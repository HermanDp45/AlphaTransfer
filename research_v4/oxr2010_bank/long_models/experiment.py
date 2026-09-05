"""Matched OXR backextension and bank fusion; frozen temporal model evaluation."""
from pathlib import Path
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,hashlib,time,warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from final_solution.training import core_experiment as core
from research_v3.models import experiment as old
from research_v4.continuation.oxr import experiment as oxr
from research_v4.continuation.oxr.assess import point
from research_v4.liquidity import experiment as bank
from research_v4.kazakhstan.experiment import Adapted
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';SNAPSHOT=HERE.parent/'input_oxr_snapshot.csv'
PREMIUM=['bank_oxr_premium','bank_oxr_premium_chg1','bank_oxr_premium_chg5','bank_oxr_quote_age_gap']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def fp(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).to_numpy().tobytes()).hexdigest()
def specifications():
 out=[]
 def add(name,months=120,family='none',since='2010-01-01',bank=False,adapt=False,local=False,premium=False):
  out.append(dict(name=name,months=months,family=family,since=since,bank=bank,adapt=adapt,local=local,premium=premium))
 for m in (120,180):
  add(f'v3_{m}m',m)
  for year in (2018,2010):add(f'oxr_basis_{year}_{m}m',m,'basis',f'{year}-06-17' if year==2018 else '2010-01-01')
 add('oxr_full_2010_120m',family='full')
 add('oxr_coverage_2010_120m',family='coverage')
 add('kzt_local_120m',local=True)
 add('kzt_shrink_120m',adapt=True,local=True)
 add('halyk_local_120m',bank=True,local=True)
 add('halyk_shrink_120m',bank=True,adapt=True,local=True)
 for year in (2018,2010):
  since=f'{year}-06-17' if year==2018 else '2010-01-01'
  add(f'kzt_oxr_{year}_120m',family='basis',since=since,adapt=True,local=True)
  add(f'fusion_{year}_120m',family='basis',since=since,bank=True,adapt=True,local=True)
 add('fusion_premium_2010_120m',family='basis',bank=True,adapt=True,local=True,premium=True)
 add('fusion_2010_180m',180,'basis',bank=True,adapt=True,local=True)
 return out
def build_views():
 raw=pd.read_csv(SNAPSHOT);banks={};panels={}
 for lag in (1,2):
  banks[lag],groups,_=bank.build_panel(True,lag)
 cols=groups['halyk']+['halyk_personal_rub_log_price','halyk_personal_rub_observation_date','halyk_personal_rub_available_date']
 for since in ('2010-01-01','2018-06-17'):
  for delay in (24,48):
   p=oxr.build_panel(delay,since,raw=raw)
   for lag in (1,2):
    pd.testing.assert_frame_equal(p[['date','corridor',*oxr.BASE]],banks[lag][['date','corridor',*oxr.BASE]],check_exact=True)
    q=p.copy();q[cols]=banks[lag][cols]
    k=q[q.corridor.eq('KZT')].set_index('date')
    q['bank_oxr_premium']=q.halyk_personal_rub_log_price+q.date.map(k.oxr_log_rate)
    q['bank_oxr_quote_age_gap']=(q.halyk_personal_rub_observation_date-q.date.map(k.observed_date)).dt.days
    q['bank_oxr_premium_chg1']=q.groupby('corridor').bank_oxr_premium.diff()
    q['bank_oxr_premium_chg5']=q.groupby('corridor').bank_oxr_premium.diff(5)
    q=core.add_target(q,5);q['label_available_date']=q.groupby('corridor').date.shift(-5)
    panels[since,delay,lag]=q
 return panels,groups['halyk']
def feature_list(spec,bankcols):
 extras={'none':[],'basis':oxr.BASIS+oxr.COVER,'full':oxr.BASIS+oxr.RET+oxr.COVER,'coverage':oxr.COVER}[spec['family']]
 return oxr.BASE+extras+(bankcols if spec['bank'] else [])+(PREMIUM if spec['premium'] else [])
def new_model(features):
 model=core.make_model('hist_gradient_boosting',features)
 model.named_steps['classifier'].set_params(early_stopping=False)
 return model
def stress_view(views,spec,cutoff,delay,lag):
 """Splice the outage at cutoff, preserving normal rolling feature history."""
 normal=views[spec['since'],24,1]
 if (delay,lag)==(24,1):return normal
 q=views[spec['since'],delay,lag].copy();before=q.date.lt(cutoff)
 q.loc[before,:]=normal.loc[before,:]
 for _,g in q.groupby('corridor'):
  b=g.oxr_log_basis
  q.loc[g.index,'oxr_basis_chg1']=b.diff()
  q.loc[g.index,'oxr_basis_chg5']=b.diff(5)
  q.loc[g.index,'oxr_basis_z20']=(b-b.rolling(20,min_periods=10).mean())/b.rolling(20,min_periods=10).std().clip(lower=1e-6)
  for n in (1,5):q.loc[g.index,f'bank_oxr_premium_chg{n}']=g.bank_oxr_premium.diff(n)
 return q
def run(views,bankcols,spec,cutoff):
 clock=time.monotonic();cutoff=pd.Timestamp(cutoff);end=pd.Timestamp(cutoff.year+1,1,1)
 p=views[spec['since'],24,1];features=feature_list(spec,bankcols)
 tr,va,te=old.temporal_split(p,5,cutoff,end,old.Spec(spec['name'],months=spec['months'],extended=True))
 cal_start=cutoff-pd.DateOffset(years=1)
 assert tr.label_available_date.max()<cal_start and va.label_available_date.max()<cutoff
 history=p[p.date.ge(cal_start)&p.date.lt(cutoff)].copy()
 if spec['local']:
  va=va[va.corridor.eq('KZT')];te=te[te.corridor.eq('KZT')];history=history[history.corridor.eq('KZT')]
 model=new_model(features)
 if spec['adapt']:model=Adapted(model,'residual_shrink')
 model.fit(tr[features+['corridor']],tr.target.astype(int))
 if spec['adapt']:
  x=va[features+['corridor']];pooled=model.pooled.predict_proba(x)[:,1];adapted=model.predict_proba(x)[:,1]
  model.weight=min((0.,.25,.5,.75,1.),key=lambda w:np.mean((w*adapted+(1-w)*pooled-va.target)**2))
 raw=model.predict_proba(va[features+['corridor']])[:,1];cal=core.fit_platt_calibrator(raw,va.target)
 prob=core.apply_platt(cal,raw);threshold,_,_=core.choose_frequency_threshold(va,prob)
 hp=core.apply_platt(cal,model.predict_proba(history[features+['corridor']])[:,1])
 hi=core.select_per_corridor_with_cooldown(history,hp,threshold)
 state=core.corridor_selection_state(history,hi)
 ps=core.selection_state(history,core.select_portfolio_from_candidates(history,hp,hi))
 bundle=dict(model=model,calibrator=cal,threshold=threshold,initial_state=state,portfolio_state=ps,features=features,spec=spec,cutoff=str(cutoff.date()))
 name=spec['name']+'_'+str(cutoff.date());checkpoint=OUT/(name+'.pkl');checkpoint.write_bytes(pickle.dumps(bundle))
 modes={'normal':(24,1)}
 if spec['family']!='none':modes['oxr_delayed']=(48,1)
 if spec['bank']:modes['bank_delayed']=(24,2)
 if spec['bank'] and spec['family']!='none':modes['both_delayed']=(48,2)
 outputs=[]
 for mode,(delay,lag) in modes.items():
  test=stress_view(views,spec,cutoff,delay,lag).loc[te.index]
  pd.testing.assert_frame_equal(te[['date','corridor','target']],test[['date','corridor','target']],check_exact=True)
  raw=model.predict_proba(test[features+['corridor']])[:,1];prob=core.apply_platt(cal,raw)
  chosen=core.select_per_corridor_with_cooldown(test,prob,threshold,state)
  portfolio=core.select_portfolio_from_candidates(test,prob,chosen,ps)
  pred=test[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date']].copy()
  pred['raw_probability']=raw;pred['probability']=prob;pred['candidate_signal']=pred.index.isin(chosen);pred['signal']=pred.index.isin(portfolio)
  pred['config_id']=spec['name'];pred['cutoff']=str(cutoff.date());pred['fold_test_year']=cutoff.year;pred['mode']=mode
  outputs.append(pred)
 pred=pd.concat(outputs,ignore_index=True);dest=OUT/(name+'.csv.gz');pred.to_csv(dest,index=False)
 receipt=dict(spec=spec,cutoff=cutoff,train_rows=len(tr),train_min=tr.date.min(),train_max=tr.date.max(),train_latest_label=tr.label_available_date.max(),validation_min=va.date.min(),validation_max=va.date.max(),validation_latest_label=va.label_available_date.max(),validation_rows=len(va),test_rows=len(te),test_min=te.date.min(),test_max=te.date.max(),history_max=history.date.max(),features=features,shrink_weight=getattr(model,'weight',None),train_oxr_coverage=float(tr.oxr_available.mean()),train_halyk_coverage=float(tr.halyk_personal_rub_log_price.notna().mean()),train_feature_fingerprint=fp(tr[['date','corridor',*features]]),checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(dest),source_sha256=sha(SNAPSHOT),seconds=time.monotonic()-clock)
 save(OUT/(name+'.json'),receipt)
 print(name,'rows',len(te),'seconds',round(receipt['seconds'],2),flush=True)
 return pred
def summary(p):
 rows=[]
 for (config,mode),g in p.groupby(['config_id','mode']):
  for scope in (['all','KZT'] if g.corridor.nunique()>1 else ['KZT']):
   x=g if scope=='all' else g[g.corridor.eq('KZT')]
   rows.append(dict(config_id=config,mode=mode,scope=scope,**point(x)))
 return pd.DataFrame(rows)
def main():
 OUT.mkdir(exist_ok=True);specs=specifications()
 protocol=dict(created_unix=time.time(),specifications=specs,primary_contrasts=['oxr_basis_2010_120m vs oxr_basis_2018_120m','fusion_2010_120m vs fusion_2018_120m','fusion_2010_120m vs halyk_shrink_120m'],decision='CBR frozen reference10:05Moscow; OXR max(day_end,published)+24h; Halyk effective_date+1calendar day',stress='fixed model/calibration/threshold and pre-cutoff normal state; extra one day OXR and/or Halyk AFTER cutoff',selection='development2023-2025 normal Brier within all-corridor and KZT bank families, controls eligible; frozen before new2026 computation',test='Jan/Mar2026 temporal cutoff, previously inspected history, NOT pristine holdout',source_sha256=sha(SNAPSHOT),code_sha256=sha(__file__))
 save(HERE/'protocol.json',protocol)
 frames=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  views,bankcols=build_views()
  for cutoff in ('2023-01-01','2024-01-01','2025-01-01'):
   for spec in specs:frames.append(run(views,bankcols,spec,cutoff))
  dev=pd.concat(frames,ignore_index=True);dev.to_csv(HERE/'development_predictions.csv.gz',index=False)
  metrics=summary(dev);metrics.to_csv(HERE/'development_summary.csv',index=False)
  normal=metrics[metrics['mode'].eq('normal')]
  a=normal[normal.scope.eq('all')];b=normal[normal.scope.eq('KZT')&normal.config_id.isin([s['name'] for s in specs if s['local']])]
  save(HERE/'selection.json',dict(global_candidate=a.loc[a.brier.idxmin(),'config_id'],bank_candidate=b.loc[b.brier.idxmin(),'config_id'],selection_unix=time.time(),uses_2026=False,development_sha256=sha(HERE/'development_predictions.csv.gz'),protocol_sha256=sha(HERE/'protocol.json')))
  for cutoff in ('2026-01-01','2026-03-01'):
   for spec in specs:frames.append(run(views,bankcols,spec,cutoff))
 pred=pd.concat(frames,ignore_index=True);pred.to_csv(HERE/'all_predictions.csv.gz',index=False)
 rows=[]
 for cutoff,g in pred.groupby('cutoff'):
  m=summary(g);m['cutoff']=cutoff;rows.append(m)
 pd.concat(rows).to_csv(HERE/'metrics_by_cutoff.csv',index=False)
 save(HERE/'completion.json',dict(status='complete',fits=len(specs)*5,prediction_rows=len(pred),specifications=len(specs),selection= json.loads((HERE/'selection.json').read_text()),source_sha256=sha(SNAPSHOT),code_sha256=sha(__file__)))
if __name__=='__main__':main()
