"""Separate CLOSING endpoint model, fixed 4-fit Treasury/Halyk KZT PoC."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,hashlib,time,warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from final_solution.training import core_experiment as core
from research_v3.models import experiment as old
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
from research_v4.liquidity import experiment as bank
from research_v4.kazakhstan.experiment import Adapted
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';H=5
ARMS=('closing_treasury_shrink120m','closing_treasury_halyk_shrink120m')
BASE=old.BASE_FEATURES+[old.BASIS]
CONTRACT='CLOSING:R[t+5]>R[t];R=RUB_per_recipient_unit;h=5_effective_CBR_rows;tau=0'
SOURCE_PATHS=[ROOT/'research_v3/models/panel_extended.pkl',ROOT/'research_v3/external_data/feature_panel.parquet',ROOT/'research_v4/liquidity/halyk_sell_daily.csv',Path(bank.__file__),ROOT/'research_v4/kazakhstan/experiment.py',Path(core.__file__)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def fp(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()

def build_views():
 views={}
 for lag in (1,2):
  p,groups,_=bank.build_panel(True,lag);p=core.add_target(augment_panel(p),H)
  p['now_target']=p.target;p['future_endpoint_rate']=p.groupby('corridor').rub_per_unit.shift(-H);p['label_available_date']=p.groupby('corridor').date.shift(-H)
  p['closing_target']=(p.future_endpoint_rate>p.rub_per_unit).astype(float).where(p.future_endpoint_rate.notna());p['target']=p.closing_target;p['endpoint_bps']=10000*(p.future_endpoint_rate/p.rub_per_unit-1)
  p['recent_low_rank5']=p.groupby('corridor').pr60.transform(lambda s:s.shift(1).rolling(5,min_periods=1).min())
  p['recent_low_rate5']=p.groupby('corridor').rub_per_unit.transform(lambda s:s.shift(1).rolling(5,min_periods=1).min())
  p['change_from_recent_low_bps']=10000*(p.rub_per_unit/p.recent_low_rate5-1)
  p['now_text_gate']=p.pr60.le(.2)|p.ret1.lt(0)
  p['closing_text_gate']=p.ret1.gt(0)
  p['closing_rebound_low_evidence']=p.closing_text_gate&p.recent_low_rank5.le(.2)
  views[lag]=p
 features={ARMS[0]:BASE+TREASURY_LAG7_FEATURES,ARMS[1]:BASE+groups['halyk']+TREASURY_LAG7_FEATURES}
 return views,features

def split(p,cutoff):
 cut=pd.Timestamp(cutoff);end=pd.Timestamp(cut.year+1,1,1)
 tr,va,te=old.temporal_split(p,H,cut,end,old.Spec('closing',months=120,extended=True))
 va=va[va.corridor.eq('KZT')].copy();te=te[te.corridor.eq('KZT')].copy();history=p[p.corridor.eq('KZT')&p.date.ge(cut-pd.DateOffset(years=1))&p.date.lt(cut)].copy()
 assert tr.label_available_date.max()<cut-pd.DateOffset(years=1) and va.label_available_date.max()<cut
 return tr,va,te,history

def fit(p,features,cutoff,arm):
 tr,va,te,history=split(p,cutoff);m=core.make_model('hist_gradient_boosting',features);m.named_steps['classifier'].set_params(early_stopping=False);m=Adapted(m,'residual_shrink');m.fit(tr[features+['corridor']],tr.target.astype(int))
 x=va[features+['corridor']];base=m.pooled.predict_proba(x)[:,1];adapted=m.predict_proba(x)[:,1];m.weight=min((0.,.25,.5,.75,1.),key=lambda w:float(np.mean((w*adapted+(1-w)*base-va.target)**2)))
 raw=m.predict_proba(x)[:,1];cal=core.fit_platt_calibrator(raw,va.target);prob=core.apply_platt(cal,raw);threshold,_,_=core.choose_frequency_threshold(va,prob)
 hp=core.apply_platt(cal,m.predict_proba(history[features+['corridor']])[:,1]);selected=core.select_per_corridor_with_cooldown(history,hp,threshold);state=core.corridor_selection_state(history,selected)
 return {'model':m,'calibrator':cal,'threshold':threshold,'initial_state':state,'features':features,'cutoff':cutoff,'arm':arm,'prior_baseline_rate':float(va.target.mean()),'scenario':'CLOSING','target_contract':CONTRACT},(tr,va,te,history)

def probabilities(bundle,frame):
 raw=bundle['model'].predict_proba(frame[bundle['features']+['corridor']])[:,1];return raw,core.apply_platt(bundle['calibrator'],raw)

def output_frame(bundle,frame,mode,phase='test'):
 fields=['date','corridor','session_ordinal','rub_per_unit','ret1','pr60','recent_low_rank5','recent_low_rate5','change_from_recent_low_bps','now_text_gate','closing_text_gate','closing_rebound_low_evidence','label_available_date','target','closing_target','now_target','forward_bps','endpoint_bps','symmetric_bps','regret_bps']
 out=frame[fields].copy();raw,prob=probabilities(bundle,frame);out['raw_probability']=raw;out['probability']=prob;out['model_id']=bundle['arm'];out['config_id']=bundle['arm'];out['cutoff']=bundle['cutoff'];out['mode']=mode;out['scenario']='CLOSING';out['target_contract']=CONTRACT;out['horizon']=H;out['phase']=phase;out['prior_baseline_rate']=bundle['prior_baseline_rate'];out['threshold']=bundle['threshold']['KZT'];out['fold_test_year']=pd.Timestamp(bundle['cutoff']).year;out['calibration_eligible']=frame.label_available_date.lt(pd.Timestamp(bundle['cutoff']))
 if phase=='history':
  out.loc[~out.calibration_eligible,['target','closing_target','now_target','forward_bps','endpoint_bps','symmetric_bps','regret_bps']]=np.nan
  initial=None
 else:initial=bundle['initial_state']
 selected=core.select_per_corridor_with_cooldown(frame,prob,bundle['threshold'],initial);out['candidate_signal']=out.index.isin(selected)
 out['source_known_at']=out.date.dt.strftime('%Y-%m-%d')
 return out

def metrics(g):
 s=g[g.candidate_signal];base=g.target.mean();selected_dates=s.date.dt.to_period('W-SUN');weeks=pd.period_range(g.date.min(),g.date.max(),freq='W-SUN');counts=selected_dates.value_counts().reindex(weeks,fill_value=0)
 return {'rows':len(g),'market_dates':g.date.nunique(),'brier':float(np.mean((g.probability-g.target)**2)),'prior_prevalence_brier':float(np.mean((g.prior_baseline_rate-g.target)**2)),'log_loss':float(log_loss(g.target,g.probability,labels=[0,1])),'baseline_hit_rate':float(base),'candidate_count':len(s),'closing_hit_rate':float(s.target.mean()),'closing_lift':float(s.target.mean()/base),'forward_bps_mean':float(s.forward_bps.mean()),'forward_delta_bps':float(s.forward_bps.mean()-g.forward_bps.mean()),'endpoint_bps_mean':float(s.endpoint_bps.mean()),'endpoint_delta_bps':float(s.endpoint_bps.mean()-g.endpoint_bps.mean()),'all_observed_weeks':len(weeks),'weeks_1_to_2_share':float(counts.between(1,2).mean()),'silent_week_share':float(counts.eq(0).mean()),'weeks_above2_share':float(counts.gt(2).mean()),'closing_text_gate_share_on_candidates':float(s.closing_text_gate.mean()),'rebound_low_evidence_share_on_candidates':float(s.closing_rebound_low_evidence.mean())}

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);before={str(p.relative_to(ROOT)):sha(p) for p in SOURCE_PATHS};allpred=[];histories=[];receipts=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning);views,features=build_views()
  for cut in ('2025-01-01','2026-01-01'):
   for arm in ARMS:
    bundle,(tr,va,te,history)=fit(views[1],features[arm],cut,arm);path=OUT/'checkpoints'/f'{arm}_{cut}.pkl';path.write_bytes(pickle.dumps(bundle));normal=output_frame(bundle,te,'normal');delay=output_frame(bundle,views[2].loc[te.index],'bank_delayed');pd.testing.assert_frame_equal(normal[['date','corridor','target']],delay[['date','corridor','target']],check_exact=True)
    hist=output_frame(bundle,history,'normal','history');allpred.extend([normal,delay]);histories.append(hist)
    receipts.append({'model_id':arm,'cutoff':cut,'features':features[arm],'checkpoint':str(path.relative_to(HERE)),'checkpoint_sha256':sha(path),'train_rows':len(tr),'train_min':tr.date.min(),'train_max':tr.date.max(),'train_last_label':tr.label_available_date.max(),'validation_rows':len(va),'validation_min':va.date.min(),'validation_max':va.date.max(),'validation_last_label':va.label_available_date.max(),'history_rows':len(history),'history_last_date':history.date.max(),'history_unmatured_labels_blanked':int((~hist.calibration_eligible).sum()),'test_rows':len(te),'test_min':te.date.min(),'test_max':te.date.max(),'shrink_weight':bundle['model'].weight,'threshold':bundle['threshold'],'prior_baseline_rate':bundle['prior_baseline_rate'],'train_fingerprint':fp(tr[features[arm]+['corridor','target']]),'validation_fingerprint':fp(va[features[arm]+['corridor','target']])});print(arm,cut,'test',len(te),flush=True)
 pred=pd.concat(allpred,ignore_index=True);hist=pd.concat(histories,ignore_index=True);pred.to_csv(OUT/'closing_predictions.csv.gz',index=False);hist.to_csv(OUT/'closing_history.csv.gz',index=False)
 rows=[]
 for (arm,cut,mode),g in pred.groupby(['model_id','cutoff','mode']):rows.append({'model_id':arm,'cutoff':cut,'mode':mode,**metrics(g)})
 pd.DataFrame(rows).to_csv(OUT/'closing_metrics.csv',index=False);save(OUT/'model_receipts.json',receipts);save(OUT/'feature_groups.json',features)
 assert before=={p:sha(ROOT/p) for p in before}
 save(OUT/'manifest.json',{'scenario':'CLOSING','target_contract':CONTRACT,'source_inputs_sha256':before,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.md'),'prediction_sha256':sha(OUT/'closing_predictions.csv.gz'),'history_sha256':sha(OUT/'closing_history.csv.gz'),'fits':4,'seconds':time.monotonic()-started,'now_target_preserved_separately':True,'shared_policy_not_claimed':'Root combines scenarios under one cap with per-scenario metrics; standalone policy is diagnostic','bank_delay':'frozen L1 checkpoint applied L2 after cutoff; unchanged normal prior state'})
 print(pd.DataFrame(rows)[['model_id','cutoff','mode','brier','closing_lift','candidate_count','weeks_1_to_2_share']].to_string(index=False),flush=True)
if __name__=='__main__':main()
