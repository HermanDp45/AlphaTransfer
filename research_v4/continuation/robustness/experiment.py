#!/usr/bin/env python3
"""Fixed-family Halyk delay stress; train/calibration strictly before cutoff."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,dataclasses,hashlib,json,pickle,time,warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import log_loss
from research_v4.liquidity import experiment as liquidity
from research_v4.kazakhstan import experiment as kz
from final_solution.training import core_experiment as core
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';H=5
BASE=kz.FEATURES
RULES=('v3_long_globalcal','v3_long_localcal','halyk_l1','halyk_l2_retrained','lag_augmentation','feature_dropout','lag_ensemble','minimax_shrink_v3')
ROBUST=('lag_augmentation','feature_dropout','lag_ensemble','minimax_shrink_v3')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()
def build_views():
 views={}
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  for lag in (1,2,3):
   p,groups,_=liquidity.build_panel(True,lag);p=core.add_target(p,H)
   p['label_available_date']=p.groupby('corridor').date.shift(-H)
   views[lag]=p
 for lag in (2,3):
  pd.testing.assert_frame_equal(views[1][['date','corridor',*BASE,'target']],views[lag][['date','corridor',*BASE,'target']],check_exact=True)
 return views,groups['halyk']

def splits(frame,cutoff,end):
 cal_start=cutoff-pd.DateOffset(months=12);train_start=cal_start-pd.DateOffset(months=120)
 valid=frame.target.notna()
 train=frame[valid&(frame.date>=train_start)&(frame.date<cal_start)&(frame.label_available_date<cal_start)].copy()
 calibration=frame[valid&(frame.date>=cal_start)&(frame.date<cutoff)&(frame.label_available_date<cutoff)].copy()
 history=frame[(frame.date>=cal_start)&(frame.date<cutoff)].copy()
 test=frame[valid&(frame.date>=cutoff)&(frame.date<end)].copy()
 if end<=frame.date.max():test=test[test.label_available_date<end].copy()
 assert train.label_available_date.max()<cal_start and calibration.label_available_date.max()<cutoff
 assert len(calibration)>25 and len(test)>0 and train.date.max()<calibration.date.min()
 return train,calibration,history,test

def date_bucket(frame,salt,modulus):
 return np.array([int.from_bytes(hashlib.sha256(f'{salt}|{pd.Timestamp(d).date()}'.encode()).digest()[:8],'big')%modulus for d in frame.date])

def new_model(features):
 m=core.make_model('hist_gradient_boosting',features);m.named_steps['classifier'].set_params(early_stopping=False);return m

def base_probability(bundle,frames,name,lag):
 return bundle['models'][name].predict_proba(frames[lag])[:,1]

def raw_probability(bundle,frames,rule,delay=0):
 if rule.startswith('v3_long_'):return base_probability(bundle,frames,'v3',1)
 if rule=='halyk_l1':return base_probability(bundle,frames,'h1',1+delay)
 if rule=='halyk_l2_retrained':return base_probability(bundle,frames,'h2',2+delay)
 if rule=='lag_augmentation':return base_probability(bundle,frames,'aug',1+delay)
 if rule=='feature_dropout':return base_probability(bundle,frames,'drop',1+delay)
 if rule=='lag_ensemble':return .5*(base_probability(bundle,frames,'h1',1+delay)+base_probability(bundle,frames,'h2',2+delay))
 if rule=='minimax_shrink_v3':return bundle['blend_alpha']*probability(bundle,frames,'halyk_l1',delay)+(1-bundle['blend_alpha'])*probability(bundle,frames,'v3_long_globalcal',0)
 raise ValueError(rule)

def probability(bundle,frames,rule,delay=0):
 p=raw_probability(bundle,frames,rule,delay)
 return p if rule=='minimax_shrink_v3' else core.apply_platt(bundle['calibrators'][rule],p)

def fit(views,extra,cutoff,end):
 start=time.monotonic();train,val,history,test=splits(views[1],cutoff,end);features=BASE+extra;v=val[val.corridor.eq('KZT')]
 frames={lag:p.loc[v.index,features+['corridor']] for lag,p in views.items()};y=v.target
 models={};weights={};models['v3']=new_model(BASE).fit(train[BASE+['corridor']],train.target.astype(int))
 training_fingerprints={}
 for name,lag in [('h1',1),('h2',2),('aug',1),('drop',1)]:
  x=views[lag].loc[train.index,features+['corridor']].copy()
  if name=='aug':
   mask=date_bucket(train,'lagaugmentation-v1',2).astype(bool);x.loc[mask,extra]=views[2].loc[train.index[mask],extra].to_numpy()
  elif name=='drop':x.loc[date_bucket(train,'halykdropout-v1',4)==0,extra]=np.nan
  training_fingerprints[name]=fingerprint(x)
  m=kz.Adapted(new_model(features),'residual_shrink').fit(x,train.target.astype(int));models[name]=m
  pair=[1,2] if name in ('aug','drop') else [lag]
  pooled=[m.pooled.predict_proba(frames[l])[:,1] for l in pair];adapted=[m.predict_proba(frames[l])[:,1] for l in pair]
  choices=[]
  for w in (0.,.25,.5,.75,1.):
   losses=[float(np.mean((w*a+(1-w)*b-y.to_numpy())**2)) for a,b in zip(adapted,pooled)];choices.append((max(losses),w))
  _,m.weight=min(choices);weights[name]={'weight':m.weight,'criterion':'worst_validation_raw_brier' if len(pair)==2 else 'normal_validation_raw_brier','views':pair}
 bundle={'models':models,'calibrators':{},'thresholds':{},'initial_states':{},'features':features,'cutoff':str(cutoff.date()),'end':str(end.date()),'extra':extra,'fit_metadata':{'train_start':str(train.date.min().date()),'train_end':str(train.date.max().date()),'train_max_label_available':str(train.label_available_date.max().date()),'calibration_start':str(val.date.min().date()),'calibration_end':str(val.date.max().date()),'calibration_max_label_available':str(val.label_available_date.max().date()),'history_start':str(history.date.min().date()),'history_end':str(history.date.max().date()),'train_rows':len(train),'kzt_calibration_dates':len(v),'training_feature_fingerprints':training_fingerprints,'shrink_weights':weights}}
 for rule in RULES:
  if rule=='minimax_shrink_v3':continue
  if rule=='v3_long_globalcal':
   raw=models['v3'].predict_proba(val[BASE+['corridor']])[:,1];labels=val.target
  elif rule in ('lag_augmentation','feature_dropout','lag_ensemble'):
   raw=np.concatenate([raw_probability(bundle,frames,rule,d) for d in (0,1)]);labels=pd.Series(np.tile(y.to_numpy(),2))
  else:raw=raw_probability(bundle,frames,rule);labels=y
  bundle['calibrators'][rule]=core.fit_platt_calibrator(raw,labels)
 bp=probability(bundle,frames,'v3_long_globalcal');hp=[probability(bundle,frames,'halyk_l1',d) for d in (0,1)]
 choices=[]
 for alpha in (0.,.25,.5,.75,1.):choices.append((max(float(np.mean((alpha*p+(1-alpha)*bp-y.to_numpy())**2)) for p in hp),alpha))
 _,bundle['blend_alpha']=min(choices)
 hist=history[history.corridor.eq('KZT')];hframes={lag:p.loc[hist.index,features+['corridor']] for lag,p in views.items()}
 tuning=[]
 for rule in RULES:
  p=probability(bundle,frames,rule);threshold,frequency,error=core.choose_frequency_threshold(v,p);bundle['thresholds'][rule]=threshold
  selected=core.select_per_corridor_with_cooldown(hist,probability(bundle,hframes,rule),threshold);bundle['initial_states'][rule]=core.corridor_selection_state(hist,selected)
  for delay in (0,1):
   pp=probability(bundle,frames,rule,delay);tuning.append({'cutoff':bundle['cutoff'],'rule':rule,'delay':delay,'validation_brier':float(np.mean((pp-y.to_numpy())**2)),'threshold':threshold['KZT'],'validation_signal_frequency':frequency,'validation_weighted_error':error,'calibration_observations':len(y),'calibration_view_rows':len(y)*(2 if rule in ('lag_augmentation','feature_dropout','lag_ensemble') else 1)})
 bundle['fit_metadata']['seconds']=time.monotonic()-start
 return bundle,pd.DataFrame(tuning),test

def predict(bundle,views,test):
 t=test[test.corridor.eq('KZT')].copy();frames={lag:p.loc[t.index,bundle['features']+['corridor']] for lag,p in views.items()};rows=[]
 for rule in RULES:
  for delay in (0,1):
   probability_values=probability(bundle,frames,rule,delay);raw=raw_probability(bundle,frames,rule,delay)
   selected=core.select_per_corridor_with_cooldown(t,probability_values,bundle['thresholds'][rule],bundle['initial_states'][rule]);chosen=set(selected)
   output=t[['date','corridor','target','forward_bps','regret_bps','rub_per_unit','session_ordinal','label_available_date']].copy()
   output['rule']=rule;output['delay']=delay;output['cutoff']=bundle['cutoff'];output['probability']=probability_values;output['raw_probability']=raw;output['candidate_signal']=[i in chosen for i in t.index];output['threshold']=bundle['thresholds'][rule]['KZT'];rows.append(output)
 return pd.concat(rows,ignore_index=True)

def candidate_comparison(g):
 select=g[g.candidate_signal].copy()
 baseline=g.groupby(['cutoff','corridor']).agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
 select=select.join(baseline,on=['cutoff','corridor'])
 expected_hits=float(select.base_hit.sum())
 standardized=float(select.target.sum()/expected_hits) if expected_hits else np.nan
 return {'candidate_count':len(select),'candidate_quality':float(select.target.mean()),
         'candidate_lift':standardized,'candidate_lift_standardized':standardized,
         'candidate_lift_unstandardized':float(select.target.mean()/g.target.mean()),
         'forward_delta_bps':float((select.forward_bps-select.base_forward).mean()),
         'forward_bps_absolute':float(select.forward_bps.mean()),
         'forward_bps_per_candidate':float(select.forward_bps.mean()),
         'regret_bps_per_candidate':float(select.regret_bps.mean())}

def metrics(pred):
 results=[]
 for (cutoff,rule,delay),g in pred.groupby(['cutoff','rule','delay']):
  counts=g.groupby(g.date.dt.to_period('W-SUN')).candidate_signal.sum();span=(g.date.max()-g.date.min()).days+1
  results.append(dict(cutoff=cutoff,rule=rule,delay=int(delay),rows=len(g),first_date=str(g.date.min().date()),last_date=str(g.date.max().date()),brier=float(np.mean((g.probability-g.target)**2)),log_loss=float(log_loss(g.target,g.probability,labels=[0,1])),raw_brier=float(np.mean((g.raw_probability-g.target)**2)),candidate_frequency_per_week=int(g.candidate_signal.sum())*7/span,weeks_1_to_2_share=float(counts.between(1,2).mean()),threshold=float(g.threshold.iloc[0]),**candidate_comparison(g)))
 return pd.DataFrame(results)

def development_summary(pred):
 results=[]
 for (rule,delay),g in pred.groupby(['rule','delay']):
  results.append(dict(rule=rule,delay=int(delay),rows=len(g),brier=float(np.mean((g.probability-g.target)**2)),raw_brier=float(np.mean((g.raw_probability-g.target)**2)),**candidate_comparison(g)))
 return pd.DataFrame(results)

def old_parity(pred):
 rows=[]
 paths={'v3_long_globalcal':ROOT/'research_v3/models/basis_train_120m_h5_predictions.csv.gz','v3_long_localcal':ROOT/'research_v4/kazakhstan/kzt_pooled_120m_predictions.csv.gz','halyk_l1':ROOT/'research_v4/kazakhstan/kzt_residual_shrink_120m__halyk_lag1_predictions.csv.gz','halyk_l2_retrained':ROOT/'research_v4/kazakhstan/kzt_residual_shrink_120m__halyk_lag2_predictions.csv.gz'}
 for rule,p in paths.items():
  old=pd.read_csv(p,parse_dates=['date']);old=old[old.corridor.eq('KZT')];new=pred[(pred.rule==rule)&pred.delay.eq(0)&pred.cutoff.str.endswith('-01-01')]
  match=new.merge(old,on=['date','corridor'],suffixes=('_new','_old'),validate='one_to_one');assert len(match)==len(new)
  error=float(np.max(np.abs(match.probability_new-match.probability_old)));rawerror=float(np.max(np.abs(match.raw_probability_new-match.raw_probability_old)));cand=int((match.candidate_signal_new!=match.candidate_signal_old).sum())
  rows.append(dict(rule=rule,rows=len(match),max_probability_error=error,max_raw_error=rawerror,candidate_mismatches=cand));assert error<1e-12 and rawerror<1e-12 and cand==0,(rule,error,rawerror,cand)
 return rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);clock=time.monotonic();views,extra=build_views();before={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'research_v3/manifest.json',ROOT/'research_v4/continuation/previous_v4_manifest.json',ROOT/'research_v3/models/panel_extended.pkl',ROOT/'research_v4/liquidity/halyk_sell_daily.csv',ROOT/'research_v4/liquidity/kase_spot_daily.csv',Path(kz.__file__),Path(liquidity.__file__),Path(core.__file__)]};preds=[];tuning=[];receipts=[]
 def run(cutoff,end):
  cutoff=pd.Timestamp(cutoff);end=pd.Timestamp(end);bundle,cal,test=fit(views,extra,cutoff,end);prediction=predict(bundle,views,test);name=str(cutoff.date());path=OUT/'checkpoints'/f'freeze_{name}.pkl';path.write_bytes(pickle.dumps(bundle));dest=OUT/f'predictions_{name}.csv.gz';prediction.to_csv(dest,index=False);receipts.append({'cutoff':name,**bundle['fit_metadata'],'blend_alpha':bundle['blend_alpha'],'checkpoint':str(path.relative_to(HERE)),'checkpoint_sha256':digest(path),'prediction':str(dest.relative_to(HERE)),'prediction_sha256':digest(dest)});preds.append(prediction);tuning.append(cal);print(name,'fit_seconds',round(bundle['fit_metadata']['seconds'],2),'KZT_dates',len(test[test.corridor.eq('KZT')]),flush=True)
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  for year in (2023,2024,2025):run(f'{year}-01-01',f'{year+1}-01-01')
  development=pd.concat(preds,ignore_index=True);dev=development_summary(development);dev.to_csv(OUT/'development_summary.csv',index=False)
  scores=dev.pivot(index='rule',columns='delay',values='brier');scores['worst_brier']=scores[[0,1]].max(axis=1);scores['normal_minus_current_l1']=scores[0]-scores.loc['halyk_l1',0];scores['delayed_minus_v3']=scores[1]-scores.loc['v3_long_globalcal',0];scores['strict_no_quality_loss']=(scores.normal_minus_current_l1<=0)&(scores.delayed_minus_v3<=0)
  winner=scores.loc[list(ROBUST)].sort_values(['worst_brier'],kind='stable').index[0];selection={'selected_rule':winner,'criterion':'minimum worst normal/delayed development2023-2025 Brier among fixed robust family','normal_reference':float(scores.loc['halyk_l1',0]),'delay_reference':float(scores.loc['v3_long_globalcal',0]),'strict_no_loss_robust_methods':[x for x in ROBUST if bool(scores.loc[x,'strict_no_quality_loss'])],'selection_uses_2026':False,'2026_previously_inspected_in_V4':True,'selection_before_new_2026_computation':True,'protocol_sha256':digest(HERE/'PROTOCOL.md')}
  scores.to_csv(OUT/'robust_selection_table.csv');(OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTED',winner,'strict no-loss',selection['strict_no_loss_robust_methods'],flush=True)
  for cutoff in ('2026-01-01','2026-03-01'):run(cutoff,'2027-01-01')
 allpred=pd.concat(preds,ignore_index=True);allpred.to_csv(OUT/'all_predictions.csv.gz',index=False);metrics(allpred).to_csv(OUT/'metrics_by_cutoff.csv',index=False);common=allpred[(allpred.date>='2026-03-05')&allpred.cutoff.isin(['2026-01-01','2026-03-01'])];metrics(common).to_csv(OUT/'common_march5_metrics.csv',index=False);pd.concat(tuning).to_csv(OUT/'validation_diagnostics.csv',index=False)
 parity=old_parity(allpred);(OUT/'old_parity.json').write_text(json.dumps(parity,indent=2)+'\n');(OUT/'model_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n');assert before=={name:digest(ROOT/name) for name in before}
 manifest={'status':'complete','nature':'Retrospective robust-source stress; not pristine holdout or causal user gain','horizon':H,'rules':RULES,'robust_family':ROBUST,'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(HERE/'PROTOCOL.md'),'input_sha256':before,'runtime_seconds':time.monotonic()-clock,'selection':selection,'final_source_date':str(views[1].date.max().date()),'last_mature_h5_target_date':str(allpred.date.max().date()),'delayed_deployment':'one extra calendar day of Halyk delay starts at cutoff; no test recalibration; history remains normal','training_lag_augmentation':'one deterministic date-hash view per row; no duplicated market dates/corridors','calibration_views':'robust Platt uses two possible views per existing KZT validation date; not independent sample augmentation'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('COMPLETE',round(time.monotonic()-clock,2),flush=True)
if __name__=='__main__':main()
