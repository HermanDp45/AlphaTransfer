"""Bounded, explicitly post-inspection followup using observable quote age."""
from __future__ import annotations
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.continuation.robustness import experiment as e
HERE=e.HERE;OUT=e.OUT
RULES={'age_fallback_v3':'v3_long_globalcal','age_fallback_minimax':'minimax_shrink_v3'}
OBS_COLUMNS=['halyk_personal_rub_observation_date','halyk_legal_rub_observation_date','halyk_personal_usd_observation_date']

def quote_age(frame):
 ages=pd.concat([(frame.date-frame[col]).dt.total_seconds()/86400 for col in OBS_COLUMNS],axis=1)
 result=ages.max(axis=1).to_numpy(copy=True);result[ages.isna().any(axis=1).to_numpy()]=np.inf
 assert (result>=0).all()
 return result

def gate_probability(age,normal,alternative,cutoff):
 fallback=np.asarray(age)>cutoff
 return np.where(fallback,alternative,normal),fallback

def scores(bundle,views,frame,rule,delay,cutoff):
 features={lag:p.loc[frame.index,bundle['features']+['corridor']] for lag,p in views.items()};main=e.probability(bundle,features,'halyk_l1',delay);alternative=e.probability(bundle,features,RULES[rule],delay)
 age=quote_age(views[1+delay].loc[frame.index]);p,fallback=gate_probability(age,main,alternative,cutoff)
 return p,age,fallback

def fit_age(bundle,views):
 cut=pd.Timestamp(bundle['cutoff']);end=pd.Timestamp(bundle['end']);_,validation,history,test=e.splits(views[1],cut,end);v=validation[validation.corridor.eq('KZT')];h=history[history.corridor.eq('KZT')];params={};trials=[]
 for rule in RULES:
  choices=[]
  for agecut in (2,3):
   losses=[]
   for delay in (0,1):
    p,age,fallback=scores(bundle,views,v,rule,delay,agecut);loss=float(np.mean((p-v.target.to_numpy())**2));losses.append(loss);trials.append({'cutoff':bundle['cutoff'],'rule':rule,'age_cutoff':agecut,'delay':delay,'validation_brier':loss,'validation_dates':len(v),'fallback_share':float(fallback.mean())})
   choices.append((max(losses),agecut))
  loss,agecut=min(choices);vp,_,_=scores(bundle,views,v,rule,0,agecut);threshold,frequency,error=e.core.choose_frequency_threshold(v,vp);hp,_,_=scores(bundle,views,h,rule,0,agecut);selected=e.core.select_per_corridor_with_cooldown(h,hp,threshold);state=e.core.corridor_selection_state(h,selected)
  params[rule]={'age_cutoff':agecut,'threshold':threshold,'initial_state':state,'worst_validation_brier':loss,'criterion':'minimax past validation Brier over fixed age cutoffs2/3, normal/delayed views','validation_candidate_frequency':frequency,'validation_weighted_error':error}
 return params,pd.DataFrame(trials),test

def predict(bundle,views,test,params):
 rows=[];t=test[test.corridor.eq('KZT')]
 for rule in RULES:
  for delay in (0,1):
   p,age,fallback=scores(bundle,views,t,rule,delay,params[rule]['age_cutoff']);selected=e.core.select_per_corridor_with_cooldown(t,p,params[rule]['threshold'],params[rule]['initial_state']);chosen=set(selected)
   out=t[['date','corridor','target','forward_bps','regret_bps','rub_per_unit','session_ordinal','label_available_date']].copy();out['rule']=rule;out['delay']=delay;out['cutoff']=bundle['cutoff'];out['probability']=p;out['raw_probability']=p;out['candidate_signal']=[i in chosen for i in t.index];out['threshold']=params[rule]['threshold']['KZT'];out['age_calendar_days']=age;out['age_cutoff']=params[rule]['age_cutoff'];out['fallback']=fallback;rows.append(out)
 return pd.concat(rows,ignore_index=True)

def main():
 views,extra=e.build_views();receipts=json.loads((OUT/'model_receipts.json').read_text());preds=[];trials=[];newreceipts=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  for receipt in receipts:
   path=HERE/receipt['checkpoint'];assert e.digest(path)==receipt['checkpoint_sha256']
   with path.open('rb') as f:bundle=pickle.load(f)
   params,table,test=fit_age(bundle,views);pred=predict(bundle,views,test,params);preds.append(pred);trials.append(table);newreceipts.append({'cutoff':bundle['cutoff'],'params':params,'base_checkpoint':receipt['checkpoint'],'base_checkpoint_sha256':receipt['checkpoint_sha256']})
 allpred=pd.concat(preds,ignore_index=True);allpred.to_csv(OUT/'age_predictions.csv.gz',index=False);pd.concat(trials).to_csv(OUT/'age_validation_trials.csv',index=False);e.metrics(allpred).to_csv(OUT/'age_metrics_by_cutoff.csv',index=False);e.development_summary(allpred[allpred.date<'2026-01-01']).to_csv(OUT/'age_development_summary.csv',index=False);e.metrics(allpred[(allpred.date>='2026-03-05')&allpred.cutoff.isin(['2026-01-01','2026-03-01'])]).to_csv(OUT/'age_common_march5_metrics.csv',index=False)
 (OUT/'age_receipts.json').write_text(json.dumps(newreceipts,indent=2)+'\n');(OUT/'age_manifest.json').write_text(json.dumps({'status':'complete','status_detail':'bounded followup added after primary2026 results inspected; no2026test labels select age threshold','code_sha256':e.digest(Path(__file__)),'protocol_sha256':e.digest(HERE/'AGE_FOLLOWUP_PROTOCOL.md'),'predictions_sha256':e.digest(OUT/'age_predictions.csv.gz'),'input_base_manifest_sha256':e.digest(OUT/'manifest.json'),'age_observation_columns':OBS_COLUMNS,'normal_delayed_flag_used_by_gate':False,'original_selection_unchanged':json.loads((OUT/'selection.json').read_text())},indent=2)+'\n')
 print(e.development_summary(allpred[allpred.date<'2026-01-01']).to_string(index=False))
if __name__=='__main__':main()
