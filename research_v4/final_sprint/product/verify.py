"""Independent proof of CLOSING labels, cutoffs, typed scenarios and joint counts."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,io,json,pickle,time,traceback,unittest,warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.product import closing_experiment as c
from research_v4.final_sprint.product import joint_evaluate as j
from research_v4.final_sprint.product import test_scenario_adapter as adapter_tests
HERE=c.HERE;OUT=c.OUT

def bundle(r):
 path=HERE/r['checkpoint'];assert c.sha(path)==r['checkpoint_sha256'];return pickle.loads(path.read_bytes())

def manual_probability(b,p):
 model=b['model'];x=p[b['features']+['corridor']];original=model.pooled.predict_proba(x)[:,1];z=model.pooled.named_steps['preprocessor'].transform(x);assert z.dtype==np.float64
 correction=np.zeros(len(x))
 for tree in model.trees:correction+=.025*tree.predict(z)
 adapted=expit(logit(original.clip(1e-7,1-1e-7))+correction);raw=model.weight*adapted+(1-model.weight)*original;cal=b['calibrator']
 prob=raw if cal.model is None else expit(cal.intercept+cal.slope*logit(raw.clip(1e-6,1-1e-6)))
 return raw,prob

def sources():
 m=json.loads((OUT/'manifest.json').read_text());assert m['code_sha256']==c.sha(HERE/'closing_experiment.py')
 for p,v in m['source_inputs_sha256'].items():assert c.sha(ROOT/p)==v,p
 assert c.sha(OUT/'closing_predictions.csv.gz')==m['prediction_sha256'] and c.sha(OUT/'closing_history.csv.gz')==m['history_sha256']
 return {'immutable_model_source_hashes':len(m['source_inputs_sha256']),'prediction_and_history_hashes':True}

def labels_maturity(views,receipts):
 p=views[1];checks=[]
 for corridor,g in p.groupby('corridor'):
  r=g.rub_per_unit.to_numpy();dates=g.date.to_numpy();valid=g.iloc[:-5]
  np.testing.assert_array_equal(valid.closing_target,(r[5:]>r[:-5]).astype(float));np.testing.assert_array_equal(valid.label_available_date,dates[5:]);np.testing.assert_allclose(valid.endpoint_bps,10000*(r[5:]/r[:-5]-1),rtol=0,atol=1e-12)
 for r in receipts:
  tr,va,te,hi=c.split(p,r['cutoff']);cut=pd.Timestamp(r['cutoff']);calstart=cut-pd.DateOffset(years=1)
  for name,part,boundary in [('train',tr,calstart),('validation',va,cut)]:
   for corridor,g in part.groupby('corridor'):
    dates=p[p.corridor.eq(corridor)].date.to_numpy();idx=np.searchsorted(dates,g.date.to_numpy());assert (dates[idx+5]<np.datetime64(boundary)).all()
   checks.append({'cutoff':r['cutoff'],'arm':r['model_id'],'part':name,'last_label':str(part.label_available_date.max())})
 history=pd.read_csv(OUT/'closing_history.csv.gz');fields=['target','closing_target','now_target','forward_bps','endpoint_bps','symmetric_bps','regret_bps'];assert history.loc[~history.calibration_eligible,fields].isna().all().all()
 # The two event definitions are mathematically distinct, not renamed scores.
 paths=[([100,99,101,102,103,105],False,True),([100]*6,True,False)]
 for rates,now,closing in paths:assert (rates[0]<=min(rates[1:]))==now and (rates[5]>rates[0])==closing
 return {'independent_endpoint_and_fifth_session_labels':True,'different_NOW_CLOSING_fixtures':True,'unmatured_history_outcomes_all_blank':True,'maturity':checks}

def checkpoint_parity(views,receipts):
 p=views[1];saved=pd.read_csv(OUT/'closing_predictions.csv.gz',parse_dates=['date']);history=pd.read_csv(OUT/'closing_history.csv.gz',parse_dates=['date']);results=[]
 for r in receipts:
  b=bundle(r);tr,va,te,hi=c.split(p,r['cutoff']);assert r['train_fingerprint']==c.fp(tr[b['features']+['corridor','target']]);assert r['validation_fingerprint']==c.fp(va[b['features']+['corridor','target']])
  for mode,lag in [('normal',1),('bank_delayed',2)]:
   z=views[lag].loc[te.index];raw,prob=manual_probability(b,z);old=saved[saved.model_id.eq(r['model_id'])&saved.cutoff.eq(r['cutoff'])&saved['mode'].eq(mode)];np.testing.assert_allclose(raw,old.raw_probability,rtol=0,atol=1e-12);np.testing.assert_allclose(prob,old.probability,rtol=0,atol=1e-12)
   results.append({'model_id':r['model_id'],'cutoff':r['cutoff'],'mode':mode,'rows':len(z),'manual_probability_max_error':float(np.max(abs(prob-old.probability.to_numpy())))})
  _,hp=manual_probability(b,hi);saved_hi=history[history.model_id.eq(r['model_id'])&history.cutoff.eq(r['cutoff'])];np.testing.assert_allclose(hp,saved_hi.probability,rtol=0,atol=1e-12)
 return results

def future_fit(views,features,receipts):
 rows=[]
 for r in receipts:
  cut=pd.Timestamp(r['cutoff']);q=views[1].copy();future=q.date>=cut;q.loc[future,features[r['model_id']]]*=100;unmatured=q.label_available_date.ge(cut)&q.target.notna();q.loc[unmatured,'target']=1-q.loc[unmatured,'target'];b,_=c.fit(q,features[r['model_id']],r['cutoff'],r['model_id']);old=bundle(r);_,_,test,_=c.split(views[1],r['cutoff'])
  assert b['threshold']==old['threshold'] and b['initial_state']==old['initial_state'] and b['model'].weight==old['model'].weight
  for a,z in zip(c.probabilities(b,test),c.probabilities(old,test)):np.testing.assert_array_equal(a,z)
  rows.append({'model_id':r['model_id'],'cutoff':r['cutoff'],'future_feature_x100_and_unmatured_label_flip_exact_fit_identity':True})
 return rows

def raw_prefix(views,features):
 original_csv=pd.read_csv;original_parquet=pd.read_parquet;cut=pd.Timestamp('2025-03-01')
 def read_csv(path,*args,**kwargs):
  f=original_csv(path,*args,**kwargs)
  if Path(path).name=='halyk_sell_daily.csv':f.loc[pd.to_datetime(f.date)>cut,'value']*=100
  return f
 def read_parquet(path,*args,**kwargs):
  f=original_parquet(path,*args,**kwargs)
  if Path(path).name=='feature_panel.parquet':f.loc[f.date>cut,c.TREASURY_LAG7_FEATURES]*=100
  return f
 with patch.object(pd,'read_csv',side_effect=read_csv),patch.object(pd,'read_parquet',side_effect=read_parquet):new,_=c.build_views()
 cols=['date','corridor',*features[c.ARMS[1]],'recent_low_rank5','recent_low_rate5','now_text_gate','closing_text_gate']
 for lag in (1,2):pd.testing.assert_frame_equal(views[lag].loc[views[lag].date<=cut,cols],new[lag].loc[new[lag].date<=cut,cols],check_exact=True)
 return {'Halyk_Treasury_future_values_x100':True,'lag1_lag2_past_features_and_fact_gates_exact':True}

def joint_checks(directory):
 manifest=json.loads((directory/'manifest.json').read_text());engine=HERE/manifest['preserved_engine_snapshot'] if 'preserved_engine_snapshot' in manifest else HERE/'joint_evaluate.py';assert c.sha(engine)==manifest['code_sha256']
 for rootpath,item in manifest['preserved_input_snapshots'].items():assert c.sha(HERE/item['path'])==item['sha256']
 data=pd.read_csv(directory/'predictions.csv.gz',parse_dates=['date']);reported=pd.read_csv(directory/'metrics.csv');policies=json.loads((directory/'policies.json').read_text());checks=[]
 for (cutoff,mode,variant),g in data.groupby(['cutoff','mode','variant']):
  policy=next(p for p in policies if p['cutoff']==cutoff);base=policy['past_baseline_rates'];threshold=policy['thresholds'];initial=policy['initial_states'][variant];fresh,_=j.schedule(g,threshold,base,policy['cooldown'],variant,initial)
  np.testing.assert_array_equal(fresh.signal,g.signal);np.testing.assert_array_equal(fresh.closing_annotation,g.closing_annotation)
  poison=g.copy();poison['now_target']=999;poison['closing_target']=-99;poison['forward_bps']=1e9;changed,_=j.schedule(poison,threshold,base,policy['cooldown'],variant,initial);np.testing.assert_array_equal(changed.signal,g.signal);np.testing.assert_array_equal(changed.closing_annotation,g.closing_annotation)
  prefix=g.iloc[:50];prefix_scored,_=j.schedule(prefix,threshold,base,policy['cooldown'],variant,initial);np.testing.assert_array_equal(prefix_scored.signal,g.signal.iloc[:50])
  row=reported[reported.cutoff.eq(cutoff)&reported['mode'].eq(mode)&reported.variant.eq(variant)].iloc[0]
  for scenario,label in [('NOW','now_target'),('CLOSING','closing_target')]:
   mask=(g.signal if scenario=='NOW' else g.closing_annotation) if variant=='dual_annotations' else g.selected_scenario.eq(scenario)
   assert int(mask.sum())==int(row[scenario+'_contacts'])
   if mask.any():assert abs(g.loc[mask,label].mean()/g[label].mean()-row[scenario+'_lift'])<1e-12
  if variant in ('NOW_only','NOW_confirmed_closing','dual_annotations'):np.testing.assert_array_equal(g.signal,g.root_now_signal)
  if variant=='dual_annotations':
   assert (g.closing_annotation<=g.signal).all();assert g.loc[g.signal,'selected_scenario'].eq('NOW').all();assert int(g.signal.sum())==int(row.NOW_contacts)
  checks.append({'cutoff':cutoff,'mode':mode,'variant':variant,'rows':len(g),'signals':int(g.signal.sum()),'future_outcome_and_prefix_invariant':True})
 return {'separate_truth_and_denominators_verified':True,'dual_tags_not_new_contacts':True,'checks':checks}

def adapter_suite():
 suite=unittest.defaultTestLoader.loadTestsFromModule(adapter_tests);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream).run(suite);assert result.wasSuccessful(),stream.getvalue();return {'tests':result.testsRun,'status':'PASS','stdout':stream.getvalue()}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--joint-dir',default=str(OUT/'joint'));args=parser.parse_args();started=time.monotonic();rows=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  views,features=c.build_views();receipts=json.loads((OUT/'model_receipts.json').read_text())
  tests=[('source_sha',sources),('independent_labels_maturity',lambda:labels_maturity(views,receipts)),('manual_checkpoint_and_history_parity',lambda:checkpoint_parity(views,receipts)),('future_fit_invariance',lambda:future_fit(views,features,receipts)),('raw_source_prefix_invariance',lambda:raw_prefix(views,features)),('joint_typed_metrics_and_causality',lambda:joint_checks(Path(args.joint_dir))),('scenario_adapter_suite',adapter_suite)]
  for name,fn in tests:
   clock=time.monotonic()
   try:details=fn();row={'check':name,'status':'PASS','details':details}
   except Exception as exc:row={'check':name,'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()}
   row['seconds']=time.monotonic()-clock;rows.append(row);print(name,row['status'],round(row['seconds'],2),flush=True)
 result={'status':'PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL','checks':rows,'joint_directory':str(Path(args.joint_dir).resolve().relative_to(HERE)),'verifier_sha256':c.sha(Path(__file__)),'seconds':time.monotonic()-started,'not_proven':['prospective performance','scenario joint robustness across all years','historical publication timing','bank execution savings']};c.save(OUT/'verification.json',result)
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
