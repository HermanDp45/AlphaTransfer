"""Independent numeric, immutable-input and cutoff tests for robustness continuation."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import copy,json,pickle,time,traceback,warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_v4.continuation.robustness import experiment as e
from research_v4.continuation.robustness import age_followup as a
OUT=e.OUT;HERE=e.HERE

def read_bundle(receipt):
 path=HERE/receipt['checkpoint'];assert e.digest(path)==receipt['checkpoint_sha256']
 with path.open('rb') as f:return pickle.load(f)

def manual_model(model,frame):
 if not hasattr(model,'pooled'):return model.predict_proba(frame)[:,1]
 p=model.pooled.predict_proba(frame)[:,1];z=model.pooled.named_steps['preprocessor'].transform(frame);assert z.dtype==np.float64
 correction=np.zeros(len(frame))
 for tree in model.trees:correction+=model.learning_rate*tree.predict(z)
 adapted=expit(logit(p.clip(1e-7,1-1e-7))+correction);weight=getattr(model,'weight',1.)
 return weight*adapted+(1-weight)*p

def manual_score(bundle,frames,rule,delay):
 m=bundle['models']
 if rule.startswith('v3_long_'):raw=manual_model(m['v3'],frames[1])
 elif rule=='halyk_l1':raw=manual_model(m['h1'],frames[1+delay])
 elif rule=='halyk_l2_retrained':raw=manual_model(m['h2'],frames[2+delay])
 elif rule=='lag_augmentation':raw=manual_model(m['aug'],frames[1+delay])
 elif rule=='feature_dropout':raw=manual_model(m['drop'],frames[1+delay])
 elif rule=='lag_ensemble':raw=.5*(manual_model(m['h1'],frames[1+delay])+manual_model(m['h2'],frames[2+delay]))
 elif rule=='minimax_shrink_v3':return bundle['blend_alpha']*manual_score(bundle,frames,'halyk_l1',delay)+(1-bundle['blend_alpha'])*manual_score(bundle,frames,'v3_long_globalcal',0)
 else:raise AssertionError(rule)
 cal=bundle['calibrators'][rule]
 return raw if cal.model is None else expit(cal.intercept+cal.slope*logit(raw.clip(1e-6,1-1e-6)))

def manual_selection(frame,p,threshold,last=-10000):
 choices=[]
 for s,prob in zip(frame.session_ordinal,p):
  yes=bool(prob>=threshold and int(s)-last>3);choices.append(yes)
  if yes:last=int(s)
 return np.array(choices),last

def immutable_seals():
 count=0;by_package={};document_backups=[]
 for package in ('research_v3','research_v4'):
  path=ROOT/package/('manifest.json' if package=='research_v3' else 'continuation/previous_v4_manifest.json');manifest=json.loads(path.read_text());by_package[package]=len(manifest['files'])
  for relative,item in manifest['files'].items():
   expected=item if isinstance(item,str) else item['sha256'];source=ROOT/relative if package=='research_v3' else ROOT/package/relative
   if package=='research_v4' and relative in ('REPORT.md','README.md'):
    source=ROOT/package/'continuation/prior_reports'/relative;document_backups.append(str(source.relative_to(ROOT)))
   assert e.digest(source)==expected,str(source);count+=1
 for relative,expected in json.loads((OUT/'manifest.json').read_text())['input_sha256'].items():
  source=ROOT/'research_v4/continuation/previous_v4_manifest.json' if relative=='research_v4/artifact_manifest.json' else ROOT/relative
  assert e.digest(source)==expected,relative
 return {'sealed_old_artifact_files':count,'by_package':by_package,'old_manifests_and_sources_unchanged':True,'old_v4_manifest':'research_v4/continuation/previous_v4_manifest.json','old_root_documents_verified_at_backups':document_backups,'all_other_old_files_verified_live':True}

def numeric_reconstruction(views,receipts):
 results=[]
 for r in receipts:
  bundle=read_bundle(r);dest=HERE/r['prediction'];assert e.digest(dest)==r['prediction_sha256'];saved=pd.read_csv(dest,parse_dates=['date']);p=views[1];indexer=pd.Series(p.index,index=pd.MultiIndex.from_frame(p[['date','corridor']]))
  cutoff=pd.Timestamp(r['cutoff']);history=p[p.corridor.eq('KZT')&(p.date>=cutoff-pd.DateOffset(months=12))&(p.date<cutoff)]
  history_frames={lag:f.loc[history.index,bundle['features']+['corridor']] for lag,f in views.items()}
  for (rule,delay),g in saved.groupby(['rule','delay']):
   idx=indexer.loc[pd.MultiIndex.from_frame(g[['date','corridor']])].to_numpy();frame=p.loc[idx];frames={lag:f.loc[idx,bundle['features']+['corridor']] for lag,f in views.items()};prob=manual_score(bundle,frames,rule,int(delay));error=float(np.max(np.abs(prob-g.probability.to_numpy())));assert error<1e-12
   hp=manual_score(bundle,history_frames,rule,0);_,last=manual_selection(history,hp,bundle['thresholds'][rule]['KZT']);assert last==bundle['initial_states'][rule].get('KZT',-10000)
   selected,_=manual_selection(frame,prob,bundle['thresholds'][rule]['KZT'],last);assert np.array_equal(selected,g.candidate_signal.to_numpy())
   results.append({'cutoff':r['cutoff'],'rule':rule,'delay':int(delay),'rows':len(g),'manual_probability_max_error':error,'candidate_exact_match':True,'explicit_cutoff_history_state_exact_match':True})
 return results

def maturity_and_parity(views,receipts):
 rows=[];p=views[1]
 for r in receipts:
  cutoff=pd.Timestamp(r['cutoff']);tr,va,hi,te=e.splits(p,cutoff,pd.Timestamp('2027-01-01') if cutoff.year==2026 else pd.Timestamp(cutoff.year+1,1,1));cal_start=cutoff-pd.DateOffset(months=12)
  for label,part,boundary in [('train',tr,cal_start),('calibration',va,cutoff)]:
   latest=[]
   for c,g in part.groupby('corridor'):
    all_dates=p[p.corridor.eq(c)].date.to_numpy();pos=np.searchsorted(all_dates,g.date.to_numpy());mature=all_dates[pos+5];assert np.all(mature<np.datetime64(boundary));latest.append(pd.Timestamp(mature.max()))
   rows.append({'cutoff':r['cutoff'],'part':label,'rows':len(part),'max_independently_computed_label_date':str(max(latest).date()),'boundary':str(boundary.date())})
  assert (hi.date>=cutoff-pd.DateOffset(months=12)).all() and (hi.date<cutoff).all()
 assert pd.Timestamp(receipts[-1]['calibration_end']).year==2026
 parity=e.old_parity(pd.read_csv(OUT/'all_predictions.csv.gz',parse_dates=['date']))
 return {'maturity':rows,'annual_old_parity':parity,'January_March_common_dates':120}

def future_fit_invariance(views,extra,receipts):
 records=[]
 for cut in ('2026-01-01','2026-03-01'):
  receipt=next(r for r in receipts if r['cutoff']==cut);original=read_bundle(receipt);cutoff=pd.Timestamp(cut);changed={}
  for lag,p in views.items():
   q=p.copy();future=q.date>=cutoff;q.loc[future,e.BASE+extra]=q.loc[future,e.BASE+extra]*100
   unavailable=q.label_available_date>=cutoff;q.loc[unavailable,'target']=1-q.loc[unavailable,'target'];changed[lag]=q
  fitted,_,test=e.fit(changed,extra,cutoff,pd.Timestamp('2027-01-01'));assert fitted['blend_alpha']==original['blend_alpha'];assert fitted['thresholds']==original['thresholds'];assert fitted['initial_states']==original['initial_states'];assert fitted['fit_metadata']['training_feature_fingerprints']==original['fit_metadata']['training_feature_fingerprints']
  probe=views[1][views[1].corridor.eq('KZT')&(views[1].date>=cutoff)].iloc[:70];frames={lag:p.loc[probe.index,e.BASE+extra+['corridor']] for lag,p in views.items()}
  maxerr=0.
  for rule in e.RULES:
   for delay in (0,1):
    err=float(np.max(np.abs(manual_score(original,frames,rule,delay)-manual_score(fitted,frames,rule,delay))));maxerr=max(maxerr,err);assert err<1e-14
  oldage,_,_=a.fit_age(original,views);newage,_,_=a.fit_age(fitted,changed);assert oldage==newage
  records.append({'cutoff':cut,'future_feature_multiplier':100,'unmatured_and_future_targets_flipped':True,'all_fit_probabilities_max_error':maxerr,'thresholds_and_history_states_unchanged':True,'age_threshold_and_history_fit_unchanged':True})
 return records

def raw_future_prefix_lag3(views):
 original=pd.read_csv;cutoff=pd.Timestamp('2026-03-01');counts={}
 def poisoned(path,*args,**kwargs):
  f=original(path,*args,**kwargs)
  if Path(path).name=='halyk_sell_daily.csv':
   mask=pd.to_datetime(f.date)>cutoff;f.loc[mask,'value']=f.loc[mask,'value']*100;counts[str(Path(path).name)]=int(mask.sum())
  return f
 with patch.object(pd,'read_csv',side_effect=poisoned):other,_=e.build_views()
 for lag in (1,2,3):pd.testing.assert_frame_equal(views[lag][views[lag].date<=cutoff],other[lag][other[lag].date<=cutoff],check_exact=True)
 assert counts['halyk_sell_daily.csv']>0
 return {'lag_views':[1,2,3],'cutoff':str(cutoff.date()),'changed_raw_rows':counts,'all_past_features_and_age_metadata_exactly_unchanged':True}

def observable_age_and_causal_policy(views,receipts):
 frame=pd.DataFrame({'date':pd.to_datetime(['2026-02-20']*3),**{c:pd.to_datetime(['2026-02-19','2026-02-16',None]) for c in a.OBS_COLUMNS}});ages=a.quote_age(frame);np.testing.assert_array_equal(ages,[1,4,np.inf]);p,mask=a.gate_probability(ages,np.array([.8]*3),np.array([.3]*3),2);np.testing.assert_array_equal(p,[.8,.3,.3]);assert mask.tolist()==[False,True,True]
 copyframe=frame.assign(hidden_delay_flag=[1,0,0]);np.testing.assert_array_equal(a.quote_age(copyframe),ages)
 bundle=read_bundle(receipts[-1]);_,_,_,test=e.splits(views[1],pd.Timestamp(bundle['cutoff']),pd.Timestamp(bundle['end']));full=e.predict(bundle,views,test);tail_cut=test[test.corridor.eq('KZT')].date.iloc[40];prefix=e.predict(bundle,views,test[test.date<=tail_cut]);cols=['date','rule','delay','probability','candidate_signal'];pd.testing.assert_frame_equal(full[full.date<=tail_cut][cols].reset_index(drop=True),prefix[cols].reset_index(drop=True))
 poisoned=test.copy();poisoned['target']=999;poisoned['forward_bps']=-1e9;check=e.predict(bundle,views,poisoned);pd.testing.assert_frame_equal(full[cols],check[cols])
 return {'age_gate_uses_visible_observation_dates_only':True,'missing_feed_causes_fallback':True,'fresh_probability_is_exact_original':True,'future_test_rows_do_not_change_prefix_candidates':True,'test_outcomes_do_not_change_probabilities_or_candidates':True}

def standardized_metric_contract():
 # Deliberately unequal yearly base rates; naive pooled lift is 2.5, standardized1.6.
 fixture=pd.DataFrame({'cutoff':['2023-01-01']*2+['2024-01-01']*8,'corridor':['KZT']*10,'target':[1,0,1,0,0,0,0,0,0,0],'forward_bps':[10,30,0,10,20,30,40,50,60,70],'regret_bps':[0]*10,'candidate_signal':[True,False,False,False,False,False,False,False,False,True]})
 x=e.candidate_comparison(fixture);assert abs(x['candidate_lift_standardized']-1.6)<1e-12;assert abs(x['candidate_lift_unstandardized']-2.5)<1e-12;assert abs(x['forward_delta_bps']-12.5)<1e-12
 from research_v3.models import experiment as old
 observed=pd.read_csv(OUT/'development_summary.csv');path=ROOT/'research_v3/models/basis_train_120m_h5_predictions.csv.gz';baseline=pd.read_csv(path,parse_dates=['date']);baseline=baseline[baseline.corridor.eq('KZT')];legacy=old.summarize(baseline);legacy=legacy[legacy.track.eq('development_2023_2025')].iloc[0];now=observed[observed.rule.eq('v3_long_globalcal')&observed.delay.eq(0)].iloc[0]
 assert abs(legacy.lift-now.candidate_lift_standardized)<1e-12 and abs(legacy.forward_delta_bps-now.forward_delta_bps)<1e-10
 return {'unequal_year_prevalence_fixture':x,'old_v3_long_standardized_lift_match':True,'old_v3_long_forward_delta_match':True}

def main():
 start=time.monotonic();results=[];receipts=json.loads((OUT/'model_receipts.json').read_text())
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning);views,extra=e.build_views()
  tests=[('old_seals_immutable',immutable_seals),('standardized_metric_contract',standardized_metric_contract),('manual_checkpoint_and_candidate_reconstruction',lambda:numeric_reconstruction(views,receipts)),('independent_label_maturity_and_old_parity',lambda:maturity_and_parity(views,receipts)),('future_labels_and_features_cannot_change_fit',lambda:future_fit_invariance(views,extra,receipts)),('raw_halyk_future_prefix_lag3',lambda:raw_future_prefix_lag3(views)),('observable_age_and_policy_causality',lambda:observable_age_and_causal_policy(views,receipts))]
  for name,fn in tests:
   t=time.monotonic()
   try:detail=fn();r={'check':name,'status':'PASS','detail':detail}
   except Exception as exc:r={'check':name,'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()}
   r['seconds']=time.monotonic()-t;results.append(r);print(name,r['status'],round(r['seconds'],2),flush=True)
 result={'status':'PASS' if all(x['status']=='PASS' for x in results) else 'FAIL','runtime_seconds':time.monotonic()-start,'verifier_sha256':e.digest(Path(__file__)),'checks':results,'not_proven':'Actual historical publication timestamps, prospective model performance, customer uplift.'};(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 if result['status']!='PASS':sys.exit(1)
if __name__=='__main__':main()
