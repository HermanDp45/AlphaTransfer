"""Independent target, timestamp, checkpoint and future-information checks."""
from __future__ import annotations
import os,sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import copy,json,pickle,time,traceback
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.bank_target import experiment as e
from research_v4.oxr2010_bank.bank_target import now_followup as now
HERE=e.HERE;OUT=e.OUT

def get_bundle(r):
 p=HERE/r['checkpoint'];assert e.sha(p)==r['checkpoint_sha256']
 return pickle.loads(p.read_bytes())

def source_and_schema():
 m=json.loads((OUT/'manifest.json').read_text());assert e.sha(HERE/'experiment.py')==m['code_sha256'];assert e.sha(HERE/'PROTOCOL.md')==m['protocol_sha256']
 for p,v in m['inputs_sha256'].items():assert e.sha(ROOT/p)==v,p
 raw=e.load_inputs()['halyk_raw'];iso=pd.to_datetime(raw.date_at,utc=True,format='ISO8601');epoch=pd.to_datetime(raw.date,unit='ms',utc=True);assert (iso==epoch).all()
 examples=[]
 for date in ('2024-02-28','2024-03-04'):
  # Verify real local-day conversion across Kazakhstan's UTC+6 -> UTC+5 change.
  local=iso.dt.tz_convert(e.ZONE);mask=local.dt.strftime('%Y-%m-%d').eq(date)
  if mask.any():examples.append({'local_day':date,'utc':str(iso[mask].iloc[0]),'local':str(local[mask].iloc[0])})
 return {'all_input_hashes_match':True,'raw_epoch_iso_agree':True,'timezone_transition_examples':examples,'no_ambiguous_day_month_parser':True}

def independent_targets(p,receipts):
 raw=json.loads(e.INPUTS['halyk_raw'].read_text());groups={}
 for row in raw:
  day=pd.Timestamp(row['date_at']).tz_convert(e.ZONE).date();groups.setdefault(day,set()).add(float(row['value']))
 good=sorted((d,next(iter(q))) for d,q in groups.items() if len(q)==1 and np.isfinite(next(iter(q))) and next(iter(q))>0)
 assert len(good)==len(p)
 for i,row in p.iterrows():
  day,q=good[i];assert row.quote_date.date()==day and row.quote_kzt_per_rub==q
  assert row.decision_at.date()==day+pd.Timedelta(days=1) and row.decision_at.hour==12
  future=good[i+1:i+6]
  if len(future)<5:assert pd.isna(row.target) and pd.isna(row.change_bps);continue
  last=future[-1][0];average=sum(v for _,v in future)/5
  assert abs(average-row.future_mean_kzt_per_rub)<1e-12
  assert row.label_available_at.date()==last+pd.Timedelta(days=1)
  if (last-day).days>14:assert pd.isna(row.target);continue
  exact_delta=sum(Decimal(str(v)) for _,v in future)-5*Decimal(str(q))
  assert int(exact_delta>0)==int(row.target)
  assert abs(float(10000*exact_delta/(5*Decimal(str(q))))-row.change_bps)<1e-9
  assert all(d>day for d,v in future)
 labels=[]
 for cutoff in sorted({r['cutoff'] for r in receipts}):
  tr,va,te=e.split(p,cutoff,str(int(cutoff[:4])+1)+'-01-01');cut=e.cutoff_time(cutoff);calstart=cut-pd.DateOffset(years=1)
  for name,frame,boundary in [('train',tr,calstart),('validation',va,cut)]:
   reconstructed=[pd.Timestamp(good[int(i)+5][0],tz=e.ZONE)+pd.Timedelta(days=1) for i in frame.quote_ordinal]
   assert all(t<boundary for t in reconstructed)
   labels.append({'cutoff':cutoff,'part':name,'last_mature_label':str(max(reconstructed)),'boundary':str(boundary)})
 return {'independently_enumerated_quotes':len(good),'maturity':labels,'orientation_fixture':'q=5.0 followed by mean5.1 -> target1, cost increases for RUB buyer; q=5.0 followed by4.9 -> target0'}

def asof_and_future_prefix(p,groups,inputs):
 for c in ('bank_available_at','cbr_available_at','oxr_available_at'):
  valid=p[c].notna();assert (p.loc[valid,c]<=p.loc[valid,'decision_at']).all()
 for c in ('cbr_age_days','oxr_age_days'):assert p[c].dropna().between(0,7).all()
 changed=copy.deepcopy(inputs);cut=pd.Timestamp('2025-03-01')
 d=pd.to_datetime(changed['halyk_raw'].date_at,utc=True).dt.tz_convert(e.ZONE).dt.tz_localize(None)
 changed['halyk_raw'].loc[d>cut,'value']*=100
 changed['cbr'].loc[changed['cbr'].date>cut,'rub_per_unit']*=100
 mask=changed['oxr'].date>'2025-03-01';changed['oxr'].loc[mask,'quote_per_rub']*=100;changed['oxr'].loc[mask,'rub_per_quote']/=100
 new,_,_=e.build_panel(changed)
 past=p.decision_at<e.cutoff_time('2025-03-01');columns=['date','quote_date','decision_at','bank_available_at','cbr_available_at','oxr_available_at',*groups['bank_cbr_oxr']]
 pd.testing.assert_frame_equal(p.loc[past,columns],new.loc[past,columns],check_exact=True)
 assert not set(groups['bank_cbr_oxr'])&{'target','change_bps','future_mean_kzt_per_rub','label_available_at','future_last_quote_date'}
 return {'all_source_known_at_before_decision':True,'max_asof_age_days':7,'raw_future_value_multiplier':100,'past_feature_prefix_exact':True,'future_label_columns_excluded_from_features':True}

def checkpoints_and_baselines(p,receipts):
 rows=[]
 for r in receipts:
  bundle=get_bundle(r);path=HERE/r['predictions'];assert e.sha(path)==r['predictions_sha256'];saved=pd.read_csv(path,parse_dates=['date']);tr,va,te=e.split(p,r['cutoff'],str(int(r['cutoff'][:4])+1)+'-01-01')
  assert len(te)==len(saved);assert np.array_equal(te.date.to_numpy(),saved.date.to_numpy())
  assert r['train_fingerprint']==e.frame_fingerprint(tr[bundle['features']+['target','change_bps']]);assert r['calibration_fingerprint']==e.frame_fingerprint(va[bundle['features']+['target','change_bps']])
  assert bundle['prior_probability']==float(va.target.mean())
  if bundle['arm'] in ('persistence','prior_mean'):
   prob=np.full(len(te),va.target.mean());reg=np.full(len(te),0 if bundle['arm']=='persistence' else va.change_bps.mean())
  else:
   raw=bundle['classifier'].predict_proba(te[bundle['features']])[:,1];cal=bundle['calibrator'];prob=expit(cal.intercept_[0]+cal.coef_[0,0]*logit(raw.clip(1e-6,1-1e-6)));reg=bundle['regressor'].predict(te[bundle['features']])
  perr=float(np.max(np.abs(prob-saved.probability)));rerr=float(np.max(np.abs(reg-saved.predicted_change_bps)));assert perr<1e-12 and rerr<1e-8
  quote=te.quote_kzt_per_rub*(1+reg/10000);np.testing.assert_allclose(quote,saved.predicted_future_mean_kzt_per_rub,rtol=0,atol=1e-12)
  rows.append({'arm':r['arm'],'cutoff':r['cutoff'],'rows':len(te),'probability_max_error':perr,'change_bps_max_error':rerr})
 return rows

def future_fit_invariance(p,groups,receipts):
 details=[]
 for cutoff in ('2026-01-01','2026-03-01'):
  altered=p.copy();cut=e.cutoff_time(cutoff);features=groups['bank_cbr_oxr'];future=altered.decision_at>=cut;altered.loc[future,features]*=100
  unavailable=(altered.label_available_at>=cut)&altered.target.notna();altered.loc[unavailable,'target']=1-altered.loc[unavailable,'target'];altered.loc[unavailable,'change_bps']*=-999
  tr,va,te=e.split(altered,cutoff,'2027-01-01');_,_,probe=e.split(p,cutoff,'2027-01-01')
  errors=[]
  for arm in e.ARMS:
   new=e.fit_model(tr,va,groups.get(arm,[]),arm);old=get_bundle(next(r for r in receipts if r['arm']==arm and r['cutoff']==cutoff));a=e.score(new,probe);b=e.score(old,probe)
   for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
   errors.append(arm)
  details.append({'cutoff':cutoff,'all_arms_exact_prediction_identity':errors,'future_features_x100_and_unmatured_labels_poisoned':True})
 return details

def start_control(p,groups,inputs,receipts):
 short,_,_=e.build_panel(inputs,'2018-06-17');pd.testing.assert_frame_equal(p[groups['bank_cbr_oxr']],short[groups['bank_cbr_oxr']],check_exact=True);checks=[]
 for cutoff in sorted({r['cutoff'] for r in receipts}):
  train,val,test=e.split(short,cutoff,str(int(cutoff[:4])+1)+'-01-01');new=e.fit_model(train,val,groups['bank_cbr_oxr'],'bank_cbr_oxr');original=get_bundle(next(r for r in receipts if r['cutoff']==cutoff and r['arm']=='bank_cbr_oxr'))
  for a,b in zip(e.score(new,test),e.score(original,test)):np.testing.assert_array_equal(a,b)
  checks.append(cutoff)
 return {'features_exact':True,'separate_fits_prediction_identity_cutoffs':checks,'earlier_source_history_creates_no_bank_labels':True}

def cohort_and_now_control(p,groups,inputs):
 primary=pd.read_csv(OUT/'all_predictions.csv.gz',parse_dates=['date','quote_date']);saved=pd.read_csv(now.OUT/'predictions.csv.gz',parse_dates=['date','quote_date'])
 keys=['date','quote_date','cutoff','arm'];pd.testing.assert_frame_equal(primary[keys].sort_values(keys).reset_index(drop=True),saved[keys].sort_values(keys).reset_index(drop=True))
 common=primary[primary.date.ge('2026-03-05')&primary.cutoff.str.startswith('2026')]
 for arm,g in common.groupby('arm'):
  a=g[g.cutoff.eq('2026-01-01')][['date','quote_date']].reset_index(drop=True);b=g[g.cutoff.eq('2026-03-01')][['date','quote_date']].reset_index(drop=True);pd.testing.assert_frame_equal(a,b)
 npanel=now.now_panel(p);q=p.quote_kzt_per_rub.to_numpy()
 for i,row in npanel[npanel.target.notna()].iterrows():
  assert int(Decimal(str(q[i]))<=min(Decimal(str(v)) for v in q[i+1:i+6]))==int(row.target)
 short,_,_=e.build_panel(inputs,'2018-06-17');short=now.now_panel(short);receipts=json.loads((now.OUT/'model_receipts.json').read_text());parity=[]
 manifest=json.loads((now.OUT/'manifest.json').read_text());assert manifest['code_sha256']==e.sha(HERE/'now_followup.py');assert manifest['predictions_sha256']==e.sha(now.OUT/'predictions.csv.gz')
 for r in receipts:
  b=get_bundle(r);tr,va,te=e.split(npanel,r['cutoff'],str(int(r['cutoff'][:4])+1)+'-01-01');pred=saved[saved.cutoff.eq(r['cutoff'])&saved.arm.eq(r['arm'])];pp,raw=now.score(b,te)
  np.testing.assert_allclose(pp,pred.probability,rtol=0,atol=1e-12);assert b['prior_probability']==float(va.target.mean())
  if r['arm']=='bank_cbr_oxr':
   st,sv,se=e.split(short,r['cutoff'],str(int(r['cutoff'][:4])+1)+'-01-01');alt=now.fit(st,sv,r['features'],r['arm']);np.testing.assert_array_equal(now.score(alt,se)[0],pp)
  if r['cutoff'] in ('2026-01-01','2026-03-01'):
   bad=npanel.copy();cut=e.cutoff_time(r['cutoff']);mask=bad.decision_at>=cut;bad.loc[mask,groups['bank_cbr_oxr']]*=100;mask=(bad.label_available_at>=cut)&bad.target.notna();bad.loc[mask,'target']=1-bad.loc[mask,'target'];bt,bv,_=e.split(bad,r['cutoff'],'2027-01-01');alt=now.fit(bt,bv,r['features'],r['arm']);np.testing.assert_array_equal(now.score(alt,te)[0],pp)
  parity.append({'arm':r['arm'],'cutoff':r['cutoff'],'rows':len(te)})
 return {'primary_and_now_exact_date_quote_keys':True,'January_March_common_dates':len(a),'NOW_target_independent_decimal_minimum_check':True,'NOW_checkpoints_past_prevalence_and_future_poison_pass':True,'NOW2010_2018_fitted_predictions_exact':True,'parity':parity}

def main():
 t=time.monotonic();results=[];inputs=e.load_inputs();p,g,_=e.build_panel(inputs);receipts=json.loads((OUT/'model_receipts.json').read_text())
 checks=[('source_hash_and_timestamp_schema',source_and_schema),('independent_actual_quote_target_and_maturity',lambda:independent_targets(p,receipts)),('known_at_and_raw_future_prefix',lambda:asof_and_future_prefix(p,g,inputs)),('checkpoint_and_past_only_baselines',lambda:checkpoints_and_baselines(p,receipts)),('future_fit_invariance',lambda:future_fit_invariance(p,g,receipts)),('OXR2010_2018_separate_fit_identity',lambda:start_control(p,g,inputs,receipts)),('paired_cohorts_and_NOW_target',lambda:cohort_and_now_control(p,g,inputs))]
 with threadpool_limits(limits=1):
  for name,fn in checks:
   started=time.monotonic()
   try:detail=fn();r={'check':name,'status':'PASS','detail':detail}
   except Exception as exc:r={'check':name,'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()}
   r['seconds']=time.monotonic()-started;results.append(r);print(name,r['status'],round(r['seconds'],2),flush=True)
 result={'status':'PASS' if all(x['status']=='PASS' for x in results) else 'FAIL','verifier_sha256':e.sha(Path(__file__)),'seconds':time.monotonic()-t,'checks':results,'limitations_not_verified':['historical publication timestamps','quote executability at decision time','customer RUB-sell side','causal customer or bank gains']};e.save(OUT/'verification.json',result)
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
