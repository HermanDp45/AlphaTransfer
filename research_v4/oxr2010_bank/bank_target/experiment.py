"""Predict archived BANK SELL RUB quotes; never a remittance execution backtest."""
from __future__ import annotations
import os,sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json,pickle,time,warnings
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';ZONE='Asia/Almaty'
INPUTS={'halyk_raw':ROOT/'research_v4/liquidity/raw/halyk_personal_rub.json','halyk_clean':ROOT/'research_v4/liquidity/halyk_sell_daily.csv','cbr':ROOT/'research_v3/models/panel_extended.pkl','oxr':HERE.parent/'input_oxr_snapshot.csv'}
ARMS=('persistence','prior_mean','bank_history','cbr_history','bank_cbr','bank_cbr_oxr')
HISTORY=['log_quote','ret1','ret5','ret20','vol20','rank60']
CALENDAR=['dow','month_sin','month_cos']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n')
def local_midnight(dates):return pd.to_datetime(dates).dt.tz_localize(ZONE)
def cutoff_time(x):return pd.Timestamp(x,tz=ZONE)
def history_features(q,prefix):
 log=np.log(q);out={prefix+'log_quote':log}
 for n in (1,5,20):out[prefix+f'ret{n}']=log.diff(n)
 # Window-local calculation avoids accumulated floating-point state depending
 # on an irrelevant older source prefix (important for the 2010/2018 control).
 out[prefix+'vol20']=log.diff().rolling(20,min_periods=10).apply(lambda x:np.std(x,ddof=1),raw=True)
 out[prefix+'rank60']=q.rolling(60,min_periods=20).rank(pct=True)
 return pd.DataFrame(out,index=q.index)

def load_inputs():return {'halyk_raw':pd.DataFrame(json.loads(INPUTS['halyk_raw'].read_text())),'cbr':pd.read_pickle(INPUTS['cbr']),'oxr':pd.read_csv(INPUTS['oxr'])}

def clean_bank(raw):
 raw=raw.copy();iso=pd.to_datetime(raw.date_at,utc=True,format='ISO8601');epoch=pd.to_datetime(raw.date,unit='ms',utc=True)
 assert (iso==epoch).all(),'Archive ISO/epoch date disagreement'
 raw['quote_date']=iso.dt.tz_convert(ZONE).dt.tz_localize(None).dt.normalize();raw['value']=pd.to_numeric(raw.value,errors='coerce')
 conflicts=raw.groupby('quote_date').value.nunique().gt(1);bad=conflicts[conflicts].index;valid=np.isfinite(raw.value)&raw.value.gt(0)
 clean=raw[~raw.quote_date.isin(bad)&valid].drop_duplicates('quote_date').sort_values('quote_date').reset_index(drop=True)
 audit={'raw_rows':len(raw),'iso_epoch_exact_matches':int((iso==epoch).sum()),'conflicting_dates_excluded':len(bad),'nonpositive_or_nonfinite_rows_excluded':int((~valid).sum()),'clean_quotes':len(clean),'first_quote':str(clean.quote_date.min().date()),'last_quote':str(clean.quote_date.max().date()),'date_interpretation':'ISO UTC and epoch milliseconds agree; effective day in historical Asia/Almaty zone, not publication timestamp','conflict_dates':[str(x.date()) for x in bad],'side':'BANK SELLS RUB; KZT paid per RUB; lower is better for customer buying RUB'}
 return clean[['quote_date','value']],audit

def asof(left,right,key,prefix,feature_columns):
 right=right.sort_values(key);out=pd.merge_asof(left.sort_values('decision_at'),right,left_on='decision_at',right_on=key,direction='backward',tolerance=pd.Timedelta(days=7))
 valid=out[key].notna();assert (out.loc[valid,key]<=out.loc[valid,'decision_at']).all()
 out[prefix+'age_days']=(out.decision_at-out[key]).dt.total_seconds()/86400
 out[prefix+'available']=valid.astype(float)
 return out

def build_panel(inputs=None,oxr_since='2010-01-01'):
 inputs=load_inputs() if inputs is None else inputs
 bank,audit=clean_bank(inputs['halyk_raw']);p=bank.rename(columns={'value':'quote_kzt_per_rub'}).copy();p['quote_ordinal']=np.arange(len(p));p['date']=p.quote_date+pd.Timedelta(days=1)
 p['bank_available_at']=local_midnight(p.date);p['decision_at']=p.bank_available_at+pd.Timedelta(hours=12)
 p=pd.concat([p,history_features(p.quote_kzt_per_rub,'bank_')],axis=1)
 p['bank_previous_gap_days']=p.quote_date.diff().dt.days
 p['dow']=p.date.dt.dayofweek;p['month_sin']=np.sin(2*np.pi*p.date.dt.month/12);p['month_cos']=np.cos(2*np.pi*p.date.dt.month/12)
 # Exact next-five observations; skip the anchor, do not fill gaps.
 # Raw personal RUB quotes have two decimal digits. Integer tick comparisons
 # preserve exact ties; binary float means otherwise flipped three flat labels.
 ticks=np.rint(p.quote_kzt_per_rub*100).astype('int64')
 assert np.allclose(p.quote_kzt_per_rub*100,ticks,rtol=0,atol=1e-10)
 future=pd.concat({str(n):ticks.shift(-n) for n in range(1,6)},axis=1)
 future_sum=future.sum(axis=1,min_count=5)
 p['future_mean_kzt_per_rub']=future_sum/500
 p['future_last_quote_date']=p.quote_date.shift(-5);p['label_available_at']=local_midnight(p.future_last_quote_date+pd.Timedelta(days=1))
 p['horizon_calendar_days']=(p.future_last_quote_date-p.quote_date).dt.days
 valid=p.future_mean_kzt_per_rub.notna()&p.horizon_calendar_days.le(14)
 p['change_bps']=10000*(future_sum-5*ticks)/(5*ticks)
 p['target']=(future_sum>5*ticks).astype(float).where(valid)
 p.loc[~valid,'change_bps']=np.nan
 c=inputs['cbr'];c=c[c.corridor.eq('KZT')][['date','rub_per_unit']].sort_values('date').reset_index(drop=True)
 c['cbr_observation_date']=c.date;c['cbr_quote']=1/c.rub_per_unit
 c=pd.concat([c,history_features(c.cbr_quote,'cbr_')],axis=1)
 c['cbr_available_at']=(pd.to_datetime(c.date)+pd.Timedelta(days=1)).dt.tz_localize('Europe/Moscow').dt.tz_convert(ZONE)
 p=asof(p,c[['cbr_available_at','cbr_observation_date','cbr_quote',*['cbr_'+x for x in HISTORY]]],'cbr_available_at','cbr_',[])
 o=inputs['oxr'];o=o[o.quote.eq('KZT')&o.date.ge(oxr_since)].sort_values('date').copy().reset_index(drop=True)
 assert not o.date.duplicated().any()
 o['oxr_observation_date']=pd.to_datetime(o.date);o['oxr_published_at']=pd.to_datetime(o.published_at_utc,utc=True)
 assert (o.quote_per_rub>0).all() and np.allclose(o.quote_per_rub*o.rub_per_quote,1,atol=1e-10)
 complete=o.oxr_observation_date.dt.tz_localize('UTC')+pd.Timedelta(days=1)
 o['oxr_available_at']=(pd.concat([complete,o.oxr_published_at],axis=1).max(axis=1)+pd.Timedelta(hours=24)).dt.tz_convert(ZONE)
 o['oxr_quote']=o.quote_per_rub;o=pd.concat([o,history_features(o.oxr_quote,'oxr_')],axis=1)
 p=asof(p,o[['oxr_available_at','oxr_observation_date','oxr_published_at','oxr_quote',*['oxr_'+x for x in HISTORY]]],'oxr_available_at','oxr_',[])
 p['bank_cbr_basis']=np.log(p.quote_kzt_per_rub/p.cbr_quote);p['bank_oxr_basis']=np.log(p.quote_kzt_per_rub/p.oxr_quote);p['cbr_oxr_basis']=np.log(p.cbr_quote/p.oxr_quote)
 groups={'bank_history':['bank_'+x for x in HISTORY]+['bank_previous_gap_days']+CALENDAR,'cbr_history':['cbr_'+x for x in HISTORY]+['cbr_available','cbr_age_days']+CALENDAR}
 groups['bank_cbr']=list(dict.fromkeys(groups['bank_history']+groups['cbr_history']+['bank_cbr_basis']))
 groups['bank_cbr_oxr']=groups['bank_cbr']+['oxr_'+x for x in HISTORY]+['oxr_available','oxr_age_days','bank_oxr_basis','cbr_oxr_basis']
 audit.update({'decision_rows':len(p),'eligible_labels':int(p.target.notna().sum()),'windows_over_14_days':int(p.horizon_calendar_days.gt(14).sum()),'unknown_final_labels':int(p.future_mean_kzt_per_rub.isna().sum()),'median_horizon_calendar_days':float(p.horizon_calendar_days.median()),'max_horizon_calendar_days':float(p.horizon_calendar_days.max()),'exact_flat_future_mean_rows':int((p.future_mean_kzt_per_rub==p.quote_kzt_per_rub).sum()),'available_cbr_fraction':float(p.cbr_available.mean()),'available_oxr_fraction':float(p.oxr_available.mean()),'oxr_start':str(o.oxr_observation_date.min().date()),'oxr_end':str(o.oxr_observation_date.max().date()),'bank_history_limits_supervised_training':'2020 start; older OXR does not create bank labels'})
 return p,groups,audit

def split(p,cutoff,end):
 cut=cutoff_time(cutoff);start=cut-pd.DateOffset(years=1);end=cutoff_time(end);ok=p.target.notna()
 train=p[ok&(p.decision_at<start)&(p.label_available_at<start)].copy()
 validation=p[ok&(p.decision_at>=start)&(p.decision_at<cut)&(p.label_available_at<cut)].copy()
 test=p[ok&(p.decision_at>=cut)&(p.decision_at<end)&(p.label_available_at<end)].copy()
 assert train.label_available_at.max()<start and validation.label_available_at.max()<cut
 assert len(train)>100 and len(validation)>100
 return train,validation,test

def make_model(reg=False):
 cls=HistGradientBoostingRegressor if reg else HistGradientBoostingClassifier
 return cls(max_iter=120,max_depth=2,learning_rate=.05,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=1701)

def fit_model(train,validation,features,arm):
 bundle={'arm':arm,'features':features,'prior_probability':float(validation.target.mean()),'prior_mean_bps':float(validation.change_bps.mean())}
 if arm in ('persistence','prior_mean'):return bundle
 x=train[features];classifier=make_model().fit(x,train.target.astype(int));regressor=make_model(True).fit(x,train.change_bps)
 raw=classifier.predict_proba(validation[features])[:,1];cal=LogisticRegression(C=1.,solver='lbfgs',max_iter=500).fit(logit(raw.clip(1e-6,1-1e-6)).reshape(-1,1),validation.target.astype(int))
 bundle.update(classifier=classifier,regressor=regressor,calibrator=cal);return bundle

def score(bundle,p):
 if bundle['arm'] in ('persistence','prior_mean'):
  return np.full(len(p),bundle['prior_probability']),np.full(len(p),0. if bundle['arm']=='persistence' else bundle['prior_mean_bps']),np.full(len(p),bundle['prior_probability'])
 raw=bundle['classifier'].predict_proba(p[bundle['features']])[:,1]
 prob=bundle['calibrator'].predict_proba(logit(raw.clip(1e-6,1-1e-6)).reshape(-1,1))[:,1]
 return prob,bundle['regressor'].predict(p[bundle['features']]),raw

def frame_fingerprint(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()

def evaluate(p,groups,cutoff,end):
 train,val,test=split(p,cutoff,end);rows=[];receipts=[]
 for arm in ARMS:
  f=groups.get(arm,[]);t=time.monotonic();bundle=fit_model(train,val,f,arm);bundle.update(cutoff=cutoff,end=end)
  proba,delta,raw=score(bundle,test)
  output=test[['date','quote_date','decision_at','label_available_at','future_last_quote_date','quote_ordinal','quote_kzt_per_rub','future_mean_kzt_per_rub','target','change_bps','horizon_calendar_days']].copy()
  output['probability']=proba;output['raw_probability']=raw;output['predicted_change_bps']=delta;output['predicted_future_mean_kzt_per_rub']=output.quote_kzt_per_rub*(1+delta/10000);output['arm']=arm;output['cutoff']=cutoff
  ckpt=OUT/'checkpoints'/f'{arm}_{cutoff}.pkl';ckpt.write_bytes(pickle.dumps(bundle));dest=OUT/'predictions'/f'{arm}_{cutoff}.csv.gz';output.to_csv(dest,index=False)
  receipt={'arm':arm,'cutoff':cutoff,'features':f,'train_rows':len(train),'train_min':train.decision_at.min(),'train_max':train.decision_at.max(),'train_latest_label':train.label_available_at.max(),'validation_rows':len(val),'validation_min':val.decision_at.min(),'validation_max':val.decision_at.max(),'validation_latest_label':val.label_available_at.max(),'test_rows':len(test),'test_min':test.decision_at.min(),'test_max':test.decision_at.max(),'prior_probability':bundle['prior_probability'],'prior_mean_bps':bundle['prior_mean_bps'],'train_fingerprint':frame_fingerprint(train[f+['target','change_bps']]),'calibration_fingerprint':frame_fingerprint(val[f+['target','change_bps']]),'checkpoint':str(ckpt.relative_to(HERE)),'checkpoint_sha256':sha(ckpt),'predictions':str(dest.relative_to(HERE)),'predictions_sha256':sha(dest),'seconds':time.monotonic()-t}
  rows.append(output);receipts.append(receipt)
 print(cutoff,'train',len(train),'cal',len(val),'test',len(test),flush=True)
 return pd.concat(rows,ignore_index=True),receipts

def metric(g):
 y=g.target.to_numpy();p=g.probability.to_numpy();err=(g.predicted_change_bps-g.change_bps).to_numpy();qerr=(g.predicted_future_mean_kzt_per_rub-g.future_mean_kzt_per_rub).to_numpy()
 return {'rows':len(g),'dates':g.date.nunique(),'positive_share':float(y.mean()),'brier':float(np.mean((p-y)**2)),'raw_brier':float(np.mean((g.raw_probability-y)**2)),'log_loss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,'mae_change_bps':float(np.mean(np.abs(err))),'rmse_change_bps':float(np.sqrt(np.mean(err**2))),'mae_future_quote_kzt_per_rub':float(np.mean(np.abs(qerr))),'rmse_future_quote_kzt_per_rub':float(np.sqrt(np.mean(qerr**2)))}

def summarize(pred,aggregate=False):
 keys=['arm'] if aggregate else ['cutoff','arm'];rows=[]
 for key,g in pred.groupby(keys):
  values=key if isinstance(key,tuple) else (key,);rows.append({**dict(zip(keys,values)),**metric(g)})
 out=pd.DataFrame(rows)
 if aggregate:
  ref=out[out.arm.eq('persistence')].iloc[0];out['brier_skill_vs_past_prevalence']=1-out.brier/ref.brier;out['mae_skill_vs_persistence']=1-out.mae_change_bps/ref.mae_change_bps
 else:
  refs=out[out.arm.eq('persistence')].set_index('cutoff');out['brier_skill_vs_past_prevalence']=1-out.brier/out.cutoff.map(refs.brier);out['mae_skill_vs_persistence']=1-out.mae_change_bps/out.cutoff.map(refs.mae_change_bps)
 return out

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
 before={str(path.relative_to(ROOT)):sha(path) for path in INPUTS.values()};data=load_inputs();p,groups,audit=build_panel(data);p18,groups18,audit18=build_panel(data,'2018-06-17');features=groups['bank_cbr_oxr'];pd.testing.assert_frame_equal(p[features],p18[features],check_exact=True)
 old=pd.read_csv(INPUTS['halyk_clean'],parse_dates=['date']);old=old[old.client.eq('personal')&old.currency.eq('RUB')];assert old.bank_side.eq('sell').all();fresh=p[['quote_date','quote_kzt_per_rub']].rename(columns={'quote_date':'date','quote_kzt_per_rub':'value'}).reset_index(drop=True);pd.testing.assert_frame_equal(fresh,old[['date','value']].reset_index(drop=True),check_exact=True)
 save(OUT/'source_audit.json',audit);save(OUT/'feature_groups.json',groups);save(OUT/'source_start_control.json',{'start2010_vs2018_features_exact':True,'bank_rows':len(p),'features':len(features),'source2010_start':audit['oxr_start'],'source2018_start':audit18['oxr_start'],'early_oxr_cannot_create_bank_labels':True})
 p.to_pickle(OUT/'panel.pkl');pred=[];receipts=[]
 with threadpool_limits(limits=1):
  for cut in ('2023-01-01','2024-01-01','2025-01-01'):
   x,r=evaluate(p,groups,cut,str(int(cut[:4])+1)+'-01-01');pred.append(x);receipts.extend(r)
  development=pd.concat(pred,ignore_index=True);summarize(development,True).to_csv(OUT/'development_summary.csv',index=False)
  save(OUT/'development_protocol_result.json',{'primary_contrast':'bank_cbr_oxr minus bank_cbr','all_arms_reported':True,'no_2026_selection':True,'protocol_sha256':sha(HERE/'PROTOCOL.md'),'frozen_before_new_2026_fit':True})
  for cut in ('2026-01-01','2026-03-01'):
   x,r=evaluate(p,groups,cut,'2027-01-01');pred.append(x);receipts.extend(r)
 allpred=pd.concat(pred,ignore_index=True);allpred.to_csv(OUT/'all_predictions.csv.gz',index=False);summarize(allpred).to_csv(OUT/'metrics_by_cutoff.csv',index=False)
 common=allpred[(allpred.date>=pd.Timestamp('2026-03-05'))&allpred.cutoff.str.startswith('2026')];summarize(common).to_csv(OUT/'common_march5_summary.csv',index=False)
 save(OUT/'model_receipts.json',receipts);assert before=={path:sha(ROOT/path) for path in before}
 save(OUT/'manifest.json',{'status':'complete','code_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'PROTOCOL.md'),'inputs_sha256':before,'seconds':time.monotonic()-started,'rows':len(p),'source_start_identity':True,'target':'next5 actual BANK SELL RUB quote mean; KZT per RUB, lower favors buying RUB','availability':'archive effective local day+1 proxy; publication timestamps and stale-quote executability unverified','supervised_training_history':'2020 onward only; OXR2010 cannot synthesize bank labels','2026_status':'retrospective, previously inspected market history'})
 print(summarize(development,True).round(5).to_string(index=False),flush=True)
 print('COMPLETE',round(time.monotonic()-started,2),flush=True)
if __name__=='__main__':main()
