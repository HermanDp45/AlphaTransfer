#!/usr/bin/env python3
"""Independent, read-only artifact/temporal verification of V4 market research.
Only market_verification.json is written. Training, sources and sealed V3 stay intact.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import contextlib,copy,csv,gzip,hashlib,io,json,pickle,runpy,sys,time,traceback,warnings
from pathlib import Path
from datetime import date
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.kazakhstan import experiment as kz
from research_v4.liquidity import experiment as liq


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def check_hash(path,expected):
 got=digest(path)
 if got!=expected:raise AssertionError(f'SHA256 mismatch {path}: expected {expected}, got {got}')

def receipts():
 result={'kazakhstan_models':0,'kazakhstan_checkpoints':0,'liquidity_models':0,'liquidity_checkpoints':0,'input_hashes':0,'source_raw_receipts':0}
 for path in sorted((HERE/'kazakhstan').glob('*_receipt.json')):
  r=json.loads(path.read_text());check_hash(path.with_name(path.name.replace('_receipt.json','_predictions.csv.gz')),r['prediction_sha256']);result['kazakhstan_models']+=1
  for f in r['fits']:check_hash(HERE/'kazakhstan'/f['checkpoint'],f['sha256']);result['kazakhstan_checkpoints']+=1
  for p,h in r.get('inputs_sha256',{}).items():check_hash(ROOT/p,h);result['input_hashes']+=1
  check_hash(ROOT/r['source_panel'],r['source_sha256'])
 for name in ('model_receipts.json','combo_receipts.json'):
  path=HERE/'liquidity'/name
  if not path.exists():raise AssertionError(f'Missing final receipt {name}')
  for r in json.loads(path.read_text()):
   check_hash(HERE/'liquidity/predictions'/f"{r['name']}.csv.gz",r['prediction_sha256']);check_hash(HERE/'liquidity/checkpoints'/f"{r['name']}.pkl",r['checkpoint_sha256']);result['liquidity_models']+=1;result['liquidity_checkpoints']+=1
   for p,h in r.get('inputs_sha256',{}).items():check_hash(ROOT/p,h);result['input_hashes']+=1
 for path in sorted((HERE/'liquidity').glob('feature_receipt_*.json')):
  r=json.loads(path.read_text())
  for p,h in r['inputs_sha256'].items():check_hash(ROOT/p,h);result['input_hashes']+=1
 for name in ('kase_receipts.json','halyk_receipts.json'):
  r=json.loads((HERE/'liquidity'/name).read_text())
  for row in r['rows']:
   if 'file' in row and 'sha256' in row:check_hash(HERE/'liquidity'/row['file'],row['sha256']);result['source_raw_receipts']+=1
 return result


def checkpoint_recompute():
 panel_cache={};model_cache={};rows=[];bases=[]
 def model(path):
  if path not in model_cache:
   with path.open('rb') as f:model_cache[path]=pickle.load(f)
  return model_cache[path]
 for rp in sorted((HERE/'kazakhstan').glob('*_receipt.json')):
  r=json.loads(rp.read_text());features=r['features'];extra=any(s.startswith(('halyk_','kase_')) for s in features)
  if extra:
   lag=2 if '_lag2' in r['name'] else 1;key=('extended_liquidity',lag)
   if key not in panel_cache:panel_cache[key]=liq.build_panel(True,lag)[0].set_index(['date','corridor'])
  else:
   key=r['source_panel']
   if key not in panel_cache:panel_cache[key]=pd.read_pickle(ROOT/key).set_index(['date','corridor'])
  panel=panel_cache[key];pred=pd.read_csv(rp.with_name(rp.name.replace('_receipt.json','_predictions.csv.gz')),parse_dates=['date']);assert len(pred)==883 and pred.corridor.eq('KZT').all() and not pred.duplicated(['date','corridor']).any()
  total=0;maxerr=0.
  for fit in r['fits']:
   p=pred[pred.fold_test_year==fit['year']];index=pd.MultiIndex.from_frame(p[['date','corridor']]);x=panel.loc[index].reset_index()[features+['corridor']];m=model(HERE/'kazakhstan'/fit['checkpoint']);prob=m.predict_proba(x)[:,1]
   err=float(np.max(np.abs(prob-p.raw_probability.to_numpy())));assert err<1e-12,(r['name'],fit['year'],err);total+=len(p);maxerr=max(maxerr,err)
   if r['strategy'].startswith('residual'):
    pooled_name=r['name'].replace('kzt_residual_shrink_','kzt_pooled_').replace('kzt_residual_','kzt_pooled_');pooled=model(HERE/'kazakhstan/checkpoints'/f'{pooled_name}_{fit["year"]}.pkl')
    original=m.pooled.predict_proba(x);independent=pooled.predict_proba(x);base_error=float(np.max(np.abs(original-independent)));assert base_error<1e-14,(r['name'],fit['year'],'pooled initialization mismatch',base_error)
    z=m.pooled.named_steps['preprocessor'].transform(x);assert z.dtype==np.float64,('Transformed frozen base input not float64',z.dtype)
    trees=m.trees
    try:
     m.trees=[];zero=m.predict_proba(x)
    finally:m.trees=trees
    identity_error=float(np.max(np.abs(zero-original)));assert identity_error<1e-14,(r['name'],fit['year'],'zero-tree identity failed',identity_error)
    # Reconstruct correction independently of wrapper and its predict_proba.
    from scipy.special import expit,logit
    correction=np.zeros(len(x),dtype=np.float64)
    for tree in trees:correction+=m.learning_rate*tree.predict(z)
    adjusted=expit(logit(original[:,1].clip(1e-7,1-1e-7))+correction);weight=getattr(m,'weight',1.)
    reconstructed=weight*adjusted+(1-weight)*original[:,1];reconstruction_error=float(np.max(np.abs(reconstructed-prob)));assert reconstruction_error<1e-14
    bases.append({'model':r['name'],'year':fit['year'],'rows':len(x),'float64_transformed_input':True,'pooled_base_max_abs_error':base_error,'zero_tree_max_abs_error':identity_error,'manual_newton_reconstruction_max_abs_error':reconstruction_error,'trees':len(trees),'shrink_weight':weight})
  assert total==883;rows.append({'model':r['name'],'rows':total,'raw_probability_max_abs_error':maxerr})
 return {'models':rows,'continuation_checks':bases,'models_checked':len(rows),'raw_prediction_rows_recomputed':sum(r['rows'] for r in rows),'note':'883 actual OOT KZT rows per model; deterministic parity, not 883 independent regimes.'}


def future_perturbation():
 records=[];original_read_csv=pd.read_csv
 for extended in (False,True):
  for lag in (1,2):
   base,groups,_=liq.build_panel(extended,lag)
   for cutoff in (pd.Timestamp('2023-06-30'),pd.Timestamp('2024-12-31')):
    changed={}
    def poison(path,*args,**kwargs):
     df=original_read_csv(path,*args,**kwargs)
     if Path(path).name=='kase_spot_daily.csv':d=pd.to_datetime(df.date_trade).dt.normalize()
     elif Path(path).name=='halyk_sell_daily.csv':d=pd.to_datetime(df.date)
     else:return df
     mask=d>cutoff;columns=df.select_dtypes(include='number').columns
     assert mask.any() and len(columns)>0
     df.loc[mask,columns]=df.loc[mask,columns]*100
     changed[Path(path).name]={'future_rows_changed':int(mask.sum()),'numeric_columns':list(columns)};return df
    with patch.object(pd,'read_csv',side_effect=poison):modified,mg,_=liq.build_panel(extended,lag)
    past=base[base.date<=cutoff].reset_index(drop=True);mp=modified[modified.date<=cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(past,mp,check_exact=True);assert groups==mg and len(changed)==2
    records.append({'extended':extended,'lag':lag,'cutoff':str(cutoff.date()),'unchanged_prefix_rows':len(past),'columns_compared':len(past.columns),'perturbations':changed,'exact_prefix_invariant':True})
 return records


def sealed_v3_readonly():
 path=ROOT/'research_v3/verify.py';success=path.with_name('_SUCCESS.json');before=digest(success) if success.exists() else None;writes=[];capture=io.StringIO()
 def intercept_write(target,text,*args,**kwargs):
  if target.resolve()!=success.resolve():raise AssertionError(f'Unexpected write from read-only verifier: {target}')
  writes.append(json.loads(text));return len(text)
 with patch.object(sys,'argv',[str(path)]),patch.object(Path,'write_text',intercept_write),contextlib.redirect_stdout(capture):runpy.run_path(str(path),run_name='__main__')
 after=digest(success) if success.exists() else None;assert before==after and len(writes)==1 and writes[0]['status']=='PASS'
 return {'executed':'research_v3/verify.py via runpy as __main__','write_intercepted':'Only _SUCCESS.json write captured in memory; sealed files and success artifact unchanged','result':writes[0],'success_file_unchanged':True}


def preview_invariance():
 from research_v4 import market_preview as preview
 models=['kzt_residual_shrink_120m__kase_prices_lag1','combo_treasury_halyk_120m'];results=[];original_gzip_open=gzip.open
 for name in models:
  p,_=preview.locate(name)
  with original_gzip_open(p,'rt',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
  eligible=[r for r in rows if r['corridor']=='KZT' and r['date'].startswith('2025-')];day=date.fromisoformat(eligible[len(eligible)//2]['date'])
  modified=copy.deepcopy(rows);outcome_columns=[x for x in fields if any(w in x for w in ('target','forward','regret','symmetric'))]
  future_rows=0
  for row in modified:
   for col in outcome_columns:row[col]='FORBIDDEN_OUTCOME_NOT_A_NUMBER'
   if date.fromisoformat(row['date'])>day:
    future_rows+=1
    for col in ('probability','session_ordinal','candidate_signal'):row[col]='FORBIDDEN_FUTURE_NOT_A_NUMBER'
  assert outcome_columns and future_rows>0
  buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=fields);writer.writeheader();writer.writerows(modified);fixture=buffer.getvalue()
  def fake_open(path,*args,**kwargs):
   if Path(path).resolve()==p.resolve():return io.StringIO(fixture)
   return original_gzip_open(path,*args,**kwargs)
  for policy in ('legacy','selective'):
   original=preview.preview(name,day,policy=policy)
   with patch.object(gzip,'open',side_effect=fake_open):altered=preview.preview(name,day,policy=policy)
   assert original==altered and original['eligible_to_send'] is False and original['external_message_sent'] is False
   results.append({'model':name,'policy':policy,'as_of':str(day),'outcome_columns_poisoned':outcome_columns,'future_rows_poisoned':future_rows,'byte_equivalent_json':json.dumps(original,sort_keys=True)==json.dumps(altered,sort_keys=True),'no_send':True})
 return results


def main():
 started=time.monotonic();checks=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  for name,fn in [('receipts_and_raw_source_sha256',receipts),('saved_kzt_checkpoint_recomputation',checkpoint_recompute),('future_source_perturbation_prefix_invariance',future_perturbation),('historical_preview_future_outcome_invariance',preview_invariance),('sealed_v3_verifier_read_only',sealed_v3_readonly)]:
   t=time.monotonic()
   try:detail=fn();entry={'check':name,'status':'PASS','detail':detail}
   except Exception as exc:entry={'check':name,'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()}
   entry['seconds']=time.monotonic()-t;checks.append(entry);print(name,entry['status'],round(entry['seconds'],2),flush=True)
 result={'status':'PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL','seconds':time.monotonic()-started,'verifier_sha256':digest(Path(__file__)),'checks':checks,'scope':'Independent deterministic/temporal verification; not source publication-time proof, production promotion or market significance.'}
 (HERE/'market_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 if result['status']!='PASS':sys.exit(1)
if __name__=='__main__':main()
