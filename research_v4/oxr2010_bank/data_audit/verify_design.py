"""Read-only independent onset repair and historical-control checks; no fitting."""
from pathlib import Path
import hashlib,json,sys,warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from research_v4.oxr2010_bank.long_models import experiment as engine

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 checks=[]
 def check(name,passed,**details):
  checks.append(dict(check=name,passed=bool(passed),**details));print(name,'PASS' if passed else 'FAIL',flush=True)
 views,_=engine.build_views()
 columns=['oxr_basis_chg1','oxr_basis_chg5','oxr_basis_z20','bank_oxr_premium_chg1','bank_oxr_premium_chg5']
 # Construct the first twenty post-cutoff values from an explicit online buffer.
 # This avoids reusing stress_view's pandas rolling/diff implementation.
 for since in ('2010-01-01','2018-06-17'):
  for cutoff in ('2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'):
   for delay,lag in ((48,1),(24,2),(48,2)):
    normal=views[since,24,1];delayed=views[since,delay,lag]
    actual=engine.stress_view(views,{'since':since},pd.Timestamp(cutoff),delay,lag)
    before=normal.date.lt(cutoff)
    maxerror=0.;equal_missing=True
    for _,g in normal.groupby('corridor'):
     idx=g.index;past=idx[g.date.lt(cutoff)];future=idx[g.date.ge(cutoff)][:21]
     basis=list(normal.loc[past,'oxr_log_basis'].to_numpy())
     premium=list(normal.loc[past,'bank_oxr_premium'].to_numpy())
     for i in future:
      basis.append(delayed.loc[i,'oxr_log_basis']);premium.append(delayed.loc[i,'bank_oxr_premium'])
      window=np.asarray(basis[-20:]);window=window[np.isfinite(window)]
      z=(basis[-1]-window.mean())/max(window.std(ddof=1),1e-6) if len(window)>=10 else np.nan
      expected=np.array([basis[-1]-basis[-2],basis[-1]-basis[-6],z,premium[-1]-premium[-2],premium[-1]-premium[-6]])
      observed=actual.loc[i,columns].to_numpy(float)
      equal_missing &= bool(np.array_equal(np.isnan(expected),np.isnan(observed)))
      if np.isfinite(expected).any():maxerror=max(maxerror,float(np.nanmax(abs(expected-observed))))
    check(f'online_stitch_{since}_{cutoff}_O{delay}_B{lag}',
          actual.loc[before].equals(normal.loc[before]) and equal_missing and maxerror<1e-10,
          maximum_feature_error=maxerror,pre_cutoff_frame_exact=actual.loc[before].equals(normal.loc[before]))

 total=0
 for folder in (engine.HERE,engine.HERE/'treasury'):
  before=pd.read_csv(folder/'before_onset_repair/all_predictions.csv.gz')
  after=pd.read_csv(folder/'all_predictions.csv.gz')
  a=before[before['mode'].eq('normal')];b=after[after['mode'].eq('normal')]
  keys=['config_id','cutoff','date','corridor','mode']
  merged=a.merge(b,on=keys,suffixes=['_old','_new'],validate='one_to_one')
  errors={c:float((merged[c+'_new'].astype(float)-merged[c+'_old'].astype(float)).abs().max()) for c in
          ['raw_probability','probability','candidate_signal','signal','target','forward_bps']}
  check(folder.name+'_normal_repair_parity',len(a)==len(b)==len(merged) and max(errors.values())<1e-12,
        rows=len(merged),maximum_errors=errors)
  selections=json.loads((folder/'selection.json').read_text())
  check(folder.name+'_original_selection_input_preserved',
        selections['development_sha256']==sha(folder/'before_onset_repair/development_predictions.csv.gz'))
  errors=[]
  for ck in (folder/'output').glob('*.pkl'):
   receipt=json.loads(ck.with_suffix('.json').read_text());total+=1
   if not (sha(ck)==receipt['checkpoint_sha256'] and sha(ck.with_suffix('.csv.gz'))==receipt['predictions_sha256']
           and receipt.get('model_refitted') is False):errors.append(ck.name)
  check(folder.name+'_checkpoint_and_prediction_hashes',not errors,errors=errors)
 check('all_120_preserved_checkpoints',total==120,checkpoints=total)

 new=pd.read_csv(engine.HERE/'all_predictions.csv.gz');new=new[new['mode'].eq('normal') & new.cutoff.ne('2026-03-01')]
 mappings=[('v3_120m','research_v3/models/basis_train_120m_h5_predictions.csv.gz',None),
           ('kzt_local_120m','research_v4/kazakhstan/kzt_pooled_120m_predictions.csv.gz',None),
           ('kzt_shrink_120m','research_v4/kazakhstan/kzt_residual_shrink_120m_predictions.csv.gz',None),
           ('halyk_local_120m','research_v4/kazakhstan/kzt_pooled_120m__halyk_lag1_predictions.csv.gz',None),
           ('halyk_shrink_120m','research_v4/kazakhstan/kzt_residual_shrink_120m__halyk_lag1_predictions.csv.gz',None)]
 # Previous OXR annual output is stored in separate development/test aggregates.
 older_oxr=pd.concat([pd.read_csv(ROOT/'research_v4/continuation/oxr'/f'{phase}_predictions.csv.gz') for phase in ('development','test')])
 older_oxr=older_oxr[older_oxr.config_id.eq('oxr_basis_120m_delay24h') & older_oxr.cutoff.ne('2026-03-01')]
 def compare(name,a,b):
  m=a.merge(b,on=['date','corridor'],suffixes=['_new','_old'],validate='one_to_one')
  errors={c:float((m[c+'_new'].astype(float)-m[c+'_old'].astype(float)).abs().max()) for c in
          ['probability','target','candidate_signal','signal','forward_bps']}
  check(name,len(a)==len(b)==len(m) and max(errors.values())<1e-12,rows=len(m),maximum_errors=errors)
 for name,path,_ in mappings:compare('historical_control_'+name,new[new.config_id.eq(name)],pd.read_csv(ROOT/path))
 compare('historical_control_oxr2018',new[new.config_id.eq('oxr_basis_2018_120m')],older_oxr)
 foundation=pd.read_csv(HERE.parent/'foundation/development_predictions.csv.gz')
 previous=pd.read_csv(ROOT/'research_v4/continuation/foundation/extended_contract/predictions.csv.gz',low_memory=False)
 previous=previous[previous.fold_test_year.le(2025)]
 for name,oldname in [('synth_no_oxr','chronos2_synth_zs_head10y'),('small_no_oxr','chronos2_small_ft_head10y')]:
  compare('foundation_development_control_'+name,foundation[foundation.config_id.eq(name)],previous[previous.config_id.eq(oldname)])
 result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',passed=sum(c['passed'] for c in checks),
             failed=sum(not c['passed'] for c in checks),checks=checks,tree_fits=0,api_calls=0,
             source_sha256=sha(engine.SNAPSHOT),engine_sha256=sha(engine.__file__),
             foundation_code_sha256=sha(HERE.parent/'foundation/experiment.py'),verifier_sha256=sha(__file__),
             scope='Independent online rolling-buffer splice, unchanged normal outputs and historical baseline parity; no new statistical model fits.')
 (HERE/'design_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:result[k] for k in ('status','passed','failed')},indent=2))
 if result['failed']:raise SystemExit(1)
if __name__=='__main__':
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning);main()
