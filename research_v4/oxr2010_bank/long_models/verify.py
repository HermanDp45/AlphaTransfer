"""Read-only fit/prediction receipts and frozen-baseline reproduction checks."""
from pathlib import Path
import sys,json,pickle,hashlib,warnings,platform
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel
HERE=Path(__file__).resolve().parent
def main():
 checks=[];preds=[];replays=0
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  base,cols=e.build_views()
  for folder in (HERE,HERE/'treasury',HERE/'bank_controls'):
   views=base if folder==HERE else {k:augment_panel(v) for k,v in base.items()}
   for path in sorted((folder/'output').glob('*.pkl')):
    receipt=json.loads(path.with_suffix('.json').read_text());assert e.sha(path)==receipt['checkpoint_sha256']
    dest=path.with_suffix('.csv.gz');assert e.sha(dest)==receipt['predictions_sha256'];assert receipt['source_sha256']==e.sha(e.SNAPSHOT)
    bundle=pickle.loads(path.read_bytes());spec=bundle['spec'];cutoff=pd.Timestamp(bundle['cutoff']);f=bundle['features']
    assert pd.Timestamp(receipt['train_latest_label'])<cutoff-pd.DateOffset(years=1)
    assert pd.Timestamp(receipt['validation_latest_label'])<cutoff
    p=views[spec['since'],24,1]
    tr,va,te=e.old.temporal_split(p,5,cutoff,pd.Timestamp(cutoff.year+1,1,1),e.old.Spec(spec['name'],months=spec['months'],extended=True))
    assert e.fp(tr[['date','corridor',*f]])==receipt['train_feature_fingerprint']
    prediction=pd.read_csv(dest,parse_dates=['date']);preds.append(prediction)
    for mode,g in prediction.groupby('mode'):
     delay,lag={'normal':(24,1),'oxr_delayed':(48,1),'bank_delayed':(24,2),'both_delayed':(48,2)}[mode]
     view=e.stress_view(views,spec,cutoff,delay,lag)
     lookup=view.reset_index().set_index(['date','corridor'])['index'];idx=lookup.loc[pd.MultiIndex.from_frame(g[['date','corridor']])].to_numpy();test=view.loc[idx]
     raw=bundle['model'].predict_proba(test[f+['corridor']])[:,1];proba=e.core.apply_platt(bundle['calibrator'],raw)
     assert np.max(np.abs(g.raw_probability.to_numpy()-raw))<1e-12
     assert np.max(np.abs(g.probability.to_numpy()-proba))<1e-12
     chosen=e.core.select_per_corridor_with_cooldown(test,proba,bundle['threshold'],bundle['initial_state'])
     assert np.array_equal(g.candidate_signal.to_numpy(),test.index.isin(chosen))
     replays+=1
    checks.append(dict(checkpoint=str(path.relative_to(HERE)),rows=len(prediction),train_maturity=True,calibration_maturity=True,input_fingerprint=True,checkpoint_and_predictions_sha=True,replay=True))
 pred=pd.concat(preds,ignore_index=True);normal=pred[pred['mode'].eq('normal')&pred.cutoff.str.endswith('-01-01')]
 references={'v3_120m':'research_v3/models/basis_train_120m_h5_predictions.csv.gz','kzt_local_120m':'research_v4/kazakhstan/kzt_pooled_120m_predictions.csv.gz','kzt_shrink_120m':'research_v4/kazakhstan/kzt_residual_shrink_120m_predictions.csv.gz','halyk_shrink_120m':'research_v4/kazakhstan/kzt_residual_shrink_120m__halyk_lag1_predictions.csv.gz'}
 parity=[]
 for name,source in references.items():
  old=pd.read_csv(ROOT/source,parse_dates=['date']);new=normal[normal.config_id.eq(name)]
  both=new.merge(old,on=['date','corridor'],suffixes=('_new','_old'),validate='one_to_one');assert len(both)==len(new)
  error=float(np.max(np.abs(both.probability_new-both.probability_old)));assert error<1e-12
  assert np.array_equal(both.candidate_signal_new,both.candidate_signal_old)
  parity.append(dict(config_id=name,rows=len(new),max_probability_error=error,candidate_mismatches=0))
 for folder in (HERE,HERE/'treasury'):
  selection=json.loads((folder/'selection.json').read_text());assert selection.get('uses_2026') is False
  assert selection['development_sha256']==e.sha(folder/'before_onset_repair/development_predictions.csv.gz')
  protocol=json.loads((folder/'protocol.json').read_text());engine=protocol.get('engine_sha256',protocol['code_sha256'])
  assert engine==e.sha(HERE/'fit_engine_snapshot.py')
 e.save(HERE/'verification.json',dict(status='PASS',fits=len(checks),model_view_replays=replays,checks=checks,baseline_parity=parity,delay_repair='all normal probabilities and candidate decisions unchanged; exact pre-cutoff source state retained',runtime=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__)))
 print('PASS',len(checks),'models',replays,'view replays',flush=True)
if __name__=='__main__':main()
