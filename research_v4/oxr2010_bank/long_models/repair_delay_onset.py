"""Refresh only inference after audited onset fix; preserve every fitted model."""
from pathlib import Path
import sys,shutil,json,pickle,warnings
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel
HERE=Path(__file__).resolve().parent
def main():
 records=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  raw,cols=e.build_views()
  for folder in (HERE,HERE/'treasury'):
   views=raw if folder==HERE else {k:augment_panel(v) for k,v in raw.items()}
   backup=folder/'before_onset_repair';backup.mkdir(exist_ok=True)
   for f in folder.glob('*.csv*'):shutil.copy2(f,backup/f.name)
   outputs=[]
   for path in sorted((folder/'output').glob('*.pkl')):
    bundle=pickle.loads(path.read_bytes());spec=bundle['spec'];cutoff=pd.Timestamp(bundle['cutoff']);f=bundle['features']
    dest=path.with_suffix('.csv.gz');before=pd.read_csv(dest,parse_dates=['date','label_available_date']);after=before.copy()
    receipt_path=path.with_suffix('.json');receipt=json.loads(receipt_path.read_text());oldsha=e.sha(dest)
    for mode,g in before.groupby('mode'):
     delay,lag={'normal':(24,1),'oxr_delayed':(48,1),'bank_delayed':(24,2),'both_delayed':(48,2)}[mode]
     panel=e.stress_view(views,spec,cutoff,delay,lag)
     lookup=panel.reset_index().set_index(['date','corridor'])['index']
     idx=lookup.loc[pd.MultiIndex.from_frame(g[['date','corridor']])].to_numpy()
     test=panel.loc[idx]
     probability_raw=bundle['model'].predict_proba(test[f+['corridor']])[:,1]
     probability=e.core.apply_platt(bundle['calibrator'],probability_raw)
     chosen=e.core.select_per_corridor_with_cooldown(test,probability,bundle['threshold'],bundle['initial_state'])
     portfolio=e.core.select_portfolio_from_candidates(test,probability,chosen,bundle['portfolio_state'])
     if mode=='normal':
      assert np.max(np.abs(g.probability.to_numpy()-probability))<1e-12
      assert np.array_equal(g.candidate_signal.to_numpy(),test.index.isin(chosen))
     after.loc[g.index,'raw_probability']=probability_raw;after.loc[g.index,'probability']=probability
     after.loc[g.index,'candidate_signal']=test.index.isin(chosen);after.loc[g.index,'signal']=test.index.isin(portfolio)
     records.append(dict(checkpoint=str(path.relative_to(HERE)),mode=mode,rows=len(g),changed_probability_rows=int((np.abs(g.probability.to_numpy()-probability)>1e-12).sum()),max_probability_difference=float(np.max(np.abs(g.probability.to_numpy()-probability)))))
    assert e.sha(path)==receipt['checkpoint_sha256']
    after.to_csv(dest,index=False);receipt['before_onset_repair_predictions_sha256']=oldsha;receipt['predictions_sha256']=e.sha(dest);receipt['onset_repair_code_sha256']=e.sha(__file__);receipt['model_refitted']=False;e.save(receipt_path,receipt)
    outputs.append(after)
   pred=pd.concat(outputs,ignore_index=True);pred.to_csv(folder/'all_predictions.csv.gz',index=False)
   dev=pred[pred.fold_test_year.lt(2026)];dev.to_csv(folder/'development_predictions.csv.gz',index=False);e.summary(dev).to_csv(folder/'development_summary.csv',index=False)
   metrics=[]
   for cutoff,g in pred.groupby('cutoff'):
    m=e.summary(g);m['cutoff']=cutoff;metrics.append(m)
   pd.concat(metrics).to_csv(folder/'metrics_by_cutoff.csv',index=False)
   # Existing selection is immutable and refers to backed-up original development.
   selection=json.loads((folder/'selection.json').read_text())
   assert selection['development_sha256']==e.sha(backup/'development_predictions.csv.gz')
   e.save(folder/'onset_repair.json',dict(status='PASS',reason='Delayed feature rolling state now preserves normal pre-cutoff history; native source returns unchanged',fit_engine_snapshot_sha256=e.sha(HERE/'fit_engine_snapshot.py'),current_engine_sha256=e.sha(e.__file__),code_sha256=e.sha(__file__),original_selection_development='before_onset_repair/development_predictions.csv.gz',normal_predictions_unchanged=True,selection_unchanged=True,refits=0))
 pd.DataFrame(records).to_csv(HERE/'onset_repair_changes.csv',index=False)
 print('Replayed',len(records),'model/view combinations without refits',flush=True)
if __name__=='__main__':main()
