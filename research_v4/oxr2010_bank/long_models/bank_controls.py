"""Attribution controls for the promising Halyk+Treasury research candidate."""
from pathlib import Path
import sys,time,warnings
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
HERE=Path(__file__).resolve().parent/'bank_controls'
def main():
 HERE.mkdir(exist_ok=True);e.OUT=HERE/'output';e.OUT.mkdir(exist_ok=True)
 specs=[]
 for base in ('kzt_local_120m','kzt_shrink_120m','halyk_local_120m'):
  s=next(s.copy() for s in e.specifications() if s['name']==base);s['name']='treasury_'+base;specs.append(s)
 e.save(HERE/'protocol.json',dict(created_unix=time.time(),specifications=specs,reason='Post-readout attribution only: control Treasury effect, bank effect and local residual adaptation separately for previously fitted Treasury+Halyk candidate; no tuning',selection='No new candidate selection; all controls retained; 2026 already inspected',source_sha256=e.sha(e.SNAPSHOT),code_sha256=e.sha(__file__),engine_sha256=e.sha(e.__file__)))
 original=e.feature_list;e.feature_list=lambda spec,cols:original(spec,cols)+TREASURY_LAG7_FEATURES
 outputs=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  views,cols=e.build_views();views={k:augment_panel(v) for k,v in views.items()}
  for cutoff in ('2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'):
   for spec in specs:outputs.append(e.run(views,cols,spec,cutoff))
 pred=pd.concat(outputs,ignore_index=True);pred.to_csv(HERE/'all_predictions.csv.gz',index=False)
 e.summary(pred[pred.fold_test_year.lt(2026)]).to_csv(HERE/'development_summary.csv',index=False)
 e.save(HERE/'completion.json',dict(status='complete',fits=15,source_sha256=e.sha(e.SNAPSHOT),code_sha256=e.sha(__file__)))
if __name__=='__main__':main()
