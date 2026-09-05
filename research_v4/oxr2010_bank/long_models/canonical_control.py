"""Isolate historical OXR availability from rolling accumulation roundoff."""
from pathlib import Path
import sys,time,warnings,json,pickle
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
from research_v4.continuation.oxr.assess import paired
HERE=Path(__file__).resolve().parent/'canonical'
def main():
 HERE.mkdir(exist_ok=True);e.OUT=HERE/'output';e.OUT.mkdir(exist_ok=True)
 names=['oxr_basis_2010_120m','oxr_basis_2010_180m','kzt_oxr_2010_120m','fusion_2010_120m','treasury_oxr_basis_2010_120m','treasury_oxr_full_2010_120m','treasury_fusion_2010_120m']
 specs=[]
 for name in names:
  base=name.removeprefix('treasury_');s=next(s.copy() for s in e.specifications() if s['name']==base)
  s.update(name='canonical_'+name,treasury=name.startswith('treasury_'));specs.append(s)
 e.save(HERE/'protocol.json',dict(created_unix=time.time(),reason='Independent audit found rolling.std roundoff on identical common observations can change training bins; this numerical attribution control was added after readout without hyperparameter selection',specifications=specs,common_start='2018-09-01',canonical='Copy OXR RET/BASIS/COVER on common post-warmup rows from unchanged2018 panel to2010 panel; preserve every pre2018-09-01 value',targets='unchanged CBRNOW h5',uses_2026_for_tuning=False,code_sha256=e.sha(__file__),engine_sha256=e.sha(e.__file__)))
 feature_list=e.feature_list;e.feature_list=lambda s,cols:feature_list(s,cols)+(TREASURY_LAG7_FEATURES if s['treasury'] else [])
 outputs=[];checks=[]
 with threadpool_limits(limits=1),warnings.catch_warnings():
  warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
  views,cols=e.build_views();features=e.oxr.RET+e.oxr.BASIS+e.oxr.COVER
  for delay in (24,48):
   for lag in (1,2):
    a=views['2018-06-17',delay,lag];b=views['2010-01-01',delay,lag]
    common=a.date.ge('2018-09-01');assert a.loc[common,features].notna().equals(b.loc[common,features].notna())
    before=e.fp(b.loc[~common,['date','corridor',*features]])
    b.loc[common,features]=a.loc[common,features]
    pd.testing.assert_frame_equal(a.loc[common,features],b.loc[common,features],check_exact=True)
    assert before==e.fp(b.loc[~common,['date','corridor',*features]])
    checks.append(dict(delay=delay,bank_lag=lag,common_rows=int(common.sum()),common_oxr_features_identical=True,earlier_features_preserved=True))
  views={k:augment_panel(v) for k,v in views.items()}
  for cutoff in ('2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01'):
   for s in specs:outputs.append(e.run(views,cols,s,cutoff))
 pred=pd.concat(outputs,ignore_index=True);pred.to_csv(HERE/'all_predictions.csv.gz',index=False)
 original=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in (HERE.parent/'all_predictions.csv.gz',HERE.parent/'treasury/all_predictions.csv.gz')],ignore_index=True)
 allp=pd.concat([original,pred],ignore_index=True);jan=allp[allp.cutoff.eq('2026-01-01')];march=allp[allp.cutoff.eq('2026-03-01')];common=sorted(set(jan.date)&set(march.date))
 tracks={'development_2023_2025':allp[allp.fold_test_year.lt(2026)],'2026_january':jan,'2026_common_january':jan[jan.date.isin(common)],'2026_common_march':march[march.date.isin(common)]}
 summaries=[];intervals=[]
 refs={'oxr_basis_2010_120m':'oxr_basis_2018_120m','oxr_basis_2010_180m':'oxr_basis_2018_180m','kzt_oxr_2010_120m':'kzt_oxr_2018_120m','fusion_2010_120m':'fusion_2018_120m','treasury_oxr_basis_2010_120m':'treasury_oxr_basis_2018_120m','treasury_oxr_full_2010_120m':'treasury_v3_120m','treasury_fusion_2010_120m':'treasury_halyk_shrink_120m'}
 with threadpool_limits(limits=1):
  for track,q in tracks.items():
   sums=e.summary(q[q.config_id.str.startswith('canonical_')]);sums['track']=track;summaries.append(sums)
   normal=q[q['mode'].eq('normal')]
   for name in names:
    b=normal[normal.config_id.eq('canonical_'+name)]
    for label,ref in [('numerical_attribution',name),('history_or_incremental_control',refs[name])]:
     a=normal[normal.config_id.eq(ref)]
     for scope in (['all','KZT'] if b.corridor.nunique()>1 else ['KZT']):
      x=a if scope=='all' else a[a.corridor.eq('KZT')];y=b if scope=='all' else b[b.corridor.eq('KZT')]
      for block in ('month',20,60):intervals.append(dict(track=track,contrast=label,baseline=ref,candidate='canonical_'+name,scope=scope,block=str(block),**paired(x,y,block)))
 pd.concat(summaries).to_csv(HERE/'summary.csv',index=False);pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
 e.save(HERE/'verification.json',dict(status='PASS',fits=35,common_feature_checks=checks,prior_results_unchanged=True,code_sha256=e.sha(__file__),source_sha256=e.sha(e.SNAPSHOT),classification='Post-audit numerical attribution; all controls retained, no model selection on2026'))
 print('Canonical controls complete',flush=True)
if __name__=='__main__':main()
