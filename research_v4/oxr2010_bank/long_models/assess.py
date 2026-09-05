"""Paired date-block inference for fixed backextension/bank contrasts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.continuation.oxr.assess import paired,point
HERE=Path(__file__).resolve().parent
def main():
 pred=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in (HERE/'all_predictions.csv.gz',HERE/'treasury/all_predictions.csv.gz',HERE/'bank_controls/all_predictions.csv.gz')],ignore_index=True)
 jan=pred[pred.cutoff.eq('2026-01-01')];march=pred[pred.cutoff.eq('2026-03-01')]
 common=sorted(set(jan.date)&set(march.date))
 tracks={'development_2023_2025':pred[pred.fold_test_year.lt(2026)],'2026_january':jan,'2026_common_january':jan[jan.date.isin(common)],'2026_common_march':march[march.date.isin(common)]}
 contrasts=[
 ('history_primary','oxr_basis_2018_120m','oxr_basis_2010_120m','all'),
 ('history_longer','oxr_basis_2018_180m','oxr_basis_2010_180m','all'),
 ('history_bank_primary','fusion_2018_120m','fusion_2010_120m','KZT'),
 ('oxr_vs_v3','v3_120m','oxr_basis_2010_120m','all'),
 ('oxr_old_vs_v3','v3_120m','oxr_basis_2018_120m','all'),
 ('bank_fusion_primary','halyk_shrink_120m','fusion_2010_120m','KZT'),
 ('bank_fusion_old','halyk_shrink_120m','fusion_2018_120m','KZT'),
 ('bank_increment_oldoxr','kzt_oxr_2018_120m','fusion_2018_120m','KZT'),
 ('bank_increment_newoxr','kzt_oxr_2010_120m','fusion_2010_120m','KZT'),
 ('bank_increment_nooxr','kzt_shrink_120m','halyk_shrink_120m','KZT'),
 ('bank_vs_v3','v3_120m','halyk_shrink_120m','KZT'),
 ('premium','fusion_2010_120m','fusion_premium_2010_120m','KZT'),
 ('treasury_history','treasury_oxr_basis_2018_120m','treasury_oxr_basis_2010_120m','all'),
 ('treasury_oxr','treasury_v3_120m','treasury_oxr_basis_2010_120m','all'),
 ('treasury_full','treasury_v3_120m','treasury_oxr_full_2010_120m','all'),
 ('treasury_bank_fusion','treasury_halyk_shrink_120m','treasury_fusion_2010_120m','KZT'),
 ('treasury_plus_bank_vs_v3','v3_120m','treasury_halyk_shrink_120m','KZT'),
 ('treasury_increment_bank','halyk_shrink_120m','treasury_halyk_shrink_120m','KZT'),
 ('bank_increment_treasury','treasury_kzt_shrink_120m','treasury_halyk_shrink_120m','KZT'),
 ('adaptation_increment_treasury_bank','treasury_halyk_local_120m','treasury_halyk_shrink_120m','KZT'),
 ('treasury_bank_vs_treasury_local','treasury_kzt_local_120m','treasury_halyk_shrink_120m','KZT'),
 ]
 summaries=[];intervals=[];cells=[];stress=[]
 with threadpool_limits(limits=1):
  for track,p in tracks.items():
   for (name,mode),g in p.groupby(['config_id','mode']):
    for scope in (['all','KZT'] if g.corridor.nunique()>1 else ['KZT']):
     x=g if scope=='all' else g[g.corridor.eq('KZT')]
     summaries.append(dict(track=track,config_id=name,mode=mode,scope=scope,**point(x)))
     if mode=='normal':
      for (year,corridor),part in x.groupby(['fold_test_year','corridor']):cells.append(dict(track=track,config_id=name,scope=scope,year=year,corridor=corridor,**point(part)))
   normal=p[p['mode'].eq('normal')]
   for label,baseline,candidate,scope in contrasts:
    a=normal[normal.config_id.eq(baseline)];b=normal[normal.config_id.eq(candidate)]
    if scope=='KZT':a=a[a.corridor.eq('KZT')];b=b[b.corridor.eq('KZT')]
    for block in ('month',20,60):intervals.append(dict(track=track,contrast=label,baseline=baseline,candidate=candidate,scope=scope,block=str(block),**paired(a,b,block)))
   for name in ('halyk_shrink_120m','fusion_2018_120m','fusion_2010_120m','oxr_basis_2010_120m','treasury_halyk_shrink_120m'):
    a=normal[normal.config_id.eq(name)]
    for mode,b in p[p.config_id.eq(name)&p['mode'].ne('normal')].groupby('mode'):
     stress.append(dict(track=track,config_id=name,mode=mode,**paired(a,b)))
   print('assessed',track,flush=True)
 pd.DataFrame(summaries).to_csv(HERE/'summary.csv',index=False);pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
 pd.DataFrame(cells).to_csv(HERE/'cells.csv',index=False);pd.DataFrame(stress).to_csv(HERE/'delay_intervals.csv',index=False)
 (HERE/'inference_receipt.json').write_text(json.dumps(dict(status='complete',bootstrap_repetitions=10000,seed=20260905,interval='paired percentile95%, jointly resample all currencies within date blocks and calendar-year strata;20/60date blocks sensitivity',policy_baseline='reestimated inside every month-bootstrap resample for each year/corridor',multiplicity='unadjusted descriptive intervals; primary3 contrasts preregistered in protocol; historical data and prior2026 insights reused',summary_rows=len(summaries),paired_rows=len(intervals)),indent=2)+'\n')
if __name__=='__main__':main()
