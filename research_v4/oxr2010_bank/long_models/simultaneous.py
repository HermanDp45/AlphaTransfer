"""Selection-aware simultaneous Brier bands across this branch's27 models."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
def bands(p,block):
 p=p[p['mode'].eq('normal')&p.corridor.eq('KZT')].copy()
 losses=p.assign(loss=(p.probability-p.target)**2).pivot(index=['fold_test_year','date'],columns='config_id',values='loss')
 assert losses.notna().all().all();delta=losses.subtract(losses['v3_120m'],axis=0)
 candidates=[c for c in delta if c!='v3_120m'];delta=delta[candidates]
 days=delta.index.get_level_values('date');years=delta.index.get_level_values('fold_test_year')
 blocks=days.to_period('M').astype(str) if block=='month' else np.arange(len(days))//int(block)
 sums=delta.groupby([years,blocks]).sum();counts=delta.groupby([years,blocks]).size()
 rng=np.random.default_rng(20260905);weights=np.zeros((10000,len(sums)))
 for year in sums.index.get_level_values(0).unique():
  ix=np.flatnonzero(sums.index.get_level_values(0)==year);draws=rng.integers(0,len(ix),(10000,len(ix)))
  for j,col in enumerate(ix):weights[:,col]=(draws==j).sum(axis=1)
 draws=(weights@sums.to_numpy())/(weights@counts.to_numpy())[:,None]
 observed=delta.mean().to_numpy();se=draws.std(axis=0,ddof=1);assert (se>0).all()
 max_stat=np.max(np.abs((draws-observed)/se),axis=1);critical=float(np.quantile(max_stat,.95))
 return [dict(config_id=name,baseline='v3_120m',scope='KZT',block=str(block),dates=len(days),models_in_family=len(losses.columns),delta_brier=float(observed[i]),bootstrap_se=float(se[i]),critical_max_abs_t=critical,simultaneous_ci_low=float(observed[i]-critical*se[i]),simultaneous_ci_high=float(observed[i]+critical*se[i])) for i,name in enumerate(candidates)]
def main():
 p=pd.concat([pd.read_csv(f,parse_dates=['date']) for f in [HERE/'all_predictions.csv.gz',HERE/'treasury/all_predictions.csv.gz',HERE/'bank_controls/all_predictions.csv.gz']],ignore_index=True)
 jan=p[p.cutoff.eq('2026-01-01')];march=p[p.cutoff.eq('2026-03-01')];common=sorted(set(jan.date)&set(march.date))
 tracks={'development_2023_2025':p[p.fold_test_year.lt(2026)],'2026_january':jan,'2026_common_january':jan[jan.date.isin(common)],'2026_common_march':march[march.date.isin(common)]}
 rows=[]
 with threadpool_limits(limits=1):
  for track,q in tracks.items():
   for block in ('month',20,60):rows.extend([dict(track=track,**r) for r in bands(q,block)])
 pd.DataFrame(rows).to_csv(HERE/'simultaneous_intervals.csv',index=False)
 (HERE/'simultaneous_receipt.json').write_text(json.dumps(dict(status='complete',method='Centered standardized block-bootstrap maximum absolute statistic; simultaneous95% bands around observed paired Brier differences',replicates=10000,scope='27 normal-view HGB configurations in this new branch; KZT vs same V3long',limitations='Approximate fixed-model sampling intervals; does not cover previousV3/V4/neural research, task/metric selection, latency selection or model training uncertainty;2026 previously inspected'),indent=2)+'\n')
if __name__=='__main__':main()
