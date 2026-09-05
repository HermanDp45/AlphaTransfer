"""Post-selection descriptive intervals; no retraining or reranking."""
from pathlib import Path
import os,sys,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.continuation.oxr.assess import paired
HERE=Path(__file__).resolve().parent

def single(g):
    g=g.copy();g['month']=g.date.dt.to_period('M').astype(str)
    blocks=sorted(g.groupby(['fold_test_year','month']).groups);cells=sorted(g.groupby(['fold_test_year','corridor']).groups)
    bi={k:i for i,k in enumerate(blocks)};ci={k:i for i,k in enumerate(cells)};x=np.zeros((len(blocks),len(cells),5))
    for row in g.itertuples():
        signal=float(row.candidate_signal);x[bi[row.fold_test_year,row.month],ci[row.fold_test_year,row.corridor]] += [1,row.target,row.forward_bps,signal,signal*row.forward_bps]
    rng=np.random.default_rng(20260905);w=np.zeros((10000,len(blocks)))
    for year in sorted(g.fold_test_year.unique()):
        ids=[i for i,k in enumerate(blocks) if k[0]==year];draws=rng.integers(0,len(ids),(10000,len(ids)))
        for j,idx in enumerate(ids):w[:,idx]=(draws==j).sum(axis=1)
    z=(w@x.reshape(len(blocks),-1)).reshape(10000,len(cells),5)
    base=z[:,:,2]/z[:,:,0];count=z[:,:,3].sum(axis=1);utility=(z[:,:,4]-z[:,:,3]*base).sum(axis=1)/count
    return dict(forward_delta_bps_ci_low=float(np.nanquantile(utility,.025)),forward_delta_bps_ci_high=float(np.nanquantile(utility,.975)),finite_bootstrap_draws=int(np.isfinite(utility).sum()),bootstrap_draws=10000)

def main():
    p=pd.read_csv(HERE/'predictions.csv.gz',parse_dates=['date'],float_precision='round_trip');selections=json.loads((HERE/'selection.json').read_text());rows=[];pairs=[]
    for s in selections:
        if s['period']!='retrospective_2024_2026':continue
        g=p[(p.config_id==s['config_id'])&(p.train_horizon==s['train_horizon'])&(p.policy==s['policy'])]
        if s['evaluation_scope']=='KZT':g=g[g.corridor=='KZT']
        for year in (2024,2025,2026,'all'):
            z=g if year=='all' else g[g.fold_test_year==year]
            rows.append(dict(config_id=s['config_id'],train_horizon=s['train_horizon'],evaluation_scope=s['evaluation_scope'],policy=s['policy'],year=year,inference='descriptive_post_selection',**single(z)))
        a=p[(p.config_id=='v3')&(p.train_horizon==s['train_horizon'])&(p.policy=='strict05')]
        if s['evaluation_scope']=='KZT':a=a[a.corridor=='KZT']
        pairs.append(dict(candidate=s['config_id']+'::'+s['policy'],baseline='v3::strict05',train_horizon=s['train_horizon'],evaluation_scope=s['evaluation_scope'],inference='descriptive_post_selection',**paired(a,g)))
        if s['evaluation_scope']=='KZT':
            a=p[(p.config_id==s['config_id'])&(p.train_horizon==s['train_horizon'])&(p.policy=='cadence85')&(p.corridor=='KZT')]
            pairs.append(dict(candidate=s['config_id']+'::'+s['policy'],baseline=s['config_id']+'::cadence85',train_horizon=s['train_horizon'],evaluation_scope=s['evaluation_scope'],inference='same_weights_policy_change_post_selection',**paired(a,g)))
    pd.DataFrame(rows).to_csv(HERE/'selected_utility_intervals.csv',index=False);pd.DataFrame(pairs).to_csv(HERE/'retrospective_paired_intervals.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
