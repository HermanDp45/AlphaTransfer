"""Independent extra post-selection interval audit, no fits or reranking."""
from pathlib import Path
import os,sys,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.architecture_2023_2025.audit.verify import bootstrap
HERE=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pred=pd.read_csv(HERE/'predictions.csv.gz',parse_dates=['date'],float_precision='round_trip',low_memory=False)
    rows=pd.read_csv(HERE/'selected_utility_intervals.csv',float_precision='round_trip');checks=[]
    for r in rows.to_dict('records'):
        g=pred[pred.config_id.eq(r['config_id'])&pred.train_horizon.eq(r['train_horizon'])&pred.policy.eq(r['policy'])]
        if r['evaluation_scope']=='KZT':g=g[g.corridor.eq('KZT')]
        if str(r['year'])!='all':g=g[g.fold_test_year.eq(int(r['year']))]
        # Selecting every eligible day has exactly zero relative utility in each
        # resampled year×corridor cell. Difference vs this control equals the
        # candidate's absolute relative utility; independent paired engine.
        zero=g.copy();zero['candidate_signal']=True
        d=bootstrap(zero,g)['forward_delta_bps']
        assert np.isclose(d['low'],r['forward_delta_bps_ci_low'],atol=1e-10,rtol=1e-11)
        assert np.isclose(d['high'],r['forward_delta_bps_ci_high'],atol=1e-10,rtol=1e-11)
        assert d['finite']==r['finite_bootstrap_draws']
        checks.append(dict(type='single_utility',candidate=r['config_id'],horizon=r['train_horizon'],scope=r['evaluation_scope'],year=r['year'],status='PASS',**d))
    pairs=pd.read_csv(HERE/'retrospective_paired_intervals.csv',float_precision='round_trip')
    for r in pairs.to_dict('records'):
        bs,bp=r['baseline'].split('::');cs,cp=r['candidate'].split('::')
        a=pred[pred.config_id.eq(bs)&pred.policy.eq(bp)&pred.train_horizon.eq(r['train_horizon'])]
        b=pred[pred.config_id.eq(cs)&pred.policy.eq(cp)&pred.train_horizon.eq(r['train_horizon'])]
        if r['evaluation_scope']=='KZT':a=a[a.corridor.eq('KZT')];b=b[b.corridor.eq('KZT')]
        d=bootstrap(a,b)
        for m,x in d.items():
            prefix='' if m=='brier' else m+'_'
            for side in ('low','high'):assert np.isclose(x[side],r[prefix+'ci_'+side],atol=1e-10,rtol=1e-11,equal_nan=True)
        checks.append(dict(type='paired',candidate=r['candidate'],baseline=r['baseline'],horizon=r['train_horizon'],scope=r['evaluation_scope'],status='PASS',metrics=d))
    receipt=dict(status='PASS',passed=len(checks),checks=checks,source_sha256={p:sha(HERE/p) for p in ['predictions.csv.gz','uncertainty.py','selected_utility_intervals.csv','retrospective_paired_intervals.csv']},audit_sha256=sha(__file__),interpretation='Conditional descriptive intervals after retrospective selection. These are not selection-adjusted or independent confirmation.')
    (HERE/'audit/uncertainty_verification.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks))))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
