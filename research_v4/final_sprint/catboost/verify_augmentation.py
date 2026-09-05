"""Verify two duplicate-normal controls against actual delayed test features."""
from pathlib import Path
import sys,json,pickle,warnings
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import experiment as x
import verify as v
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from research_v4.final_sprint import hgb

def main():
    dest=HERE/'augmentation_controls';checks=[]
    def check(name,passed,**details):checks.append(dict(check=name,passed=bool(passed),**details))
    completion=json.loads((dest/'completion.json').read_text());views,_=x.e.build_views();views={k:x.augment_panel(p) for k,p in views.items()}
    panel=views['2010-01-01',24,1]
    for year in (2025,2026):
        cutoff=pd.Timestamp(year,1,1);key=str(cutoff.date());stem='attribution_hgb_60m_c12_duplicate_normal_'+key
        checkpoint=dest/'output'/(stem+'.pkl');b=pickle.loads(checkpoint.read_bytes());r=json.loads(checkpoint.with_suffix('.json').read_text())
        original=pickle.loads((hgb.OUT/('hgb_60m_c12_d0_bank_treasury_aug_'+key+'.pkl')).read_bytes())
        tr,va,history,te=hgb.split(panel,b['spec'],cutoff)
        check(f'{year}_duplicate_count_maturity_identical_model_params',
              len(tr)==completion['unique_train_rows'][key] and r['fit_rows']==2*len(tr)
              and tr.label_available_date.max()<cutoff-pd.DateOffset(years=1)
              and b['features']==original['features']
              and b['model'].pooled.named_steps['classifier'].get_params()==original['model'].pooled.named_steps['classifier'].get_params()
              and x.sha(checkpoint)==r['checkpoint_sha256'])
        saved=pd.read_csv(checkpoint.with_suffix('.csv.gz'),parse_dates=['date']);errors=[];maximum=0.
        for mode,lag in (('normal',1),('bank_delayed',2)):
            z=x.e.stress_view(views,dict(since='2010-01-01'),cutoff,24,lag).loc[te.index].copy()
            raw=b['model'].predict_proba(z[b['features']+['corridor']])[:,1]
            z['probability']=x.e.core.apply_platt(b['calibrator'],raw)
            for policy_name,policy in b['policies'].items():
                q=saved[saved['mode'].eq(mode)&saved.policy.eq(policy_name)]
                ids,_=v.direct_select(z,policy,policy['initial_state'])
                maximum=max(maximum,float(np.max(abs(raw-q.raw_probability.to_numpy()))),float(np.max(abs(z.probability-q.probability.to_numpy()))))
                if not np.array_equal(z.index.isin(ids),q.candidate_signal):errors.append(mode+policy_name)
        check(f'{year}_real_normal_and_delayed_inference_replay',not errors and maximum<1e-12,
              maximum_probability_error=maximum,errors=errors)
    leaderboard=pd.read_csv(HERE.parent/'leaderboard.csv')
    check('attribution_controls_excluded_from_candidate_selection',not leaderboard.config_id.str.startswith('attribution_').any())
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,
                passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),tree_fits=0,
                verifier_sha256=x.sha(__file__),control_code_sha256=x.sha(HERE/'augmentation_control.py'))
    x.save(dest/'verification.json',result);print(result['status'],result['passed'],result['failed'])
    if result['failed']:raise SystemExit(1)

if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning);main()
