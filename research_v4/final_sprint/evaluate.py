"""Complete retrospective leaderboard and fixed-prediction block intervals."""
from pathlib import Path
import os,sys,json
os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.common import POLICIES,fit_policy,apply_policy,metrics
from research_v4.continuation.oxr.assess import paired
HERE=Path(__file__).resolve().parent

def read(path):return pd.read_csv(path,parse_dates=['date'])

def re_policy(pred,cal,hist):
    frames=[];policies=[]
    for (name,cutoff),g in pred.groupby(['config_id','cutoff']):
        v=cal[cal.config_id.eq(name)&cal.cutoff.eq(cutoff)&cal.corridor.eq('KZT')].copy().reset_index(drop=True)
        hh=hist[hist.config_id.eq(name)&hist.cutoff.eq(cutoff)&hist.corridor.eq('KZT')].copy().reset_index(drop=True)
        assert v.target.notna().all() and v.date.max()<pd.Timestamp(cutoff)
        for pname in POLICIES:
            policy=fit_policy(v,hh,pname);policies.append(dict(config_id=name,cutoff=cutoff,**policy))
            for mode,x in g[g.corridor.eq('KZT')].groupby('mode'):
                q=apply_policy(x,policy);frames.append(q)
    return pd.concat(frames,ignore_index=True),policies

def absolute_intervals(g):
    """Four-week contiguous clusters, fixed model/signals; selection not repeated."""
    q=g.copy();start=q.date.min().to_period('W-SUN')
    q['block']=[(d.to_period('W-SUN').ordinal-start.ordinal)//4 for d in q.date]
    q['hits']=q.target*q.candidate_signal;q['selected_forward']=q.forward_bps*q.candidate_signal
    a=q.groupby('block').agg(n=('target','size'),y=('target','sum'),s=('candidate_signal','sum'),h=('hits','sum'),f=('forward_bps','sum'),sf=('selected_forward','sum')).to_numpy(float)
    rng=np.random.default_rng(20260905);draw=rng.integers(len(a),size=(10000,len(a)));s=a[draw].sum(axis=1)
    lift=(s[:,3]/s[:,2])/(s[:,1]/s[:,0]);benefit=s[:,5]/s[:,2]-s[:,4]/s[:,0]
    return dict(blocks=len(a),lift_ci95=np.nanquantile(lift,[.025,.975]).tolist(),forward_delta_bps_ci95=np.nanquantile(benefit,[.025,.975]).tolist())

def main():
    frames=[];policyregistry=[]
    for year in (2025,2026):
        for p in (HERE/'hgb').glob(f'*_{year}-01-01.csv.gz'):frames.append(read(p))
        for p in (HERE/'monthly').glob(f'*_{year}.csv.gz'):frames.append(read(p))
    frames.append(read(HERE/'baseline_controls/all_predictions.csv.gz'))
    cb=HERE/'catboost'
    if (cb/'candidates_predictions.csv.gz').exists():
        q,policies=re_policy(read(cb/'candidates_predictions.csv.gz'),read(cb/'candidates_calibration_predictions.csv.gz'),read(cb/'candidates_history_predictions.csv.gz'))
        frames.append(q);policyregistry+=policies
    tp=HERE/'tabm/policy_predictions.csv.gz'
    if tp.exists():frames.append(read(tp))
    pred=pd.concat(frames,ignore_index=True);pred=pred[pred.corridor.eq('KZT')].copy()
    # Explicit exact cohort checks; never hide missing dates by intersection.
    references={year:pred[pred.config_id.eq('v3_120m')&pred.policy.eq('legacy')&pred['mode'].eq('normal')&pred.fold_test_year.eq(year)].sort_values('date') for year in (2025,2026)}
    summaries=[]
    for (name,cutoff,mode,policy),g in pred.groupby(['config_id','cutoff','mode','policy']):
        ref=references[pd.Timestamp(cutoff).year];g=g.sort_values('date')
        assert g.date.astype(str).tolist()==ref.date.astype(str).tolist(),(name,cutoff,mode,policy,len(g),len(ref))
        assert np.array_equal(g.target,ref.target) and np.allclose(g.forward_bps,ref.forward_bps),(name,mode)
        summaries.append(dict(config_id=name,cutoff=cutoff,mode=mode,policy=policy,**metrics(g)))
    m=pd.DataFrame(summaries);m.to_csv(HERE/'leaderboard.csv',index=False)
    pred.to_csv(HERE/'policy_predictions.csv.gz',index=False)
    (HERE/'external_policy_registry.json').write_text(json.dumps(policyregistry,indent=2,default=str))
    ranks=[]
    for (name,policy),g in m[m.cutoff.eq('2026-01-01')].groupby(['config_id','policy']):
        normal=g[g['mode'].eq('normal')].iloc[0]
        expected={'normal'} if name=='v3_120m' else {'normal','bank_delayed'}
        if ('oxr' in name or 'tabm' in name):expected|={'oxr_delayed','both_delayed'}
        assert expected.issubset(set(g['mode'])),(name,'missing stress',expected-set(g['mode']))
        # Source-independent controls are invariant under bank delay.
        normal_stress=g[g['mode'].isin(['normal','bank_delayed','oxr_delayed','both_delayed'])]
        minlift=normal_stress.lift.min();mincov=normal_stress.weeks_1_2.min();minutility=normal_stress.forward_delta_bps.min()
        past=m[m.config_id.eq(name)&m.policy.eq(policy)&m.cutoff.eq('2025-01-01')]
        pn=past[past['mode'].eq('normal')]
        pass2026=bool(normal.lift>=1.3-1e-12 and normal.weeks_1_2>=.85 and normal.forward_delta_bps>0)
        pass2025=bool(len(pn)==1 and pn.lift.min()>=1.3-1e-12 and pn.weeks_1_2.min()>=.85 and pn.forward_delta_bps.min()>0)
        ranks.append(dict(config_id=name,policy=policy,passed=pass2026 and pass2025,passed_2026=pass2026,passed_2025=pass2025,min_lift=minlift,min_coverage=mincov,min_utility_bps=minutility,normal_lift=normal.lift,normal_coverage=normal.weeks_1_2,normal_utility_bps=normal.forward_delta_bps,normal_2025_lift=pn.lift.min(),min_2025_lift=past.lift.min(),min_2025_coverage=past.weeks_1_2.min(),redundancy=int(name.startswith('blend_')),policy_complexity=int(policy!='legacy')))
    ranking=pd.DataFrame(ranks).sort_values(['passed','normal_lift','normal_utility_bps','normal_2025_lift','redundancy','policy_complexity'],ascending=[False,False,False,False,True,True])
    ranking.to_csv(HERE/'ranking.csv',index=False)
    champion=ranking[ranking.passed].iloc[0].to_dict() if ranking.passed.any() else None
    out=dict(status='provisional' if not tp.exists() else 'complete',selection_uses_2026=True,claim='Retrospective selected model; block intervals conditional on fixed predictions, not selection-adjusted.',fresh_bank_override=True,source_delay_is_selection_gate=False,champion=champion,model_policy_count=len(ranking),fits_hgb_annual=64,fits_hgb_monthly=54)
    if champion:
        intervals=[]
        for year in (2025,2026):
            cand=pred[pred.config_id.eq(champion['config_id'])&pred.policy.eq(champion['policy'])&pred.fold_test_year.eq(year)]
            for mode,b in cand.groupby('mode'):
                intervals.append(dict(year=year,mode=mode,kind='absolute_four_week',**absolute_intervals(b)))
                for base in ('v3_120m','treasury_halyk_shrink_120m'):
                    a=pred[pred.config_id.eq(base)&pred.policy.eq('legacy')&pred.fold_test_year.eq(year)&pred['mode'].eq('normal')]
                    for block in ('month',20,60):
                        intervals.append(dict(year=year,mode=mode,baseline=base,kind='paired_'+str(block),**paired(a,b,block)))
        (HERE/'selected_intervals.json').write_text(json.dumps(intervals,indent=2,default=str))
        pred[pred.config_id.eq(champion['config_id'])&pred.policy.eq(champion['policy'])].to_csv(HERE/'selected_predictions.csv.gz',index=False)
    (HERE/'selection.json').write_text(json.dumps(out,indent=2,default=str));print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
