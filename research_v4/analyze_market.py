"""Shared V4 market readout; matched outcomes, joint month uncertainty.

Intervals condition on already trained and historically inspected models.
They are not selection-adjusted confirmation or bank customer savings.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'research_v3/models'))
import numpy as np
import pandas as pd
from experiment import summarize,core
from assess import bootstrap_paired
import decision_uncertainty as du
HERE=Path(__file__).resolve().parent

def load_models():
    results={}
    for name in ('baseline_reproduction','basis_train_120m'):
        results[name]=pd.read_csv(ROOT/f'research_v3/models/{name}_h5_predictions.csv.gz',parse_dates=['date'])
    ext=pd.read_csv(ROOT/'research_v3/external_data/long_combo/predictions.csv.gz',parse_dates=['date'])
    for name,g in ext.groupby('config_id'):
        if 'treasury_lag7' in name:results[name]=g.copy()
    for folder,pattern in [('liquidity/predictions','*.csv.gz'),('kazakhstan','kzt_*_predictions.csv.gz')]:
        for path in sorted((HERE/folder).glob(pattern)):
            p=pd.read_csv(path,parse_dates=['date']);results[str(p.config_id.iloc[0])]=p
    return results

def paired_policy(base,candidate,reps=10000):
    a=base.sort_values(['date','corridor']).reset_index(drop=True)
    b=candidate.sort_values(['date','corridor']).reset_index(drop=True)
    assert a[['date','corridor']].equals(b[['date','corridor']])
    exposure,signals,months=du.aggregate_block_arrays(a,{'base':a.candidate_signal.to_numpy(bool),'candidate':b.candidate_signal.to_numpy(bool)})
    weights=du.block_weights(months,reps);n_cells=exposure.shape[1]
    e=(weights@exposure.reshape(len(months),-1)).reshape(reps,n_cells,5)
    points={};draws={}
    for name,s in signals.items():
        points[name]=du.evaluate_sums(exposure.sum(axis=0),s.sum(axis=0))
        draws[name]=du.evaluate_sums(e,(weights@s.reshape(len(months),-1)).reshape(reps,n_cells,5))
    result={}
    for metric in du.METRICS:
        diff=draws['candidate'][metric]-draws['base'][metric];finite=np.isfinite(diff)
        bounds=np.quantile(diff[finite],[.025,.975]) if finite.any() else [np.nan,np.nan]
        result.update({metric+'_delta':float(points['candidate'][metric]-points['base'][metric]),metric+'_ci_low':bounds[0],metric+'_ci_high':bounds[1]})
    return result

def main():
    models=load_models();summary=[];cells=[];uncertainty=[];policy=[];parity=[]
    basenames=('baseline_reproduction','basis_train_120m')
    for name,p in models.items():
        p=p.sort_values(['date','corridor']).reset_index(drop=True)
        scopes=['all','KZT'] if p.corridor.nunique()>1 else ['KZT']
        for scope in scopes:
            q=p if scope=='all' else p[p.corridor.eq('KZT')].copy()
            s=summarize(q);s['scope']=scope
            s['mean_signals_per_corridor_week']=[du.cadence(q[q.fold_test_year.le(2025) if row.track.startswith('development') else q.fold_test_year.eq(2026)],q.loc[q.fold_test_year.le(2025) if row.track.startswith('development') else q.fold_test_year.eq(2026),'candidate_signal'].to_numpy(bool))['mean_signals_per_corridor_week'] for row in s.itertuples()]
            summary.append(s)
            for (year,corridor),g in q.groupby(['fold_test_year','corridor']):
                cells.append({'config_id':name,'scope':scope,'year':year,'corridor':corridor,'rows':len(g),'brier':float(((g.probability-g.target)**2).mean()),'signals':int(g.candidate_signal.sum())})
            if name in basenames:continue
            references=list(basenames)
            if name.startswith('kzt_'):
                window='120m' if '120m' in name else '24m'
                references.append('kzt_pooled_'+window)
                if '__' in name:references.append('kzt_pooled_'+window+'__'+name.split('__',1)[1])
            if name.startswith('combo_'):
                references.extend(k for k in models if k.startswith('basis_train_120m__treasury_lag7'))
            for baseline in dict.fromkeys(references):
                if baseline==name:continue
                b=models[baseline];b=b[b.corridor.isin(q.corridor.unique())]
                cols=['date','corridor','target','probability','raw_probability','candidate_signal','forward_bps','symmetric_bps','regret_bps']
                j=q.merge(b[cols].rename(columns={c:'base_'+c for c in cols if c not in ['date','corridor']}),on=['date','corridor'],validate='one_to_one')
                assert len(j)==len(b)==len(q),(name,scope,len(j),len(b),len(q))
                for key in ('target','forward_bps','symmetric_bps','regret_bps'):
                    assert np.allclose(j[key],j['base_'+key],rtol=1e-10,atol=1e-8,equal_nan=True),(name,key)
                parity.append({'name':name,'scope':scope,'baseline':baseline,'matched_rows':len(j),'target_and_payoff_exact':True,'max_raw_probability_delta':float((j.raw_probability-j.base_raw_probability).abs().max())})
                for track,part in [('development_2023_2025',j[j.fold_test_year<=2025]),('diagnostic_2026',j[j.fold_test_year==2026])]:
                    for block in ('month',20,60):
                        uncertainty.append({'config_id':name,'baseline':baseline,'scope':scope,'track':track,'block':str(block),**bootstrap_paired(part,block=block)})
                    # Full family is descriptive; no posthoc winner is hidden.
                    a=b[b.date.isin(part.date)]
                    c=q[q.date.isin(part.date)]
                    policy.append({'config_id':name,'baseline':baseline,'scope':scope,'track':track,'trim_dates':0,**paired_policy(a,c)})
                    if scope=='KZT' and ('halyk_lag' in name or 'kase_prices_lag' in name):
                        dates=part.groupby('fold_test_year').date.apply(lambda x:sorted(x.unique())[20:])
                        keep=set(d for ds in dates for d in ds)
                        policy.append({'config_id':name,'baseline':baseline,'scope':scope,'track':track,'trim_dates':20,**paired_policy(a[a.date.isin(keep)],c[c.date.isin(keep)])})
    pd.concat(summary,ignore_index=True).to_csv(HERE/'MARKET_COMPARISON.csv',index=False)
    pd.DataFrame(cells).to_csv(HERE/'market_cells.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(HERE/'market_paired_uncertainty.csv',index=False)
    pd.DataFrame(policy).to_csv(HERE/'market_policy_paired_uncertainty.csv',index=False)
    (HERE/'market_parity.json').write_text(json.dumps({'status':'PASS','comparisons':parity,'repetitions':10000,'method':'year-stratified joint calendar-month blocks; random-day baseline recomputed in policy draws; 20/60-row-block sensitivity for Brier','scope':'retrospective conditional on fixed models; no correction for historical selection'},indent=2))
    print('PASS',len(models),'models,',len(uncertainty),'proper-score intervals,',len(policy),'policy comparisons',flush=True)
if __name__=='__main__':main()
