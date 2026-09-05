"""Paired time-block scores; no fitting or test-based model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
REPS=10000

def point(g):
    cells=g.groupby(['fold_test_year','corridor']).agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
    selected=g[g.candidate_signal].join(cells,on=['fold_test_year','corridor'])
    return dict(rows=len(g),dates=g.date.nunique(),brier=float(np.mean((g.probability-g.target)**2)),signals=len(selected),lift=float(selected.target.sum()/selected.base_hit.sum()),forward_delta_bps=float((selected.forward_bps-selected.base_forward).mean()))

def paired(a,b,block='month'):
    a=a.sort_values(['date','corridor']).reset_index(drop=True)
    b=b.sort_values(['date','corridor']).reset_index(drop=True)
    assert a[['date','corridor']].equals(b[['date','corridor']])
    assert np.array_equal(a.target,b.target)
    assert np.allclose(a.forward_bps,b.forward_bps)
    x=a[['date','corridor','fold_test_year','target','forward_bps']].copy()
    x['delta']=(b.probability-b.target)**2-(a.probability-a.target)**2
    if block=='month':x['block']=x.date.dt.to_period('M').astype(str)
    else:
        ordinal={d:i for i,d in enumerate(sorted(x.date.unique()))}
        x['block']=x.date.map(ordinal)//int(block)
    grouped=x.groupby(['fold_test_year','block']).agg(total=('delta','sum'),n=('delta','size'))
    rng=np.random.default_rng(20260905)
    weights=np.zeros((REPS,len(grouped)))
    for year in grouped.index.get_level_values(0).unique():
        idx=np.flatnonzero(grouped.index.get_level_values(0)==year)
        draws=rng.integers(0,len(idx),(REPS,len(idx)))
        for j in range(len(idx)):weights[:,idx[j]]=(draws==j).sum(axis=1)
    dist=(weights@grouped.total.to_numpy())/(weights@grouped.n.to_numpy())
    result=dict(delta_brier=float(x.delta.mean()),ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),blocks=len(grouped),improved_years=int(x.groupby('fold_test_year').delta.mean().lt(0).sum()))
    if block!='month':return result
    cell_keys=list(x.groupby(['fold_test_year','corridor']).groups)
    block_index={key:i for i,key in enumerate(grouped.index)}
    cell_index={key:i for i,key in enumerate(cell_keys)}
    arrays=np.zeros((3,len(grouped),len(cell_keys),4))
    for i,row in x.iterrows():
        bi=block_index[(row.fold_test_year,row.block)];ci=cell_index[(row.fold_test_year,row.corridor)]
        value=np.array([1.,row.target,row.forward_bps,row.delta])
        arrays[0,bi,ci]+=value
        if a.candidate_signal.iloc[i]:arrays[1,bi,ci]+=value
        if b.candidate_signal.iloc[i]:arrays[2,bi,ci]+=value
    sums=[(weights@v.reshape(len(grouped),-1)).reshape(REPS,len(cell_keys),4) for v in arrays]
    exposure=sums[0]
    hitmean=np.divide(exposure[:,:,1],exposure[:,:,0],out=np.zeros_like(exposure[:,:,0]),where=exposure[:,:,0]>0)
    fmean=np.divide(exposure[:,:,2],exposure[:,:,0],out=np.zeros_like(exposure[:,:,0]),where=exposure[:,:,0]>0)
    values=[]
    for s in sums[1:]:
        count=s[:,:,0].sum(axis=1)
        expected=(s[:,:,0]*hitmean).sum(axis=1)
        values.append(dict(lift=np.divide(s[:,:,1].sum(axis=1),expected,out=np.full(REPS,np.nan),where=expected>0),forward_delta_bps=np.divide((s[:,:,2]-s[:,:,0]*fmean).sum(axis=1),count,out=np.full(REPS,np.nan),where=count>0)))
    pa,pb=point(a),point(b)
    for metric in ('lift','forward_delta_bps'):
        delta=values[1][metric]-values[0][metric]
        result[metric+'_delta']=pb[metric]-pa[metric]
        result[metric+'_ci_low']=float(np.nanquantile(delta,.025))
        result[metric+'_ci_high']=float(np.nanquantile(delta,.975))
    return result

def main():
    dev=pd.read_csv(HERE/'development_predictions.csv.gz',parse_dates=['date'])
    test=pd.read_csv(HERE/'test_predictions.csv.gz',parse_dates=['date'])
    if (HERE/'sensitivity_development_predictions.csv.gz').exists():
        dev=pd.concat([dev,pd.read_csv(HERE/'sensitivity_development_predictions.csv.gz',parse_dates=['date'])],ignore_index=True)
        test=pd.concat([test,pd.read_csv(HERE/'sensitivity_test_predictions.csv.gz',parse_dates=['date'])],ignore_index=True)
    jan=test[test.cutoff.eq('2026-01-01')]
    march=test[test.cutoff.eq('2026-03-01')]
    common=sorted(set(jan.date)&set(march.date))
    tracks={'development_2023_2025':dev,'test2026_january_freeze':jan,'test2026_common_january_freeze':jan[jan.date.isin(common)],'test2026_common_march_freeze':march[march.date.isin(common)]}
    specifications={s['name']:s for s in json.loads((HERE/'specifications.json').read_text())}
    if (HERE/'history_sensitivity_protocol.json').exists():
        specifications.update({s['name']:s for s in json.loads((HERE/'history_sensitivity_protocol.json').read_text())['specs']})
    rows=[];intervals=[];cells=[]
    for track,p in tracks.items():
        for scope in ('all','KZT'):
            q=p if scope=='all' else p[p.corridor.eq('KZT')]
            for name,b in q.groupby('config_id'):
                rows.append(dict(track=track,scope=scope,config_id=name,**point(b)))
                for (year,corridor),part in b.groupby(['fold_test_year','corridor']):
                    cells.append(dict(track=track,scope=scope,config_id=name,year=year,corridor=corridor,**point(part)))
                if name.startswith('v3_'):continue
                reference=f"v3_{specifications[name]['months']}m"
                a=q[q.config_id.eq(reference)]
                for block in ('month',20,60):
                    intervals.append(dict(track=track,scope=scope,config_id=name,baseline=reference,block=str(block),**paired(a,b,block)))
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    pd.DataFrame(cells).to_csv(HERE/'cells.csv',index=False)
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    refit=[]
    for name in specifications:
        for scope in ('all','KZT'):
            a=tracks['test2026_common_january_freeze'];b=tracks['test2026_common_march_freeze']
            a=a[a.config_id.eq(name)];b=b[b.config_id.eq(name)]
            if scope=='KZT':a=a[a.corridor.eq('KZT')];b=b[b.corridor.eq('KZT')]
            refit.append(dict(config_id=name,scope=scope,**paired(a,b)))
    pd.DataFrame(refit).to_csv(HERE/'march_vs_january.csv',index=False)
    history=[]
    for track,p in tracks.items():
        for scope in ('all','KZT'):
            q=p if scope=='all' else p[p.corridor.eq('KZT')]
            for family in ('full','basis'):
                complete=f'oxr_{family}_120m_delay24h'
                for since in (2020,2022):
                    reference=f'oxr_{family}_120m_since{since}'
                    if reference not in specifications:continue
                    a=q[q.config_id.eq(reference)];b=q[q.config_id.eq(complete)]
                    history.append(dict(track=track,scope=scope,family=family,baseline=reference,candidate=complete,**paired(a,b)))
    pd.DataFrame(history).to_csv(HERE/'source_history_pairs.csv',index=False)
    print('Scored',len(rows),'summaries',len(intervals),'paired intervals',flush=True)

if __name__=='__main__':main()
