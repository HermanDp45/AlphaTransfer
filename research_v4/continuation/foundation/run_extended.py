#!/usr/bin/env python3
"""Exact V3 extended-panel contract, reusing all immutable forecast arrays."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_heads as heads
from assess_heads import score,paired

HERE=heads.OUT

def panel_exact():
    p=pd.read_pickle(heads.ROOT/'research_v3/models/panel_extended.pkl')
    p=p[['date','corridor','rub_per_unit','session_ordinal',*dict.fromkeys(heads.BASE)]].copy()
    return heads.core.add_target(p,heads.H)

def assess(directory,control_directory=None):
    frames=[pd.read_csv(f,parse_dates=['date']) for f in sorted((directory/'output').glob('*_predictions.csv.gz'))]
    for cid in ['baseline_reproduction','basis_train_120m']:
        f=pd.read_csv(heads.ROOT/'research_v3/models'/f'{cid}_h5_predictions.csv.gz',parse_dates=['date']);f['config_id']=cid;frames.append(f)
    if control_directory is not None:
        for f in sorted((control_directory/'output').glob('chronos2_small_ft_head10y_*_predictions.csv.gz')):
            frames.append(pd.read_csv(f,parse_dates=['date']))
    data=pd.concat(frames,ignore_index=True);data.to_csv(directory/'predictions.csv.gz',index=False)
    rows=[];intervals=[];cells=[]
    for (cid,year,corridor),f in data.groupby(['config_id','fold_test_year','corridor']):cells.append({'config_id':cid,'fold_test_year':year,'corridor':corridor,**score(f)})
    pd.DataFrame(cells).to_csv(directory/'cells.csv',index=False)
    for stage,years in [('development_2023_2025',[2023,2024,2025]),('inspected_retrospective_2026',[2026])]:
        subset=data[data.fold_test_year.isin(years)]
        for scope in ['all','KZT']:
            scoped=subset if scope=='all' else subset[subset.corridor.eq('KZT')]
            grouped={cid:f.sort_values(['date','corridor']).reset_index(drop=True) for cid,f in scoped.groupby('config_id')}
            for cid,f in grouped.items():
                if '_kzt_' in cid and scope!='KZT':continue
                rows.append({'stage':stage,'scope':scope,'config_id':cid,**score(f)})
                alternatives=['baseline_reproduction','basis_train_120m']
                if 'ft900' in cid:alternatives.append(cid.replace('ft900','ft'))
                if '_ft_kzt_' in cid:alternatives.append(cid.replace('_ft_kzt_','_ft_'))
                for other in dict.fromkeys(alternatives):
                    if other==cid or other not in grouped:continue
                    with threadpool_limits(limits=2):result=paired(f,grouped[other],10000)
                    intervals.append({'stage':stage,'scope':scope,'config_id':cid,'benchmark':other,**result})
    pd.DataFrame(rows).to_csv(directory/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(directory/'paired_intervals.csv',index=False)
    print(pd.DataFrame(rows).query("scope=='all'")[['stage','config_id','brier','lift','forward_delta_bps','signals']].to_string(index=False))
    if 'base_control_head10y' in set(data.config_id):
        a=data[data.config_id.eq('base_control_head10y')].sort_values(['date','corridor']).reset_index(drop=True)
        b=data[data.config_id.eq('basis_train_120m')].sort_values(['date','corridor']).reset_index(drop=True)
        assert a[['date','corridor']].equals(b[['date','corridor']])
        diff=float(abs(a.probability-b.probability).max());mismatch=int((a.candidate_signal!=b.candidate_signal).sum())
        assert diff<1e-12 and mismatch==0
        heads.save(directory/'v3long_reproduction.json',{'probability_max_abs_diff':diff,'candidate_signal_mismatches':mismatch,'rows':len(a),'passed':True})

def main():
    heads.OUT=HERE/'extended_contract'
    p=panel_exact();wide,_=heads.legacy.load_history()
    for variant in heads.VARIANTS:
        for year in [2023,2024,2025,2026]:
            if variant=='base_control':frame,features=p,heads.BASE
            else:
                path=HERE/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_"+str(year)}.npz'
                item=np.load(path)
                frame,extra=heads.legacy.forecast_features(p,wide,pd.DatetimeIndex(item['dates']),item['quantiles'],item['grid']);features=heads.BASE+extra
            heads.run_head(frame,features,year,variant,10)
    assess(heads.OUT)
    heads.save(heads.OUT/'receipt.json',{'head_count':20,'panel':'exact immutable panel_extended.pkl','panel_sha256':heads.sha(heads.ROOT/'research_v3/models/panel_extended.pkl'),'code_sha256':heads.sha(__file__),'protocol_sha256':heads.sha(heads.OUT/'protocol.json'),'forecast_arrays':'read-only ../forecasts; identical arrays to matched-window experiment'})

if __name__=='__main__':main()
