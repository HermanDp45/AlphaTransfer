#!/usr/bin/env python3
"""Paired external-data ablations on fixed neural forecasts."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('foundation_fusion',OUT/'experiment.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
score,paired=fusion.score,fusion.paired


def comparisons(cid):
    if cid.startswith('small_'):return ['small_no_oxr','v3_long']
    if cid.startswith('synth_kzt_'):
        alternatives=['synth_kzt_no_oxr','synth_no_oxr','v3_long']
        if 'basis2010_halyk' in cid:
            alternatives+=['synth_kzt_basis2010_D2','synth_kzt_halyk_D2']
        if cid.endswith('_D3'):
            alternatives.append(cid.removesuffix('_D3')+'_D2')
        if cid.endswith('_D1'):alternatives.append(cid.removesuffix('_D1')+'_D2')
        if cid.endswith('_H1O2'):alternatives+=['synth_kzt_halyk_D1','synth_kzt_basis2010_halyk_D2']
        return alternatives
    alternatives=['synth_no_oxr','v3_long']
    if cid=='synth_basis2010_D2':alternatives+=['synth_basis2018_D2','synth_availability2010_D2']
    if cid=='synth_full2010_D2':alternatives+=['synth_basis2010_D2','synth_availability2010_D2']
    if cid.endswith('_D3'):alternatives.append(cid.removesuffix('_D3')+'_D2')
    return alternatives


def main():
    dev=pd.read_csv(OUT/'development_predictions.csv.gz',parse_dates=['date'])
    test=pd.read_csv(OUT/'test_predictions.csv.gz',parse_dates=['date'])
    data=pd.concat([dev,test],ignore_index=True)
    references=[]
    for cutoff in ['2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01']:
        path=fusion.ROOT/'research_v4/continuation/oxr/output'/f'v3_120m_{cutoff}.csv.gz'
        assert path.exists(),path
        frame=pd.read_csv(path,parse_dates=['date']);frame['cutoff']=cutoff;frame['config_id']='v3_long';references.append(frame)
    data=pd.concat([data,*references],ignore_index=True)
    data.to_csv(OUT/'predictions.csv.gz',index=False)
    cells=[]
    for (cid,cutoff,corridor),f in data.groupby(['config_id','cutoff','corridor']):
        cells.append({'config_id':cid,'cutoff':cutoff,'corridor':corridor,**score(f)})
    pd.DataFrame(cells).to_csv(OUT/'cells.csv',index=False)
    summaries=[];intervals=[]
    stages={'development_2023_2025':data.cutoff.lt('2026-01-01'),'jan2026_retrospective':data.cutoff.eq('2026-01-01'),'march2026_retrospective':data.cutoff.eq('2026-03-01')}
    for stage,mask in stages.items():
        for scope in ['all','KZT']:
            scoped=data[mask]
            if scope=='KZT':scoped=scoped[scoped.corridor.eq('KZT')]
            grouped={cid:f.sort_values(['date','corridor']).reset_index(drop=True) for cid,f in scoped.groupby('config_id') if not (scope=='all' and cid.startswith('synth_kzt_'))}
            reference=grouped['v3_long']
            for cid,frame in grouped.items():
                assert frame[['date','corridor']].equals(reference[['date','corridor']]),(cid,stage,scope)
                for key in ['target','forward_bps','symmetric_bps','regret_bps']:assert np.allclose(frame[key],reference[key]),(cid,key)
                summaries.append({'stage':stage,'scope':scope,'config_id':cid,'date_first':frame.date.min(),'date_last':frame.date.max(),**score(frame)})
                for baseline in dict.fromkeys(comparisons(cid)):
                    if baseline==cid or baseline not in grouped:continue
                    with threadpool_limits(limits=1):result=paired(frame,grouped[baseline],10000)
                    intervals.append({'stage':stage,'scope':scope,'config_id':cid,'benchmark':baseline,**result})
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    # Same March-onward dates isolate cutoff changes from evaluation-period mix.
    common=data[data.date.ge('2026-03-01')&data.cutoff.isin(['2026-01-01','2026-03-01'])]
    common.to_csv(OUT/'common_march_predictions.csv.gz',index=False)
    rows=[];cis=[]
    for scope in ['all','KZT']:
        scoped=common if scope=='all' else common[common.corridor.eq('KZT')]
        for cid,f in scoped.groupby('config_id'):
            if scope=='all' and cid.startswith('synth_kzt_'):continue
            grouped={cutoff:g.sort_values(['date','corridor']).reset_index(drop=True) for cutoff,g in f.groupby('cutoff')}
            for cutoff,g in grouped.items():rows.append({'scope':scope,'config_id':cid,'cutoff':cutoff,'date_first':g.date.min(),'date_last':g.date.max(),**score(g)})
            with threadpool_limits(limits=1):result=paired(grouped['2026-03-01'],grouped['2026-01-01'],10000)
            cis.append({'scope':scope,'config_id':cid,'comparison':'March freeze minus January freeze, same March-onward dates',**result})
    pd.DataFrame(rows).to_csv(OUT/'common_march_summary.csv',index=False)
    pd.DataFrame(cis).to_csv(OUT/'common_march_paired_intervals.csv',index=False)
    fusion.save(OUT/'assessment_receipt.json',{'configurations':len(fusion.specs()),'fits':85,'all_keys_and_outcomes_match_v3':True,'bootstrap_repetitions':10000,'bootstrap':'year-stratified paired calendar-month blocks; allcorridors shared; cell baselines recomputed','sources_sha256':{str(p.relative_to(OUT)):fusion.sha(p) for p in [OUT/'development_predictions.csv.gz',OUT/'test_predictions.csv.gz']},'code_sha256':fusion.sha(__file__),'selection_sha256':fusion.sha(OUT/'selection.json')})
    print(pd.DataFrame(summaries).query("scope=='all'")[['stage','config_id','brier','lift','forward_delta_bps','signals']].to_string(index=False))


if __name__=='__main__':main()
