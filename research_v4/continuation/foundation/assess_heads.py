#!/usr/bin/env python3
"""Read-only result aggregation with paired month blocks and exact-key controls."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_heads import OUT, ROOT, OLD, sha, save, panel, legacy

spec=importlib.util.spec_from_file_location('old_foundation_assess',OLD/'assess.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
score,paired=helpers.score,helpers.paired


def main():
    frames=[]
    for path in sorted((OUT/'output').glob('*_predictions.csv.gz')):
        frame=pd.read_csv(path,parse_dates=['date']);frames.append(frame)
    assert len(frames)==40, f'Expected 40 annual heads, found {len(frames)}'
    for cid in ['baseline_reproduction','basis_train_120m']:
        frame=pd.read_csv(ROOT/'research_v3/models'/f'{cid}_h5_predictions.csv.gz',parse_dates=['date'])
        frame['config_id']=cid;frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(OUT/'predictions.csv.gz',index=False)
    cell_rows=[]
    for (cid,year,corridor),frame in data.groupby(['config_id','fold_test_year','corridor']):
        cell_rows.append({'config_id':cid,'fold_test_year':year,'corridor':corridor,**score(frame)})
    pd.DataFrame(cell_rows).to_csv(OUT/'cells.csv',index=False)
    summaries=[];intervals=[]
    for stage,years in [('development_2023_2025',[2023,2024,2025]),('inspected_retrospective_2026',[2026])]:
        subset=data[data.fold_test_year.isin(years)]
        for scope in ['all','KZT']:
            scoped=subset if scope=='all' else subset[subset.corridor.eq('KZT')]
            grouped={cid:frame.sort_values(['date','corridor']).reset_index(drop=True) for cid,frame in scoped.groupby('config_id')}
            reference=grouped['baseline_reproduction']
            for cid,frame in grouped.items():
                assert frame[['date','corridor']].equals(reference[['date','corridor']]),cid
                for field in ['target','forward_bps','symmetric_bps','regret_bps']:
                    assert np.allclose(frame[field],reference[field]),(cid,field)
                if '_kzt_' in cid and scope!='KZT':continue
                summaries.append({'stage':stage,'scope':scope,'config_id':cid,**score(frame)})
                alternatives=['baseline_reproduction','basis_train_120m']
                if cid.endswith('_head10y'): alternatives.append(cid.removesuffix('_head10y')+'_head2y')
                if cid.startswith('chronos2_'):
                    alternatives.append('base_control_'+cid.rsplit('_',1)[-1])
                if '_ft_kzt_' in cid: alternatives.append(cid.replace('_ft_kzt_','_ft_'))
                for baseline in dict.fromkeys(alternatives):
                    if cid==baseline or baseline not in grouped:continue
                    with threadpool_limits(limits=2):
                        result=paired(frame,grouped[baseline],10000)
                    intervals.append({'stage':stage,'scope':scope,'config_id':cid,'benchmark':baseline,**result})
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    # Matched two-year controls must reproduce prior numerical outputs exactly.
    mapping={'base_control_head2y':'baseline_reproduction','chronos2_small_ft_head2y':'chronos2_small_ft_hgb_aug','chronos2_synth_ft_head2y':'chronos2_synth_ft_hgb_aug','chronos2_synth_ft_kzt_head2y':'chronos2_synth_ft_kzt_hgb_aug','chronos2_synth_zs_head2y':'chronos2_synth_zs_hgb_aug'}
    old=pd.read_csv(OLD/'predictions.csv.gz',parse_dates=['date'])
    checks=[]
    for cid,prior in mapping.items():
        a=data[data.config_id.eq(cid)].sort_values(['date','corridor']).reset_index(drop=True)
        b=old[old.config_id.eq(prior)].sort_values(['date','corridor']).reset_index(drop=True)
        assert a[['date','corridor']].equals(b[['date','corridor']]),prior
        diff=np.max(abs(a.probability-b.probability))
        rawdiff=np.max(abs(a.raw_probability-b.raw_probability))
        mismatch=int((a.candidate_signal!=b.candidate_signal).sum())
        checks.append({'new_config':cid,'old_config':prior,'rows':len(a),'probability_max_abs_diff':float(diff),'raw_probability_max_abs_diff':float(rawdiff),'candidate_signal_mismatches':mismatch,'passed':bool(diff<1e-12 and rawdiff<1e-12 and mismatch==0)})
    save(OUT/'reproduction_checks.json',checks)
    save(OUT/'assessment_receipt.json',{'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'repetitions':10000,'bootstrap':'year-stratified paired whole-month blocks with all corridors shared and cell baselines recomputed; fixed models and signals; not selection-proof','annual_heads':40,'prediction_rows':len(data),'all_date_outcome_keys_match_V3':True,'old_2y_reproduction_pass':all(x['passed'] for x in checks),'input_sha256':{str(x.relative_to(OUT)):sha(x) for x in sorted((OUT/'output').glob('*_predictions.csv.gz'))}})
    print(pd.DataFrame(summaries).query("scope=='all'")[['stage','config_id','brier','lift','forward_delta_bps','signals']].to_string(index=False))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
