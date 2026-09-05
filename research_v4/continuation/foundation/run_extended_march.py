#!/usr/bin/env python3
"""Same explicit Jan/March replay on the exact V3 extended feature contract."""
import datetime
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_heads import OUT,VARIANTS,BASE,legacy,save,sha
from run_extended import panel_exact
from run_march import frame_for,evaluate
from assess_heads import score,paired

def main():
    output=OUT/'extended_contract/march';output.mkdir(exist_ok=True)
    protocol=output/'protocol.json'
    if not protocol.exists():save(protocol,{'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'all five prespecified backbones/base controls,10y heads, exact immutable V3extended base feature panel; same Jan2026 backbone for Jan and Marchcutoffs','cutoffs':['2026-01-01','2026-03-01'],'parent_protocol_sha256':sha(OUT/'protocol.json'),'head_features':'exact extended contract already prespecified; no candidate selection from2026','calibration':'12months immediately before each cutoff, purged5; replay own full calibration history','test':'same mature March3-Aug25 dates, incomplete6calendar months; previously inspected retrospective diagnostic'})
    p=panel_exact();wide,_=legacy.load_history()
    annual=pd.read_csv(OUT/'extended_contract/predictions.csv.gz',parse_dates=['date'],low_memory=False)
    frames=[];checks=[]
    with threadpool_limits(limits=2):
        for variant in VARIANTS:
            frame,features=frame_for(variant,p,wide);cid=variant+'_head10y'
            jan,_,_=evaluate(frame,features,'2026-01-01',10,cid)
            old=annual[annual.config_id.eq(cid)&annual.fold_test_year.eq(2026)].sort_values(['date','corridor']).reset_index(drop=True)
            jan=jan.sort_values(['date','corridor']).reset_index(drop=True)
            assert np.array_equal(jan.date.to_numpy(),old.date.to_numpy()) and np.array_equal(jan.corridor.to_numpy(),old.corridor.to_numpy())
            diff=float(abs(jan.probability-old.probability).max());mismatch=int((jan.candidate_signal!=old.candidate_signal).sum())
            assert diff<1e-12 and mismatch==0
            checks.append({'config_id':cid,'probability_max_abs_diff':diff,'candidate_signal_mismatches':mismatch,'passed':True})
            march,pack,receipt=evaluate(frame,features,'2026-03-01',10,cid)
            path=output/f'{cid}_checkpoint.joblib';joblib.dump(pack,path)
            save(output/f'{cid}_receipt.json',{**receipt,'head_sha256':sha(path),'parent_evaluation_code_sha256':receipt['code_sha256'],'driver_code_sha256':sha(__file__),'protocol_sha256':sha(protocol)})
            frames.extend([jan,march])
    pred=pd.concat(frames,ignore_index=True);pred.to_csv(output/'predictions.csv.gz',index=False)
    common=pred[pred.date.ge('2026-03-01')];common.to_csv(output/'common_march_onward_predictions.csv.gz',index=False)
    rows=[];intervals=[]
    for scope in ['all','KZT']:
        data=common if scope=='all' else common[common.corridor.eq('KZT')]
        grouped={(cid,cutoff):f.sort_values(['date','corridor']).reset_index(drop=True) for (cid,cutoff),f in data.groupby(['config_id','cutoff'])}
        for (cid,cutoff),f in grouped.items():
            if '_kzt_' in cid and scope!='KZT':continue
            rows.append({'scope':scope,'config_id':cid,'cutoff':cutoff,'first_date':f.date.min(),'last_date':f.date.max(),**score(f)})
            if cutoff=='2026-03-01':
                with threadpool_limits(limits=2):result=paired(f,grouped[(cid,'2026-01-01')],10000)
                intervals.append({'scope':scope,'config_id':cid,'cutoff':cutoff,'benchmark':cid,'benchmark_cutoff':'2026-01-01',**result})
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False);pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    save(output/'jan_replay_parity.json',checks)
    save(output/'receipt.json',{'same_test_dates_for_jan_vs_march':True,'distinct_dates':common.date.nunique(),'test_first':common.date.min(),'test_last':common.date.max(),'six_calendar_months_complete':False,'code_sha256':sha(__file__),'protocol_sha256':sha(protocol),'all5_jan_replays_pass':True})
    print(pd.DataFrame(rows).query("scope=='all'")[['config_id','cutoff','brier','forward_delta_bps','signals']].to_string(index=False))

if __name__=='__main__':main()
