#!/usr/bin/env python3
"""March freeze using an explicit calibration-history replay and Jan26 backbone."""
from __future__ import annotations
import json
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_heads import OUT, ROOT, OLD, BASE, VARIANTS, core, H, legacy, panel, save, sha
from assess_heads import score, paired


def frame_for(variant,p,wide):
    if variant=='base_control':return p,BASE
    file=OUT/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_2026"}.npz'
    item=np.load(file)
    frame,extra=legacy.forecast_features(p,wide,pd.DatetimeIndex(item['dates']),item['quantiles'],item['grid'])
    return frame,BASE+extra


def evaluate(p,features,cutoff,years,cid):
    cutoff=pd.Timestamp(cutoff);cal_start=cutoff-pd.DateOffset(years=1)
    train_start=cal_start-pd.DateOffset(years=years)
    eligible=p[p.target.notna()]
    train=core.purge_tail(eligible[eligible.date.ge(train_start)&eligible.date.lt(cal_start)],H)
    cal=core.purge_tail(eligible[eligible.date.ge(cal_start)&eligible.date.lt(cutoff)],H)
    test=eligible[eligible.date.ge(cutoff)&eligible.date.lt('2027-01-01')].copy()
    history=p[p.date.ge(cal_start)&p.date.lt(cutoff)].copy()
    future=p.groupby('corridor').date.shift(-H)
    assert (future.loc[train.index]<cal_start).all()
    assert (future.loc[cal.index]<cutoff).all()
    model=core.make_model('hist_gradient_boosting',features)
    model.named_steps['classifier'].set_params(early_stopping=False)
    columns=features+['corridor'];t=time.perf_counter()
    model.fit(train[columns],train.target.astype(int))
    raw_cal=model.predict_proba(cal[columns])[:,1]
    calibrator=core.fit_platt_calibrator(raw_cal,cal.target)
    threshold,_,_=core.choose_frequency_threshold(cal,core.apply_platt(calibrator,raw_cal))
    hp=core.apply_platt(calibrator,model.predict_proba(history[columns])[:,1])
    hi=core.select_per_corridor_with_cooldown(history,hp,threshold)
    state=core.corridor_selection_state(history,hi)
    portfolio_state=core.selection_state(history,core.select_portfolio_from_candidates(history,hp,hi))
    raw=model.predict_proba(test[columns])[:,1]
    prob=core.apply_platt(calibrator,raw)
    selected=core.select_per_corridor_with_cooldown(test,prob,threshold,state)
    portfolio=core.select_portfolio_from_candidates(test,prob,selected,portfolio_state)
    pred=test[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal']].copy()
    pred['probability']=prob;pred['raw_probability']=raw
    pred['candidate_signal']=pred.index.isin(selected);pred['signal']=pred.index.isin(portfolio)
    pred['fold_test_year']=2026;pred['config_id']=cid;pred['cutoff']=str(cutoff.date())
    receipt={'config_id':cid,'cutoff':cutoff,'head_years':years,'train_first':train.date.min(),'train_last':train.date.max(),'train_latest_label':future.loc[train.index].max(),'train_rows':len(train),'calibration_first':cal.date.min(),'calibration_last':cal.date.max(),'calibration_latest_label':future.loc[cal.index].max(),'calibration_rows':len(cal),'history_first':history.date.min(),'history_last':history.date.max(),'history_initial_candidate_state_for_test':state,'test_first':test.date.min(),'test_last':test.date.max(),'test_rows':len(test),'backbone':'same read-only Jan2026 checkpoint trained through2024, never fine-tuned on2026','features':features,'seconds':time.perf_counter()-t,'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),'train_and_calibration_label_maturity_pass':True}
    return pred,{'model':model,'calibrator':calibrator,'threshold':threshold,'features':features},receipt


def main():
    p=panel();wide,_=legacy.load_history()
    annual=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['date'])
    out=OUT/'march';out.mkdir(exist_ok=True)
    frames=[];checks=[]
    with threadpool_limits(limits=2):
        for variant in VARIANTS:
            frame,features=frame_for(variant,p,wide)
            for years in [2,10]:
                cid=f'{variant}_head{years}y'
                # Full Jan replay reconstruction verifies the bespoke routine.
                jan,_,_=evaluate(frame,features,'2026-01-01',years,cid)
                old=annual[annual.config_id.eq(cid)&annual.fold_test_year.eq(2026)].sort_values(['date','corridor']).reset_index(drop=True)
                jan=jan.sort_values(['date','corridor']).reset_index(drop=True)
                assert np.array_equal(jan.date.to_numpy(),old.date.to_numpy())
                assert np.array_equal(jan.corridor.to_numpy(),old.corridor.to_numpy())
                diff=float(abs(jan.probability-old.probability).max())
                mismatch=int((jan.candidate_signal!=old.candidate_signal).sum())
                assert diff<1e-12 and mismatch==0,(cid,diff,mismatch)
                checks.append({'config_id':cid,'probability_max_abs_diff':diff,'candidate_signal_mismatches':mismatch,'passed':True})
                march,pack,receipt=evaluate(frame,features,'2026-03-01',years,cid)
                march.to_csv(out/f'{cid}_predictions.csv.gz',index=False)
                path=out/f'{cid}_checkpoint.joblib';joblib.dump(pack,path)
                save(out/f'{cid}_receipt.json',{**receipt,'head_sha256':sha(path)})
                frames.extend([jan,march]);print(cid,'March freeze complete',flush=True)
    pred=pd.concat(frames,ignore_index=True)
    pred.to_csv(out/'predictions.csv.gz',index=False)
    summaries=[];intervals=[]
    # Compare cutoffs on exactly the same March onward row set.
    common=pred[pred.date.ge('2026-03-01')]
    common.to_csv(out/'common_march_onward_predictions.csv.gz',index=False)
    for scope in ['all','KZT']:
        data=common if scope=='all' else common[common.corridor.eq('KZT')]
        grouped={(cid,cutoff):f.sort_values(['date','corridor']).reset_index(drop=True) for (cid,cutoff),f in data.groupby(['config_id','cutoff'])}
        for (cid,cutoff),f in grouped.items():
            if '_kzt_' in cid and scope!='KZT':continue
            summaries.append({'scope':scope,'config_id':cid,'cutoff':cutoff,'first_date':f.date.min(),'last_date':f.date.max(),**score(f)})
            comparisons=[]
            if cutoff=='2026-03-01':comparisons.append((cid,'2026-01-01'))
            if cid.endswith('_head10y'):comparisons.append((cid.removesuffix('_head10y')+'_head2y',cutoff))
            for other in comparisons:
                with threadpool_limits(limits=2):result=paired(f,grouped[other],10000)
                intervals.append({'scope':scope,'config_id':cid,'cutoff':cutoff,'benchmark':other[0],'benchmark_cutoff':other[1],**result})
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    save(out/'jan_replay_parity.json',checks)
    save(out/'receipt.json',{'same_test_dates_for_jan_vs_march':True,'jan_replay_matches_original_for_all10_heads':True,'test_first':common.date.min(),'test_last':common.date.max(),'distinct_dates':common.date.nunique(),'protocol_sha256':sha(OUT/'protocol.json'),'code_sha256':sha(__file__),'six_calendar_months_complete':False,'interpretation':'March-August interval truncated at last mature public CBR label; six full months are unavailable.'})
    print(pd.DataFrame(summaries).query("scope=='all'")[['config_id','cutoff','brier','lift','forward_delta_bps','signals']].to_string(index=False))


if __name__=='__main__':main()
