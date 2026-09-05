#!/usr/bin/env python3
"""Reconstruct every model and policy, plus actual future-data perturbations."""
import importlib.util
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('foundation_fusion_verify',OUT/'experiment.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)


def same_keys(a,b):
    assert np.array_equal(a.date.to_numpy(),b.date.to_numpy())
    assert np.array_equal(a.corridor.to_numpy(),b.corridor.to_numpy())


def main():
    original_inputs=json.loads((OUT/'input_receipt.json').read_text())
    inputs={f:fusion.sha(fusion.ROOT/f)==h for f,h in original_inputs['input_sha256'].items()};assert all(inputs.values())
    builder=fusion.FeatureBuilder();checks=[];cache_checks={}
    selection=json.loads((OUT/'selection.json').read_text())
    predictions=pd.concat([pd.read_csv(OUT/f'{phase}_predictions.csv.gz',parse_dates=['date']) for phase in ['development','test']],ignore_index=True)
    specs={s['name']:s for s in fusion.specs()}
    for cid,cutoff in predictions.groupby(['config_id','cutoff']).groups:
        spec=specs[cid];date=pd.Timestamp(cutoff)
        frame,features,cache=builder.build(spec,date.year)
        name=cid+'_'+cutoff;receipt=json.loads((OUT/'output'/f'{name}_receipt.json').read_text())
        assert all(receipt['checks'].values())
        assert fusion.sha(cache)==receipt['forecast_cache_sha256']
        if cache.name not in cache_checks:
            before=json.loads(cache.with_suffix('.json').read_text());weight=Path(before['checkpoint'])/'model.safetensors'
            assert fusion.sha(weight)==before['weights_sha256']
            if 'small_ft' in cache.name:
                old_fit=json.loads((weight.parent.parent/'fit_receipt.json').read_text())
                assert pd.Timestamp(old_fit['backbone_last_observation'])<pd.Timestamp(date.year-1,1,1)
            cache_checks[cache.name]={'cache_sha256':fusion.sha(cache),'weights_sha256':before['weights_sha256'],'weights_path':str(weight),'passed':True}
        filepath=OUT/'output'/f'{name}_head.joblib';assert fusion.sha(filepath)==receipt['head_sha256']
        pack=joblib.load(filepath);x=features+['corridor']
        train,cal,test=fusion.classic.temporal_split(frame,5,date,pd.Timestamp(date.year+1,1,1),fusion.classic.Spec(cid,months=120,extended=True))
        history=frame[frame.date.ge(date-pd.DateOffset(years=1))&frame.date.lt(date)]
        if spec['local_kzt_calibration']:
            cal=cal[cal.corridor.eq('KZT')];test=test[test.corridor.eq('KZT')];history=history[history.corridor.eq('KZT')]
        future=frame.groupby('corridor').date.shift(-5)
        assert (future.loc[train.index]<date-pd.DateOffset(years=1)).all() and (future.loc[cal.index]<date).all()
        with threadpool_limits(limits=1):
            raw=pack['model'].predict_proba(test[x])[:,1]
            hp=fusion.core.apply_platt(pack['calibrator'],pack['model'].predict_proba(history[x])[:,1])
        prob=fusion.core.apply_platt(pack['calibrator'],raw)
        hsel=fusion.core.select_per_corridor_with_cooldown(history,hp,pack['threshold'])
        state=fusion.core.corridor_selection_state(history,hsel)
        pstate=fusion.core.selection_state(history,fusion.core.select_portfolio_from_candidates(history,hp,hsel))
        chosen=fusion.core.select_per_corridor_with_cooldown(test,prob,pack['threshold'],state)
        portfolio=fusion.core.select_portfolio_from_candidates(test,prob,chosen,pstate)
        pred=predictions[predictions.config_id.eq(cid)&predictions.cutoff.eq(cutoff)].sort_values(['date','corridor']).reset_index(drop=True)
        same_keys(test,pred)
        diff=float(abs(prob-pred.probability).max());rawdiff=float(abs(raw-pred.raw_probability).max())
        mismatch=int((test.index.isin(chosen)!=pred.candidate_signal).sum());pmismatch=int((test.index.isin(portfolio)!=pred.signal).sum())
        assert diff<1e-12 and rawdiff<1e-12 and mismatch==0 and pmismatch==0
        if date.year==2026:assert selection['selection_unix']<(OUT/'output'/f'{name}_predictions.csv.gz').stat().st_mtime
        checks.append({'config_id':cid,'cutoff':cutoff,'raw_probability_max_abs_diff':rawdiff,'probability_max_abs_diff':diff,'candidate_signal_mismatches':mismatch,'portfolio_signal_mismatches':pmismatch,'head_hash_and_label_maturity_pass':True})
    assert len(checks)==85
    # Recover prior strong models on every original year and the explicitMarch fit.
    parity=[]
    for cid,oldname in [('synth_no_oxr','chronos2_synth_zs_head10y'),('small_no_oxr','chronos2_small_ft_head10y')]:
        for cutoff in ['2023-01-01','2024-01-01','2025-01-01','2026-01-01','2026-03-01']:
            a=predictions[predictions.config_id.eq(cid)&predictions.cutoff.eq(cutoff)].sort_values(['date','corridor']).reset_index(drop=True)
            if cutoff.endswith('03-01'):
                b=pd.read_csv(fusion.PREVIOUS/'extended_contract/march/predictions.csv.gz',parse_dates=['date'])
                b=b[b.config_id.eq(oldname)&b.cutoff.eq(cutoff)]
            else:b=pd.read_csv(fusion.PREVIOUS/'extended_contract/output'/f'{oldname}_{cutoff[:4]}_predictions.csv.gz',parse_dates=['date'])
            b=b.sort_values(['date','corridor']).reset_index(drop=True);same_keys(a,b)
            diff=float(abs(a.probability-b.probability).max());mismatch=int((a.candidate_signal!=b.candidate_signal).sum());assert diff<1e-12 and mismatch==0
            parity.append({'config_id':cid,'cutoff':cutoff,'max_probability_diff':diff,'signal_mismatches':mismatch,'passed':True})
    # OXR future perturbation runs the actual public-source transformation.
    cutoff=pd.Timestamp('2024-07-01');changed=builder.raw.copy();change=pd.to_datetime(changed.date).ge(cutoff)
    changed.loc[change,'rub_per_quote']*=7
    altered=fusion.oxr.build_panel(delay=24,since='2010-01-01',raw=changed,extended=True)
    original=builder.panels[('2010-01-01',24)]
    columns=fusion.oxr.BASIS+fusion.oxr.RET+fusion.oxr.COVER
    earlier=original.date.le(cutoff)
    assert np.allclose(original.loc[earlier,columns],altered.loc[earlier,columns],equal_nan=True)
    future_changed=bool((abs(original.loc[~earlier,'oxr_log_basis']-altered.loc[~earlier,'oxr_log_basis'])>1e-6).any());assert future_changed
    # Halyk perturbation intercepts read_csv only in this process; no file writes.
    bank_path=fusion.ROOT/'research_v4/liquidity/halyk_sell_daily.csv'
    read_csv=pd.read_csv
    def changed_reader(path,*args,**kwargs):
        data=read_csv(path,*args,**kwargs)
        if Path(path).resolve()==bank_path.resolve():
            mask=pd.to_datetime(data.date).ge(cutoff);data.loc[mask,'value']*=7
        return data
    try:
        pd.read_csv=changed_reader
        changed_bank,groups,_=fusion.banks.build_panel(extended=True,lag=2)
    finally:pd.read_csv=read_csv
    original_bank,bank_columns=builder.bankframes[2]
    original_bank=original_bank.sort_values(['date','corridor']).reset_index(drop=True)
    changed_bank=changed_bank.sort_values(['date','corridor']).reset_index(drop=True)
    same_keys(original_bank,changed_bank);earlier=original_bank.date.le(cutoff)
    assert np.allclose(original_bank.loc[earlier,bank_columns],changed_bank.loc[earlier,bank_columns],equal_nan=True)
    bank_future_changed=bool((abs(original_bank.loc[~earlier,bank_columns[0]]-changed_bank.loc[~earlier,bank_columns[0]])>1e-6).any());assert bank_future_changed
    intervals=pd.read_csv(OUT/'paired_intervals.csv');assert intervals.repetitions.eq(10000).all()
    for scope,choice in selection['choices'].items():
        computed={cid:float(np.mean((predictions.loc[predictions.config_id.eq(cid)&predictions.cutoff.lt('2026-01-01'),'probability']-predictions.loc[predictions.config_id.eq(cid)&predictions.cutoff.lt('2026-01-01'),'target'])**2)) for cid in choice['brier']}
        assert choice['selected']==min(computed,key=computed.get)
        assert all(abs(computed[k]-v)<1e-12 for k,v in choice['brier'].items())
    result={'status':'PASS','models_reconstructed':85,'configurations':17,'new_neural_fits':0,'new_neural_forecasts':0,'source_input_hashes_unchanged':inputs,'forecast_and_backbone_checks':cache_checks,'saved_head_and_policy_checks':checks,'prior_model_reproduction_checks':parity,'all_same_date_outcome_comparisons':True,'train_and_calibration_maturity_proved':True,'development_selection_verified_before2026_predictions':True,'future_perturbations':{'OXR_actual_pipeline_unchanged_before_cutoff':True,'OXR_future_canary_changes':future_changed,'Halyk_actual_pipeline_unchanged_before_cutoff':True,'Halyk_future_canary_changes':bank_future_changed,'cutoff':cutoff},'paired_block_bootstrap_repetitions':10000,'source_snapshot_sha256':fusion.sha(fusion.SNAPSHOT),'protocol_sha256':fusion.sha(OUT/'protocol.json'),'pretest_halyk_lag1_addendum_sha256':fusion.sha(OUT/'protocol_addendum_halyk_lag1.json'),'code_sha256':fusion.sha(__file__),'experiment_code_sha256':fusion.sha(OUT/'experiment.py'),'limitations':['2026 already inspected retrospective test','Halyk sell-side chart not customer all-in execution quote','Small earlyhead forecasts in-sample w.r.t frozen neural fit','calendar-month bootstrap conditional on fitted models and historical hypothesis search notadjusted']}
    fusion.save(OUT/'verification.json',result)
    print('PASS:85heads,10prior-model reproductions,OXR/Halyk actualfuture perturbations,selectionandpolicy replay.')


if __name__=='__main__':main()
