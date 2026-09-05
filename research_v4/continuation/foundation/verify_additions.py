#!/usr/bin/env python3
"""Reconstruct additional heads and score fixed900-step raw quantile forecasts."""
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from run_heads import OUT,ROOT,OLD,BASE,H,legacy,save,sha
from run_extended import panel_exact
from run_march import frame_for
from assess_heads import helpers,score,paired

def main():
    p=panel_exact();wide,logs=legacy.load_history();checks=[];losses=[]
    for directory in [OUT/'extended_contract',OUT/'budget900',OUT/'budget900/cpu300']:
        for file in sorted((directory/'output').glob('*_predictions.csv.gz')):
            pred=pd.read_csv(file,parse_dates=['date']);cid=str(pred.config_id.iloc[0]);year=int(pred.fold_test_year.iloc[0]);variant=cid.removesuffix('_head10y')
            receipt=json.loads((directory/'output'/f'{cid}_{year}_receipt.json').read_text())
            if variant=='base_control':frame,features=p,BASE
            else:
                forecast_root=directory if ('ft900' in variant or 'cpu300' in variant) else OUT
                path=forecast_root/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_"+str(year)}.npz'
                item=np.load(path);dates=pd.DatetimeIndex(item['dates'])
                assert np.isfinite(item['quantiles']).all()
                frame,extra=legacy.forecast_features(p,wide,dates,item['quantiles'],item['grid']);features=BASE+extra
                if 'ft900' in variant:
                    fit=json.loads((directory/'checkpoints/chronos2_small'/str(year)/'fit_receipt.json').read_text())
                    ckpt=directory/'checkpoints/chronos2_small'/str(year)/'finetuned-ckpt/model.safetensors'
                    assert sha(ckpt)==fit['weights_sha256'] and fit['steps']==900
                    assert pd.Timestamp(fit['backbone_last_observation'])<pd.Timestamp(year-1,1,1)
                    assert fit['parameters_changed']>fit['parameters']*.5
                    losses.append(legacy.forecast_metrics(wide,logs,dates,item['quantiles'],item['grid'],year,variant,pred[['date','corridor']]))
            modelpath=directory/'output'/f'{cid}_{year}_head.joblib';assert sha(modelpath)==receipt['head_sha256']
            model=joblib.load(modelpath)
            merged=frame.merge(pred[['date','corridor','raw_probability']],on=['date','corridor'],validate='one_to_one')
            with threadpool_limits(limits=2):actual=model.predict_proba(merged[features+['corridor']])[:,1]
            diff=float(abs(actual-merged.raw_probability).max());assert diff<1e-12
            checks.append({'contract':directory.name,'config_id':cid,'year':year,'raw_probability_max_abs_diff':diff,'weights_hashes_pass':True,'head_maturity_checks_pass':all(receipt['checks'].values())})
    assert len(checks)==28
    cpu_frames=[]
    for folder,pattern in [(OUT/'budget900/output','*_predictions.csv.gz'),(OUT/'budget900/cpu300/output','*_predictions.csv.gz'),(OUT/'extended_contract/output','chronos2_small_ft_head10y_*_predictions.csv.gz')]:
        cpu_frames.extend(pd.read_csv(f,parse_dates=['date']) for f in sorted(folder.glob(pattern)))
    cpu_data=pd.concat(cpu_frames,ignore_index=True);cpu_data.to_csv(OUT/'budget900/cpu_matched_predictions.csv.gz',index=False)
    cpu_rows=[];cpu_intervals=[]
    for stage,years in [('development_2023_2025',[2023,2024,2025]),('inspected_retrospective_2026',[2026])]:
        for scope in ['all','KZT']:
            scoped=cpu_data[cpu_data.fold_test_year.isin(years)]
            if scope=='KZT':scoped=scoped[scoped.corridor.eq('KZT')]
            grouped={cid:f.sort_values(['date','corridor']).reset_index(drop=True) for cid,f in scoped.groupby('config_id')}
            for cid,f in grouped.items():cpu_rows.append({'stage':stage,'scope':scope,'config_id':cid,**score(f)})
            for cid,benchmark in [('chronos2_small_ft900_head10y','chronos2_small_ft_cpu300_head10y'),('chronos2_small_ft_cpu300_head10y','chronos2_small_ft_head10y')]:
                with threadpool_limits(limits=2):result=paired(grouped[cid],grouped[benchmark],10000)
                cpu_intervals.append({'stage':stage,'scope':scope,'config_id':cid,'benchmark':benchmark,**result})
    pd.DataFrame(cpu_rows).to_csv(OUT/'budget900/cpu_matched_summary.csv',index=False)
    pd.DataFrame(cpu_intervals).to_csv(OUT/'budget900/cpu_matched_paired_intervals.csv',index=False)
    # Compare forecast proper scores with immutable original300-step forecasts.
    original=pd.read_csv(OLD/'forecast_losses.csv.gz',parse_dates=['date'])
    original=original[original.config_id.isin(['chronos2_small_ft','random_walk_gaussian60'])]
    data=pd.concat([original,*losses],ignore_index=True)
    data.to_csv(OUT/'budget900/forecast_losses.csv.gz',index=False)
    rows=[];intervals=[]
    for stage,years in [('development_2023_2025',[2023,2024,2025]),('inspected_retrospective_2026',[2026])]:
        for scope in ['all','KZT']:
            scoped=data[data.fold_test_year.isin(years)]
            if scope=='KZT':scoped=scoped[scoped.corridor.eq('KZT')]
            grouped={cid:f.sort_values(['date','corridor']).reset_index(drop=True) for cid,f in scoped.groupby('config_id')}
            for cid,f in grouped.items():rows.append({'stage':stage,'scope':scope,'config_id':cid,'rows':len(f),**f[['pinball_bps','mae_bps','coverage80']].mean().to_dict()})
            a=grouped['chronos2_small_ft900']
            for benchmark in ['chronos2_small_ft','random_walk_gaussian60']:
                b=grouped[benchmark]
                assert a[['date','corridor']].equals(b[['date','corridor']])
                result={'stage':stage,'scope':scope,'config_id':'chronos2_small_ft900','benchmark':benchmark,'repetitions':10000}
                for metric in ['pinball_bps','mae_bps','coverage80']:
                    blocks=pd.DataFrame({'month':a.date.dt.to_period('M'),'delta':a[metric]-b[metric]}).groupby('month').delta.agg(['sum','count'])
                    weights=helpers.block_weights(list(blocks.index),10000)
                    distribution=(weights@blocks['sum'].to_numpy())/(weights@blocks['count'].to_numpy())
                    result.update({metric+'_delta':float((a[metric]-b[metric]).mean()),metric+'_delta_ci95_low':float(np.quantile(distribution,.025)),metric+'_delta_ci95_high':float(np.quantile(distribution,.975))})
                intervals.append(result)
    pd.DataFrame(rows).to_csv(OUT/'budget900/forecast_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'budget900/forecast_paired_intervals.csv',index=False)
    march_checks=[]
    for directory,pp in [(OUT/'march',heads_panel()),(OUT/'extended_contract/march',p)]:
        data=pd.read_csv(directory/'predictions.csv.gz',parse_dates=['date'])
        data=data[data.cutoff.eq('2026-03-01')]
        for path in sorted(directory.glob('*_checkpoint.joblib')):
            cid=path.name.removesuffix('_checkpoint.joblib');variant=cid.rsplit('_head',1)[0]
            frame,features=frame_for(variant,pp,wide)
            pack=joblib.load(path);pred=data[data.config_id.eq(cid)].sort_values(['date','corridor']).reset_index(drop=True)
            frame=frame.sort_values(['date','corridor']).reset_index(drop=True)
            test=frame[frame.target.notna()&frame.date.ge('2026-03-01')&frame.date.lt('2027-01-01')]
            history=frame[frame.date.ge('2025-03-01')&frame.date.lt('2026-03-01')]
            with threadpool_limits(limits=2):
                raw=pack['model'].predict_proba(test[features+['corridor']])[:,1]
                hp=helpers.core.apply_platt(pack['calibrator'],pack['model'].predict_proba(history[features+['corridor']])[:,1])
            prob=helpers.core.apply_platt(pack['calibrator'],raw)
            hi=helpers.core.select_per_corridor_with_cooldown(history,hp,pack['threshold'])
            state=helpers.core.corridor_selection_state(history,hi)
            chosen=helpers.core.select_per_corridor_with_cooldown(test,prob,pack['threshold'],state)
            actual=test.index.isin(chosen)
            diff=float(abs(prob-pred.probability).max());mismatch=int((actual!=pred.candidate_signal).sum())
            assert diff<1e-12 and mismatch==0
            march_checks.append({'contract':str(directory.relative_to(OUT)),'config_id':cid,'saved_probability_max_abs_diff':diff,'full_cooldown_replay_signal_mismatches':mismatch,'passed':True})
    assert len(march_checks)==15
    save(OUT/'additional_verification_receipt.json',{'head_checks':checks,'additional_head_count':28,'march_checks':march_checks,'march_saved_models_reconstructed':15,'actual900step_backbone_checkpoints':4,'full_parameter_update_and_precalibration_dates_pass':True,'code_sha256':sha(__file__)})
    print('All28additional saved annual heads,15March heads and4full900-step backbones verified.')


def heads_panel():
    from run_heads import panel
    return panel()

if __name__=='__main__':main()
