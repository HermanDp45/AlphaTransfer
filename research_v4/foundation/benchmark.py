#!/usr/bin/env python3
"""Local retrospective foundation-model benchmark; no network during execution.

Forecasts are 5 marginal quantile distributions, not a joint path distribution.
NOW adapters learn the original path-survival label on purged historical rows.
Full-weight fine tuning uses only public CBR levels before calibration year.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
from chronos import Chronos2Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from final_solution.training import core_experiment as core

H, CONTEXT, STEPS, SEED = 5, 256, 300, 20260905
BASE_FEATURES = core.CORE_FEATURES + core.VOL_FEATURES + ['moex_cny_close_minus_fixing_same_session']
HF_CACHE = Path('/private/tmp/alphatransfer-hf/hub')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n')


def load_panel():
    # Frozen V3 snapshot is read-only. Only listed BASE_FEATURES enter any fit.
    p = pd.read_pickle(ROOT / 'research_v3/models/panel_v2.pkl')
    p = p[['date', 'corridor', 'rub_per_unit', 'session_ordinal', *dict.fromkeys(BASE_FEATURES)]].copy()
    return core.add_target(p, H)


def load_history():
    r = pd.read_csv(ROOT / 'research_v3/models/data/cbr_extended.csv', parse_dates=['date'])
    w = r.pivot(index='date', columns='corridor', values='rub_per_unit').sort_index()
    # A single 2010-01-01 KZT-only row precedes all tested/trained windows.
    w = w.dropna()
    assert np.isfinite(w).all().all() and (w > 0).all().all()
    return w, np.log(w.to_numpy(dtype='float64')).T.astype('float32')


def forecast(pipeline, wide, logs, dates, cache):
    """One group contains five currencies at exactly the same information cutoff."""
    dates = pd.DatetimeIndex(sorted(dates))
    indexes = wide.index.get_indexer(dates)
    assert np.all(indexes >= CONTEXT - 1)
    if cache.exists():
        item = np.load(cache)
        assert np.array_equal(item['dates'], dates.values.astype('datetime64[D]'))
        return item['quantiles'], item['grid']
    start = time.perf_counter()
    inputs = [logs[:, i-CONTEXT+1:i+1].copy() for i in indexes]
    # cross_learning=True would mix different dates and leak later histories!
    result = pipeline.predict(inputs, prediction_length=H, context_length=CONTEXT,
                              batch_size=40, cross_learning=False)
    q = np.stack([x.numpy() for x in result])  # date x currency x quantile x horizon
    grid = np.asarray(pipeline.quantiles)
    assert q.shape == (len(dates), len(wide.columns), len(grid), H)
    assert np.isfinite(q).all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, dates=dates.values.astype('datetime64[D]'), quantiles=q, grid=grid)
    save_json(cache.with_suffix('.json'), {'rows':len(dates)*len(wide.columns),
        'seconds':time.perf_counter()-start, 'context':CONTEXT, 'cross_learning':False,
        'task_group':'five currencies at one identical cutoff', 'grid':grid.tolist(),
        'cache_sha256':sha(cache)})
    return q, grid


def forecast_features(panel, wide, dates, quantiles, grid):
    """CDF inversion gives marginal summaries and Frechet bounds, never joint p."""
    dates = pd.DatetimeIndex(sorted(dates))
    q = np.sort(quantiles, axis=2)  # Monotone rearrangement, defined before labels.
    current = np.log(wide.loc[dates].to_numpy())
    relative = (q - current[:, :, None, None]) * 10000
    flat = relative.reshape(-1, len(grid), H)
    surv = np.empty((len(flat), H))
    for i in range(len(flat)):
        for j in range(H):
            surv[i,j] = 1-np.interp(0., flat[i,:,j], grid, left=0., right=1.)
    med = flat[:,np.argmin(abs(grid-.5)),:]
    width = flat[:,np.argmin(abs(grid-.9)),:]-flat[:,np.argmin(abs(grid-.1)),:]
    rows = pd.MultiIndex.from_product([dates, wide.columns], names=['date','corridor']).to_frame(index=False)
    for j in range(H):
        rows[f'fm_marginal_survival_{j+1}'] = surv[:,j]
        rows[f'fm_median_bps_{j+1}'] = med[:,j]
    rows['fm_frechet_lower'] = np.maximum(0., surv.sum(axis=1)-(H-1))
    rows['fm_frechet_upper'] = surv.min(axis=1)
    rows['fm_width80_mean_bps'] = width.mean(axis=1)
    rows['fm_width80_h5_bps'] = width[:,-1]
    features = [c for c in rows if c.startswith('fm_')]
    out = panel.merge(rows, on=['date','corridor'], how='left', validate='one_to_one')
    out.index = panel.index
    return out, features


def forecast_metrics(wide, logs, dates, q, grid, year, name, test_keys):
    dates = pd.DatetimeIndex(sorted(dates))
    test_dates = dates[dates.year == year]
    selection = np.flatnonzero(dates.year == year)
    indexes = wide.index.get_indexer(test_dates)
    valid = indexes + H < len(wide)
    selection, indexes = selection[valid], indexes[valid]
    test_dates = test_dates[valid]
    current = logs[:,indexes].T
    realized = np.stack([logs[:,indexes+h].T for h in range(1,H+1)], axis=-1)
    q = np.sort(q[selection], axis=2)
    err = (realized[:,:,None,:]-q)*10000
    pin = np.maximum(grid[None,None,:,None]*err, (grid[None,None,:,None]-1)*err).mean(axis=(2,3))
    med = q[:,:,np.argmin(abs(grid-.5)),:]
    mae = np.abs(realized-med).mean(axis=-1)*10000
    cover = ((realized >= q[:,:,np.argmin(abs(grid-.1)),:]) &
             (realized <= q[:,:,np.argmin(abs(grid-.9)),:])).mean(axis=-1)
    rows = pd.MultiIndex.from_product([test_dates, wide.columns], names=['date','corridor']).to_frame(index=False)
    rows['pinball_bps'] = pin.ravel()
    rows['mae_bps'] = mae.ravel()
    rows['coverage80'] = cover.ravel()
    rows['config_id'] = name
    rows['fold_test_year'] = year
    return rows.merge(test_keys, on=['date','corridor'], validate='one_to_one')


def run_adapter(p, features, year, name, kind, output):
    group = 'foundation_current'
    core.FEATURE_GROUPS[group] = features
    original = core.make_model
    model_kind = 'hist_gradient_boosting'
    if kind == 'logit':
        def factory(ignored, fs):
            model = original('logistic', fs)
            model.set_params(classifier=LogisticRegression(C=.1, max_iter=2000, random_state=SEED))
            return model
        core.make_model = factory
    fitted = []
    current = core.make_model
    def retain(kind_, fs):
        model = current(kind_, fs)
        fitted.append(model)
        return model
    core.make_model = retain
    try:
        spec = core.Experiment(name, model_kind, group)
        cells, predictions, _ = core.run_fold(spec, H, year, p)
    finally:
        core.make_model = original
    predictions['config_id'] = name
    joblib.dump(fitted[0], output / f'{name}_{year}_event_adapter.joblib')
    pd.DataFrame(cells).to_csv(output / f'{name}_{year}_cells.csv', index=False)
    predictions.to_csv(output / f'{name}_{year}_predictions.csv.gz', index=False)
    return predictions


def fit(pipeline, wide, logs, year, name, steps, kzt_only=False):
    start_date, end_date = pd.Timestamp(year-11,1,1), pd.Timestamp(year-1,1,1)
    mask = (wide.index >= start_date) & (wide.index < end_date)
    path = OUT / 'checkpoints' / name / str(year)
    ckpt = path / 'finetuned-ckpt'
    if ckpt.exists():
        return Chronos2Pipeline.from_pretrained(str(ckpt), device_map='cpu')
    start = time.perf_counter()
    train_values=logs[:,mask].copy()
    if kzt_only:
        train_values=train_values[[list(wide.columns).index('KZT')],:]
    lr=2e-6 if kzt_only else 1e-5
    tuned = pipeline.fit(inputs=[train_values], prediction_length=H,
        finetune_mode='full', context_length=CONTEXT, learning_rate=lr,
        num_steps=steps, batch_size=20, output_dir=path, min_past=64,
        optim='adamw_torch', disable_tqdm=True, logging_steps=50, seed=SEED+year,
        data_seed=SEED+year, report_to='none')
    delta2, initial2, changed, total = 0., 0., 0, 0
    initial = dict(pipeline.model.named_parameters())
    for key, value in tuned.model.named_parameters():
        orig = initial[key]
        d = (value.detach()-orig.detach()).double()
        delta2 += float((d*d).sum())
        initial2 += float((orig.detach().double()**2).sum())
        changed += int(torch.count_nonzero(d))
        total += value.numel()
    save_json(path / 'fit_receipt.json', {
        'test_year':year, 'backbone_train_start':wide.index[mask].min(),
        'backbone_last_observation':wide.index[mask].max(), 'calibration_start':end_date,
        'training_cbr_dates':int(mask.sum()), 'training_currencies':['KZT'] if kzt_only else list(wide.columns),
        'max_train_context_label_end':'strictly before calibration year',
        'mode':'full', 'steps':steps, 'batch_size':20, 'context':CONTEXT,
        'learning_rate':lr, 'parameters':total, 'parameters_changed':changed,
        'parameter_delta_l2':delta2**.5, 'relative_delta_l2':(delta2/initial2)**.5,
        'fit_seconds':time.perf_counter()-start, 'seed':SEED+year,
        'weights_sha256':sha(ckpt/'model.safetensors'),
        'raw_public_cbr_sha256':sha(ROOT/'research_v3/models/data/cbr_extended.csv')})
    assert changed > total*.5, 'Full fine tuning did not update expected weights'
    return tuned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['small','synth'])
    parser.add_argument('--years', nargs='+', type=int, default=[2023,2024,2025,2026])
    parser.add_argument('--steps', type=int, default=STEPS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    p, (wide, logs) = load_panel(), load_history()
    output = OUT/'output'
    output.mkdir(exist_ok=True)
    provenance = {'python':platform.python_version(), 'device':'cpu', 'threads':2,
        'packages':{k:importlib.metadata.version(k) for k in ['torch','chronos-forecasting','transformers','numpy','pandas','scikit-learn']},
        'code_sha256':sha(__file__), 'panel_sha256':sha(ROOT/'research_v3/models/panel_v2.pkl'),
        'raw_cbr_sha256':sha(ROOT/'research_v3/models/data/cbr_extended.csv'),
        'horizon':H, 'context':CONTEXT, 'full_finetune_steps':args.steps,
        'event_adapter':'2-year training; preceding full year calibration and policy; purge5',
        'years':args.years, 'status':'retrospective development2023-25; inspected diagnostic2026'}
    save_json(OUT/'protocol.json',provenance)
    for model in args.models:
        name = f'chronos2_{model}'
        meta = json.loads((OUT/f'chronos-2-{model}_hub_metadata.json').read_text())
        checkpoint = HF_CACHE/f'models--autogluon--chronos-2-{model}'/'snapshots'/meta['sha']
        pipeline = Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cpu')
        # A fixed backbone can safely reuse causal forecasts across outer years.
        dates = p.date.unique()
        q, grid = forecast(pipeline,wide,logs,dates,OUT/'forecasts'/f'{name}_zs.npz')
        zero, features = forecast_features(p,wide,dates,q,grid)
        for year in args.years:
            _,_,test = core.split_for_year(p,H,year)
            keys = test[['date','corridor']]
            for mode, frame, qq, dd in [('zs',zero,q,dates)]:
                cid=f'{name}_{mode}'
                run_adapter(frame,features,year,cid,'logit',output)
                run_adapter(frame,BASE_FEATURES+features,year,cid+'_hgb_aug','hgb',output)
                forecast_metrics(wide,logs,dd,qq,grid,year,cid,keys).to_csv(output/f'{cid}_{year}_forecast_losses.csv',index=False)
            tuned=fit(pipeline,wide,logs,year,name,args.steps)
            dates_ft=p.loc[p.date >= pd.Timestamp(year-3,1,1),'date'].unique()
            dates_ft=dates_ft[dates_ft < np.datetime64(f'{year+1}-01-01')]
            tq,tgrid=forecast(tuned,wide,logs,dates_ft,OUT/'forecasts'/f'{name}_ft_{year}.npz')
            frame,features=forecast_features(p,wide,dates_ft,tq,tgrid)
            cid=f'{name}_ft'
            run_adapter(frame,features,year,cid,'logit',output)
            run_adapter(frame,BASE_FEATURES+features,year,cid+'_hgb_aug','hgb',output)
            forecast_metrics(wide,logs,dates_ft,tq,tgrid,year,cid,keys).to_csv(output/f'{cid}_{year}_forecast_losses.csv',index=False)
            if model == 'synth':
                specialized=fit(tuned,wide,logs,year,name+'_kzt',100,kzt_only=True)
                kq,kgrid=forecast(specialized,wide,logs,dates_ft,OUT/'forecasts'/f'{name}_ft_kzt_{year}.npz')
                kp,kfeatures=forecast_features(p,wide,dates_ft,kq,kgrid)
                kcid=f'{name}_ft_kzt'
                run_adapter(kp,kfeatures,year,kcid,'logit',output)
                run_adapter(kp,BASE_FEATURES+kfeatures,year,kcid+'_hgb_aug','hgb',output)
                forecast_metrics(wide,logs,dates_ft,kq,kgrid,year,kcid,keys).to_csv(output/f'{kcid}_{year}_forecast_losses.csv',index=False)
                del specialized
            print(json.dumps({'model':name,'year':year,'complete':True}),flush=True)
            del tuned
            gc.collect()
        del pipeline
        gc.collect()


if __name__ == '__main__':
    main()
