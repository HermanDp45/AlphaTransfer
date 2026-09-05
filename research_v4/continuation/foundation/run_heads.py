#!/usr/bin/env python3
"""Matched 2/10-year NOW heads on immutable Chronos backbones and caches."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HOME', '/private/tmp/alphatransfer-hf')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import joblib
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT/'research_v4/foundation'
spec = importlib.util.spec_from_file_location('benchmark', OLD/'benchmark.py')
legacy = importlib.util.module_from_spec(spec)
sys.modules['benchmark'] = legacy
spec.loader.exec_module(legacy)
core, H, CONTEXT = legacy.core, legacy.H, legacy.CONTEXT
BASE = legacy.BASE_FEATURES
VARIANTS = ['base_control','chronos2_small_ft','chronos2_synth_ft','chronos2_synth_ft_kzt','chronos2_synth_zs']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')


def panel():
    # Append earlier causal history, preserving ALL original shared feature values.
    p=pd.read_pickle(ROOT/'research_v3/models/panel_extended.pkl')
    old=pd.read_pickle(ROOT/'research_v3/models/panel_v2.pkl')
    cols=['date','corridor','rub_per_unit','session_ordinal',*dict.fromkeys(BASE)]
    p=p[cols].sort_values(['date','corridor']).reset_index(drop=True)
    indexed=p.set_index(['date','corridor'])
    old=old.set_index(['date','corridor'])
    indexed.loc[old.index,BASE]=old[BASE].to_numpy()
    p=indexed.reset_index()
    assert np.allclose(p.set_index(['date','corridor']).loc[old.index,BASE],old[BASE],equal_nan=True)
    return core.add_target(p,H)


def checkpoint(variant, year):
    if variant.endswith('_zs'):
        meta=json.loads((OLD/'chronos-2-synth_hub_metadata.json').read_text())
        return legacy.HF_CACHE/'models--autogluon--chronos-2-synth'/'snapshots'/meta['sha']
    name='chronos2_synth_kzt' if variant.endswith('_kzt') else variant.removesuffix('_ft')
    return OLD/'checkpoints'/name/str(year)/'finetuned-ckpt'


def old_cache(variant,year):
    return OLD/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_"+str(year)}.npz'


def predict(pipe,logs,indexes):
    inputs=[logs[:,i-CONTEXT+1:i+1].copy() for i in indexes]
    with torch.inference_mode():
        result=pipe.predict(inputs,prediction_length=H,context_length=CONTEXT,batch_size=40,cross_learning=False)
    return np.stack([x.detach().cpu().numpy() for x in result])


def smoke():
    wide,logs=legacy.load_history()
    ii=np.linspace(wide.index.searchsorted(pd.Timestamp('2012-01-10')),len(wide)-1,32).astype(int)
    receipt={'torch':torch.__version__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'groups':len(ii),'currencies_per_group':5,'context':CONTEXT,'batch_size':40,'models':[]}
    for variant in ['chronos2_small_ft','chronos2_synth_ft']:
        ckpt=checkpoint(variant,2026); entry={'variant':variant,'weights_sha256':sha(ckpt/'model.safetensors')}
        results={}
        for device in ['cpu','mps']:
            if device=='mps' and not torch.backends.mps.is_available():continue
            pipe=Chronos2Pipeline.from_pretrained(str(ckpt),device_map=device)
            predict(pipe,logs,ii[:2])
            t=time.perf_counter();results[device]=predict(pipe,logs,ii)
            entry[device+'_seconds']=time.perf_counter()-t
            del pipe;gc.collect()
            if device=='mps':torch.mps.empty_cache()
        if 'mps' in results:
            diff=abs(results['mps']-results['cpu'])
            entry.update(max_abs_log_quantile_diff=float(diff.max()),mean_abs_log_quantile_diff=float(diff.mean()),max_abs_logbps_diff=float(diff.max()*10000),parity_tolerance_log_levels=5e-5,parity_pass=bool(diff.max()<5e-5))
        receipt['models'].append(entry)
        print(json.dumps(entry),flush=True)
    save(OUT/'device_smoke.json',receipt)


def expanded_forecast(variant,year,wide,logs,device):
    # The unchanged zero-shot backbone permits one cache shared by outer folds.
    start=pd.Timestamp(2012 if variant.endswith('_zs') else year-11,1,1)
    end=pd.Timestamp(2027 if variant.endswith('_zs') else year+1,1,1)
    dates=wide.index[(wide.index>=start)&(wide.index<end)]
    cache=OUT/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_"+str(year)}.npz'
    receipt=cache.with_suffix('.json')
    ckpt=checkpoint(variant,year); previous=old_cache(variant,year)
    signature={'weights_sha256':sha(ckpt/'model.safetensors'),'old_cache_sha256':sha(previous),'cbr_sha256':sha(ROOT/'research_v3/models/data/cbr_extended.csv'),'context':CONTEXT,'cross_learning':False,'date_start':str(dates.min()),'date_end':str(dates.max()),'dates':len(dates),'code_sha256':sha(__file__)}
    if cache.exists():
        saved=json.loads(receipt.read_text())
        assert all(saved[k]==v for k,v in signature.items()),'Cache identity mismatch'
        item=np.load(cache);assert np.array_equal(item['dates'],dates.values.astype('datetime64[D]'))
        return dates,item['quantiles'],item['grid']
    item=np.load(previous)
    previous_dates=pd.DatetimeIndex(item['dates'])
    grid=item['grid']; oldq=item['quantiles']
    q=np.empty((len(dates),5,len(grid),H),dtype=oldq.dtype)
    old_indexes=previous_dates.get_indexer(dates)
    mask=old_indexes>=0
    q[mask]=oldq[old_indexes[mask]]
    missing=np.flatnonzero(~mask)
    indexes=wide.index.get_indexer(dates[missing]);assert np.all(indexes>=CONTEXT-1)
    pipe=Chronos2Pipeline.from_pretrained(str(ckpt),device_map=device)
    assert np.array_equal(np.asarray(pipe.quantiles),grid)
    t=time.perf_counter()
    for offset in range(0,len(missing),128):
        ix=missing[offset:offset+128]
        q[ix]=predict(pipe,logs,indexes[offset:offset+128])
        print(f'forecast {variant} {year}: {min(offset+128,len(missing))}/{len(missing)} new groups; {time.perf_counter()-t:.1f}s',flush=True)
    del pipe;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    assert np.isfinite(q).all()
    assert np.array_equal(q[mask],oldq[old_indexes[mask]])
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache,dates=dates.values.astype('datetime64[D]'),quantiles=q,grid=grid)
    save(receipt,{**signature,'checkpoint':str(ckpt),'old_cache':str(previous),'forecast_device_for_new_rows':device,'old_rows_device':'cpu; reused bitwise unchanged','reused_date_groups':int(mask.sum()),'new_date_groups':int((~mask).sum()),'seconds':time.perf_counter()-t,'cache_sha256':sha(cache),'reused_quantiles_bitwise_equal':True,'max_context_date_equals_prediction_date':True,'source_backbone_receipt':None if variant.endswith('_zs') else json.loads((ckpt.parent/'fit_receipt.json').read_text())})
    return dates,q,grid


def run_head(p,features,year,variant,years):
    name=f'{variant}_head{years}y'
    output=OUT/'output';output.mkdir(exist_ok=True)
    receipt=output/f'{name}_{year}_receipt.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['code_sha256']==sha(__file__)
        print(f'reuse head {name} {year}',flush=True);return
    group='continuation_foundation'
    core.FEATURE_GROUPS[group]=features
    old_factory,old_years=core.make_model,core.TRAIN_WINDOW_YEARS
    fitted=[]
    def factory(kind,fs):
        model=old_factory(kind,fs)
        model.named_steps['classifier'].set_params(early_stopping=False)
        fitted.append(model);return model
    core.make_model=factory;core.TRAIN_WINDOW_YEARS=years
    t=time.perf_counter()
    try:
        train,cal,test=core.split_for_year(p,H,year)
        assert train.date.max()<cal.date.min()<test.date.min()
        # Explicitly verify target maturity before each next split boundary.
        next_date=p.groupby('corridor').date.shift(-H)
        assert (next_date.loc[train.index]<pd.Timestamp(year-1,1,1)).all()
        assert (next_date.loc[cal.index]<pd.Timestamp(year,1,1)).all()
        if any(x.startswith('fm_') for x in features):
            assert p.loc[train.index.union(cal.index).union(test.index),[f for f in features if f.startswith('fm_')]].notna().all().all()
        with threadpool_limits(limits=2):
            cells,pred,_=core.run_fold(core.Experiment(name,'hist_gradient_boosting',group),H,year,p)
    finally:
        core.make_model=old_factory;core.TRAIN_WINDOW_YEARS=old_years
    pred['session_ordinal']=p.loc[pred.index,'session_ordinal']
    pred['head_years']=years
    pred['variant']=variant
    pred.to_csv(output/f'{name}_{year}_predictions.csv.gz',index=False)
    pd.DataFrame(cells).to_csv(output/f'{name}_{year}_cells.csv',index=False)
    checkpoint_path=output/f'{name}_{year}_head.joblib'
    joblib.dump(fitted[0],checkpoint_path)
    save(receipt,{'variant':variant,'head_years':years,'fold_test_year':year,'train_first':train.date.min(),'train_last':train.date.max(),'train_rows':len(train),'cal_first':cal.date.min(),'cal_last':cal.date.max(),'cal_rows':len(cal),'test_first':test.date.min(),'test_last':test.date.max(),'test_rows':len(test),'head_parameters':fitted[0].named_steps['classifier'].get_params(),'features':features,'seconds':time.perf_counter()-t,'head_sha256':sha(checkpoint_path),'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'checks':{'train_labels_mature_before_calibration':True,'calibration_labels_mature_before_test':True,'all_forecast_features_present':True,'early_stopping_disabled_both_windows':True},'backbone_training_feature_status':'Earlier head features are in-sample with respect to frozen neural fit, not cross-fitted.'})
    print(f'head {name} {year}: {len(train)} train rows; {time.perf_counter()-t:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--device',choices=['cpu','mps'],default='mps');parser.add_argument('--variants',nargs='+',default=VARIANTS);parser.add_argument('--years',nargs='+',type=int,default=[2023,2024,2025,2026]);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    assert (OUT/'protocol.json').exists()
    if args.smoke:smoke();return
    if args.device=='mps':
        receipt=json.loads((OUT/'device_smoke.json').read_text())
        assert torch.backends.mps.is_available() and all(x['parity_pass'] for x in receipt['models'])
    p=panel();wide,logs=legacy.load_history()
    for variant in args.variants:
        for year in args.years:
            if variant=='base_control':frame,features=p,BASE
            else:
                dates,q,grid=expanded_forecast(variant,year,wide,logs,args.device)
                frame,extra=legacy.forecast_features(p,wide,dates,q,grid)
                features=BASE+extra
            for years in [2,10]:run_head(frame,features,year,variant,years)
    print('All requested matched heads complete.',flush=True)


if __name__=='__main__':main()
