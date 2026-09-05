#!/usr/bin/env python3
"""Prespecified compact 900-versus300 full-weight training-budget control."""
import gc
import json
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
import run_heads as heads
from run_extended import panel_exact,assess

def main():
    here=heads.OUT;out=here/'budget900';heads.OUT=out;heads.legacy.OUT=out
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    meta=json.loads((heads.OLD/'chronos-2-small_hub_metadata.json').read_text())
    base=heads.legacy.HF_CACHE/'models--autogluon--chronos-2-small'/'snapshots'/meta['sha']
    p=panel_exact();wide,logs=heads.legacy.load_history()
    pipeline=Chronos2Pipeline.from_pretrained(str(base),device_map='cpu')
    for year in [2023,2024,2025,2026]:
        tuned=heads.legacy.fit(pipeline,wide,logs,year,'chronos2_small',900)
        dates=wide.index[(wide.index>=pd.Timestamp(year-11,1,1))&(wide.index<pd.Timestamp(year+1,1,1))]
        cache=out/'forecasts'/f'chronos2_small_ft900_{year}.npz'
        q,grid=heads.legacy.forecast(tuned,wide,logs,dates,cache)
        receipt=json.loads(cache.with_suffix('.json').read_text())
        receipt.update(weights_sha256=heads.sha(out/'checkpoints/chronos2_small'/str(year)/'finetuned-ckpt/model.safetensors'),code_sha256=heads.sha(__file__),raw_cbr_sha256=heads.sha(heads.ROOT/'research_v3/models/data/cbr_extended.csv'),date_first=str(dates.min()),date_last=str(dates.max()))
        heads.save(cache.with_suffix('.json'),receipt)
        frame,extra=heads.legacy.forecast_features(p,wide,dates,q,grid)
        heads.run_head(frame,heads.BASE+extra,year,'chronos2_small_ft900',10)
        del tuned;gc.collect()
    assess(out,here/'extended_contract')
    heads.save(out/'receipt.json',{'backbone_fits':4,'full_steps_per_fit':900,'total_full_steps':3600,'device':'cpu','head_years':10,'panel_contract':'exact V3 panel_extended','code_sha256':heads.sha(__file__),'protocol_sha256':heads.sha(out/'protocol.json'),'base_weights_sha256':heads.sha(base/'model.safetensors'),'compared_with':'same-panel chronos2_small_ft_head10y with original300step checkpoints','test_selection':'none'})

if __name__=='__main__':main()
