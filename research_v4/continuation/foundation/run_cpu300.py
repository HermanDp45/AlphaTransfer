#!/usr/bin/env python3
"""CPU300 nuisance control isolates training budget from tiny MPS rounding."""
import datetime
import gc
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
import run_heads as heads
from run_extended import panel_exact

def main():
    root=heads.OUT;out=root/'budget900/cpu300';out.mkdir(exist_ok=True)
    protocol=out/'protocol.json'
    if not protocol.exists():heads.save(protocol,{'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Numerical nuisance control, not model selection: reforecast existing immutable300-step Small weights entirely onCPU, because900 usesCPU while primary extended300 cache combines oldCPU and earlyMPS. Same exact V3extended panel10y head; four fixed annual folds.','selection':'none; all four folds; report both original300 and CPU300 comparisons','no_neural_refit':True,'parent_protocol_sha256':heads.sha(root/'budget900/protocol.json')})
    heads.OUT=out;torch.set_num_threads(2)
    p=panel_exact();wide,logs=heads.legacy.load_history()
    for year in [2023,2024,2025,2026]:
        ckpt=heads.checkpoint('chronos2_small_ft',year)
        pipe=Chronos2Pipeline.from_pretrained(str(ckpt),device_map='cpu')
        dates=wide.index[(wide.index>=pd.Timestamp(year-11,1,1))&(wide.index<pd.Timestamp(year+1,1,1))]
        path=out/'forecasts'/f'chronos2_small_ft_cpu300_{year}.npz'
        q,grid=heads.legacy.forecast(pipe,wide,logs,dates,path)
        frame,extra=heads.legacy.forecast_features(p,wide,dates,q,grid)
        heads.run_head(frame,heads.BASE+extra,year,'chronos2_small_ft_cpu300',10)
        del pipe;gc.collect()
    heads.save(out/'receipt.json',{'heads':4,'new_neural_fits':0,'source_weights':'immutable original300-step annual Small checkpoints','all_forecasts_device':'cpu','panel_contract':'exact V3extended','code_sha256':heads.sha(__file__),'protocol_sha256':heads.sha(protocol)})

if __name__=='__main__':main()
