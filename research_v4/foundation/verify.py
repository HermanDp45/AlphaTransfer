#!/usr/bin/env python3
"""Meaningful contract checks for actual saved model artifacts and forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from benchmark import OUT,ROOT,CONTEXT,H,Chronos2Pipeline,load_history,save_json,sha

base=pd.read_csv(ROOT/'research_v3/models/baseline_reproduction_h5_predictions.csv.gz',parse_dates=['date'])
ordinals=pd.read_pickle(ROOT/'research_v3/models/panel_v2.pkl')[['date','corridor','session_ordinal']]
checks=[]
for file in sorted((OUT/'output').glob('*_predictions.csv.gz')):
    p=pd.read_csv(file,parse_dates=['date']).sort_values(['date','corridor']).reset_index(drop=True)
    year=int(p.fold_test_year.iloc[0])
    expected=base[base.fold_test_year.eq(year)].sort_values(['date','corridor']).reset_index(drop=True)
    assert p[['date','corridor']].equals(expected[['date','corridor']]),str(file)
    for field in ['target','forward_bps','symmetric_bps','regret_bps']:
        assert np.allclose(p[field],expected[field],equal_nan=False),(file,field)
    assert p.probability.between(0,1).all()
    if 'session_ordinal' not in p:
        p=p.merge(ordinals,on=['date','corridor'],validate='one_to_one')
    for _,g in p[p.candidate_signal.astype(bool)].groupby('corridor'):
        assert (g.session_ordinal.diff().dropna()>3).all()
    checks.append({'file':file.name,'rows':len(p),'year':year,'sha256':sha(file)})
assert len(checks)==44, len(checks)
receipts=[]
for file in sorted((OUT/'checkpoints').glob('*/*/fit_receipt.json')):
    r=json.loads(file.read_text())
    assert pd.Timestamp(r['backbone_last_observation'])<pd.Timestamp(r['calibration_start'])
    assert r['parameters_changed']>r['parameters']*.5
    assert sha(file.parent/'finetuned-ckpt/model.safetensors')==r['weights_sha256']
    receipts.append({'checkpoint':str(file.parent.relative_to(OUT)),'changed_fraction':r['parameters_changed']/r['parameters']})
assert len(receipts)==12
wide,logs=load_history()
torch.set_num_threads(2)
meta=json.loads((OUT/'chronos-2-small_hub_metadata.json').read_text())
path=Path('/private/tmp/alphatransfer-hf/hub/models--autogluon--chronos-2-small/snapshots')/meta['sha']
p=Chronos2Pipeline.from_pretrained(str(path),device_map='cpu')
i=wide.index.get_loc('2023-02-01')
a=logs[:,i-CONTEXT+1:i+1].copy()
b=logs[:,i-CONTEXT+2:i+2].copy()
def predict(inputs):
    return p.predict(inputs,prediction_length=H,context_length=CONTEXT,cross_learning=False)[0].numpy()
one=predict([a]);paired=predict([a,b]);modified=predict([a,b+100])
diff=float(np.max(abs(one-paired)));changed=float(np.max(abs(one-modified)))
assert diff<1e-5 and changed<1e-5
save_json(OUT/'verification_receipt.json',{'prediction_contracts':checks,'training_receipts':receipts,
    'single_vs_batch_max_abs_lograte':diff,'other_group_perturbation_max_abs_lograte':changed,
    'cutoff':'2023-02-01','status':'passed','code_sha256':sha(__file__)})
print('44 prediction contracts, 12 real checkpoints and batch causality checks passed')
