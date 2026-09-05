#!/usr/bin/env python3
"""Material verification of cache reuse, head reconstruction and information cutoffs."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from threadpoolctl import threadpool_limits
from run_heads import OUT, ROOT, OLD, BASE, VARIANTS, H, CONTEXT, sha, save, panel, checkpoint, old_cache, predict, legacy, core


def main():
    torch.set_num_threads(2)
    protocol=json.loads((OUT/'protocol.json').read_text())
    sources={name:sha(ROOT/name)==expected for name,expected in protocol['input_sha256'].items()}
    assert all(sources.values())
    p=panel();wide,logs=legacy.load_history()
    receipts=[];cache_checks=[];head_checks=[]
    for variant in VARIANTS:
        for year in [2023,2024,2025,2026]:
            if variant=='base_control':frame,features=p,BASE
            else:
                cache=OUT/'forecasts'/f'{variant}{"" if variant.endswith("_zs") else "_"+str(year)}.npz'
                item=np.load(cache);dates=pd.DatetimeIndex(item['dates'])
                before=np.load(old_cache(variant,year))
                indices=pd.DatetimeIndex(before['dates']).get_indexer(dates)
                overlap=indices>=0
                assert np.array_equal(item['quantiles'][overlap],before['quantiles'][indices[overlap]])
                receipt=json.loads(cache.with_suffix('.json').read_text())
                assert receipt['cache_sha256']==sha(cache)
                assert receipt['weights_sha256']==sha(checkpoint(variant,year)/'model.safetensors')
                if not variant.endswith('_zs'):
                    fit=receipt['source_backbone_receipt']
                    assert pd.Timestamp(fit['backbone_last_observation'])<pd.Timestamp(year-1,1,1)
                    assert fit['weights_sha256']==receipt['weights_sha256']
                cache_checks.append({'variant':variant,'year':year,'overlap_date_groups':int(overlap.sum()),'prior_quantiles_bitwise_preserved':True,'cache_sha256_match':True,'backbone_weights_sha256_match':True,'backbone_before_calibration':True if not variant.endswith('_zs') else 'synthetic zero-shot; publisher-only-synthetic claim'})
                frame,extra=legacy.forecast_features(p,wide,dates,item['quantiles'],item['grid'])
                features=BASE+extra
            for years in [2,10]:
                cid=f'{variant}_head{years}y'
                file=OUT/'output'/f'{cid}_{year}_predictions.csv.gz'
                pred=pd.read_csv(file,parse_dates=['date'])
                receipt=json.loads(file.with_name(f'{cid}_{year}_receipt.json').read_text())
                modelpath=file.with_name(f'{cid}_{year}_head.joblib')
                assert sha(modelpath)==receipt['head_sha256']
                model=joblib.load(modelpath)
                merged=frame.merge(pred[['date','corridor','raw_probability','probability']],on=['date','corridor'],validate='one_to_one')
                with threadpool_limits(limits=2):
                    actual=model.predict_proba(merged[features+['corridor']])[:,1]
                maxdiff=float(abs(actual-merged.raw_probability).max())
                assert maxdiff<1e-12
                sig=pred[pred.candidate_signal]
                assert all((g.session_ordinal.diff().dropna()>core.COOLDOWN_SESSIONS).all() for _,g in sig.groupby('corridor'))
                head_checks.append({'config_id':cid,'year':year,'saved_head_raw_probability_max_abs_diff':maxdiff,'within_test_cooldown_pass':True,'saved_head_sha256_match':True})
                receipts.append(receipt)
    # Same-date grouping must prevent later task contexts influencing this date.
    # Test the real package inference path; fixed weights may still have seen the
    # early training period, a separate issue explicitly disclosed in protocol.
    pipe=Chronos2Pipeline.from_pretrained(str(checkpoint('chronos2_small_ft',2026)),device_map='cpu')
    i=wide.index.searchsorted(pd.Timestamp('2018-07-01'))
    single=predict(pipe,logs,[i])[0]
    mixed=predict(pipe,logs,[len(wide)-1,i,i-300])[1]
    grouped_diff=float(abs(single-mixed).max())
    assert grouped_diff<5e-6
    altered=logs.copy();altered[:,i+1:]+=10
    perturbed=predict(pipe,altered,[i])[0]
    future_diff=float(abs(single-perturbed).max())
    assert future_diff==0
    # Recompute literal target semantics independently on 100 spread-out rows.
    sample=p[p.target.notna()&p.date.ge('2023-01-01')].iloc[np.linspace(0,len(p[p.target.notna()&p.date.ge('2023-01-01')])-1,100).astype(int)]
    for row in sample.itertuples():
        series=p[p.corridor.eq(row.corridor)].set_index('date').rub_per_unit
        j=series.index.get_loc(row.date)
        assert int(series.iloc[j]<=series.iloc[j+1:j+H+1].min())==int(row.target)
    report={'source_inputs_unchanged':sources,'cache_checks':cache_checks,'head_checks':head_checks,'group_isolation':{'actual_single_vs_mixed_past_future_context_quantile_max_abs_diff':grouped_diff,'tolerance':5e-6,'cross_learning':False,'passed':True},'future_perturbation':{'actual_future_log_levels_add10_prediction_max_abs_diff':future_diff,'passed':True,'scope':'input causality with fixed weights; not cross-fitting of historical neural weights'},'independent_target_recomputations':100,'all_head_split_label_maturity_checks_pass':all(all(r['checks'].values()) for r in receipts),'head_count':len(head_checks),'code_sha256':sha(__file__),'protocol_sha256':sha(OUT/'protocol.json'),'reproduction':json.loads((OUT/'reproduction_checks.json').read_text()),'march_jan_replay':json.loads((OUT/'march/jan_replay_parity.json').read_text()) if (OUT/'march/jan_replay_parity.json').exists() else None}
    assert all(x['passed'] for x in report['reproduction'])
    save(OUT/'verification_receipt.json',report)
    print(json.dumps({'sources_unchanged':True,'head_count':len(head_checks),'group_isolation_max_abs_diff':grouped_diff,'future_perturbation_max_abs_diff':future_diff,'all_pass':True},indent=2))


if __name__=='__main__':main()
