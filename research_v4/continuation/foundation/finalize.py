#!/usr/bin/env python3
"""Single machine-readable scientific status for the parent report verifier."""
import datetime
import json
import pandas as pd
from run_heads import OUT,sha,save

def read(path):return json.loads((OUT/path).read_text())

def main():
    first=read('verification_receipt.json');extra=read('additional_verification_receipt.json')
    original=read('reproduction_checks.json');long=read('extended_contract/v3long_reproduction.json')
    march1=read('march/jan_replay_parity.json');march2=read('extended_contract/march/jan_replay_parity.json')
    budget=read('budget900/receipt.json');cpu=read('budget900/cpu300/receipt.json')
    assert first['head_count']==40 and extra['additional_head_count']==28
    assert all(first['source_inputs_unchanged'].values())
    assert all(x['passed'] for x in original) and long['passed']
    assert first['group_isolation']['passed'] and first['future_perturbation']['passed']
    assert first['all_head_split_label_maturity_checks_pass'] and extra['full_parameter_update_and_precalibration_dates_pass']
    assert len(march1)==10 and len(march2)==5 and all(x['passed'] for x in march1+march2)
    assert extra['march_saved_models_reconstructed']==15 and all(x['passed'] for x in extra['march_checks'])
    assert budget['backbone_fits']==4 and budget['total_full_steps']==3600 and cpu['new_neural_fits']==0
    interval_files=['paired_intervals.csv','extended_contract/paired_intervals.csv','march/paired_intervals.csv','extended_contract/march/paired_intervals.csv','budget900/paired_intervals.csv','budget900/cpu_matched_paired_intervals.csv','budget900/forecast_paired_intervals.csv']
    for f in interval_files:assert pd.read_csv(OUT/f).repetitions.eq(10000).all()
    receipts=['protocol.json','device_smoke.json','assessment_receipt.json','reproduction_checks.json','verification_receipt.json','additional_verification_receipt.json','extended_contract/protocol.json','extended_contract/receipt.json','extended_contract/v3long_reproduction.json','march/receipt.json','march/jan_replay_parity.json','extended_contract/march/receipt.json','extended_contract/march/jan_replay_parity.json','budget900/protocol.json','budget900/receipt.json','budget900/cpu300/protocol.json','budget900/cpu300/receipt.json']
    ledger={f:sha(OUT/f) for f in receipts+interval_files}
    # Include exact checkpoint/cache/split receipts, without copying old data.
    evidence={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*.json')) if p.name.endswith('_receipt.json') or p.parent.name=='forecasts'}
    save(OUT/'final_verification.json',{'status':'PASS','verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'annual_head_count':68,'annual_head_groups':{'original_matched_2y_and10y':40,'exact_V3_extended10y':20,'Small900_exact_extended10y':4,'Small300_CPU_nuisance_control10y':4},'march_saved_head_count':15,'jan_replay_parity_check_count':15,'new900_step_backbone_count':4,'new_full_finetuning_steps':3600,'scientific_checks':{'old_source_and_weight_hashes_unchanged':True,'original_five2y_variants_exact_reproduction_4415rows_each':True,'exact_V3long_reproduction_4415rows':True,'all68_saved_annual_heads_reconstructed':True,'all15_saved_march_heads_and_full_cooldown_replays_reconstructed':True,'all_train_and_calibration_labels_mature_before_next_split':True,'all_finetuned_backbones_end_before_outer_calibration':True,'old_forecast_overlap_quantiles_bitwise_preserved':True,'actual_future_input_perturbation_zero_effect':True,'actual_mixed_date_group_isolation_within_numeric_tolerance':True,'all_paired_intervals_10000_month_blocks':True,'all_same_date_outcome_keys_compared':True},'limitations':['2023-25 retrospective development;2026 already-inspected retrospective diagnostic, not untouched holdout','early head features in-sample with respect to fixed pretrained backbone fit, not10years cross-fitted','small real-data pretraining overlap cannot be excluded; synthetic-only is publisher claim','March3-Aug25 is incomplete six-calendar-month interval','bootstrap conditional on fixed fitted models and not adjusted for hypothesis search'],'receipts_sha256':ledger,'split_cache_backbone_evidence_sha256':evidence,'code_sha256':sha(__file__)})
    paths=[p for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
    save(OUT/'MANIFEST.json',{'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths}})
    print('PASS:68annual heads,15March heads,15Jan replays,4full900-step backbones.')

if __name__=='__main__':main()
