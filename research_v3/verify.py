"""Verify the delivered V3 research package using only the Python standard library.

--seal records current files only after structural, provenance and prediction
parity checks; ordinary invocation compares them to that frozen manifest.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8',newline='') as f:
        yield from csv.DictReader(f)


def verify_required():
    required=['README.md','REPORT.md','METHODOLOGY_REVIEW.md','selection.json','COMPARISON.csv','COMPARISON_MANIFEST.json','FINAL_COMPARISON.md','models/parity.json','models/paired_uncertainty.csv','models/decision_policy_intervals.csv','models/decision_policy_paired_deltas.csv','models/external_combo_vs_incumbent_ci.csv','external_data/REPORT.md','external_data/verification.json','external_data/source_access_decisions.csv','external_data/long_combo/verification.json','tabm/RESULTS.md','tabm/output/incumbent_parity.json','tabm/output/paired_brier_intervals.csv','behavior/REPORT.md','behavior/EVIDENCE.md','behavior/results/integrity_checks.json','behavior/results/headline_metrics.json','validation.json']
    for h in (1,3,5,10,20):required.append(f'models/summary_h{h}.csv')
    for name in required:
        if not (HERE/name).is_file():raise AssertionError(f'Missing deliverable: {name}')
    checks=[]
    lock=json.loads((ROOT/'final_solution/inputs.lock.json').read_text())
    for name,expected in lock['files'].items():
        if digest(ROOT/name)!=expected:raise AssertionError(f'Frozen V2 input changed: {name}')
    data_manifest=json.loads((ROOT/'final_solution/data/data_manifest.json').read_text())
    for name,record in data_manifest['artifacts'].items():
        if digest(ROOT/'final_solution/data'/name)!=record['sha256']:raise AssertionError(f'Normalized source changed: {name}')
    checks.append({'check':'frozen_input_lock_and_normalized_sources','locked_files':len(lock['files']),'normalized_artifacts':len(data_manifest['artifacts']),'status':'PASS'})
    receipts=list((HERE/'models').glob('*_receipt.json'))
    if len(receipts)<55:raise AssertionError(f'Only {len(receipts)} model/horizon receipts, expected at least 55')
    for receipt in receipts:
        r=json.loads(receipt.read_text())
        output=receipt.with_name(receipt.name.replace('_receipt.json','_predictions.csv.gz'))
        if r['status']!='complete' or digest(output)!=r['predictions_sha256']:raise AssertionError(f'Prediction receipt failed: {receipt}')
    checks.append({'check':'model_prediction_receipts','count':len(receipts),'status':'PASS'})
    base={(r['date'],r['corridor']):r for r in read_rows(HERE/'models/baseline_reproduction_h5_predictions.csv.gz')}
    compared=0;max_error=0.
    for name in ['development_h5_predictions.csv','diagnostic_2026_predictions.csv']:
        for r in read_rows(ROOT/'final_solution/model_bundle'/name):
            if r['config_id']!='hgb_plus_cnyrub_basis':continue
            b=base[(r['date'],r['corridor'])]
            err=abs(float(r['probability'])-float(b['probability']))
            if err>1e-12 or r['candidate_signal'].lower()!=b['candidate_signal'].lower():raise AssertionError('Frozen baseline numeric parity failed')
            max_error=max(max_error,err);compared+=1
    if compared!=4415 or len(base)!=4415:raise AssertionError('Baseline coverage incomplete')
    checks.append({'check':'independent_v2_prediction_parity','rows':compared,'max_abs_error':max_error,'status':'PASS'})
    table=list(read_rows(HERE/'COMPARISON.csv'))
    if len(table)<150:raise AssertionError('Incomplete experiment ledger')
    checks.append({'check':'comparison_ledger','rows':len(table),'status':'PASS'})
    val=json.loads((HERE/'validation.json').read_text())
    if val['status']!='PASS' or any(x['returncode'] for x in val['commands']):raise AssertionError('Required validation failed')
    checks.append({'check':'validation_commands','count':len(val['commands']),'status':'PASS'})
    behavior=json.loads((HERE/'behavior/results/headline_metrics.json').read_text())
    if behavior['empirical_customer_behavior_validation'] is not False or behavior['causal_incremental_revenue'] is not None:raise AssertionError('Synthetic effects mislabeled')
    selection=json.loads((HERE/'selection.json').read_text())
    if selection['production_ready'] or selection['real_customer_effect_identified']:raise AssertionError('Unsupported production claim')
    checks.append({'check':'simulation_and_production_boundaries','status':'PASS'})
    return checks


def files_to_seal():
    exclusions={'__pycache__','smoke','preview_output'}
    for path in HERE.rglob('*'):
        rel=path.relative_to(HERE)
        if not path.is_file() or any(p in exclusions for p in rel.parts):continue
        if path.suffix in {'.pyc','.pkl','.log'} or path.name in {'manifest.json','_SUCCESS.json'} and path.parent==HERE:continue
        yield path
    for pattern in ['alphatransfer_final/*.py','tests/*.py']:
        yield from (ROOT/'final_solution').glob(pattern)
    for name in ['main.py','README.md','APPROACH.md','inputs.lock.json']:
        yield ROOT/'final_solution'/name
    yield ROOT/'product_artifacts/V3_DECISION_CONTRACT.md'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seal',action='store_true');a=parser.parse_args()
    checks=verify_required()
    manifest=HERE/'manifest.json'
    if a.seal:
        payload={'schema_version':3,'status':'complete','nature':'Retrospective exploratory research; no real customer uplift or production promotion','checks':checks,'files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files_to_seal()))}}
        manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    else:
        payload=json.loads(manifest.read_text())
        for name,expected in payload['files'].items():
            if digest(ROOT/name)!=expected:raise AssertionError(f'Sealed artifact changed: {name}')
    result={'status':'PASS','sealed_files':len(payload['files']),'checks':checks,'manifest_sha256':digest(manifest)}
    (HERE/'_SUCCESS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
