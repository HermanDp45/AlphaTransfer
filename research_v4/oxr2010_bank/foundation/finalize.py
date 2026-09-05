#!/usr/bin/env python3
"""Seal only this new foundation extension after scientific verification."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    verified=json.loads((OUT/'verification.json').read_text())
    assert verified['status']=='PASS' and verified['models_reconstructed']==85
    assert verified['new_neural_fits']==verified['new_neural_forecasts']==0
    assert verified['pretest_halyk_lag1_addendum_sha256']==sha(OUT/'protocol_addendum_halyk_lag1.json')
    model_count=85
    if (OUT/'canonical/verification.json').exists():
        canonical=json.loads((OUT/'canonical/verification.json').read_text())
        assert canonical['status']=='PASS' and canonical['models_reconstructed']==15
        new=pd.read_csv(OUT/'canonical/predictions.csv.gz',parse_dates=['date'])
        old=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['date'],low_memory=False)
        checks=[]
        for (cid,cutoff),a in new.groupby(['config_id','cutoff']):
            a=a.sort_values(['date','corridor']).reset_index(drop=True)
            b=old[old.config_id.eq(cid.removesuffix('_canonical'))&old.cutoff.eq(cutoff)].sort_values(['date','corridor']).reset_index(drop=True)
            assert np.array_equal(a.date.to_numpy(),b.date.to_numpy()) and np.array_equal(a.corridor.to_numpy(),b.corridor.to_numpy())
            diff=float(abs(a.probability-b.probability).max());rawdiff=float(abs(a.raw_probability-b.raw_probability).max())
            signals=int((a.candidate_signal!=b.candidate_signal).sum());portfolio=int((a.signal!=b.signal).sum())
            assert diff==0 and rawdiff==0 and signals==0 and portfolio==0
            checks.append({'config_id':cid,'cutoff':cutoff,'probability_max_abs_diff':diff,'raw_probability_max_abs_diff':rawdiff,'candidate_signal_mismatches':signals,'portfolio_signal_mismatches':portfolio,'matches_previously_verified_original_exactly':True})
        assert len(checks)==15
        (OUT/'canonical/reproduction_to_original.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2)+'\n')
        model_count+=15
        (OUT/'final_verification.json').write_text(json.dumps({'status':'PASS','models':model_count,'original_models':85,'post_audit_canonical_models':15,'new_neural_fits':0,'new_neural_forecasts':0,'all_original_probability_and_policy_replays_pass':True,'all_canonical_heads_reconstructed_and_predicted_policies_equal_verified_originals':True,'common_OXR_features_exactly_canonical_since2018_09_01':True,'earlier_actual2010_history_preserved':True,'original_selection_unchanged':True,'original_verification_sha256':sha(OUT/'verification.json'),'canonical_verification_sha256':sha(OUT/'canonical/verification.json'),'canonical_reproduction_sha256':sha(OUT/'canonical/reproduction_to_original.json'),'canonical_post_audit_protocol_sha256':sha(OUT/'canonical/protocol.json')},indent=2)+'\n')
    for file in OUT.glob('*.py'):ast.parse(file.read_text())
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
    (OUT/'MANIFEST.json').write_text(json.dumps({'status':'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification_sha256':sha(OUT/'verification.json'),'models':model_count,'new_neural_fits':0,'new_neural_forecasts':0,'files':{str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}},indent=2,ensure_ascii=False)+'\n')
    for name in ['REPORT.md','README.md']:
        file=OUT/name
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text()):
            if not target.startswith('http'):assert (file.parent/target).resolve().exists(),target
    print('PASS:'+str(model_count)+'heads,zero new neural fits/forecasts,allreportlinks;files='+str(len(files)))
if __name__=='__main__':main()
