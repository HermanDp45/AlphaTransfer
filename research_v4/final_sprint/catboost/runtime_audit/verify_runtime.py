"""Read-only profile audit; all generated inputs/outputs remain in this directory."""
from pathlib import Path
import sys,json,hashlib,importlib.util,pickle,warnings
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PROFILE=ROOT/'final_solution/final_sprint'
spec=importlib.util.spec_from_file_location('audited_profile_predict',PROFILE/'predict.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    checks=[]
    def check(name,passed,**details):
        checks.append(dict(check=name,passed=bool(passed),**details));print(name,'PASS' if passed else 'FAIL',flush=True)
    cfg=json.loads((PROFILE/'model.json').read_text());frame=pd.read_csv(PROFILE/'example_features.csv',parse_dates=['date'],float_precision='round_trip')
    profile_files=[p for p in PROFILE.rglob('*') if p.is_file() and 'output' not in p.parts]
    hashes={str(p.relative_to(PROFILE)):sha(p) for p in profile_files}
    full=HERE/'full.csv';state=HERE/'full_state.json'
    runtime.run(PROFILE/'example_features.csv',full,state_out=state)
    output=pd.read_csv(full,parse_dates=['date'],float_precision='round_trip')
    saved=pd.read_csv(PROFILE/'selected_predictions.csv.gz',parse_dates=['date'],float_precision='round_trip')
    saved=saved[saved.cutoff.eq(cfg['cutoff'])&saved['mode'].eq('normal')]
    merged=output.merge(saved,on=['date','corridor'],suffixes=['_runtime','_saved'],validate='one_to_one')
    errors={c:float(np.max(abs(merged[c+'_runtime']-merged[c+'_saved']))) for c in ('raw_probability','probability')}
    check('full_selected_profile_replay',len(merged)==156 and max(errors.values())<2e-7
          and np.array_equal(merged.candidate_signal_runtime,merged.candidate_signal_saved),
          rows=len(merged),maximum_probability_errors=errors,actions_exact=True,
          note='Tolerance covers original float32 inference/calibration versus decimal CSV and float64 exported calibration arithmetic.')
    pieces=[];state_in=None
    for i,indices in enumerate(np.array_split(np.arange(len(frame)),3)):
        inp=HERE/f'chunk{i}.csv';dest=HERE/f'chunk{i}_predictions.csv';next_state=HERE/f'chunk{i}_state.json'
        frame.iloc[indices].to_csv(inp,index=False)
        runtime.run(inp,dest,state_in=state_in,state_out=next_state)
        pieces.append(pd.read_csv(dest,parse_dates=['date'],float_precision='round_trip'));state_in=next_state
    joined=pd.concat(pieces,ignore_index=True)
    errors={c:float(np.max(abs(output[c]-joined[c]))) for c in ('raw_probability','probability','closing_probability')}
    check('three_chunks_match_full_profile_contacts_and_annotations',max(errors.values())<2e-7
          and output.candidate_signal.equals(joined.candidate_signal)
          and output.closing_annotation.equals(joined.closing_annotation)
          and json.loads(state.read_text())==json.loads(state_in.read_text()),
          maximum_probability_errors=errors,actions_and_final_state_exact=True)
    forbidden={'target','forward_bps','symmetric_bps','regret_bps','label_available_date','candidate_signal'}
    check('outcomes_absent_from_both_model_feature_lists',not forbidden.intersection(cfg['features']+cfg['closing_features']))
    poison=frame.copy()
    for name in forbidden:poison[name]=999999.
    poison.to_csv(HERE/'poisoned_outcomes.csv',index=False)
    runtime.run(HERE/'poisoned_outcomes.csv',HERE/'poisoned_predictions.csv')
    poisoned=pd.read_csv(HERE/'poisoned_predictions.csv',parse_dates=['date'],float_precision='round_trip')
    check('outcome_columns_cannot_change_predictions_or_contacts',output.equals(poisoned))
    def rejects(name,content,state_in=None):
        inp=HERE/(name+'.csv');content.to_csv(inp,index=False)
        try:
            runtime.run(inp,HERE/(name+'_unexpected_output.csv'),state_in=state_in)
            check(name,False,error='Input accepted')
        except ValueError as error:check(name,True,rejection=str(error))
    rejects('missing_middle_sessions_rejected',frame.iloc[[0,-1]])
    rejects('missing_first_session_without_state_rejected',frame.iloc[1:])
    rejects('overlap_including_previously_processed_nonsignal_rejected',frame.iloc[51:54],HERE/'chunk0_state.json')
    rejects('skipped_incremental_session_rejected',frame.iloc[53:55],HERE/'chunk0_state.json')
    wrong=json.loads((HERE/'chunk0_state.json').read_text());wrong['config_id']='other_model'
    (HERE/'wrong_model_state.json').write_text(json.dumps(wrong))
    rejects('wrong_model_state_rejected',frame.iloc[52:54],HERE/'wrong_model_state.json')
    for component in cfg['components']:
        meta=json.loads((PROFILE/component['path']/'model.json').read_text())
        check('packaged_component_hashes_'+component['path'],
              sha(PROFILE/component['path']/'weights.pt')==meta['weights_sha256']
              and sha(PROFILE/component['path']/'preprocess.joblib')==meta['preprocessor_sha256']
              and component['features']==meta['features'])
    stem=cfg['config_id'].rsplit('_',1)[0]
    original=pickle.loads((ROOT/'research_v4/final_sprint/tabm/output'/(stem+'_'+cfg['cutoff']+'_policies.pkl')).read_bytes())['kztcal']['calibrator']
    cal=cfg['calibration']
    check('calibration_contract_matches_selected_KZT_calibrator',cal['method']==original.method and cal['intercept']==original.intercept and cal['slope']==original.slope)
    check('profile_files_unchanged_by_audit',all(sha(PROFILE/name)==value for name,value in hashes.items()))
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),checks=checks,
                config_id=cfg['config_id'],profile_hashes=hashes,tree_fits=0,neural_fits=0,external_messages=0,
                verifier_sha256=sha(__file__),scope='Executable historical profile on frozen feature rows. This does not supply real-time PIT features or authorize contacts.')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','passed','failed')}),flush=True)
    if result['failed']:raise SystemExit(1)

if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning);main()
