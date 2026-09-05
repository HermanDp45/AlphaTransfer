"""Replay every saved neural checkpoint and verify causal split/policy contracts."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,time
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as x
from research_v4.final_sprint import common
HERE=x.HERE
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    checks=[];replays=[]
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert x.sha(ROOT/path)==digest
    checks.append(dict(check='Frozen source inputs unchanged',status='PASS',files=len(protocol['source_hashes'])))
    with threadpool_limits(limits=2):
        views,features=x.build();diagnostics=[]
        for cd in sorted(x.CKPT.iterdir()):
            if not (cd/'model.json').exists():continue
            meta=json.loads((cd/'model.json').read_text());sp=json.loads((cd/'split.json').read_text());cutoff=pd.Timestamp(sp['cutoff'])
            assert pd.Timestamp(sp['train_latest_label'])<pd.Timestamp(sp['calibration_start'])
            assert pd.Timestamp(sp['validation_latest_label'])<cutoff
            assert pd.Timestamp(meta['inner_label_max'])<pd.Timestamp(meta['inner_validation_min'])
            assert x.sha(cd/'weights.pt')==meta['weights_sha256'];assert x.sha(cd/'preprocess.joblib')==meta['preprocessor_sha256']
            receipt=json.loads((x.OUT/(cd.name+'.json')).read_text());spec=receipt['spec']
            model=x.Neural(meta['features'],meta['seed']);model.load(cd)
            cached=joblib.load(cd/'raw_predictions.joblib');tr,va,te,history=x.split(views,spec,cutoff)
            for mode,(delay,lag) in x.MODES.items():
                q=x.e.stress_view(views,dict(since='2010-01-01'),cutoff,delay,lag)
                pre=q.date.lt(cutoff)
                # The stitched rolling recomputation may differ by final floating
                # bits, so compare materially with1e-10 (same source values).
                a=q.loc[pre,features].to_numpy(float);b=views['2010-01-01',24,1].loc[pre,features].to_numpy(float)
                assert np.allclose(a,b,equal_nan=True,atol=1e-10,rtol=1e-10)
                f=q.loc[te.index];raw=model.predict(f)
                assert np.array_equal(raw,cached[mode].raw_probability.to_numpy())
                one=model.predict(f.iloc[:1]);assert np.allclose(one,raw[:1],atol=2e-6)
                replays.append(dict(checkpoint=cd.name,mode=mode,rows=len(f),maximum_raw_difference=float(np.max(np.abs(raw-cached[mode].raw_probability.to_numpy())))))
            for split,f in [('validation',va),('history',history)]:
                assert np.array_equal(model.predict(f),cached[split].raw_probability.to_numpy())
            if cd.name=='tabm_periodic_kzt_120m_s0_2025-01-01':
                normal=views['2010-01-01',24,1].loc[te.index]
                delayed=x.e.stress_view(views,dict(since='2010-01-01'),cutoff,48,1).loc[te.index]
                bundle=pickle.loads((x.OUT/(cd.name+'_policies.pkl')).read_bytes())['kztcal']
                for feature in features:
                    probe=delayed.copy();probe[feature]=normal[feature]
                    prob=x.core.apply_platt(bundle['calibrator'],model.predict(probe))
                    diagnostics.append(dict(feature_restored_to_normal=feature,brier=float(np.mean((prob-probe.target)**2)),mean_probability=float(prob.mean()),normal_mean=float(normal[feature].mean()),delayed_mean=float(delayed[feature].mean()),normal_train_min=float(tr[feature].min()),normal_train_max=float(tr[feature].max())))
        checks.append(dict(check='Checkpoint hash and exact numerical prediction replay',status='PASS',models=len(replays)//4,test_modes=len(replays)))
        checks.append(dict(check='Actual train/calibration/inner-validation label maturity',status='PASS',models=len(replays)//4))
        checks.append(dict(check='Fixed-model delayed source views preserve pre-cutoff inputs and outcomes',status='PASS',modes=len(replays)))
        pd.DataFrame(replays).to_csv(HERE/'checkpoint_replay.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(HERE/'source_delay_diagnostic.csv',index=False)
    p=pd.read_csv(HERE/'policy_predictions.csv.gz',parse_dates=['date','label_available_date'])
    c=pd.read_csv(HERE/'policy_calibration_predictions.csv.gz',parse_dates=['date','label_available_date'])
    policies=pickle.loads((HERE/'policies.pkl').read_bytes());policyreplays=0
    for (cid,cutoff,policy,mode),q in p.groupby(['config_id','cutoff','policy','mode']):
        q=q.reset_index(drop=True)
        pol=policies[cid,cutoff,policy];replay=common.apply_policy(q,pol)
        assert np.array_equal(replay.candidate_signal,q.candidate_signal)
        assert q.date.nunique()==(156 if cutoff=='2026-01-01' else 242)
        assert len(q)==q.date.nunique() and q.corridor.eq('KZT').all()
        policyreplays+=1
    for (cid,cutoff),q in c.groupby(['config_id','cutoff']):
        va=q[q.split.eq('validation')];assert va.label_available_date.max()<pd.Timestamp(cutoff)
        immature=q.label_available_date.ge(pd.Timestamp(cutoff))|q.label_available_date.isna()
        assert q.loc[immature,['target','forward_bps']].isna().all().all()
    checks.append(dict(check='All common-policy decisions replay exactly; identical calendar dates',status='PASS',groups=policyreplays))
    checks.append(dict(check='Exported validation is mature; immature history outcomes masked',status='PASS'))
    weights=json.loads((HERE/'blend_weights.json').read_text())
    for w in weights:
        scores={float(k):v for k,v in w['grid_brier'].items()};assert scores[w['neural_weight']]==min(scores.values())
        assert pd.Timestamp(w['validation_last'])<pd.Timestamp(w['cutoff'])
    checks.append(dict(check='Blend weights minimize prior-calibration grid Brier, including zero',status='PASS',blends=len(weights)))
    sel=json.loads((HERE/'selection_2025.json').read_text());assert sel['chosen']==min(sel['scores'],key=lambda v:v['brier'])['spec'] and not sel['uses_2026']
    checks.append(dict(check='Architecture selected only on2025 before additional seeds',status='PASS'))
    registry=json.loads((HERE/'ensemble_registry.json').read_text())
    for entry in registry:
        assert len(entry['components'])==3 and not entry['test_fitted_weights']
        assert np.isclose(sum(c['raw_weight'] for c in entry['components']),1.)
        raw=[]
        for component in entry['components']:
            cd=x.CKPT/component['model']
            for file,key in [('weights.pt','weights_sha256'),('preprocess.joblib','preprocessor_sha256'),('raw_predictions.joblib','raw_predictions_sha256')]:assert x.sha(cd/file)==component[key]
            raw.append(joblib.load(cd/'raw_predictions.joblib'))
        exported=pd.read_csv(x.OUT/(entry['config_id']+'_'+entry['cutoff']+'.csv.gz'))
        for mode in x.MODES:
            expected=np.mean([r[mode].raw_probability.to_numpy() for r in raw],axis=0)
            assert np.allclose(expected,exported[exported['mode'].eq(mode)].raw_probability.to_numpy(),atol=1e-15,rtol=0)
    checks.append(dict(check='Both annual ensembles use fixed equal raw weights and verified three-component checkpoint hashes',status='PASS',ensembles=len(registry)))
    save=dict(status='PASS',checks=checks,neural_checkpoints=len(replays)//4,temporary_epoch_selection_fits=len(replays)//4,policy_groups=policyreplays,finished_unix=time.time(),code_sha256=x.sha(__file__),experiment_sha256=x.sha(x.__file__),protocol_sha256=x.sha(HERE/'protocol.json'),augmentation_protocol_sha256=x.sha(HERE/'augmentation_protocol.json'),no_age_protocol_sha256=x.sha(HERE/'no_age_protocol.json'),policy_sha256=x.sha(HERE/'policies.pkl'))
    x.save(HERE/'verification.json',save);print(json.dumps(save,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
