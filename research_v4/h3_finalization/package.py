"""Package selected final H3 weights and independently calibrated causal policy."""
from pathlib import Path
import os,sys,json,shutil,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from research_v4.robust_selection.evaluate import core,ranked,policy_fit,apply
from final_solution.tabm_h3.policy import empty_state,replay
from final_solution.tabm_h3.model import Predictor
HERE=Path(__file__).resolve().parent;DEST=ROOT/'final_solution/tabm_h3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def rows(g):return g[['date','corridor','session_ordinal','probability']].to_dict('records')
def main():
 final=HERE/'final_fit';raw=pd.read_csv(final/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip')
 v,h,w=[raw[raw.split==name].sort_values('date').copy() for name in ('validation','history','warmup')]
 cal=core.fit_platt_calibrator(v.raw_probability.to_numpy(),v.target)
 for z in (v,h,w):z['probability']=core.apply_platt(cal,z.raw_probability.to_numpy())
 scores=ranked(pd.concat([w,h]).sort_values('date')).set_index('date').rank_score
 for z in (v,h,w):z['rank_score']=z.date.map(scores)
 policy=policy_fit(v,h,'rank80');meta=json.loads((final/'model/model.json').read_text());receipt=json.loads((final/'receipt.json').read_text())
 (DEST/'model').mkdir(exist_ok=True);(DEST/'data').mkdir(exist_ok=True)
 for f in ('weights.pt','preprocess.joblib','model.json'):shutil.copy2(final/'model'/f,DEST/'model'/f)
 paths=json.loads((HERE/'audit/source_paths.json').read_text());packed={};source_receipts=[]
 for name,path in paths.items():
  source=ROOT/path;dst=DEST/'data'/(name+'.csv');shutil.copy2(source,dst);packed[name]=str(dst.relative_to(DEST));source_receipts.append(dict(name=name,path=packed[name],source=str(source.relative_to(ROOT)),sha256=sha(dst)))
 config=dict(schema_version=1,profile_id='tabm_kzt_h3_expanding2010',model_id='tabm_kzt_h3_expanding2010_20260905',model_cutoff='2026-09-05',train_horizon=3,corridor='KZT',features=meta['features'],seed=meta['seed'],architecture=meta['architecture'],numerical_embeddings=meta['numerical_embeddings'],model=dict(weights='model/weights.pt',preprocessor='model/preprocess.joblib',weights_sha256=sha(DEST/'model/weights.pt'),preprocessor_sha256=sha(DEST/'model/preprocess.joblib')),calibration=dict(method=cal.method,intercept=cal.intercept,slope=cal.slope),policy=dict(name='rank80',kind='rank',threshold=policy['threshold']['KZT'],window=63,min_history=20,cooldown_sessions=2,max_contacts_per_week=2),source_paths=packed,initial_state='initial_state.json',metadata=dict(training=receipt,selection=json.loads((HERE/'selection.json').read_text()),latest_feature_date='2026-09-03',default_mode='historical_smoke',demo_features='example_features.csv',model_target='NOW_H3: currentRUBperKZT <= min(next3effectiveCBRsessionrates)',historical_metrics_are_recipe_not_final_checkpoint=True,no_live_alpha_quote=True,closing_status='pending_separate_H3_validation'))
 config['source_sha256']={x['name']:x['sha256'] for x in source_receipts}
 save(DEST/'bundle.json',config)
 # Score-only warmup exactly matches the research rank reference.
 _,warmstate=replay(rows(w),config,empty_state(config),emit_candidates=False)
 emitted,endstate=replay(rows(h),config,warmstate)
 expected=apply(h,dict(policy,initial_state={}))
 assert np.array_equal([x['candidate_signal'] for x in emitted],expected.candidate_signal)
 assert np.allclose([x['rank_score'] for x in emitted],h.rank_score,atol=0,rtol=0)
 _,before_demo=replay(rows(h.iloc[:-1]),config,warmstate)
 save(DEST/'initial_state.json',before_demo);save(DEST/'operational_state.json',endstate)
 panel=pd.read_pickle(HERE/'audit/latest_panel.pkl');features=panel[panel.date==h.date.max()].copy();features['feature_known_at']=features.date.dt.strftime('%Y-%m-%d')
 features[list(dict.fromkeys(['date','corridor','session_ordinal','rub_per_unit','feature_known_at',*config['features']]))].to_csv(DEST/'example_features.csv',index=False)
 final_model=Predictor(config,DEST);sourceframe=panel[panel.date.isin(h.date)].sort_values('date');rr,pp=final_model.predict(sourceframe)
 assert np.allclose(rr,h.raw_probability.to_numpy(),atol=1e-7,rtol=0)
 assert np.allclose(pp,h.probability.to_numpy(),atol=1e-7,rtol=0),float(np.max(np.abs(pp-h.probability.to_numpy())))
 runtime_h=h.copy();runtime_h['probability']=pp
 wp=panel[panel.date.isin(w.date)].sort_values('date');_,wprob=final_model.predict(wp);runtime_w=w.copy();runtime_w['probability']=wprob
 _,runtime_warm=replay(rows(runtime_w),config,empty_state(config),emit_candidates=False)
 runtime_contacts,runtime_end=replay(rows(runtime_h),config,runtime_warm)
 _,runtime_before=replay(rows(runtime_h.iloc[:-1]),config,runtime_warm)
 save(DEST/'initial_state.json',runtime_before);save(DEST/'operational_state.json',runtime_end)
 assert [x['candidate_signal'] for x in runtime_contacts]==[x['candidate_signal'] for x in emitted]
 assert np.allclose([x['rank_score'] for x in runtime_contacts],[x['rank_score'] for x in emitted],atol=0,rtol=0)
 _,state2=replay(rows(h.iloc[-1:]),config,before_demo);assert state2==endstate
 save(DEST/'source_receipt.json',source_receipts);save(DEST/'training_receipt.json',receipt);save(DEST/'policy_calibration_receipt.json',policy)
 save(HERE/'deployment_receipt.json',dict(status='packaged',profile=config['profile_id'],model_cutoff=config['model_cutoff'],history_start='2010-01-01',train_rows=receipt['train_rows'],calibration_rows=len(v),latest_feature_date=str(h.date.max().date()),policy=config['policy'],calibration=config['calibration'],state_history_scores=len(endstate['past_scores']),calibration_history_replay_rows=len(h),rank_and_candidate_replay='exact',raw_replay_max_error=float(np.max(np.abs(rr-h.raw_probability.to_numpy()))),calibrated_replay_max_error=float(np.max(np.abs(pp-h.probability.to_numpy()))),bundle_sha256=sha(DEST/'bundle.json'),weights_sha256=config['model']['weights_sha256'],scope='KZT,H3',historical_smoke_is_not_OOT=True))
 print('Package PASS',config['policy'],'history',len(h),'latest',h.date.max(),flush=True)
if __name__=='__main__':main()
