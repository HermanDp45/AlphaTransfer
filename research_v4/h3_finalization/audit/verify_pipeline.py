"""Independent final H3 source/model/state verification. Inference only; no fits."""
from pathlib import Path
import os,sys,json,hashlib,copy,shutil
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
from final_solution.tabm_h3 import features as f,policy as pol,predict as cli
from final_solution.tabm_h3.model import Predictor
from research_v4.robust_selection.audit.verify_evaluation import reference_rank,apply,m,equal
HERE=Path(__file__).resolve().parent;P=HERE.parent;D=ROOT/'final_solution/tabm_h3';checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(name,ok,**info):
    checks.append(dict(name=name,status='PASS' if bool(ok) else 'FAIL',**info));assert ok,(name,info)
def rows(g):return [dict(date=str(r.date.date()),corridor='KZT',session_ordinal=int(r.session_ordinal),probability=float(r.probability)) for r in g.itertuples()]
def reject(name,fn):
    try:fn()
    except ValueError:check(name,True);return
    check(name,False)
def main():
    config=read(D/'bundle.json');rc=read(P/'final_fit/receipt.json');fp=read(P/'final_fit/protocol.json');meta=read(P/'final_fit/model/model.json')
    panel=pd.read_pickle(HERE/'latest_panel.pkl').sort_values('date').reset_index(drop=True)
    check('panel_input_sha_locked',sha(HERE/'latest_panel.pkl')==rc['panel_sha256']==fp['panel_sha256'])
    check('source_receipt_input_sha_locked',sha(HERE/'latest_panel_receipt.json')==fp['source_receipt_sha256'])
    prepatch=HERE/'feature_builder_snapshot.py'
    if not prepatch.exists():prepatch=Path('/private/tmp/tabm_h3_features_pre_dateguard.py')
    expected='375e052e24306d208750f24e96475e3f47ca60122fb53265f4db455fb4efc1f3'
    if prepatch.exists():
        check('original_builder_snapshot',sha(prepatch)==expected)
        if prepatch!=HERE/'feature_builder_snapshot.py':shutil.copy2(prepatch,HERE/'feature_builder_snapshot.py')
    sources=f.read_sources(config['source_paths'],D);built=f.build_features(sources,'2026-09-05')
    check('latest_standalone_features_exact',len(built)==4118 and np.array_equal(built[f.FEATURES],panel[f.FEATURES],equal_nan=True) and np.array_equal(built.session_ordinal,panel.session_ordinal) and np.array_equal(built.rub_per_unit,panel.rub_per_unit),rows=len(built),features=len(f.FEATURES),latest=str(built.date.max()))
    (HERE/'postpatch_builder_parity.json').write_text(json.dumps(dict(status='PASS',prepatch_sha256=expected,current_builder_sha256=sha(f.__file__),latest_panel_sha256=sha(HERE/'latest_panel.pkl'),feature_rows=4118,features=33,exact=True,change='Daily calendar-date input guard only. Original source hash in panel receipt remains intentional provenance.'),indent=2)+'\n')
    for src in read(D/'source_receipt.json'):
        check('source_copy_'+src['name'],sha(D/src['path'])==src['sha256']==sha(ROOT/src['source']))
    check('package_H3_KZT_full33_contract',config['train_horizon']==3 and config['corridor']=='KZT' and config['features']==f.FEATURES==meta['features'] and config['metadata']['historical_metrics_are_recipe_not_final_checkpoint'])
    for name in ('weights.pt','preprocess.joblib','model.json'):
        check('model_copy_'+name,sha(D/'model'/name)==sha(P/'final_fit/model'/name))
    # Independently derive actual H3 maturity and training/calibration cohorts.
    label=panel.date.shift(-3);rates=panel.rub_per_unit.to_numpy();target=np.full(len(panel),np.nan)
    for i in range(len(panel)-3):target[i]=float(rates[i]<=min(rates[i+1:i+4]))
    check('independent_H3_labels',np.array_equal(target,panel.target,equal_nan=True) and label.equals(pd.to_datetime(panel.label_available_date)))
    cutoff=pd.Timestamp('2026-09-05');cs=pd.Timestamp('2025-09-05')
    train=panel[panel.date.lt(cs)&label.lt(cs)].copy();val=panel[panel.date.ge(cs)&panel.date.lt(cutoff)&label.lt(cutoff)].copy();hist=panel[panel.date.ge(cs)&panel.date.lt(cutoff)].copy();warm=panel[panel.date.lt(cs)].tail(63).copy()
    check('final_maturity_splits',len(train)==3869 and len(val)==243 and len(hist)==246 and len(warm)==63 and train.label_available_date.max()<cs and val.label_available_date.max()<cutoff and set(train.date).isdisjoint(val.date),train_end=str(train.date.max()),train_label_max=str(train.label_available_date.max()),cal_end=str(val.date.max()),cal_label_max=str(val.label_available_date.max()))
    for key,value in [('train_rows',len(train)),('validation_rows',len(val)),('history_rows',len(hist)),('warmup_dates',len(warm))]:check('receipt_'+key,rc[key]==value)
    raw=pd.read_csv(P/'final_fit/raw_predictions.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip')
    check('raw_file_lock',sha(P/'final_fit/raw_predictions.csv.gz')==rc['raw_predictions_sha256'])
    check('no_fake_final_test',set(raw.split)=={'warmup','history','validation','tail'})
    for part,frame in [('validation',val),('history',hist),('warmup',warm),('tail',hist[hist.target.isna()])]:
        q=raw[raw.split.eq(part)].sort_values('date');check('raw_cohort_'+part,list(q.date)==list(frame.date))
        if part in ('warmup','tail'):check('masked_outcomes_'+part,q[['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all())
    immature=raw.split.eq('history')&(raw.label_available_date.isna()|raw.label_available_date.ge(cutoff))
    check('history_immature_masked',len(raw[immature])==3 and raw.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all())
    # No fit: independently compute median statistics and empirical quantiles.
    pre=joblib.load(D/'model/preprocess.joblib');x=train[f.FEATURES].to_numpy();med=np.nanmedian(x,axis=0);med=np.where(np.isnan(med),0.,med)
    check('train_only_medians_exact',np.array_equal(med,pre.named_steps['impute'].statistics_))
    filled=np.where(np.isnan(x),med,x);qt=pre.named_steps['gaussian'];quant=np.nanpercentile(filled,qt.references_*100,axis=0);quant=np.maximum.accumulate(quant,axis=0)
    check('train_only_quantiles_exact',np.array_equal(quant,qt.quantiles_) and qt.n_quantiles_==128)
    innerstart=train.date.iloc[-63];it=train[train.date.lt(innerstart)&train.label_available_date.lt(innerstart)];iv=train[train.date.ge(innerstart)]
    check('inner_maturity_boundary',str(it.label_available_date.max())==meta['inner_label_max'] and str(iv.date.min())==meta['inner_validation_min'] and it.label_available_date.max()<iv.date.min())
    epochs=pd.read_csv(P/'final_fit/model/inner_epochs.csv',float_precision='round_trip');best=float('inf');bestepoch=1
    for r in epochs.itertuples():
        if r.inner_brier<best-1e-6:best,bestepoch=r.inner_brier,int(r.epoch)
    refit=pd.read_csv(P/'final_fit/model/refit_epochs.csv')
    check('inner_epoch_selection_and_outer_refit',bestepoch==40==meta['selected_epochs']==len(refit) and abs(best-meta['inner_best_brier'])<1e-15 and epochs.epoch.iloc[-1]-bestepoch==15)
    # Replay all held-out/latest and warmup scores from packaged weights.
    predictor=Predictor(config,D);parts={};maxerr=0.
    for part,frame in [('history',hist),('warmup',warm),('validation',val)]:
        rr,pp=predictor.predict(frame);q=raw[raw.split.eq(part)].sort_values('date').copy()
        err=float(np.max(np.abs(rr-q.raw_probability.to_numpy())));maxerr=max(maxerr,err)
        check('checkpoint_replay_'+part,np.array_equal(rr.astype(np.float32),q.raw_probability.to_numpy().astype(np.float32)),rows=len(q),max_csv_decimal_error=err)
        q['probability']=pp;parts[part]=q
    # Check published Platt coefficients solve the prior-validation objective.
    v=raw[raw.split.eq('validation')];r=np.clip(v.raw_probability.to_numpy(),1e-6,1-1e-6);lg=np.log(r/(1-r));cc=config['calibration'];pc=expit(cc['intercept']+cc['slope']*lg);y=v.target.to_numpy()
    grad=np.array([(pc-y).mean(),np.mean((pc-y)*lg)+cc['slope']/1e6/len(v)])
    check('platt_prior_validation_optimum',np.max(np.abs(grad))<1e-4 and cc['slope']>0 and np.mean((pc-y)**2)<np.mean((r-y)**2),maximum_gradient=float(np.max(np.abs(grad))),raw_brier=float(np.mean((r-y)**2)),calibrated_brier=float(np.mean((pc-y)**2)))
    h,w=parts['history'],parts['warmup'];v=parts['validation'];allrows=pd.concat([w,h]).sort_values('date');allrows['rank_score']=reference_rank(allrows);rankmap=allrows.set_index('date').rank_score
    for q in (h,w,v):q['rank_score']=q.date.map(rankmap)
    # Independently select the prior-calibration threshold over the fixed grid.
    candidates=[]
    for threshold in np.arange(0,.81,.05):
        q,_=apply(v,{'KZT':threshold},2,rank=True);ms=m(q);candidates.append(dict(threshold=float(threshold),coverage=ms['week_coverage'],lift=ms['lift'],utility=ms['forward_delta_bps']))
    feasible=[x for x in candidates if x['coverage']>=.8];chosen=max(feasible,key=lambda x:(x['lift'],x['utility'],x['coverage']))
    check('final_prior_calibration_rank80_threshold',chosen['threshold']==config['policy']['threshold']==.6000000000000001,candidates=len(candidates),selected=chosen)
    _,warmstate=pol.replay(rows(w),config,pol.empty_state(config),emit_candidates=False);dec,end=pol.replay(rows(h),config,warmstate)
    reference,_=apply(h,{'KZT':chosen['threshold']},2,rank=True)
    check('runtime_rank_and_contact_exact',np.array_equal([x['rank_score'] for x in dec],h.rank_score) and np.array_equal([x['candidate_signal'] for x in dec],reference.candidate_signal),rows=len(h),contacts=sum(x['candidate_signal'] for x in dec))
    _,before=pol.replay(rows(h.iloc[:-1]),config,warmstate)
    check('persisted_states_are_runtime_binary_probabilities',before==read(D/'initial_state.json') and end==read(D/'operational_state.json'))
    current=copy.deepcopy(warmstate);incremental=[]
    for left,right in [(0,1),(1,63),(63,105),(105,245),(245,246)]:
        z,current=pol.replay(rows(h.iloc[left:right]),config,current);incremental.extend(z)
    check('state_batch_incremental_exact',incremental==dec and current==end)
    poisoned=h.copy();poisoned.loc[poisoned.index[150]:,'probability']=.314159
    output,_=pol.replay(rows(poisoned),config,warmstate);check('future_score_poison_prefix_exact',output[:150]==dec[:150])
    poison=hist.copy();poison['target']=1-poison.target;poison['forward_bps']=1e9;poison['regret_bps']=-1e9
    check('outcomes_excluded_from_predictor',np.array_equal(predictor.predict(poison)[0],predictor.predict(hist)[0]))
    reject('state_overlap_rejected',lambda:pol.replay(rows(h.iloc[-1:]),config,end))
    wrong=copy.deepcopy(config);wrong['calibration']['slope']+=.01;reject('state_calibration_binding_rejected',lambda:pol.validate_state(end,wrong))
    reject('intraday_asof_rejected',lambda:f.build_features(sources,'2026-09-05T01:00:00+03:00'))
    # Only dated snapshot mode can replay the current pre-model-cutoff feature date.
    result=cli.run(D/'bundle.json',HERE/'runtime_smoke','2026-09-05',mode='historical_smoke')
    check('runtime_historical_smoke_labels',result['train_horizon']==3 and result['status']=='historical_snapshot_smoke' and result['is_oot_claim'] is False and result['feature_date']=='2026-09-03' and result['authorized_contact'] is False and result['external_messages_sent']==0)
    reject('operational_precutoff_replay_rejected',lambda:cli.run(D/'bundle.json',HERE/'runtime_reject','2026-09-05',mode='operational'))
    none=cli.run(D/'bundle.json',HERE/'runtime_no_new','2026-09-05',mode='operational',state_in=D/'operational_state.json')
    check('completed_state_no_duplicate_contact',none['status']=='no_new_source_sessions' and none['NOW_contacts']==0 and none['candidate_signal'] is False)
    closing=config.get('closing',{})
    check('H5_closing_not_carried',closing.get('enabled') is False and closing.get('train_horizon')==3 and closing.get('status')=='diagnostic_only_failed_annual_annotation_gates' and result['closing']['train_horizon']==3 and result['closing']['annotation_active'] is False and result['annotations']==[] and sha(D/closing['model'])==closing['model_sha256'])
    check('standalone_no_research_imports',all('research_v' not in (D/name).read_text() for name in ('features.py','model.py','policy.py','predict.py','closing.py')))
    check('current_annual_evidence_stable',all(sha(P/path)==value for path,value in read(HERE/'annual_verification.json')['source_sha256'].items()))
    paths=[D/name for name in ('bundle.json','features.py','model.py','policy.py','predict.py','closing.py','model/closing_h3.joblib','initial_state.json','operational_state.json','source_receipt.json','training_receipt.json','policy_calibration_receipt.json')]+[P/'final_fit'/name for name in ('receipt.json','protocol.json','raw_predictions.csv.gz','model/model.json','model/inner_epochs.csv','model/refit_epochs.csv')]
    receipt=dict(status='PASS',passed=len(checks),checks=checks,maximum_raw_csv_decimal_error=maxerr,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},audit_sha256=sha(__file__),scope='Independent final H3 source/feature/maturity/epoch/preprocessor/checkpoint/calibration/policy/state checks. No training fits. Final checkpoint smoke is not a historical OOT evaluation.')
    (HERE/'pipeline_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks),max_csv_raw_error=maxerr)),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
