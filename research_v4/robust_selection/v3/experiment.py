"""Exact V3 basis_train_120m recipe, separately fitted H3/H5 annual 2024–2026."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json,pickle,time,platform
from dataclasses import asdict
import joblib,numpy as np,pandas as pd,sklearn
from threadpoolctl import threadpool_limits
from research_v3.models import experiment as old
core=old.core;HERE=Path(__file__).resolve().parent;SOURCE=ROOT/'research_v3/models/panel_extended.pkl'
SPEC=old.Spec('basis_train_120m',months=120,extended=True)
FEATURES=old.feature_list(SPEC)
KEEP=['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date','rub_per_unit','ret1','pr60']
OUTCOME=['target','forward_bps','symmetric_bps','regret_bps']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def fp(q):return hashlib.sha256(pd.util.hash_pandas_object(q,index=False).to_numpy().tobytes()).hexdigest()
def target_panel(raw,h):
    p=core.add_target(raw,h);p['label_available_date']=p.groupby('corridor').date.shift(-h);return p

def parts(p,h,year):
    cut=pd.Timestamp(year,1,1);end=pd.Timestamp(year+1,1,1);cs=pd.Timestamp(year-1,1,1)
    tr,va,te=old.temporal_split(p,h,cut,end,SPEC)
    history=p[p.date.ge(cs)&p.date.lt(cut)].copy()
    assert tr.label_available_date.lt(cs).all() and va.label_available_date.lt(cut).all()
    assert te.label_available_date.notna().all() and te.label_available_date.lt(end).all()
    assert all(x.corridor.nunique()==5 for x in (tr,va,te,history))
    return {'train':tr,'validation':va,'history':history,'test':te}

def policy(frame,prob,threshold,initial=None):
    initial=initial or {'candidate':{},'portfolio':{}}
    cand=core.select_per_corridor_with_cooldown(frame,prob,threshold,initial['candidate'])
    selected=core.select_portfolio_from_candidates(frame,prob,cand,initial['portfolio'])
    state={'candidate':core.corridor_selection_state(frame,cand,initial['candidate']),'portfolio':core.selection_state(frame,selected,initial['portfolio'])}
    return frame.index.isin(cand),frame.index.isin(selected),state

def run(p,h,year):
    start=time.monotonic();q=parts(p,h,year);tr=q['train'];model=old.make_model(SPEC,tr)
    model.fit(tr[FEATURES+['corridor']],tr.target.astype(int))
    raw={k:model.predict_proba(v[FEATURES+['corridor']])[:,1] for k,v in q.items() if k!='train'}
    calibrator=core.fit_platt_calibrator(raw['validation'],q['validation'].target)
    prob={k:core.apply_platt(calibrator,v) for k,v in raw.items()}
    threshold,_,_=core.choose_frequency_threshold(q['validation'],prob['validation'])
    _,_,history_state=policy(q['history'],prob['history'],threshold)
    _,_,strict_state=policy(q['history'],prob['history'],.5)
    bundle={'model':model,'calibrator':calibrator,'features':FEATURES,'spec':asdict(SPEC),'train_horizon':h,'year':year,'original_threshold':threshold,'strict_threshold':.5,'original_state':history_state,'strict05_state':strict_state}
    cp=HERE/'checkpoints'/f'v3_h{h}_{year}.joblib';joblib.dump(bundle,cp)
    frames=[]
    for split_name in ('validation','history','test'):
        f=q[split_name];out=f[KEEP].copy();out['raw_probability']=raw[split_name];out['probability']=prob[split_name]
        out['original_v3_probability']=prob[split_name];out['config_id']='v3';out['train_horizon']=h
        out['fold_test_year']=year;out['cutoff']=f'{year}-01-01';out['split']=split_name;out['calibration_scope']='pooled5_original_v3'
        out['candidate_signal'],out['signal'],_=policy(f,prob[split_name],threshold,history_state if split_name=='test' else None)
        out['original_candidate_signal']=out.candidate_signal;out['original_signal']=out.signal
        out['strict05_candidate_signal'],out['strict05_signal'],_=policy(f,prob[split_name],.5,strict_state if split_name=='test' else None)
        out['calibration_method']=calibrator.method;out['calibration_status']=calibrator.status
        out['platt_intercept']=calibrator.intercept;out['platt_slope']=calibrator.slope
        out['labels_matured_at_cutoff']=out.label_available_date.lt(pd.Timestamp(year,1,1))
        if split_name=='history':out.loc[~out.labels_matured_at_cutoff,OUTCOME]=np.nan
        frames.append(out)
    receipt={'spec':asdict(SPEC),'train_horizon':h,'year':year,'checkpoint_sha256':sha(cp),'checkpoint':str(cp.relative_to(HERE)),'features':FEATURES,'train_feature_target_sha256':fp(tr[['date','corridor',*FEATURES,'target','label_available_date']]),'original_threshold':threshold,'original_history_state':history_state,'strict05_history_state':strict_state,'calibrator':{k:v for k,v in asdict(calibrator).items() if k!='model'},'splits':{k:{'rows':len(f),'date_min':str(f.date.min()),'date_max':str(f.date.max()),'latest_label_date':str(f.label_available_date.max())} for k,f in q.items()},'model_parameters':model.named_steps['classifier'].get_params(),'seconds':time.monotonic()-start}
    save(HERE/f'v3_h{h}_{year}_receipt.json',receipt)
    print(f'H{h} {year}: train{len(tr)} val{len(q["validation"])} test{len(q["test"])} seconds{receipt["seconds"]:.2f}',flush=True)
    return pd.concat(frames,ignore_index=True),receipt

def point(q,signal_col):
    selected=q[q[signal_col]];cells=q.groupby('corridor').agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
    s=selected.join(cells,on='corridor');coverage=[];legacy=[]
    for _,g in q.groupby('corridor'):
        counts=g.groupby(g.date.dt.to_period('W-SUN'))[signal_col].sum();coverage.append(float(counts.between(1,2).mean()))
        legacy.append(core.cadence_diagnostics(g,g.index[g[signal_col]])['weeks_with_1_to_2_signals_share'])
    return {'rows':len(q),'dates':q.date.nunique(),'corridors':q.corridor.nunique(),'contacts':len(s),'hit_rate':float(s.target.mean()) if len(s) else None,'raw_base_hit':float(q.target.mean()),'lift_standardized':float(s.target.sum()/s.base_hit.sum()) if len(s) else None,'forward_delta_bps':float((s.forward_bps-s.base_forward).mean()) if len(s) else None,'mean_all_observed_week_coverage':float(np.mean(coverage)),'mean_legacy_full_week_coverage':float(np.mean(legacy)),'brier':float(((q.probability-q.target)**2).mean())}

def parity(pred):
    rows=[]
    for h in (3,5):
        historical=pd.read_csv(ROOT/f'research_v3/models/basis_train_120m_h{h}_predictions.csv.gz',parse_dates=['date'])
        for year in (2024,2025,2026):
            actual=pred[pred.train_horizon.eq(h)&pred.fold_test_year.eq(year)&pred.split.eq('test')].sort_values(['date','corridor']).reset_index(drop=True)
            expected=historical[historical.fold_test_year.eq(year)].sort_values(['date','corridor']).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[['date','corridor']],expected[['date','corridor']])
            errors={c:float(np.max(np.abs(actual[c].to_numpy()-expected[c].to_numpy()))) for c in OUTCOME+['raw_probability','probability']}
            mismatch={c:int((actual[c]!=expected[c]).sum()) for c in ['candidate_signal','signal']}
            assert max(errors.values())<1e-9,(h,year,errors)
            assert not any(mismatch.values()),(h,year,mismatch)
            rows.append({'train_horizon':h,'year':year,'rows':len(actual),'status':'PASS','maximum_errors':errors,'signal_mismatches':mismatch})
    save(HERE/'historical_parity.json',{'status':'PASS','checks':rows});return rows

def main():
    HERE.mkdir(exist_ok=True,parents=True);(HERE/'checkpoints').mkdir(exist_ok=True)
    inputs=[SOURCE,Path(old.__file__),Path(core.__file__),*[ROOT/f'research_v3/models/basis_train_120m_h{h}_predictions.csv.gz' for h in (3,5)]]
    protocol={'status':'fixed_recipe_reproduction','matrix':{'years':[2024,2025,2026],'separate_train_horizons':[3,5]},'spec':asdict(SPEC),'features':FEATURES,'calibration':'Original V3 pooled5 monotone Platt on mature prior12months. Positive slope and prior Brier-improvement fallback exactly unchanged. No per-corridor recalibration.','strict05':'Threshold0.5 on original pooled-calibrated probability; cooldown3 effective CBR observations; pre-cutoff state replay at same0.5; candidate and shared portfolio exported separately.','historical_policy':'Original per-corridor past-validation frequency threshold and shared portfolio selection retained for exact parity.','history':'Past12months including immature tail; all future outcomes in immature rows blank. Only scores and observed context usable for policy replay.','scope':'Separate annual fits, train horizon equals evaluated NOW horizon; rescore of frozenH5 policy atH3 labelled separately. No hyperparameter/model selection in this reproduction.','code_sha256':sha(__file__),'inputs':{str(f.relative_to(ROOT)):sha(f) for f in inputs},'python':platform.python_version(),'sklearn':sklearn.__version__}
    save(HERE/'protocol.json',protocol)
    raw=pd.read_pickle(SOURCE);frames=[];receipts=[]
    with threadpool_limits(limits=1):
        for h in (3,5):
            panel=target_panel(raw,h)
            for year in (2024,2025,2026):
                p,r=run(panel,h,year);frames.append(p);receipts.append(r)
    allp=pd.concat(frames,ignore_index=True);allp.to_csv(HERE/'raw_predictions.csv.gz',index=False)
    save(HERE/'model_receipts.json',receipts)
    parity(allp)
    metrics=[]
    for (h,year),g in allp[allp.split.eq('test')].groupby(['train_horizon','fold_test_year']):
        for scope in ['all',*sorted(g.corridor.unique())]:
            q=g if scope=='all' else g[g.corridor.eq(scope)]
            for policy_col in ['candidate_signal','strict05_candidate_signal','signal','strict05_signal']:
                metrics.append({'train_horizon':h,'year':year,'scope':scope,'policy':policy_col,**point(q,policy_col)})
    pd.DataFrame(metrics).to_csv(HERE/'original_metrics.csv',index=False)
    save(HERE/'completion.json',{'status':'complete','fits':6,'rows':len(allp),'source_files_unchanged':all(sha(ROOT/k)==v for k,v in protocol['inputs'].items()),'code_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol.json'),'predictions_sha256':sha(HERE/'raw_predictions.csv.gz')})
    print('RAW_READY',HERE/'raw_predictions.csv.gz',flush=True)
if __name__=='__main__':main()
