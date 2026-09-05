"""Bounded CatBoost / train-only stability sprint. Import has no side effects."""
from pathlib import Path
import os, sys, hashlib, json, pickle, time, warnings
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from research_v4.oxr2010_bank.long_models import experiment as e
from research_v3.external_data.benchmark import augment_panel,TREASURY_LAG7_FEATURES
from research_v4.continuation.oxr.assess import point,paired
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
ANCHORS=['ret1','ret5','vol20','moex_cny_close_minus_fixing_same_session','halyk_rub_sell_official_basis']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n')
def specifications():
    out=[]
    for months,depth,l2 in ((24,2,20),(60,3,40),(120,4,80)):
        for subset in ('full','stable'):
            out.append(dict(name=f'catboost_treasury_halyk_{months}m_{subset}',months=months,
                            subset=subset,depth=depth if subset=='full' else 3,
                            l2_leaf_reg=l2 if subset=='full' else 50))
    return out

def choose_features(train,features):
    """Four chronological KZT train-block ranks, with no held-out labels."""
    k=train[train.corridor.eq('KZT')].sort_values('date')
    blocks=np.array_split(np.arange(len(k)),4)
    records=[]
    for feature in features:
        correlations=[];ranks=[]
        for b,indices in enumerate(blocks):
            q=k.iloc[indices][[feature,'target']].dropna()
            value=q[feature].corr(q.target,method='spearman') if len(q)>=40 and q[feature].nunique()>1 and q.target.nunique()>1 else 0.
            correlations.append(0. if not np.isfinite(value) else float(value))
        records.append(dict(feature=feature,**{f'block{i+1}_rho':v for i,v in enumerate(correlations)},
                            median_abs_rho=float(np.median(np.abs(correlations))),
                            blocks_positive=sum(v>0 for v in correlations),blocks_negative=sum(v<0 for v in correlations),
                            train_nonmissing=float(k[feature].notna().mean())))
    ranks=pd.DataFrame(records)
    # A feature must keep at least a modest rank in several distinct periods.
    block_ranks=pd.concat([ranks[f'block{i}_rho'].abs().rank(pct=True) for i in range(1,5)],axis=1)
    ranks['stability_score']=block_ranks.median(axis=1)-.25*block_ranks.std(axis=1)
    ranks=ranks.sort_values(['stability_score','median_abs_rho','feature'],ascending=[False,False,True])
    selected=ANCHORS+[f for f in ranks.feature if f not in ANCHORS][:9]
    ranks['selected']=ranks.feature.isin(selected)
    metadata=dict(method='Median absolute-Spearman percentile rank over four consecutive KZT train blocks, minus0.25 rank SD; five fixed anchors plus nine remaining features',
                  train_min=k.date.min(),train_max=k.date.max(),latest_label=k.label_available_date.max(),
                  blocks=[dict(rows=len(ix),start=k.iloc[ix].date.min(),end=k.iloc[ix].date.max()) for ix in blocks])
    return selected,ranks,metadata

def matrix(frame,features):
    x=frame[features+['corridor']].copy()
    for name in features:x[name]=pd.to_numeric(x[name],errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
    x['corridor']=x.corridor.astype(str)
    return x

def cadence_all(g):
    weeks=pd.period_range(g.date.min().to_period('W-SUN'),g.date.max().to_period('W-SUN'),freq='W-SUN')
    counts=g.loc[g.candidate_signal,'date'].dt.to_period('W-SUN').value_counts().reindex(weeks,fill_value=0)
    return dict(calendar_weeks=len(weeks),weeks_1_2_share=float(counts.between(1,2).mean()),zero_weeks=int(counts.eq(0).sum()),
                weeks_over2=int(counts.gt(2).sum()),signal_forward_bps=float(g.loc[g.candidate_signal,'forward_bps'].mean()))

def fit_one(views,bankcols,spec,cutoff):
    started=time.monotonic();cutoff=pd.Timestamp(cutoff);p=views['2010-01-01',24,1]
    full=e.oxr.BASE+bankcols+TREASURY_LAG7_FEATURES
    tr,va,te=e.old.temporal_split(p,5,cutoff,pd.Timestamp(cutoff.year+1,1,1),
                                e.old.Spec(spec['name'],months=spec['months'],extended=True))
    va=va[va.corridor.eq('KZT')];te=te[te.corridor.eq('KZT')]
    start=cutoff-pd.DateOffset(years=1)
    history=p[p.corridor.eq('KZT') & p.date.ge(start) & p.date.lt(cutoff)].copy()
    assert tr.label_available_date.max()<start and va.label_available_date.max()<cutoff
    chosen,ranks,selection=choose_features(tr,full)
    features=full if spec['subset']=='full' else chosen
    name=spec['name']+'_'+str(cutoff.date())
    ranks.to_csv(OUT/(name+'_feature_stability.csv'),index=False)
    params=dict(iterations=500,learning_rate=.04,depth=spec['depth'],l2_leaf_reg=spec['l2_leaf_reg'],
                loss_function='Logloss',bootstrap_type='Bernoulli',subsample=.8,rsm=.8,random_strength=1.,
                one_hot_max_size=5,has_time=True,thread_count=1,random_seed=20260905,
                allow_writing_files=False,verbose=False,nan_mode='Min')
    model=CatBoostClassifier(**params)
    weights=np.where(tr.corridor.eq('KZT'),4.,1.)
    model.fit(matrix(tr,features),tr.target.astype(int),cat_features=['corridor'],sample_weight=weights)
    vraw=model.predict_proba(matrix(va,features))[:,1]
    cal=e.core.fit_platt_calibrator(vraw,va.target)
    vp=e.core.apply_platt(cal,vraw);threshold,_,_=e.core.choose_frequency_threshold(va,vp)
    hraw=model.predict_proba(matrix(history,features))[:,1];hp=e.core.apply_platt(cal,hraw)
    chosen=e.core.select_per_corridor_with_cooldown(history,hp,threshold)
    state=e.core.corridor_selection_state(history,chosen)
    portfolio=e.core.selection_state(history,e.core.select_portfolio_from_candidates(history,hp,chosen))
    bundle=dict(model=model,calibrator=cal,threshold=threshold,initial_state=state,portfolio_state=portfolio,
                features=features,spec=spec,cutoff=str(cutoff.date()),params=params)
    checkpoint=OUT/(name+'.pkl');checkpoint.write_bytes(pickle.dumps(bundle))
    native=OUT/(name+'.cbm');model.save_model(str(native))
    predictions=[]
    for mode,lag in (('normal',1),('bank_delayed',2)):
        test=e.stress_view(views,dict(since='2010-01-01'),cutoff,24,lag).loc[te.index]
        pd.testing.assert_frame_equal(test[['date','corridor','target']],te[['date','corridor','target']],check_exact=True)
        raw=model.predict_proba(matrix(test,features))[:,1];prob=e.core.apply_platt(cal,raw)
        chosen=e.core.select_per_corridor_with_cooldown(test,prob,threshold,state)
        selected=e.core.select_portfolio_from_candidates(test,prob,chosen,portfolio)
        pred=test[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date']].copy()
        pred['raw_probability']=raw;pred['probability']=prob
        pred['candidate_signal']=pred.index.isin(chosen);pred['signal']=pred.index.isin(selected)
        pred['config_id']=spec['name'];pred['cutoff']=str(cutoff.date());pred['fold_test_year']=cutoff.year;pred['mode']=mode
        predictions.append(pred)
    pred=pd.concat(predictions,ignore_index=True);dest=OUT/(name+'.csv.gz');pred.to_csv(dest,index=False)
    records=[]
    for split,g,raw,prob in (('calibration',va,vraw,vp),('history',history,hraw,hp)):
        q=g[['date','corridor','session_ordinal','target','forward_bps','label_available_date']].copy()
        # Preserve the final purged decision dates but hide labels not yet mature at cutoff.
        q.loc[q.label_available_date.ge(cutoff)|q.label_available_date.isna(),['target','forward_bps']]=np.nan
        q['raw_probability']=raw;q['probability']=prob;q['config_id']=spec['name'];q['cutoff']=str(cutoff.date());q['split']=split
        q['fold_test_year']=cutoff.year;records.append(q)
    receipt=dict(spec=spec,cutoff=cutoff,catboost_version=catboost.__version__,params=params,features=features,
                 feature_selection=selection,train_rows=len(tr),train_kzt_rows=int(tr.corridor.eq('KZT').sum()),
                 sample_weight_KZT=4,other_corridor_weight=1,train_min=tr.date.min(),train_max=tr.date.max(),
                 train_latest_label=tr.label_available_date.max(),validation_rows=len(va),validation_min=va.date.min(),
                 validation_max=va.date.max(),validation_latest_label=va.label_available_date.max(),history_rows=len(history),
                 history_max=history.date.max(),test_rows=len(te),test_min=te.date.min(),test_max=te.date.max(),
                 train_fingerprint=e.fp(tr[['date','corridor',*features]]),train_bank_nonmissing=float(tr[bankcols].notna().mean().mean()),
                 train_treasury_nonmissing=float(tr[TREASURY_LAG7_FEATURES].notna().mean().mean()),
                 checkpoint_sha256=sha(checkpoint),native_model_sha256=sha(native),predictions_sha256=sha(dest),
                 seconds=time.monotonic()-started)
    save(OUT/(name+'.json'),receipt)
    print(name,'rows',len(te),'seconds',round(receipt['seconds'],2),flush=True)
    return pred,pd.concat(records,ignore_index=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(HERE/'calibration').mkdir(exist_ok=True)
    specs=specifications()
    sources=[e.SNAPSHOT,ROOT/'research_v3/models/panel_extended.pkl',ROOT/'research_v3/external_data/feature_panel.parquet',
             ROOT/'research_v4/liquidity/halyk_sell_daily.csv',Path(e.__file__)]
    save(HERE/'protocol.json',dict(created_unix=time.time(),specifications=specs,cutoffs=['2025-01-01','2026-01-01'],
          fits=12,model='CatBoost pooledfive corridors, fixed KZT weight4, local KZT calibration; no residual adapter',
          features='V3base+CNYbasis+six Treasury lag7+six Halyk; no OXR columns used',
          feature_selection='Four chronological train-only KZT blocks; stable14feature subset compared with full27; full depth varies by window so cross-window comparisons also vary capacity',
          development='2025 separately reported; past retrospective2026 model selection explicitly authorized, not pristine holdout',
          training='Strict label maturity, fixed500iterations, no eval-set stopping, previous12months KZT calibration',
          lag_stress='Bank1to2 calendar days applied after cutoff; fixed model/calibrator/threshold/normal history state',
          policy='Legacy prior-calibration policy here; root separately compares common cadence policies without2026threshold fitting',
          code_sha256=sha(__file__),inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},catboost_version=catboost.__version__))
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,bankcols=e.build_views();views={k:augment_panel(v) for k,v in views.items()}
        predictions=[];histories=[]
        for cutoff in ('2025-01-01','2026-01-01'):
            for spec in specs:
                p,h=fit_one(views,bankcols,spec,cutoff);predictions.append(p);histories.append(h)
    pred=pd.concat(predictions,ignore_index=True);hist=pd.concat(histories,ignore_index=True)
    pred.to_csv(HERE/'all_predictions.csv.gz',index=False)
    hist.to_csv(HERE/'calibration/history_predictions.csv.gz',index=False)
    hist[hist.split.eq('calibration')].to_csv(HERE/'calibration_predictions.csv.gz',index=False)
    hist[hist.split.eq('history')].to_csv(HERE/'history_predictions.csv.gz',index=False)
    rows=[]
    for (name,cutoff,mode),q in pred.groupby(['config_id','cutoff','mode']):
        rows.append(dict(config_id=name,cutoff=cutoff,mode=mode,scope='KZT',**point(q),**cadence_all(q)))
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False)
    baseline=pd.read_csv(ROOT/'research_v4/oxr2010_bank/long_models/treasury/all_predictions.csv.gz',parse_dates=['date'])
    baseline['date']=baseline.date.astype(pred.date.dtype)
    intervals=[]
    for (name,cutoff,mode),q in pred.groupby(['config_id','cutoff','mode']):
        a=baseline[baseline.config_id.eq('treasury_halyk_shrink_120m')&baseline.cutoff.eq(cutoff)&baseline['mode'].eq(mode)]
        for block in ('month',20):
            intervals.append(dict(config_id=name,cutoff=cutoff,mode=mode,baseline='treasury_halyk_shrink_120m',block=str(block),**paired(a,q,block)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    save(HERE/'completion.json',dict(status='complete',fits=12,output_rows=len(pred),history_and_calibration_rows=len(hist),
          code_sha256=sha(__file__),prediction_sha256=sha(HERE/'all_predictions.csv.gz'),
          history_sha256=sha(HERE/'calibration/history_predictions.csv.gz'),seconds='See per-fit receipts'))
    print('Completed12fits',flush=True)

def finish_saved_assessment():
    """Resume metrics after a timestamp-resolution equality failure; no refitting."""
    pred=pd.read_csv(HERE/'all_predictions.csv.gz',parse_dates=['date'])
    hist=pd.read_csv(HERE/'calibration/history_predictions.csv.gz',parse_dates=['date'])
    baseline=pd.read_csv(ROOT/'research_v4/oxr2010_bank/long_models/treasury/all_predictions.csv.gz',parse_dates=['date'])
    intervals=[]
    with threadpool_limits(limits=1):
        for (name,cutoff,mode),q in pred.groupby(['config_id','cutoff','mode']):
            a=baseline[baseline.config_id.eq('treasury_halyk_shrink_120m')&baseline.cutoff.eq(cutoff)&baseline['mode'].eq(mode)]
            for block in ('month',20):
                intervals.append(dict(config_id=name,cutoff=cutoff,mode=mode,baseline='treasury_halyk_shrink_120m',block=str(block),**paired(a,q,block)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    save(HERE/'completion.json',dict(status='complete',fits=12,output_rows=len(pred),history_and_calibration_rows=len(hist),
          code_sha256=sha(__file__),fit_engine_sha256=sha(HERE/'fit_engine_snapshot.py'),
          prediction_sha256=sha(HERE/'all_predictions.csv.gz'),history_sha256=sha(HERE/'calibration/history_predictions.csv.gz'),
          assessment_resume='Initial paired equality rejected ns-vs-us date dtype despite equal date keys; reloaded both frozen CSVs at same resolution. No model or prediction changed.'))
    print('Completed12fits; saved assessment complete',flush=True)

if __name__=='__main__':
    if '--assess-only' in sys.argv:finish_saved_assessment()
    else:main()
