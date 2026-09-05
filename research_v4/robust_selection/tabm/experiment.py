"""Frozen TabM recipes, separate H3/H5 labels, annual rolling fits."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json,pickle,shutil,time,warnings
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as n
from research_v3.models import experiment as old
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';CKPT=HERE/'checkpoints'
SOURCE=ROOT/'research_v4/final_sprint/views.pkl'
FEATURE_FILE=ROOT/'research_v4/final_sprint/tabm/features.json'
FEATURES=json.loads(FEATURE_FILE.read_text());SEED=20261105;KEEP=n.KEEP
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def fp(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()
def stem(scope,h,year):return f'tabm_{scope}_h{h}_{year}'
def initialize():
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=dict(created_unix=time.time(),scopes=['kzt','pooled'],training_horizons=[3,5],years=[2024,2025,2026],feature_names=FEATURES,seed=SEED,refit_seed=SEED+1,architecture=n.ARCH,embeddings=n.EMBED,optimizer=n.OPT,train_months=120,calibration_months=12,target='Recompute target, forward_bps, symmetric_bps, regret_bps with core.add_target(panel,H), and label_available_date=groupby(corridor).date.shift(-H). No H5 labels reused for H3.',maturity='Actual next-H observation strictly before next split; temporal_split uses the same H. Neural inner split uses supplied actual label_available_date.',preprocessing='Frozen Neural class: train-only median+quantile normalization, all-feature missing flags, numerical PeriodicEmbeddings, native corridor one-hot. No hyperparameter or seed selection.',reuse='H5 only after exact full training-row/features/labels/maturity parity and saved seed/features/preprocessor/checkpoint hashes. H3 always newly fitted. Reused checkpoints copied into this new directory.',evaluation='Only raw validation/history/test outputs. Parent applies calibrated probabilities and matching policy. 2026 previously inspected; no holdout or no-selection-bias claim.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,FEATURE_FILE,Path(n.__file__),Path(old.__file__),Path(n.core.__file__)]},code_sha256=sha(__file__))
    if not (HERE/'protocol.json').exists():save(HERE/'protocol.json',protocol)
def targeted(panel,h):
    q=n.core.add_target(panel,h);q['label_available_date']=q.groupby('corridor').date.shift(-h)
    q['train_horizon']=h
    return q
def split(panel,scope,h,year):
    cutoff=pd.Timestamp(year,1,1);end=pd.Timestamp(year+1,1,1);cs=cutoff-pd.DateOffset(months=12)
    tr,va,te=old.temporal_split(panel,h,cutoff,end,old.Spec(stem(scope,h,year),months=120,validation_months=12,extended=True))
    hi=panel[panel.date.ge(cs)&panel.date.lt(cutoff)].copy()
    if scope=='kzt':tr,va,te,hi=[x[x.corridor.eq('KZT')].copy() for x in (tr,va,te,hi)]
    assert tr.label_available_date.max()<cs and va.label_available_date.max()<cutoff
    assert te.label_available_date.notna().all()
    if end<=panel.date.max():assert te.label_available_date.max()<end
    return dict(train=tr,validation=va,history=hi,test=te)
def candidate(scope,year):
    if scope=='pooled' and year in (2024,2025):
        base=ROOT/'research_v4/architecture_2023_2025/pooled'
        return base/'checkpoints'/f'tabm_full33_{year}',base/'output'/f'full33_{year}_receipt.json','training_feature_sha256','views'
    if scope=='kzt' and year==2024:
        base=ROOT/'research_v4/architecture_2023_2025/local'
        return base/f'kzt_tabm_full33_s2_{year}',base/'receipts.json','train_fingerprint','views'
    if scope=='kzt' and year in (2025,2026):
        base=ROOT/'research_v4/final_sprint/tabm/checkpoints'/f'tabm_periodic_kzt_120m_s2_{year}-01-01'
        return base,base/'split.json','features_fingerprint','original_build'
    return None
def reuse_model(model,dest,parts,scope,year,panel,legacy_views):
    item=candidate(scope,year)
    if item is None:return None
    source,receipt_path,key,kind=item
    meta=json.loads((source/'model.json').read_text());receipt=json.loads(receipt_path.read_text())
    if isinstance(receipt,list):receipt=next(r for r in receipt if r['model_id']=='kzt_tabm_full33_s2' and r['year']==year)
    assert meta['features']==FEATURES and meta['seed']==SEED and meta['refit_seed']==SEED+1
    assert meta['architecture']==n.ARCH and meta['numerical_embeddings']==n.EMBED
    tr=parts['train'];assert meta['train_rows']==len(tr)
    assert receipt[key]==fp(tr[['date','corridor',*FEATURES]])
    prior_panel=legacy_views['2010-01-01',24,1] if kind=='original_build' else panel
    prior=split(prior_panel,scope,5,year)
    # Includes labels and their maturity, not just features and row counts.
    columns=['date','corridor',*FEATURES,'target','label_available_date','forward_bps','symmetric_bps','regret_bps']
    pd.testing.assert_frame_equal(tr[columns],prior['train'][columns],check_exact=True)
    assert meta['weights_sha256']==sha(source/'weights.pt')
    assert meta['preprocessor_sha256']==sha(source/'preprocess.joblib')
    dest.mkdir(parents=True,exist_ok=True)
    for file in ('weights.pt','preprocess.joblib','model.json','inner_epochs.csv','refit_epochs.csv'):
        if (source/file).exists():shutil.copy2(source/file,dest/file)
    model.load(dest)
    independent=model.preprocessor(tr)
    np.testing.assert_array_equal(independent.named_steps['impute'].statistics_,model.pre.named_steps['impute'].statistics_)
    np.testing.assert_array_equal(independent.named_steps['gaussian'].quantiles_,model.pre.named_steps['gaussian'].quantiles_)
    return dict(source=str(source.relative_to(ROOT)),source_receipt=str(receipt_path.relative_to(ROOT)),source_receipt_sha256=sha(receipt_path),full_train_fingerprint=fp(tr[columns]),prior_full_train_fingerprint=fp(prior['train'][columns]),source_weights_sha256=sha(source/'weights.pt'),source_preprocessor_sha256=sha(source/'preprocess.joblib'),status='verified_exact_training_and_preprocessing')
def run(panels,base,legacy_views,scope,h,year):
    clock=time.monotonic();parts=split(panels[h],scope,h,year);tr=parts['train'];name=stem(scope,h,year);dest=CKPT/name
    model=n.Neural(FEATURES,SEED);reuse=None
    if (dest/'receipt.json').exists():
        previous=json.loads((dest/'receipt.json').read_text())
        assert previous['full_train_fingerprint']==fp(tr[['date','corridor',*FEATURES,'target','label_available_date']])
        assert previous['source_sha256']==sha(SOURCE);model.load(dest);reuse=previous['reuse'];meta=json.loads((dest/'model.json').read_text())
    else:
        if h==5:reuse=reuse_model(model,dest,parts,scope,year,base,legacy_views)
        meta=json.loads((dest/'model.json').read_text()) if reuse else model.fit(tr,dest)
    frames=[];raw_scores=[]
    for split_name in ('validation','history','test'):
        f=parts[split_name];raw=model.predict(f);q=f[KEEP].copy();q['raw_probability']=raw
        if split_name=='history':
            immature=q.label_available_date.ge(pd.Timestamp(year,1,1))|q.label_available_date.isna()
            q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
        q['config_id']='tabm_'+scope;q['train_horizon']=h;q['cutoff']=f'{year}-01-01';q['fold_test_year']=year;q['split']=split_name
        frames.append(q)
        if split_name!='history':raw_scores.append(dict(config_id='tabm_'+scope,train_horizon=h,year=year,split=split_name,rows=len(f),raw_brier=float(np.mean((raw-f.target)**2))))
    result=pd.concat(frames,ignore_index=True);result.to_csv(OUT/(name+'.csv.gz'),index=False)
    receipt=dict(config_id='tabm_'+scope,train_horizon=h,year=year,cutoff=f'{year}-01-01',features=FEATURES,seed=SEED,train_rows=len(tr),train_min=tr.date.min(),train_max=tr.date.max(),train_latest_label=tr.label_available_date.max(),validation_rows=len(parts['validation']),validation_min=parts['validation'].date.min(),validation_max=parts['validation'].date.max(),validation_latest_label=parts['validation'].label_available_date.max(),test_rows=len(parts['test']),test_min=parts['test'].date.min(),test_max=parts['test'].date.max(),test_latest_label=parts['test'].label_available_date.max(),full_train_fingerprint=fp(tr[['date','corridor',*FEATURES,'target','label_available_date']]),selected_epochs=meta['selected_epochs'],inner_latest_label=meta['inner_label_max'],inner_validation_start=meta['inner_validation_min'],reuse=reuse,new_neural_fit=reuse is None,source_sha256=sha(SOURCE),weights_sha256=sha(dest/'weights.pt'),preprocessor_sha256=sha(dest/'preprocess.joblib'),predictions_sha256=sha(OUT/(name+'.csv.gz')),protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(__file__),seconds=time.monotonic()-clock)
    save(dest/'receipt.json',receipt);save(OUT/(name+'_receipt.json'),receipt)
    print(name,'reuse' if reuse else 'NEW FIT','epochs',meta['selected_epochs'],'rows',len(parts['test']),'seconds',round(receipt['seconds'],2),flush=True)
    return result,raw_scores,receipt
def main():
    initialize();views,_=pickle.loads(SOURCE.read_bytes());base=views['2010-01-01',24,1]
    panels={h:targeted(base,h) for h in (3,5)}
    # Original construction is read-only and used solely to prove H5 reuse parity.
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        legacy_views,_=n.build();allrows=[];scores=[];receipts=[]
        for h in (5,3):
            for year in (2024,2025,2026):
                for scope in ('kzt','pooled'):
                    p,s,r=run(panels,base,legacy_views,scope,h,year);allrows.append(p);scores.extend(s);receipts.append(r)
    out=pd.concat(allrows,ignore_index=True);out.to_csv(HERE/'raw_predictions.csv.gz',index=False)
    pd.DataFrame(scores).to_csv(HERE/'raw_score_diagnostic.csv',index=False);save(HERE/'receipts.json',receipts)
    save(HERE/'completion.json',dict(status='complete',model_year_horizon_cells=len(receipts),new_neural_fits=sum(r['new_neural_fit'] for r in receipts),reused_h5_checkpoints=sum(r['reuse'] is not None for r in receipts),raw_prediction_rows=len(out),raw_prediction_sha256=sha(HERE/'raw_predictions.csv.gz'),code_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json')))
if __name__=='__main__':main()
