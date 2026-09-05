"""Matched KZT-only learner comparisons, fixed before backtests."""
from pathlib import Path
import os,sys,pickle,json,shutil,warnings
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as n
HERE=Path(__file__).resolve().parent;OUT=HERE/'local';OUT.mkdir(exist_ok=True)

def encoded(model,frame):
    x,c=model.encode(frame,model.pre)
    return np.column_stack([x.numpy(),np.eye(5)[c.numpy().ravel()]])

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    views,bankcols=pickle.loads((ROOT/'research_v4/final_sprint/views.pkl').read_bytes())
    features={'base15':n.e.oxr.BASE,'full33':n.e.oxr.BASE+n.e.oxr.BASIS+n.e.oxr.COVER+bankcols+n.TREASURY_LAG7_FEATURES}
    allrows=[];receipts=[];newfits=0;reuse=0
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        for year in (2023,2024,2025):
            cutoff=pd.Timestamp(year,1,1)
            tr,va,te,hist=n.split(views,dict(scope='kzt',months=120,seed_index=2),cutoff)
            for family,cols in features.items():
                for seed in ([2] if family=='base15' else [0,1,2]):
                    name=f'kzt_tabm_{family}_s{seed}';dest=OUT/(name+'_'+str(year));model=n.Neural(cols,n.SEED+seed*100)
                    old=ROOT/'research_v4/final_sprint/tabm/checkpoints'/f'tabm_periodic_kzt_120m_s{seed}_{year}-01-01'
                    if family=='full33' and year==2025:
                        model.load(old);meta=json.loads((old/'model.json').read_text());reuse+=1
                        assert meta['features']==cols and meta['train_rows']==len(tr)
                        dest.mkdir(exist_ok=True)
                        for f in ('weights.pt','preprocess.joblib','model.json'):shutil.copy2(old/f,dest/f)
                        prior=joblib.load(old/'raw_predictions.joblib')
                        assert np.max(np.abs(model.predict(te)-prior['normal'].raw_probability.to_numpy()))<2e-7
                    else:meta=model.fit(tr,dest);newfits+=1
                    source=dict(model_id=name,year=year,scope='kzt',family=family,seed=seed,features=cols,train_rows=len(tr),train_min=tr.date.min(),train_max=tr.date.max(),train_label_max=tr.label_available_date.max(),cal_start=cutoff-pd.DateOffset(years=1),validation_min=va.date.min(),validation_max=va.date.max(),validation_label_max=va.label_available_date.max(),test_rows=len(te),train_fingerprint=n.e.fp(tr[['date','corridor',*cols]]),test_keys_fingerprint=n.e.fp(te[['date','corridor','target']]),preprocessor_sha256=n.sha(dest/'preprocess.joblib'),reused=bool(year==2025 and family=='full33'))
                    receipts.append(source)
                    blocks={'validation':va,'history':hist,'test':te}
                    for split,f in blocks.items():
                        q=f[n.KEEP].copy();q['raw_probability']=model.predict(f)
                        if split=='history':q.loc[q.label_available_date.ge(cutoff)|q.label_available_date.isna(),['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
                        q['split']=split;q['config_id']=name;q['cutoff']=str(cutoff.date());q['fold_test_year']=year;allrows.append(q)
                    if seed==2:
                        classifier=HistGradientBoostingClassifier(max_iter=120,max_depth=2,learning_rate=.05,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=n.SEED)
                        classifier.fit(encoded(model,tr),tr.target.astype(int));newfits+=1
                        hid=f'kzt_hgb_{family}';hd=OUT/(hid+'_'+str(year));hd.mkdir(exist_ok=True);joblib.dump(classifier,hd/'classifier.joblib');shutil.copy2(dest/'preprocess.joblib',hd/'preprocess.joblib')
                        receipts.append(dict(source,model_id=hid,seed=n.SEED,reused=False,architecture='HGBonidenticalencodedmatrix',preprocessor_sha256=n.sha(hd/'preprocess.joblib')))
                        for split,f in blocks.items():
                            q=f[n.KEEP].copy();q['raw_probability']=classifier.predict_proba(encoded(model,f))[:,1]
                            if split=='history':q.loc[q.label_available_date.ge(cutoff)|q.label_available_date.isna(),['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
                            q['split']=split;q['config_id']=hid;q['cutoff']=str(cutoff.date());q['fold_test_year']=year;allrows.append(q)
                    n.save(OUT/'receipts.json',receipts);print(name,year,'done',flush=True)
    pred=pd.concat(allrows,ignore_index=True)
    ensembles=[]
    for (year,split),g in pred[pred.config_id.str.match(r'kzt_tabm_full33_s[012]$')].groupby(['fold_test_year','split']):
        arrays=[x.sort_values(['date','corridor']).reset_index(drop=True) for _,x in g.groupby('config_id')]
        for a in arrays[1:]:pd.testing.assert_frame_equal(arrays[0][n.KEEP],a[n.KEEP],check_exact=True)
        q=arrays[0].copy();q['raw_probability']=np.mean([a.raw_probability.to_numpy() for a in arrays],axis=0);q['config_id']='kzt_tabm_full33_ensemble3';ensembles.append(q)
    pd.concat([pred,*ensembles],ignore_index=True).to_csv(OUT/'raw_predictions.csv.gz',index=False)
    n.save(OUT/'completion.json',dict(status='complete',new_final_fits=newfits,reused_final_neural_fits=reuse,new_neural_inner_fits=9,model_year_pairs=18,ensemble_year_pairs=3))
if __name__=='__main__':main()
