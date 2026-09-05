"""Self-contained sklearn CLOSING_H3 endpoint models; no notification creation."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,time
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import QuantileTransformer,OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from research_v4.h3_finalization.long_history import experiment as e
from research_v4.h3_finalization.long_history.final_fit import make_parts
OUT=e.HERE.parent/'closing'
CONFIG=dict(max_iter=120,max_depth=2,learning_rate=.05,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=e.SEED)
def labels(panel):
    p=panel.copy();future=p.groupby('corridor').rub_per_unit.shift(-3)
    p['closing_target']=np.where(future.notna(),future.gt(p.rub_per_unit).astype(float),np.nan)
    p['closing_endpoint_bps']=(future/p.rub_per_unit-1)*10000
    return p
def new_model():
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('gaussian',QuantileTransformer(n_quantiles=128,output_distribution='normal',random_state=e.SEED))])
    pre=ColumnTransformer([('numeric',numeric,e.FEATURES),('missing',MissingIndicator(features='all'),e.FEATURES),('corridor',OneHotEncoder(categories=[list(e.n.CATS)],handle_unknown='ignore',sparse_output=False),['corridor'])],remainder='drop')
    return Pipeline([('preprocess',pre),('classifier',HistGradientBoostingClassifier(**CONFIG))])
def run(panel,parts,cutoff,name):
    start=time.monotonic();model=new_model();tr=parts['train'];dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    trainy=panel.loc[tr.index,'closing_target'];assert trainy.notna().all()
    model.fit(tr[e.FEATURES+['corridor']],trainy.astype(int))
    joblib.dump(model,dest/'model.joblib')
    rows=[]
    for split,f in parts.items():
        if split=='train':continue
        q=f[e.KEEP+['ret1','rub_per_unit']].copy();q['now_target']=q.target
        q['target']=panel.loc[f.index,'closing_target'];q['closing_target']=q.target;q['closing_endpoint_bps']=panel.loc[f.index,'closing_endpoint_bps']
        q['raw_probability']=model.predict_proba(f[e.FEATURES+['corridor']])[:,1]
        immature=q.label_available_date.isna()|q.label_available_date.ge(cutoff)
        if split in ('warmup','tail'):immature[:]=True
        if split in ('history','warmup','tail'):
            q.loc[immature,['target','closing_target','now_target','forward_bps','symmetric_bps','regret_bps','closing_endpoint_bps']]=np.nan
        q['config_id']='closing_hgb_kzt_fullhistory';q['train_horizon']=3;q['cutoff']=str(cutoff.date());q['fold_test_year']=cutoff.year;q['split']=split;q['fit_kind']=name
        rows.append(q)
    result=pd.concat(rows,ignore_index=True);result.to_csv(dest/'raw_predictions.csv.gz',index=False)
    loaded=joblib.load(dest/'model.joblib')
    assert type(loaded).__module__.startswith('sklearn.')
    maximum=0.
    for split,f in parts.items():
        if split=='train':continue
        a=loaded.predict_proba(f[e.FEATURES+['corridor']])[:,1];b=result[result.split.eq(split)].raw_probability.to_numpy()
        assert np.array_equal(a,b);maximum=max(maximum,float(np.max(np.abs(a-b))))
    receipt=dict(status='PASS',name=name,cutoff=cutoff,features=e.FEATURES,model_class='sklearn.pipeline.Pipeline',classifier_config=CONFIG,target='strict endpoint R[t+3]>R[t]; H3 CBR observations, not NOW survival',train_min=tr.date.min(),train_max=tr.date.max(),train_rows=len(tr),train_latest_label=tr.label_available_date.max(),validation_latest_label=parts['validation'].label_available_date.max(),training_labels_sha256=e.fp(pd.DataFrame({'target':trainy.to_numpy()})),checkpoint_sha256=e.sha(dest/'model.joblib'),raw_predictions_sha256=e.sha(dest/'raw_predictions.csv.gz'),maximum_replay_error=maximum,seconds=time.monotonic()-start,rule='Parent annotation: probability>=0.5 AND ret1>0 AND existing NOW signal; adds no new contacts')
    e.save(dest/'receipt.json',receipt);print('CLOSING',name,'rows',len(tr),'seconds',round(receipt['seconds'],2),flush=True)
    return result,receipt
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',type=Path,required=True);ap.add_argument('--panel-sha',required=True);args=ap.parse_args()
    assert e.sha(args.panel)==args.panel_sha;OUT.mkdir(parents=True,exist_ok=True)
    e.save(OUT/'protocol.json',dict(created_unix=time.time(),target='R[t+3]>R[t], strict endpoint closing event; separate from NOW H3 survival',scope='KZT only',history='from2010; annual2024/25/26 plus final2026-09-05; 12m calibration, actual H3 maturity',classifier_config=CONFIG,features=e.FEATURES,annotation='Fixed0.5 probability threshold AND ret1>0 AND NOW; parent owns common calibration and annotation evaluation',serialization='Only standard sklearn Pipeline/ColumnTransformer/imputer/quantile/indicator/one-hot/HGB; no project-class pickle',latest_panel_sha256=args.panel_sha,code_sha256=e.sha(__file__)))
    rows=[];receipts=[]
    with threadpool_limits(limits=1):
        panel=labels(e.load_panel())
        for year in (2024,2025,2026):
            p,receipt=run(panel,e.split(panel,year),pd.Timestamp(year,1,1),f'annual_{year}');rows.append(p);receipts.append(receipt)
        latest=pd.read_pickle(args.panel);p,parts=make_parts(latest,pd.Timestamp('2026-09-05'));p=labels(p)
        out,receipt=run(p,parts,pd.Timestamp('2026-09-05'),'final');rows.append(out);receipts.append(receipt)
    allrows=pd.concat(rows,ignore_index=True);allrows.to_csv(OUT/'raw_predictions.csv.gz',index=False)
    e.save(OUT/'receipts.json',receipts)
    e.save(OUT/'verification.json',dict(status='PASS',actual_hgb_fits=4,train_horizon=3,serialization='sklearn-only',all_checkpoint_replay_exact=True,raw_predictions_sha256=e.sha(OUT/'raw_predictions.csv.gz'),latest_panel_sha256=args.panel_sha,code_sha256=e.sha(__file__)))
if __name__=='__main__':main()
