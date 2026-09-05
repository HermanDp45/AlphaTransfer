"""Matched fixed-configuration pooled TabM/HGB factorial, 2023–2025 only."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib,json,pickle,time,warnings
import joblib,numpy as np,pandas as pd,torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as neural
from research_v3.models import experiment as temporal
HERE=Path(__file__).resolve().parent;OUT=HERE/'output';CHECKPOINTS=HERE/'checkpoints'
SOURCE=ROOT/'research_v4/final_sprint/views.pkl'
FEATURE_FILE=ROOT/'research_v4/final_sprint/tabm/features.json'
FEATURES_FULL=json.loads(FEATURE_FILE.read_text());FEATURES_BASE=FEATURES_FULL[:15]
FEATURE_SETS={'base15':FEATURES_BASE,'full33':FEATURES_FULL}
KEEP=neural.KEEP;SEED=20261105
HGB_CONFIG=dict(max_iter=120,max_depth=2,learning_rate=.05,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=SEED)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def fingerprint(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).to_numpy().tobytes()).hexdigest()
def cid(architecture,feature_set):return f'arch_pooled_{architecture}_{feature_set}'
def initialize():
    OUT.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=dict(created_unix=time.time(),matrix=dict(years=[2023,2024,2025],architectures=['tabm','hgb'],feature_sets=FEATURE_SETS,training_scope='all five corridors pooled'),neural_seed=SEED,neural_refit_seed=SEED+1,hgb_config=HGB_CONFIG,neural_architecture=neural.ARCH,numerical_embeddings=neural.EMBED,neural_optimizer=neural.OPT,neural_training='Existing immutable Neural class: max100epochs; last63training dates for inner validation; actual-maturity purge; patience15; full outer refit at chosen epochs.',outer_split='120months train ending before12months calibration; annual Jan1 cutoff; actual label maturity strictly before next split; test last5observations purged. No2026 fitting, selection, or evaluation.',preprocessing='Per-year/feature-set train-only median imputation plus quantile-to-normal and all-feature missing indicators. Shared fitted preprocessor from the corresponding TabM. HGB receives the exact same float32 numeric inputs, then five corridor one-hot columns; TabM receives same numeric inputs and same corridor categories.',outputs='Raw probabilities only, validation/history/test; history outcomes not matured at cutoff masked. Parent applies identical per-corridor calibration and policies to both architectures.',interpretation='Matched fixed configurations, not a universal architecture ranking or a matched HPO budget. TabM epoch selection versus fixed120HGB trees is explicitly part of fixed recipes.',inputs={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,FEATURE_FILE,Path(neural.__file__),Path(temporal.__file__)]},code_sha256=sha(__file__))
    if not (HERE/'protocol.json').exists():save(HERE/'protocol.json',protocol)
    (HERE/'PROTOCOL.md').write_text('# Сопоставимые фиксированные конфигурации\n\nДо расчёта результатов зафиксирована матрица: пять коридоров вместе; 2023, 2024 и 2025 годы; BASE15/FULL33; TabM/HGB. Всего шесть обучений каждого семейства. Окно обучения — 120 месяцев, калибровки — 12 месяцев. Реальные даты созревания меток проверяются на каждой границе.\n\nTabM использует seed 20261105 и неизменный обучающий класс предыдущего эксперимента; число эпох определяется внутри обучения. HGB: 120 деревьев, глубина 2, шаг 0,05, минимум 40 объектов в листе, L2=2, без early stopping. Квантильное преобразование, импутация и индикаторы пропусков полностью общие. HGB получает те же float32-входы и one-hot коридора.\n\nЗдесь сохраняются только сырые вероятности. Калибровка по коридорам и политики будут одинаково применены основным агентом. Никакого отбора по 2026 году. Результат относится к этим фиксированным рецептам, а не к преимуществу архитектуры после одинакового поиска гиперпараметров.\n')
def split(panel,year):
    cutoff=pd.Timestamp(year,1,1);end=pd.Timestamp(year+1,1,1);cs=cutoff-pd.DateOffset(years=1)
    tr,va,te=temporal.temporal_split(panel,5,cutoff,end,temporal.Spec('architecture_pooled',months=120,validation_months=12,extended=True))
    history=panel[panel.date.ge(cs)&panel.date.lt(cutoff)].copy()
    assert tr.label_available_date.max()<cs and va.label_available_date.max()<cutoff
    assert te.label_available_date.max()<end
    assert tr.corridor.nunique()==va.corridor.nunique()==te.corridor.nunique()==5
    return dict(train=tr,validation=va,history=history,test=te)
def encoded(neural_model,frame):
    x,c=neural_model.encode(frame,neural_model.pre)
    numbers=x.numpy();categories=c.numpy()[:,0];onehot=np.eye(5,dtype=np.float32)[categories]
    result=np.concatenate([numbers,onehot],axis=1)
    assert np.array_equal(result[:,:numbers.shape[1]],numbers)
    assert result.dtype==np.float32 and np.isfinite(result).all()
    return result
def export(frame,raw,architecture,feature_set,year,split_name):
    q=frame[KEEP].copy();q['raw_probability']=np.asarray(raw,float)
    q['config_id']=cid(architecture,feature_set);q['architecture']=architecture;q['feature_set']=feature_set
    q['training_scope']='pooled';q['cutoff']=f'{year}-01-01';q['fold_test_year']=year;q['split']=split_name
    if split_name=='history':
        immature=q.label_available_date.ge(pd.Timestamp(year,1,1))|q.label_available_date.isna()
        q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
    return q
def run(panel,year,feature_set):
    started=time.monotonic();features=FEATURE_SETS[feature_set];parts=split(panel,year);tr=parts['train']
    stem=f'{feature_set}_{year}';nn_dir=CHECKPOINTS/f'tabm_{stem}';tree_dir=CHECKPOINTS/f'hgb_{stem}';tree_dir.mkdir(exist_ok=True)
    network=neural.Neural(features,SEED)
    if (nn_dir/'model.json').exists():network.load(nn_dir);nnmeta=json.loads((nn_dir/'model.json').read_text())
    else:nnmeta=network.fit(tr,nn_dir)
    trainx=encoded(network,tr);tree=HistGradientBoostingClassifier(**HGB_CONFIG)
    t0=time.monotonic();tree.fit(trainx,tr.target.astype(int));tree_seconds=time.monotonic()-t0
    bundle=dict(classifier=tree,features=features,preprocessor=network.pre,category_map=neural.CATS,append_all_missing_indicators=True,cast_numeric='float32',append_corridor_one_hot=5,config=HGB_CONFIG)
    joblib.dump(bundle,tree_dir/'model.joblib')
    frames=[];predictive=[]
    for split_name in ('validation','history','test'):
        q=parts[split_name]
        for arch,raw in [('tabm',network.predict(q)),('hgb',tree.predict_proba(encoded(network,q))[:,1])]:
            assert np.isfinite(raw).all() and ((raw>=0)&(raw<=1)).all()
            frame=export(q,raw,arch,feature_set,year,split_name);frames.append(frame)
            if split_name!='history':
                predictive.append(dict(config_id=cid(arch,feature_set),year=year,split=split_name,rows=len(q),raw_brier=float(np.mean((raw-q.target)**2))))
    predictions=pd.concat(frames,ignore_index=True);predictions.to_csv(OUT/f'{stem}_raw_predictions.csv.gz',index=False)
    receipt=dict(year=year,feature_set=feature_set,features=features,train_rows=len(tr),train_start=str(tr.date.min()),train_end=str(tr.date.max()),train_latest_label=str(tr.label_available_date.max()),validation_start=str(parts['validation'].date.min()),validation_end=str(parts['validation'].date.max()),validation_latest_label=str(parts['validation'].label_available_date.max()),test_start=str(parts['test'].date.min()),test_end=str(parts['test'].date.max()),test_latest_label=str(parts['test'].label_available_date.max()),test_rows=len(parts['test']),history_rows=len(parts['history']),training_feature_sha256=fingerprint(tr[['date','corridor',*features]]),training_encoded_sha256=hashlib.sha256(trainx.tobytes()).hexdigest(),encoded_numerical_columns=2*len(features),encoded_total_hgb_columns=trainx.shape[1],hgb_config=HGB_CONFIG,hgb_seconds=tree_seconds,neural=nnmeta,shared_preprocessor_sha256=sha(nn_dir/'preprocess.joblib'),hgb_checkpoint_sha256=sha(tree_dir/'model.joblib'),raw_predictions_sha256=sha(OUT/f'{stem}_raw_predictions.csv.gz'),source_sha256=sha(SOURCE),protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(__file__),seconds=time.monotonic()-started)
    save(OUT/f'{stem}_receipt.json',receipt);save(tree_dir/'receipt.json',receipt);save(nn_dir/'split_receipt.json',receipt)
    print(year,feature_set,'NN epochs',nnmeta['selected_epochs'],'NNseconds',round(nnmeta['fit_seconds'],2),'HGBseconds',round(tree_seconds,2),'testrows',len(parts['test']),flush=True)
    return predictions,predictive
def main():
    initialize();views,_=pickle.loads(SOURCE.read_bytes());panel=views['2010-01-01',24,1]
    assert len(FEATURES_BASE)==15 and len(FEATURES_FULL)==33
    frames=[];scores=[]
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        for year in (2023,2024,2025):
            for fs in FEATURE_SETS:
                p,m=run(panel,year,fs);frames.append(p);scores.extend(m)
    p=pd.concat(frames,ignore_index=True);p.to_csv(HERE/'raw_predictions.csv.gz',index=False)
    pd.DataFrame(scores).to_csv(HERE/'raw_score_diagnostic.csv',index=False)
    save(HERE/'completion.json',dict(status='complete',neural_fits=6,hgb_fits=6,temporary_inner_neural_fits=6,rows=len(p),years=[2023,2024,2025],raw_prediction_sha256=sha(HERE/'raw_predictions.csv.gz'),code_sha256=sha(__file__),protocol_sha256=sha(HERE/'protocol.json')))
if __name__=='__main__':main()
