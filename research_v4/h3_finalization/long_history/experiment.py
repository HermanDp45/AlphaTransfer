"""H3 KZT-only TabM: expanding2010 history, fixed annual recipes."""
from pathlib import Path
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,pickle,time,warnings
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.tabm import experiment as r
n=r.n;HERE=Path(__file__).resolve().parent;OUT=HERE/'output';CKPT=HERE/'checkpoints'
H=3;SEED=r.SEED;FEATURES=r.FEATURES;KEEP=r.KEEP;SOURCE=r.SOURCE;TRAIN_START=pd.Timestamp('2010-01-01')
sha=r.sha;save=r.save;fp=r.fp
def initialize():
    OUT.mkdir(parents=True,exist_ok=True);CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=dict(created_unix=time.time(),config_id='tabm_kzt_fullhistory',train_horizon=3,scope='KZT only',annual_years=[2024,2025,2026],train_start='2010-01-01',train_end='strictly before prior12month calibration; actual H3 maturity purge',features=FEATURES,seed=SEED,refit_seed=SEED+1,architecture=n.ARCH,embeddings=n.EMBED,optimizer=n.OPT,epoch_selection='Unchanged Neural class: final63training dates, actual maturity purge, max100epochs/patience15, full refit at selected epochs.',comparison='Only history span changes against existing120m H3 baseline; same validation/history/test/panel-warmup keys and targets. No H5 or pooled models.',selection='History choice belongs to parent after annual comparison; final fit will use separately supplied specification. No implicit final fit yet.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,r.FEATURE_FILE,Path(n.__file__),Path(r.__file__)]},code_sha256=sha(__file__))
    if not (HERE/'protocol.json').exists():save(HERE/'protocol.json',protocol)
def load_panel():
    views,_=pickle.loads(SOURCE.read_bytes());p=r.targeted(views['2010-01-01',24,1],H)
    return p[p.corridor.eq('KZT')].copy()
def split(panel,year):
    cutoff=pd.Timestamp(year,1,1);cal_start=cutoff-pd.DateOffset(months=12);end=pd.Timestamp(year+1,1,1)
    eligible=panel[panel.target.notna()]
    train=n.core.purge_tail(eligible[eligible.date.ge(TRAIN_START)&eligible.date.lt(cal_start)],H)
    validation=n.core.purge_tail(eligible[eligible.date.ge(cal_start)&eligible.date.lt(cutoff)],H)
    test=eligible[eligible.date.ge(cutoff)&eligible.date.lt(end)].copy()
    if end<=panel.date.max():test=n.core.purge_tail(test,H)
    history=panel[panel.date.ge(cal_start)&panel.date.lt(cutoff)].copy()
    prior=panel[panel.date.lt(cal_start)];dates=sorted(prior.date.unique())[-63:];warmup=prior[prior.date.isin(dates)].copy()
    assert train.date.min()==TRAIN_START and train.label_available_date.max()<cal_start
    assert validation.label_available_date.max()<cutoff and test.label_available_date.notna().all()
    baseline=r.split(panel,'kzt',H,year)
    for key,part in [('validation',validation),('history',history),('test',test)]:
        pd.testing.assert_frame_equal(part[KEEP+FEATURES],baseline[key][KEEP+FEATURES],check_exact=True)
    assert baseline['train'].index.isin(train.index).all()
    return dict(train=train,validation=validation,history=history,test=test,warmup=warmup)
def export(f,p,year,split):
    q=f[KEEP].copy();q['raw_probability']=p
    if split=='warmup':q[['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
    if split=='history':
        immature=q.label_available_date.ge(pd.Timestamp(year,1,1))|q.label_available_date.isna()
        q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
    q['config_id']='tabm_kzt_fullhistory';q['train_horizon']=H;q['cutoff']=f'{year}-01-01';q['fold_test_year']=year;q['split']=split
    return q
def run(panel,year):
    parts=split(panel,year);tr=parts['train'];dest=CKPT/f'tabm_kzt_fullhistory_h3_{year}';model=n.Neural(FEATURES,SEED)
    contract=fp(tr[['date','corridor',*FEATURES,'target','label_available_date']])
    if (dest/'receipt.json').exists():
        receipt=json.loads((dest/'receipt.json').read_text());assert receipt['training_fingerprint']==contract and receipt['source_sha256']==sha(SOURCE)
        model.load(dest);meta=json.loads((dest/'model.json').read_text())
    else:meta=model.fit(tr,dest)
    frames=[];scores=[]
    for key in ('validation','history','test','warmup'):
        raw=model.predict(parts[key]);frames.append(export(parts[key],raw,year,key))
        if key in ('validation','test'):scores.append(dict(year=year,split=key,rows=len(raw),raw_brier=float(np.mean((raw-parts[key].target)**2))))
    result=pd.concat(frames,ignore_index=True);result.to_csv(OUT/f'{year}_raw.csv.gz',index=False)
    base=r.split(panel,'kzt',H,year)['train']
    receipt=dict(config_id='tabm_kzt_fullhistory',train_horizon=H,year=year,train_start=tr.date.min(),train_end=tr.date.max(),train_rows=len(tr),baseline120m_rows=len(base),extra_train_rows=len(tr)-len(base),train_latest_label=tr.label_available_date.max(),calibration_start=pd.Timestamp(year-1,1,1),validation_min=parts['validation'].date.min(),validation_max=parts['validation'].date.max(),validation_latest_label=parts['validation'].label_available_date.max(),test_min=parts['test'].date.min(),test_max=parts['test'].date.max(),test_rows=len(parts['test']),warmup_dates=parts['warmup'].date.nunique(),warmup_last=parts['warmup'].date.max(),training_fingerprint=contract,weights_sha256=sha(dest/'weights.pt'),preprocessor_sha256=sha(dest/'preprocess.joblib'),source_sha256=sha(SOURCE),protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(__file__),selected_epochs=meta['selected_epochs'],fit_seconds=meta['fit_seconds'],feature_coverage={f:float(tr[f].notna().mean()) for f in FEATURES})
    save(dest/'receipt.json',receipt);save(OUT/f'{year}_receipt.json',receipt)
    print(year,'train',len(tr),'extra',len(tr)-len(base),'epochs',meta['selected_epochs'],'seconds',round(meta['fit_seconds'],2),flush=True)
    return result,scores,receipt
def main():
    initialize();panel=load_panel();frames=[];scores=[];receipts=[]
    availability=[]
    for feature in ['rub_per_unit',*FEATURES]:
        x=panel[panel[feature].notna()];availability.append(dict(feature=feature,first_nonmissing=x.date.min(),last_nonmissing=x.date.max(),coverage_2010=float(panel.loc[panel.date.dt.year.eq(2010),feature].notna().mean())))
    pd.DataFrame(availability).to_csv(HERE/'source_availability.csv',index=False)
    save(HERE/'panel_start.json',dict(corridor='KZT',first_panel_date=panel.date.min(),last_panel_date=panel.date.max(),rows=len(panel),rows_2010=int(panel.date.dt.year.eq(2010).sum()),first_actual_oxr=panel.loc[panel.oxr_available.eq(1),'date'].min(),description='Initial missing derived/source features retained with train-only imputation and missing flags; no fictitious complete feature history.'))
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        for year in (2024,2025,2026):
            p,s,receipt=run(panel,year);frames.append(p);scores.extend(s);receipts.append(receipt)
    p=pd.concat(frames,ignore_index=True);p[p.split.ne('warmup')].to_csv(HERE/'raw_predictions.csv.gz',index=False);p[p.split.eq('warmup')].to_csv(HERE/'warmup.csv.gz',index=False)
    pd.DataFrame(scores).to_csv(HERE/'raw_score_diagnostic.csv',index=False);save(HERE/'receipts.json',receipts)
    save(HERE/'completion.json',dict(status='annual_phase_complete',new_neural_fits=3,new_inner_fits=3,train_horizon=H,rows=int(p.split.ne('warmup').sum()),warmup_rows=int(p.split.eq('warmup').sum()),source_sha256=sha(SOURCE),raw_prediction_sha256=sha(HERE/'raw_predictions.csv.gz'),warmup_sha256=sha(HERE/'warmup.csv.gz'),awaiting='Parent-selected final-fit specification'))
if __name__=='__main__':main()
