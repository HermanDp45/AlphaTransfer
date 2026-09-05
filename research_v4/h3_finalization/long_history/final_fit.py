"""Final expanding KZT H3 fit on a verified latest snapshot; no calibration fit."""
from pathlib import Path
import sys,os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,pickle,time
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_v4.h3_finalization.long_history import experiment as e
def make_parts(panel,cutoff):
    p=e.r.targeted(panel,3);p=p[p.corridor.eq('KZT')].copy();cal_start=cutoff-pd.DateOffset(months=12)
    eligible=p[p.target.notna()&p.label_available_date.notna()]
    train=eligible[eligible.date.ge(pd.Timestamp('2010-01-01'))&eligible.date.lt(cal_start)&eligible.label_available_date.lt(cal_start)].copy()
    validation=eligible[eligible.date.ge(cal_start)&eligible.date.lt(cutoff)&eligible.label_available_date.lt(cutoff)].copy()
    history=p[p.date.ge(cal_start)&p.date.lt(cutoff)].copy()
    prior=p[p.date.lt(cal_start)];warmup=prior[prior.date.isin(sorted(prior.date.unique())[-63:])].copy()
    tail=history[history.target.isna()|history.label_available_date.isna()|history.label_available_date.ge(cutoff)].copy()
    assert train.date.min()==pd.Timestamp('2010-01-01') and train.label_available_date.max()<cal_start
    assert validation.label_available_date.max()<cutoff
    assert warmup.date.nunique()==63 and warmup.date.max()==prior.date.max()
    assert history.date.max()<cutoff
    return p,dict(train=train,validation=validation,history=history,warmup=warmup,tail=tail)
def output(f,raw,cutoff,split):
    q=f[e.KEEP+['ret1','rub_per_unit']].copy();q['raw_probability']=raw
    if split in ('warmup','tail'):q[['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
    if split=='history':
        immature=q.label_available_date.isna()|q.label_available_date.ge(cutoff)
        q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
    q['config_id']='tabm_kzt_fullhistory';q['train_horizon']=3;q['cutoff']=str(cutoff.date());q['fold_test_year']=cutoff.year;q['split']=split
    return q
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',type=Path,required=True);ap.add_argument('--panel-sha',required=True);ap.add_argument('--source-receipt',type=Path,required=True);ap.add_argument('--cutoff',default='2026-09-05');ap.add_argument('--output',type=Path,default=e.HERE.parent/'final_fit');args=ap.parse_args()
    assert e.sha(args.panel)==args.panel_sha
    panel=pd.read_pickle(args.panel);assert isinstance(panel,pd.DataFrame)
    assert set(e.FEATURES+['date','corridor','session_ordinal','rub_per_unit']).issubset(panel.columns)
    cutoff=pd.Timestamp(args.cutoff);out=args.output;dest=out/'model';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=dict(created_unix=time.time(),selection='Parent explicitly selected expanding history from2010 and H3 KZT rank80 after annual comparison.',cutoff=cutoff,calibration_start=cutoff-pd.DateOffset(months=12),train_start='2010-01-01',train_horizon=3,features=e.FEATURES,seed=e.SEED,refit_seed=e.SEED+1,panel_path=str(args.panel.resolve()),panel_sha256=args.panel_sha,source_receipt_path=str(args.source_receipt.resolve()),source_receipt_sha256=e.sha(args.source_receipt),split='Training date and actual H3 maturity before2025-09-05. Calibration dates in previous12months, actual maturity before2026-09-05. No training on calibration examples.',exports='validation/history/warmup/tail; no test. Tail predictions are current final-fit snapshot scores, not historical OOT predictions for those dates.',warmup='63 panel dates strictly before calibration start including purged train tail, labels all masked.',parent_scope='Calibrator, rank80 policy, runtime package and annotation integration are parent-owned.',code_sha256=e.sha(__file__),neural_class_sha256=e.sha(e.n.__file__))
    e.save(out/'protocol.json',protocol)
    p,parts=make_parts(panel,cutoff);model=e.n.Neural(e.FEATURES,e.SEED)
    with threadpool_limits(limits=2):
        meta=model.fit(parts['train'],dest)
        frames=[output(parts[k],model.predict(parts[k]),cutoff,k) for k in ('validation','history','warmup','tail')]
    raw=pd.concat(frames,ignore_index=True);raw.to_csv(out/'raw_predictions.csv.gz',index=False)
    receipt=dict(status='fitted',train_horizon=3,scope='KZT',cutoff=cutoff,train_start=parts['train'].date.min(),train_end=parts['train'].date.max(),train_rows=len(parts['train']),train_latest_label=parts['train'].label_available_date.max(),calibration_start=cutoff-pd.DateOffset(months=12),validation_start=parts['validation'].date.min(),validation_end=parts['validation'].date.max(),validation_latest_label=parts['validation'].label_available_date.max(),validation_rows=len(parts['validation']),history_rows=len(parts['history']),history_last=parts['history'].date.max(),warmup_dates=parts['warmup'].date.nunique(),warmup_last=parts['warmup'].date.max(),tail_rows=len(parts['tail']),tail_dates=parts['tail'].date.tolist(),latest_panel_date=p.date.max(),training_fingerprint=e.fp(parts['train'][['date','corridor',*e.FEATURES,'target','label_available_date']]),selected_epochs=meta['selected_epochs'],weights_sha256=e.sha(dest/'weights.pt'),preprocessor_sha256=e.sha(dest/'preprocess.joblib'),panel_sha256=args.panel_sha,raw_predictions_sha256=e.sha(out/'raw_predictions.csv.gz'),protocol_sha256=e.sha(out/'protocol.json'),code_sha256=e.sha(__file__))
    e.save(out/'receipt.json',receipt)
    # Reload official TabM state plus sklearn preprocessor; no project class pickle.
    reloaded=e.n.Neural(e.FEATURES,e.SEED);reloaded.load(dest);replay=[]
    for k in ('validation','history','warmup','tail'):
        q=raw[raw.split.eq(k)];error=float(np.max(np.abs(reloaded.predict(parts[k])-q.raw_probability))) if len(q) else 0.
        assert error==0.;replay.append(dict(split=k,rows=len(q),maximum_raw_difference=error))
    assert not raw.duplicated(['split','date','corridor']).any()
    independent=reloaded.preprocessor(parts['train'])
    np.testing.assert_array_equal(independent.named_steps['impute'].statistics_,reloaded.pre.named_steps['impute'].statistics_)
    np.testing.assert_array_equal(independent.named_steps['gaussian'].quantiles_,reloaded.pre.named_steps['gaussian'].quantiles_)
    e.save(out/'verification.json',dict(status='PASS',replay=replay,train_only_preprocessing_exact=True,train_labels_strictly_before_calibration=True,calibration_labels_strictly_before_cutoff=True,panel_sha256=args.panel_sha,weights_sha256=e.sha(dest/'weights.pt'),raw_predictions_sha256=e.sha(out/'raw_predictions.csv.gz')))
    print('FINAL H3 FIT PASS',json.dumps(receipt,default=str),flush=True)
if __name__=='__main__':main()
