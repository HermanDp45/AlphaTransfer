"""Verify horizon semantics, exact checkpoint replay, and raw/warmup contracts."""
from pathlib import Path
import sys,pickle,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.tabm import experiment as e
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    protocol=json.loads((e.HERE/'protocol.json').read_text());assert e.sha(e.__file__)==protocol['code_sha256']
    for file,digest in protocol['source_hashes'].items():assert e.sha(ROOT/file)==digest
    views,_=pickle.loads(e.SOURCE.read_bytes());base=views['2010-01-01',24,1];panels={h:e.targeted(base,h) for h in (3,5)}
    checks=[];label_checks=[]
    for h,p in panels.items():
        for c,g in p.groupby('corridor'):
            g=g.sort_values('date');rates=g.rub_per_unit.to_numpy(float)
            future=np.stack([rates[k:len(rates)-h+k] for k in range(1,h+1)],axis=1)
            current=rates[:-h];minimum=future.min(axis=1)
            target=(((minimum/current-1)*10000+1e-12)>=0).astype(float)
            forward=(future.mean(axis=1)/current-1)*10000
            regret=(current/np.minimum(current,minimum)-1)*10000
            np.testing.assert_array_equal(g.target.iloc[:-h].to_numpy(),target)
            np.testing.assert_allclose(g.forward_bps.iloc[:-h],forward,atol=1e-9,rtol=0)
            np.testing.assert_allclose(g.regret_bps.iloc[:-h],regret,atol=1e-9,rtol=0)
            symmetric=np.lib.stride_tricks.sliding_window_view(rates,2*h+1).mean(axis=1)
            symmetric=(symmetric/rates[h:-h]-1)*10000
            np.testing.assert_allclose(g.symmetric_bps.iloc[h:-h],symmetric,atol=1e-9,rtol=0)
            assert g.label_available_date.equals(g.date.shift(-h))
            assert g.target.iloc[-h:].isna().all()
            label_checks.append(dict(train_horizon=h,corridor=c,rows=len(g),target_and_utilities='PASS'))
    comparable=panels[3].target.notna()&panels[5].target.notna()
    assert (panels[3].loc[comparable,'target']>=panels[5].loc[comparable,'target']).all()
    assert (panels[3].loc[comparable,'target']!=panels[5].loc[comparable,'target']).any()
    checks.append(dict(name='Independent H3/H5 next-observation target, all utilities and actual maturity',status='PASS',corridor_horizon_checks=len(label_checks),h3_h5_label_disagreements=int((panels[3].loc[comparable,'target']!=panels[5].loc[comparable,'target']).sum())))
    raw=pd.read_csv(e.HERE/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'])
    warm=pd.read_csv(e.HERE/'warmup.csv.gz',parse_dates=['date','label_available_date'])
    keys=['train_horizon','config_id','fold_test_year','split','date','corridor']
    assert not raw.duplicated(keys).any() and not warm.duplicated(keys).any()
    assert not raw.duplicated(['train_horizon','config_id','split','date','corridor']).any()
    checks.append(dict(name='Raw and warmup keys unique',status='PASS',raw_rows=len(raw),warmup_rows=len(warm)))
    replay=[];receipts=json.loads((e.HERE/'receipts.json').read_text())
    with threadpool_limits(limits=2):
        for receipt in receipts:
            h=receipt['train_horizon'];year=receipt['year'];scope=receipt['config_id'].removeprefix('tabm_');parts=e.split(panels[h],scope,h,year)
            dest=e.CKPT/e.stem(scope,h,year);meta=json.loads((dest/'model.json').read_text())
            assert receipt['weights_sha256']==e.sha(dest/'weights.pt') and receipt['preprocessor_sha256']==e.sha(dest/'preprocess.joblib')
            assert meta['seed']==e.SEED and meta['features']==e.FEATURES
            assert pd.Timestamp(receipt['inner_latest_label'])<pd.Timestamp(receipt['inner_validation_start'])
            tr=parts['train'];assert e.fp(tr[['date','corridor',*e.FEATURES,'target','label_available_date']])==receipt['full_train_fingerprint']
            if h==3:assert receipt['new_neural_fit'] and receipt['reuse'] is None
            if receipt['reuse']:
                r=receipt['reuse'];assert r['full_train_fingerprint']==r['prior_full_train_fingerprint']
                assert e.sha(ROOT/r['source']/'weights.pt')==r['source_weights_sha256']==receipt['weights_sha256']
            model=e.n.Neural(e.FEATURES,e.SEED);model.load(dest)
            for split,f in parts.items():
                if split=='train':continue
                q=raw[raw.train_horizon.eq(h)&raw.config_id.eq(receipt['config_id'])&raw.fold_test_year.eq(year)&raw.split.eq(split)]
                q=q.sort_values(['date','corridor']).reset_index(drop=True);f=f.sort_values(['date','corridor'])
                pd.testing.assert_frame_equal(q[['date','corridor']],f[['date','corridor']].reset_index(drop=True),check_exact=True,check_dtype=False)
                predicted=model.predict(f);error=float(np.max(np.abs(predicted-q.raw_probability)))
                assert error<2e-7
                if split=='history':
                    immature=q.label_available_date.ge(pd.Timestamp(year,1,1))|q.label_available_date.isna()
                    assert q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all()
                else:
                    np.testing.assert_array_equal(q.target,f.target)
                    np.testing.assert_allclose(q.forward_bps,f.forward_bps,atol=1e-9,rtol=0)
                replay.append(dict(config_id=receipt['config_id'],train_horizon=h,year=year,split=split,rows=len(q),maximum_raw_probability_error=error))
            prior=panels[h][panels[h].date.lt(pd.Timestamp(year-1,1,1))]
            if scope=='kzt':prior=prior[prior.corridor.eq('KZT')]
            latest=sorted(prior.date.unique())[-63:];f=prior[prior.date.isin(latest)].sort_values(['date','corridor'])
            q=warm[warm.train_horizon.eq(h)&warm.config_id.eq(receipt['config_id'])&warm.fold_test_year.eq(year)].sort_values(['date','corridor']).reset_index(drop=True)
            pd.testing.assert_frame_equal(q[['date','corridor']],f[['date','corridor']].reset_index(drop=True),check_exact=True,check_dtype=False)
            assert q[['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all()
            assert q.date.nunique()==63 and q.date.max()==prior.date.max()
            assert int((~f.index.isin(tr.index)).sum())==h*(1 if scope=='kzt' else 5)
            error=float(np.max(np.abs(model.predict(f)-q.raw_probability)));assert error<2e-7
            replay.append(dict(config_id=receipt['config_id'],train_horizon=h,year=year,split='warmup',rows=len(q),maximum_raw_probability_error=error))
    pd.DataFrame(replay).to_csv(e.HERE/'checkpoint_replay.csv',index=False)
    pd.DataFrame(label_checks).to_csv(e.HERE/'target_semantic_checks.csv',index=False)
    checks.append(dict(name='All checkpoints/actual training bounds/inner boundaries and export probabilities replay',status='PASS',models=len(receipts),split_checks=len(replay),maximum_raw_probability_error=max(r['maximum_raw_probability_error'] for r in replay)))
    checks.append(dict(name='Warmup covers exactly63 PANEL dates including H-date purged tail, all labels masked',status='PASS',model_cells=len(receipts)))
    checks.append(dict(name='Six H3 models actually trained; five H5 reuses verified with full labels/features/maturity parity',status='PASS',new_neural_fits=sum(r['new_neural_fit'] for r in receipts),h5_reuse=sum(r['reuse'] is not None for r in receipts)))
    e.save(e.HERE/'verification.json',dict(status='PASS',checks=checks,model_year_horizon_cells=len(receipts),raw_predictions_sha256=e.sha(e.HERE/'raw_predictions.csv.gz'),warmup_sha256=e.sha(e.HERE/'warmup.csv.gz'),source_sha256=e.sha(e.SOURCE),protocol_sha256=e.sha(e.HERE/'protocol.json'),warmup_protocol_sha256=e.sha(e.HERE/'warmup_protocol.json'),code_sha256=e.sha(__file__)))
    print('TABM VERIFICATION PASS',len(receipts),'model cells',len(replay),'replay groups',flush=True)
if __name__=='__main__':main()
