"""Annual expanding-H3 verification; no fitting or model selection."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np,torch
from threadpoolctl import threadpool_limits
from research_v4.h3_finalization.long_history import experiment as e
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    protocol=json.loads((e.HERE/'protocol.json').read_text());assert e.sha(e.__file__)==protocol['code_sha256']
    for file,digest in protocol['source_hashes'].items():assert e.sha(ROOT/file)==digest
    p=e.load_panel();assert p.date.min()==pd.Timestamp('2010-01-01') and p.corridor.eq('KZT').all()
    rates=p.rub_per_unit.to_numpy(float);future=np.stack([rates[k:len(rates)-3+k] for k in (1,2,3)],axis=1)
    expected=(((future.min(axis=1)/rates[:-3]-1)*10000+1e-12)>=0).astype(float)
    np.testing.assert_array_equal(p.target.iloc[:-3],expected)
    np.testing.assert_allclose(p.forward_bps.iloc[:-3],(future.mean(axis=1)/rates[:-3]-1)*10000,rtol=0,atol=1e-9)
    assert p.label_available_date.equals(p.date.shift(-3))
    raw=pd.read_csv(e.HERE/'raw_predictions.csv.gz',parse_dates=['date','label_available_date']);warm=pd.read_csv(e.HERE/'warmup.csv.gz',parse_dates=['date','label_available_date'])
    assert raw.train_horizon.eq(3).all() and warm.train_horizon.eq(3).all()
    assert raw.config_id.eq('tabm_kzt_fullhistory').all() and raw.corridor.eq('KZT').all()
    assert not raw.duplicated(['config_id','train_horizon','fold_test_year','split','date','corridor']).any()
    replay=[]
    with threadpool_limits(limits=2):
        for year in (2024,2025,2026):
            parts=e.split(p,year);tr=parts['train'];dest=e.CKPT/f'tabm_kzt_fullhistory_h3_{year}'
            receipt=json.loads((dest/'receipt.json').read_text());meta=json.loads((dest/'model.json').read_text())
            assert receipt['weights_sha256']==e.sha(dest/'weights.pt') and receipt['preprocessor_sha256']==e.sha(dest/'preprocess.joblib')
            assert meta['features']==e.FEATURES and meta['seed']==e.SEED and meta['refit_seed']==e.SEED+1
            assert pd.Timestamp(meta['inner_label_max'])<pd.Timestamp(meta['inner_validation_min'])
            assert tr.date.min()==e.TRAIN_START and tr.label_available_date.max()<pd.Timestamp(year-1,1,1)
            assert parts['validation'].label_available_date.max()<pd.Timestamp(year,1,1)
            model=e.n.Neural(e.FEATURES,e.SEED);model.load(dest)
            independent=model.preprocessor(tr)
            np.testing.assert_array_equal(independent.named_steps['impute'].statistics_,model.pre.named_steps['impute'].statistics_)
            np.testing.assert_array_equal(independent.named_steps['gaussian'].quantiles_,model.pre.named_steps['gaussian'].quantiles_)
            for split in ('validation','history','test','warmup'):
                f=parts[split].sort_values(['date','corridor'])
                table=warm if split=='warmup' else raw
                q=table[table.fold_test_year.eq(year)&table.split.eq(split)].sort_values(['date','corridor']).reset_index(drop=True)
                pd.testing.assert_frame_equal(q[['date','corridor']],f[['date','corridor']].reset_index(drop=True),check_exact=True,check_dtype=False)
                error=float(np.max(np.abs(model.predict(f)-q.raw_probability)));assert error<2e-7
                if split=='warmup':
                    assert q.date.nunique()==63 and q.date.max()==p.loc[p.date.lt(pd.Timestamp(year-1,1,1)),'date'].max()
                    assert q[['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all()
                    assert (~f.index.isin(tr.index)).sum()==3
                elif split=='history':
                    immature=q.label_available_date.ge(pd.Timestamp(year,1,1))|q.label_available_date.isna()
                    assert q.loc[immature,['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all()
                else:np.testing.assert_array_equal(q.target,f.target)
                replay.append(dict(year=year,split=split,rows=len(f),maximum_raw_error=error))
    baseline=pd.read_csv(ROOT/'research_v4/robust_selection/tabm/raw_predictions.csv.gz',parse_dates=['date','label_available_date'])
    baseline=baseline[baseline.config_id.eq('tabm_kzt')&baseline.train_horizon.eq(3)]
    common=['fold_test_year','split','date','corridor'];a=baseline.sort_values(common).reset_index(drop=True);b=raw.sort_values(common).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[common+['target','forward_bps','label_available_date']],b[common+['target','forward_bps','label_available_date']],check_exact=True)
    basewarm=pd.read_csv(ROOT/'research_v4/robust_selection/tabm/warmup.csv.gz',parse_dates=['date','label_available_date'])
    basewarm=basewarm[basewarm.config_id.eq('tabm_kzt')&basewarm.train_horizon.eq(3)].sort_values(common).reset_index(drop=True)
    pd.testing.assert_frame_equal(basewarm[common],warm.sort_values(common).reset_index(drop=True)[common],check_exact=True)
    pd.DataFrame(replay).to_csv(e.HERE/'checkpoint_replay.csv',index=False)
    e.save(e.HERE/'verification.json',dict(status='PASS',annual_models=3,new_neural_fits=3,train_horizon=3,scope='KZT only',fixed_2010_train_start=True,train_only_preprocessing_exact=True,actual_maturity_pass=True,h3_target_and_forward_independently_recomputed=True,baseline_120m_evaluation_keys_outcomes_exact=True,warmup_includes_purged_tail=True,replay_groups=len(replay),maximum_raw_error=max(x['maximum_raw_error'] for x in replay),source_sha256=e.sha(e.SOURCE),raw_predictions_sha256=e.sha(e.HERE/'raw_predictions.csv.gz'),warmup_sha256=e.sha(e.HERE/'warmup.csv.gz'),protocol_sha256=e.sha(e.HERE/'protocol.json'),code_sha256=e.sha(__file__)))
    print('EXPANDING H3 PASS',len(replay),'replay groups',flush=True)
if __name__=='__main__':main()
