"""Score 63 panel dates before calibration, including the purged train tail."""
from pathlib import Path
import sys,pickle,json,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.tabm import experiment as e
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    protocol=dict(created_unix=time.time(),request='63 latest PANEL dates strictly before prior-year calibration start, INCLUDING purged training tail; same checkpoint. Corrected per independent audit.',labels='All outcome columns masked. Most rows are in-sample training dates; final H dates are outside purged training. Used only to initialize past score ranks, not a separate validation set.',output='warmup.csv.gz, split=warmup, same schema as main raw_predictions.csv.gz',supersedes='warmup_purged_train_raw_predictions.csv.gz is retained only as a discarded diagnostic; parent must use warmup.csv.gz.',source_sha256=e.sha(e.SOURCE),code_sha256=e.sha(__file__))
    if not (e.HERE/'warmup_protocol.json').exists():e.save(e.HERE/'warmup_protocol.json',protocol)
    views,_=pickle.loads(e.SOURCE.read_bytes());panels={h:e.targeted(views['2010-01-01',24,1],h) for h in (3,5)}
    frames=[];receipts=[]
    with threadpool_limits(limits=2):
        for h in (3,5):
            for year in (2024,2025,2026):
                for scope in ('kzt','pooled'):
                    prior=panels[h][panels[h].date.lt(pd.Timestamp(year-1,1,1))]
                    if scope=='kzt':prior=prior[prior.corridor.eq('KZT')]
                    dates=sorted(prior.date.unique())[-63:];q=prior[prior.date.isin(dates)]
                    tr=e.split(panels[h],scope,h,year)['train']
                    dest=e.CKPT/e.stem(scope,h,year);model=e.n.Neural(e.FEATURES,e.SEED);model.load(dest)
                    out=q[e.KEEP].copy();out['raw_probability']=model.predict(q)
                    out[['target','forward_bps','symmetric_bps','regret_bps']]=np.nan
                    out['config_id']='tabm_'+scope;out['train_horizon']=h;out['cutoff']=f'{year}-01-01';out['fold_test_year']=year;out['split']='warmup'
                    frames.append(out)
                    assert out.date.nunique()==63 and out.date.max()<pd.Timestamp(year-1,1,1)
                    assert q.date.max()==prior.date.max()
                    receipts.append(dict(config_id='tabm_'+scope,train_horizon=h,year=year,rows=len(q),unique_dates=63,date_min=q.date.min(),date_max=q.date.max(),latest_label=q.label_available_date.max(),rows_in_purged_train_tail=int((~q.index.isin(tr.index)).sum()),weights_sha256=e.sha(dest/'weights.pt')))
    p=pd.concat(frames,ignore_index=True);p.to_csv(e.HERE/'warmup.csv.gz',index=False)
    e.save(e.HERE/'warmup_receipts.json',receipts);print('WARMUP',len(p),'rows',len(receipts),'model cells',flush=True)
if __name__=='__main__':main()
