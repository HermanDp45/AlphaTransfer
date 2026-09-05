"""Post-readout, predeclared four-fit source-delay augmentation control."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import time,json,warnings
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as x
HERE=x.HERE
def main():
    x.initialize()
    protocol=dict(created_unix=time.time(),status='Post-readout robustness addendum after observing sensitivity to delayed sources. Exploratory; no untouched holdout claim.',fits=4,cutoffs=['2025-01-01','2026-01-01'],specification='KZT-only120m seed0, identical architecture/optimizer/epoch-selection rule. Two fixed augmentation strategies, no HPO.',strategies={'bankaug':[(24,1),(24,2)],'allaug':[(24,1),(48,1),(24,2),(48,2)]},training='Replicate each matured training example for equally weighted causal source-availability views; same label. Inner epoch selection also uses equally weighted source views; full refit uses selected epoch. Train-only median/quantile preprocessing includes augmented examples.',calibration='Normal prior-year validation and normal pre-cutoff history, unchanged policy rules.',numerics='No extra winsorization or fit on future/test features.',code_sha256=x.sha(__file__),base_code_sha256=x.sha(x.__file__))
    x.save(HERE/'augmentation_protocol.json',protocol)
    original_name=x.name;original_split=x.split
    with threadpool_limits(limits=2),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,features=x.build()
        for strategy,modepairs in protocol['strategies'].items():
            x.name=lambda spec,strategy=strategy:original_name(spec)+'_'+strategy
            def augmented_split(v,s,cutoff,modepairs=modepairs):
                tr,va,te,history=original_split(v,s,cutoff)
                versions=[]
                for delay,lag in modepairs:
                    q=v['2010-01-01',delay,lag].loc[tr.index].copy()
                    pd.testing.assert_frame_equal(tr[x.KEEP],q[x.KEEP],check_exact=True)
                    q['augmentation_view']=f'oxr{delay}_bank{lag}';versions.append(q)
                return pd.concat(versions,ignore_index=True),va,te,history
            x.split=augmented_split
            for cutoff in protocol['cutoffs']:
                x.run(views,features,dict(scope='kzt',months=120,seed_index=0,augmentation=strategy),cutoff)
    p,c=x.aggregate()
    x.save(HERE/'augmentation_completion.json',dict(status='complete',neural_fits=4,temporary_inner_fits=4,protocol_sha256=x.sha(HERE/'augmentation_protocol.json'),code_sha256=x.sha(__file__),prediction_rows=len(p),calibration_history_rows=len(c)))
if __name__=='__main__':main()
