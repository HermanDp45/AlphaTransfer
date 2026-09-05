"""Two attribution-only fits: normal+normal against normal+banklag2 training."""
from pathlib import Path
import sys,warnings,time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import experiment as x
from research_v4.final_sprint import hgb
import pandas as pd
from threadpoolctl import threadpool_limits

def main():
    dest=HERE/'augmentation_controls';dest.mkdir(exist_ok=True);out=dest/'output';out.mkdir(exist_ok=True)
    spec=next(s.copy() for s in hgb.specs() if s['name']=='hgb_60m_c12_d0_bank_treasury_aug')
    spec.update(name='attribution_hgb_60m_c12_duplicate_normal',augment=False)
    x.save(dest/'protocol.json',dict(created_unix=time.time(),spec=spec,fits=2,
        purpose='Attribution only, excluded from candidate selection: isolate delayed-feature augmentation from sample duplication/effective regularization.',
        implementation='Duplicate only the normal training frame; preserve original calibration, history, actual normal and actual banklag2 test views.',
        disclosure='Requested after identifying original augmentation champion; no new model/policy selection.',
        hgb_sha256=x.sha(hgb.__file__),code_sha256=x.sha(__file__)))
    original_split=hgb.split;original_out=hgb.OUT
    unique_rows={}
    def duplicate_split(panel,s,cutoff):
        tr,va,history,te=original_split(panel,s,cutoff)
        unique_rows[str(cutoff.date())]=len(tr)
        return pd.concat([tr,tr.copy()],ignore_index=True),va,history,te
    try:
        hgb.split=duplicate_split;hgb.OUT=out
        with threadpool_limits(limits=1),warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            views,cols=x.e.build_views();views={k:x.augment_panel(v) for k,v in views.items()}
            rows=[]
            for cutoff in ('2025-01-01','2026-01-01'):rows+=hgb.run(views,cols,spec,cutoff)
    finally:
        hgb.split=original_split;hgb.OUT=original_out
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False)
    intervals=[]
    with threadpool_limits(limits=1):
        for cutoff in ('2025-01-01','2026-01-01'):
            dup=pd.read_csv(out/(spec['name']+'_'+cutoff+'.csv.gz'),parse_dates=['date'])
            aug=pd.read_csv(original_out/('hgb_60m_c12_d0_bank_treasury_aug_'+cutoff+'.csv.gz'),parse_dates=['date'])
            for (mode,policy),b in aug.groupby(['mode','policy']):
                a=dup[dup['mode'].eq(mode)&dup.policy.eq(policy)]
                for block in ('month',20):intervals.append(dict(cutoff=cutoff,mode=mode,policy=policy,block=str(block),baseline='duplicate_normal',candidate='normal_plus_banklag2',**x.paired(a,b,block)))
    pd.DataFrame(intervals).to_csv(dest/'augmentation_vs_duplicate.csv',index=False)
    x.save(dest/'completion.json',dict(status='complete',fits=2,unique_train_rows=unique_rows,
             physical_training_rows='Exactly twice unique_train_rows; equal unit weights in both copies.',
             inference='Actual unchanged normal/banklag2 source views',candidate_selection_eligible=False,
             output_hashes={p.name:x.sha(p) for p in out.glob('*.csv.gz')}))
    print('Complete2duplicate-normal attribution fits',flush=True)

if __name__=='__main__':main()
