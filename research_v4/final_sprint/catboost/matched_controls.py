"""Six explicitly requested controls isolating feature selection from depth/L2."""
from pathlib import Path
import sys,time,warnings
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import experiment as x
import pandas as pd
from threadpoolctl import threadpool_limits

def main():
    dest=HERE/'matched';dest.mkdir(exist_ok=True)
    x.OUT=dest/'output';x.OUT.mkdir(exist_ok=True)
    specs=[]
    for original in x.specifications():
        if original['subset']!='full':continue
        s=original.copy();s['name']=s['name'].removesuffix('_full')+'_stable_matched';s['subset']='stable'
        specs.append(s)
    x.save(dest/'protocol.json',dict(created_unix=time.time(),specifications=specs,fits=6,
       reason='Requested matched feature-selection attribution: original full-vs-stable also varied depth/L2. Match every full-model hyperparameter and change only selected numeric columns.',
       disclosure='Added after inspecting initial12fit results; no hyperparameter selection or2026labels used for feature selection.',
       experiment_sha256=x.sha(x.__file__),code_sha256=x.sha(__file__)))
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        views,bankcols=x.e.build_views();views={k:x.augment_panel(v) for k,v in views.items()}
        pred=[];hist=[]
        for cutoff in ('2025-01-01','2026-01-01'):
            for s in specs:
                p,h=x.fit_one(views,bankcols,s,cutoff);pred.append(p);hist.append(h)
    p=pd.concat(pred,ignore_index=True);h=pd.concat(hist,ignore_index=True)
    p.to_csv(dest/'all_predictions.csv.gz',index=False);h.to_csv(dest/'history_and_calibration.csv.gz',index=False)
    original=pd.read_csv(HERE/'all_predictions.csv.gz',parse_dates=['date'])
    original.date=original.date.astype(p.date.dtype)
    allp=pd.concat([original,p],ignore_index=True)
    allp.to_csv(HERE/'candidates_predictions.csv.gz',index=False)
    oldhist=pd.read_csv(HERE/'calibration/history_predictions.csv.gz',parse_dates=['date'])
    both=pd.concat([oldhist,h],ignore_index=True)
    both[both.split.eq('history')].to_csv(HERE/'candidates_history_predictions.csv.gz',index=False)
    both[both.split.eq('calibration')].to_csv(HERE/'candidates_calibration_predictions.csv.gz',index=False)
    rows=[];intervals=[]
    with threadpool_limits(limits=1):
        for (name,cutoff,mode),q in allp.groupby(['config_id','cutoff','mode']):
            rows.append(dict(config_id=name,cutoff=cutoff,mode=mode,scope='KZT',**x.point(q),**x.cadence_all(q)))
            if name.endswith('_stable_matched'):
                ref=name.removesuffix('_stable_matched')+'_full'
                a=allp[allp.config_id.eq(ref)&allp.cutoff.eq(cutoff)&allp['mode'].eq(mode)]
                for block in ('month',20):
                    intervals.append(dict(config_id=name,cutoff=cutoff,mode=mode,baseline=ref,block=str(block),**x.paired(a,q,block)))
    pd.DataFrame(rows).to_csv(HERE/'candidates_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(dest/'paired_feature_selection.csv',index=False)
    x.save(dest/'completion.json',dict(status='complete',additional_fits=6,total_fits=18,models=9,
          output_sha256=x.sha(dest/'all_predictions.csv.gz'),combined_sha256=x.sha(HERE/'candidates_predictions.csv.gz'),
          original_prediction_unchanged_sha256=x.sha(HERE/'all_predictions.csv.gz')))
    print('Complete6matchedcontrols; total18fits',flush=True)

if __name__=='__main__':main()
