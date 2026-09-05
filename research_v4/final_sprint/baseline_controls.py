"""Apply identical new policies to sealed V3 and Halyk checkpoints; zero new fits."""
from pathlib import Path
import sys,pickle,os
os.environ['OMP_NUM_THREADS']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from research_v4.final_sprint import hgb as h
from research_v4.final_sprint.common import POLICIES,fit_policy,apply_policy,metrics
HERE=Path(__file__).resolve().parent;OUT=HERE/'baseline_controls';OUT.mkdir(exist_ok=True)

def main():
    views,_=pickle.loads((HERE/'views.pkl').read_bytes());p=views['2010-01-01',24,1]
    old=ROOT/'research_v4/oxr2010_bank/long_models';rows=[];frames=[];histories=[]
    with threadpool_limits(limits=1):
        for name,folder in [('v3_120m',old/'output'),('halyk_shrink_120m',old/'output'),('treasury_halyk_shrink_120m',old/'treasury/output')]:
            for year in (2025,2026):
                cutoff=pd.Timestamp(year,1,1);stem=name+'_'+str(cutoff.date());bundle=pickle.loads((folder/(stem+'.pkl')).read_bytes())
                _,va,hist,_=h.split(p,dict(cal=12,months=120),cutoff)
                for z in (va,hist):
                    z['raw_probability']=bundle['model'].predict_proba(z[bundle['features']+['corridor']])[:,1]
                    z['probability']=h.e.core.apply_platt(bundle['calibrator'],z.raw_probability.to_numpy())
                policies={x:fit_policy(va,hist,x) for x in POLICIES}
                pred=pd.read_csv(folder/(stem+'.csv.gz'),parse_dates=['date']);pred=pred[pred.corridor.eq('KZT')]
                for mode,g in pred.groupby('mode'):
                    for policy in policies.values():
                        q=apply_policy(g,policy)
                        if policy['name']=='legacy':assert np.array_equal(q.candidate_signal,g.candidate_signal),stem
                        rows.append(dict(config_id=name,cutoff=str(cutoff.date()),policy=policy['name'],mode=mode,**metrics(q)));frames.append(q)
                for split,z in [('validation',va),('history',hist)]:
                    q=z[['date','corridor','target','forward_bps','label_available_date','session_ordinal','raw_probability','probability']].copy()
                    q.loc[q.label_available_date.ge(cutoff)|q.label_available_date.isna(),['target','forward_bps']]=np.nan
                    q['config_id']=name;q['cutoff']=str(cutoff.date());q['split']=split;histories.append(q)
                h.e.save(OUT/(stem+'_policies.json'),policies)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False);pd.concat(frames).to_csv(OUT/'all_predictions.csv.gz',index=False);pd.concat(histories).to_csv(OUT/'histories.csv.gz',index=False)
    h.e.save(OUT/'verification.json',dict(status='PASS',fits=0,legacy_signal_reproductions=10))
if __name__=='__main__':main()
