"""Monthly refits with actual emitted-policy state carried over across fits."""
from pathlib import Path
import os,sys,pickle,warnings,json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.final_sprint import hgb as h
from research_v4.final_sprint.common import select,POLICIES,metrics
HERE=Path(__file__).resolve().parent;OUT=HERE/'monthly';OUT.mkdir(exist_ok=True)
SPECS=[s for s in h.specs() if s['name'] in ['hgb_60m_c3_d0_bank_treasury','hgb_120m_c3_d0_bank_treasury','hgb_120m_c3_d0_bank_treasury_aug']]

def main():
    views,cols=pickle.loads((HERE/'views.pkl').read_bytes());frames=[];rows=[];fits=0
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        for year in (2025,2026):
            for spec in SPECS:
                states={};parts=[]
                normal=views['2010-01-01',24,1]
                stress=h.e.stress_view(views,dict(since='2010-01-01'),pd.Timestamp(year,1,1),24,2)
                grid=h.split(normal,spec,pd.Timestamp(year,1,1))[-1]
                for cutoff in pd.date_range(f'{year}-01-01',f'{year}-'+('08-01' if year==2026 else '12-01'),freq='MS'):
                    name=spec['name']+'_'+str(cutoff.date());checkpoint=h.OUT/(name+'.pkl')
                    if not checkpoint.exists():h.run(views,cols,spec,cutoff);fits+=1
                    bundle=pickle.loads(checkpoint.read_bytes());feat=bundle['features']
                    te=grid[grid.date.ge(cutoff)&grid.date.lt(cutoff+pd.DateOffset(months=1))]
                    if te.empty:continue
                    for mode,view in [('normal',normal),('bank_delayed',stress)]:
                        z=view.loc[te.index].copy()
                        z['raw_probability']=bundle['model'].predict_proba(z[feat+['corridor']])[:,1]
                        z['probability']=h.e.core.apply_platt(bundle['calibrator'],z.raw_probability.to_numpy())
                        for policy_name,policy in bundle['policies'].items():
                            key=(mode,policy_name)
                            ids,state=select(z,policy['threshold'],policy['cooldown'],states.get(key,policy['initial_state']))
                            states[key]=state;q=z[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal','label_available_date','raw_probability','probability','pr60','ret1','rub_per_unit']].copy()
                            q['candidate_signal']=q.index.isin(ids);q['policy']=policy_name;q['mode']=mode
                            q['config_id']='monthly_'+spec['name'];q['cutoff']=f'{year}-01-01';q['fit_cutoff']=str(cutoff.date());q['fold_test_year']=year
                            parts.append(q)
                pred=pd.concat(parts,ignore_index=True);pred.to_csv(OUT/(f'monthly_{spec["name"]}_{year}.csv.gz'),index=False);frames.append(pred)
                for (mode,policy),g in pred.groupby(['mode','policy']):rows.append(dict(config_id='monthly_'+spec['name'],cutoff=f'{year}-01-01',mode=mode,policy=policy,**metrics(g)))
                pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    h.e.save(OUT/'completion.json',dict(status='complete',new_fits=fits,reused_january_fits=6,configs=3,state='Actual contact state per mode/policy retained across all monthly refits; no reset or replay with new model.',stress='annualJanuary onset; fixed normal-trained checkpoints evaluated with delayed inference features.'))
if __name__=='__main__':main()
