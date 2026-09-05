"""Frozen h5 policy evaluated on other horizons and delayed execution proxies."""
from pathlib import Path
import sys,pickle,json
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from research_v4.final_sprint.common import metrics
from final_solution.training import core_experiment as core
HERE=Path(__file__).resolve().parent

def main():
    pred=pd.read_csv(HERE/'selected_predictions.csv.gz',parse_dates=['date'])
    pred=pred[pred.fold_test_year.eq(2026)&pred['mode'].eq('normal')]
    base=pd.read_csv(HERE/'baseline_controls/all_predictions.csv.gz',parse_dates=['date'])
    base=base[base.fold_test_year.eq(2026)&base['mode'].eq('normal')&base.policy.eq('legacy')&base.config_id.eq('v3_120m')]
    views,_=pickle.loads((HERE/'views.pkl').read_bytes());panel=views['2010-01-01',24,1]
    p=panel[panel.corridor.eq('KZT')].copy().sort_values('date')
    rows=[]
    for horizon in (1,3,5,10,20):
        z=core.add_target(panel,horizon);z=z[z.corridor.eq('KZT')&z.target.notna()]
        for name,g in [('selected',pred),('V3',base)]:
            q=g.drop(columns=['target','forward_bps']).merge(z[['date','corridor','target','forward_bps']],on=['date','corridor'],validate='one_to_one')
            m=metrics(q)
            if horizon!=5:m['brier']=np.nan
            rows.append(dict(type='horizon_sensitivity_frozen_h5_policy',model=name,horizon=horizon,execution_delay=0,**m))
    rates=p.rub_per_unit.to_numpy();q=p[['date','corridor']].copy()
    for delay in (0,1,2):
        target=np.full(len(q),np.nan);forward=np.full(len(q),np.nan)
        for i in range(len(q)-5):
            current=rates[i+delay];future=rates[i+delay+1:i+6]
            target[i]=float(current<=future.min()+1e-12);forward[i]=(future.mean()/current-1)*10000
        q['target']=target;q['forward_bps']=forward
        for name,g in [('selected',pred),('V3',base)]:
            z=g.drop(columns=['target','forward_bps']).merge(q,on=['date','corridor'],validate='one_to_one').dropna(subset=['target'])
            m=metrics(z)
            if delay:m['brier']=np.nan
            rows.append(dict(type='execution_delay_fixed_original_h5_deadline',model=name,horizon=5,execution_delay=delay,**m))
    pd.DataFrame(rows).to_csv(HERE/'horizon_execution_sensitivity.csv',index=False)
    cells=[]
    for name,g in [('selected',pred),('V3',base)]:
        for quarter,z in g.groupby(g.date.dt.to_period('Q')):cells.append(dict(model=name,quarter=str(quarter),**metrics(z)))
    pd.DataFrame(cells).to_csv(HERE/'quarter_sensitivity.csv',index=False)
    (HERE/'sensitivity_receipt.json').write_text(json.dumps(dict(status='complete',fits=0,interpretation='Same h5 model and causal notifications; other horizons are payoff sensitivity, not separately trained h1/3/10/20 heads. Execution delays countCBRobservations, not wallclockhours; original t+5 deadline is retained; executed-price proxy is CBR, not a bank fill.'),indent=2))
if __name__=='__main__':main()
