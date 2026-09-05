"""Shared causal policies and prior-calibration TabM/HGB blends, zero new fits."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import json,time,pickle
import numpy as np
import pandas as pd
from research_v4.final_sprint import common
from research_v4.continuation.oxr.assess import paired,point
from research_v4.final_sprint.tabm.experiment import HERE,save,sha,KEEP
HGB='hgb_60m_c12_d0_bank_treasury_aug'
KEY=['date','corridor']
def align(a,b):
    a=a.sort_values(KEY).reset_index(drop=True);b=b.sort_values(KEY).reset_index(drop=True)
    assert a[KEY].equals(b[KEY]);assert np.allclose(a.target,b.target,equal_nan=True)
    return a,b
def blends(p,c):
    out=[];cal=[];weights=[]
    for (cid,cutoff),q in p[p.corridor.eq('KZT')].groupby(['config_id','cutoff']):
        cp=c[c.config_id.eq(cid)&c.cutoff.eq(cutoff)&c.corridor.eq('KZT')]
        prefix=ROOT/'research_v4/final_sprint/hgb'/f'{HGB}_{cutoff}'
        hp=pd.read_csv(prefix.with_suffix('.csv.gz'),parse_dates=['date'])
        hc=pd.read_csv(prefix.with_name(prefix.name+'_calibration.csv.gz'),parse_dates=['date'])
        a,b=align(cp[cp.split.eq('validation')],hc[hc.split.eq('validation')])
        scores={w:float(np.mean((w*a.probability+(1-w)*b.probability-a.target)**2)) for w in (0.,.25,.5,.75,1.)}
        weight=min(scores,key=scores.get);bid='blend_treasH_'+cid
        weights.append(dict(config_id=bid,neural_config=cid,cutoff=cutoff,neural_weight=weight,grid_brier=scores,validation_rows=len(a),validation_last=str(a.date.max()),selection='prior-calibration Brier only; calibrated probabilities, no further calibration',hgb_checkpoint_sha256=sha(prefix.with_suffix('.pkl'))))
        for split in ('validation','history'):
            aa=cp[cp.split.eq(split)].sort_values(KEY).reset_index(drop=True)
            bb=hc[hc.split.eq(split)].sort_values(KEY).reset_index(drop=True)
            assert aa[KEY].equals(bb[KEY])
            r=aa.copy();r['probability']=weight*aa.probability+(1-weight)*bb.probability;r['raw_probability']=r.probability;r['config_id']=bid;cal.append(r)
        for mode,z in q.groupby('mode'):
            hmode='bank_delayed' if mode in ('bank_delayed','both_delayed') else 'normal'
            aa,bb=align(z,hp[hp['mode'].eq(hmode)&hp.policy.eq('legacy')])
            r=aa.copy();r['probability']=weight*aa.probability+(1-weight)*bb.probability;r['raw_probability']=r.probability;r['config_id']=bid;out.append(r)
    save(HERE/'blend_weights.json',weights)
    return pd.concat(out,ignore_index=True),pd.concat(cal,ignore_index=True)
def main():
    protocol=dict(created_unix=time.time(),hgb=HGB,weights=[0,.25,.5,.75,1],weight_selection='Only matched purged prior-year calibration Brier; calibrated component probabilities; no further recalibration',policies=common.POLICIES,stress='Fixed selected weights. HGB normal for TabM OXR-delay; HGB bank-delay for TabM both-delay.',status='Requested after initial neural results; 2026 is retrospective, never called untouched holdout',code_sha256=sha(__file__),common_sha256=sha(common.__file__))
    save(HERE/'policy_blend_protocol.json',protocol)
    p=pd.read_csv(HERE/'predictions.csv.gz',parse_dates=['date','label_available_date']);c=pd.read_csv(HERE/'calibration_predictions.csv.gz',parse_dates=['date','label_available_date'])
    bp,bc=blends(p,c);bp.to_csv(HERE/'blend_predictions.csv.gz',index=False);bc.to_csv(HERE/'blend_calibration_predictions.csv.gz',index=False)
    p=pd.concat([p[p.corridor.eq('KZT')],bp],ignore_index=True);c=pd.concat([c[c.corridor.eq('KZT')],bc],ignore_index=True)
    result=[];rows=[];bundle={}
    for (cid,cutoff),q in p.groupby(['config_id','cutoff']):
        cp=c[c.config_id.eq(cid)&c.cutoff.eq(cutoff)]
        va=cp[cp.split.eq('validation')].copy();history=cp[cp.split.eq('history')].copy()
        assert va.date.max()<pd.Timestamp(cutoff);assert va.label_available_date.max()<pd.Timestamp(cutoff)
        for policy in common.POLICIES:
            fit=common.fit_policy(va,history,policy);bundle[cid,cutoff,policy]=fit
            for mode,z in q.groupby('mode'):
                r=common.apply_policy(z,fit);result.append(r)
                rows.append(dict(config_id=cid,cutoff=cutoff,year=int(cutoff[:4]),mode=mode,policy=policy,**common.metrics(r)))
    result=pd.concat(result,ignore_index=True);result.to_csv(HERE/'policy_predictions.csv.gz',index=False)
    metrics=pd.DataFrame(rows);metrics.to_csv(HERE/'policy_summary.csv',index=False)
    (HERE/'policies.pkl').write_bytes(pickle.dumps(bundle))
    c.to_csv(HERE/'policy_calibration_predictions.csv.gz',index=False)
    # Pair against frozen incumbent predictions on exactly common dates.
    v3=pd.read_csv(ROOT/'research_v3/models/basis_train_120m_h5_predictions.csv.gz',parse_dates=['date'])
    treasury=[]
    for year in (2025,2026):
        f=ROOT/'research_v4/oxr2010_bank/long_models/treasury/output'/f'treasury_halyk_shrink_120m_{year}-01-01.csv.gz'
        t=pd.read_csv(f,parse_dates=['date']);treasury.append(t[t['mode'].eq('normal')])
    new_hgb=[]
    for year in (2025,2026):
        f=ROOT/'research_v4/final_sprint/hgb'/f'{HGB}_{year}-01-01.csv.gz'
        t=pd.read_csv(f,parse_dates=['date']);new_hgb.append(t[t['mode'].eq('normal')])
    baselines={'V3_long':v3[v3.corridor.eq('KZT')],'Treasury_Halyk_old':pd.concat(treasury),'Sprint_HGB60m_aug':pd.concat(new_hgb)}
    selected=json.loads((HERE/'selection_2025.json').read_text())['chosen']
    selected_id=f"tabm_periodic_{selected['scope']}_{selected['months']}m_s0_kztcal"
    finalists={selected_id,selected_id.replace('s0_','sensemble3_'),'blend_treasH_'+selected_id,'blend_treasH_'+selected_id.replace('s0_','sensemble3_')}
    finalists.update(selected_id.replace('s0_',f's{seed}_') for seed in (1,2))
    finalists.update('blend_treasH_'+selected_id.replace('s0_',f's{seed}_') for seed in (1,2))
    finalists.update(f'tabm_periodic_kzt_120m_s0_{a}_kztcal' for a in ('bankaug','allaug','noage'))
    # All summaries are available; intervals cover preselected architecture and
    # its ensemble/blend, not an outcome-selected best row.
    intervals=[];baseline_rows=[]
    for (year,cid,policy),b in result[result['mode'].eq('normal')&result.config_id.isin(finalists)].groupby(['fold_test_year','config_id','policy']):
        for baseline,raw in baselines.items():
            a=raw[raw.fold_test_year.eq(year)].copy()
            if 'policy' in a:a=a[a.policy.eq(policy)]
            a,b2=align(a,b)
            intervals.append(dict(year=year,config_id=cid,policy=policy,baseline=baseline,block='month',bootstrap_repetitions=10000,**paired(a,b2)))
            if policy=='legacy':baseline_rows.append(dict(year=year,baseline=baseline,**common.metrics(a)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    pd.DataFrame(baseline_rows).drop_duplicates().to_csv(HERE/'baseline_metrics.csv',index=False)
    attribution=[]
    for (year,mode),g in result[result.policy.eq('cadence90_cd3')].groupby(['fold_test_year','mode']):
        a=g[g.config_id.eq(selected_id)]
        for suffix in ('bankaug','allaug','noage'):
            cid=f'tabm_periodic_kzt_120m_s0_{suffix}_kztcal';b=g[g.config_id.eq(cid)]
            if len(b):attribution.append(dict(year=year,mode=mode,baseline=selected_id,candidate=cid,bootstrap_repetitions=10000,**paired(a,b)))
    pd.DataFrame(attribution).to_csv(HERE/'source_control_intervals.csv',index=False)
    save(HERE/'policy_completion.json',dict(status='complete',rows=len(result),config_cutoff_pairs=p[['config_id','cutoff']].drop_duplicates().shape[0],neural_weight_zero_count=sum(x['neural_weight']==0 for x in json.loads((HERE/'blend_weights.json').read_text())),intervals=len(intervals),code_sha256=sha(__file__),policies_sha256=sha(HERE/'policies.pkl')))
    print(metrics[metrics.year.eq(2026)&metrics['mode'].eq('normal')].sort_values(['lift','weeks_1_2'],ascending=False).head(12).to_string(index=False),flush=True)
if __name__=='__main__':main()
