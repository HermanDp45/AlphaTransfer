"""Common past-only calibration/policies and cell-standardized comparison."""
from pathlib import Path
import os,sys,json,pickle
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.common import fit_policy,apply_policy
from final_solution.training import core_experiment as core
from research_v4.continuation.oxr.assess import point,paired
HERE=Path(__file__).resolve().parent

def metrics(g):
    p=point(g);counts=[];coverage=[]
    for _,cell in g.groupby(['fold_test_year','corridor']):
        c=cell.groupby(cell.date.dt.to_period('W-SUN')).candidate_signal.sum()
        c=c.reindex(pd.period_range(c.index.min(),c.index.max(),freq='W-SUN'),fill_value=0)
        counts+=list(c);coverage.append(c.between(1,2).mean())
    c=pd.Series(counts);s=g[g.candidate_signal]
    p.update(hit_rate=float(s.target.mean()),hits=int(s.target.sum()),raw_brier=float(np.mean((g.raw_probability-g.target)**2)) if g.raw_probability.notna().all() else np.nan,auc=float(roc_auc_score(g.target,g.probability)),log_loss=float(log_loss(g.target,g.probability,labels=[0,1])),week_cells=len(c),weeks_with_1_2=int(c.between(1,2).sum()),week_coverage=float(c.between(1,2).mean()),min_year_corridor_coverage=float(min(coverage)),signals_per_corridor_week=float(c.mean()),regret_bps=float(s.regret_bps.mean()))
    return p

def historical():
    sys.path.insert(0,str(ROOT/'research_v3/models'))
    from decision_uncertainty import policies
    reference,masks=policies();rows=[]
    for cid in ('baseline_reproduction','basis_train_120m'):
        f=pd.read_csv(ROOT/f'research_v3/models/{cid}_h5_predictions.csv.gz',parse_dates=['date'],float_precision='round_trip').sort_values(['date','corridor']).reset_index(drop=True)
        assert f[['date','corridor']].equals(reference[['date','corridor']])
        for pol in ('legacy','threshold_0.50'):
            q=f.copy();q['candidate_signal']=masks[cid+'::'+pol];q['config_id']='historical_'+cid;q['policy']='fixed_threshold_0.50' if pol.startswith('threshold') else 'legacy';q['raw_probability']=np.nan;q['architecture_scope']='historical_pooled';rows.append(q[q.fold_test_year.le(2025)])
    return pd.concat(rows,ignore_index=True)

def main():
    raw=pd.concat([pd.read_csv(HERE/f'{scope}/raw_predictions.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip') for scope in ('local','pooled')],ignore_index=True)
    mapping={f'arch_pooled_{arch}_{family}':f'pooled_{arch}_{family}'+('_s2' if arch=='tabm' else '') for arch in ('tabm','hgb') for family in ('base15','full33')}
    raw['config_id']=raw.config_id.replace(mapping)
    output=[];cals=[];policies=[]
    for (cid,cutoff),g in raw.groupby(['config_id','cutoff']):
        v=g[g.split.eq('validation')].copy();h=g[g.split.eq('history')].copy();t=g[g.split.eq('test')].copy()
        assert v.label_available_date.max()<pd.Timestamp(cutoff) and v.target.notna().all()
        assert not h.loc[h.label_available_date.ge(pd.Timestamp(cutoff)),'target'].notna().any()
        for corridor in v.corridor.unique():
            vv=v[v.corridor.eq(corridor)];cal=core.fit_platt_calibrator(vv.raw_probability.to_numpy(),vv.target)
            for z in (v,h,t):
                mask=z.corridor.eq(corridor);z.loc[mask,'probability']=core.apply_platt(cal,z.loc[mask,'raw_probability'].to_numpy())
            cals.append(dict(config_id=cid,cutoff=cutoff,corridor=corridor,method=cal.method,intercept=cal.intercept,slope=cal.slope,validation_rows=len(vv),latest_label=vv.label_available_date.max()))
        for pname in ('legacy','cadence85_cd2','fixed_threshold_0.50'):
            if pname=='fixed_threshold_0.50':
                from research_v4.final_sprint.common import select
                thresholds={c:.5 for c in v.corridor.unique()};_,state=select(h,thresholds,3)
                policy=dict(name=pname,threshold=thresholds,cooldown=3,initial_state=state)
            else:policy=fit_policy(v,h,pname)
            q=apply_policy(t,policy);q['architecture_scope']='kzt_only' if cid.startswith('kzt_') else 'pooled';output.append(q)
            policies.append(dict(config_id=cid,cutoff=cutoff,**policy))
    pred=pd.concat([*output,historical()],ignore_index=True)
    pred.to_csv(HERE/'predictions.csv.gz',index=False)
    (HERE/'calibration.json').write_text(json.dumps(cals,indent=2,default=str));(HERE/'policies.json').write_text(json.dumps(policies,indent=2,default=str))
    summary=[];cells=[]
    for (cid,pol),g in pred.groupby(['config_id','policy']):
        for scope in (['all5','KZT'] if g.corridor.nunique()>1 else ['KZT']):
            q=g if scope=='all5' else g[g.corridor.eq('KZT')]
            summary.append(dict(config_id=cid,policy=pol,evaluation_scope=scope,**metrics(q)))
            for year,x in q.groupby('fold_test_year'):cells.append(dict(config_id=cid,policy=pol,evaluation_scope=scope,year=year,**metrics(x)))
    pd.DataFrame(summary).to_csv(HERE/'summary.csv',index=False);pd.DataFrame(cells).to_csv(HERE/'by_year.csv',index=False)
    intervals=[]
    # Matched architecture contrasts: exact same test rows, feature families,
    # training population and per-corridor calibration rule.
    for training in ('kzt','pooled'):
        for family in ('base15','full33'):
            hgb=f'{training}_hgb_{family}';tabm=f'{training}_tabm_{family}_s2'
            for policy in ('legacy','cadence85_cd2','fixed_threshold_0.50'):
                a=pred[pred.config_id.eq(hgb)&pred.policy.eq(policy)];b=pred[pred.config_id.eq(tabm)&pred.policy.eq(policy)]
                for scope in (['KZT'] if training=='kzt' else ['all5','KZT']):
                    aa=a if scope=='all5' else a[a.corridor.eq('KZT')];bb=b if scope=='all5' else b[b.corridor.eq('KZT')]
                    intervals.append(dict(baseline=hgb,candidate=tabm,policy=policy,scope=scope,contrast='matched_architecture',**paired(aa,bb)))
    for base,pol in [('historical_basis_train_120m','legacy'),('historical_basis_train_120m','fixed_threshold_0.50')]:
        a=pred[pred.config_id.eq(base)&pred.policy.eq(pol)&pred.corridor.eq('KZT')]
        b=pred[pred.config_id.eq('kzt_tabm_full33_s2')&pred.policy.eq('cadence85_cd2')]
        intervals.append(dict(baseline=base,candidate='kzt_tabm_full33_s2',policy=pol+' vs cadence85_cd2',scope='KZT',contrast='systems_not_architecture',**paired(a,b)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    (HERE/'completion.json').write_text(json.dumps(dict(status='complete',matched_contrasts=sum(x['contrast']=='matched_architecture' for x in intervals),system_contrasts=2,policies_per_matched_model=3,calibration='sameper-corridorPlatt-rule',aggregation='year×corridorstandardized',bootstrap='pairedcalendar-month10kyearstrata;conditionalonfixedmodels;retrospective'),indent=2))
    print(pd.DataFrame(summary).query("config_id=='kzt_tabm_full33_s2'").to_string(index=False))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
