"""Frozen rolling annual comparison and retrospective robustness selection."""
from pathlib import Path
import os,sys,json,warnings
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from final_solution.training import core_experiment as core
from research_v4.final_sprint.common import select
from research_v4.continuation.oxr.assess import paired
HERE=Path(__file__).resolve().parent
NAMES=['strict05','cadence80','cadence85','cadence90','rank80','rank90','rank90_shared']
def save(name,x): (HERE/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def metric(g,proper=True):
    s=g[g.candidate_signal.astype(bool)]
    cells=g.groupby(['fold_test_year','corridor']).agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
    joined=s.join(cells,on=['fold_test_year','corridor'])
    expected=joined.base_hit.sum();counts=[];cover=[];streaks=[]
    for _,z in g.groupby(['fold_test_year','corridor']):
        c=z.groupby(z.date.dt.to_period('W-SUN')).candidate_signal.sum()
        c=c.reindex(pd.period_range(c.index.min(),c.index.max(),freq='W-SUN'),fill_value=0)
        counts.extend(c);cover.append(c.between(1,2).mean());run=best=0
        for k in c:run=run+1 if k==0 else 0;best=max(best,run)
        streaks.append(best)
    c=pd.Series(counts);brier=float(np.mean((g.probability-g.target)**2)) if proper else np.nan
    return dict(rows=len(g),dates=g.date.nunique(),signals=len(s),hits=int(s.target.sum()),base_hit=float(g.target.mean()),hit_rate=float(s.target.mean()),lift=float(s.target.sum()/expected) if expected else np.nan,forward_delta_bps=float((joined.forward_bps-joined.base_forward).mean()),forward_signal_bps=float(s.forward_bps.mean()),regret_bps=float(s.regret_bps.mean()),brier=brier,raw_brier=float(np.mean((g.raw_probability-g.target)**2)) if proper else np.nan,auc=float(roc_auc_score(g.target,g.probability)) if proper and g.target.nunique()>1 else np.nan,log_loss=float(log_loss(g.target,g.probability,labels=[0,1])) if proper else np.nan,week_cells=len(c),weeks_with_1_2=int(c.between(1,2).sum()),week_coverage=float(c.between(1,2).mean()),mean_cell_week_coverage=float(np.mean(cover)),signals_per_corridor_week=float(c.mean()),silent_weeks=int(c.eq(0).sum()),max_week_signals=int(c.max()),max_silent_week_run=max(streaks),min_year_corridor_coverage=float(min(cover)))
def ranked(g):
    q=g.copy();q['rank_score']=np.nan
    for _,z in q.groupby('corridor'):
        z=z.sort_values('date');p=z.probability.to_numpy();r=np.full(len(p),np.nan)
        for i in range(20,len(p)):
            past=p[max(0,i-63):i];r[i]=(np.sum(past<p[i])+.5*np.sum(past==p[i]))/len(past)
        q.loc[z.index,'rank_score']=r
    return q

def policy_fit(v,h,name):
    cooldown=3 if name=='strict05' else 2;thresholds={};diagnostics=[]
    if name=='strict05':thresholds={c:.5 for c in v.corridor.unique()}
    else:
        target=.80 if '80' in name else .85 if '85' in name else .90
        shared=name.endswith('shared');groups=[('shared',v)] if shared else list(v.groupby('corridor'))
        for c,z in groups:
            candidates=[]
            for q in np.arange(0,.81,.05):
                t=float(q) if name.startswith('rank') else float(z.probability.quantile(q))
                zz=z.copy()
                if name.startswith('rank'):zz['probability']=zz.rank_score
                ids,_=select(zz,t,cooldown);zz['candidate_signal']=zz.index.isin(ids)
                ms=[metric(x,False) for _,x in zz.groupby('corridor')];m=metric(zz,False)
                coverage=min(x['week_coverage'] for x in ms)
                candidates.append(dict(q=float(q),threshold=t,coverage=coverage,lift=m['lift'],forward_delta_bps=m['forward_delta_bps'],feasible=coverage>=target))
            feasible=[x for x in candidates if x['feasible']]
            chosen=max(feasible,key=lambda x:(x['lift'],x['forward_delta_bps'],x['coverage'])) if feasible else max(candidates,key=lambda x:(x['coverage'],x['lift']))
            for cc in (z.corridor.unique() if shared else [c]):thresholds[str(cc)]=chosen['threshold']
            diagnostics.append(dict(corridor=c,target=target,chosen=chosen,candidates=candidates))
    hh=h.copy()
    if name.startswith('rank'):hh['probability']=hh.rank_score
    _,state=select(hh,thresholds,cooldown)
    return dict(name=name,threshold=thresholds,cooldown=cooldown,initial_state=state,calibration_diagnostics=diagnostics)

def apply(t,pol):
    q=t.copy();scored=q.copy()
    if pol['name'].startswith('rank'):scored['probability']=scored.rank_score
    ids,_=select(scored,pol['threshold'],pol['cooldown'],pol['initial_state']);q['candidate_signal']=q.index.isin(ids);q['policy']=pol['name'];return q

def main():
    raws=[]
    for branch in ('v3','tabm'):
        z=pd.read_csv(HERE/branch/'raw_predictions.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip');raws.append(z)
        if (HERE/branch/'warmup.csv.gz').exists():raws.append(pd.read_csv(HERE/branch/'warmup.csv.gz',parse_dates=['date','label_available_date'],float_precision='round_trip'))
    raw=pd.concat(raws,ignore_index=True);output=[];policies=[];calibrators=[]
    for (cid,horizon,year),g in raw.groupby(['config_id','train_horizon','fold_test_year']):
        cutoff=pd.Timestamp(year,1,1);v=g[g.split.eq('validation')].copy();h=g[g.split.eq('history')].copy();t=g[g.split.eq('test')].copy();w=g[g.split.eq('warmup')].copy()
        assert len(w)>0,('missing rank warmup',cid,horizon,year)
        assert v.label_available_date.lt(cutoff).all() and v.target.notna().all()
        if cid!='v3':
            for c,z in v.groupby('corridor'):
                cal=core.fit_platt_calibrator(z.raw_probability.to_numpy(),z.target)
                for f in (v,h,t,w):
                    m=f.corridor.eq(c);f.loc[m,'probability']=core.apply_platt(cal,f.loc[m,'raw_probability'].to_numpy())
                calibrators.append(dict(config_id=cid,train_horizon=int(horizon),year=int(year),corridor=c,method=cal.method,intercept=cal.intercept,slope=cal.slope))
        # Rank references use history+test scores and strictly preceding rows.
        allrank=ranked(pd.concat([w,h,t]).sort_values(['date','corridor']))
        v['rank_score']=v.index.map(lambda i: np.nan)
        rankmap=allrank.set_index(['date','corridor']).rank_score
        for f in (v,h,t):f['rank_score']=pd.MultiIndex.from_frame(f[['date','corridor']]).map(rankmap).to_numpy()
        for name in NAMES:
            pol=policy_fit(v,h,name);q=apply(t,pol);q['evaluation_horizon']=horizon;q['cohort']='native_matured';output.append(q)
            policies.append(dict(config_id=cid,train_horizon=int(horizon),year=int(year),**pol))
            if cid=='v3' and name=='strict05' and 'strict05_candidate_signal' in t:
                assert np.array_equal(q.candidate_signal,t.strict05_candidate_signal),('V3 strict parity',year,horizon)
    pred=pd.concat(output,ignore_index=True);pred.to_csv(HERE/'predictions.csv.gz',index=False);save('policies.json',policies);save('calibration.json',calibrators)
    summaries=[];annual=[];cells=[]
    for (cid,hor,pol),g in pred.groupby(['config_id','train_horizon','policy']):
        for scope in (['all5','KZT'] if g.corridor.nunique()==5 else ['KZT']):
            q=g if scope=='all5' else g[g.corridor.eq('KZT')]
            keys=dict(config_id=cid,train_horizon=hor,evaluation_horizon=hor,policy=pol,evaluation_scope=scope,cohort='native_matured')
            summaries.append(dict(**keys,period='2024-2026',**metric(q)))
            for year,z in q.groupby('fold_test_year'):annual.append(dict(**keys,year=year,**metric(z)))
        for (year,c),z in g.groupby(['fold_test_year','corridor']):cells.append(dict(config_id=cid,train_horizon=hor,evaluation_horizon=hor,policy=pol,year=year,corridor=c,**metric(z)))
    summary=pd.DataFrame(summaries);yearly=pd.DataFrame(annual);cell=pd.DataFrame(cells)
    summary.to_csv(HERE/'summary.csv',index=False);yearly.to_csv(HERE/'by_year.csv',index=False);cell.to_csv(HERE/'by_year_corridor.csv',index=False)
    # Cross-horizon rescore with frozen contacts, common mature H5 date cohort.
    targets=raw[(raw.config_id=='v3')&(raw.split=='test')].drop_duplicates(['train_horizon','fold_test_year','date','corridor'])
    cross=[]
    for (cid,train_h,pol,year),g in pred.groupby(['config_id','train_horizon','policy','fold_test_year']):
        h5=targets[(targets.train_horizon==5)&(targets.fold_test_year==year)]
        keys=['date','corridor'];common=g.merge(h5[keys],on=keys,how='inner')
        for eval_h in (3,5):
            labels=targets[(targets.train_horizon==eval_h)&(targets.fold_test_year==year)][keys+['target','forward_bps','regret_bps','label_available_date']]
            z=common.drop(columns=['target','forward_bps','regret_bps','label_available_date']).merge(labels,on=keys,how='inner')
            for scope in (['all5','KZT'] if z.corridor.nunique()==5 else ['KZT']):
                zz=z if scope=='all5' else z[z.corridor.eq('KZT')]
                cross.append(dict(config_id=cid,train_horizon=train_h,evaluation_horizon=eval_h,policy=pol,year=year,evaluation_scope=scope,cohort='common_H5_dates_frozen_contacts',**metric(zz,train_h==eval_h)))
    pd.DataFrame(cross).to_csv(HERE/'cross_horizon.csv',index=False)
    # Frozen deterministic selection: no weighted averaging can hide a failed cell.
    ranks=[];selections=[]
    for period,years in [('development_2024_2025',[2024,2025]),('retrospective_2024_2026',[2024,2025,2026])]:
        for scope in ('all5','KZT'):
            rows=[]
            for (cid,hor,pol),g in cell[cell.year.isin(years)].groupby(['config_id','train_horizon','policy']):
                if scope=='all5' and cid=='tabm_kzt':continue
                if scope=='KZT':g=g[g.corridor.eq('KZT')]
                coverage=g.week_coverage.to_numpy();lift=g.lift.fillna(0).to_numpy();utility=g.forward_delta_bps.fillna(-1e6).to_numpy()
                qualified=bool(np.all(coverage>=.8)&np.all(lift>=1.3)&np.all(utility>0))
                preferred=qualified and bool(np.all(coverage>=.9))
                shortfall=float(np.maximum(0,(.8-coverage)/.8).mean()+np.maximum(0,(1.3-lift)/1.3).mean()+np.mean(utility<=0))
                rows.append(dict(period=period,evaluation_scope=scope,config_id=cid,train_horizon=hor,policy=pol,qualified=qualified,preferred90=preferred,gate_shortfall=shortfall,min_cell_coverage=float(min(coverage)),min_cell_lift=float(min(lift)),min_cell_forward_delta_bps=float(min(utility)),mean_brier=float(g.brier.mean()),cells=len(g),passed_cells=int(((coverage>=.8)&(lift>=1.3)&(utility>0)).sum())))
            rows=sorted(rows,key=lambda x:(not x['qualified'],not x['preferred90'],x['gate_shortfall'] if not x['qualified'] else 0,-x['min_cell_lift'],-x['min_cell_forward_delta_bps'],x['mean_brier']))
            for i,r in enumerate(rows):r['rank']=i+1
            ranks.extend(rows);selected=dict(rows[0]);selected['selection_status']='qualified' if selected['qualified'] else 'fallback_no_qualifying_model'
            if period=='development_2024_2025':
                g=cell[(cell.config_id==selected['config_id'])&(cell.train_horizon==selected['train_horizon'])&(cell.policy==selected['policy'])&(cell.year==2026)]
                if scope=='KZT':g=g[g.corridor=='KZT']
                selected['audit2026_qualified']=bool((g.week_coverage.ge(.8)&g.lift.ge(1.3)&g.forward_delta_bps.gt(0)).all());selected['audit2026_cells']=g.to_dict('records')
            selections.append(selected)
    pd.DataFrame(ranks).to_csv(HERE/'selection_ranking.csv',index=False);save('selection.json',selections)
    intervals=[]
    # Compare frozen development choice to V3strict at same trained horizon.
    for chosen in selections:
        if chosen['period']!='development_2024_2025':continue
        scope=chosen['evaluation_scope'];hor=chosen['train_horizon']
        a=pred[(pred.config_id=='v3')&(pred.train_horizon==hor)&(pred.policy=='strict05')]
        b=pred[(pred.config_id==chosen['config_id'])&(pred.train_horizon==hor)&(pred.policy==chosen['policy'])]
        if scope=='KZT':a=a[a.corridor=='KZT'];b=b[b.corridor=='KZT']
        for year in (2024,2025,2026,'all'):
            aa=a if year=='all' else a[a.fold_test_year==year];bb=b if year=='all' else b[b.fold_test_year==year]
            intervals.append(dict(baseline='v3::strict05',candidate=chosen['config_id']+'::'+chosen['policy'],train_horizon=hor,evaluation_horizon=hor,evaluation_scope=scope,year=year,**paired(aa,bb)))
    pd.DataFrame(intervals).to_csv(HERE/'paired_intervals.csv',index=False)
    save('completion.json',dict(status='complete',models=sorted(pred.config_id.unique()),policy_count=len(NAMES),rows=len(pred),selection=selections))
    print(pd.DataFrame(selections)[['period','evaluation_scope','config_id','train_horizon','policy','selection_status','min_cell_coverage','min_cell_lift']].to_string(index=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);main()
