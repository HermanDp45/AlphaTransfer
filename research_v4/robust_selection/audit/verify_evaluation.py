"""Read-only numerical audit of robust_selection. No model fitting."""
from pathlib import Path
import os,sys,json,hashlib,copy
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
from research_v4.robust_selection import evaluate as implementation
from research_v4.architecture_2023_2025.audit.verify import bootstrap
HERE=Path(__file__).resolve().parent.parent;OUT=HERE/'audit';CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return pd.read_csv(p,parse_dates=['date'],float_precision='round_trip',low_memory=False)
def check(name,ok=True,**info):
    CHECKS.append(dict(name=name,status='PASS' if bool(ok) else 'FAIL',**info))
    assert ok,(name,info)
def key(f):return list(zip(f.date,f.corridor))
def equal(a,b):return np.isclose(a,b,rtol=1e-11,atol=1e-10,equal_nan=True)
def m(g,proper=True):
    obs=expected=selected=fw=weeks=covered=0;silent=0;maxrun=0;maxcontact=0;cellcover=[]
    for _,z in g.groupby(['fold_test_year','corridor']):
        s=z[z.candidate_signal];obs+=s.target.sum();selected+=len(s)
        expected+=len(s)*z.target.mean();fw+=(s.forward_bps-z.forward_bps.mean()).sum()
        pp=z.date.dt.to_period('W-SUN');c=z.groupby(pp).candidate_signal.sum().reindex(pd.period_range(pp.min(),pp.max(),freq='W-SUN'),fill_value=0)
        weeks+=len(c);covered+=c.between(1,2).sum();silent+=c.eq(0).sum();cellcover.append(c.between(1,2).mean())
        maxcontact=max(maxcontact,int(c.max()));run=0
        for n in c:
            run=run+1 if n==0 else 0;maxrun=max(maxrun,run)
    sel=g[g.candidate_signal]
    return dict(rows=len(g),dates=g.date.nunique(),signals=selected,hits=int(obs),base_hit=g.target.mean(),hit_rate=sel.target.mean(),lift=obs/expected if expected else np.nan,
        forward_delta_bps=fw/selected if selected else np.nan,forward_signal_bps=sel.forward_bps.mean(),regret_bps=sel.regret_bps.mean(),
        week_cells=weeks,weeks_with_1_2=covered,week_coverage=covered/weeks,mean_cell_week_coverage=np.mean(cellcover),signals_per_corridor_week=selected/weeks,silent_weeks=silent,max_week_signals=maxcontact,max_silent_week_run=maxrun,min_year_corridor_coverage=min(cellcover),
        brier=float(np.mean((g.probability-g.target)**2)) if proper else np.nan,raw_brier=float(np.mean((g.raw_probability-g.target)**2)) if proper else np.nan,
        auc=roc_auc_score(g.target,g.probability) if proper and g.target.nunique()>1 else np.nan,log_loss=log_loss(g.target,g.probability,labels=[0,1]) if proper else np.nan)
def reference_rank(g):
    out=pd.Series(np.nan,index=g.index)
    for _,z in g.groupby('corridor'):
        past=[]
        for row in z.sort_values('date').itertuples():
            assert np.isfinite(row.probability)
            if len(past)>=20:
                less=sum(x<row.probability for x in past);ties=sum(x==row.probability for x in past)
                out.loc[row.Index]=(less+.5*ties)/len(past)
            past.append(row.probability);past=past[-63:]
    return out
def selection(g,threshold,cd,initial=None,rank=False):
    state=copy.deepcopy(initial or {});chosen=[]
    for r in g.sort_values(['date','corridor']).itertuples():
        week=str(r.date.to_period('W-SUN'));c=r.corridor
        st=state.setdefault(c,dict(last=-10000,week=week,count=0))
        if week!=st['week']:st.update(week=week,count=0)
        p=r.rank_score if rank else r.probability;t=threshold[c] if isinstance(threshold,dict) else threshold
        if np.isfinite(p) and p>=t and r.session_ordinal-st['last']>cd and st['count']<2:
            chosen.append(r.Index);st.update(last=int(r.session_ordinal),count=st['count']+1)
    return np.array(chosen,dtype=int),state
def apply(g,threshold,cd,initial=None,rank=False):
    q=g.copy();ids,st=selection(q,threshold,cd,initial,rank);q['candidate_signal']=q.index.isin(ids);return q,st

def audit_policy(pred):
    sources=[HERE/x/'raw_predictions.csv.gz' for x in ('v3','tabm')]+[HERE/'v3/warmup.csv.gz',HERE/'tabm/warmup.csv.gz']
    raw=pd.concat([load(p) for p in sources],ignore_index=True);raw['label_available_date']=pd.to_datetime(raw.label_available_date)
    cals=json.loads((HERE/'calibration.json').read_text());policies=json.loads((HERE/'policies.json').read_text())
    lookup={(x['config_id'],x['train_horizon'],x['year'],x['corridor']):x for x in cals}
    replayrows=0;maximum=0.;rankerror=0.;chosen_errors=[];future_checks=0
    for (cid,hor,year),g in raw.groupby(['config_id','train_horizon','fold_test_year']):
        g=g.copy();cutoff=pd.Timestamp(year,1,1);calstart=pd.Timestamp(year-1,1,1)
        w=g[g.split.eq('warmup')];v=g[g.split.eq('validation')];h=g[g.split.eq('history')];t=g[g.split.eq('test')]
        check(f'maturity_{cid}_{hor}_{year}',v.label_available_date.lt(cutoff).all() and t.label_available_date.notna().all() and w.date.lt(calstart).all() and t.date.ge(cutoff).all() and t.label_available_date.lt(pd.Timestamp(year+1,1,1)).all())
        check(f'masked_warmup_history_{cid}_{hor}_{year}',w[['target','forward_bps','regret_bps']].isna().all().all() and h.loc[~h.label_available_date.lt(cutoff),['target','forward_bps','regret_bps']].isna().all().all())
        for c,z in g.groupby('corridor'):
            if cid=='v3':continue
            cal=lookup[cid,int(hor),int(year),c];p=z.raw_probability.to_numpy()
            if cal['method']!='identity':
                p=np.clip(p,1e-6,1-1e-6);p=expit(cal['intercept']+cal['slope']*np.log(p/(1-p)))
            g.loc[z.index,'probability']=p
        allrank=g[g.split.isin(['warmup','history','test'])].copy().sort_values(['date','corridor'])
        check(f'unique_rank_reference_{cid}_{hor}_{year}',not allrank.duplicated(['date','corridor']).any() and all(len(x)==63 for _,x in w.groupby('corridor')))
        ref=reference_rank(allrank);actual=implementation.ranked(allrank).rank_score
        check(f'rank_formula_{cid}_{hor}_{year}',np.array_equal(ref,actual,equal_nan=True))
        day=sorted(t.date.unique())[len(t.date.unique())//2]
        poisoned=allrank.copy();poisoned.loc[poisoned.date.gt(day),'probability']=np.linspace(0,1,poisoned.date.gt(day).sum())
        poisonrank=implementation.ranked(poisoned).rank_score
        prefix=implementation.ranked(allrank[allrank.date.le(day)]).rank_score
        check(f'future_poison_prefix_{cid}_{hor}_{year}',np.array_equal(actual.loc[prefix.index],prefix,equal_nan=True) and np.array_equal(actual[allrank.date.le(day)],poisonrank[allrank.date.le(day)],equal_nan=True))
        future_checks+=1;allrank['rank_score']=ref
        rankmap=allrank.set_index(['date','corridor']).rank_score
        g['rank_score']=[rankmap.loc[(r.date,r.corridor)] for r in g.itertuples()]
        w,v,h,t=[g[g.split.eq(s)].copy() for s in ('warmup','validation','history','test')]
        for pol in (p for p in policies if p['config_id']==cid and p['train_horizon']==hor and p['year']==year):
            name=pol['name'];rank=name.startswith('rank');cd=3 if name=='strict05' else 2
            thresholds={c:.5 for c in v.corridor.unique()};diagnostics=[]
            if name!='strict05':
                target=.8 if '80' in name else .85 if '85' in name else .9
                groups=[('shared',v)] if name.endswith('shared') else list(v.groupby('corridor'))
                for c,z in groups:
                    candidates=[]
                    for q in np.arange(0,.81,.05):
                        th=float(q) if rank else float(z.probability.quantile(q));selected,_=apply(z,th,cd,rank=rank)
                        mm=m(selected,False);cov=min(m(x,False)['week_coverage'] for _,x in selected.groupby('corridor'))
                        candidates.append(dict(q=float(q),threshold=th,coverage=cov,lift=mm['lift'],forward_delta_bps=mm['forward_delta_bps'],feasible=cov>=target))
                    feasible=[x for x in candidates if x['feasible']]
                    choice=max(feasible,key=lambda x:(x['lift'],x['forward_delta_bps'],x['coverage'])) if feasible else max(candidates,key=lambda x:(x['coverage'],x['lift']))
                    for ccc in z.corridor.unique():thresholds[ccc]=choice['threshold']
                    diagnostics.append(dict(corridor=c,chosen=choice))
                check(f'past_calibration_threshold_{cid}_{hor}_{year}_{name}',all(equal(thresholds[c],pol['threshold'][c]) for c in thresholds))
                check(f'calibration_feasibility_{cid}_{hor}_{year}_{name}',all(x['chosen']['feasible']==d['chosen']['feasible'] for x,d in zip(diagnostics,pol['calibration_diagnostics'])))
            if name.endswith('shared'):check(f'shared_parameter_{cid}_{hor}_{year}',len(set(pol['threshold'].values()))==1)
            _,state=selection(h,thresholds,cd,rank=rank)
            q,_=apply(t,thresholds,cd,state,rank)
            saved=pred[pred.config_id.eq(cid)&pred.train_horizon.eq(hor)&pred.fold_test_year.eq(year)&pred.policy.eq(name)].sort_values(['date','corridor'])
            q=q.sort_values(['date','corridor']);replayrows+=len(q)
            check(f'test_contact_replay_{cid}_{hor}_{year}_{name}',key(q)==key(saved) and np.array_equal(q.candidate_signal,saved.candidate_signal) and state==pol['initial_state'])
            maximum=max(maximum,float(np.max(np.abs(q.probability.to_numpy()-saved.probability.to_numpy()))))
            rankerror=max(rankerror,float(np.nanmax(np.abs(q.rank_score.to_numpy()-saved.rank_score.to_numpy()))))
    check('all_calibrated_probabilities_and_ranks',maximum<1e-12 and rankerror==0,rows=replayrows,max_probability_error=maximum,max_rank_error=rankerror,future_poison_groups=future_checks)
    return raw

def audit_metrics(pred,raw):
    errors=[]
    for filename in ('summary.csv','by_year.csv','by_year_corridor.csv'):
        table=pd.read_csv(HERE/filename,float_precision='round_trip')
        for row in table.to_dict('records'):
            g=pred[pred.config_id.eq(row['config_id'])&pred.train_horizon.eq(row['train_horizon'])&pred.policy.eq(row['policy'])]
            if 'year' in row:g=g[g.fold_test_year.eq(row['year'])]
            if 'corridor' in row:g=g[g.corridor.eq(row['corridor'])]
            elif row['evaluation_scope']=='KZT':g=g[g.corridor.eq('KZT')]
            for k,v in m(g).items():
                if not equal(v,row[k]):errors.append([filename,row['config_id'],row['train_horizon'],row['policy'],k,v,row[k]])
        check('metrics_'+filename,not errors,rows=len(table),errors=errors)
    cross=pd.read_csv(HERE/'cross_horizon.csv',float_precision='round_trip')
    targets=raw[raw.config_id.eq('v3')&raw.split.eq('test')]
    cross_errors=[]
    for row in cross.to_dict('records'):
        g=pred[pred.config_id.eq(row['config_id'])&pred.train_horizon.eq(row['train_horizon'])&pred.fold_test_year.eq(row['year'])&pred.policy.eq(row['policy'])]
        common=set(key(targets[targets.train_horizon.eq(5)&targets.fold_test_year.eq(row['year'])]));g=g[[k in common for k in key(g)]]
        lab=targets[targets.train_horizon.eq(row['evaluation_horizon'])&targets.fold_test_year.eq(row['year'])]
        cols=['target','forward_bps','regret_bps'];g=g.drop(columns=cols).merge(lab[['date','corridor',*cols]],on=['date','corridor'],validate='1:1')
        if row['evaluation_scope']=='KZT':g=g[g.corridor.eq('KZT')]
        for k,v in m(g,row['train_horizon']==row['evaluation_horizon']).items():
            if not equal(v,row[k]):cross_errors.append([row['config_id'],row['train_horizon'],row['evaluation_horizon'],row['year'],row['policy'],k,v,row[k]])
    check('cross_horizon_common_cohort_proper_scores',not cross_errors,rows=len(cross),errors=cross_errors)

def audit_selection(pred):
    choices=[];ranking=pd.read_csv(HERE/'selection_ranking.csv',float_precision='round_trip')
    selected=json.loads((HERE/'selection.json').read_text())
    cells=pd.read_csv(HERE/'by_year_corridor.csv',float_precision='round_trip')
    for period,years in [('development_2024_2025',[2024,2025]),('retrospective_2024_2026',[2024,2025,2026])]:
        for scope in ('all5','KZT'):
            rows=[]
            for (cid,hor,pol),g in cells[cells.year.isin(years)].groupby(['config_id','train_horizon','policy']):
                if scope=='all5' and cid=='tabm_kzt':continue
                if scope=='KZT':g=g[g.corridor.eq('KZT')]
                assert len(g)==len(years)*(5 if scope=='all5' else 1)
                cov=g.week_coverage.to_numpy();li=np.nan_to_num(g.lift.to_numpy(),nan=0);ut=np.nan_to_num(g.forward_delta_bps.to_numpy(),nan=-1e6)
                qualified=all(c>=.8 and l>=1.3 and u>0 for c,l,u in zip(cov,li,ut));pref=qualified and min(cov)>=.9
                shortfall=sum(max(0,(.8-c)/.8) for c in cov)/len(cov)+sum(max(0,(1.3-l)/1.3) for l in li)/len(li)+sum(u<=0 for u in ut)/len(ut)
                rows.append(dict(config_id=cid,train_horizon=int(hor),policy=pol,qualified=qualified,preferred90=pref,gate_shortfall=shortfall,min_cell_coverage=min(cov),min_cell_lift=min(li),min_cell_forward_delta_bps=min(ut),mean_brier=g.brier.mean()))
            rows.sort(key=lambda x:(not x['qualified'],not x['preferred90'],x['gate_shortfall'] if not x['qualified'] else 0,-x['min_cell_lift'],-x['min_cell_forward_delta_bps'],x['mean_brier']))
            saved=ranking[ranking.period.eq(period)&ranking.evaluation_scope.eq(scope)].sort_values('rank')
            ids=lambda r:(r['config_id'],int(r['train_horizon']),r['policy'])
            check(f'complete_selector_ranking_{period}_{scope}',[ids(r) for r in rows]==[ids(r) for r in saved.to_dict('records')],candidates=len(rows),qualified=sum(x['qualified'] for x in rows))
            chosen=next(s for s in selected if s['period']==period and s['evaluation_scope']==scope)
            check(f'winner_{period}_{scope}',ids(rows[0])==ids(chosen) and rows[0]['qualified']==chosen['qualified'] and ((chosen['selection_status']=='qualified')==chosen['qualified']))
            if period.startswith('development'):
                g=cells[cells.config_id.eq(chosen['config_id'])&cells.train_horizon.eq(chosen['train_horizon'])&cells.policy.eq(chosen['policy'])&cells.year.eq(2026)]
                if scope=='KZT':g=g[g.corridor.eq('KZT')]
                check('frozen_2026_audit_'+scope,len(g)==(5 if scope=='all5' else 1) and bool((g.week_coverage.ge(.8)&g.lift.ge(1.3)&g.forward_delta_bps.gt(0)).all())==chosen['audit2026_qualified'])
            choices.append(dict(period=period,scope=scope,**rows[0],qualified_count=sum(x['qualified'] for x in rows)))
    (OUT/'independent_selection.json').write_text(json.dumps(choices,indent=2,default=str)+'\n')
    intervals=pd.read_csv(HERE/'paired_intervals.csv',float_precision='round_trip');diag=[];errors=[]
    for row in intervals.to_dict('records'):
        cid,pol=row['candidate'].split('::');a=pred[pred.config_id.eq('v3')&pred.train_horizon.eq(row['train_horizon'])&pred.policy.eq('strict05')];b=pred[pred.config_id.eq(cid)&pred.train_horizon.eq(row['train_horizon'])&pred.policy.eq(pol)]
        if row['evaluation_scope']=='KZT':a=a[a.corridor.eq('KZT')];b=b[b.corridor.eq('KZT')]
        if str(row['year'])!='all':a=a[a.fold_test_year.eq(int(row['year']))];b=b[b.fold_test_year.eq(int(row['year']))]
        d=bootstrap(a,b)
        for met,val in d.items():
            prefix='' if met=='brier' else met+'_'
            for side in ('low','high'):
                if not equal(val[side],row[prefix+'ci_'+side]):errors.append([row['candidate'],row['year'],met,side,val[side],row[prefix+'ci_'+side]])
        diag.append(dict(candidate=row['candidate'],horizon=row['train_horizon'],scope=row['evaluation_scope'],year=row['year'],metrics=d))
    check('paired_month_bootstrap_intervals',not errors,comparisons=len(intervals),errors=errors)
    (OUT/'bootstrap_finite_draws.json').write_text(json.dumps(diag,indent=2,default=str)+'\n')

def main():
    pred=load(HERE/'predictions.csv.gz');check('matrix',set(pred.config_id)=={'v3','tabm_kzt','tabm_pooled'} and set(pred.train_horizon)=={3,5} and set(pred.fold_test_year)=={2024,2025,2026} and pred.candidate_signal.dtype==bool)
    with threadpool_limits(limits=1):
        raw=audit_policy(pred);audit_metrics(pred,raw);audit_selection(pred)
    paths=['protocol.json','evaluate.py','report.py','predictions.csv.gz','summary.csv','by_year.csv','by_year_corridor.csv','cross_horizon.csv','policies.json','calibration.json','selection.json','selection_ranking.csv','paired_intervals.csv','tabm/raw_predictions.csv.gz','tabm/warmup.csv.gz','v3/raw_predictions.csv.gz','v3/warmup.csv.gz']
    receipt=dict(status='PASS',passed=len(CHECKS),checks=CHECKS,source_sha256={p:sha(HERE/p) for p in paths},audit_sha256=sha(__file__),scope='Independent read-only numerical protocol, rank, selection and interval audit; no model fitting.')
    (OUT/'evaluation_verification.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n');print(json.dumps(dict(status='PASS',checks=len(CHECKS))))
if __name__=='__main__':main()
