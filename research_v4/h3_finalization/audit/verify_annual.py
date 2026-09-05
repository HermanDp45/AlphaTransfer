"""Independent H3 history selection and annual scoring; no fitting."""
from pathlib import Path
import os,sys,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from research_v4.robust_selection.audit.verify_evaluation import m,reference_rank,selection,apply,equal,key
from research_v4.architecture_2023_2025.audit.verify import bootstrap
HERE=Path(__file__).resolve().parent.parent;OUT=HERE/'audit';checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return pd.read_csv(p,parse_dates=['date'],float_precision='round_trip',low_memory=False)
def check(name,ok,**info):
    checks.append(dict(name=name,status='PASS' if bool(ok) else 'FAIL',**info));assert ok,(name,info)
def main():
    pred=load(HERE/'predictions.csv.gz');old=load(ROOT/'research_v4/robust_selection/predictions.csv.gz')
    baseline=old[old.config_id.eq('tabm_kzt')&old.train_horizon.eq(3)]
    check('H3_KZT_only',set(pred.config_id)=={'tabm_kzt_120m','tabm_kzt_fullhistory'} and set(pred.train_horizon)=={3} and set(pred.corridor)=={'KZT'} and set(pred.fold_test_year)=={2024,2025,2026})
    for pol,g in baseline.groupby('policy'):
        q=pred[pred.config_id.eq('tabm_kzt_120m')&pred.policy.eq(pol)].sort_values(['date','corridor']);g=g.sort_values(['date','corridor'])
        check('exact_old_120m_'+pol,key(q)==key(g) and all(np.array_equal(q[c],g[c],equal_nan=True) for c in ('probability','raw_probability','rank_score','candidate_signal','target','forward_bps')))
    raws=[]
    for folder,isbase in [(ROOT/'research_v4/robust_selection/tabm',True),(HERE/'long_history',False)]:
        for file in ('raw_predictions.csv.gz','warmup.csv.gz'):
            q=load(folder/file)
            if isbase:q=q[q.config_id.eq('tabm_kzt')&q.train_horizon.eq(3)].copy();q['config_id']='tabm_kzt_120m'
            raws.append(q)
    raw=pd.concat(raws,ignore_index=True);raw['label_available_date']=pd.to_datetime(raw.label_available_date)
    calibrations=json.loads((HERE/'calibration.json').read_text());policies=json.loads((HERE/'policies.json').read_text());maxp=maxrank=0.;rows=0
    for (cid,year),g in raw.groupby(['config_id','fold_test_year']):
        g=g.copy();cal=next(c for c in calibrations if c['config_id']==cid and c['year']==year);p=g.raw_probability.to_numpy()
        if cal['method']!='identity':p=np.clip(p,1e-6,1-1e-6);p=expit(cal['intercept']+cal['slope']*np.log(p/(1-p)))
        g['probability']=p;v=g[g.split.eq('validation')];h=g[g.split.eq('history')];w=g[g.split.eq('warmup')]
        check(f'maturity_{cid}_{year}',len(w)==63 and w.date.lt(pd.Timestamp(year-1,1,1)).all() and v.label_available_date.lt(pd.Timestamp(year,1,1)).all() and h.loc[~h.label_available_date.lt(pd.Timestamp(year,1,1)),['target','forward_bps','regret_bps']].isna().all().all())
        ref=g[g.split.isin(['warmup','history','test'])].copy();check(f'unique_rank_reference_{cid}_{year}',not ref.duplicated(['date','corridor']).any())
        ref['rank_score']=reference_rank(ref);rmap=ref.set_index('date').rank_score;g['rank_score']=g.date.map(rmap)
        h=g[g.split.eq('history')];t=g[g.split.eq('test')]
        for pol in [q for q in policies if q['config_id']==cid and q['year']==year]:
            rank=pol['name'].startswith('rank');_,state=selection(h,pol['threshold'],pol['cooldown'],rank=rank)
            q,_=apply(t,pol['threshold'],pol['cooldown'],state,rank);q=q.sort_values('date')
            s=pred[pred.config_id.eq(cid)&pred.fold_test_year.eq(year)&pred.policy.eq(pol['name'])].sort_values('date')
            check(f'policy_replay_{cid}_{year}_{pol["name"]}',key(q)==key(s) and state==pol['initial_state'] and np.array_equal(q.candidate_signal,s.candidate_signal))
            maxp=max(maxp,float(np.max(np.abs(q.probability.to_numpy()-s.probability.to_numpy()))));maxrank=max(maxrank,float(np.nanmax(np.abs(q.rank_score.to_numpy()-s.rank_score.to_numpy()))));rows+=len(s)
    check('calibrated_probability_and_rank_replay',maxp==0 and maxrank==0,rows=rows,max_probability_error=maxp,max_rank_error=maxrank)
    for filename in ('summary.csv','by_year.csv'):
        table=pd.read_csv(HERE/filename,float_precision='round_trip');errors=[]
        for row in table.to_dict('records'):
            g=pred[pred.config_id.eq(row['config_id'])&pred.policy.eq(row['policy'])]
            if 'year' in row:g=g[g.fold_test_year.eq(row['year'])]
            for k,v in m(g).items():
                if not equal(v,row[k]):errors.append([row['config_id'],row['policy'],k,v,row[k]])
        check('independent_'+filename,not errors,rows=len(table),errors=errors)
    annual=pd.read_csv(HERE/'by_year.csv',float_precision='round_trip');ranking=[]
    for (cid,pol),g in annual.groupby(['config_id','policy']):
        assert set(g.year)=={2024,2025,2026};cov=g.week_coverage.to_numpy();li=np.nan_to_num(g.lift.to_numpy());ut=g.forward_delta_bps.to_numpy()
        qual=bool(all(c>=.8 and l>=1.3 and u>0 for c,l,u in zip(cov,li,ut)));preferred=bool(qual and min(cov)>=.9)
        short=sum(max(0,(.8-c)/.8) for c in cov)/3+sum(max(0,(1.3-l)/1.3) for l in li)/3+sum(u<=0 for u in ut)/3
        ranking.append(dict(config_id=cid,policy=pol,qualified=qual,preferred90=preferred,gate_shortfall=short,min_coverage=min(cov),min_lift=min(li),min_utility=min(ut),brier=g.brier.mean()))
    ranking.sort(key=lambda x:(not x['qualified'],not x['preferred90'],x['gate_shortfall'] if not x['qualified'] else 0,-x['min_lift'],-x['min_utility'],x['brier']))
    saved=pd.read_csv(HERE/'ranking.csv',float_precision='round_trip').sort_values('rank');chosen=json.loads((HERE/'selection.json').read_text())
    ids=lambda x:(x['config_id'],x['policy'])
    check('full_history_promotion_ranking',[ids(x) for x in ranking]==[ids(x) for x in saved.to_dict('records')] and ids(ranking[0])==ids(chosen['chosen']) and ranking[0]['qualified']==chosen['chosen']['qualified'],candidates=len(ranking),winner=ids(ranking[0]))
    ci=pd.read_csv(HERE/'paired_intervals.csv',float_precision='round_trip');diagnostics=[]
    for row in ci.to_dict('records'):
        a=pred[pred.config_id.eq('tabm_kzt_120m')&pred.policy.eq(row['policy'])];b=pred[pred.config_id.eq('tabm_kzt_fullhistory')&pred.policy.eq(row['policy'])];d=bootstrap(a,b)
        for met,x in d.items():
            prefix='' if met=='brier' else met+'_'
            for side in ('low','high'):assert equal(x[side],row[prefix+'ci_'+side]),(row['policy'],met,side)
        diagnostics.append(dict(policy=row['policy'],metrics=d))
    check('all_same_policy_month_intervals',True,comparisons=len(ci));(OUT/'annual_bootstrap_finite_draws.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    paths=['protocol.json','evaluate.py','predictions.csv.gz','summary.csv','by_year.csv','ranking.csv','selection.json','policies.json','calibration.json','paired_intervals.csv','long_history/raw_predictions.csv.gz','long_history/warmup.csv.gz','long_history/experiment.py','long_history/receipts.json']
    receipt=dict(status='PASS',passed=len(checks),checks=checks,source_sha256={p:sha(HERE/p) for p in paths},audit_sha256=sha(__file__),scope='Independent annual raw-output calibration/rank/policy replay, old120m exact parity, metric aggregation and retrospective history ranking; no model fitting.')
    (OUT/'annual_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks))))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
