"""Independent artifact audit. No estimator fit, source changes, or network calls."""
from pathlib import Path
import os, sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import json,pickle,hashlib,warnings
import numpy as np,pandas as pd,torch,joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
from research_v4.final_sprint.tabm import experiment as neural

HERE=Path(__file__).resolve().parent.parent
OUT=HERE/'audit'
checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,condition=True,**details):
    checks.append(dict(name=name,status='PASS' if bool(condition) else 'FAIL',**details))
    assert condition,(name,details)
def read(p):return pd.read_csv(p,parse_dates=['date'],float_precision='round_trip',low_memory=False)
def keys(x):
    q=x[['date','corridor']].reset_index(drop=True).copy()
    q['date']=q.date.astype('datetime64[ns]')
    q['corridor']=q.corridor.astype(object)
    return q
def fp(x):return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).to_numpy().tobytes()).hexdigest()
def matrix(model,f):
    n,c=model.encode(f,model.pre)
    return np.column_stack([n.numpy(),np.eye(5)[c.numpy().ravel()]])

def independent_metrics(g):
    hits=expected=selected=weighted_forward=weeks=covered=0
    for _,z in g.groupby(['fold_test_year','corridor']):
        s=z[z.candidate_signal]
        hits+=s.target.sum();selected+=len(s);expected+=len(s)*z.target.mean()
        weighted_forward+=(s.forward_bps-z.forward_bps.mean()).sum()
        period=z.date.dt.to_period('W-SUN')
        counts=z.groupby(period).candidate_signal.sum().reindex(pd.period_range(period.min(),period.max(),freq='W-SUN'),fill_value=0)
        weeks+=len(counts);covered+=counts.between(1,2).sum()
    return dict(rows=len(g),signals=selected,brier=float(((g.probability-g.target)**2).mean()),
                lift=hits/expected if expected else np.nan,
                forward_delta_bps=weighted_forward/selected if selected else np.nan,
                week_cells=weeks,weeks_with_1_2=covered,week_coverage=covered/weeks)

def bootstrap(a,b,reps=10000):
    """Independent vectorized sufficient statistics, common resampled months."""
    a=a.sort_values(['date','corridor']).reset_index(drop=True)
    b=b.sort_values(['date','corridor']).reset_index(drop=True)
    assert keys(a).equals(keys(b)) and np.array_equal(a.target,b.target)
    assert np.array_equal(a.fold_test_year,b.fold_test_year)
    z=a[['date','corridor','fold_test_year','target','forward_bps']].copy()
    z['block']=z.date.dt.to_period('M').astype(str)
    z['error_delta']=(b.probability-b.target)**2-(a.probability-a.target)**2
    z['a']=a.candidate_signal.astype(float);z['b']=b.candidate_signal.astype(float)
    blocks=list(z.groupby(['fold_test_year','block']).groups)
    cells=list(z.groupby(['fold_test_year','corridor']).groups)
    weights=np.zeros((reps,len(blocks)),dtype=int);rng=np.random.default_rng(20260905)
    for y in sorted(z.fold_test_year.unique()):
        idx=np.array([i for i,key in enumerate(blocks) if key[0]==y])
        draws=rng.integers(len(idx),size=(reps,len(idx)))
        # One multinomial count row per replicate; all corridors share its months.
        for r,d in enumerate(draws):weights[r,idx]=np.bincount(d,minlength=len(idx))
    features=['n','hit','fwd','delta','an','ah','af','bn','bh','bf']
    z['n']=1.;z['hit']=z.target;z['fwd']=z.forward_bps;z['delta']=z.error_delta
    for p in ('a','b'):
        z[p+'n']=z[p];z[p+'h']=z[p]*z.target;z[p+'f']=z[p]*z.forward_bps
    stat=np.zeros((len(blocks),len(cells),len(features)))
    for ci,(y,c) in enumerate(cells):
        zz=z[z.fold_test_year.eq(y)&z.corridor.eq(c)].groupby(['fold_test_year','block'])[features].sum()
        for bi,key in enumerate(blocks):
            if key in zz.index:stat[bi,ci]=zz.loc[key].to_numpy()
    totals=(weights@stat.reshape(len(blocks),-1)).reshape(reps,len(cells),len(features))
    n,h,f=totals[:,:,0],totals[:,:,1],totals[:,:,2]
    base_h=np.divide(h,n,out=np.zeros_like(h),where=n>0)
    base_f=np.divide(f,n,out=np.zeros_like(f),where=n>0)
    dist={'brier':totals[:,:,3].sum(axis=1)/n.sum(axis=1)}
    vals=[]
    for j in (4,7):
        sn,sh,sf=totals[:,:,j],totals[:,:,j+1],totals[:,:,j+2]
        expected=(sn*base_h).sum(axis=1);count=sn.sum(axis=1)
        lift=np.divide(sh.sum(axis=1),expected,out=np.full(reps,np.nan),where=expected>0)
        utility=np.divide((sf-sn*base_f).sum(axis=1),count,out=np.full(reps,np.nan),where=count>0)
        vals.append((lift,utility))
    dist['lift']=vals[1][0]-vals[0][0]
    dist['forward_delta_bps']=vals[1][1]-vals[0][1]
    return {k:dict(nonfinite=int((~np.isfinite(v)).sum()),finite=int(np.isfinite(v).sum()),
                   low=float(np.nanquantile(v,.025)) if np.isfinite(v).any() else np.nan,
                   high=float(np.nanquantile(v,.975)) if np.isfinite(v).any() else np.nan) for k,v in dist.items()}

def audit_local():
    views,_=pickle.loads((ROOT/'research_v4/final_sprint/views.pkl').read_bytes())
    panel=views['2010-01-01',24,1]
    check('binary_source_hash',sha(ROOT/'research_v4/final_sprint/views.pkl')=='b3cda0736e28200e465f980293f9c7dafd69b834b274788549fb0ab7c6375961')
    raw=read(HERE/'local/raw_predictions.csv.gz')
    raw['label_available_date']=pd.to_datetime(raw.label_available_date)
    receipts=json.loads((HERE/'local/receipts.json').read_text())
    check('local_matrix',len(receipts)==18 and len(raw.config_id.unique())==7,
          model_year_pairs=len(receipts),ensemble_year_pairs=3)
    actual=panel.sort_values(['corridor','date']).groupby('corridor').date.shift(-5)
    check('actual_fifth_future_date',panel.label_available_date.equals(actual.reindex(panel.index)))
    for year in (2023,2024,2025):
        cutoff=pd.Timestamp(year,1,1);calstart=cutoff-pd.DateOffset(years=1)
        tr,va,te,hist=neural.split(views,dict(scope='kzt',months=120,seed_index=2),cutoff)
        parts=dict(validation=va,history=hist,test=te)
        check(f'maturity_{year}',tr.label_available_date.max()<calstart and va.label_available_date.max()<cutoff and te.label_available_date.max()<pd.Timestamp(year+1,1,1)
              and set(tr.corridor)=={'KZT'} and set(va.corridor)==set(te.corridor)=={'KZT'},
              train_rows=len(tr),validation_rows=len(va),test_rows=len(te),train_label_max=str(tr.label_available_date.max()),validation_label_max=str(va.label_available_date.max()))
        for r in (x for x in receipts if x['year']==year):
            cid=r['model_id'];cols=r['features'];d=HERE/'local'/f'{cid}_{year}'
            check(f'train_fingerprint_{cid}_{year}',r['train_fingerprint']==fp(tr[['date','corridor',*cols]]) and r['test_keys_fingerprint']==fp(te[['date','corridor','target']]))
            seed=neural.SEED+(r['seed'] if 'tabm' in cid else 2)*100
            nn=neural.Neural(cols,seed)
            if 'tabm' in cid:
                nn.load(d);meta=json.loads((d/'model.json').read_text())
                check(f'checkpoint_metadata_{cid}_{year}',meta['features']==cols and meta['weights_sha256']==sha(d/'weights.pt') and meta['preprocessor_sha256']==sha(d/'preprocess.joblib') and pd.Timestamp(meta['inner_label_max'])<pd.Timestamp(meta['inner_validation_min']))
                if year==2025 and r['family']=='full33':
                    old=ROOT/'research_v4/final_sprint/tabm/checkpoints'/f'tabm_periodic_kzt_120m_s{r["seed"]}_2025-01-01'
                    oldsplit=json.loads((old/'split.json').read_text())
                    check(f'old2025_reuse_{cid}',all(sha(old/f)==sha(d/f) for f in ('weights.pt','preprocess.joblib','model.json')) and oldsplit['features_fingerprint']==r['train_fingerprint'])
                    oldraw=joblib.load(old/'raw_predictions.joblib')['normal']
                    current=raw[raw.config_id.eq(cid)&raw.fold_test_year.eq(year)&raw.split.eq('test')].reset_index(drop=True)
                    dmax=float(np.max(np.abs(oldraw.raw_probability.to_numpy()-current.raw_probability.to_numpy())))
                    check(f'old2025_raw_reuse_{cid}',keys(oldraw).equals(keys(current)) and dmax<2e-7,max_abs_error=dmax,rows=len(current))
            else:
                nn.pre=joblib.load(d/'preprocess.joblib');tree=joblib.load(d/'classifier.joblib')
                sibling=HERE/'local'/f'kzt_tabm_{r["family"]}_s2_{year}'
                cfg=tree.get_params()
                check(f'matched_preprocessor_{cid}_{year}',sha(d/'preprocess.joblib')==sha(sibling/'preprocess.joblib') and cfg['max_iter']==120 and cfg['max_depth']==2 and cfg['l2_regularization']==2 and not cfg['early_stopping'])
            # Fitted quantiles independently calculated without calling fit.
            train=tr[cols].to_numpy(float)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(train,axis=0)
            med=np.nan_to_num(med,nan=0.)
            pre=nn.pre;imputed=np.where(np.isnan(train),med,train)
            quant=np.nanpercentile(imputed,pre.named_steps['gaussian'].references_*100,axis=0)
            check(f'train_only_statistics_{cid}_{year}',np.array_equal(med,pre.named_steps['impute'].statistics_) and np.allclose(quant,pre.named_steps['gaussian'].quantiles_,rtol=0,atol=1e-14))
            maxdiff=0.;nrows=0
            for split,part in parts.items():
                saved=raw[raw.config_id.eq(cid)&raw.fold_test_year.eq(year)&raw.split.eq(split)].reset_index(drop=True)
                check(f'keys_{cid}_{year}_{split}',keys(saved).equals(keys(part)))
                pred=nn.predict(part) if 'tabm' in cid else tree.predict_proba(matrix(nn,part))[:,1]
                diff=float(np.max(np.abs(pred-saved.raw_probability)));maxdiff=max(maxdiff,diff);nrows+=len(saved)
                if split=='history':
                    immature=part.label_available_date.ge(cutoff)|part.label_available_date.isna()
                    check(f'immature_mask_{cid}_{year}',saved.loc[immature.to_numpy(),['target','forward_bps','symmetric_bps','regret_bps']].isna().all().all())
            check(f'checkpoint_raw_replay_{cid}_{year}',maxdiff<2e-7,rows=nrows,max_abs_error=maxdiff)
    for (y,s),g in raw[raw.config_id.str.contains('full33')].groupby(['fold_test_year','split']):
        a=[g[g.config_id.eq(f'kzt_tabm_full33_s{k}')].sort_values(['date','corridor']) for k in (0,1,2)]
        ens=g[g.config_id.eq('kzt_tabm_full33_ensemble3')].sort_values(['date','corridor'])
        diff=float(np.max(np.abs(np.mean([x.raw_probability.to_numpy() for x in a],axis=0)-ens.raw_probability)))
        check(f'ensemble_{y}_{s}',diff<1e-15,max_abs_error=diff)

def audit_evaluation():
    pred=read(HERE/'predictions.csv.gz');summary=pd.read_csv(HERE/'summary.csv',float_precision='round_trip')
    intervals=pd.read_csv(HERE/'paired_intervals.csv',float_precision='round_trip')
    check('evaluation_scope',set(pred.fold_test_year)=={2023,2024,2025} and pred.candidate_signal.dtype==bool)
    ref=read(ROOT/'research_v3/models/basis_train_120m_h5_predictions.csv.gz')
    ref=ref[ref.fold_test_year.le(2025)].sort_values(['date','corridor'])
    for (cid,pol),g in pred.groupby(['config_id','policy']):
        base=ref if g.corridor.nunique()==5 else ref[ref.corridor.eq('KZT')]
        z=g.sort_values(['date','corridor'])
        check(f'common_test_cohort_{cid}_{pol}',keys(base).equals(keys(z)) and np.array_equal(base.target,z.target) and np.allclose(base.forward_bps,z.forward_bps,atol=1e-10,rtol=0))
    errors=[]
    for row in summary.to_dict('records'):
        g=pred[pred.config_id.eq(row['config_id'])&pred.policy.eq(row['policy'])]
        if row['evaluation_scope']=='KZT':g=g[g.corridor.eq('KZT')]
        m=independent_metrics(g)
        for k,v in m.items():
            if not np.isclose(v,row[k],atol=1e-10,rtol=1e-12,equal_nan=True):errors.append([row['config_id'],row['policy'],k,v,row[k]])
    check('all_summary_aggregate_values',not errors,summary_rows=len(summary),errors=errors)
    selected=pred[pred.config_id.eq('kzt_tabm_full33_s2')&pred.policy.eq('cadence85_cd2')]
    check('annual_week_denominator',independent_metrics(selected)['week_cells']==152,correct=152,incorrect_continuous=155)
    diagnostics=[];interval_errors=[]
    for row in intervals.to_dict('records'):
        pa,pb=(row['policy'].split(' vs ') if ' vs ' in row['policy'] else [row['policy']]*2)
        a=pred[pred.config_id.eq(row['baseline'])&pred.policy.eq(pa)]
        b=pred[pred.config_id.eq(row['candidate'])&pred.policy.eq(pb)]
        if row['scope']=='KZT':a=a[a.corridor.eq('KZT')];b=b[b.corridor.eq('KZT')]
        result=bootstrap(a,b)
        for k,x in result.items():
            prefix='' if k=='brier' else k+'_'
            for side in ('low','high'):
                if not np.isclose(x[side],row[prefix+'ci_'+side],atol=1e-10,rtol=1e-10,equal_nan=True):interval_errors.append([row['baseline'],row['policy'],k,side,x[side],row[prefix+'ci_'+side]])
        diagnostics.append(dict(baseline=row['baseline'],candidate=row['candidate'],policy=row['policy'],scope=row['scope'],metrics=result))
    check('all_paired_month_intervals',not interval_errors,comparisons=len(intervals),errors=interval_errors)
    (OUT/'bootstrap_finite_draws.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=True)+'\n')
    headline=summary[summary.config_id.isin(['kzt_tabm_full33_s2','kzt_hgb_full33'])&summary.policy.eq('cadence85_cd2')]
    headline.to_csv(OUT/'independent_headline.csv',index=False)
    return pred

def replay_calibration_policy(pred):
    """Use frozen numeric calibrator parameters and independent sequential selector."""
    raw=pd.concat([read(HERE/p/'raw_predictions.csv.gz') for p in ('local','pooled')],ignore_index=True)
    mapping={f'arch_pooled_{a}_{f}':f'pooled_{a}_{f}'+('_s2' if a=='tabm' else '') for a in ('tabm','hgb') for f in ('base15','full33')}
    raw.config_id=raw.config_id.replace(mapping)
    cals=json.loads((HERE/'calibration.json').read_text());policies=json.loads((HERE/'policies.json').read_text())
    lookup={(x['config_id'],x['cutoff'],x['corridor']):x for x in cals}
    failures=[];maxprob=0.;count=0
    for policy in policies:
        cid,cutoff,name=policy['config_id'],policy['cutoff'],policy['name']
        g=raw[raw.config_id.eq(cid)&raw.cutoff.eq(cutoff)].copy();probs=[]
        for row in g.itertuples():
            cal=lookup[cid,cutoff,row.corridor];p=row.raw_probability
            if cal['method']!='identity':
                p=np.clip(p,1e-6,1-1e-6);p=expit(cal['intercept']+cal['slope']*np.log(p/(1-p)))
            probs.append(p)
        g['probability']=probs
        state={};chosen=[]
        for split in ('history','test'):
            for row in g[g.split.eq(split)].sort_values(['date','corridor']).itertuples():
                w=str(row.date.to_period('W-SUN'));st=state.setdefault(row.corridor,dict(last=-10000,week=w,count=0))
                if w!=st['week']:st['week']=w;st['count']=0
                if row.probability>=policy['threshold'][row.corridor] and row.session_ordinal-st['last']>policy['cooldown'] and st['count']<2:
                    st.update(last=int(row.session_ordinal),count=st['count']+1)
                    if split=='test':chosen.append((row.date,row.corridor))
            if split=='history' and state!=policy['initial_state']:failures.append([cid,cutoff,name,'history_state'])
        saved=pred[pred.config_id.eq(cid)&pred.cutoff.eq(cutoff)&pred.policy.eq(name)].sort_values(['date','corridor'])
        test=g[g.split.eq('test')].sort_values(['date','corridor'])
        maxprob=max(maxprob,float(np.max(np.abs(saved.probability.to_numpy()-test.probability.to_numpy()))))
        decisions=[(r.date,r.corridor) in chosen for r in saved.itertuples()]
        if not np.array_equal(decisions,saved.candidate_signal):failures.append([cid,cutoff,name,'contacts'])
        count+=len(saved)
    check('frozen_calibrator_and_independent_policy_replay',not failures and maxprob<1e-12,rows=count,max_probability_error=maxprob,failures=failures)

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        p=audit_evaluation();replay_calibration_policy(p);audit_local()
    files=[HERE/f for f in ('protocol.json','local.py','evaluate.py','summary.csv','by_year.csv','paired_intervals.csv','predictions.csv.gz','local/receipts.json','local/raw_predictions.csv.gz','pooled/protocol.json','pooled/experiment.py','pooled/raw_predictions.csv.gz')]
    receipt=dict(status='PASS',checks=checks,passed=len(checks),scope='Independent read-only audit; no fitting or source mutation.',input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},audit_sha256=sha(__file__))
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks))))
if __name__=='__main__':main()
