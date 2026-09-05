"""Saved-model verification plus independent root policy/split audit; no tree fits."""
from pathlib import Path
import sys,pickle,json,warnings
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import experiment as x
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research_v4.final_sprint import common,hgb

def direct_select(frame,policy,initial=None):
    state={} if initial is None else {k:dict(v) for k,v in initial.items()};yes=[]
    for idx,row in frame.sort_values(['date','corridor']).iterrows():
        currency=str(row.corridor);week=str(row.date.to_period('W-SUN'))
        previous=state.get(currency,dict(last=-10000,week=week,count=0))
        if previous['week']!=week:previous['week']=week;previous['count']=0
        limit=policy['threshold'][currency]
        if np.isfinite(row.probability) and row.probability>=limit and row.session_ordinal>previous['last']+policy['cooldown'] and previous['count']<2:
            yes.append(idx);previous['last']=int(row.session_ordinal);previous['count']+=1
        state[currency]=previous
    return yes,state

def main():
    checks=[]
    def check(name,passed,**data):
        checks.append(dict(check=name,passed=bool(passed),**data));print(name,'PASS' if passed else 'FAIL',flush=True)
    views,bankcols=x.e.build_views();views={k:x.augment_panel(v) for k,v in views.items()}
    p=views['2010-01-01',24,1]
    artifacts=list((HERE/'output').glob('*.pkl'))+list((HERE/'matched/output').glob('*.pkl'))
    for path in artifacts:
        saved=pickle.loads(path.read_bytes());r=json.loads(path.with_suffix('.json').read_text())
        cutoff=pd.Timestamp(saved['cutoff']);s=saved['spec'];features=saved['features']
        tr,va,te=x.e.old.temporal_split(p,5,cutoff,pd.Timestamp(cutoff.year+1,1,1),x.e.old.Spec(s['name'],months=s['months'],extended=True))
        va=va[va.corridor.eq('KZT')];te=te[te.corridor.eq('KZT')]
        chosen,rank,_=x.choose_features(tr,x.e.oxr.BASE+bankcols+x.TREASURY_LAG7_FEATURES)
        check(path.stem+'_train_selection_hash_purge',
              x.e.fp(tr[['date','corridor',*features]])==r['train_fingerprint']
              and x.sha(path)==r['checkpoint_sha256'] and x.sha(path.with_suffix('.csv.gz'))==r['predictions_sha256']
              and x.sha(path.with_suffix('.cbm'))==r['native_model_sha256']
              and tr.label_available_date.max()<cutoff-pd.DateOffset(years=1)
              and va.label_available_date.max()<cutoff and (s['subset']=='full' or chosen==features))
        model=saved['model'];vraw=model.predict_proba(x.matrix(va,features))[:,1]
        calibrator=x.e.core.fit_platt_calibrator(vraw,va.target)
        vp=x.e.core.apply_platt(calibrator,vraw)
        threshold,_,_=x.e.core.choose_frequency_threshold(va,vp)
        original_vp=x.e.core.apply_platt(saved['calibrator'],vraw)
        check(path.stem+'_calibration_and_threshold_replay',np.max(abs(vp-original_vp))<1e-12 and threshold==saved['threshold'])
        pred=pd.read_csv(path.with_suffix('.csv.gz'),parse_dates=['date'])
        maximum=0.;mask_exact=True
        for mode,lag in (('normal',1),('bank_delayed',2)):
            z=x.e.stress_view(views,dict(since='2010-01-01'),cutoff,24,lag).loc[te.index]
            raw=model.predict_proba(x.matrix(z,features))[:,1];prob=x.e.core.apply_platt(saved['calibrator'],raw)
            ids=x.e.core.select_per_corridor_with_cooldown(z,prob,saved['threshold'],saved['initial_state'])
            portfolio=x.e.core.select_portfolio_from_candidates(z,prob,ids,saved['portfolio_state'])
            q=pred[pred['mode'].eq(mode)]
            maximum=max(maximum,float(np.max(abs(raw-q.raw_probability.to_numpy()))),float(np.max(abs(prob-q.probability.to_numpy()))))
            mask_exact &= np.array_equal(q.candidate_signal,z.index.isin(ids)) and np.array_equal(q.signal,z.index.isin(portfolio))
        check(path.stem+'_all_view_prediction_replay',maximum<1e-12 and mask_exact,maximum_probability_error=maximum)
    check('CatBoost18models_verified',len(artifacts)==18)
    hist=pd.read_csv(HERE/'candidates_history_predictions.csv.gz',parse_dates=['date','label_available_date'])
    cal=pd.read_csv(HERE/'candidates_calibration_predictions.csv.gz',parse_dates=['date','label_available_date'])
    for frame in (hist,cal):frame['label_available_date']=pd.to_datetime(frame.label_available_date,format='mixed')
    late=hist.label_available_date.ge(pd.to_datetime(hist.cutoff))|hist.label_available_date.isna()
    check('history_keeps_unmatured_dates_hides_labels',hist.loc[late,['target','forward_bps']].isna().all().all()
          and (hist.date<pd.to_datetime(hist.cutoff)).all() and late.any(),unmatured_history_rows=int(late.sum()))
    check('calibration_all_labels_mature',cal.target.notna().all() and (cal.label_available_date<pd.to_datetime(cal.cutoff)).all())
    # Root receipts for all annual and monthly checkpoints: inspect but do not refit.
    root_errors=[];count=0
    for file in hgb.OUT.glob('*.json'):
        if not file.with_suffix('.pkl').exists():continue
        r=json.loads(file.read_text());s=r['spec'];cutoff=pd.Timestamp(file.stem[-10:]);count+=1
        if not (pd.Timestamp(r['latest_train_label'])<cutoff-pd.DateOffset(months=s['cal'])
                and pd.Timestamp(r['latest_validation_label'])<cutoff
                and x.sha(file.with_suffix('.pkl'))==r['checkpoint_sha256']):root_errors.append(file.name)
    check('root_all_existing_annual_monthly_checkpoint_maturities',not root_errors,checkpoints=count,errors=root_errors)
    for year in (2025,2026):
        name=f'hgb_60m_c12_d0_bank_treasury_aug_{year}-01-01'
        file=hgb.OUT/(name+'.pkl');bundle=pickle.loads(file.read_bytes());s=bundle['spec'];cutoff=pd.Timestamp(year,1,1)
        tr,va,history,te=hgb.split(p,s,cutoff);features=bundle['features']
        augment=views['2010-01-01',24,2].loc[tr.index]
        check(f'root_{year}_augmentation_same_labels_and_mature_times',
              tr[['date','corridor','target','label_available_date']].equals(augment[['date','corridor','target','label_available_date']])
              and (augment.label_available_date<cutoff-pd.DateOffset(months=s['cal'])).all())
        for z in (va,history):
            raw=bundle['model'].predict_proba(z[features+['corridor']])[:,1]
            z['probability']=x.e.core.apply_platt(bundle['calibrator'],raw)
        for policy_name,policy in bundle['policies'].items():
            recomputed=common.fit_policy(va,history,policy_name)
            check(f'root_{year}_{policy_name}_past_only_policy_fit',recomputed==policy)
        saved=pd.read_csv(file.with_suffix('.csv.gz'),parse_dates=['date'])
        errors=[];maximum=0.
        for mode,lag in (('normal',1),('bank_delayed',2)):
            z=x.e.stress_view(views,dict(since='2010-01-01'),cutoff,24,lag).loc[te.index].copy()
            raw=bundle['model'].predict_proba(z[features+['corridor']])[:,1]
            z['probability']=x.e.core.apply_platt(bundle['calibrator'],raw)
            for policy_name,policy in bundle['policies'].items():
                ids,state=direct_select(z,policy,policy['initial_state'])
                q=saved[saved['mode'].eq(mode)&saved.policy.eq(policy_name)]
                maximum=max(maximum,float(np.max(abs(z.probability-q.probability.to_numpy()))))
                if not np.array_equal(z.index.isin(ids),q.candidate_signal):errors.append(mode+policy_name)
                # Target/outcome mutation and append-only future decisions cannot change past contacts.
                poisoned=z.copy();poisoned[['target','forward_bps']]=999999.
                altered,_=direct_select(poisoned,policy,policy['initial_state'])
                if ids!=altered:errors.append('outcome_poison')
                tail=z.iloc[-1:].copy();tail.index=[int(z.index.max())+1]
                tail['date']=z.date.max()+pd.Timedelta(days=100);tail['probability']=1.;tail['session_ordinal']=z.session_ordinal.max()+100
                extended,_=direct_select(pd.concat([z,tail]),policy,policy['initial_state'])
                if [i for i in extended if i in z.index]!=ids:errors.append('future_append')
                if year==2026 and policy_name=='cadence90_cd3':
                    a=z.copy();a['candidate_signal']=a.index.isin(ids)
                    point=common.metrics(a)
                    check(f'root_champion2026_{mode}_objective_and_units',
                          point['weeks']==33 and point['rows']==156 and point['lift']>=1.3
                          and point['weeks_1_2']>=.85 and point['forward_delta_bps']>0,
                          **point,note='Forward delta is relative to all eligible-date mean, not net executable P&L.')
        check(f'root_{year}_champion_prediction_masks_and_causality',not errors and maximum<1e-12,
              errors=errors,maximum_probability_error=maximum)
    # Reconstruct every saved monthly contact stream with actual carried state.
    for file in (HERE.parent/'monthly').glob('monthly_*.csv.gz'):
        saved=pd.read_csv(file,parse_dates=['date']);errors=[]
        for (mode,policy_name),g in saved.groupby(['mode','policy']):
            state=None;have=[]
            for fit_cutoff,z in g.groupby('fit_cutoff',sort=True):
                stem=g.config_id.iloc[0].removeprefix('monthly_')+'_'+fit_cutoff
                bundle=pickle.loads((hgb.OUT/(stem+'.pkl')).read_bytes());policy=bundle['policies'][policy_name]
                if state is None:state=policy['initial_state']
                ids,state=direct_select(z,policy,state);have.extend(ids)
            if not np.array_equal(g.index.isin(have),g.candidate_signal):errors.append(mode+policy_name)
        check(file.stem+'_monthly_actual_state_replay',not errors,errors=errors)
    result=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',passed=sum(c['passed'] for c in checks),
                failed=sum(not c['passed'] for c in checks),checks=checks,tree_fits=0,api_calls=0,
                verification_code_sha256=x.sha(__file__),root_code_sha256={f:x.sha(HERE.parent/f) for f in ('common.py','hgb.py','monthly.py','protocol.json')},
                limitations='Retrospective model/policy selection2026 explicitly authorized; no untouched confirmation or execution-profit claim. Duplicate-normal attribution is a separate requested control.')
    x.save(HERE/'verification.json',result)
    print(json.dumps({k:result[k] for k in ('status','passed','failed')}),flush=True)
    if result['failed']:raise SystemExit(1)

if __name__=='__main__':
    with threadpool_limits(limits=1),warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning);main()
