"""OXR ablation with explicit frozen cutoffs and source publication delays."""
from __future__ import annotations
import argparse, hashlib, json, pickle, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from final_solution.training import core_experiment as core
from research_v3.models import experiment as old

BASE = old.BASE_FEATURES + [old.BASIS]
SOURCE = ROOT / 'data/open_exchange_rates/rub_cis_daily.csv'
SNAPSHOT = HERE / 'input_oxr_snapshot.csv'
RET = ['oxr_ret1', 'oxr_ret5', 'oxr_ret20', 'oxr_vol20']
BASIS = ['oxr_log_basis', 'oxr_basis_chg1', 'oxr_basis_chg5', 'oxr_basis_z20']
COVER = ['oxr_available', 'oxr_age_days']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n')

def specs():
    out = [dict(name=f'v3_{m}m', months=m, family='none', delay=24, since='2018-06-17') for m in (24,120)]
    for months in (24,120):
        for family in ('returns','basis','full'):
            for delay in (24,48):
                out.append(dict(name=f'oxr_{family}_{months}m_delay{delay}h',months=months,family=family,delay=delay,since='2018-06-17'))
    out.append(dict(name='oxr_coverage_120m',months=120,family='coverage',delay=24,since='2018-06-17'))
    for since in ('2020-01-01','2022-01-01'):
        out.append(dict(name=f'oxr_full_120m_since{since[:4]}',months=120,family='full',delay=24,since=since))
    return out

def build_panel(delay=24,since='2018-06-17',raw=None,extended=True):
    source_panel=ROOT/'research_v3/models'/('panel_extended.pkl' if extended else 'panel_v2.pkl')
    p = pd.read_pickle(source_panel).copy()
    raw = pd.read_csv(SNAPSHOT) if raw is None else raw.copy()
    raw = raw[raw.date.ge(since)].copy()
    raw['observed_date'] = pd.to_datetime(raw.date)
    raw['published'] = pd.to_datetime(raw.published_at_utc,utc=True)
    completed_day=raw.observed_date.dt.tz_localize('UTC')+pd.Timedelta(days=1)
    raw['available_at'] = pd.concat([raw.published,completed_day],axis=1).max(axis=1)+pd.Timedelta(hours=delay)
    pieces=[]
    for corridor, left in p.groupby('corridor'):
        r=raw[raw.quote.eq(corridor)].sort_values('observed_date').copy()
        assert not r.observed_date.duplicated().any()
        v=np.log(r.rub_per_quote)
        for n in (1,5,20):r[f'oxr_ret{n}']=v.diff(n)
        r['oxr_vol20']=v.diff().rolling(20,min_periods=10).std()
        r['oxr_log_rate']=v
        left=left.sort_values('date').copy()
        # Frozen reference pipeline assumes today's CBR rate known at this batch.
        left['decision_at']=left.date.dt.tz_localize('Europe/Moscow')+pd.Timedelta(hours=10,minutes=5)
        left['decision_at']=left.decision_at.dt.tz_convert('UTC')
        q=pd.merge_asof(left,r[['available_at','published','observed_date','oxr_log_rate',*RET]],left_on='decision_at',right_on='available_at',direction='backward',tolerance=pd.Timedelta(days=7))
        valid=q.available_at.notna()
        assert (q.loc[valid,'available_at'] <= q.loc[valid,'decision_at']).all()
        q['oxr_available']=valid.astype(float)
        q['oxr_age_days']=(q.decision_at-q.published).dt.total_seconds()/86400
        q['oxr_log_basis']=q.oxr_log_rate-np.log(q.rub_per_unit)
        b=q.oxr_log_basis
        q['oxr_basis_chg1']=b.diff()
        q['oxr_basis_chg5']=b.diff(5)
        q['oxr_basis_z20']=(b-b.rolling(20,min_periods=10).mean())/b.rolling(20,min_periods=10).std().clip(lower=1e-6)
        pieces.append(q)
    return pd.concat(pieces,ignore_index=True).sort_values(['date','corridor']).reset_index(drop=True)

def features(spec):
    family=spec['family']
    extras={'none':[],'returns':RET+COVER,'basis':BASIS+COVER,'full':RET+BASIS+COVER,'coverage':COVER}[family]
    return BASE+extras

def evaluate(panel,spec,cutoff,end):
    p=core.add_target(panel,5)
    cutoff=pd.Timestamp(cutoff);end=pd.Timestamp(end)
    val_start=cutoff-pd.DateOffset(years=1)
    contract=old.Spec(spec['name'],months=spec['months'],extended=True)
    tr,va,te=old.temporal_split(p,5,cutoff,end,contract)
    # Prove label maturity using actual fifth future observation dates.
    future=p.groupby('corridor').date.shift(-5)
    assert (future.loc[tr.index] < val_start).all()
    assert (future.loc[va.index] < cutoff).all()
    f=features(spec)
    model=core.make_model('hist_gradient_boosting',f)
    model.named_steps['classifier'].set_params(early_stopping=False)
    started=time.monotonic()
    model.fit(tr[f+['corridor']],tr.target.astype(int))
    vraw=model.predict_proba(va[f+['corridor']])[:,1]
    cal=core.fit_platt_calibrator(vraw,va.target)
    vp=core.apply_platt(cal,vraw)
    threshold,_,_=core.choose_frequency_threshold(va,vp)
    history=p[p.date.ge(val_start)&p.date.lt(cutoff)].copy()
    hp=core.apply_platt(cal,model.predict_proba(history[f+['corridor']])[:,1])
    hi=core.select_per_corridor_with_cooldown(history,hp,threshold)
    state=core.corridor_selection_state(history,hi)
    portfolio_state=core.selection_state(history,core.select_portfolio_from_candidates(history,hp,hi))
    raw=model.predict_proba(te[f+['corridor']])[:,1]
    prob=core.apply_platt(cal,raw)
    selected=core.select_per_corridor_with_cooldown(te,prob,threshold,state)
    portfolio=core.select_portfolio_from_candidates(te,prob,selected,portfolio_state)
    pred=te[['date','corridor','target','forward_bps','symmetric_bps','regret_bps','session_ordinal']].copy()
    pred['raw_probability']=raw;pred['probability']=prob
    pred['candidate_signal']=pred.index.isin(selected);pred['signal']=pred.index.isin(portfolio)
    pred['config_id']=spec['name'];pred['fold_test_year']=cutoff.year;pred['cutoff']=str(cutoff.date())
    out=HERE/'output';out.mkdir(exist_ok=True)
    name=spec['name']+'_'+str(cutoff.date())
    checkpoint=out/(name+'.pkl')
    checkpoint.write_bytes(pickle.dumps(dict(model=model,calibrator=cal,threshold=threshold,spec=spec,features=f)))
    pred.to_csv(out/(name+'.csv.gz'),index=False)
    receipt=dict(spec=spec,cutoff=cutoff,train_min=tr.date.min(),train_max=tr.date.max(),train_rows=len(tr),train_latest_label=future.loc[tr.index].max(),validation_min=va.date.min(),validation_max=va.date.max(),validation_latest_label=future.loc[va.index].max(),history_max=history.date.max(),test_min=te.date.min(),test_max=te.date.max(),test_rows=len(te),features=f,train_oxr_coverage=float(tr.oxr_available.mean()),validation_oxr_coverage=float(va.oxr_available.mean()),test_oxr_coverage=float(te.oxr_available.mean()),elapsed_seconds=time.monotonic()-started,checkpoint_sha256=sha(checkpoint),predictions_sha256=sha(out/(name+'.csv.gz')),source_sha256=sha(SNAPSHOT),feature_frame_sha256=hashlib.sha256(pd.util.hash_pandas_object(p[['date','corridor',*f]],index=False).to_numpy().tobytes()).hexdigest())
    save(out/(name+'.json'),receipt)
    print(name,len(pred),round(receipt['elapsed_seconds'],2),flush=True)
    return pred

def metrics(pred):
    rows=[]
    for (config,cutoff),g in pred.groupby(['config_id','cutoff']):
        for scope in ('all','KZT'):
            x=g if scope=='all' else g[g.corridor.eq('KZT')]
            base=x.groupby(['fold_test_year','corridor']).agg(base_hit=('target','mean'),base_forward=('forward_bps','mean'))
            s=x[x.candidate_signal].join(base,on=['fold_test_year','corridor'])
            rows.append(dict(config_id=config,cutoff=cutoff,scope=scope,rows=len(x),dates=x.date.nunique(),brier=float(np.mean((x.probability-x.target)**2)),signals=len(s),hit_rate=float(s.target.mean()),lift=float(s.target.mean()/s.base_hit.mean()),forward_delta_bps=float((s.forward_bps-s.base_forward).mean())))
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['development','test'],required=True);args=parser.parse_args()
    if not SNAPSHOT.exists():shutil.copy2(SOURCE,SNAPSHOT)
    config=specs();save(HERE/'specifications.json',config)
    if args.phase=='test':
        assert (HERE/'selection.json').exists(),'Development selection must be frozen first'
    frames={}
    outputs=[]
    cutoffs=['2023-01-01','2024-01-01','2025-01-01'] if args.phase=='development' else ['2026-01-01','2026-03-01']
    with threadpool_limits(limits=1):
        for spec in config:
            key=(spec['delay'],spec['since'],spec['months']==120)
            if key not in frames:frames[key]=build_panel(key[0],key[1],extended=key[2])
            for cutoff in cutoffs:
                end=f'{pd.Timestamp(cutoff).year+1}-01-01'
                outputs.append(evaluate(frames[key],spec,cutoff,end))
    pred=pd.concat(outputs,ignore_index=True)
    pred.to_csv(HERE/(args.phase+'_predictions.csv.gz'),index=False)
    result=metrics(pred);result.to_csv(HERE/(args.phase+'_metrics.csv'),index=False)
    if args.phase=='development':
        eligible=[s['name'] for s in config if s['months']==120 and s['family'] in ('returns','basis','full') and s['delay']==24 and s['since']=='2018-06-17']
        m=result[result.scope.eq('all') & result.config_id.isin(eligible)].groupby('config_id').apply(lambda g:float(np.average(g.brier,weights=g.rows)))
        winner=str(m.idxmin())
        save(HERE/'selection.json',dict(primary_candidate=winner,criterion='minimum paired-development all-corridor Brier among predefined120m primarydelay OXR families; no2026labels',scores=m.to_dict(),development_predictions_sha256=sha(HERE/'development_predictions.csv.gz'),selection_unix=time.time()))

if __name__=='__main__':main()
