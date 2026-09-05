#!/usr/bin/env python3
"""Evidence-informed, explicitly non-empirical personalization stress test.

No synthetic user feature is input to the FX model. All policy returns use the
same frozen, previously inspected out-of-time predictions and official FX path.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
import argparse, dataclasses, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = 'hgb_plus_cnyrub_basis'
CORRIDORS = ('AMD','KGS','KZT','TJS','UZS')
SEGMENTS = ('frequent','monthly','occasional')
POLICIES = ('market_only','frequency_gate','user_aware','matched_random_market','weekly_fixed','matched_calendar')
# Mixtures are scenarios, not estimates of the bank's customer composition.
MIXES = {'active_bank_assumption': (.15,.50,.35), 'balanced': (1/3,1/3,1/3),
         'historical_tajik_monthly_share_stress': (0,.34,.66), 'frequent_heavy': (.50,.35,.15)}

@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str = 'base'
    flexibility_scale: float = 1.
    urgent_share: float = .20
    intent_sensitivity: float = .70
    intent_false_positive: float = .03
    balance_sensitivity: float = .90
    balance_false_positive: float = .40
    response: float = .35
    fatigue_per_contact: float = .05
    phase_noise_days: float = 2.
    contact_cost_rub: float = 1.
    incremental_execution_cost_bps: float = 0.

SCENARIOS = [Scenario(),
    Scenario(name='no_behavioral_response',response=0),
    Scenario(name='no_fatigue',fatigue_per_contact=0),
    Scenario(name='narrow_flexibility',flexibility_scale=.25),
    Scenario(name='wide_flexibility',flexibility_scale=2),
    Scenario(name='urgent_majority',urgent_share=.75),
    Scenario(name='weak_intent',intent_sensitivity=.30,intent_false_positive=.15,balance_sensitivity=.75,phase_noise_days=7),
    Scenario(name='high_contact_cost',contact_cost_rub=10),
    Scenario(name='execution_drag_25bps',incremental_execution_cost_bps=25),
    Scenario(name='execution_drag_50bps',incremental_execution_cost_bps=50),
    Scenario(name='low_response',response=.10),
    Scenario(name='high_response',response=.70)]


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load_market():
    cols=['date','corridor','config_id','fold_test_year','candidate_signal','probability','target','forward_bps']
    frames=[]
    paths=[]
    for name in ['development_h5_predictions.csv','diagnostic_2026_predictions.csv']:
        p=ROOT/'final_solution/model_bundle'/name;paths.append(p)
        for df in pd.read_csv(p,usecols=cols,chunksize=15000):
            frames.append(df[df.config_id.eq(MODEL)])
    market=pd.concat(frames,ignore_index=True)
    market['date']=pd.to_datetime(market.date)
    if market.duplicated(['date','corridor']).any(): raise AssertionError('duplicate market rows')
    p=ROOT/'final_solution/data/cbr_daily.csv';paths.append(p)
    rates=pd.read_csv(p,parse_dates=['date']).pivot(index='date',columns='corridor',values='rub_per_unit').sort_index()
    return market,rates,paths


def make_users(seed, n_per_segment):
    """Synthetic 12-event pre-period; every forecast feature computed from it only."""
    rng=np.random.default_rng(seed)
    users=[]
    for si,seg in enumerate(SEGMENTS):
        for j in range(n_per_segment):
            uid=si*n_per_segment+j
            interval=float(rng.choice([7.,14.])) if si==0 else (30.4375 if si==1 else float(rng.choice([60.,90.,180.],p=[.35,.50,.15])))
            jitter={0:1.,1:2.,2:7.}[si]
            # Evidence motivates variable transfers; ruble magnitudes are illustrative.
            median_amount=[18000.,45000.,90000.][si]*np.exp(rng.normal(0,.35))
            payroll_phase=rng.uniform(0,interval)
            pre_due=payroll_phase+np.arange(-14,0)*interval+rng.normal(0,jitter,14)
            pre_due=np.sort(pre_due[pre_due<0])[-13:]
            assert len(pre_due)>=12 and pre_due[-1]<0
            gap=float(np.median(np.diff(pre_due)))
            users.append(dict(user_id=uid,segment=seg,corridor=CORRIDORS[j%5],interval=interval,
                              phase=payroll_phase,jitter=jitter,estimated_interval=gap,
                              last_preperiod_transfer=float(pre_due[-1]),median_amount=median_amount,
                              base_flex=[2.,5.,10.][si]))
    return users


def make_world(user, year, n_days, seed, scenario):
    """Generate latent needs and observed noisy signals before evaluating policies.

    Counterfactual needs are fixed across policies. No treatment creates amount,
    count or demand. Liquidity only permits advancing a planned transfer.
    """
    rng=np.random.default_rng(np.random.SeedSequence([seed,year,user['user_id']]))
    cycle=np.arange(-2,int(n_days/user['interval'])+3)
    planned=user['phase']+cycle*user['interval']
    due=np.rint(planned+rng.normal(0,user['jitter'],len(cycle))).astype(int)
    urgent=rng.random(len(cycle))<scenario.urgent_share
    flex=np.rint(user['base_flex']*scenario.flexibility_scale*rng.uniform(.5,1.5,len(cycle))).astype(int)
    flex[urgent]=0
    # Each need has one transfer. Clip to ensure windows do not overlap.
    flex=np.minimum(flex,max(0,int(user['interval'])-2))
    amount=user['median_amount']*np.exp(rng.normal(-.5*.40**2,.40,len(cycle)))
    keep=(due>=0)&(due<n_days)
    due,flex,amount,urgent=due[keep],flex[keep],amount[keep],urgent[keep]
    order=np.argsort(due);due,flex,amount,urgent=due[order],flex[order],amount[order],urgent[order]
    if len(np.unique(due))!=len(due): raise AssertionError('overlapping due dates')
    ready=np.maximum(0,due-flex)
    state=np.zeros(n_days,dtype=bool)
    event_at=np.full(n_days,-1,dtype=int)
    for e,(lo,hi) in enumerate(zip(ready,due)):
        state[lo:hi]=True;event_at[lo:hi]=e
    # Observed bank/app features are noisy measurements, not future due-date labels.
    intent_u=rng.random(n_days); balance_u=rng.random(n_days)
    observed_intent=intent_u<np.where(state,scenario.intent_sensitivity,scenario.intent_false_positive)
    observed_balance=balance_u<np.where(state,scenario.balance_sensitivity,scenario.balance_false_positive)
    response_u=rng.random(n_days)
    # Frozen start-of-period recurrent-date forecast; phase noise is ex ante.
    last=user['last_preperiod_transfer']+rng.normal(0,scenario.phase_noise_days)
    elapsed=np.arange(n_days)-last
    until_next=user['estimated_interval']-np.mod(elapsed,user['estimated_interval'])
    phase_window=until_next<=min(10.,max(2.,user['base_flex']*1.5))
    weekly_hash=rng.random(n_days)
    return dict(due=due,ready=ready,amount=amount,urgent=urgent,state=state,event_at=event_at,
                intent=observed_intent,balance=observed_balance,response_u=response_u,
                phase_window=phase_window,random_rank=weekly_hash)


def cap_dates(dates, limit=2):
    accepted=[]
    for d in dates:
        if sum(d-6<=x for x in accepted[-2:])<limit:accepted.append(int(d))
    return np.array(accepted,dtype=int)


def matched_calendar_dates(valid_days, quota, rank):
    """Exact ex-post count matching; dates use no FX scores, needs or outcomes.

    Evenly spaced calendar targets, greedily fill missing slots if cap binds.
    This is a descriptive comparator, not a deployable online quota policy.
    """
    if quota==0:return np.array([],dtype=int)
    ix=np.rint(np.linspace(0,len(valid_days)-1,quota)).astype(int)
    out=cap_dates(valid_days[ix])
    for d in valid_days[np.argsort(rank[valid_days])]:
        if len(out)==quota:break
        if d in out:continue
        trial=np.sort(np.append(out,d))
        if max(sum((trial>=t-6)&(trial<=t)) for t in trial)<=2:out=trial
    if len(out)!=quota:raise AssertionError('calendar budget cannot be matched')
    return out


def run_policy(policy, candidate_days, valid_days, dates, rates, target, forward, world, scenario, quota=None):
    if policy=='weekly_fixed':
        # First actual market-observation day each ISO week, independent of score.
        weeks=dates[valid_days].isocalendar()
        _,ix=np.unique(np.array([f'{y}-{w}' for y,w in zip(weeks.year,weeks.week)]),return_index=True)
        contacts=cap_dates(valid_days[np.sort(ix)])
    elif policy=='matched_random_market':
        take=np.argsort(world['random_rank'][candidate_days])[:quota]
        contacts=np.sort(candidate_days[take])
    elif policy=='matched_calendar':
        contacts=matched_calendar_dates(valid_days,quota,world['random_rank'])
    else:contacts=candidate_days
    due=world['due'];amount=world['amount']; moved=np.zeros(len(due),bool)
    accepted=[]; shifted=[]; changed_date=due.copy(); relevant=[]
    last_transfer=-10000
    due_map={int(d):e for e,d in enumerate(due)}
    contact_set=set(contacts)
    for day in range(len(dates)):
        # Organic transfers occur before contact on their due date: never delay.
        if day in due_map and not moved[due_map[day]]:last_transfer=day
        if day not in contact_set:continue
        if policy=='frequency_gate' and not world['phase_window'][day]:continue
        if policy=='user_aware':
            if not ((world['phase_window'][day] or world['intent'][day]) and world['balance'][day]):continue
            if day-last_transfer<3:continue
        if sum(day-6<=x for x in accepted[-2:])>=2:continue
        e=int(world['event_at'][day]); is_relevant=e>=0 and not moved[e]
        accepted.append(day);relevant.append(int(is_relevant))
        if not is_relevant:continue
        prior_contacts=sum(day-29<=x<day for x in accepted)
        response=scenario.response*np.exp(-scenario.fatigue_per_contact*prior_contacts)
        if world['response_u'][day]<response:
            moved[e]=True;changed_date[e]=day;shifted.append(e);last_transfer=day
    contacts=np.array(accepted,dtype=int)
    # LCY bought now / LCY under fixed RUB amount at organic date minus one.
    bps=(rates[due]/rates[changed_date]-1)*10000
    rub=amount*bps/10000
    incremental_cost=amount*scenario.incremental_execution_cost_bps/10000*moved
    gross=float(rub.sum()); net=float((rub-incremental_cost).sum()-scenario.contact_cost_rub*len(contacts))
    count=len(contacts); n_events=len(due)
    return dict(contacts=count,relevant_contacts=int(sum(relevant)),shifted_transfers=int(moved.sum()),
                planned_transfers=n_events,completed_transfers=n_events,
                organic_volume_rub=float(amount.sum()),policy_volume_rub=float(amount.sum()),
                gross_timing_value_rub=gross,net_scenario_value_rub=net,
                execution_drag_rub=float(incremental_cost.sum()),
                shifted_amount_rub=float(amount[moved].sum()),sum_advance_days=int((due-changed_date).sum()),
                hurt_transfers=int((bps < -1e-10).sum()),
                contact_target_sum=float(target[contacts].sum()),
                contact_forward_bps_sum=float(forward[contacts].sum()),
                cap_max=max([sum((contacts>=d-6)&(contacts<=d)) for d in contacts],default=0),
                no_delay=bool((changed_date<=due).all()),
                no_unfunded=bool((changed_date>=world['ready']).all()),
                no_urgent_shift=bool((changed_date[world['urgent']]==due[world['urgent']]).all()),
                _contact_days=contacts)


def summarize(frame, keys):
    sums=['contacts','relevant_contacts','shifted_transfers','planned_transfers','completed_transfers','organic_volume_rub',
          'policy_volume_rub','gross_timing_value_rub','net_scenario_value_rub','execution_drag_rub','shifted_amount_rub',
          'sum_advance_days','hurt_transfers','contact_target_sum','contact_forward_bps_sum']
    g=frame.groupby(keys,dropna=False)
    out=g[sums].sum()
    out['client_periods']=g.size()
    out['exposure_weeks']=g['exposure_weeks'].sum()
    out=out.reset_index()
    d=out.contacts.replace(0,np.nan); n=out.client_periods
    out['contacts_per_client']=out.contacts/n
    out['contacts_per_client_week']=out.contacts/out.exposure_weeks
    out['relevance_rate']=out.relevant_contacts/d
    out['shift_rate_per_contact']=out.shifted_transfers/d
    out['shifted_fraction_of_transfers']=out.shifted_transfers/out.planned_transfers
    out['gross_value_rub_per_client']=out.gross_timing_value_rub/n
    out['net_scenario_value_rub_per_client']=out.net_scenario_value_rub/n
    out['timing_bps_all_planned_volume']=10000*out.gross_timing_value_rub/out.organic_volume_rub
    out['timing_bps_shifted_volume']=10000*out.gross_timing_value_rub/out.shifted_amount_rub.replace(0,np.nan)
    out['conditional_contact_hit_rate']=out.contact_target_sum/d
    out['conditional_contact_forward_bps']=out.contact_forward_bps_sum/d
    out['incremental_volume_rub']=out.policy_volume_rub-out.organic_volume_rub
    return out


def weighted_mixes(frame):
    # Segment-stratified Monte Carlo sample; reweight before computing ratios.
    nums=frame.select_dtypes(include='number').columns.tolist()
    nums=[x for x in nums if x not in ('seed','year','user_id','cap_max')]
    output=[]
    for mix,weights in MIXES.items():
        for keys,g in frame.groupby(['scenario','seed','year','policy']):
            means=g.groupby('segment')[nums].mean().reindex(SEGMENTS)
            row=dict(zip(['scenario','seed','year','policy'],keys));row['mix']=mix
            for c in nums:row[c]=float(np.dot(weights,means[c]))
            row['relevance_rate']=row['relevant_contacts']/row['contacts'] if row['contacts'] else None
            row['timing_bps_all_planned_volume']=10000*row['gross_timing_value_rub']/row['organic_volume_rub']
            row['incremental_volume_rub']=row['policy_volume_rub']-row['organic_volume_rub']
            output.append(row)
    return pd.DataFrame(output)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--users-per-segment',type=int,default=150)
    ap.add_argument('--seeds',type=int,nargs='+',default=[4101,4102,4103]);ap.add_argument('--output',type=Path,default=HERE/'results')
    ap.add_argument('--scenarios',nargs='*');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    market,rate_frame,source_paths=load_market()
    market_fingerprint_before=hashlib.sha256(pd.util.hash_pandas_object(market,index=True).values.tobytes()).hexdigest()
    users_by_seed={seed:make_users(seed,args.users_per_segment) for seed in args.seeds}
    rows=[];checks=[];stylized=[];market_quality=[];sample_contacts=[]
    scenarios=[s for s in SCENARIOS if not args.scenarios or s.name in args.scenarios]
    for year,m in market.groupby('fold_test_year'):
        start=m.date.min();end=m.date.max(); dates=pd.date_range(start,end,freq='D')
        rates=rate_frame.reindex(dates).ffill()
        if rates.isna().any().any():raise AssertionError('missing rates')
        corridor_data={}
        for c,g in m.groupby('corridor'):
            g=g.set_index('date').reindex(dates)
            valid=np.flatnonzero(g.probability.notna());candidate=cap_dates(np.flatnonzero(g.candidate_signal.fillna(False).to_numpy(bool)))
            corridor_data[c]=(valid,candidate,rates[c].to_numpy(),g.target.fillna(0).to_numpy(),g.forward_bps.fillna(0).to_numpy())
            market_quality.append(dict(year=year,corridor=c,eligible_rows=len(valid),candidate_signals=len(candidate),
                raw_market_brier=float(((g.probability-g.target)**2).mean()),
                raw_candidate_hit_rate=float(g.loc[dates[candidate],'target'].mean()),
                raw_candidate_forward_bps=float(g.loc[dates[candidate],'forward_bps'].mean())))
        for scenario in scenarios:
            for seed,users in users_by_seed.items():
                for user in users:
                    valid,candidate,fx,targets,forward=corridor_data[user['corridor']]
                    world=make_world(user,int(year),len(dates),seed,scenario)
                    outcomes={}
                    for policy in POLICIES:
                        q=outcomes['user_aware']['contacts'] if policy.startswith('matched_') else None
                        result=run_policy(policy,candidate,valid,dates,fx,targets,forward,world,scenario,q)
                        outcomes[policy]=result
                        cdays=result.pop('_contact_days')
                        if scenario.name=='base' and seed==args.seeds[0] and user['user_id']<5:
                            sample_contacts.extend(dict(year=year,user_id=user['user_id'],corridor=user['corridor'],policy=policy,
                                                        contact_date=dates[d].date(),latent_ready=bool(world['state'][d]),
                                                        observed_intent=bool(world['intent'][d]),observed_balance=bool(world['balance'][d]),
                                                        phase_window=bool(world['phase_window'][d])) for d in cdays)
                        row=dict(scenario=scenario.name,seed=seed,year=year,user_id=user['user_id'],segment=user['segment'],
                                 corridor=user['corridor'],policy=policy,exposure_weeks=len(dates)/7.,**result)
                        rows.append(row)
                    if scenario.name=='base':
                        stylized.append(dict(seed=seed,year=year,user_id=user['user_id'],segment=user['segment'],
                            interval_days=user['interval'],estimated_interval=user['estimated_interval'],
                            transfers=len(world['due']),transfers_per_year=len(world['due'])/len(dates)*365.25,
                            urgent_fraction=float(world['urgent'].mean()) if len(world['due']) else np.nan,
                            median_advance_window=float(np.median(world['due']-world['ready'])),
                            amount_cv=float(np.std(world['amount'])/np.mean(world['amount'])) if len(world['amount'])>1 else np.nan,
                            readiness_day_fraction=float(world['state'].mean()),
                            intent_tpr=float(world['intent'][world['state']].mean()) if world['state'].any() else np.nan,
                            intent_fpr=float(world['intent'][~world['state']].mean())))
                print(f'done year={year} scenario={scenario.name} seed={seed}',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'client_policy_results.csv',index=False)
    summary=summarize(frame,['scenario','year','policy']);summary.to_csv(args.output/'policy_summary.csv',index=False)
    segments=summarize(frame,['scenario','year','segment','policy']);segments.to_csv(args.output/'segment_summary.csv',index=False)
    seed_summary=summarize(frame,['scenario','year','seed','policy']);seed_summary.to_csv(args.output/'monte_carlo_seed_summary.csv',index=False)
    mix=weighted_mixes(frame);mix.to_csv(args.output/'population_mix_sensitivity.csv',index=False)
    pd.DataFrame(stylized).to_csv(args.output/'synthetic_stylized_facts.csv',index=False)
    pd.DataFrame(market_quality).to_csv(args.output/'frozen_market_quality.csv',index=False)
    pd.DataFrame(sample_contacts).to_csv(args.output/'contact_ledger_sample.csv',index=False)
    # Null + invariant tests guard inferential category errors, timing and budgets.
    keys=['scenario','seed','year','user_id']
    quotas=frame.pivot(index=keys,columns='policy',values='contacts')
    check={'all_no_delay':bool(frame.no_delay.all()),'all_funded':bool(frame.no_unfunded.all()),
           'all_urgent_events_unshifted':bool(frame.no_urgent_shift.all()),'rolling_7d_cap_le_2':bool(frame.cap_max.le(2).all()),
           'no_created_transfers':bool((frame.completed_transfers==frame.planned_transfers).all()),
           'zero_incremental_volume_by_construction':bool((frame.policy_volume_rub==frame.organic_volume_rub).all()),
           'exact_random_market_contact_budget':bool(quotas.user_aware.eq(quotas.matched_random_market).all()),
           'exact_calendar_contact_budget':bool(quotas.user_aware.eq(quotas.matched_calendar).all()),
           'input_market_forecasts_never_modified':market_fingerprint_before==hashlib.sha256(pd.util.hash_pandas_object(market,index=True).values.tobytes()).hexdigest(),
           'external_validity_for_bank_clients':False,'causal_business_uplift_identified':False}
    null=frame[frame.scenario.eq('no_behavioral_response')]
    if len(null):check['null_response_has_zero_timing_value']=bool(null.gross_timing_value_rub.eq(0).all())
    true_checks=[v for k,v in check.items() if k not in ('external_validity_for_bank_clients','causal_business_uplift_identified')]
    if not all(true_checks):raise AssertionError(check)
    (args.output/'integrity_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    receipt=dict(model=MODEL,scenarios=[dataclasses.asdict(s) for s in scenarios],seeds=args.seeds,
        synthetic_users_per_segment_per_seed=args.users_per_segment,records=len(frame),
        source_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in source_paths},script_sha256=file_hash(__file__),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        status='SCENARIO_RESEARCH_ONLY',market_status='retrospective OOT 2023-2025; already inspected diagnostic 2026',
        timing='Same CBR effective-date/publication-proxy convention as frozen model, not executable bank quotes',
        causal_uplift='NOT IDENTIFIED; new count and volume are identically zero by construction')
    (args.output/'run_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(summary[summary.scenario.eq('base')][['year','policy','contacts_per_client','relevance_rate','gross_value_rub_per_client','net_scenario_value_rub_per_client','timing_bps_all_planned_volume']].to_string(index=False))

if __name__=='__main__':main()
