#!/usr/bin/env python3
"""Prior-year segment market-policy selection on frozen V3 scores.
Synthetic users do not augment FX training or the count of independent markets.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(name,'1')
import argparse,dataclasses,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research_v3.behavior import simulate as v3
HERE=Path(__file__).resolve().parent
SEGMENTS=v3.SEGMENTS
HORIZONS=(1,3,5,10,20)
MODEL='baseline_reproduction' # fixed before this experiment; no model tournament
# WB2016 KG identifies >=monthly 40/99, below-monthly59/99 only.
# 10% of the >=monthly bin assigned frequent is an explicit planning assumption.
MIXES={'wb2016_kg_planning':(.04/0.99,.36/0.99,.59/.99),
 'wb2016_kg_no_frequent':(0,.40/.99,.59/.99),
 'wb2016_kg_frequent_quarter_of_monthly':(.10/.99,.30/.99,.59/.99),
 'wb2016_kg_frequent_half_of_monthly':(.20/.99,.20/.99,.59/.99),
 'wb2016_tj_same_split':(.041/.99,.369/.99,.58/.99),
 'iom2024_returnees_scenario':(.09/.99,.57/.99,.33/.99),
 'v3_balanced':(1/3,1/3,1/3), 'frequent_heavy_stress':(.5,.35,.15)}
# Microdata shape is external retrospective sensitivity, not a historical available covariate.
_bands=pd.read_csv(HERE/'data/l2kgz_regularity_bands.csv')
_reg=float(_bands[(_bands.period=='full2021_2025')&(_bands.min_observed_months==6)&(_bands.band=='regular_gt75pct_months')].derived_household_weighted_share.iloc[0])
MIXES['l2kgz_regular10pct_frequent_scenario']=(.1*_reg,.9*_reg,1-_reg)
MAIN='wb2016_kg_planning'
GRID=[(h,q,c) for h,q,c in itertools.product(HORIZONS,(.25,.50,.75),(3,7,14,28))]
ALLOWED={'frequent':(3,7),'monthly':(7,14),'occasional':(14,28)}
SCENARIOS=[v3.Scenario(),v3.Scenario(name='zero_response',response=0),v3.Scenario(name='low_response',response=.10),v3.Scenario(name='high_response',response=.70),
 v3.Scenario(name='weak_intent',intent_sensitivity=.30,intent_false_positive=.15,balance_sensitivity=.75,phase_noise_days=7),
 v3.Scenario(name='narrow_flexibility',flexibility_scale=.25),v3.Scenario(name='contact_cost10',contact_cost_rub=10),
 v3.Scenario(name='execution_drag25bps',incremental_execution_cost_bps=25)]
SUMS=['contacts','relevant_contacts','shifted_transfers','planned_transfers','completed_transfers','organic_volume_rub','policy_volume_rub','gross_timing_value_rub','net_scenario_value_rub','execution_drag_rub','shifted_amount_rub','sum_advance_days','hurt_transfers','contact_target_sum','contact_forward_bps_sum','own_h_target_sum','all_date_h5_base_sum','exposure_weeks']

def identity(cfg):return f'h{cfg[0]}_q{cfg[1]:.2f}_cad{cfg[2]}'
def load_market():
 frames=[];paths=[]
 for h in HORIZONS:
  p=ROOT/f'research_v3/models/{MODEL}_h{h}_predictions.csv.gz';paths.append(p)
  x=pd.read_csv(p,parse_dates=['date']); assert not x.duplicated(['date','corridor']).any()
  x=x[['date','corridor','fold_test_year','probability','target','forward_bps','candidate_signal']].rename(columns={a:f'{a}_h{h}' for a in ('probability','target','forward_bps','candidate_signal')});frames.append(x)
 panel=frames[0]
 for f in frames[1:]:panel=panel.merge(f,on=['date','corridor','fold_test_year'],how='inner',validate='one_to_one')
 p=ROOT/'final_solution/data/cbr_daily.csv';paths.append(p)
 rates=pd.read_csv(p,parse_dates=['date']).pivot(index='date',columns='corridor',values='rub_per_unit').sort_index()
 markets={}
 for y,g in panel.groupby('fold_test_year'):
  # Preserve a full calendar year of organic needs. Only contact on common observed score dates.
  dates=pd.date_range(f'{y}-01-01',g.date.max(),freq='D');fx=rates.reindex(rates.index.union(dates)).sort_index().ffill().reindex(dates)
  assert fx.notna().all().all()
  markets[int(y)]={'dates':dates,'corridors':{}}
  for c,a in g.groupby('corridor'):
   a=a.set_index('date').reindex(dates);valid=np.flatnonzero(a.probability_h5.notna())
   markets[int(y)]['corridors'][c]={'valid':valid,'rates':fx[c].to_numpy(), 'target':a.target_h5.fillna(0).to_numpy(),'forward':a.forward_bps_h5.fillna(0).to_numpy(),
    'scores':{h:a[f'probability_h{h}'].to_numpy() for h in HORIZONS},'own_targets':{h:a[f'target_h{h}'].fillna(0).to_numpy() for h in HORIZONS},'v3_days':v3.cap_dates(np.flatnonzero(a.candidate_signal_h5.fillna(False).to_numpy(bool)))}
 return panel,markets,paths

def thresholds(panel,years):
 old=panel[panel.fold_test_year.isin(years)]
 return {(h,q):float(old[f'probability_h{h}'].quantile(q)) for h in HORIZONS for q in (.25,.50,.75)}

def causal_cadence(days,gap):
 out=[]
 for day in days:
  if not out or day-out[-1]>=gap:out.append(int(day))
 return np.array(out,dtype=int)

def schedule(data,cfg,threshold):
 if cfg=='random':return data['valid']
 if cfg=='weekly':return causal_cadence(data['valid'],7)
 if cfg=='v3':return data['v3_days']
 h,q,c=cfg
 return causal_cadence(data['valid'][data['scores'][h][data['valid']]>=threshold[(h,q)]],c)

def pool(markets,years,seeds,n,scenario):
 records=[]
 for seed in seeds:
  for user in v3.make_users(seed,n):
   for year in years:
    market=markets[year];world=v3.make_world(user,year,len(market['dates']),seed,scenario)
    records.append((seed,user,year,world))
 return records

def evaluate(markets,records,scenario,configs,threshold,thin=1.,detailed=False):
 rows=[]; contacts=[]; cache={}
 for seed,user,year,world in records:
  c=user['corridor'];seg=user['segment'];cfg=configs[seg];market=markets[year];d=market['corridors'][c]
  key=(year,c,str(cfg))
  if key not in cache:cache[key]=schedule(d,cfg,threshold)
  days=cache[key];days=days[world['random_rank'][days]<thin]
  out=v3.run_policy('user_aware',days,d['valid'],market['dates'],d['rates'],d['target'],d['forward'],world,scenario)
  selected=out.pop('_contact_days');h=cfg[0] if isinstance(cfg,tuple) else 5
  assert out['cap_max']<=2 and out['no_delay'] and out['no_unfunded'] and out['no_urgent_shift']
  assert out['completed_transfers']==out['planned_transfers'] and out['policy_volume_rub']==out['organic_volume_rub']
  out.update(seed=seed,user_id=user['user_id'],segment=seg,corridor=c,year=year,own_h_target_sum=float(d['own_targets'][h][selected].sum()),all_date_h5_base_sum=float(d['target'][d['valid']].mean()*len(selected)),exposure_weeks=len(market['dates'])/7)
  rows.append(out)
  if detailed:
   contacts.extend(dict(seed=seed,user_id=user['user_id'],segment=seg,corridor=c,year=year,date=market['dates'][day],horizon=h,h5_hit=d['target'][day],forward_bps=d['forward'][day]) for day in selected)
 return pd.DataFrame(rows),pd.DataFrame(contacts)

def weighted(frame,weights):
 means=frame.groupby('segment')[SUMS].mean().reindex(SEGMENTS).fillna(0)
 a=dict(zip(SUMS,np.array(weights)@means.to_numpy()))
 n=a['contacts'];a.update(contacts_per_week=n/a['exposure_weeks'],relevance=a['relevant_contacts']/n if n else np.nan,h5_quality=a['contact_target_sum']/n if n else np.nan,own_h_quality=a['own_h_target_sum']/n if n else np.nan,h5_unconditional_same_dates=a['all_date_h5_base_sum']/n if n else np.nan,forward_bps_per_contact=a['contact_forward_bps_sum']/n if n else np.nan,timing_bps_all_volume=a['gross_timing_value_rub']/a['organic_volume_rub']*10000)
 a['h5_lift_vs_unconditional']=a['h5_quality']-a['h5_unconditional_same_dates'];return a

def fit_policy(train_rows):
 stats=[]
 for cfg,frame in train_rows.items():
  agg=weighted(frame,MIXES[MAIN]);stats.append(dict(config=identity(cfg),segment='all',**agg))
  for s in SEGMENTS:
   weights=tuple(float(x==s) for x in SEGMENTS);stats.append(dict(config=identity(cfg),segment=s,**weighted(frame,weights)))
 stats=pd.DataFrame(stats)
 # Objective fixed in advance: joint timing RUB less 1 RUB/contact; not bank CM.
 # Equal model/menu size universal and each subgroup, minimum contacts prevents 1-event winner.
 def best(s,allowed=None):
  a=stats[(stats.segment==s)&(stats.contacts>=2)].copy()
  if allowed:a=a[a.config.isin([identity(c) for c in GRID if c[2] in allowed])]
  if len(a)==0:a=stats[stats.segment==s].copy()
  row=a.sort_values(['net_scenario_value_rub','contacts','config'],ascending=[False,True,True]).iloc[0]
  return next(c for c in GRID if identity(c)==row.config)
 common=best('all');uni={s:common for s in SEGMENTS};group={s:best(s,ALLOWED[s]) for s in SEGMENTS}
 unconstrained={s:best(s) for s in SEGMENTS}
 return uni,group,unconstrained,stats

def calibrate_budget(markets,records,scenario,config,threshold,target):
 # Prior-only expected budget matching; subsequent test count can drift.
 candidates=[]
 for thin in np.linspace(0,1,21):
  a,_=evaluate(markets,records,scenario,config,threshold,float(thin));num=weighted(a,MIXES[MAIN])['contacts'];candidates.append((abs(num-target),float(thin),num))
 _,p,n=min(candidates);return p,n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--train-users',type=int,default=35);ap.add_argument('--test-users',type=int,default=60);ap.add_argument('--output',type=Path,default=HERE/'results');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 panel,markets,paths=load_market();before={str(p.relative_to(ROOT)):v3.file_hash(p) for p in paths};rows=[];contactrows=[];fits=[];selections=[];summary=[]
 for testyear in (2024,2025,2026):
  prior=[y for y in markets if y<testyear];threshold=thresholds(panel,prior);base=SCENARIOS[0];training=pool(markets,prior,[7101],args.train_users,base)
  print('Fit',testyear,'prior',prior,'users',len(training),flush=True)
  alltrain={}
  for cfg in GRID:alltrain[cfg]=evaluate(markets,training,base,{s:cfg for s in SEGMENTS},threshold)[0]
  universal,groupaware,unconstrained,stats=fit_policy(alltrain);stats['test_year']=testyear;fits.append(stats)
  uni,_=evaluate(markets,training,base,universal,threshold);grp,_=evaluate(markets,training,base,groupaware,threshold)
  budget=min(weighted(uni,MIXES[MAIN])['contacts'],weighted(grp,MIXES[MAIN])['contacts'])
  policies={'universal':(universal,1.),'group_aware':(groupaware,1.),'group_unconstrained_exploratory':(unconstrained,1.),'v3_readiness':({s:'v3' for s in SEGMENTS},1.)}
  budget_receipts={}
  for name,config in [('universal_expected_budget',universal),('group_expected_budget',groupaware),('group_unconstrained_expected_budget',unconstrained),('random_expected_budget',{s:'random' for s in SEGMENTS}),('weekly_expected_budget',{s:'weekly' for s in SEGMENTS})]:
   p,realized=calibrate_budget(markets,training,base,config,threshold,budget);policies[name]=(config,p);budget_receipts[name]={'thin':p,'prior_contacts':realized}
  selections.append(dict(test_year=testyear,prior_years=prior,max_prior_score_date=str(panel[panel.fold_test_year.isin(prior)].date.max().date()),universal={s:identity(c) for s,c in universal.items()},group_aware={s:identity(c) for s,c in groupaware.items()},group_unconstrained_exploratory={s:identity(c) for s,c in unconstrained.items()},thresholds={f'h{h}_q{q}':v for (h,q),v in threshold.items()},expected_budget_target=budget,budget_calibration=budget_receipts))
  print('Chosen',selections[-1]['universal'],selections[-1]['group_aware'],flush=True)
  for scenario in SCENARIOS:
   records=pool(markets,[testyear],[8101,8102,8103],args.test_users,scenario)
   for name,(config,thin) in policies.items():
    frame,contacts=evaluate(markets,records,scenario,config,threshold,thin,detailed=scenario.name=='base')
    frame['policy']=name;frame['scenario']=scenario.name;rows.append(frame)
    if len(contacts):contacts['policy']=name;contactrows.append(contacts)
    for mix,weights in MIXES.items():summary.append(dict(year=testyear,scenario=scenario.name,policy=name,mix=mix,**weighted(frame,weights)))
  print('Done',testyear,flush=True)
 raw=pd.concat(rows,ignore_index=True);raw.to_csv(args.output/'client_outcomes.csv.gz',index=False);pd.concat(contactrows).to_csv(args.output/'base_contacts.csv.gz',index=False)
 pd.concat(fits).to_csv(args.output/'prior_policy_frontier.csv',index=False);pd.DataFrame(summary).to_csv(args.output/'weighted_by_year.csv',index=False)
 pooled=[]
 for (scenario,policy),frame in raw[raw.year<2026].groupby(['scenario','policy']):
  for mix,weights in MIXES.items():pooled.append(dict(scenario=scenario,policy=policy,mix=mix,**weighted(frame,weights)))
 pd.DataFrame(pooled).to_csv(args.output/'weighted_development.csv',index=False)
 sg=[]
 for (scenario,policy,segment),frame in raw[raw.year<2026].groupby(['scenario','policy','segment']):sg.append(dict(scenario=scenario,policy=policy,segment=segment,**weighted(frame,tuple(float(s==segment) for s in SEGMENTS))))
 pd.DataFrame(sg).to_csv(args.output/'by_segment_development.csv',index=False)
 (args.output/'selected_policy_receipts.json').write_text(json.dumps(selections,indent=2)+'\n')
 assert before=={str(p.relative_to(ROOT)):v3.file_hash(p) for p in paths}
 assert np.allclose(raw.policy_volume_rub,raw.organic_volume_rub)
 assert (raw[raw.scenario=='zero_response'].gross_timing_value_rub==0).all()
 manifest={'status':'retrospective_simulation_not_causal_uplift','model':MODEL,'horizons':HORIZONS,'grid':[identity(x) for x in GRID],'mixtures':MIXES,'exploratory_extension':'Unconstrained group policy added after seeing first constrained-cadence results; selection uses prior years only but this is an inspected retrospective followup, not untouched holdout.','input_sha256':before,'code_sha256':v3.file_hash(Path(__file__)),'v3_simulator_sha256':v3.file_hash(ROOT/'research_v3/behavior/simulate.py'),'train_seed':[7101],'eval_seeds':[8101,8102,8103],'train_users_per_segment':args.train_users,'test_users_per_segment_per_seed':args.test_users,'test_years':[2024,2025,2026],'development_years':[2024,2025],'diagnostic_years':[2026],'unique_market_rows':len(panel),'unique_market_dates':panel.date.nunique(),'common_market_rows_by_year':panel.groupby('fold_test_year').size().to_dict(),'synthetic_client_periods':len(raw),'tuning':'past OOT years only; all their h20 labels matured before following year; not independently uninspected OOT','no_new_transfer_count':True,'no_new_volume':True,'no_input_mutation':True,'cap2_per_rolling7days':bool((raw.cap_max<=2).all()),'future2026_production_eligible':False}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('Saved',args.output,flush=True)
if __name__=='__main__':main()
