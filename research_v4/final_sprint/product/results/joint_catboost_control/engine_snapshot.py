"""Parameterised fixed joint scheduling for the actual root NOW champion."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,time
import numpy as np
import pandas as pd
from research_v4.final_sprint import common
from research_v4.final_sprint.product import closing_experiment as c
HERE=c.HERE;OUT=c.OUT/'joint';SPRINT=HERE.parent
CLOSING='closing_treasury_halyk_shrink120m'
VARIANTS=('NOW_only','joint','joint_closing_fact_gate','joint_both_fact_gates','NOW_confirmed_closing','dual_annotations')

def history_source(config,explicit=None):
 if explicit:return [Path(explicit)]
 selected=SPRINT/'selected_history.csv.gz'
 if selected.exists():return [selected]
 if config.startswith('hgb_'):return [SPRINT/'hgb'/f'{config}_{cut}_calibration.csv.gz' for cut in ('2025-01-01','2026-01-01')]
 if config.startswith('catboost_'):
  p=SPRINT/'catboost'/('matched/history_and_calibration.csv.gz' if config.endswith('_matched') else 'candidates_calibration_predictions.csv.gz')
  return [p]
 raise ValueError('Provide --now-history for this selected model family')

def schedule(frame,thresholds,base,cooldown,variant,initial=None):
 state={'last':-10000,'week':None,'count':0} if initial is None else dict(initial);chosen=[]
 for row in frame.sort_values(['date','corridor']).itertuples():
  week=str(row.date.to_period('W-SUN'))
  if state['week']!=week:state.update(week=week,count=0)
  emitted=None
  if int(row.session_ordinal)-state['last']>cooldown and state['count']<2:
   if variant in ('NOW_confirmed_closing','dual_annotations'):
    if np.isfinite(row.now_probability) and row.now_probability>=thresholds['NOW']:
     emitted='CLOSING' if variant=='NOW_confirmed_closing' and np.isfinite(row.closing_probability) and row.closing_probability>=thresholds['CLOSING'] and bool(row.closing_text_gate) else 'NOW'
     state.update(last=int(row.session_ordinal),count=state['count']+1)
    chosen.append({'index':row.Index,'selected_scenario':emitted});continue
   eligible=[]
   for scenario,prob in [('NOW',row.now_probability),('CLOSING',row.closing_probability)]:
    if scenario=='CLOSING' and variant=='NOW_only':continue
    if scenario=='CLOSING' and variant in ('joint_closing_fact_gate','joint_both_fact_gates') and not bool(row.closing_text_gate):continue
    if scenario=='NOW' and variant=='joint_both_fact_gates' and not bool(row.now_text_gate):continue
    if np.isfinite(prob) and prob>=thresholds[scenario] and base[scenario]>0:eligible.append((float(prob)/base[scenario],scenario=='NOW',scenario))
   if eligible:
    emitted=max(eligible)[2];state.update(last=int(row.session_ordinal),count=state['count']+1)
  chosen.append({'index':row.Index,'selected_scenario':emitted})
 out=frame.copy();mapping={x['index']:x['selected_scenario'] for x in chosen};out['selected_scenario']=[mapping[i] for i in out.index];out['signal']=out.selected_scenario.notna();out['variant']=variant
 out['closing_annotation']=bool(variant=='dual_annotations')&out.signal&out.closing_probability.ge(thresholds['CLOSING'])&out.closing_text_gate
 return out,state

def metrics(g):
 selected=g[g.signal];weeks=pd.period_range(g.date.min(),g.date.max(),freq='W-SUN');counts=selected.date.dt.to_period('W-SUN').value_counts().reindex(weeks,fill_value=0)
 result={'rows':len(g),'dates':g.date.nunique(),'combined_contacts':len(selected),'weeks':len(weeks),'weeks_1_to_2_share':float(counts.between(1,2).mean()),'silent_weeks':int(counts.eq(0).sum()),'weeks_above2':int(counts.gt(2).sum()),'combined_forward_delta_bps':float(selected.forward_bps.mean()-g.forward_bps.mean())}
 for scenario,target in [('NOW','now_target'),('CLOSING','closing_target')]:
  mask=(g.signal if scenario=='NOW' else g.closing_annotation) if g.variant.eq('dual_annotations').all() else g.selected_scenario.eq(scenario)
  s=g[mask];base=float(g[target].mean());hit=float(s[target].mean());result.update({scenario+'_contacts':len(s),scenario+'_baseline_hit':base,scenario+'_hit_rate':hit,scenario+'_lift':hit/base if len(s) and base else np.nan,scenario+'_forward_delta_bps':float(s.forward_bps.mean()-g.forward_bps.mean()),scenario+'_endpoint_delta_bps':float(s.endpoint_bps.mean()-g.endpoint_bps.mean())})
 result['scenario_evaluation']='NOW primary on all contacts; CLOSING optional secondary subset, nonexclusive' if g.variant.eq('dual_annotations').all() else 'exclusive selected scenario'
 result['both_scenario_lift_ge1_3']=bool(result['NOW_contacts']>0 and result['CLOSING_contacts']>0 and result['NOW_lift']>=1.3 and result['CLOSING_lift']>=1.3)
 result['joint_acceptance']=bool(result['both_scenario_lift_ge1_3'] and result['weeks_1_to_2_share']>=.85)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--now-predictions',default=str(SPRINT/'selected_predictions.csv.gz'));parser.add_argument('--now-history');parser.add_argument('--selection',default=str(SPRINT/'selection.json'));args=parser.parse_args();OUT.mkdir(exist_ok=True)
 selection=json.loads(Path(args.selection).read_text());winner=selection['champion'];config=winner['config_id'];policyname=winner['policy'];nowpath=Path(args.now_predictions);sources=history_source(config,args.now_history)
 source_sha={str(p.relative_to(ROOT)):c.sha(p) for p in [Path(args.selection),nowpath,*sources]};now=pd.read_csv(nowpath,parse_dates=['date']);now=now[now.config_id.eq(config)&now.policy.eq(policyname)]
 nh=pd.concat([pd.read_csv(p,parse_dates=['date']) for p in sources],ignore_index=True);nh=nh[nh.config_id.eq(config)].copy();assert len(nh)>0
 closing=pd.read_csv(c.OUT/'closing_predictions.csv.gz',parse_dates=['date','label_available_date']);closing=closing[closing.model_id.eq(CLOSING)];ch=pd.read_csv(c.OUT/'closing_history.csv.gz',parse_dates=['date','label_available_date']);ch=ch[ch.model_id.eq(CLOSING)]
 outputs=[];policies=[];rows=[];parity=[]
 for cutoff in ('2025-01-01','2026-01-01'):
  h=ch[ch.cutoff.eq(cutoff)].copy();cv=h[h.calibration_eligible].copy();cv['probability']=cv['probability'];cp=common.fit_policy(cv,h,policyname)
  hist=nh[nh.cutoff.eq(cutoff)&nh['split'].eq('history')].copy();nv=nh[nh.cutoff.eq(cutoff)&nh['split'].isin(['validation','calibration'])].copy()
  # HGB exports may omit label_available_date; never infer maturity from target presence.
  if 'label_available_date' not in nv:nv=nv.merge(h[['date','corridor','label_available_date']],on=['date','corridor'],validate='one_to_one')
  else:nv['label_available_date']=pd.to_datetime(nv.label_available_date)
  assert nv.label_available_date.lt(pd.Timestamp(cutoff)).all() and nv.target.notna().all();np_=common.fit_policy(nv,hist,policyname)
  assert cp['cooldown']==np_['cooldown'];cd=cp['cooldown'];thresholds={'NOW':np_['threshold']['KZT'],'CLOSING':cp['threshold']['KZT']};base={'NOW':float(nv.target.mean()),'CLOSING':float(cv.target.mean())}
  joint_history=h.rename(columns={'probability':'closing_probability'}).merge(hist[['date','corridor','probability']].rename(columns={'probability':'now_probability'}),on=['date','corridor'],how='outer',validate='one_to_one').sort_values('date').reset_index(drop=True)
  assert joint_history.session_ordinal.notna().all()
  states={variant:schedule(joint_history,thresholds,base,cd,variant)[1] for variant in VARIANTS}
  policies.append({'cutoff':cutoff,'now_config':config,'policy':policyname,'thresholds':thresholds,'past_baseline_rates':base,'cooldown':cd,'initial_states':states,'closing_policy':cp,'now_policy':np_,'history_start':str(joint_history.date.min().date()),'history_end':str(joint_history.date.max().date())})
  for mode in ('normal','bank_delayed'):
   n=now[now.cutoff.eq(cutoff)&now['mode'].eq(mode)].copy();q=closing[closing.cutoff.eq(cutoff)&closing['mode'].eq(mode)].copy()
   z=q.rename(columns={'probability':'closing_probability'}).merge(n[['date','corridor','probability','target','candidate_signal']].rename(columns={'probability':'now_probability','target':'root_now_target','candidate_signal':'root_now_signal'}),on=['date','corridor'],validate='one_to_one')
   assert len(z)==len(n)==len(q);np.testing.assert_array_equal(z.now_target,z.root_now_target)
   for variant in VARIANTS:
    scored,_=schedule(z,thresholds,base,cd,variant,states[variant]);scored['cutoff']=cutoff;scored['mode']=mode;scored['now_model_id']=config;outputs.append(scored)
    if variant in ('NOW_only','NOW_confirmed_closing','dual_annotations'):
     mismatch=int((scored.signal!=scored.root_now_signal).sum());assert mismatch==0,('NOW reproduction',cutoff,mode,mismatch);parity.append({'cutoff':cutoff,'mode':mode,'variant':variant,'rows':len(z),'candidate_mismatches':mismatch})
    rows.append({'cutoff':cutoff,'mode':mode,'variant':variant,'now_model_id':config,'policy':policyname,**metrics(scored)})
 pred=pd.concat(outputs,ignore_index=True);pred.to_csv(OUT/'predictions.csv.gz',index=False);summary=pd.DataFrame(rows);summary.to_csv(OUT/'metrics.csv',index=False);summary[summary.variant.eq('dual_annotations')].to_csv(OUT/'annotations_metrics.csv',index=False);c.save(OUT/'policies.json',policies);c.save(OUT/'now_parity.json',parity)
 c.save(OUT/'manifest.json',{'now_selection':selection,'now_input_sha256':source_sha,'now_history_sources':[str(p.relative_to(ROOT)) for p in sources],'code_sha256':c.sha(Path(__file__)),'protocol_sha256':c.sha(HERE/'JOINT_PROTOCOL.md'),'semantic_routing_addendum_sha256':c.sha(HERE/'ROUTING_ADDENDUM.md'),'closing_predictions_sha256':c.sha(c.OUT/'closing_predictions.csv.gz'),'closing_history_sha256':c.sha(c.OUT/'closing_history.csv.gz'),'prediction_sha256':c.sha(OUT/'predictions.csv.gz'),'selection_uses_2026':selection.get('selection_uses_2026',True),'joint_test_threshold_search':False,'metric_contract':'separate NOW and CLOSING hits/lift; only cadence and reference forward utility pooled'})
 print(pd.DataFrame(rows)[['cutoff','mode','variant','combined_contacts','weeks_1_to_2_share','NOW_contacts','NOW_lift','CLOSING_contacts','CLOSING_lift','joint_acceptance']].to_string(index=False))
if __name__=='__main__':main()
