"""Stdlib research scenario resolver. No sends, outcomes or hidden future inputs."""
from __future__ import annotations
import argparse,csv,gzip,json,math
from datetime import date,datetime,timedelta
from pathlib import Path
NOW_CONTRACT='NOW:R[t]<=min(R[t+1:t+5]);R=RUB_per_recipient_unit;h=5_effective_CBR_rows;tau=0'
CLOSING_CONTRACT='CLOSING:R[t+5]>R[t];R=RUB_per_recipient_unit;h=5_effective_CBR_rows;tau=0'
CONTRACTS={'NOW':NOW_CONTRACT,'CLOSING':CLOSING_CONTRACT}
CANDIDATE_FIELDS={'scenario','target_contract','as_of','known_at','corridor','probability','threshold','past_baseline_rate','model_id','model_cutoff','calibration_end','factual_context'}
FACT_FIELDS={'known_at','ret1','pr60','rub_per_unit','recent_low_rank5','change_from_recent_low_bps'}

def _day(value):
 if isinstance(value,datetime):return value.date()
 if isinstance(value,date):return value
 text=str(value)
 if 'T' in text or ' ' in text:
  dt=datetime.fromisoformat(text.replace('Z','+00:00'))
  if dt.tzinfo is None:raise ValueError('datetime_requires_timezone')
  return dt.date()
 return date.fromisoformat(text)

def _number(value,lo=None,hi=None):
 x=float(value)
 if not math.isfinite(x) or (lo is not None and x<lo) or (hi is not None and x>hi):raise ValueError('invalid_numeric_value')
 return x

def factual_copy(candidate):
 fact=candidate['factual_context'];ret=_number(fact['ret1'],-10,10);rank=_number(fact['pr60'],0,1);rate=_number(fact['rub_per_unit'],0)
 if rate==0:raise ValueError('nonpositive_reference_rate')
 scenario=candidate['scenario'];day=candidate['as_of'];corridor=candidate['corridor']
 if scenario=='NOW':
  if rank<=.2:body=f'Курс ЦБ на {day} находится в нижних 20% последних 60 наблюдений.'
  elif ret<0:body=f'Курс ЦБ на {day} снизился на {100*abs(math.expm1(ret)):.2f}% с предыдущего наблюдения.'
  else:body=f'Официальный курс ЦБ на {day}: {rate:.4f} ₽ за единицу валюты.'
 else:
  if ret>0:body=f'Курс ЦБ на {day} вырос на {100*math.expm1(ret):.2f}% с предыдущего наблюдения.'
  else:body=f'Официальный курс ЦБ на {day}: {rate:.4f} ₽ за единицу валюты.'
  if ret>0 and fact.get('recent_low_rank5') is not None and _number(fact['recent_low_rank5'],0,1)<=.2:
   body+=' В последних пяти наблюдениях он находился в нижних 20% диапазона за 60 наблюдений.'
 strong=(rank<=.2 or ret<0) if scenario=='NOW' else ret>0
 return {'title':f'Курс для перевода в {corridor}','body':body+' Курс перевода и сумму к получению проверьте в приложении.','model_verdict_label':'Модельный сигнал: '+('выгодно сейчас' if scenario=='NOW' else 'окно закрывается'),'strong_historical_fact_gate':strong,'contains_future_promise':False,'source':'CBR official reference; not bank execution'},None

def resolve_scenarios(candidates,as_of,context=None,state=None):
 """Return a research contact proposal and copied state; never mutate inputs.

 Candidate probability and threshold refer to exactly one event contract. The
 policy ranks p / past-event-prevalence - 1. A shared rolling-seven-day cap and
 same-date dedup apply across both scenarios and all corridors for this client.
 """
 asof=_day(as_of);context=dict(context or {});state=dict(state or {});contacts=[dict(x) for x in state.get('contacts',[])];reasons=[];eligible=[];facts_seen={}
 if set(state)-{'contacts'}:return {'status':'rejected','reasons':['unexpected_state_fields'],'research_only':True}
 if set(context)-{'max_contacts_7d','other_contacts_7d','relevant_corridors','urgent_transfer','min_expected_lift','cooldown_calendar_days','require_strong_fact','routing_mode'}:return {'status':'rejected','reasons':['unexpected_context_fields'],'research_only':True}
 try:
  cap=int(context.get('max_contacts_7d',2));other=int(context.get('other_contacts_7d',0));gap=int(context.get('cooldown_calendar_days',2));minimum=_number(context.get('min_expected_lift',0.),0)
  if context.get('routing_mode','dual_annotations') not in ('relative_quality','now_confirmed_closing','dual_annotations'):raise ValueError('invalid_routing_mode')
  if cap<0 or cap>2 or other<0 or gap<0:raise ValueError('invalid_contact_policy')
  for x in contacts:
   if set(x)-{'date','corridor','scenario','model_id'} or _day(x['date'])>asof:raise ValueError('future_or_invalid_contact_state')
 except (ValueError,TypeError,KeyError) as exc:return {'status':'rejected','reasons':[str(exc)],'research_only':True}
 recent=[x for x in contacts if asof-timedelta(days=7)<_day(x['date'])<=asof]
 if context.get('urgent_transfer'):reasons.append('urgent_transfer_excluded_from_proactive_contact')
 if len(recent)+other>=cap:reasons.append('shared_contact_cap_exhausted')
 if any((asof-_day(x['date'])).days<gap for x in recent):reasons.append('shared_calendar_cooldown')
 if any(_day(x['date'])==asof for x in recent):reasons.append('already_contacted_on_date')
 for candidate in candidates:
  try:
   c=dict(candidate)
   if set(c)-CANDIDATE_FIELDS:raise ValueError('unexpected_candidate_fields_including_outcomes')
   scenario=c['scenario']
   if scenario not in CONTRACTS or c['target_contract']!=CONTRACTS[scenario]:raise ValueError('scenario_target_contract_mismatch')
   if _day(c['as_of'])!=asof or _day(c['known_at'])>asof:raise ValueError('candidate_date_or_known_at_invalid')
   if _day(c['model_cutoff'])>asof or _day(c['calibration_end'])>=_day(c['model_cutoff']):raise ValueError('model_or_calibration_future_information')
   if context.get('relevant_corridors') and c['corridor'] not in context['relevant_corridors']:raise ValueError('irrelevant_corridor')
   p=_number(c['probability'],0,1);threshold=_number(c['threshold'],0,1);base=_number(c['past_baseline_rate'],0,1)
   if base<=0:raise ValueError('missing_positive_past_event_baseline')
   fact=dict(c['factual_context'])
   if set(fact)-FACT_FIELDS or _day(fact['known_at'])!=asof:raise ValueError('stale_future_or_unexpected_fact_fields')
   signature=tuple(_number(fact[k]) for k in ('ret1','pr60','rub_per_unit'))
   if c['corridor'] in facts_seen and facts_seen[c['corridor']]!=signature:raise ValueError('inconsistent_market_facts_between_heads')
   facts_seen[c['corridor']]=signature
   c['factual_context']=fact;copy,error=factual_copy(c)
   if error:raise ValueError(error)
   if context.get('require_strong_fact') and not copy['strong_historical_fact_gate']:raise ValueError('optional_strong_fact_gate_failed')
   if p<threshold:raise ValueError('below_scenario_past_fitted_threshold')
   if p/base<minimum:raise ValueError('below_required_expected_relative_quality')
   eligible.append({'scenario':scenario,'corridor':c['corridor'],'target_contract':c['target_contract'],'model_id':c['model_id'],'probability':p,'past_baseline_rate':base,'expected_relative_quality':p/base,'ranking_utility':p/base-1,'copy':copy})
  except (ValueError,TypeError,KeyError) as exc:reasons.append(str(exc))
 blocked=any(r in reasons for r in ('urgent_transfer_excluded_from_proactive_contact','shared_contact_cap_exhausted','shared_calendar_cooldown','already_contacted_on_date','inconsistent_market_facts_between_heads'))
 eligible.sort(key=lambda x:(-x['ranking_utility'],x['scenario']!='NOW',x['corridor'],x['model_id']))
 chosen=eligible[0] if eligible and not blocked else None
 annotations=[]
 if context.get('routing_mode','dual_annotations') in ('now_confirmed_closing','dual_annotations') and not blocked:
  now=[x for x in eligible if x['scenario']=='NOW'];chosen=now[0] if now else None
  if chosen:
   closing=[x for x in eligible if x['scenario']=='CLOSING' and x['corridor']==chosen['corridor'] and x['copy']['strong_historical_fact_gate']]
   if closing:
    if context.get('routing_mode','dual_annotations')=='now_confirmed_closing':chosen=closing[0]
    else:annotations=[closing[0]]
  elif eligible:reasons.append('closing_requires_same_day_now_confirmation')
 if chosen:contacts.append({'date':asof.isoformat(),'corridor':chosen['corridor'],'scenario':chosen['scenario'],'model_id':chosen['model_id']})
 return {'status':'research_proposal' if chosen else 'suppressed','as_of':asof.isoformat(),'selected':chosen,'annotations':annotations,'eligible_scenarios':[x['scenario'] for x in eligible],'reasons':sorted(set(reasons)),'next_state':{'contacts':contacts},'research_only':True,'authorized_contact':False,'routing_mode':context.get('routing_mode','dual_annotations'),'conflict_rule':'Default NOW primary; optional CLOSING secondary confirmation; one shared contact. Other routing modes remain explicit diagnostics.','metric_contract':'NOW evaluated on every primary contact; CLOSING evaluated separately on annotated subset. Tags are nonexclusive, no extra contacts and no unqualified pooled lift.'}

def candidate_from_closing_row(row,calibration_end):
 """Explicit allowlist: ignore all outcome columns in offline prediction CSV."""
 return {'scenario':'CLOSING','target_contract':CLOSING_CONTRACT,'as_of':str(row['date'])[:10],'known_at':str(row['source_known_at'])[:10],'corridor':row['corridor'],'probability':float(row['probability']),'threshold':float(row['threshold']),'past_baseline_rate':float(row['prior_baseline_rate']),'model_id':row['model_id'],'model_cutoff':row['cutoff'],'calibration_end':calibration_end,'factual_context':{'known_at':str(row['source_known_at'])[:10],'ret1':float(row['ret1']),'pr60':float(row['pr60']),'rub_per_unit':float(row['rub_per_unit']),'recent_low_rank5':float(row['recent_low_rank5']),'change_from_recent_low_bps':float(row['change_from_recent_low_bps'])}}

def preview_closing(path,as_of,model_id,cutoff,context=None,state=None):
 """Historical local preview. Exact date only; missing dates remain unavailable."""
 opener=gzip.open if str(path).endswith('.gz') else open;rows=[]
 with opener(path,'rt',encoding='utf-8',newline='') as f:
  for row in csv.DictReader(f):
   if row['date']==str(as_of) and row['model_id']==model_id and row['cutoff']==cutoff and row['mode']=='normal':rows.append(row)
 if len(rows)!=1:return {'status':'unavailable','reasons':['no_unique_scored_row_for_exact_date'],'research_only':True}
 # The model's previous-year calibration is purged, so the day before cutoff
 # is a conservative upper bound for its latest permissible observed outcome.
 candidate=candidate_from_closing_row(rows[0],(_day(cutoff)-timedelta(days=1)).isoformat())
 return resolve_scenarios([candidate],as_of,{'routing_mode':'relative_quality',**(context or {})},state)

def main():
 p=argparse.ArgumentParser();p.add_argument('--as-of',required=True);p.add_argument('--cutoff',default='2026-01-01');p.add_argument('--model',default='closing_treasury_halyk_shrink120m');a=p.parse_args();path=Path(__file__).resolve().parent/'results/closing_predictions.csv.gz';print(json.dumps(preview_closing(path,a.as_of,a.model,a.cutoff),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
