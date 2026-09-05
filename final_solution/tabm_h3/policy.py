"""Pure, causal rank/fixed policy with explicit persistent score history."""
from __future__ import annotations
from datetime import date,datetime
from copy import deepcopy
import hashlib,json,math


def binding(config):
    fields={k:config[k] for k in ('model_id','model_cutoff','train_horizon','corridor','features','model','calibration','policy')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def empty_state(config):
    return {'schema_version':1,'binding_sha256':binding(config),'last_processed_date':None,'last_processed_session':None,'last_candidate_session':None,'week':None,'week_candidates':0,'past_scores':[]}


def _day(value):
    if isinstance(value,datetime):
        if value.tzinfo is not None or any((value.hour,value.minute,value.second,value.microsecond)):raise ValueError('Policy rows use calendar dates, not intraday timestamps')
        return value.date()
    if isinstance(value,date):return value
    return date.fromisoformat(str(value))


def validate_state(state,config):
    if state.get('schema_version')!=1 or state.get('binding_sha256')!=binding(config):raise ValueError('State belongs to another model/calibration/policy')
    last=state['last_processed_session'];seen=state['last_processed_date'];hist=state['past_scores'];window=config['policy']['window']
    if len(hist)>window:raise ValueError('State exceeds score history window')
    if last is None:
        if seen is not None or hist or state['last_candidate_session'] is not None:raise ValueError('Empty-state chronology is inconsistent')
    else:
        if seen is None or not hist:raise ValueError('Incomplete previous score state')
        if hist[-1]['session_ordinal']!=last or _day(hist[-1]['date'])!=_day(seen):raise ValueError('Score history endpoint differs from processed state')
        for left,right in zip(hist,hist[1:]):
            if right['session_ordinal']!=left['session_ordinal']+1 or _day(right['date'])<=_day(left['date']):raise ValueError('Score state omits or reorders sessions')
        if state['last_candidate_session'] is not None and state['last_candidate_session']>last:raise ValueError('Future candidate in state')
    if not 0<=state['week_candidates']<=config['policy']['max_contacts_per_week']:raise ValueError('Invalid weekly candidate count')
    for row in hist:
        if not math.isfinite(row['probability']) or not 0<=row['probability']<=1:raise ValueError('Invalid historical probability')
    return state


def replay(rows,config,state=None,*,emit_candidates=True):
    """Score rows must be chronological and contiguous; input state is never mutated.

    Rank uses only previous probabilities, and current probability is appended
    after the verdict. emit_candidates=False initializes score-only warmup.
    """
    policy=config['policy'];kind=policy['kind'];threshold=float(policy['threshold'])
    if kind not in ('rank','fixed') or not 0<=threshold<=1:raise ValueError('Invalid policy kind/threshold')
    if not 1<=policy['min_history']<=policy['window']==63:raise ValueError('Expected window63 and valid min_history')
    if not 0<=policy['cooldown_sessions'] or policy['max_contacts_per_week']!=2:raise ValueError('Invalid cooldown or weekly cap')
    current=deepcopy(validate_state(state or empty_state(config),config));out=[]
    for raw in rows:
        if set(raw)-{'date','corridor','session_ordinal','probability'}:raise ValueError('Only observed score fields may enter policy')
        if raw['corridor']!='KZT':raise ValueError('KZT-only policy')
        day=_day(raw['date']);session=int(raw['session_ordinal']);p=float(raw['probability'])
        if not math.isfinite(p) or not 0<=p<=1:raise ValueError('Probability must be finite in [0,1]')
        if current['last_processed_session'] is not None:
            if session!=current['last_processed_session']+1 or day<=_day(current['last_processed_date']):raise ValueError('Input overlaps or skips processed sessions')
        week=f'{day.isocalendar().year}-W{day.isocalendar().week:02d}'
        if week!=current['week']:current['week']=week;current['week_candidates']=0
        history=current['past_scores'];rank=None
        if len(history)>=policy['min_history']:
            rank=(sum(x['probability']<p for x in history)+.5*sum(x['probability']==p for x in history))/len(history)
        score=rank if kind=='rank' else p
        reasons=[]
        if score is None:reasons.append('insufficient_prior_score_history')
        elif score<threshold:reasons.append('below_past_fitted_threshold')
        if current['last_candidate_session'] is not None and session-current['last_candidate_session']<=policy['cooldown_sessions']:reasons.append('market_cooldown')
        if current['week_candidates']>=policy['max_contacts_per_week']:reasons.append('market_weekly_cap')
        if not emit_candidates:reasons.append('score_only_warmup')
        candidate=not reasons
        if candidate:current['last_candidate_session']=session;current['week_candidates']+=1
        out.append({'date':day.isoformat(),'corridor':'KZT','session_ordinal':session,'probability':p,'rank_score':rank,'prior_score_count':len(history),'policy_score':score,'threshold':threshold,'candidate_signal':candidate,'reason_codes':reasons or ['NOW_H3_model_policy_pass']})
        current['past_scores']=(history+[{'date':day.isoformat(),'session_ordinal':session,'probability':p}])[-policy['window']:]
        current['last_processed_session']=session;current['last_processed_date']=day.isoformat()
    validate_state(current,config)
    return out,current
