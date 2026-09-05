"""Optional V4 segment-policy display. No sends, no production promotion."""
from __future__ import annotations
from datetime import date
import math,re
from final_solution.alphatransfer_final.behavior import build_behavior_preview

def build_segment_policy_preview(context,market_scores,as_of,receipt,mode='universal'):
 """Compose V3 readiness with a previously frozen V4 market-policy receipt.

 context has frequency_segment and last_market_candidate_at in addition to V3
 fields. market_scores has known_at and probabilities keyed by h1/h3/h5/h10/h20.
 No score is changed. The chosen horizon's quality cannot replace common-h5 quality.
 """
 gate=build_behavior_preview(context,as_of)
 out={'status':'research_preview','simulation':gate['simulation'],'production_eligible':False,'causal_uplift_estimated':False,'fx_probability_modified':False,'readiness':gate,'candidate':False,'reasons':list(gate['suppression_reasons'])}
 if gate['status'] in ('disabled','unavailable','rejected'):
  out['status']=gate['status'];return out
 try:
  if mode not in ('universal','group_aware','group_unconstrained_exploratory'):raise ValueError('unsupported_policy_mode')
  if receipt['test_year']!=as_of.year:raise ValueError('policy_year_mismatch')
  if date.fromisoformat(receipt['max_prior_score_date'])>=date(as_of.year,1,1):raise ValueError('policy_not_fit_before_test_year')
  if not receipt['prior_years'] or max(receipt['prior_years'])>=as_of.year:raise ValueError('policy_contains_future_fit_year')
  segment=context['frequency_segment']
  config=receipt[mode][segment];m=re.fullmatch(r'h(1|3|5|10|20)_q(0\.\d+)_cad(\d+)',config)
  if not m:raise ValueError('invalid_policy_configuration')
  horizon=int(m[1]);quantile=float(m[2]);cadence=int(m[3]);threshold=float(receipt['thresholds'][f'h{horizon}_q{quantile}'])
  if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid_policy_threshold')
  if date.fromisoformat(market_scores['known_at'])!=as_of:raise ValueError('market_scores_not_from_decision_day')
  score=market_scores['probabilities'][f'h{horizon}']
  if isinstance(score,bool) or not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('invalid_market_probability')
  last=context.get('last_market_candidate_at')
  if last is not None and (as_of-date.fromisoformat(last)).days<cadence:out['reasons'].append('segment_market_candidate_cadence')
  if score<threshold:out['reasons'].append('segment_market_threshold')
  out.update(frequency_segment=segment,policy_mode=mode,selected_horizon_cbr_rows=horizon,probability=score,threshold=threshold,minimum_market_candidate_gap_calendar_days=cadence,candidate=bool(gate['preview_ready'] and not out['reasons']))
  out['effect_claim']='Retrospective scenario policy; group cadence underperformed universal policy in V4. No delivery or causal uplift authorized.'
 except (KeyError,TypeError,ValueError) as exc:
  out['status']='rejected';out['reasons'].append(str(exc));out['candidate']=False
 return out


def main():
 import argparse,json
 from pathlib import Path
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--context',type=Path,required=True);p.add_argument('--scores',type=Path,required=True);p.add_argument('--as-of',type=date.fromisoformat,required=True);p.add_argument('--receipt',type=Path,default=Path(__file__).parent/'results/selected_policy_receipts.json');p.add_argument('--policy',choices=['universal','group_aware','group_unconstrained_exploratory'],default='universal');a=p.parse_args()
 receipts=json.loads(a.receipt.read_text());receipt=next((r for r in receipts if r['test_year']==a.as_of.year),{})
 print(json.dumps(build_segment_policy_preview(json.loads(a.context.read_text()),json.loads(a.scores.read_text()),a.as_of,receipt,a.policy),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
