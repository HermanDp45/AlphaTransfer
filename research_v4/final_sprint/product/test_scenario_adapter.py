"""Meaningful stdlib API checks; no real contacts are sent."""
import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from research_v4.final_sprint.product import scenario_adapter as a
DAY='2026-04-10'

def candidate(scenario='NOW',probability=.55,baseline=.25):
 return {'scenario':scenario,'target_contract':a.CONTRACTS[scenario],'as_of':DAY,'known_at':DAY,'corridor':'KZT','probability':probability,'threshold':.4,'past_baseline_rate':baseline,'model_id':scenario+'_model','model_cutoff':'2026-01-01','calibration_end':'2025-12-31','factual_context':{'known_at':DAY,'ret1':.01,'pr60':.1,'rub_per_unit':.15,'recent_low_rank5':.1,'change_from_recent_low_bps':20}}

class ScenarioAdapterTest(unittest.TestCase):
 def test_separate_event_contract_required(self):
  x=candidate('CLOSING');x['target_contract']=a.NOW_CONTRACT;r=a.resolve_scenarios([x],DAY)
  self.assertIsNone(r['selected']);self.assertIn('scenario_target_contract_mismatch',r['reasons'])
 def test_compare_relative_quality_not_raw_probability(self):
  n=candidate('NOW',.55,.25);c=candidate('CLOSING',.8,.6);r=a.resolve_scenarios([c,n],DAY,context={'routing_mode':'relative_quality'})
  self.assertEqual(r['selected']['scenario'],'NOW');self.assertAlmostEqual(r['selected']['probability'],.55)
 def test_future_outcomes_and_future_calibration_rejected(self):
  for field,value in [('target',1),('forward_bps',100),('future_min',.1)]:
   x=candidate();x[field]=value;self.assertIsNone(a.resolve_scenarios([x],DAY)['selected'])
  x=candidate();x['calibration_end']='2026-02-01';self.assertIsNone(a.resolve_scenarios([x],DAY)['selected'])
 def test_shared_budget_and_input_state_immutable(self):
  st={'contacts':[{'date':'2026-04-05','corridor':'AMD','scenario':'CLOSING','model_id':'a'},{'date':'2026-04-07','corridor':'KZT','scenario':'NOW','model_id':'b'}]};old=copy.deepcopy(st);r=a.resolve_scenarios([candidate()],DAY,state=st)
  self.assertIsNone(r['selected']);self.assertIn('shared_contact_cap_exhausted',r['reasons']);self.assertEqual(st,old)
 def test_repeated_date_is_deduplicated(self):
  x=candidate();r=a.resolve_scenarios([x],DAY);self.assertIsNotNone(r['selected']);s=a.resolve_scenarios([x],DAY,state=r['next_state']);self.assertIsNone(s['selected'])
 def test_model_verdict_remains_with_neutral_factual_copy(self):
  x=candidate('NOW');x['factual_context']['pr60']=.8;r=a.resolve_scenarios([x],DAY)
  self.assertEqual(r['selected']['scenario'],'NOW');self.assertIn('Модельный сигнал',r['selected']['copy']['model_verdict_label']);self.assertFalse(r['selected']['copy']['contains_future_promise'])
  self.assertIsNone(a.resolve_scenarios([x],DAY,context={'require_strong_fact':True})['selected'])
 def test_routing_requires_both_heads_and_keeps_closing_score(self):
  n=candidate('NOW',.6,.25);c=candidate('CLOSING',.65,.55);ctx={'routing_mode':'now_confirmed_closing'};r=a.resolve_scenarios([n,c],DAY,context=ctx)
  self.assertEqual(r['selected']['scenario'],'CLOSING');self.assertEqual(r['selected']['probability'],.65);self.assertIsNone(a.resolve_scenarios([c],DAY,context=ctx)['selected'])
 def test_default_annotation_preserves_now_claim_and_one_contact(self):
  n=candidate('NOW',.6,.25);c=candidate('CLOSING',.65,.55);r=a.resolve_scenarios([n,c],DAY)
  self.assertEqual(r['selected']['scenario'],'NOW');self.assertEqual(r['selected']['probability'],.6);self.assertEqual(len(r['annotations']),1);self.assertEqual(r['annotations'][0]['target_contract'],a.CLOSING_CONTRACT);self.assertEqual(r['annotations'][0]['probability'],.65);self.assertEqual(len(r['next_state']['contacts']),1)
 def test_disagreeing_same_day_market_facts_fail_closed(self):
  n=candidate();c=candidate('CLOSING');c['factual_context']['rub_per_unit']=.18;r=a.resolve_scenarios([n,c],DAY);self.assertIsNone(r['selected'])
 def test_csv_outcomes_cannot_influence_candidate(self):
  row={'date':DAY,'source_known_at':DAY,'corridor':'KZT','probability':'.7','threshold':'.5','prior_baseline_rate':'.55','model_id':'closing','cutoff':'2026-01-01','ret1':'.01','pr60':'.1','rub_per_unit':'.15','recent_low_rank5':'.15','change_from_recent_low_bps':'20','target':'1','forward_bps':'100'}
  first=a.candidate_from_closing_row(row,'2025-12-31');row.update(target='0',forward_bps='-999999',future_rate='99999');self.assertEqual(first,a.candidate_from_closing_row(row,'2025-12-31'))
 def test_stale_facts_and_future_contact_state_rejected(self):
  x=candidate();x['factual_context']['known_at']='2025-01-01';self.assertIsNone(a.resolve_scenarios([x],DAY)['selected'])
  r=a.resolve_scenarios([candidate()],DAY,state={'contacts':[{'date':'2026-04-11'}]});self.assertEqual(r['status'],'rejected')
if __name__=='__main__':unittest.main()
