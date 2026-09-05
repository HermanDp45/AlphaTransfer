import copy,json,sys,unittest
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research_v4.segments import experiment as e
from research_v4.segments.preview import build_segment_policy_preview

class TemporalAndCounterfactualTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.panel,cls.markets,_=e.load_market()
 def test_future_score_poison_does_not_change_prior_thresholds(self):
  x=self.panel.copy();x.loc[x.fold_test_year>=2024,[f'probability_h{h}' for h in e.HORIZONS]]=1000
  self.assertEqual(e.thresholds(self.panel,[2023]),e.thresholds(x,[2023]))
 def test_causal_schedule_prefix_invariance(self):
  d=self.markets[2025]['corridors']['KZT'];threshold=e.thresholds(self.panel,[2023,2024]);cut=150
  for cfg in e.GRID:
   full=e.schedule(d,cfg,threshold);truncated=dict(d,valid=d['valid'][d['valid']<cut]);short=e.schedule(truncated,cfg,threshold)
   np.testing.assert_array_equal(full[full<cut],short)
 def test_common_grid_horizons_and_matured_labels_before_next_year(self):
  rates=pd.read_csv(ROOT/'final_solution/data/cbr_daily.csv',parse_dates=['date'])
  for (year,c),g in self.panel.groupby(['fold_test_year','corridor']):
   available=rates[rates.corridor==c].date.to_numpy();idx=np.searchsorted(available,g.date.to_numpy())
   if year<2026:self.assertTrue((available[idx+20]<np.datetime64(f'{year+1}-01-01')).all())
   self.assertTrue(g[[f'probability_h{h}' for h in e.HORIZONS]].notna().all().all())
 def test_zero_response_preserves_organic_path_and_caps_under_all_horizons(self):
  scenario=e.v3.Scenario(name='zero',response=0);records=e.pool(self.markets,[2025],[5551],3,scenario);threshold=e.thresholds(self.panel,[2023,2024])
  worlds_before=[{k:v.copy() if isinstance(v,np.ndarray) else v for k,v in r[3].items()} for r in records]
  for h in e.HORIZONS:
   f,_=e.evaluate(self.markets,records,scenario,{s:(h,.25,3) for s in e.SEGMENTS},threshold)
   self.assertTrue((f.gross_timing_value_rub==0).all());self.assertTrue((f.policy_volume_rub==f.organic_volume_rub).all());self.assertTrue((f.net_scenario_value_rub==-f.contacts).all());self.assertLessEqual(f.cap_max.max(),2)
  for before,r in zip(worlds_before,records):
   for k in before:np.testing.assert_array_equal(before[k],r[3][k])

class PreviewTests(unittest.TestCase):
 def setUp(self):
  self.context={'source':'synthetic','known_at':'2025-06-02','historical_transfer_count':12,'expected_transfer_date':'2025-06-04','available_balance_sufficient':True,'recent_explicit_intent':True,'last_completed_transfer':'2025-05-25','urgency_declared':False,'allow_fx_notifications':True,'remaining_shared_crm_slots_7d':1,'frequency_segment':'monthly','last_market_candidate_at':'2025-05-20'}
  self.receipt=json.loads((e.HERE/'results/selected_policy_receipts.json').read_text())[1];self.scores={'known_at':'2025-06-02','probabilities':{f'h{h}':1. for h in e.HORIZONS}}
 def test_simulation_and_future_refusal(self):
  a=build_segment_policy_preview(self.context,self.scores,date(2025,6,2),self.receipt);self.assertTrue(a['simulation']);self.assertTrue(a['candidate']);self.assertFalse(a['production_eligible']);self.assertFalse(a['fx_probability_modified'])
  bad=copy.deepcopy(self.context);bad['last_market_candidate_at']='2025-06-03';b=build_segment_policy_preview(bad,self.scores,date(2025,6,2),self.receipt);self.assertEqual(b['status'],'rejected');self.assertFalse(b['candidate'])
 def test_receipt_and_market_must_be_available_in_decision_year(self):
  bad=copy.deepcopy(self.receipt);bad['prior_years'].append(2025)
  self.assertEqual(build_segment_policy_preview(self.context,self.scores,date(2025,6,2),bad)['status'],'rejected')
  self.scores['known_at']='2025-06-03';self.assertEqual(build_segment_policy_preview(self.context,self.scores,date(2025,6,2),self.receipt)['status'],'rejected')
 def test_urgent_and_shared_cap_are_not_overridden_by_perfect_market_score(self):
  for field,value in [('urgency_declared',True),('remaining_shared_crm_slots_7d',0)]:
   c=dict(self.context);c[field]=value;a=build_segment_policy_preview(c,self.scores,date(2025,6,2),self.receipt);self.assertFalse(a['candidate']);self.assertFalse(a['production_eligible'])

if __name__=='__main__':unittest.main()
