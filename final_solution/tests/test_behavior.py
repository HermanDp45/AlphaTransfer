"""Safety/temporal contracts for optional behavior preview, without FX retraining."""
from datetime import date
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alphatransfer_final.behavior import build_behavior_preview


class BehaviorPreviewTest(unittest.TestCase):
    def context(self, **changes):
        c={'source':'bank_observed','known_at':'2026-09-04','historical_transfer_count':5,
           'expected_transfer_date':'2026-09-08','available_balance_sufficient':True,
           'recent_explicit_intent':False,'last_completed_transfer':'2026-08-08',
           'urgency_declared':False,'allow_fx_notifications':True,'remaining_shared_crm_slots_7d':1}
        c.update(changes)
        return c

    def run_gate(self, **changes):
        return build_behavior_preview(self.context(**changes),date(2026,9,4))

    def test_missing_data_never_fabricates_eligibility(self):
        self.assertEqual(build_behavior_preview(None,date(2026,9,4))['status'],'disabled')
        self.assertEqual(build_behavior_preview({},date(2026,9,4))['status'],'disabled')
        result=self.run_gate(available_balance_sufficient=None)
        self.assertEqual(result['status'],'unavailable')
        self.assertIsNone(result['preview_ready'])

    def test_future_history_and_observation_are_rejected_but_future_plan_allowed(self):
        self.assertTrue(self.run_gate()['preview_ready'])
        for field in ['known_at','balance_observed_at','intent_observed_at','history_through','last_completed_transfer']:
            with self.subTest(field=field):
                r=self.run_gate(**{field:'2026-09-05'})
                self.assertEqual(r['status'],'rejected')
                self.assertFalse(r['production_eligible'])
                self.assertIn('future_behavior_observation:'+field,r['suppression_reasons'])

    def test_snapshot_cannot_contain_data_observed_after_it(self):
        r=self.run_gate(known_at='2026-09-03',balance_observed_at='2026-09-04')
        self.assertEqual(r['status'],'rejected')
        self.assertIn('observation_after_context_known_at',r['suppression_reasons'])

    def test_simulation_never_promotes_and_is_explicit(self):
        for source in ['demo','synthetic']:
            r=self.run_gate(source=source)
            self.assertEqual(r['status'],'simulation')
            self.assertTrue(r['simulation'])
            self.assertTrue(r['preview_ready'])
            self.assertFalse(r['production_eligible'])
            self.assertFalse(r['causal_uplift_estimated'])
            self.assertFalse(r['fx_probability_modified'])
        self.assertEqual(self.run_gate(synthetic=True)['status'],'rejected')

    def test_urgency_consent_and_budget_each_override_readiness(self):
        for changes,reason in [({'urgency_declared':True},'urgent_transfer_use_organic_flow'),
                               ({'allow_fx_notifications':False},'fx_notifications_not_consented'),
                               ({'remaining_shared_crm_slots_7d':0},'shared_crm_cap'),
                               ({'last_completed_transfer':'2026-09-03'},'recent_transfer')]:
            r=self.run_gate(**changes)
            self.assertFalse(r['preview_ready'])
            self.assertIn(reason,r['suppression_reasons'])

    def test_cold_start_needs_explicit_intent_and_confirmed_money(self):
        self.assertFalse(self.run_gate(historical_transfer_count=0)['preview_ready'])
        self.assertTrue(self.run_gate(historical_transfer_count=0,recent_explicit_intent=True)['preview_ready'])
        self.assertFalse(self.run_gate(historical_transfer_count=0,recent_explicit_intent=True,available_balance_sufficient=False)['preview_ready'])

    def test_json_strings_cannot_be_coerced_to_true(self):
        r=self.run_gate(available_balance_sufficient='false')
        self.assertEqual(r['status'],'rejected')
        self.assertEqual(self.run_gate(last_completed_transfer='tomorrow')['status'],'rejected')

if __name__=='__main__':unittest.main()
