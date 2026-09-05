"""Regression checks for the exposed V3 as-of contract."""
import sys
from pathlib import Path
from datetime import date
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research_v3.preview import decision


class PreviewV3Test(unittest.TestCase):
    def test_explicit_empty_corridors_never_broadens_audience(self):
        with self.assertRaisesRegex(ValueError,"empty"):
            decision(date(2025,12,16),corridors=[])

    def test_invalid_probability_is_rejected_by_python_api(self):
        for threshold in [float('nan'),float('inf'),-.1,1.2]:
            with self.assertRaisesRegex(ValueError,"finite probability"):
                decision(date(2025,12,16),threshold=threshold)

    def test_future_behavior_surfaces_in_delivery_suppression(self):
        result=decision(date(2025,12,16),policy='legacy',context={'source':'bank_observed','known_at':'2026-01-01'})
        self.assertEqual(result['behavior']['status'],'rejected')
        self.assertTrue(any(s.startswith('behavior:future_behavior') for s in result['suppression_reasons']))
        self.assertFalse(result['eligible_to_send'])

    def test_historical_candidate_has_supported_copy(self):
        result=decision(date(2025,12,16),policy='legacy',corridors=['KZT'])
        selected=result['selected']
        self.assertEqual(selected['corridor'],'KZT')
        self.assertFalse(selected['factual_evidence']['lower_quintile'])
        self.assertNotEqual(selected['copy']['scenario'],'HISTORICAL_LOW_REFERENCE')


if __name__=='__main__':unittest.main()
