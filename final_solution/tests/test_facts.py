"""Semantic and temporal checks for financially meaningful notification copy."""
from datetime import date
from pathlib import Path
import csv
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alphatransfer_final.facts import historical_fact, factual_copy


class FactsTest(unittest.TestCase):
    def test_future_rows_cannot_change_the_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rates.csv"
            header = "date,corridor,rub_per_unit\n"
            prefix = "2025-01-01,AMD,10\n2025-01-02,AMD,11\n"
            path.write_text(header + prefix)
            before = historical_fact(path, "AMD", date(2025, 1, 2))
            path.write_text(header + prefix + "2025-01-03,AMD,0.001\n2025-01-04,AMD,999999\n")
            self.assertEqual(before, historical_fact(path, "AMD", date(2025, 1, 2)))

    def test_high_price_is_never_described_as_historical_low(self):
        path = Path(__file__).resolve().parents[1] / "data/cbr_daily.csv"
        fact = historical_fact(path, "KZT", date(2025, 12, 16))
        self.assertGreater(fact["percentile_midrank"], .5)
        copy = factual_copy(fact, "казахстанского тенге", "Казахстан")
        self.assertNotEqual(copy["scenario"], "HISTORICAL_LOW_REFERENCE")
        self.assertNotIn("благоприятн", copy["body"])
        self.assertIn("Курс ЦБ", copy["body"])

    def test_low_level_requires_a_full_history_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rates.csv"
            path.write_text("date,corridor,rub_per_unit\n2025-01-01,AMD,10\n2025-01-02,AMD,1\n")
            fact = historical_fact(path, "AMD", date(2025, 1, 2))
            self.assertFalse(fact["lower_quintile"])


if __name__ == "__main__":
    unittest.main()
