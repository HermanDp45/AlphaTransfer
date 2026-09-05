"""Auditable time-series contracts checked against raw dates and frozen OOT rows.

These tests perform no model fitting, network calls, or artifact/cache mutations.
Run: python -m unittest discover -s research_v3/tests -p 'test_temporal.py' -v
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from final_solution.training import core_experiment as core
from research_v3.models import experiment


class TemporalContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_path = ROOT / 'final_solution/data/cbr_daily.csv'
        # Build directly from raw input; do not load or create model panel caches.
        cls.features = core.build_panel(cls.raw_path)
        cls.timeline = {
            c: pd.DatetimeIndex(g.sort_values('date').date)
            for c, g in cls.features.groupby('corridor')
        }
        cls.targets = {
            h: core.add_target(cls.features, h) for h in (1, 3, 5, 10, 20)
        }
        frozen = []
        names = ('development_h5_predictions.csv', 'diagnostic_2026_predictions.csv')
        for name in names:
            path = ROOT / 'final_solution/model_bundle' / name
            for chunk in pd.read_csv(path, usecols=['date', 'corridor', 'config_id'], chunksize=10000):
                chunk = chunk[chunk.config_id.eq('hgb_plus_cnyrub_basis')]
                frozen.extend(zip(pd.to_datetime(chunk.date), chunk.corridor))
        cls.frozen_oot = set(frozen)
        if len(cls.frozen_oot) != len(frozen):
            raise AssertionError('Frozen baseline contains duplicate date/corridor rows')

    def assert_label_matured_before(self, frame, horizon, boundary):
        """Use independently indexed raw corridor sessions, not purge_tail itself."""
        self.assertGreater(len(frame), 0, 'A vacuous empty split cannot pass')
        self.assertEqual(set(frame.corridor), set(self.timeline))
        for corridor, group in frame.groupby('corridor'):
            timeline = self.timeline[corridor]
            positions = timeline.get_indexer(pd.DatetimeIndex(group.date))
            self.assertTrue((positions >= 0).all())
            self.assertTrue((positions + horizon < len(timeline)).all())
            last_future_observation = timeline.take(positions + horizon)
            bad = last_future_observation >= boundary
            if bad.any():
                row = int(np.flatnonzero(bad)[0])
                self.fail(
                    f'{corridor} h={horizon}: label at {group.date.iloc[row]} '
                    f'uses {last_future_observation[row]}, not before {boundary}'
                )

    def test_train_and_calibration_labels_mature_before_next_boundary(self):
        # Include annual, monthly, short calibration, and h=20 cases. This catches
        # a row-vs-calendar-day purge error on the real CBR holiday/weekend grid.
        cases = [
            (1, '2023-01-01', '2024-01-01', 24, 1),
            (3, '2024-01-01', '2025-01-01', 24, 3),
            (5, '2023-01-01', '2024-01-01', 24, 12),
            (5, '2025-02-01', '2025-03-01', 6, 1),
            (10, '2026-01-01', '2027-01-01', 24, 3),
            (20, '2024-07-01', '2024-08-01', 24, 3),
        ]
        for horizon, start, end, months, validation_months in cases:
            with self.subTest(horizon=horizon, start=start, validation_months=validation_months):
                start = pd.Timestamp(start)
                val_start = start - pd.DateOffset(months=validation_months)
                spec = experiment.Spec('temporal_test', months=months, validation_months=validation_months)
                train, val, test = experiment.temporal_split(
                    self.targets[horizon], horizon, start, pd.Timestamp(end), spec,
                )
                self.assert_label_matured_before(train, horizon, val_start)
                self.assert_label_matured_before(val, horizon, start)
                train_keys = set(zip(train.date, train.corridor))
                val_keys = set(zip(val.date, val.corridor))
                test_keys = set(zip(test.date, test.corridor))
                self.assertFalse(train_keys & val_keys)
                self.assertFalse((train_keys | val_keys) & test_keys)
                self.assertTrue((test.date >= start).all())
                self.assertTrue((test.date < pd.Timestamp(end)).all())

    def test_h5_annual_oot_set_is_frozen_across_training_windows(self):
        for months, validation_months in ((3, 12), (24, 12), (60, 3), (0, 6)):
            with self.subTest(months=months, validation_months=validation_months):
                keys = set()
                spec = experiment.Spec('temporal_test', months=months, validation_months=validation_months)
                for year in (2023, 2024, 2025, 2026):
                    _, _, test = experiment.temporal_split(
                        self.targets[5], 5, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1), spec,
                    )
                    keys.update(zip(test.date, test.corridor))
                self.assertEqual(keys, self.frozen_oot,
                    'Training/calibration recency must not change the scored OOT universe')

    def test_historical_features_do_not_depend_on_future_raw_rows(self):
        # Preserve original numeric strings in the truncated file, preventing
        # CSV float round-trip differences from masquerading as feature leakage.
        for cutoff in ('2022-03-10', '2025-12-16'):
            with self.subTest(cutoff=cutoff), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'prefix.csv'
                with self.raw_path.open(newline='') as source, path.open('w', newline='') as target:
                    reader = csv.DictReader(source)
                    writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    writer.writerows(row for row in reader if row['date'] <= cutoff)
                prefix = core.build_panel(path)
                expected = self.features[self.features.date.le(pd.Timestamp(cutoff))]
                # Every generated past feature is compared, including common RUB
                # factors and rolling ranks. No future target/return labels exist
                # in build_panel, so excluding outcomes cannot hide feature drift.
                prefix = prefix.set_index(['date', 'corridor']).sort_index()
                expected = expected.set_index(['date', 'corridor']).sort_index()
                assert_frame_equal(prefix, expected, check_exact=True)

    def test_failed_annual_fit_restores_shared_model_and_split_functions(self):
        # Failure injection exercises the critical cleanup path without fitting
        # models or writing under the real model output directory.
        class FitFailure(RuntimeError):
            pass
        original_make = core.make_model
        original_split = core.split_for_year
        name = 'temporal_failure_injection'
        previous_features = core.FEATURE_GROUPS.get(name)
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(experiment, 'OUT', Path(directory)), \
                 patch.object(experiment, 'spec_panel', return_value=self.features), \
                 patch.object(core, 'run_fold', side_effect=FitFailure('intentional failed fit')):
                with self.assertRaisesRegex(FitFailure, 'intentional failed fit'):
                    experiment.run_annual(experiment.Spec(name), 5)
                self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertIs(core.make_model, original_make)
            self.assertIs(core.split_for_year, original_split)
        finally:
            if previous_features is None:
                core.FEATURE_GROUPS.pop(name, None)
            else:
                core.FEATURE_GROUPS[name] = previous_features
            # Avoid contaminating later tests even if cleanup assertions fail.
            core.make_model = original_make
            core.split_for_year = original_split


if __name__ == '__main__':
    unittest.main()
