#!/usr/bin/env python3
"""Regression tests for the final quant/macro training pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOLUTION_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from final_solution.data_pipeline import fetch_open_data as fetcher  # noqa: E402
from final_solution.training import train_and_evaluate as quant  # noqa: E402


class AvailabilityTest(unittest.TestCase):
    """Checks that source values cannot cross their declared availability date."""

    def test_asof_join_never_uses_future_source(self) -> None:
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "corridor": ["KZT", "KZT"],
            }
        )
        feature = pd.DataFrame(
            {
                "source_date": pd.to_datetime(["2026-01-02"]),
                "available_date": pd.to_datetime(["2026-01-03"]),
                "value": [42.0],
            }
        )

        joined = quant.asof_join(panel, feature, "toy", max_age_days=7)

        self.assertTrue(np.isnan(joined.loc[joined["date"].eq("2026-01-02"), "value"].iloc[0]))
        self.assertEqual(joined.loc[joined["date"].eq("2026-01-03"), "value"].iloc[0], 42.0)
        self.assertGreaterEqual(joined["toy_age_days"].dropna().min(), 0)

    def test_primary_real_panel_has_no_negative_source_age(self) -> None:
        core = quant.load_core(REPO_ROOT)
        panel = quant.build_feature_panel(
            REPO_ROOT,
            SOLUTION_ROOT / "data" / "normalized",
            core,
            quant.PROFILES["primary"],
        )

        age_columns = [column for column in panel if column.endswith("_age_days")]
        availability_columns = [column for column in panel if column.endswith("_available_date")]
        self.assertGreater(len(age_columns), 10)
        for column in age_columns:
            self.assertGreaterEqual(panel[column].dropna().min(), 0, column)
        for column in availability_columns:
            available = panel[column].notna()
            self.assertTrue((panel.loc[available, column] <= panel.loc[available, "date"]).all(), column)


class AdaptiveCadenceTest(unittest.TestCase):
    """Checks that the adaptive policy uses only past scores and obeys cooldown."""

    @staticmethod
    def frame(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
                "corridor": ["KZT"] * rows,
                "session_ordinal": np.arange(rows),
            }
        )

    def test_score_enters_reference_only_after_its_decision(self) -> None:
        scores = np.r_[np.zeros(40), 10.0, np.zeros(3), 1.0]
        core = quant.load_core(REPO_ROOT)

        selected, histories, _ = quant.select_with_rolling_quantile(
            core, self.frame(len(scores)), scores, 0.99
        )

        self.assertIn(40, selected.tolist())
        self.assertNotIn(44, selected.tolist())
        self.assertEqual(histories["__portfolio__"][-1], 1.0)

    def test_cooldown_separates_signals_by_four_sessions(self) -> None:
        scores = np.r_[np.zeros(40), np.ones(16)]
        core = quant.load_core(REPO_ROOT)

        selected, _, _ = quant.select_with_rolling_quantile(
            core, self.frame(len(scores)), scores, 0.5
        )

        self.assertTrue(np.all(np.diff(selected) >= 4))

    def test_portfolio_policy_selects_at_most_one_corridor_per_date(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]),
                "corridor": ["AMD", "KZT", "AMD", "KZT"],
                "session_ordinal": [0, 0, 1, 1],
            }
        )
        core = quant.load_core(REPO_ROOT)

        selected = core.select_with_cooldown(frame, np.array([0.7, 0.9, 0.8, 0.6]), 0.5)

        selected_dates = frame.loc[selected, "date"]
        self.assertFalse(selected_dates.duplicated().any())
        self.assertEqual(selected.tolist(), [1])

    def test_exact_score_ties_are_neutral_and_independent_of_row_order(self) -> None:
        core = quant.load_core(REPO_ROOT)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"] * 3),
                "corridor": ["AMD", "KZT", "UZS"],
                "session_ordinal": [0, 0, 0],
            },
            index=[10, 11, 12],
        )
        scores = np.ones(3)

        original = core.portfolio_best_candidates(frame, scores)
        reversed_frame = frame.iloc[::-1]
        reversed_result = core.portfolio_best_candidates(reversed_frame, scores[::-1])

        self.assertEqual(original["corridor"].iloc[0], reversed_result["corridor"].iloc[0])
        self.assertEqual(int(original["tie_count"].iloc[0]), 3)
        self.assertEqual(original["tie_rule"].iloc[0], core.PORTFOLIO_TIE_BREAK_RULE)

    def test_candidate_stream_is_per_corridor_before_portfolio_budget(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"]
                ),
                "corridor": ["AMD", "KZT", "AMD", "KZT", "AMD", "KZT"],
                "session_ordinal": [0, 0, 1, 1, 4, 4],
            }
        )
        scores = np.array([0.7, 0.9, 0.8, 0.6, 0.75, 0.85])
        core = quant.load_core(REPO_ROOT)

        candidates = core.select_per_corridor_with_cooldown(frame, scores, 0.5)
        portfolio = core.select_portfolio_from_candidates(frame, scores, candidates)

        self.assertEqual(candidates.tolist(), [0, 1, 4, 5])
        self.assertEqual(portfolio.tolist(), [1, 5])
        self.assertFalse(frame.loc[portfolio, "date"].duplicated().any())

    def test_threshold_uses_purged_validation_loss_inside_candidate_cadence_band(self) -> None:
        core = quant.load_core(REPO_ROOT)
        dates = pd.bdate_range("2026-01-01", periods=65)
        validation = pd.DataFrame(
            {
                "date": dates,
                "corridor": ["KZT"] * len(dates),
                "session_ordinal": np.arange(len(dates)),
                "target": np.r_[np.zeros(45), np.ones(20)],
            }
        )
        scores = np.linspace(0.0, 1.0, len(validation))

        threshold, frequency, weighted_error = core.choose_frequency_threshold(
            validation,
            scores,
        )
        selected = core.select_per_corridor_with_cooldown(validation, scores, threshold)

        self.assertIn("KZT", threshold)
        self.assertGreaterEqual(frequency, 1.0)
        self.assertLessEqual(frequency, 2.0)
        self.assertAlmostEqual(
            weighted_error,
            core.candidate_weighted_error(validation, selected),
        )

    def test_cadence_diagnostics_keep_zero_signal_weeks(self) -> None:
        core = quant.load_core(REPO_ROOT)
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05", periods=21, freq="D"),
                "corridor": ["KZT"] * 21,
            }
        )

        diagnostics = core.cadence_diagnostics(frame, [0])

        self.assertEqual(diagnostics["eligible_weeks"], 3.0)
        self.assertAlmostEqual(diagnostics["silent_week_share"], 2.0 / 3.0)
        self.assertGreaterEqual(
            diagnostics["maximum_calendar_gap_days_including_boundaries"],
            20.0,
        )

    def test_final_market_portfolio_is_subset_of_corridor_policy_signals(self) -> None:
        core = quant.load_core(REPO_ROOT)
        panel = core.add_target(
            core.build_panel(SOLUTION_ROOT / "data" / "cbr_daily.csv"),
            5,
        )
        experiment = core.Experiment("test_logit", "logistic", "core10")

        rows, predictions, _ = core.run_fold(experiment, 5, 2025, panel)

        self.assertTrue((~predictions["signal"] | predictions["candidate_signal"]).all())
        self.assertEqual(sum(row["signal_count"] for row in rows), int(predictions["signal"].sum()))
        self.assertEqual(
            sum(row["candidate_signal_count"] for row in rows),
            int(predictions["candidate_signal"].sum()),
        )

    def test_threshold_does_not_use_unresolved_validation_tail_labels(self) -> None:
        core = quant.load_core(REPO_ROOT)
        panel = core.add_target(
            core.build_panel(SOLUTION_ROOT / "data" / "cbr_daily.csv"),
            5,
        )
        mutated = panel.copy()
        tail_dates = (
            panel.loc[panel["date"].dt.year.eq(2024), "date"]
            .drop_duplicates()
            .sort_values()
            .tail(5)
        )
        mask = mutated["date"].isin(tail_dates)
        mutated.loc[mask, "target"] = 1 - mutated.loc[mask, "target"]
        experiment = core.Experiment("test_logit", "logistic", "core10")

        original_rows, original_predictions, _ = core.run_fold(experiment, 5, 2025, panel)
        mutated_rows, mutated_predictions, _ = core.run_fold(experiment, 5, 2025, mutated)

        self.assertEqual(original_rows[0]["threshold"], mutated_rows[0]["threshold"])
        np.testing.assert_array_equal(
            original_predictions["candidate_signal"],
            mutated_predictions["candidate_signal"],
        )


class StatisticalTest(unittest.TestCase):
    """Checks the sign and confidence logic of the paired block procedure."""

    def test_consistent_negative_loss_delta_is_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=365, freq="D"),
                "loss_delta": -np.ones(365),
            }
        )

        result = quant.stratified_month_bootstrap(frame, 2_000, np.random.default_rng(7))

        self.assertLess(result["ci95_high"], 0)
        self.assertLess(result["one_sided_block_signflip_p_improvement"], 0.01)

    def test_materiality_threshold_can_only_remove_positive_targets(self) -> None:
        core = quant.load_core(REPO_ROOT)
        panel = core.build_panel(SOLUTION_ROOT / "data" / "cbr_daily.csv")

        exact = core.add_target(panel, 5, materiality_bps=0.0)
        material = core.add_target(panel, 5, materiality_bps=25.0)

        self.assertTrue((material["target"].fillna(0) <= exact["target"].fillna(0)).all())

    def test_platt_rejects_rank_inverting_calibration(self) -> None:
        core = quant.load_core(REPO_ROOT)
        probabilities = np.array([0.9, 0.8, 0.2, 0.1])
        labels = pd.Series([0, 0, 1, 1])

        calibrator = core.fit_platt_calibrator(probabilities, labels)
        calibrated = core.apply_platt(calibrator, probabilities)

        self.assertEqual(calibrator.method, "identity")
        self.assertEqual(calibrator.status, "identity_fallback_non_monotone_platt")
        self.assertEqual(
            calibrator.applied_validation_brier,
            calibrator.raw_validation_brier,
        )
        self.assertTrue(np.isfinite(calibrator.attempted_platt_validation_brier))
        np.testing.assert_allclose(calibrated, probabilities)

    def test_platt_single_class_fallback_is_explicit(self) -> None:
        core = quant.load_core(REPO_ROOT)
        probabilities = np.array([0.2, 0.4, 0.6])

        calibrator = core.fit_platt_calibrator(probabilities, pd.Series([1, 1, 1]))

        self.assertEqual(calibrator.method, "identity")
        self.assertEqual(calibrator.status, "identity_fallback_single_class_validation")
        self.assertEqual(
            calibrator.applied_validation_brier,
            calibrator.raw_validation_brier,
        )
        self.assertTrue(np.isnan(calibrator.attempted_platt_validation_brier))


class DataIntegrityTest(unittest.TestCase):
    """Checks source repairs, freshness and provenance digests."""

    def test_boundary_copy_is_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "effective_date": pd.to_datetime(
                    ["2021-01-01", "2021-01-01", "2026-01-01", "2026-01-01"]
                ),
                "symbol": ["USD", "CNY", "USD", "CNY"],
                "normalized_value": [450.0, 68.0, 450.0, 68.0],
            }
        )

        repaired, repairs = fetcher.remove_boundary_copy(frame, "toy")

        self.assertEqual(repaired["effective_date"].min(), pd.Timestamp("2026-01-01"))
        self.assertEqual(len(repairs), 1)

    def test_real_nbk_boundary_corruption_is_absent(self) -> None:
        frame = pd.read_csv(
            SOLUTION_ROOT / "data" / "normalized" / "project_nbk_fx_snapshot.csv"
        )

        self.assertNotEqual(frame["effective_date"].min(), "2021-05-07")

    def test_all_artifact_hashes_and_pipeline_hash_match_manifest(self) -> None:
        manifest = json.loads(
            (SOLUTION_ROOT / "data" / "data_manifest.json").read_text()
        )

        for relative_path, metadata in manifest["artifacts"].items():
            path = SOLUTION_ROOT / "data" / relative_path
            self.assertEqual(quant.sha256(path), metadata["sha256"], relative_path)
        self.assertEqual(
            quant.sha256(SOLUTION_ROOT / "data_pipeline" / "fetch_open_data.py"),
            manifest["pipeline_sha256"],
        )

    def test_every_frequency_source_passes_declared_freshness(self) -> None:
        manifest = json.loads(
            (SOLUTION_ROOT / "data" / "data_manifest.json").read_text()
        )

        for source in manifest["sources"]:
            freshness = source.get("freshness")
            if freshness is None:
                continue
            checks = freshness.values() if "status" not in freshness else [freshness]
            for check in checks:
                self.assertEqual(check["status"], "pass", source["source_id"])

class SelectionTest(unittest.TestCase):
    """Checks that failed gates never masquerade as promoted selections."""

    def test_failed_candidate_is_diagnostic_only(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "fast_operational_candidate": True,
                    "research_source_class": "public_source_research",
                    "relative_brier_improvement": 0.1,
                    "eligible_for_prospective_shadow": False,
                }
            ]
        )

        diagnostic = quant.choose_diagnostic_candidates(audit)
        policy_audit = pd.DataFrame(columns=["config_id", "passes_policy_uncertainty_gate"])
        corridor_audit = pd.DataFrame(
            columns=[
                "config_id",
                "corridor",
                "passes_exploratory_corridor_policy_screen",
            ]
        )
        prospective = quant.choose_prospective_shadow_candidates(
            audit,
            policy_audit,
            corridor_audit,
        )

        self.assertIn("public_source_research", diagnostic)
        self.assertEqual(prospective, {})

    def test_policy_audit_without_candidates_is_machine_readable(self) -> None:
        family = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "eligible_for_policy_uncertainty_audit": False,
                }
            ]
        )

        audit = quant.development_policy_audit(pd.DataFrame(), family, 10)

        self.assertEqual(len(audit), 1)
        self.assertIn("status", audit)
        self.assertFalse(bool(audit.loc[0, "passes_policy_uncertainty_gate"]))

    def test_diagnostic_track_gets_policy_uncertainty_without_promotion_eligibility(self) -> None:
        family = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "eligible_for_policy_uncertainty_audit": False,
                }
            ]
        )
        frame = pd.DataFrame(
            {
                "config_id": ["candidate"] * 4,
                "horizon_cbr_rows_pub_proxy": [5] * 4,
                "date": pd.to_datetime(
                    ["2023-01-02", "2023-02-02", "2024-01-02", "2024-02-02"]
                ),
                "fold_test_year": [2023, 2023, 2024, 2024],
                "corridor": ["KZT"] * 4,
                "target": [0, 1, 0, 1],
                "forward_bps": [1.0, 2.0, 1.0, 2.0],
                "symmetric_bps": [1.0, 2.0, 1.0, 2.0],
                "regret_bps": [2.0, 1.0, 2.0, 1.0],
                "candidate_signal": [False, True, False, True],
                "signal": [False, True, False, True],
            }
        )

        audit = quant.development_policy_audit(
            frame,
            family,
            20,
            {"public_source_research": {"config_id": "candidate"}},
        )

        self.assertEqual(audit.loc[0, "status"], "evaluated")
        self.assertEqual(audit.loc[0, "audit_scope"], "diagnostic_track_only")

    def test_prospective_selection_requires_two_passing_corridors(self) -> None:
        audit = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "eligible_for_prospective_shadow": True,
                    "research_source_class": "public_source_research",
                    "relative_brier_improvement": 0.1,
                }
            ]
        )
        policy = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "passes_corridor_candidate_policy_uncertainty_gate": True,
                }
            ]
        )
        one_corridor = pd.DataFrame(
            [
                {
                    "config_id": "candidate",
                    "corridor": "KZT",
                    "passes_exploratory_corridor_policy_screen": True,
                }
            ]
        )
        two_corridors = pd.concat(
            [
                one_corridor,
                pd.DataFrame(
                    [
                        {
                            "config_id": "candidate",
                            "corridor": "AMD",
                            "passes_exploratory_corridor_policy_screen": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        rejected = quant.choose_prospective_shadow_candidates(audit, policy, one_corridor)
        accepted = quant.choose_prospective_shadow_candidates(audit, policy, two_corridors)

        self.assertEqual(rejected, {})
        self.assertIn("public_source_research", accepted)


if __name__ == "__main__":
    unittest.main()
