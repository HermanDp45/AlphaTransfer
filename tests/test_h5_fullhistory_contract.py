"""Contract tests for the selected H5 history window and rare-contact policy."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_v4" / "h5_fullhistory"
PRODUCTION = ROOT / "final_solution" / "tabm_h5"
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(
    (RESEARCH / "selection.json").is_file(),
    "full-history H5 experiment has not produced its sealed selection yet",
)
class H5FullHistoryResearchContractTest(unittest.TestCase):
    """Guard the facts used by production packaging and public reporting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = read_json(RESEARCH / "selection.json")
        cls.summary = pd.read_csv(RESEARCH / "summary.csv")
        cls.by_year = pd.read_csv(RESEARCH / "by_year.csv")
        cls.predictions = pd.read_csv(
            RESEARCH / "predictions.csv.gz",
            parse_dates=["date", "label_available_date"],
            low_memory=False,
        )
        cls.verification = read_json(RESEARCH / "verification.json")

    def test_artifact_schema_and_selected_profile_are_explicit(self) -> None:
        required = (
            "selection.json",
            "summary.csv",
            "by_year.csv",
            "predictions.csv.gz",
            "paired_intervals.csv",
            "policies.json",
            "protocol.json",
            "receipts.json",
            "verification.json",
        )
        for name in required:
            self.assertTrue((RESEARCH / name).is_file(), name)

        self.assertIn(self.selection["selected_recipe"], {"full_history", "120m"})
        self.assertEqual(self.selection["policy"], "rare65")
        self.assertEqual(int(self.selection["horizon"]), 5)
        expected_config = {
            "full_history": "tabm_kzt_h5_fullhistory",
            "120m": "tabm_kzt_h5_120m",
        }[self.selection["selected_recipe"]]
        self.assertEqual(self.selection["selected_config_id"], expected_config)
        self.assertIn(
            self.selection["selection_status"],
            {"qualified", "fallback_minimum_gate_violations"},
        )
        self.assertIn("gates", self.selection)
        self.assertIn("candidates", self.selection)

        metric_columns = {
            "config_id",
            "history_mode",
            "policy",
            "signals_per_corridor_week",
            "lift",
            "forward_delta_bps",
            "brier",
        }
        self.assertTrue(metric_columns <= set(self.summary.columns))
        self.assertTrue(metric_columns | {"year"} <= set(self.by_year.columns))

    def test_control_and_full_history_use_identical_test_cohorts(self) -> None:
        frame = self.predictions[self.predictions["policy"].eq("rare65")].copy()
        self.assertEqual(set(frame["history_mode"].unique()), {"full_history", "120m"})
        identity = ["fold_test_year", "date"]
        if "corridor" in frame:
            identity.append("corridor")
        if "session_ordinal" in frame:
            identity.append("session_ordinal")

        cohorts = {
            mode: set(map(tuple, rows[identity].itertuples(index=False, name=None)))
            for mode, rows in frame.groupby("history_mode")
        }
        self.assertEqual(cohorts["full_history"], cohorts["120m"])
        self.assertFalse(frame[identity + ["history_mode"]].duplicated().any())

    def test_selected_metrics_obey_gates_or_record_fallback(self) -> None:
        config_id = self.selection["selected_config_id"]
        aggregate = self.summary[
            self.summary["config_id"].eq(config_id)
            & self.summary["policy"].eq("rare65")
        ]
        annual = self.by_year[
            self.by_year["config_id"].eq(config_id)
            & self.by_year["policy"].eq("rare65")
        ].sort_values("year")

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(annual["year"].astype(int).tolist(), [2024, 2025, 2026])
        aggregate_ok = aggregate["signals_per_corridor_week"].between(0.60, 0.70).all()
        annual_ok = (
            annual["signals_per_corridor_week"].between(0.45, 0.85).all()
            and annual["lift"].ge(1.30).all()
            and annual["forward_delta_bps"].gt(0).all()
        )
        qualified = bool(aggregate_ok and annual_ok)
        self.assertEqual(self.selection["selection_status"] == "qualified", qualified)
        if not qualified:
            chosen = next(
                item
                for item in self.selection["candidates"]
                if item["config_id"] == config_id
            )
            self.assertGreater(int(chosen["violated_gates"]), 0)

    def test_selection_replays_the_frozen_lexicographic_rule(self) -> None:
        def key(row: dict) -> tuple:
            qualified = bool(row["qualified"])
            return (
                not qualified,
                int(row["violated_gates"]) if not qualified else 0,
                -float(row["min_annual_forward_delta_bps"]),
                -float(row["min_annual_lift"]),
                -float(row["aggregate_lift"]),
                -float(row["aggregate_forward_delta_bps"]),
                float(row["aggregate_brier"]),
                row["history_mode"] != "120m",
            )

        replay = sorted(self.selection["candidates"], key=key)
        self.assertEqual(replay[0]["config_id"], self.selection["selected_config_id"])
        self.assertEqual([int(row["rank"]) for row in replay], list(range(1, len(replay) + 1)))

    def test_maturity_preprocessing_and_policy_checks_are_sealed(self) -> None:
        for key in (
            "cohort_parity",
            "label_maturity",
            "train_only_preprocessing",
            "causal_rank_policy",
        ):
            self.assertIs(self.verification.get(key), True, key)

        self.assertTrue((self.predictions["label_available_date"] >= self.predictions["date"]).all())
        self.assertEqual(set(self.predictions["fold_test_year"].astype(int)), {2024, 2025, 2026})

    def test_full_history_receipts_prove_expanding_train_and_mature_labels(self) -> None:
        receipts = read_json(RESEARCH / "receipts.json")
        self.assertEqual([int(item["year"]) for item in receipts], [2024, 2025, 2026])
        for item in receipts:
            full = item["full_history"]
            self.assertEqual(full["config_id"], "tabm_kzt_h5_fullhistory")
            self.assertTrue(str(full["train_start"]).startswith("2010-01-01"))
            self.assertGreater(int(full["extra_train_rows"]), 0)
            self.assertLess(
                pd.Timestamp(full["train_latest_label"]),
                pd.Timestamp(full["calibration_start"]),
            )
            self.assertLess(
                pd.Timestamp(full["validation_latest_label"]),
                pd.Timestamp(f"{int(item['year'])}-01-01"),
            )
            self.assertEqual(len(full["source_sha256"]), 64)
            self.assertEqual(len(full["weights_sha256"]), 64)
            self.assertEqual(len(full["preprocessor_sha256"]), 64)

            coverage = full["feature_coverage"]
            bank_macro = {
                name: float(value)
                for name, value in coverage.items()
                if name.startswith("halyk_") or name.startswith("treasury_")
            }
            self.assertTrue(bank_macro)
            self.assertTrue(any(value < 1.0 for value in bank_macro.values()))

    def test_rare65_policy_is_calibrated_only_from_past_scores(self) -> None:
        policies = read_json(RESEARCH / "policies.json")
        self.assertEqual(len(policies), 6)
        self.assertEqual({item["history_mode"] for item in policies}, {"full_history", "120m"})
        self.assertEqual({int(item["year"]) for item in policies}, {2024, 2025, 2026})
        for item in policies:
            self.assertEqual(item["policy"], "rare65")
            self.assertAlmostEqual(float(item["target_signals_per_week"]), 0.65)
            self.assertEqual(int(item["rank_lookback"]), 63)
            self.assertEqual(int(item["cooldown"]), 2)
            self.assertEqual(int(item["max_contacts_per_week"]), 2)
            self.assertIn("strictly preceding", item["rank_contract"])

        scored = self.predictions.copy()
        scored["calendar_week"] = scored["date"].dt.to_period("W-SUN").astype(str)
        weekly = scored.groupby(["config_id", "fold_test_year", "calendar_week"])["candidate_signal"].sum()
        self.assertLessEqual(int(weekly.max()), 2)
        # Silence is an allowed product outcome; the policy does not backfill
        # an otherwise signal-free week merely to meet its calibration target.
        self.assertTrue(weekly.eq(0).any())

    def test_bootstrap_is_paired_and_marked_post_selection(self) -> None:
        intervals = pd.read_csv(RESEARCH / "paired_intervals.csv")
        expected = {
            "baseline",
            "candidate",
            "period",
            "delta_brier",
            "ci_low",
            "ci_high",
            "lift_delta",
            "lift_ci_low",
            "lift_ci_high",
            "forward_delta_bps_delta",
            "forward_delta_bps_ci_low",
            "forward_delta_bps_ci_high",
        }
        self.assertTrue(expected <= set(intervals.columns))
        self.assertEqual(set(intervals["baseline"]), {"120m"})
        self.assertEqual(set(intervals["candidate"]), {"full_history"})
        self.assertEqual(set(intervals["period"].astype(str)), {"2024", "2025", "2026", "all"})
        marker = " ".join(str(x) for x in intervals.get("inference", pd.Series(dtype=str)))
        marker += " " + json.dumps(self.selection, ensure_ascii=False)
        self.assertIn("post-selection", marker.lower().replace("_", "-"))


class H5ProductionContractTest(unittest.TestCase):
    def test_research_bundle_readiness_requires_ignored_heavy_artifacts(self) -> None:
        from final_solution.core import H5Runner

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "model": {
                            "weights": "model/weights.pt",
                            "preprocessor": "model/preprocess.joblib",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(H5Runner._research_bundle_ready(bundle))
            (root / "model").mkdir()
            (root / "model" / "weights.pt").write_bytes(b"weights")
            self.assertFalse(H5Runner._research_bundle_ready(bundle))
            (root / "model" / "preprocess.joblib").write_bytes(b"preprocessor")
            self.assertTrue(H5Runner._research_bundle_ready(bundle))

    @unittest.skipUnless(
        (PRODUCTION / "bundle.json").is_file(),
        "production H5 bundle has not been packaged yet",
    )
    def test_portable_csv_sources_rebuild_the_selected_training_frame(self) -> None:
        from research_v4.h5_fullhistory import experiment
        from research_v4.h5_fullhistory.final_fit import load_training_panel, make_parts

        panel, source, _ = load_training_panel(
            Path("/definitely/missing/alphatransfer-panel.pkl"),
            ROOT / "research_v4/h3_finalization/audit/latest_panel_receipt.json",
            pd.Timestamp("2026-09-05"),
        )
        _, parts = make_parts(panel, pd.Timestamp("2026-09-05"))
        receipt = read_json(PRODUCTION / "final_fit_receipt.json")
        fingerprint = experiment.fingerprint(
            parts["train"][["date", "corridor", *experiment.FEATURES, "target", "label_available_date"]]
        )
        self.assertEqual(source, "portable_bundle_csv")
        self.assertEqual(fingerprint, receipt["training_fingerprint"])
        self.assertEqual(len(parts["train"]), int(receipt["train_rows"]))

    def test_h5_cli_resolves_horizon_and_policy_without_internal_flags(self) -> None:
        from final_solution import core

        captured: dict = {}

        class StubRunner:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def infer(self, *, force: bool = False):
                return {"status": "ok"}

        with mock.patch.object(core, "H5Runner", StubRunner):
            core.execute(["--model", "h5", "--action", "infer", "--skip-data-prefetch"])

        self.assertEqual(captured["horizon"], 5)
        # ``None`` deliberately delegates to the sealed selection; an explicit
        # default is also valid.  The packaged-bundle test fixes the result to rare65.
        self.assertIn(captured["policy"], {None, "rare65"})

    def test_readme_does_not_link_internal_agents_map(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("(AGENTS.md)", readme)
        self.assertNotIn("Карта проекта для агентов", readme)

    @unittest.skipUnless(
        (PRODUCTION / "bundle.json").is_file(),
        "production H5 bundle has not been packaged yet",
    )
    def test_metrics_export_uses_selected_recipe_and_common_schema(self) -> None:
        from final_solution.core import H5Runner

        with tempfile.TemporaryDirectory() as directory:
            frame = H5Runner(
                policy=None,
                output_dir=Path(directory),
                action="metrics",
                horizon=5,
                corridor_filter="KZT",
                skip_data_prefetch=True,
            ).metrics()

        required = {
            "model",
            "dataset",
            "period",
            "corridor_filter",
            "horizon",
            "threshold_policy",
            "hit_rate",
            "lift",
            "forward_delta_bps",
            "regret_bps",
            "signals_per_corridor_week",
            "mean_cell_week_coverage",
        }
        self.assertTrue(required <= set(frame.columns))
        self.assertEqual(set(frame["model"]), {"h5"})
        self.assertEqual(set(frame["horizon"].astype(int)), {5})
        self.assertEqual(set(frame["threshold_policy"]), {"rare65"})
        aggregate = frame[frame["period"].astype(str).eq("2024-2026")]
        self.assertEqual(len(aggregate), 1)
        self.assertAlmostEqual(float(aggregate.iloc[0]["signals_per_corridor_week"]), 0.6592592592592592)

    @unittest.skipUnless(
        (PRODUCTION / "bundle.json").is_file() and (RESEARCH / "selection.json").is_file(),
        "production H5 bundle has not been packaged yet",
    )
    def test_production_bundle_is_tied_to_sealed_selection(self) -> None:
        selection_path = RESEARCH / "selection.json"
        selection = read_json(selection_path)
        bundle = read_json(PRODUCTION / "bundle.json")
        receipt = read_json(PRODUCTION / "selection_receipt.json")

        for name in (
            "bundle.json",
            "feature_contract.json",
            "policy.json",
            "training_receipt.json",
            "source_receipt.json",
            "selection_receipt.json",
        ):
            self.assertTrue((PRODUCTION / name).is_file(), name)

        self.assertEqual(int(bundle["train_horizon"]), 5)
        self.assertEqual(bundle["history_mode"], selection["selected_recipe"])
        self.assertEqual(bundle["policy"]["name"], "rare65")
        self.assertEqual(str(bundle["model_cutoff"])[:10], "2026-09-05")
        self.assertEqual(receipt["selection_sha256"], sha256(selection_path))
        self.assertEqual(receipt["selected_config_id"], selection["selected_config_id"])


if __name__ == "__main__":
    unittest.main()
