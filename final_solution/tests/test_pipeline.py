"""Regression tests for the central AlphaTransfer pipeline."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest


SOLUTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOLUTION_ROOT.parent
sys.path.insert(0, str(SOLUTION_ROOT))

from alphatransfer_final.artifacts import load_json, sha256, verify_lock  # noqa: E402
from alphatransfer_final.config import PipelineConfig  # noqa: E402
from alphatransfer_final.pipeline import run_pipeline  # noqa: E402
from alphatransfer_final.product import build_product_decision, load_candidates  # noqa: E402
from alphatransfer_final.scorecard import build_scorecard  # noqa: E402


class PipelineTest(unittest.TestCase):
    """Verify metrics, fail-closed policy and output receipts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = PipelineConfig.load(SOLUTION_ROOT / "config.toml", REPO_ROOT)
        cls.run_dir = cls.config.path("canonical_run_dir")

    def test_locked_inputs_are_unchanged(self) -> None:
        """The fast path must reject silently changed research inputs."""
        observed = verify_lock(REPO_ROOT, self.config.path("input_lock"))
        self.assertEqual(len(observed), 13)

    def test_selected_candidate_metrics_match_audited_values(self) -> None:
        """Headline metrics must be read from the audited selected row."""
        scorecard = build_scorecard(self.config, self.run_dir)
        lift = scorecard["headline_metrics"]["cell_standardized_lift"]
        bps = scorecard["headline_metrics"]["forward_official_reference_advantage_bps"]
        coverage = scorecard["headline_metrics"]["weekly_coverage_1_to_2_candidates"]
        self.assertAlmostEqual(lift["value"], 1.4351906688933171)
        self.assertAlmostEqual(bps["value"], 47.68564601517411)
        self.assertAlmostEqual(coverage["value"], 2 / 3)
        self.assertTrue(lift["gate_passed"])
        self.assertTrue(bps["gate_passed"])
        self.assertFalse(coverage["gate_passed"])

    def test_product_decision_never_promotes_research_preview(self) -> None:
        """A strong offline signal must still be suppressed without production gates."""
        scorecard = build_scorecard(self.config, self.run_dir)
        decision = build_product_decision(
            self.config,
            scorecard,
            date(2025, 12, 16),
        )
        self.assertEqual(decision["market_signal"]["corridor"], "RUB_AMD")
        self.assertFalse(decision["market_signal"]["client_copy_contains_forecast"])
        self.assertFalse(decision["delivery"]["eligible_to_send"])
        self.assertFalse(decision["delivery"]["external_message_sent"])
        self.assertIn(
            "model_not_production_promoted",
            decision["delivery"]["suppressed_reasons"],
        )

    def test_unknown_demo_date_fails_instead_of_faking_abstention(self) -> None:
        """A date absent from the tiny demo snapshot must require full predictions."""
        with self.assertRaisesRegex(RuntimeError, "--predictions"):
            load_candidates(self.config, date(2025, 12, 15), None)

    def test_selection_is_personalized_before_shared_contact_budget(self) -> None:
        """A client must be ranked only across corridors relevant to that client."""
        scorecard = build_scorecard(self.config, self.run_dir)
        context = {
            "client_id": "synthetic-kzt-only",
            "timezone": "Asia/Almaty",
            "relevant_corridors": ["KZT"],
            "marketing_contacts_last_7d": 0,
            "urgent_transfer": False,
            "has_saved_recipient": True,
            "alpha_quote": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(json.dumps(context), encoding="utf-8")
            decision = build_product_decision(
                self.config,
                scorecard,
                date(2025, 12, 16),
                client_context_path=path,
            )
        self.assertEqual(decision["market_signal"]["corridor"], "RUB_KZT")
        self.assertEqual(
            decision["delivery"]["candidate_generated_at_local"],
            "2025-12-16T12:05:00+05:00",
        )

    def test_end_to_end_outputs_have_valid_success_hashes(self) -> None:
        """One invocation must create the complete machine-readable package."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_pipeline(
                config=self.config,
                run_dir=self.run_dir,
                output_dir=output,
                as_of=date(2025, 12, 16),
                predictions_path=None,
                client_context_path=None,
                verify_locked_inputs=True,
            )
            self.assertEqual(result["status"], "complete")
            success = load_json(output / "_SUCCESS.json")
            self.assertEqual(success["status"], "complete")
            for name, expected in success["outputs"].items():
                self.assertEqual(sha256(output / name), expected)
            with (output / "signal_decision.json").open(encoding="utf-8") as source:
                payload = json.load(source)
            self.assertFalse(payload["delivery"]["eligible_to_send"])


if __name__ == "__main__":
    unittest.main()
