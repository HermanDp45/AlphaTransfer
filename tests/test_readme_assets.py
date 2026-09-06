from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_readme_assets.py"
SPEC = importlib.util.spec_from_file_location("build_readme_assets", SCRIPT)
assert SPEC and SPEC.loader
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)


class ReadmeAssetsTest(unittest.TestCase):
    def test_selected_profiles_and_public_rows(self) -> None:
        h3, h5 = assets.validate_and_load()
        self.assertEqual((h3.config_id, h3.horizon, h3.policy), ("tabm_kzt_fullhistory", 3, "rank80"))
        self.assertEqual((h5.config_id, h5.horizon, h5.policy), ("tabm_kzt_h5_fullhistory", 5, "rare65"))
        self.assertEqual(len(h3.annual) + len(h5.annual), 6)
        rows = assets.snapshot_rows((h3, h5))
        self.assertEqual(len(rows), 8)
        self.assertAlmostEqual(float(h3.aggregate["lift"]), 1.5042333624865076)
        self.assertAlmostEqual(float(h5.aggregate["lift"]), 1.9858935404634515)
        receipts = json.loads((ROOT / "research_v4/h5_fullhistory/receipts.json").read_text())
        self.assertEqual(len(receipts), 3)
        self.assertTrue(all(row["full_history"]["train_start"].startswith("2010-01-01") for row in receipts))

    def test_all_public_annual_cells_pass_minimum_gates(self) -> None:
        profiles = assets.validate_and_load()
        for profile in profiles:
            for row in profile.annual:
                self.assertGreaterEqual(float(row["lift"]), 1.3)
                self.assertGreater(float(row["forward_delta_bps"]), 0.0)
                if profile.model == "h3":
                    self.assertGreaterEqual(float(row["week_coverage"]), 0.8)
                else:
                    self.assertGreaterEqual(float(row["signals_per_corridor_week"]), 0.45)
                    self.assertLessEqual(float(row["signals_per_corridor_week"]), 0.85)

    def test_document_links_exist(self) -> None:
        for document in (ROOT / "README.md", ROOT / "docs" / "MODEL_METRICS.md"):
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                path = (document.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(path.exists(), f"broken link in {document}: {target}")

    def test_generated_assets_are_current(self) -> None:
        assets.check()
        for stem in ("07_signal_timeline", "08_trigger_map", "09_signal_pipeline"):
            self.assertTrue((ROOT / f"docs/assets/model-metrics/{stem}.png").is_file())
            self.assertTrue((ROOT / f"docs/assets/model-metrics/{stem}.svg").is_file())

    def test_agents_map_covers_current_entrypoints(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        required = (
            "final_solution/main.py",
            "final_solution/core.py",
            "final_solution/tabm_h3/bundle.json",
            "final_solution/tabm_h5/{bundle.json",
            "research_v4/h5_fullhistory/selection.json",
            "scripts/build_readme_assets.py --check",
            "--model h3 --action metrics",
            "--model h5 --action metrics",
        )
        for value in required:
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
