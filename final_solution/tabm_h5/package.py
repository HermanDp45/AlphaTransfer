#!/usr/bin/env python3
"""Package the selected H5 research recipe into a standalone runtime bundle.

The research directory is deliberately treated as an input contract.  It may
change which history recipe won, while production always receives the same
layout and validates the copied checkpoint before publishing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from final_solution.tabm_h5.model import load_config
from final_solution.tabm_h5.policy import binding

DEFAULT_RESEARCH = REPO / "research_v4" / "h5_fullhistory"
DEFAULT_DEST = REPO / "final_solution" / "tabm_h5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _paths_in(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                found.append(item)
            found.extend(_paths_in(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(_paths_in(item, keys))
    return found


def _resolve(research: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for base in (REPO, research):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (research / path).resolve()


def _first_existing(research: Path, selection: Any, keys: set[str], conventional: list[Path]) -> Path:
    candidates = [_resolve(research, value) for value in _paths_in(selection, keys)] + conventional
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No selected artifact found. Checked: " + ", ".join(str(x) for x in candidates))


def package(research: Path = DEFAULT_RESEARCH, dest: Path = DEFAULT_DEST) -> dict[str, Any]:
    selection_path = research / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing H5 selection: {selection_path}")
    selection = read_json(selection_path)
    bundle_path = _first_existing(
        research,
        selection,
        {"production_bundle", "bundle", "bundle_path", "final_bundle"},
        [research / "production_bundle" / "bundle.json", research / "final_fit" / "bundle.json"],
    )
    if bundle_path.is_dir():
        bundle_path = bundle_path / "bundle.json"
    config, source_dir = load_config(bundle_path)
    selected_recipe = selection.get("selected_recipe")
    if selected_recipe not in {"full_history", "120m"}:
        raise ValueError("selection.json selected_recipe must be 'full_history' or '120m'")
    selected_config_id = selection.get("selected_config_id")
    if not selected_config_id:
        raise ValueError("selection.json must contain selected_config_id")
    if selection.get("policy") != "rare65" or int(selection.get("horizon", -1)) != 5:
        raise ValueError("Production H5 selection must use horizon=5 and policy=rare65")

    model_dir = dest / "model"
    data_dir = dest / "data"
    evaluation_dir = dest / "evaluation"
    for directory in (model_dir, data_dir, evaluation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Copy only files named by the validated bundle contract.
    for key in ("weights", "preprocessor"):
        source = source_dir / config["model"][key]
        target = dest / config["model"][key]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    optional_bundle_files = {
        "model/model.json": "model/model.json",
        "receipt.json": "final_fit_receipt.json",
        "verification.json": "runtime_verification.json",
        "policy_calibration_receipt.json": "policy_calibration_receipt.json",
    }
    for source_name, target_name in optional_bundle_files.items():
        source = source_dir / source_name
        if source.is_file():
            target = dest / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    packaged_sources: dict[str, str] = {}
    for name, relative in config["source_paths"].items():
        source = _resolve(source_dir, relative)
        suffix = "".join(source.suffixes) or ".csv"
        # Keep stable short paths in production even when research recorded a
        # repository-relative source location.
        target_relative = f"data/{name}{suffix}"
        target = dest / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        packaged_sources[name] = target_relative
    config["source_paths"] = packaged_sources
    initial = source_dir / config["initial_state"]
    # The history recipe is part of the public bundle contract, not inferred
    # later from a directory name.
    config["history_mode"] = selected_recipe
    config["policy"]["name"] = "rare65"
    write_json(dest / "bundle.json", config)
    state = read_json(initial)
    if "binding_sha256" not in state:
        selector = state.get("selector_state", {}).get("KZT", {})
        last_date = state.get("last_processed_date")
        if last_date:
            day = __import__("datetime").date.fromisoformat(last_date)
            iso = day.isocalendar()
            week = f"{iso.year}-W{iso.week:02d}"
        else:
            week = None
        state = {
            "schema_version": 1,
            "binding_sha256": binding(config),
            "last_processed_date": last_date,
            "last_processed_session": state.get("last_processed_session"),
            "last_candidate_session": selector.get("last"),
            "week": week,
            "week_candidates": int(selector.get("count", 0)),
            "past_scores": state.get("past_scores", [])[-int(config["policy"]["window"]):],
        }
    write_json(dest / config["initial_state"], state)
    shutil.copy2(selection_path, evaluation_dir / "selection.json")

    metric_files: dict[str, Path] = {}
    conventions = {
        "by_year.csv": [research / "selected_by_year.csv", research / "by_year.csv"],
        "summary.csv": [research / "selected_summary.csv", research / "summary.csv"],
        "predictions.csv.gz": [research / "selected_predictions.csv.gz", research / "predictions.csv.gz"],
    }
    key_map = {
        "by_year.csv": {"by_year", "by_year_path", "selected_by_year"},
        "summary.csv": {"summary", "summary_path", "selected_summary"},
        "predictions.csv.gz": {"predictions", "predictions_path", "selected_predictions"},
    }
    for target_name, candidates in conventions.items():
        source = _first_existing(research, selection, key_map[target_name], candidates)
        target = evaluation_dir / target_name
        shutil.copy2(source, target)
        metric_files[target_name] = target

    # Validate the package rather than trusting research metadata.
    packaged, base = load_config(dest / "bundle.json")
    checks = {
        "weights": sha(base / packaged["model"]["weights"]),
        "preprocessor": sha(base / packaged["model"]["preprocessor"]),
    }
    if checks["weights"] != packaged["model"]["weights_sha256"] or checks["preprocessor"] != packaged["model"]["preprocessor_sha256"]:
        raise RuntimeError("Packaged model hashes differ from bundle.json")
    source_receipt = [
        {"name": name, "path": relative, "sha256": sha(dest / relative)}
        for name, relative in config["source_paths"].items()
    ]
    write_json(dest / "source_receipt.json", source_receipt)
    write_json(dest / "feature_contract.json", {
        "schema_version": 1, "corridor": "KZT", "train_horizon": 5,
        "features": config["features"], "source_paths": config["source_paths"],
        "missing_values": "train-only imputation plus one missing indicator per feature",
    })
    write_json(dest / "policy.json", config["policy"])
    selection_receipt = {
        "schema_version": 1, "selection_sha256": sha(selection_path),
        "selected_recipe": selected_recipe, "selected_config_id": selected_config_id,
        "policy": "rare65", "horizon": 5,
    }
    write_json(dest / "selection_receipt.json", selection_receipt)
    receipt = {
        "schema_version": 1,
        "model": "h5",
        "status": "packaged",
        "selected_recipe": selected_recipe,
        "selected_config_id": selected_config_id,
        "selection_source": display_path(selection_path),
        "selection_sha256": sha(selection_path),
        "bundle_source": display_path(bundle_path),
        "bundle_sha256": sha(dest / "bundle.json"),
        "artifacts": {name: sha(path) for name, path in metric_files.items()},
        **{f"{name}_sha256": value for name, value in checks.items()},
    }
    write_json(dest / "training_receipt.json", receipt)
    write_json(dest / "_SUCCESS.json", {"status": "complete", "training_receipt_sha256": sha(dest / "training_receipt.json")})
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.research_dir, args.dest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
