#!/usr/bin/env python3
"""Fail-closed verification of the canonical AlphaTransfer research bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[2])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=script.parent / "results-final-20260904-v2",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(
    base: Path,
    expected: dict[str, str],
    group: str,
    failures: list[str],
) -> None:
    for relative_path, expected_hash in expected.items():
        path = base / relative_path
        if not path.is_file():
            failures.append(f"{group}: missing {relative_path}")
        elif sha256(path) != expected_hash:
            failures.append(f"{group}: sha256 mismatch for {relative_path}")


def verify_prediction_clock(path: Path, failures: list[str]) -> tuple[int, set[str], set[str]]:
    row_count = 0
    available_columns: set[str] = set()
    age_columns: set[str] = set()

    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            failures.append(f"predictions: missing date column in {path.name}")
            return row_count, available_columns, age_columns

        available_columns = {
            column for column in reader.fieldnames if column.endswith("_available_date")
        }
        age_columns = {column for column in reader.fieldnames if column.endswith("_age_days")}

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            decision_date = date.fromisoformat(row["date"])
            for column in available_columns:
                value = row[column]
                if value and date.fromisoformat(value) > decision_date:
                    failures.append(
                        f"predictions: {path.name}:{row_number} {column} is after decision date"
                    )
            for column in age_columns:
                value = row[column]
                if value and float(value) < 0:
                    failures.append(
                        f"predictions: {path.name}:{row_number} {column} is negative"
                    )

    return row_count, available_columns, age_columns


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    failures: list[str] = []

    manifest_path = run_dir / "experiment_manifest.json"
    success_path = run_dir / "_SUCCESS.json"
    selection_path = run_dir / "selection.json"
    manifest = load_json(manifest_path)
    success = load_json(success_path)
    selection = load_json(selection_path)

    if success.get("status") != "complete":
        failures.append("success marker: status is not complete")
    if success.get("run_tier") != "final" or manifest.get("run_tier") != "final":
        failures.append("run tier: canonical bundle is not final")
    if success.get("experiment_manifest_sha256") != sha256(manifest_path):
        failures.append("success marker: manifest sha256 mismatch")
    if manifest.get("bootstrap_repetitions", 0) < 10_000:
        failures.append("bootstrap: fewer than 10,000 repetitions")
    if manifest.get("inferential_status") != "final_retrospective_exploratory":
        failures.append("inferential status: unexpected promotion state")
    if selection != manifest.get("selection"):
        failures.append("selection: selection.json differs from manifest")
    if selection.get("production_promoted_tracks") != []:
        failures.append("selection: production tracks must be empty")
    if selection.get("prospective_shadow_tracks") != {}:
        failures.append("selection: prospective shadow tracks must be empty")

    expected_files = set(manifest["outputs"]) | {
        "_SUCCESS.json",
        "experiment_manifest.json",
        "selection.json",
    }
    actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        failures.append(
            "output allowlist mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    verify_hashes(repo_root, manifest["code"], "code", failures)
    verify_hashes(repo_root, manifest["inputs"], "input", failures)
    verify_hashes(run_dir, manifest["outputs"], "output", failures)

    quant_dir = repo_root / "review_artifacts" / "quant_macro"
    data_manifest = load_json(quant_dir / "data_manifest.json")
    for relative_path, metadata in data_manifest["artifacts"].items():
        path = quant_dir / relative_path
        if not path.is_file():
            failures.append(f"data artifact: missing {relative_path}")
            continue
        if path.stat().st_size != metadata["bytes"]:
            failures.append(f"data artifact: byte count mismatch for {relative_path}")
        if sha256(path) != metadata["sha256"]:
            failures.append(f"data artifact: sha256 mismatch for {relative_path}")

    prediction_rows = 0
    available_columns: set[str] = set()
    age_columns: set[str] = set()
    for name in sorted(manifest["outputs"]):
        if name.endswith("predictions.csv"):
            rows, available, ages = verify_prediction_clock(run_dir / name, failures)
            prediction_rows += rows
            available_columns.update(available)
            age_columns.update(ages)

    summary = {
        "status": "pass" if not failures else "fail",
        "run": run_dir.name,
        "run_tier": manifest.get("run_tier"),
        "inferential_status": manifest.get("inferential_status"),
        "bootstrap_repetitions": manifest.get("bootstrap_repetitions"),
        "verified_code_hashes": len(manifest["code"]),
        "verified_input_hashes": len(manifest["inputs"]),
        "verified_output_hashes": len(manifest["outputs"]),
        "verified_source_families": len(data_manifest["sources"]),
        "verified_data_artifacts": len(data_manifest["artifacts"]),
        "prediction_rows_checked": prediction_rows,
        "available_date_columns_checked": len(available_columns),
        "age_columns_checked": len(age_columns),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
