#!/usr/bin/env python3
"""Независимые проверки согласованности выгруженных витрин."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


SEGMENTS = ("card_transfer", "cash", "crypto")
DIRECTIONS = ("rub_to_kzt", "kzt_to_rub")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dates_between(start: date, end: date) -> list[str]:
    days = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-validation", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.output / "run_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((args.output / "quality_report.json").read_text(encoding="utf-8"))

    for artifact in manifest["outputs"]:
        path = args.output / artifact["file"]
        assert path.stat().st_size == artifact["size_bytes"], artifact["file"]
        assert sha256(path) == artifact["sha256"], artifact["file"]

    confirmed = read_csv(args.output / "confirmed_rate_observations.csv")
    assert len(confirmed) == quality["extraction"]["by_quality_status"]["accepted"] == 27
    assert all(row["quality_status"] == "accepted" for row in confirmed)
    refs = [row["message_ref"] for row in confirmed]
    assert len(refs) == len(set(refs))
    assert set(row["segment"] for row in confirmed) == set(SEGMENTS)
    assert set(row["direction"] for row in confirmed) == set(DIRECTIONS)

    manual = read_csv(args.manual_validation)
    assert {row["message_ref"] for row in manual} == set(refs)
    checks = ("manual_relevant", "manual_segment_correct", "manual_direction_correct", "manual_rate_correct")
    assert all(row[field] == "1" for row in manual for field in checks)

    daily = read_csv(args.output / "daily_rates.csv")
    start = date.fromisoformat(min(item["first_message_day"] for item in quality["source_metadata"]))
    end = date.fromisoformat(max(item["last_message_day"] for item in quality["source_metadata"]))
    calendar = dates_between(start, end)
    assert len(daily) == len(calendar) * len(SEGMENTS) * len(DIRECTIONS)
    counts = Counter(row["date"] for row in daily)
    assert list(counts) == calendar
    assert set(counts.values()) == {len(SEGMENTS) * len(DIRECTIONS)}

    daily_keyed = {(row["date"], row["segment"], row["direction"]): row for row in daily}
    assert len(daily_keyed) == len(daily)
    for row in daily:
        if row["fill_method"] == "forward_fill":
            assert row["is_observed"] == "False"
            assert int(row["days_since_observed"]) > 0
        elif row["fill_method"] == "observed":
            assert row["is_observed"] == "True"
            assert row["effective_source_date"] == row["date"]
            assert row["days_since_observed"] == "0"
            assert row["effective_rate_kzt_per_rub"] == row["observed_rate_kzt_per_rub"]
        else:
            assert row["fill_method"] == "no_prior_observation"
            assert row["effective_rate_kzt_per_rub"] == ""

    for segment in SEGMENTS:
        rows = read_csv(args.output / f"daily_{segment}_rates.csv")
        assert rows == [row for row in daily if row["segment"] == segment]

    wide = read_csv(args.output / "daily_rates_wide.csv")
    assert [row["date"] for row in wide] == calendar
    for row in wide:
        for segment in SEGMENTS:
            for direction in DIRECTIONS:
                long_row = daily_keyed[(row["date"], segment, direction)]
                prefix = f"{segment}_{direction}"
                assert row[f"{prefix}_rate"] == long_row["effective_rate_kzt_per_rub"]
                assert row[f"{prefix}_is_observed"] == long_row["is_observed"]
                assert row[f"{prefix}_age_days"] == long_row["days_since_observed"]

    observed_cells = defaultdict(int)
    for row in daily:
        observed_cells[(row["segment"], row["direction"])] += row["is_observed"] == "True"
    for (segment, direction), count in observed_cells.items():
        assert count == quality["daily_coverage"][f"{segment}:{direction}"]["observed_days"]

    print(json.dumps({
        "status": "ok",
        "manifest_files_verified": len(manifest["outputs"]),
        "confirmed_rows_verified": len(confirmed),
        "manual_rows_verified": len(manual),
        "daily_rows_verified": len(daily),
        "calendar_days_verified": len(calendar),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
