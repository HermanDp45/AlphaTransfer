#!/usr/bin/env python3
"""Consistency checks for privacy-safe aggregate outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = ROOT.parent / "result.json"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def main() -> None:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    p2p = json.loads((DATA / "p2p_validation.json").read_text(encoding="utf-8"))
    monthly = read_csv("monthly.csv")
    daily = read_csv("daily.csv")
    categories = read_csv("categories.csv")
    directions = read_csv("directions.csv")
    author_bins = read_csv("author_bins.csv")

    source = summary["source"]
    assert source["events"] == 480_746
    assert source["messages"] == 480_713
    assert source["text_messages"] == 473_949
    assert source["authors"] == 36_236
    assert source["duplicate_ids"] == 0
    assert source["out_of_order_timestamps"] == 0
    assert source["missing_reply_targets"] == 12_858
    assert sum(int(row["messages"]) for row in monthly) == source["messages"]
    assert sum(int(row["messages"]) for row in daily) == source["messages"]
    assert sum(int(row["authors"]) for row in author_bins) == source["authors"]
    assert sum(int(row["messages"]) for row in author_bins) == source["messages"]
    assert sum(int(row["messages"]) for row in directions) == summary["marketplace"]["strict_listings"]

    category_summary = {row["category"]: row["messages"] for row in summary["categories"]}
    category_csv = {row["category"]: int(row["messages"]) for row in categories}
    assert category_summary == category_csv

    sha256 = hashlib.sha256()
    with SOURCE.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1 << 20), b""):
            sha256.update(chunk)
    assert sha256.hexdigest() == source["sha256"]
    assert monthly[0]["month"] == "2022-06"
    assert monthly[-1]["month"] == "2026-09"
    assert daily[0]["date"] == "2022-06-09"
    assert daily[-1]["date"] == "2026-09-03"
    assert p2p["primary_universe"]["messages"] == 123
    assert p2p["primary_universe"]["unique_authors"] == 100
    assert p2p["manual_validation"]["relevant_p2p"] + p2p["manual_validation"]["not_p2p"] == 100
    assert p2p["manual_validation"]["precision"] == 0.89

    print("OK: aggregate counts, distributions, dates, and source hash are consistent")


if __name__ == "__main__":
    main()
