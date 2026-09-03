#!/usr/bin/env python3
"""Derive robustness slices from privacy-safe daily and monthly aggregates."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_chat import pearson, percentile, spearman


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def to_number(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def event_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["market_abs_return_1d"] is not None and row["messages"] >= 20]
    cutoff = percentile([row["market_abs_return_1d"] for row in eligible], 0.90)
    high = [row for row in eligible if row["market_abs_return_1d"] >= cutoff]
    other = [row for row in eligible if row["market_abs_return_1d"] < cutoff]

    def means(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n_days": len(selected),
            "rate_information_share": statistics.fmean(row["rate_information_share"] for row in selected),
            "timing_forecast_share": statistics.fmean(row["timing_forecast_share"] for row in selected),
            "strict_listing_share": statistics.fmean(row["strict_listing_share"] for row in selected),
        }

    high_means = means(high)
    other_means = means(other)
    lifts = {
        field: high_means[field] / other_means[field] if other_means[field] else None
        for field in ("rate_information_share", "timing_forecast_share", "strict_listing_share")
    }

    correlation_fields = {
        "abs_return_vs_rate_information_share": ("market_abs_return_1d", "rate_information_share"),
        "abs_return_vs_timing_forecast_share": ("market_abs_return_1d", "timing_forecast_share"),
        "rate_percentile_vs_listing_share": ("market_90d_percentile", "strict_listing_share"),
    }
    correlations = {}
    for name, (x_field, y_field) in correlation_fields.items():
        pairs = [(row[x_field], row[y_field]) for row in eligible if row[x_field] is not None and row[y_field] is not None]
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        correlations[name] = {"n_days": len(pairs), "pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}

    return {
        "eligible_days": len(eligible),
        "abs_return_p90": cutoff,
        "high_volatility": high_means,
        "other": other_means,
        "ratio_high_to_other": lifts,
        "correlations": correlations,
    }


def main() -> None:
    with (DATA / "daily.csv").open(encoding="utf-8", newline="") as source:
        raw_rows = list(csv.DictReader(source))
    numeric_fields = [
        "messages",
        "market_abs_return_1d",
        "market_90d_percentile",
        "strict_listing_share",
        "rate_information_share",
        "timing_forecast_share",
    ]
    rows = []
    for raw in raw_rows:
        row: dict[str, Any] = {"date": raw["date"]}
        for field in numeric_fields:
            row[field] = to_number(raw.get(field, ""))
        rows.append(row)

    slices = {
        "full_complete_days": ("2022-06-10", "2026-09-02"),
        "post_2022": ("2023-01-01", "2026-09-02"),
        "recent_24_complete_months": ("2024-09-01", "2026-08-31"),
    }
    output: dict[str, Any] = {}
    for name, (start, end) in slices.items():
        selected = [row for row in rows if start <= row["date"] <= end]
        output[name] = {"start": start, "end": end, **event_slice(selected)}

    with (DATA / "monthly.csv").open(encoding="utf-8", newline="") as source:
        monthly = list(csv.DictReader(source))
    annual: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"messages": [], "authors": []})
    for row in monthly:
        if row["month"] in {"2022-06", "2026-09"}:
            continue
        year = row["month"][:4]
        annual[year]["messages"].append(int(row["messages"]))
        annual[year]["authors"].append(int(row["unique_authors"]))
    output["annual_monthly_distribution"] = {
        year: {
            "n_months": len(values["messages"]),
            "messages_median": statistics.median(values["messages"]),
            "messages_min": min(values["messages"]),
            "messages_max": max(values["messages"]),
            "authors_median": statistics.median(values["authors"]),
            "authors_min": min(values["authors"]),
            "authors_max": max(values["authors"]),
        }
        for year, values in sorted(annual.items())
    }
    (DATA / "robustness_slices.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
