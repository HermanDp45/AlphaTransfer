"""Canonical, unit-labelled source observations."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    effective_date: date
    available_at: datetime
    source: str
    symbol: str
    field: str
    raw_nominal: float
    raw_value: float
    normalized_value: float
    normalized_unit: str
    is_observation: bool = True

    def __post_init__(self) -> None:
        if self.raw_nominal <= 0 or self.normalized_value <= 0:
            raise ValueError("nominal and normalized rate must be positive")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")


FIELDS = ["effective_date", "available_at", "source", "symbol", "field", "raw_nominal", "raw_value", "normalized_value", "normalized_unit", "is_observation"]


def write_observations(rows: Iterable[Observation], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.effective_date, r.source, r.symbol, r.field)):
            item = asdict(row)
            item["effective_date"] = row.effective_date.isoformat()
            item["available_at"] = row.available_at.isoformat()
            item["is_observation"] = "1" if row.is_observation else "0"
            writer.writerow(item)


def read_observations(path: str | Path) -> list[Observation]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or set(FIELDS) - set(reader.fieldnames):
            raise ValueError(f"observation schema changed: expected {FIELDS}")
        return [Observation(
            date.fromisoformat(r["effective_date"]), datetime.fromisoformat(r["available_at"]).astimezone(timezone.utc),
            r["source"], r["symbol"], r["field"], float(r["raw_nominal"]), float(r["raw_value"]),
            float(r["normalized_value"]), r["normalized_unit"], r["is_observation"] == "1",
        ) for r in reader]
