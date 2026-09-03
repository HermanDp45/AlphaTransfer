"""Calendar representation without turning non-publication days into prices."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

from .cbr import Quote


def calendar_daily(quotes: Iterable[Quote]) -> list[dict]:
    """Forward-fill only for charts; ``is_publication_day`` protects analysis.

    CBR values take effect over weekends/holidays, but repeated values are not
    new market observations. Indicators in this project use publication rows.
    """
    grouped: dict[str, dict[date, Quote]] = {}
    for q in quotes:
        grouped.setdefault(q.corridor, {})[q.date] = q
    rows: list[dict] = []
    for corridor, by_date in grouped.items():
        if not by_date:
            continue
        current = min(by_date)
        end = max(by_date)
        last: Quote | None = None
        while current <= end:
            if current in by_date:
                last = by_date[current]
                observed = True
            else:
                observed = False
            if last:
                rows.append({"date": current, "corridor": corridor, "rub_per_unit": last.rub_per_unit, "is_publication_day": observed})
            current += timedelta(days=1)
    return sorted(rows, key=lambda r: (r["corridor"], r["date"]))
