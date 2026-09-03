"""Deterministic RU/KZ statutory calendar features.

MOEX candles remain the authority for whether a session actually occurred;
holiday flags are contextual features and never create an eligible day.
"""

from __future__ import annotations

import calendar as _calendar
from datetime import date, timedelta


def _observed(days: set[date]) -> set[date]:
    result = set(days)
    for day in sorted(days):
        if day.weekday() == 5:
            result.add(day + timedelta(days=2))
        elif day.weekday() == 6:
            result.add(day + timedelta(days=1))
    return result


def holidays(country: str, year: int) -> set[date]:
    if country == "RU":
        base = {date(year, 1, d) for d in range(1, 9)} | {
            date(year, 2, 23), date(year, 3, 8), date(year, 5, 1),
            date(year, 5, 9), date(year, 6, 12), date(year, 11, 4),
        }
    elif country == "KZ":
        base = {
            date(year, 1, 1), date(year, 1, 2), date(year, 1, 7),
            date(year, 3, 8), date(year, 3, 21), date(year, 3, 22), date(year, 3, 23),
            date(year, 5, 1), date(year, 5, 7), date(year, 5, 9), date(year, 7, 6),
            date(year, 8, 30), date(year, 10, 25), date(year, 12, 16),
        }
    else:
        raise ValueError("country must be RU or KZ")
    return _observed(base)


def _distance(day: date, set_: set[date]) -> tuple[int, int]:
    before = min(((day - h).days for h in set_ if h <= day), default=366)
    after = min(((h - day).days for h in set_ if h >= day), default=366)
    return before, after


def calendar_features(day: date) -> dict[str, int]:
    ru = holidays("RU", day.year - 1) | holidays("RU", day.year) | holidays("RU", day.year + 1)
    kz = holidays("KZ", day.year - 1) | holidays("KZ", day.year) | holidays("KZ", day.year + 1)
    ru_before, ru_after = _distance(day, ru)
    kz_before, kz_after = _distance(day, kz)
    last_day = _calendar.monthrange(day.year, day.month)[1]
    return {
        "weekday": day.weekday(), "week_of_month": (day.day - 1) // 7 + 1,
        "month": day.month, "is_month_end": int(day.day >= last_day - 2),
        "is_ru_weekend": int(day.weekday() >= 5), "is_kz_weekend": int(day.weekday() >= 5),
        "is_ru_holiday": int(day in ru), "is_kz_holiday": int(day in kz),
        "days_since_ru_holiday": min(ru_before, 30), "days_to_ru_holiday": min(ru_after, 30),
        "days_since_kz_holiday": min(kz_before, 30), "days_to_kz_holiday": min(kz_after, 30),
    }

