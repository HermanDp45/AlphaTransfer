"""Auditable historical facts for notification copy, independent of ML scores."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any


def historical_fact(path: Path, corridor: str, as_of: date, window: int = 60) -> dict[str, Any]:
    """Read a prefix only. Effective CBR dates remain explicitly labelled as such."""
    observations = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["corridor"] == corridor and date.fromisoformat(row["date"]) <= as_of:
                rate = float(row["rub_per_unit"])
                if rate <= 0:
                    raise ValueError("Official reference rate must be positive")
                observations.append((row["date"], rate))
    observations.sort()
    if not observations:
        return {"available": False, "reason": "no_official_history_at_decision"}
    recent = observations[-window:]
    current_date, current = recent[-1]
    percentile = (sum(v < current for _, v in recent) + 0.5 * sum(v == current for _, v in recent)) / len(recent)
    previous = recent[-2][1] if len(recent) > 1 else current
    return {
        "available": True,
        "source": "Bank of Russia official reference; not an executable Alpha quote",
        "effective_date": current_date,
        "decision_date": as_of.isoformat(),
        "publication_timestamp_verified": False,
        "rub_per_unit": current,
        "window_observations": len(recent),
        "window_start": recent[0][0],
        "percentile_midrank": percentile,
        "lower_quintile": len(recent) >= window and percentile <= 0.20,
        "change_since_previous_observation_pct": 100 * (current / previous - 1),
        "previous_observation_date": recent[-2][0] if len(recent) > 1 else None,
    }


def factual_copy(fact: dict[str, Any], currency_name: str, country: str) -> dict[str, str]:
    """Forecast confidence cannot substitute for a factual predicate."""
    if not fact.get("available"):
        return {"scenario": "NO_FACT", "title": f"Перевод в {country}", "body": "Актуальный курс и сумму к получению можно проверить в приложении."}
    if fact["lower_quintile"]:
        return {
            "scenario": "HISTORICAL_LOW_REFERENCE",
            "title": f"Курс {currency_name}: историческое сравнение",
            "body": f"Курс ЦБ на {fact['effective_date']} — среди 20% самых низких за последние {fact['window_observations']} значений. Курс перевода и сумму к получению проверьте в приложении.",
        }
    delta = fact["change_since_previous_observation_pct"]
    if abs(delta) >= 0.01:
        direction = "вырос" if delta > 0 else "снизился"
        return {
            "scenario": "OBSERVED_RISE" if delta > 0 else "OBSERVED_FALL",
            "title": f"Курс {currency_name} изменился",
            "body": f"Курс ЦБ на {fact['effective_date']} {direction} на {abs(delta):.2f}% с {fact['previous_observation_date']}. Курс перевода и сумму к получению проверьте в приложении.",
        }
    return {
        "scenario": "REFERENCE_LEVEL",
        "title": f"Курс для перевода в {country}",
        "body": f"Официальный курс на {fact['effective_date']}: {fact['rub_per_unit']:.4f} ₽ за единицу валюты. Курс перевода в приложении может отличаться.",
    }
