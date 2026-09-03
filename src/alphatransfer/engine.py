"""Explainable indicators and a strictly point-in-time signal API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Iterable

from .cbr import Quote


@dataclass(frozen=True)
class SignalConfig:
    """Parameters are measured in CBR *publication* days, not calendar days."""

    lookback: int = 60
    momentum_days: int = 3
    low_percentile: float = 0.20
    rebound_window: int = 20
    cooldown_observations: int = 3
    min_strength: float = 0.55

    def __post_init__(self) -> None:
        if self.lookback < 10 or self.momentum_days < 2 or self.rebound_window < 3:
            raise ValueError("lookback >= 10, momentum_days >= 2, rebound_window >= 3 are required")
        if not 0 < self.low_percentile < 0.5:
            raise ValueError("low_percentile must be in (0, .5)")
        if self.cooldown_observations < 0:
            raise ValueError("cooldown_observations cannot be negative")


def _group(quotes: Iterable[Quote], as_of: date | None = None) -> dict[str, list[Quote]]:
    out: dict[str, list[Quote]] = {}
    for quote in quotes:
        if as_of is None or quote.date <= as_of:
            out.setdefault(quote.corridor, []).append(quote)
    for corridor in out:
        out[corridor].sort(key=lambda q: q.date)
        if len({q.date for q in out[corridor]}) != len(out[corridor]):
            raise ValueError(f"Duplicate publication date in {corridor}")
    return out


def indicator_rows(quotes: Iterable[Quote], config: SignalConfig = SignalConfig(), *, as_of: date | None = None) -> list[dict]:
    """Calculate features using only prices published on or before each row.

    This function must never use a future index.  It is intentionally written
    as an incremental loop rather than vectorized shifts so that the invariant
    is straightforward to audit.
    """
    rows: list[dict] = []
    for corridor, series in _group(quotes, as_of).items():
        values = [q.rub_per_unit for q in series]
        for i, quote in enumerate(series):
            prior_and_now = values[max(0, i - config.lookback + 1) : i + 1]
            percentile = sum(x <= quote.rub_per_unit for x in prior_and_now) / len(prior_and_now)
            decreasing_run = 0
            j = i
            while j > 0 and values[j] < values[j - 1]:
                decreasing_run += 1
                j -= 1

            previous_window = values[max(0, i - config.rebound_window) : i]
            was_at_bottom = bool(previous_window) and values[i - 1] <= min(previous_window)
            rebound = i > 0 and values[i] > values[i - 1] and was_at_bottom

            # A calendar profile is learned only from complete observations in
            # earlier years. This is a weak, explainable contextual feature.
            historical_same_month = [
                values[k] for k in range(i) if series[k].date.month == quote.date.month and series[k].date.year < quote.date.year
            ]
            seasonal_below_history = bool(historical_same_month) and quote.rub_per_unit <= median(historical_same_month)
            volatility_bps = 0.0
            if len(prior_and_now) >= 2:
                changes = [abs((prior_and_now[k] / prior_and_now[k - 1] - 1) * 10_000) for k in range(1, len(prior_and_now))]
                volatility_bps = median(changes)
            rows.append(
                {
                    "date": quote.date,
                    "corridor": corridor,
                    "rub_per_unit": quote.rub_per_unit,
                    "observation_index": i,
                    "low_percentile": percentile,
                    "decreasing_run": decreasing_run,
                    "momentum_down": decreasing_run >= config.momentum_days,
                    "low_level": len(prior_and_now) >= config.lookback and percentile <= config.low_percentile,
                    "rebound_up": rebound,
                    "seasonal_below_history": seasonal_below_history,
                    "volatility_bps": volatility_bps,
                }
            )
    return sorted(rows, key=lambda r: (r["date"], r["corridor"]))


def _raw_events(features: Iterable[dict], config: SignalConfig) -> list[dict]:
    events: list[dict] = []
    for row in features:
        sources: list[str] = []
        if row["momentum_down"]:
            sources.append("momentum_down")
        if row["low_level"]:
            sources.append("low_level")
        if row["seasonal_below_history"]:
            sources.append("seasonal_low")
        # For an immediate-transfer communication, require two independent
        # present/past facts: a low level plus a downward move/context.
        if row["low_level"] and (row["momentum_down"] or row["seasonal_below_history"]):
            strength = min(1.0, 0.45 + 0.10 * len(sources) + (config.low_percentile - row["low_percentile"]) / (2 * config.low_percentile))
            events.append({
                "date": row["date"], "corridor": row["corridor"], "indicator": "favourable_now",
                "direction": "lower_rub_per_unit", "strength": round(strength, 4), "speed": "fast",
                "scenario": "favourable_now", "source_indicators": ",".join(sources),
                "rub_per_unit": row["rub_per_unit"], "low_percentile": row["low_percentile"],
                "decreasing_run": row["decreasing_run"], "volatility_bps": row["volatility_bps"],
                "observation_index": row["observation_index"],
            })
        if row["rebound_up"]:
            strength = min(1.0, 0.55 + (1.0 - row["low_percentile"]) * 0.30)
            events.append({
                "date": row["date"], "corridor": row["corridor"], "indicator": "window_closing",
                "direction": "higher_rub_per_unit", "strength": round(strength, 4), "speed": "slow",
                "scenario": "window_closing", "source_indicators": "rebound_up",
                "rub_per_unit": row["rub_per_unit"], "low_percentile": row["low_percentile"],
                "decreasing_run": row["decreasing_run"], "volatility_bps": row["volatility_bps"],
                "observation_index": row["observation_index"],
            })
    return events


def _thin(events: Iterable[dict], config: SignalConfig) -> list[dict]:
    """Keep one strongest event per corridor/direction cooldown period."""
    selected: list[dict] = []
    last_index: dict[tuple[str, str], int] = {}
    for event in sorted(events, key=lambda e: (e["corridor"], e["date"], -e["strength"])):
        key = (event["corridor"], event["direction"])
        # Publication index, rather than calendar-day distance, prevents a
        # long holiday from being treated as several independent observations.
        event_index = event["observation_index"]
        if key not in last_index:
            last_index[key] = event_index
            selected.append(event)
        elif event_index - last_index[key] > config.cooldown_observations:
            last_index[key] = event_index
            selected.append(event)
    for e in selected:
        e.pop("observation_index", None)
        if e["strength"] < config.min_strength:
            continue
    return [e for e in selected if e["strength"] >= config.min_strength]


def signals_as_of(quotes: Iterable[Quote], as_of: date | str, config: SignalConfig = SignalConfig()) -> list[dict]:
    """Return the complete trigger stream that would have existed at ``as_of``.

    Input later than ``as_of`` is ignored.  This is the audit entrypoint for
    the no-look-ahead requirement, and is also safe for daily production use.
    """
    cutoff = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
    features = indicator_rows(quotes, config, as_of=cutoff)
    events = _raw_events(features, config)
    return sorted(_thin(events, config), key=lambda e: (e["date"], e["corridor"], e["indicator"]))


def signal_at(quotes: Iterable[Quote], as_of: date | str, config: SignalConfig = SignalConfig()) -> list[dict]:
    """Signals eligible to be sent on exactly one publication date."""
    cutoff = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
    return [event for event in signals_as_of(quotes, cutoff, config) if event["date"] == cutoff]
