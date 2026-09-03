"""Point-in-time feature matrix, evaluated only on fresh MOEX sessions."""

from __future__ import annotations

import csv
from datetime import date, timedelta
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from .market_calendar import calendar_features
from .normalization import cross_rub_per_kzt
from .schema import Observation


def _latest(rows: list[Observation], day: date) -> Observation | None:
    eligible = [r for r in rows if r.effective_date <= day]
    return max(eligible, key=lambda r: (r.effective_date, r.available_at)) if eligible else None


def _return(values: list[float], i: int, n: int) -> float:
    return values[i] / values[i - n] - 1.0 if i >= n else 0.0


def _percentile(window: list[float], current: float) -> float:
    return sum(v <= current for v in window) / len(window) if window else 0.5


def _basis(a: float | None, b: float | None) -> float:
    return ((a / b) - 1.0) * 10_000 if a and b else 0.0


def build_feature_rows(observations: Iterable[Observation], cfg: dict) -> list[dict]:
    obs = list(observations)
    by_key: dict[tuple[str, str, str], list[Observation]] = {}
    for row in obs:
        by_key.setdefault((row.source, row.symbol, row.field), []).append(row)
    for rows in by_key.values():
        rows.sort(key=lambda r: r.effective_date)

    moex_closes = by_key.get(("MOEX", "KZTRUB_TOM", "close"), [])
    if not moex_closes:
        return []
    cbr_kzt = by_key.get(("CBR", "KZT", "close"), [])
    cbr_usd = by_key.get(("CBR", "USD", "close"), [])
    cbr_cny = by_key.get(("CBR", "CNY", "close"), [])
    nbk_rub = by_key.get(("NBK", "RUB", "close"), [])
    nbk_usd = by_key.get(("NBK", "USD", "close"), [])
    nbk_cny = by_key.get(("NBK", "CNY", "close"), [])
    cbr_fresh_values = [x.normalized_value for x in cbr_kzt]
    cbr_fresh_index = {x.effective_date: i for i, x in enumerate(cbr_kzt)}
    moex_values = [x.normalized_value for x in moex_closes]
    result: list[dict] = []

    for mi, close in enumerate(moex_closes):
        day = close.effective_date
        cbr = _latest(cbr_kzt, day)
        nbk = _latest(nbk_rub, day)
        if cbr is None or nbk is None:
            continue
        ci = cbr_fresh_index[cbr.effective_date]
        cv = cbr_fresh_values
        row: dict[str, object] = {
            "date": day.isoformat(), "moex_session_index": mi,
            "cbr_rate": cbr.normalized_value, "nbk_rate": nbk.normalized_value,
            "moex_open": _latest(by_key.get(("MOEX", "KZTRUB_TOM", "open"), []), day).normalized_value,
            "moex_high": _latest(by_key.get(("MOEX", "KZTRUB_TOM", "high"), []), day).normalized_value,
            "moex_low": _latest(by_key.get(("MOEX", "KZTRUB_TOM", "low"), []), day).normalized_value,
            "moex_close": close.normalized_value,
            "cbr_age_days": (day - cbr.effective_date).days, "nbk_age_days": (day - nbk.effective_date).days,
            "cbr_is_fresh": int(cbr.effective_date == day), "nbk_is_fresh": int(nbk.effective_date == day),
            "moex_is_fresh": 1,
        }
        for n in cfg["return_windows"]:
            row[f"return_{n}"] = _return(cv, ci, int(n)) if cbr.effective_date == day else 0.0
        decreasing = 0
        j = ci
        while j > 0 and cv[j] < cv[j - 1]:
            decreasing += 1
            j -= 1
        row["decreasing_run"] = decreasing if cbr.effective_date == day else 0
        for n in cfg["streaks"]:
            row[f"falling_{n}"] = int(cbr.effective_date == day and decreasing >= int(n))
        for width in cfg["level_windows"]:
            w = cv[max(0, ci - int(width) + 1):ci + 1]
            row[f"percentile_{width}"] = _percentile(w, cbr.normalized_value)
            row[f"distance_min_{width}_bps"] = (cbr.normalized_value / min(w) - 1) * 10_000
        for width in cfg["rebound_windows"]:
            w = cv[max(0, ci - int(width) + 1):ci + 1]
            row[f"rebound_{width}_bps"] = (cbr.normalized_value / min(w) - 1) * 10_000 if cbr.effective_date == day else 0.0
        returns = [_return(cv, k, 1) for k in range(max(1, ci - int(cfg["volatility_window"]) + 1), ci + 1)]
        row["volatility_20_bps"] = pstdev(returns) * 10_000 if len(returns) > 1 else 0.0
        long_returns = [_return(cv, k, 1) for k in range(max(1, ci - 60 + 1), ci + 1)]
        long_vol = pstdev(long_returns) * 10_000 if len(long_returns) > 1 else row["volatility_20_bps"]
        row["volatility_regime"] = float(row["volatility_20_bps"]) / max(float(long_vol), 1e-9)
        row["speed_3_bps"] = float(row["return_3"]) * 10_000 / 3
        op, hi, lo, cl = (float(row[x]) for x in ("moex_open", "moex_high", "moex_low", "moex_close"))
        row["candle_body_bps"] = (cl / op - 1) * 10_000
        row["candle_range_bps"] = (hi / lo - 1) * 10_000
        row["candle_close_position"] = (cl - lo) / max(hi - lo, 1e-12)
        row["gap_bps"] = (op / moex_values[mi - 1] - 1) * 10_000 if mi else 0.0
        row["cbr_nbk_basis_bps"] = _basis(cbr.normalized_value, nbk.normalized_value)
        row["cbr_moex_basis_bps"] = _basis(cbr.normalized_value, cl)
        row["nbk_moex_basis_bps"] = _basis(nbk.normalized_value, cl)
        cu, nu = _latest(cbr_usd, day), _latest(nbk_usd, day)
        cc, nc = _latest(cbr_cny, day), _latest(nbk_cny, day)
        usd_cross = cross_rub_per_kzt(cu.normalized_value, nu.normalized_value) if cu and nu else None
        cny_cross = cross_rub_per_kzt(cc.normalized_value, nc.normalized_value) if cc and nc else None
        row["usd_cross_rate"] = usd_cross or 0.0
        row["cny_cross_rate"] = cny_cross or 0.0
        row["usd_cross_basis_bps"] = _basis(usd_cross, cbr.normalized_value)
        row["cny_cross_basis_bps"] = _basis(cny_cross, cbr.normalized_value)
        row.update(calendar_features(day))
        result.append(row)
    return result


def build_calendar_rows(observations: Iterable[Observation]) -> list[dict]:
    """Preserve every calendar date; filled values remain explicitly stale."""
    obs = list(observations)
    if not obs:
        return []
    cbr = sorted((r for r in obs if (r.source, r.symbol, r.field) == ("CBR", "KZT", "close")), key=lambda r: r.effective_date)
    nbk = sorted((r for r in obs if (r.source, r.symbol, r.field) == ("NBK", "RUB", "close")), key=lambda r: r.effective_date)
    moex = {r.effective_date for r in obs if (r.source, r.symbol, r.field) == ("MOEX", "KZTRUB_TOM", "close")}
    start, end = min(r.effective_date for r in obs), max(r.effective_date for r in obs)
    rows, day = [], start
    while day <= end:
        cr, nr = _latest(cbr, day), _latest(nbk, day)
        row: dict[str, object] = {
            "date": day.isoformat(), "moex_is_fresh": int(day in moex),
            "cbr_rate": cr.normalized_value if cr else "", "cbr_age_days": (day - cr.effective_date).days if cr else "",
            "cbr_is_fresh": int(bool(cr and cr.effective_date == day)),
            "nbk_rate": nr.normalized_value if nr else "", "nbk_age_days": (day - nr.effective_date).days if nr else "",
            "nbk_is_fresh": int(bool(nr and nr.effective_date == day)),
        }
        row.update(calendar_features(day)); rows.append(row); day += timedelta(days=1)
    return rows


def feature_names(rows: list[dict]) -> list[str]:
    excluded = {"date", "moex_session_index", "cbr_rate", "nbk_rate", "moex_open", "moex_high", "moex_low", "moex_close"}
    return [k for k in rows[0] if k not in excluded] if rows else []


def write_features(rows: list[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no feature rows")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_features(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict] = []
        for raw in reader:
            row: dict = {"date": raw["date"]}
            for key, value in raw.items():
                if key != "date":
                    row[key] = float(value)
            rows.append(row)
        return rows


def write_calendar(rows: list[dict], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_calendar(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_minimum_label(rows: list[dict], index: int, horizon: int, rate_key: str = "cbr_rate") -> int:
    if index < horizon or index + horizon >= len(rows) or not rows[index]["cbr_is_fresh"]:
        return 0
    current = float(rows[index][rate_key])
    window = [float(r[rate_key]) for r in rows[index - horizon:index + horizon + 1]]
    return int(current == min(window) and window.count(current) == 1)


def window_closing_label(rows: list[dict], index: int, lookback: int, rebound_bps: float) -> int:
    if index <= 0:
        return 0
    start = max(0, index - lookback)
    prior = [float(r["cbr_rate"]) for r in rows[start:index]]
    return int(bool(prior) and (float(rows[index]["cbr_rate"]) / min(prior) - 1) * 10_000 >= rebound_bps)
