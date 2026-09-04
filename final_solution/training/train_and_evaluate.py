#!/usr/bin/env python3
"""Point-in-time exogenous feature audit for the AlphaTransfer NOW target.

Protocol in brief:

* 2023--2025 are development out-of-time folds;
* h=5 CBR effective-date rows is the primary exploratory horizon; a frozen
  compact model subset is reported on h=1/3/10/20 as sensitivity;
* the already-inspected partial 2026 fold is diagnostic only;
* source-family selection is based on proper scoring, not on the most
  flattering signal-policy metric;
* every external observation is joined by an explicit availability proxy.

The result remains exploratory because the CBR snapshot has no true
``published_at`` and several sources do not expose immutable vintages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from . import core_experiment as core_experiment
except ImportError:  # Direct script execution: python final_solution/training/...
    import core_experiment


SEED = 20_260_903
PRIMARY_HORIZON = 5
DEVELOPMENT_YEARS = (2023, 2024, 2025)
DIAGNOSTIC_YEAR = 2026
HORIZONS = (1, 3, 5, 10, 20)
MIN_RELATIVE_BRIER_IMPROVEMENT = 0.005
MIN_IMPROVED_YEARS = 3
MIN_IMPROVED_CELLS = 8
MIN_POINT_LIFT = 1.3
PORTFOLIO_CADENCE_RANGE = (0.8, 1.2)
CANDIDATE_CADENCE_RANGE = (1.0, 2.0)
MIN_WEEKLY_CADENCE_FULFILLMENT = 0.90
MAX_TOP_FOLD_SIGNAL_SHARE = 0.50
MIN_FINAL_BOOTSTRAP_REPETITIONS = 10_000
# Compatibility alias for code and historical artifacts that used one cadence.
PRIMARY_CADENCE_RANGE = PORTFOLIO_CADENCE_RANGE
ADAPTIVE_SCORE_WINDOW = 126
ADAPTIVE_MIN_HISTORY = 40
ADAPTIVE_COOLDOWN_SESSIONS = 3
ADAPTIVE_QUANTILE_GRID = np.linspace(0.50, 0.995, 100)


@dataclass(frozen=True)
class AvailabilityProfile:
    name: str
    moex_days: int
    fed_h10_days: int
    treasury_days: int
    eia_days: int
    local_fx_days: int
    cbr_rate_days: int
    nbk_rate_days: int


PROFILES = {
    "optimistic": AvailabilityProfile("optimistic", 1, 2, 1, 5, 0, 0, 0),
    "primary": AvailabilityProfile("primary", 1, 8, 1, 10, 1, 1, 1),
    "stale": AvailabilityProfile("stale", 3, 12, 3, 14, 3, 2, 2),
    # Deliberately illegal positive control: if this dominates, timing matters.
    "illegal_lead5": AvailabilityProfile("illegal_lead5", -5, -5, -5, -5, -5, -5, -5),
}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=script.parents[2])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script.parents[1] / "data" / "normalized",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script.parents[1] / "work" / "model_final_10000_self_contained",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--run-tier", choices=("smoke", "final"), default="final")
    return parser.parse_args()


def load_core(repo_root: Path):
    """Return the colocated core module; keep the argument for API compatibility."""
    del repo_root
    return core_experiment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_log_return(values: pd.Series, lag: int) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").where(lambda series: series > 0)
    return np.log(clean).diff(lag)


def market_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy().sort_values("TRADEDATE")
    result["source_date"] = pd.to_datetime(result["TRADEDATE"])
    close_column = "CLOSE_RUB_PER_UNIT" if "CLOSE_RUB_PER_UNIT" in result else "CLOSE"
    close = pd.to_numeric(result[close_column], errors="coerce").where(lambda values: values > 0)
    result = result.loc[close.notna()].copy()
    close = close.loc[result.index]
    source_dates = result["source_date"]
    maximum_spans = {1: 4, 5: 14, 20: 45}
    for lag in (1, 5, 20):
        elapsed_days = (source_dates - source_dates.shift(lag)).dt.days
        result[f"{prefix}_ret{lag}"] = finite_log_return(close, lag).where(
            elapsed_days <= maximum_spans[lag]
        )
    result[f"{prefix}_vol20"] = result[f"{prefix}_ret1"].rolling(20, min_periods=12).std()
    result[f"{prefix}_log_level"] = np.log(close)
    if {"HIGH", "LOW"}.issubset(result.columns):
        high_column = "HIGH_RUB_PER_UNIT" if "HIGH_RUB_PER_UNIT" in result else "HIGH"
        low_column = "LOW_RUB_PER_UNIT" if "LOW_RUB_PER_UNIT" in result else "LOW"
        high = pd.to_numeric(result[high_column], errors="coerce")
        low = pd.to_numeric(result[low_column], errors="coerce")
        result[f"{prefix}_range"] = (high - low) / close
    if "NUMTRADES" in result:
        trades = pd.to_numeric(result["NUMTRADES"], errors="coerce").clip(lower=0)
        result[f"{prefix}_log_trades"] = np.log1p(trades)
        result[f"{prefix}_trades_ratio20"] = trades / trades.rolling(20, min_periods=8).median()
    feature_columns = [column for column in result if column.startswith(f"{prefix}_")]
    return result[["source_date", *feature_columns]].sort_values("source_date")


def simple_price_features(
    frame: pd.DataFrame,
    date_column: str,
    value_columns: dict[str, str],
) -> pd.DataFrame:
    frame = frame.copy().sort_values(date_column)
    result = pd.DataFrame({"source_date": pd.to_datetime(frame[date_column])})
    for raw_column, prefix in value_columns.items():
        values = pd.to_numeric(frame[raw_column], errors="coerce")
        result[f"{prefix}_level"] = values
        for lag in (1, 5, 20):
            result[f"{prefix}_ret{lag}"] = finite_log_return(values, lag)
        result[f"{prefix}_vol20"] = finite_log_return(values, 1).rolling(20, min_periods=12).std()
    return result.sort_values("source_date")


def pivot_official_fx(path: Path, source_prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    pivot = frame.pivot_table(
        index="effective_date",
        columns="symbol",
        values="normalized_value",
        aggfunc="last",
    ).sort_index()
    pivot.columns = [str(column).lower() for column in pivot.columns]
    result = pd.DataFrame({"source_date": pivot.index})
    for symbol in pivot.columns:
        values = pd.to_numeric(pivot[symbol], errors="coerce")
        prefix = f"{source_prefix}_{symbol.lower()}"
        result[f"{prefix}_level"] = values.to_numpy()
        for lag in (1, 5, 20):
            result[f"{prefix}_ret{lag}"] = finite_log_return(values, lag).to_numpy()
        result[f"{prefix}_vol20"] = finite_log_return(values, 1).rolling(20, min_periods=12).std().to_numpy()
    return result.reset_index(drop=True)


def rate_features(path: Path, date_column: str, value_column: str, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values(date_column)
    frame["source_date"] = pd.to_datetime(frame[date_column])
    rate = pd.to_numeric(frame[value_column], errors="coerce")
    result = frame[["source_date"]].copy()
    result[f"{prefix}_level"] = rate
    result[f"{prefix}_delta"] = rate.diff()
    result[f"{prefix}_delta20"] = rate.diff(20)
    return result.sort_values("source_date")


def market_rate_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = frame.copy().sort_values("TRADEDATE")
    result = pd.DataFrame({"source_date": pd.to_datetime(frame["TRADEDATE"])})
    rate = pd.to_numeric(frame["CLOSE"], errors="coerce")
    result[f"{prefix}_level"] = rate
    result[f"{prefix}_chg1"] = rate.diff()
    result[f"{prefix}_chg5"] = rate.diff(5)
    result[f"{prefix}_chg20"] = rate.diff(20)
    return result.sort_values("source_date")


def ecb_ciss_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("date")
    result = pd.DataFrame({"source_date": pd.to_datetime(frame["date"])})
    for column in ("ecb_ciss_euro_area", "ecb_ciss_us", "ecb_ciss_fx"):
        values = pd.to_numeric(frame[column], errors="coerce")
        result[f"{column}_level"] = values
        result[f"{column}_chg1"] = values.diff()
        result[f"{column}_chg5"] = values.diff(5)
        result[f"{column}_mean20"] = values.rolling(20, min_periods=12).mean()
    return result.sort_values("source_date")


def world_bank_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_start", "period_end", "available_date_proxy"],
    ).sort_values("period_start")
    price_columns = [column for column in frame if column.startswith("wb_")]
    returns: dict[str, pd.Series] = {}
    for column in price_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        for lag in (3, 12):
            returns[f"{column}_ret{lag}m"] = finite_log_return(values, lag)
    result = pd.DataFrame(returns)
    result["source_date"] = frame["period_end"]
    result["available_date"] = frame["available_date_proxy"]
    result["wb_ru_export_basket_ret3m"] = result[
        ["wb_brent_ret3m", "wb_europe_gas_ret3m", "wb_nickel_ret3m", "wb_gold_ret3m"]
    ].mean(axis=1)
    result["wb_kz_export_basket_ret3m"] = result[
        ["wb_brent_ret3m", "wb_copper_ret3m", "wb_zinc_ret3m", "wb_gold_ret3m"]
    ].mean(axis=1)
    keep = [
        "source_date",
        "available_date",
        "wb_ru_export_basket_ret3m",
        "wb_kz_export_basket_ret3m",
        "wb_brent_ret12m",
        "wb_europe_gas_ret3m",
        "wb_wheat_ret3m",
        "wb_urea_ret3m",
        "wb_aluminum_ret3m",
        "wb_copper_ret3m",
        "wb_gold_ret3m",
    ]
    return result[keep].sort_values("available_date")


def kz_cpi_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
            "kz_cpi_mom_pct": pd.to_numeric(frame["kz_cpi_mom_index"], errors="coerce") - 100.0,
            "kz_cpi_yoy_pct": pd.to_numeric(frame["kz_cpi_yoy_index"], errors="coerce") - 100.0,
        }
    )
    result["kz_cpi_mom_accel1m"] = result["kz_cpi_mom_pct"].diff()
    result["kz_cpi_yoy_accel3m"] = result["kz_cpi_yoy_pct"].diff(3)
    return result.sort_values("available_date")


def nbk_reserve_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
        }
    )
    for raw_column, prefix in (
        ("kz_gross_reserves_usd_mn", "kz_gross_reserves"),
        ("kz_net_reserves_usd_mn", "kz_net_reserves"),
        ("kz_national_fund_fx_assets_usd_mn", "kz_national_fund_fx_assets"),
    ):
        values = pd.to_numeric(frame[raw_column], errors="coerce")
        result[f"{prefix}_ret3m"] = finite_log_return(values, 3)
        result[f"{prefix}_ret12m"] = finite_log_return(values, 12)
    return result.sort_values("available_date")


def nbk_current_account_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    values = pd.to_numeric(frame["kz_current_account_usd_mn"], errors="coerce")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
            "kz_current_account_usd_mn": values,
            "kz_current_account_yoy_delta_usd_mn": values.diff(4),
        }
    )
    return result.sort_values("available_date")


def cbr_reserve_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    values = pd.to_numeric(frame["ru_international_reserves_usd_bn"], errors="coerce")
    result = pd.DataFrame({"source_date": frame["date"]})
    result["ru_reserves_ret1w"] = finite_log_return(values, 1)
    result["ru_reserves_ret4w"] = finite_log_return(values, 4)
    result["ru_reserves_ret13w"] = finite_log_return(values, 13)
    result["ru_reserves_vol13w"] = result["ru_reserves_ret1w"].rolling(13, min_periods=8).std()
    return result.sort_values("source_date")


def cbr_business_climate_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
        }
    )
    for column in (
        "ru_business_climate",
        "ru_business_climate_current",
        "ru_business_climate_expectations",
        "ru_business_price_expectations",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        result[f"{column}_level"] = values
        result[f"{column}_chg1m"] = values.diff()
        result[f"{column}_chg3m"] = values.diff(3)
    return result.sort_values("available_date")


def cbr_current_account_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    values = pd.to_numeric(frame["ru_current_account_usd_mn"], errors="coerce")
    return pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
            "ru_current_account_usd_mn": values,
            "ru_current_account_yoy_delta_usd_mn": values.diff(4),
        }
    ).sort_values("available_date")


def nbk_business_activity_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    economy = pd.to_numeric(frame["kz_bai_economy"], errors="coerce")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
            "kz_bai_economy_gap50": economy - 50.0,
            "kz_bai_economy_chg1m": economy.diff(),
            "kz_bai_economy_chg3m": economy.diff(3),
            "kz_bai_sector_share_above_50": pd.to_numeric(
                frame["kz_bai_sector_share_above_50"], errors="coerce"
            ),
            "kz_bai_sector_dispersion": pd.to_numeric(
                frame["kz_bai_sector_dispersion"], errors="coerce"
            ),
        }
    )
    return result.sort_values("available_date")


def nbk_inflation_expectation_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        parse_dates=["period_end", "available_date_proxy"],
    ).sort_values("period_end")
    result = pd.DataFrame(
        {
            "source_date": frame["period_end"],
            "available_date": frame["available_date_proxy"],
        }
    )
    for column in (
        "kz_perceived_inflation_12m_pct",
        "kz_expected_inflation_12m_pct",
        "kz_expected_inflation_5y_pct",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        result[f"{column}_level"] = values
        result[f"{column}_chg1m"] = values.diff()
        result[f"{column}_chg3m"] = values.diff(3)
    return result.sort_values("available_date")


def add_availability(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    result = frame.copy()
    result["available_date"] = result["source_date"] + pd.Timedelta(days=days)
    return result


def asof_join(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    source_id: str,
    max_age_days: int | None,
) -> pd.DataFrame:
    right = features.copy().sort_values("available_date")
    right = right.drop_duplicates("available_date", keep="last")
    observation_column = f"{source_id}_observation_date"
    availability_column = f"{source_id}_available_date"
    right = right.rename(
        columns={"source_date": observation_column, "available_date": availability_column}
    )
    left = panel.sort_values("date")
    result = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=availability_column,
        direction="backward",
        allow_exact_matches=True,
    )
    age_column = f"{source_id}_age_days"
    if (result[availability_column].dropna() > result.loc[result[availability_column].notna(), "date"]).any():
        raise ValueError(f"{source_id}: future observation crossed the decision boundary")
    result[age_column] = (result["date"] - result[observation_column]).dt.days
    feature_columns = [
        column
        for column in right.columns
        if column not in {observation_column, availability_column}
    ]
    if max_age_days is not None:
        stale = result[age_column] > max_age_days
        result.loc[stale, feature_columns] = np.nan
    result[age_column] = result[age_column].clip(lower=0)
    return result.sort_values(["date", "corridor"]).copy()


def build_feature_panel(repo_root: Path, data_dir: Path, core: Any, profile: AvailabilityProfile) -> pd.DataFrame:
    panel = core.build_panel(repo_root / "final_solution" / "data" / "cbr_daily.csv")
    sources: list[tuple[str, pd.DataFrame, int | None]] = []

    for file_name, prefix, max_age in (
        ("moex_cnyrub_tom.csv", "moex_cnyrub", 7),
        ("moex_kztrub_tom.csv", "moex_kztrub", 14),
        ("moex_amdrub_tom.csv", "moex_amdrub", 7),
        ("moex_kgsrub_tom.csv", "moex_kgsrub", 14),
        ("moex_uzsrub_tom.csv", "moex_uzsrub", 21),
        ("moex_tjsrub_tom.csv", "moex_tjsrub", 45),
        ("moex_imoex.csv", "moex_imoex", 7),
        ("moex_rgbitr.csv", "moex_rgbitr", 7),
        ("moex_rvi.csv", "moex_rvi", 7),
        ("moex_cny_fixing.csv", "moex_cny_fixing", 7),
    ):
        raw = pd.read_csv(data_dir / file_name)
        sources.append((prefix, add_availability(market_features(raw, prefix), profile.moex_days), max_age))

    for file_name, prefix in (
        ("moex_rusfar.csv", "moex_rusfar"),
        ("moex_rusfar_cny.csv", "moex_rusfar_cny"),
    ):
        raw = pd.read_csv(data_dir / file_name)
        sources.append((prefix, add_availability(market_rate_features(raw, prefix), profile.moex_days), 7))

    fed_raw = pd.read_csv(data_dir / "fed_h10_daily_indexes.csv")
    fed = simple_price_features(
        fed_raw,
        "date",
        {
            "H10/H10/JRXWTFB_N.B": "fed_broad_usd",
            "H10/H10/JRXWTFO_N.B": "fed_eme_usd",
        },
    )
    sources.append(("fed_h10", add_availability(fed, profile.fed_h10_days), 21))

    ecb = ecb_ciss_features(data_dir / "ecb_ciss_daily.csv")
    sources.append(("ecb_ciss", add_availability(ecb, 1), 10))

    treasury_raw = pd.read_csv(data_dir / "us_treasury_curve.csv")
    treasury = pd.DataFrame({"source_date": pd.to_datetime(treasury_raw["date"])})
    for column in ("us_2y", "us_10y", "us_30y"):
        values = pd.to_numeric(treasury_raw[column], errors="coerce")
        treasury[f"{column}_level"] = values
        treasury[f"{column}_chg5"] = values.diff(5)
        treasury[f"{column}_chg20"] = values.diff(20)
    treasury["us_curve_2s10s"] = treasury["us_10y_level"] - treasury["us_2y_level"]
    treasury["us_curve_10s30s"] = treasury["us_30y_level"] - treasury["us_10y_level"]
    sources.append(("us_treasury", add_availability(treasury, profile.treasury_days), 14))

    eia_raw = pd.read_csv(data_dir / "eia_brent_daily.csv")
    eia = simple_price_features(eia_raw, "date", {"brent_usd_per_barrel": "eia_brent"})
    sources.append(("eia", add_availability(eia, profile.eia_days), 21))

    cbr_fx = pivot_official_fx(data_dir / "project_cbr_fx_snapshot.csv", "cbr")
    nbk_fx = pivot_official_fx(data_dir / "project_nbk_fx_snapshot.csv", "nbk")
    sources.append(("cbr_fx", add_availability(cbr_fx, profile.local_fx_days), 10))
    sources.append(("nbk_fx", add_availability(nbk_fx, profile.local_fx_days), 10))

    cbr_rate = rate_features(data_dir / "cbr_key_rate.csv", "date", "cbr_key_rate", "cbr_key")
    nbk_rate = rate_features(
        data_dir / "nbk_base_rate_events.csv",
        "effective_date",
        "nbk_base_rate",
        "nbk_base",
    )
    sources.append(("cbr_rate", add_availability(cbr_rate, profile.cbr_rate_days), 14))
    sources.append(("nbk_rate", add_availability(nbk_rate, profile.nbk_rate_days), 400))

    sources.append(("world_bank", world_bank_features(data_dir / "world_bank_commodities_monthly.csv"), 75))
    sources.append(("kz_cpi", kz_cpi_features(data_dir / "kz_cpi_monthly.csv"), 75))
    sources.append(("nbk_reserves", nbk_reserve_features(data_dir / "nbk_reserves.csv"), 120))
    sources.append(
        (
            "nbk_current_account",
            nbk_current_account_features(data_dir / "nbk_current_account.csv"),
            220,
        )
    )
    cbr_reserves = cbr_reserve_features(data_dir / "cbr_international_reserves_weekly.csv")
    sources.append(("cbr_reserves", add_availability(cbr_reserves, 7), 28))
    sources.append(
        (
            "cbr_business_climate",
            cbr_business_climate_features(data_dir / "cbr_business_climate_monthly.csv"),
            75,
        )
    )
    sources.append(
        (
            "cbr_current_account",
            cbr_current_account_features(data_dir / "cbr_current_account_quarterly.csv"),
            220,
        )
    )
    sources.append(
        (
            "nbk_business_activity",
            nbk_business_activity_features(data_dir / "nbk_business_activity_monthly.csv"),
            75,
        )
    )
    sources.append(
        (
            "nbk_inflation_expectations",
            nbk_inflation_expectation_features(
                data_dir / "nbk_inflation_expectations_monthly.csv"
            ),
            90,
        )
    )

    for source_id, features, max_age in sources:
        panel = asof_join(panel, features, source_id, max_age)

    panel = panel.copy()
    # The scraped NBK decision history failed completeness QA, so policy-rate
    # features are retained only in the raw source inventory, not in models.
    same_fixing_session = panel["moex_cnyrub_observation_date"].eq(
        panel["moex_cny_fixing_observation_date"]
    )
    panel["moex_cny_close_minus_fixing_same_session"] = (
        panel["moex_cnyrub_log_level"] - panel["moex_cny_fixing_log_level"]
    ).where(same_fixing_session)
    panel["moex_rusfar_cny_minus_rub"] = (
        panel["moex_rusfar_cny_level"] - panel["moex_rusfar_level"]
    )
    is_kzt = panel["corridor"].eq("KZT").astype(float)
    for column in ("nbk_usd_ret1", "nbk_usd_ret5", "nbk_rub_ret5", "moex_kztrub_ret5"):
        panel[f"{column}_x_kzt"] = panel[column] * is_kzt
    for column in (
        "kz_cpi_mom_pct",
        "kz_cpi_yoy_pct",
        "kz_cpi_mom_accel1m",
        "kz_cpi_yoy_accel3m",
        "kz_gross_reserves_ret3m",
        "kz_national_fund_fx_assets_ret3m",
        "kz_current_account_yoy_delta_usd_mn",
        "kz_bai_economy_gap50",
        "kz_bai_economy_chg1m",
        "kz_bai_sector_share_above_50",
        "kz_expected_inflation_12m_pct_level",
        "kz_expected_inflation_12m_pct_chg1m",
    ):
        panel[f"{column}_x_kzt"] = panel[column] * is_kzt
    direct_prefix = {
        "AMD": "moex_amdrub",
        "KGS": "moex_kgsrub",
        "KZT": "moex_kztrub",
        "UZS": "moex_uzsrub",
    }
    for suffix in (
        "ret1",
        "ret5",
        "ret20",
        "vol20",
        "log_level",
        "log_trades",
        "trades_ratio20",
        "range",
    ):
        panel[f"direct_corridor_market_{suffix}"] = np.nan
        for corridor, prefix in direct_prefix.items():
            mask = panel["corridor"].eq(corridor)
            panel.loc[mask, f"direct_corridor_market_{suffix}"] = panel.loc[
                mask, f"{prefix}_{suffix}"
            ]
    panel["direct_corridor_market_age_days"] = np.nan
    for corridor, prefix in direct_prefix.items():
        mask = panel["corridor"].eq(corridor)
        panel.loc[mask, "direct_corridor_market_age_days"] = panel.loc[
            mask, f"{prefix}_age_days"
        ]
    panel["direct_corridor_market_minus_official_log_basis"] = (
        panel["direct_corridor_market_log_level"] - np.log(panel["rub_per_unit"])
    )
    direct_basis = panel["direct_corridor_market_minus_official_log_basis"].dropna()
    if not direct_basis.empty and direct_basis.abs().median() > 1.0:
        raise ValueError("direct MOEX quote scale is inconsistent with RUB per currency unit")
    return panel.sort_values(["date", "corridor"]).reset_index(drop=True)


def feature_groups(core: Any) -> dict[str, list[str]]:
    base = core.CORE_FEATURES + core.VOL_FEATURES
    rub_official = [
        "cbr_usd_ret5",
        "cbr_usd_ret20",
        "cbr_cny_ret5",
        "cbr_cny_ret20",
    ]
    cnyrub_ret1 = ["moex_cnyrub_ret1"]
    cnyrub_spot = [
        "moex_cnyrub_ret1",
        "moex_cnyrub_ret5",
        "moex_cnyrub_ret20",
        "moex_cnyrub_vol20",
        "moex_cnyrub_range",
        "moex_cnyrub_log_trades",
        "moex_cnyrub_trades_ratio20",
    ]
    cnyrub_fixing = [
        "moex_cny_fixing_ret1",
        "moex_cny_fixing_ret5",
        "moex_cny_close_minus_fixing_same_session",
    ]
    cnyrub_basis = ["moex_cny_close_minus_fixing_same_session"]
    rub_risk_market = [
        "moex_imoex_ret5",
        "moex_imoex_ret20",
        "moex_rgbitr_ret5",
        "moex_rgbitr_ret20",
        "moex_rvi_log_level",
        "moex_rvi_ret5",
    ]
    rub_funding_market = [
        "moex_rusfar_level",
        "moex_rusfar_chg5",
        "moex_rusfar_cny_level",
        "moex_rusfar_cny_minus_rub",
    ]
    rub_market = cnyrub_spot + cnyrub_fixing + rub_risk_market + rub_funding_market
    global_market = [
        "eia_brent_ret1",
        "eia_brent_ret5",
        "eia_brent_ret20",
        "eia_brent_vol20",
        "fed_broad_usd_ret5",
        "fed_broad_usd_ret20",
        "fed_eme_usd_ret5",
        "fed_eme_usd_ret20",
        "us_2y_chg5",
        "us_10y_chg5",
        "us_curve_2s10s",
        "ecb_ciss_euro_area_level",
        "ecb_ciss_euro_area_chg5",
        "ecb_ciss_us_level",
        "ecb_ciss_us_chg5",
        "ecb_ciss_fx_level",
        "ecb_ciss_fx_chg5",
    ]
    kazakhstan_official = [
        "nbk_usd_ret1",
        "nbk_usd_ret5",
        "nbk_usd_ret20",
        "nbk_rub_ret1",
        "nbk_rub_ret5",
        "nbk_rub_ret20",
        "nbk_usd_ret1_x_kzt",
        "nbk_usd_ret5_x_kzt",
        "nbk_rub_ret5_x_kzt",
    ]
    kazakhstan_market = [
        "moex_kztrub_ret1",
        "moex_kztrub_ret5",
        "moex_kztrub_ret20",
        "moex_kztrub_vol20",
        "moex_kztrub_log_trades",
        "moex_kztrub_trades_ratio20",
        "moex_kztrub_ret5_x_kzt",
    ]
    direct_corridor_market = [
        "direct_corridor_market_ret1",
        "direct_corridor_market_ret5",
        "direct_corridor_market_ret20",
        "direct_corridor_market_vol20",
        "direct_corridor_market_minus_official_log_basis",
        "direct_corridor_market_log_trades",
        "direct_corridor_market_trades_ratio20",
        "direct_corridor_market_range",
        "direct_corridor_market_age_days",
    ]
    direct_corridor_basis = ["direct_corridor_market_minus_official_log_basis"]
    slow_commodities = [
        "wb_ru_export_basket_ret3m",
        "wb_kz_export_basket_ret3m",
        "wb_brent_ret12m",
        "wb_europe_gas_ret3m",
        "wb_wheat_ret3m",
        "wb_urea_ret3m",
        "wb_aluminum_ret3m",
        "wb_copper_ret3m",
        "wb_gold_ret3m",
    ]
    kz_inflation = [
        "kz_cpi_mom_pct",
        "kz_cpi_yoy_pct",
        "kz_cpi_mom_accel1m",
        "kz_cpi_yoy_accel3m",
        "kz_cpi_mom_pct_x_kzt",
        "kz_cpi_yoy_pct_x_kzt",
        "kz_cpi_mom_accel1m_x_kzt",
        "kz_cpi_yoy_accel3m_x_kzt",
    ]
    kz_external_balance = [
        "kz_gross_reserves_ret3m",
        "kz_gross_reserves_ret12m",
        "kz_net_reserves_ret3m",
        "kz_net_reserves_ret12m",
        "kz_national_fund_fx_assets_ret3m",
        "kz_national_fund_fx_assets_ret12m",
        "kz_current_account_usd_mn",
        "kz_current_account_yoy_delta_usd_mn",
        "kz_gross_reserves_ret3m_x_kzt",
        "kz_national_fund_fx_assets_ret3m_x_kzt",
        "kz_current_account_yoy_delta_usd_mn_x_kzt",
    ]
    ru_reserves = [
        "ru_reserves_ret1w",
        "ru_reserves_ret4w",
        "ru_reserves_ret13w",
        "ru_reserves_vol13w",
    ]
    ru_external_balance = [
        "ru_current_account_usd_mn",
        "ru_current_account_yoy_delta_usd_mn",
    ]
    ru_business_climate = [
        "ru_business_climate_level",
        "ru_business_climate_chg1m",
        "ru_business_climate_chg3m",
        "ru_business_climate_current_level",
        "ru_business_climate_current_chg1m",
        "ru_business_climate_expectations_level",
        "ru_business_climate_expectations_chg1m",
        "ru_business_price_expectations_level",
        "ru_business_price_expectations_chg1m",
    ]
    kz_leading_surveys = [
        "kz_bai_economy_gap50",
        "kz_bai_economy_chg1m",
        "kz_bai_economy_chg3m",
        "kz_bai_sector_share_above_50",
        "kz_bai_sector_dispersion",
        "kz_expected_inflation_12m_pct_level",
        "kz_expected_inflation_12m_pct_chg1m",
        "kz_expected_inflation_12m_pct_chg3m",
        "kz_perceived_inflation_12m_pct_level",
        "kz_expected_inflation_5y_pct_level",
        "kz_bai_economy_gap50_x_kzt",
        "kz_bai_economy_chg1m_x_kzt",
        "kz_bai_sector_share_above_50_x_kzt",
        "kz_expected_inflation_12m_pct_level_x_kzt",
        "kz_expected_inflation_12m_pct_chg1m_x_kzt",
    ]
    public_macro_slow = (
        slow_commodities
        + kz_inflation
        + kz_external_balance
        + ru_reserves
        + ru_external_balance
        + ru_business_climate
        + kz_leading_surveys
    )
    all_public_fast = base + rub_official + global_market + kazakhstan_official
    all_contract_market = base + rub_market + kazakhstan_market + direct_corridor_market
    full_research = all_public_fast + rub_market + kazakhstan_market + direct_corridor_market
    return {
        "base": base,
        "cnyrub_basis_only": cnyrub_basis,
        "plus_rub_official": base + rub_official,
        "plus_global_market": base + global_market,
        "plus_kazakhstan_official": base + kazakhstan_official,
        "plus_cnyrub_ret1": base + cnyrub_ret1,
        "plus_cnyrub_basis": base + cnyrub_basis,
        "plus_cnyrub_basis_ret1": base + cnyrub_basis + cnyrub_ret1,
        "plus_cnyrub_spot": base + cnyrub_spot,
        "plus_cnyrub_spot_fixing": base + cnyrub_spot + cnyrub_fixing,
        "plus_rub_risk_market": base + rub_risk_market,
        "plus_rub_funding_market": base + rub_funding_market,
        "plus_kztrub_market": base + kazakhstan_market,
        "plus_direct_corridor_market": base + direct_corridor_market,
        "plus_direct_corridor_basis": base + direct_corridor_basis,
        "all_contract_market": all_contract_market,
        "plus_slow_commodities": base + slow_commodities,
        "plus_kz_inflation": base + kz_inflation,
        "plus_kz_external_balance": base + kz_external_balance,
        "plus_ru_reserves": base + ru_reserves,
        "plus_ru_external_balance": base + ru_external_balance,
        "plus_ru_business_climate": base + ru_business_climate,
        "plus_kz_leading_surveys": base + kz_leading_surveys,
        "plus_public_macro_slow": base + public_macro_slow,
        "all_public_fast": all_public_fast,
        "full_research": full_research,
    }


def experiment_specs(core: Any) -> list[Any]:
    return [
        core.Experiment("logit_base", "logistic", "base"),
        core.Experiment("logit_cnyrub_basis_only", "logistic", "cnyrub_basis_only"),
        core.Experiment("logit_plus_rub_official", "logistic", "plus_rub_official"),
        core.Experiment("logit_plus_global_market", "logistic", "plus_global_market"),
        core.Experiment("logit_plus_kazakhstan_official", "logistic", "plus_kazakhstan_official"),
        core.Experiment("logit_plus_cnyrub_ret1", "logistic", "plus_cnyrub_ret1"),
        core.Experiment("logit_plus_cnyrub_basis", "logistic", "plus_cnyrub_basis"),
        core.Experiment("logit_plus_cnyrub_spot", "logistic", "plus_cnyrub_spot"),
        core.Experiment("logit_plus_kztrub_market", "logistic", "plus_kztrub_market"),
        core.Experiment(
            "logit_plus_direct_corridor_market",
            "logistic",
            "plus_direct_corridor_market",
        ),
        core.Experiment(
            "logit_plus_direct_corridor_basis",
            "logistic",
            "plus_direct_corridor_basis",
        ),
        core.Experiment("logit_plus_slow_commodities", "logistic", "plus_slow_commodities"),
        core.Experiment("logit_plus_kz_inflation", "logistic", "plus_kz_inflation"),
        core.Experiment(
            "logit_plus_kz_external_balance",
            "logistic",
            "plus_kz_external_balance",
        ),
        core.Experiment("logit_plus_ru_reserves", "logistic", "plus_ru_reserves"),
        core.Experiment(
            "logit_plus_ru_external_balance",
            "logistic",
            "plus_ru_external_balance",
        ),
        core.Experiment(
            "logit_plus_ru_business_climate",
            "logistic",
            "plus_ru_business_climate",
        ),
        core.Experiment(
            "logit_plus_kz_leading_surveys",
            "logistic",
            "plus_kz_leading_surveys",
        ),
        core.Experiment(
            "logit_plus_public_macro_slow",
            "logistic",
            "plus_public_macro_slow",
        ),
        core.Experiment("logit_all_public_fast", "logistic", "all_public_fast"),
        core.Experiment("logit_all_contract_market", "logistic", "all_contract_market"),
        core.Experiment("hgb_base", "hist_gradient_boosting", "base"),
        core.Experiment("hgb_plus_cnyrub_ret1", "hist_gradient_boosting", "plus_cnyrub_ret1"),
        core.Experiment("hgb_plus_cnyrub_basis", "hist_gradient_boosting", "plus_cnyrub_basis"),
        core.Experiment(
            "hgb_plus_cnyrub_basis_ret1",
            "hist_gradient_boosting",
            "plus_cnyrub_basis_ret1",
        ),
        core.Experiment(
            "hgb_plus_cnyrub_spot",
            "hist_gradient_boosting",
            "plus_cnyrub_spot",
        ),
        core.Experiment(
            "hgb_plus_cnyrub_spot_fixing",
            "hist_gradient_boosting",
            "plus_cnyrub_spot_fixing",
        ),
        core.Experiment(
            "hgb_plus_rub_risk_market",
            "hist_gradient_boosting",
            "plus_rub_risk_market",
        ),
        core.Experiment(
            "hgb_plus_rub_funding_market",
            "hist_gradient_boosting",
            "plus_rub_funding_market",
        ),
        core.Experiment(
            "hgb_plus_kztrub_market",
            "hist_gradient_boosting",
            "plus_kztrub_market",
        ),
        core.Experiment(
            "hgb_plus_direct_corridor_market",
            "hist_gradient_boosting",
            "plus_direct_corridor_market",
        ),
        core.Experiment(
            "hgb_plus_direct_corridor_basis",
            "hist_gradient_boosting",
            "plus_direct_corridor_basis",
        ),
        core.Experiment("hgb_all_public_fast", "hist_gradient_boosting", "all_public_fast"),
        core.Experiment("hgb_all_contract_market", "hist_gradient_boosting", "all_contract_market"),
        core.Experiment("hgb_full_research", "hist_gradient_boosting", "full_research"),
        core.Experiment(
            "hgb_plus_public_macro_slow",
            "hist_gradient_boosting",
            "plus_public_macro_slow",
        ),
        core.Experiment(
            "hgb_cnyrub_spot_stump",
            "hist_gradient_boosting_stump",
            "plus_cnyrub_spot",
        ),
        core.Experiment(
            "hgb_cnyrub_basis_stump",
            "hist_gradient_boosting_stump",
            "plus_cnyrub_basis",
        ),
        core.Experiment(
            "hgb_contract_market_leaf100",
            "hist_gradient_boosting_leaf100",
            "all_contract_market",
        ),
    ]


def run_matrix(
    core: Any,
    panel: pd.DataFrame,
    experiments: Iterable[Any],
    horizons: Iterable[int],
    years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    oot_frames: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        targeted = core.add_target(panel, horizon)
        for experiment in experiments:
            for year in years:
                rows, oot, coefficients = core.run_fold(experiment, horizon, year, targeted)
                metric_rows.extend(rows)
                oot["model_kind"] = experiment.model_kind
                oot["feature_set"] = experiment.feature_set
                oot_frames.append(oot)
                coefficient_rows.extend(coefficients)
    metrics = pd.DataFrame(metric_rows)
    oot = pd.concat(oot_frames, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    return metrics, oot, coefficients


def probability_scores(frame: pd.DataFrame) -> dict[str, float]:
    clean = frame.dropna(subset=["probability", "target"])
    labels = clean["target"].astype(int).to_numpy()
    probabilities = np.clip(clean["probability"].to_numpy(float), 1e-8, 1 - 1e-8)
    one_class = len(np.unique(labels)) < 2
    return {
        "prediction_rows": len(clean),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": math.nan if one_class else float(roc_auc_score(labels, probabilities)),
    }


def predictive_cell_metrics(oot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "config_id",
        "model_kind",
        "feature_set",
        "horizon_cbr_rows_pub_proxy",
        "fold_test_year",
        "corridor",
    ]
    for key, group in oot.groupby(keys):
        row = dict(zip(keys, key))
        row.update(probability_scores(group))
        rows.append(row)
    return pd.DataFrame(rows)


def predictive_metrics(oot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["config_id", "model_kind", "feature_set", "horizon_cbr_rows_pub_proxy"]
    cells = predictive_cell_metrics(oot)
    for key, group in oot.groupby(keys):
        row = dict(zip(keys, key))
        row.update(probability_scores(group))
        matching_cells = cells
        for column, value in row.items():
            if column in keys:
                matching_cells = matching_cells[matching_cells[column].eq(value)]
        for metric in ("brier", "log_loss", "average_precision", "roc_auc"):
            row[f"macro_fold_corridor_{metric}"] = float(matching_cells[metric].mean())
            row[f"worst_fold_corridor_{metric}"] = (
                float(matching_cells[metric].max())
                if metric in {"brier", "log_loss"}
                else float(matching_cells[metric].min())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_diagnostics(
    oot: pd.DataFrame,
    probability_column: str = "probability",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["config_id", "horizon_cbr_rows_pub_proxy", "fold_test_year"]
    for key, pooled_group in oot.groupby(keys):
        scopes = [("__ALL__", pooled_group), *pooled_group.groupby("corridor")]
        for corridor, group in scopes:
            clean = group.dropna(subset=[probability_column, "target"]).copy()
            probabilities = np.clip(clean[probability_column].to_numpy(float), 1e-6, 1 - 1e-6)
            labels = clean["target"].astype(int).to_numpy()
            if len(np.unique(probabilities)) < 2:
                ece = float(abs(probabilities.mean() - labels.mean()))
            else:
                bins = pd.qcut(probabilities, 10, duplicates="drop")
                binned = pd.DataFrame(
                    {"probability": probabilities, "target": labels, "bin": bins}
                ).groupby("bin", observed=True).agg(
                    count=("target", "size"),
                    mean_probability=("probability", "mean"),
                    event_rate=("target", "mean"),
                )
                ece = float(
                    np.average(
                        np.abs(binned["mean_probability"] - binned["event_rate"]),
                        weights=binned["count"],
                    )
                )
            logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
            if len(np.unique(labels)) == 2:
                calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
                calibrator.fit(logits, labels)
                intercept = float(calibrator.intercept_[0])
                slope = float(calibrator.coef_[0, 0])
            else:
                intercept = math.nan
                slope = math.nan
            row = dict(zip(keys, key))
            row.update(
                {
                    "corridor": str(corridor),
                    "rows": len(clean),
                    "probability_kind": probability_column,
                    "mean_probability": float(probabilities.mean()),
                    "event_rate": float(labels.mean()),
                    "quantile_ece": ece,
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def target_dependence_diagnostics(panel: pd.DataFrame, core: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    targeted = core.add_target(panel, PRIMARY_HORIZON)
    eligible = targeted[targeted["target"].notna()]
    pivot = eligible.pivot(index="date", columns="corridor", values="target")
    correlation_matrix = pivot.corr().rename_axis(index="corridor_a", columns="corridor_b")
    correlations = correlation_matrix.stack().rename("target_correlation").reset_index()
    correlations = correlations[correlations["corridor_a"] < correlations["corridor_b"]]
    date_summary = pd.DataFrame(
        {
            "decision_dates": [len(pivot)],
            "share_all_corridors_zero": [float(pivot.eq(0).all(axis=1).mean())],
            "share_all_corridors_one": [float(pivot.eq(1).all(axis=1).mean())],
            "share_all_corridors_agree": [
                float((pivot.eq(0).all(axis=1) | pivot.eq(1).all(axis=1)).mean())
            ],
            "mean_pairwise_target_correlation": [float(correlations["target_correlation"].mean())],
        }
    )
    return correlations, date_summary


def univariate_rank_diagnostics(
    panel: pd.DataFrame,
    core: Any,
    features: Iterable[str],
) -> pd.DataFrame:
    targeted = core.add_target(panel, PRIMARY_HORIZON)
    targeted = targeted[targeted["target"].notna()].copy()
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        (
            "development",
            "2023-2025",
            targeted[targeted["date"].dt.year.isin(DEVELOPMENT_YEARS)],
        ),
        (
            "diagnostic_year",
            str(DIAGNOSTIC_YEAR),
            targeted[targeted["date"].dt.year.eq(DIAGNOSTIC_YEAR)],
        ),
    ]
    scopes.extend(
        ("year", str(year), targeted[targeted["date"].dt.year.eq(year)])
        for year in (*DEVELOPMENT_YEARS, DIAGNOSTIC_YEAR)
    )
    scopes.extend(
        ("corridor", str(corridor), group)
        for corridor, group in targeted.groupby("corridor")
    )
    scopes.extend(
        [
            (
                "market_regime",
                "before_2024-06-13",
                targeted[targeted["date"] < pd.Timestamp("2024-06-13")],
            ),
            (
                "market_regime",
                "from_2024-06-13",
                targeted[targeted["date"] >= pd.Timestamp("2024-06-13")],
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_value, scope in scopes:
        for feature in features:
            clean = scope.dropna(subset=[feature, "target"])
            if len(clean) < 30 or clean["target"].nunique() < 2 or clean[feature].nunique() < 2:
                continue
            raw_auc = float(roc_auc_score(clean["target"].astype(int), clean[feature]))
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "feature": feature,
                    "rows": len(clean),
                    "raw_auc": raw_auc,
                    "posthoc_oriented_auc": max(raw_auc, 1.0 - raw_auc),
                    "orientation": "higher" if raw_auc >= 0.5 else "lower",
                    "posthoc_diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def stratified_month_bootstrap(
    merged: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    # Five corridor rows on one date share the same market shock and are not
    # independent. Collapse them before resampling or sign flipping.
    work = merged.groupby("date", as_index=False)["loss_delta"].mean()
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.to_period("M").astype(str)
    blocks = {
        year: [group["loss_delta"].to_numpy(float) for _, group in year_frame.groupby("month")]
        for year, year_frame in work.groupby("year")
    }
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled: list[np.ndarray] = []
        for year_blocks in blocks.values():
            choices = rng.integers(0, len(year_blocks), size=len(year_blocks))
            sampled.extend(year_blocks[index] for index in choices)
        estimates[repetition] = np.concatenate(sampled).mean()
    observed = float(work["loss_delta"].mean())
    flat_blocks = [block for year_blocks in blocks.values() for block in year_blocks]
    block_sums = np.asarray([block.sum() for block in flat_blocks], dtype=float)
    null_estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(block_sums))
        null_estimates[repetition] = float(np.dot(signs, block_sums) / len(work))
    return {
        "decision_dates": float(len(work)),
        "observed_delta_brier": observed,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "one_sided_block_signflip_p_improvement": float(
            (1 + np.sum(null_estimates <= observed)) / (repetitions + 1)
        ),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def paired_feature_audit(
    oot: pd.DataFrame,
    aggregate: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    primary = oot[oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)].copy()
    rng = np.random.default_rng(SEED)
    results: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    comparison_bases = {
        experiment_id: "hgb_base"
        for experiment_id in primary.loc[
            primary["model_kind"].str.startswith("hist_gradient_boosting"), "config_id"
        ].unique()
        if experiment_id != "hgb_base"
    }
    contract_required = {
        "logit_cnyrub_basis_only",
        "logit_plus_cnyrub_ret1",
        "logit_plus_cnyrub_basis",
        "logit_plus_cnyrub_spot",
        "logit_plus_kztrub_market",
        "logit_plus_direct_corridor_market",
        "logit_plus_direct_corridor_basis",
        "logit_all_contract_market",
        "hgb_plus_cnyrub_ret1",
        "hgb_plus_cnyrub_basis",
        "hgb_plus_cnyrub_basis_ret1",
        "hgb_plus_cnyrub_spot",
        "hgb_plus_cnyrub_spot_fixing",
        "hgb_plus_rub_risk_market",
        "hgb_plus_rub_funding_market",
        "hgb_plus_kztrub_market",
        "hgb_plus_direct_corridor_market",
        "hgb_plus_direct_corridor_basis",
        "hgb_all_contract_market",
        "hgb_full_research",
        "hgb_cnyrub_spot_stump",
        "hgb_cnyrub_basis_stump",
        "hgb_contract_market_leaf100",
    }
    for config_id, candidate in primary.groupby("config_id"):
        if config_id in {"logit_base", "hgb_base"}:
            continue
        comparison_base_id = comparison_bases.get(config_id, "logit_base")
        if config_id == comparison_base_id:
            comparison_base_id = "logit_base"
        base = primary[primary["config_id"].eq(comparison_base_id)][
            ["date", "corridor", "fold_test_year", "target", "probability"]
        ].rename(columns={"probability": "base_probability"})
        base_aggregate = aggregate[
            aggregate["config_id"].eq(comparison_base_id)
            & aggregate["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)
        ].iloc[0]
        merged = candidate.merge(
            base,
            on=["date", "corridor", "fold_test_year", "target"],
            how="inner",
            validate="one_to_one",
        )
        merged["loss_delta"] = (
            merged["probability"] - merged["target"]
        ) ** 2 - (merged["base_probability"] - merged["target"]) ** 2
        bootstrap = stratified_month_bootstrap(merged, repetitions, rng)
        years_improved = 0
        cells_improved = 0
        for _, group in merged.groupby("fold_test_year"):
            years_improved += int(group["loss_delta"].mean() < 0)
        for _, group in merged.groupby(["fold_test_year", "corridor"]):
            cells_improved += int(group["loss_delta"].mean() < 0)
        candidate_aggregate = aggregate[
            aggregate["config_id"].eq(config_id)
            & aggregate["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)
        ].iloc[0]
        base_brier = float(np.mean((merged["base_probability"] - merged["target"]) ** 2))
        candidate_brier = float(np.mean((merged["probability"] - merged["target"]) ** 2))
        relative_improvement = 1.0 - candidate_brier / base_brier
        cell_deltas = merged.groupby(["fold_test_year", "corridor"])["loss_delta"].mean()
        fold_frequencies = []
        for _, fold in candidate.groupby("fold_test_year"):
            span_days = max(1, (fold["date"].max() - fold["date"].min()).days + 1)
            fold_frequencies.append(float(fold["signal"].sum() / span_days * 7.0))
        candidate_fold_corridor_frequencies = []
        for _, cell in candidate.groupby(["fold_test_year", "corridor"]):
            span_days = max(1, (cell["date"].max() - cell["date"].min()).days + 1)
            candidate_fold_corridor_frequencies.append(
                float(cell["candidate_signal"].sum() / span_days * 7.0)
            )
        row = {
            "config_id": config_id,
            "comparison_base_id": comparison_base_id,
            "model_kind": candidate["model_kind"].iloc[0],
            "feature_set": candidate["feature_set"].iloc[0],
            "comparison_rows": len(merged),
            "base_brier": base_brier,
            "candidate_brier": candidate_brier,
            "relative_brier_improvement": relative_improvement,
            "macro_fold_corridor_delta_brier": float(cell_deltas.mean()),
            "years_brier_improved": years_improved,
            "fold_corridor_cells_brier_improved": cells_improved,
            "portfolio_cell_standardized_lift": candidate_aggregate["cell_standardized_lift"],
            "base_portfolio_cell_standardized_lift": base_aggregate["cell_standardized_lift"],
            "corridor_candidate_cell_standardized_lift": candidate_aggregate[
                "candidate_cell_standardized_lift"
            ],
            "base_corridor_candidate_cell_standardized_lift": base_aggregate[
                "candidate_cell_standardized_lift"
            ],
            "portfolio_forward_bps_delta": candidate_aggregate[
                "cell_standardized_forward_bps_delta"
            ],
            "base_portfolio_forward_bps_delta": base_aggregate[
                "cell_standardized_forward_bps_delta"
            ],
            "portfolio_forward_delta_vs_base_policy": (
                candidate_aggregate["cell_standardized_forward_bps_delta"]
                - base_aggregate["cell_standardized_forward_bps_delta"]
            ),
            "corridor_candidate_forward_bps_delta": candidate_aggregate[
                "candidate_cell_standardized_forward_bps_delta"
            ],
            "base_corridor_candidate_forward_bps_delta": base_aggregate[
                "candidate_cell_standardized_forward_bps_delta"
            ],
            "corridor_candidate_forward_delta_vs_base_policy": (
                candidate_aggregate["candidate_cell_standardized_forward_bps_delta"]
                - base_aggregate["candidate_cell_standardized_forward_bps_delta"]
            ),
            "portfolio_mean_signals_per_week": candidate_aggregate[
                "mean_portfolio_signals_per_week"
            ],
            "minimum_fold_signals_per_week": min(fold_frequencies),
            "maximum_fold_signals_per_week": max(fold_frequencies),
            "all_fold_portfolio_cadence_in_range": all(
                PORTFOLIO_CADENCE_RANGE[0] <= frequency <= PORTFOLIO_CADENCE_RANGE[1]
                for frequency in fold_frequencies
            ),
            "candidate_mean_signals_per_corridor_week": float(
                np.mean(candidate_fold_corridor_frequencies)
            ),
            "minimum_fold_corridor_candidate_signals_per_week": min(
                candidate_fold_corridor_frequencies
            ),
            "maximum_fold_corridor_candidate_signals_per_week": max(
                candidate_fold_corridor_frequencies
            ),
            "all_fold_corridor_candidate_cadence_in_range": all(
                CANDIDATE_CADENCE_RANGE[0] <= frequency <= CANDIDATE_CADENCE_RANGE[1]
                for frequency in candidate_fold_corridor_frequencies
            ),
            "minimum_candidate_weeks_with_1_to_2_signals_share": candidate_aggregate[
                "minimum_candidate_weeks_with_1_to_2_signals_share"
            ],
            "mean_candidate_silent_week_share": candidate_aggregate[
                "mean_candidate_silent_week_share"
            ],
            "maximum_candidate_calendar_gap_days": candidate_aggregate[
                "maximum_candidate_calendar_gap_days"
            ],
            "candidate_top_fold_signal_share": candidate_aggregate[
                "candidate_top_fold_signal_share"
            ],
            "minimum_synthetic_portfolio_weeks_with_1_to_2_signals_share": candidate_aggregate[
                "minimum_portfolio_weeks_with_1_to_2_signals_share"
            ],
            "mean_synthetic_portfolio_silent_week_share": candidate_aggregate[
                "mean_portfolio_silent_week_share"
            ],
            "maximum_synthetic_portfolio_calendar_gap_days": candidate_aggregate[
                "maximum_portfolio_calendar_gap_days"
            ],
            "portfolio_min_fold_lift": candidate_aggregate["min_fold_lift"],
            "corridor_candidate_min_fold_lift": candidate_aggregate[
                "candidate_min_fold_lift"
            ],
            **bootstrap,
        }
        latest_vintage_or_slow = {
            "logit_plus_slow_commodities",
            "logit_plus_kz_inflation",
            "logit_plus_kz_external_balance",
            "logit_plus_ru_reserves",
            "logit_plus_ru_external_balance",
            "logit_plus_ru_business_climate",
            "logit_plus_kz_leading_surveys",
            "logit_plus_public_macro_slow",
            "hgb_plus_public_macro_slow",
        }
        row["fast_operational_candidate"] = config_id not in latest_vintage_or_slow
        row["positive_brier_skill_vs_prior_year"] = bool(
            candidate_aggregate["brier_skill_vs_prior_year"] > 0
        )
        # Snapshot hashes prove byte-level replay, not historical publication timing
        # or contractual permission for non-display/derived production use.
        row["snapshot_reproducible"] = True
        row["historical_point_in_time_verified"] = False
        row["production_data_rights_verified"] = False
        row["source_lineage_and_policy_reproducible"] = False
        row["per_signal_model_explanation_available"] = False
        row["research_source_class"] = (
            "contract_required_market_data"
            if config_id in contract_required
            else "public_source_research"
        )
        row["passes_exploratory_quality_cadence_gates"] = bool(
            row["fast_operational_candidate"]
            and row["positive_brier_skill_vs_prior_year"]
            and row["source_lineage_and_policy_reproducible"]
            and relative_improvement >= MIN_RELATIVE_BRIER_IMPROVEMENT
            and years_improved >= MIN_IMPROVED_YEARS
            and cells_improved >= MIN_IMPROVED_CELLS
            and bootstrap["ci95_high"] < 0
            and row["all_fold_corridor_candidate_cadence_in_range"]
            and row["minimum_candidate_weeks_with_1_to_2_signals_share"]
            >= MIN_WEEKLY_CADENCE_FULFILLMENT
            and row["candidate_top_fold_signal_share"] <= MAX_TOP_FOLD_SIGNAL_SHARE
            and candidate_aggregate["candidate_cell_standardized_lift"] >= MIN_POINT_LIFT
            and candidate_aggregate["candidate_min_fold_lift"] > 1.0
            and row["corridor_candidate_forward_delta_vs_base_policy"] >= 0
        )
        row["passes_exploratory_development_gates"] = bool(
            row["passes_exploratory_quality_cadence_gates"]
            and row["per_signal_model_explanation_available"]
        )
        p_values[config_id] = bootstrap["one_sided_block_signflip_p_improvement"]
        results.append(row)
    adjusted = holm_adjust(p_values)
    for row in results:
        row["holm_p_brier_improvement"] = adjusted[row["config_id"]]
        row["passes_exploratory_multiplicity_screen"] = row["holm_p_brier_improvement"] < 0.05
        row["eligible_for_policy_uncertainty_audit"] = bool(
            row["passes_exploratory_quality_cadence_gates"]
            and row["passes_exploratory_multiplicity_screen"]
        )
        row["eligible_for_prospective_shadow"] = (
            row["eligible_for_policy_uncertainty_audit"]
            and row["per_signal_model_explanation_available"]
        )
        failures: list[str] = []
        if not row["positive_brier_skill_vs_prior_year"]:
            failures.append("nonpositive_brier_skill_vs_prior_year")
        if not row["source_lineage_and_policy_reproducible"]:
            failures.append("source_lineage_or_policy_not_reproducible")
        if not row["historical_point_in_time_verified"]:
            failures.append("historical_published_at_or_vintage_not_verified")
        if not row["production_data_rights_verified"]:
            failures.append("production_data_rights_not_verified")
        if not row["per_signal_model_explanation_available"]:
            failures.append("per_signal_model_explanation_not_implemented")
        if not row["passes_exploratory_quality_cadence_gates"]:
            failures.append("one_or_more_development_quality_or_cadence_gates_failed")
        if not row["passes_exploratory_multiplicity_screen"]:
            failures.append("holm_multiplicity_screen_failed")
        row["gate_failures"] = json.dumps(failures)
        row["production_promotable"] = False
    return pd.DataFrame(results).sort_values(
        ["eligible_for_prospective_shadow", "relative_brier_improvement"],
        ascending=[False, False],
    )


def benchmark_comparison_audit(
    oot: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    """Report pre-specified benchmark and incremental-source comparisons."""
    primary = oot[oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)].copy()
    comparisons = [
        ("public_hgb_vs_hgb_base", "hgb_all_public_fast", "hgb_base"),
        ("cny_ret1_hgb_vs_hgb_base", "hgb_plus_cnyrub_ret1", "hgb_base"),
        ("cny_basis_hgb_vs_hgb_base", "hgb_plus_cnyrub_basis", "hgb_base"),
        ("cny_basis_ret1_hgb_vs_hgb_base", "hgb_plus_cnyrub_basis_ret1", "hgb_base"),
        ("cny_ret1_increment_vs_basis", "hgb_plus_cnyrub_basis_ret1", "hgb_plus_cnyrub_basis"),
        ("cny_spot_hgb_vs_hgb_base", "hgb_plus_cnyrub_spot", "hgb_base"),
        ("cny_spot_fixing_hgb_vs_hgb_base", "hgb_plus_cnyrub_spot_fixing", "hgb_base"),
        ("rub_risk_hgb_vs_hgb_base", "hgb_plus_rub_risk_market", "hgb_base"),
        ("rub_funding_hgb_vs_hgb_base", "hgb_plus_rub_funding_market", "hgb_base"),
        ("kztrub_hgb_vs_hgb_base", "hgb_plus_kztrub_market", "hgb_base"),
        (
            "direct_corridor_hgb_vs_hgb_base",
            "hgb_plus_direct_corridor_market",
            "hgb_base",
        ),
        (
            "direct_corridor_basis_hgb_vs_hgb_base",
            "hgb_plus_direct_corridor_basis",
            "hgb_base",
        ),
        ("contract_market_hgb_vs_hgb_base", "hgb_all_contract_market", "hgb_base"),
        ("full_research_hgb_vs_hgb_base", "hgb_full_research", "hgb_base"),
        ("contract_market_increment_vs_cny_basis", "hgb_all_contract_market", "hgb_plus_cnyrub_basis"),
        ("full_research_increment_vs_cny_basis", "hgb_full_research", "hgb_plus_cnyrub_basis"),
        ("fixing_increment_vs_cny_spot", "hgb_plus_cnyrub_spot_fixing", "hgb_plus_cnyrub_spot"),
        ("public_increment_vs_contract", "hgb_full_research", "hgb_all_contract_market"),
        ("contract_increment_vs_public", "hgb_full_research", "hgb_all_public_fast"),
        ("cny_stump_vs_standard", "hgb_cnyrub_spot_stump", "hgb_plus_cnyrub_spot"),
        ("cny_basis_stump_vs_standard", "hgb_cnyrub_basis_stump", "hgb_plus_cnyrub_basis"),
        ("contract_leaf100_vs_standard", "hgb_contract_market_leaf100", "hgb_all_contract_market"),
    ]
    rng = np.random.default_rng(SEED + 101)
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for comparison_id, candidate_id, base_id in comparisons:
        candidate = primary[primary["config_id"].eq(candidate_id)][
            ["date", "corridor", "fold_test_year", "target", "probability", "signal"]
        ].rename(columns={"signal": "challenger_portfolio_signal"})
        base = primary[primary["config_id"].eq(base_id)][
            ["date", "corridor", "fold_test_year", "target", "probability", "signal"]
        ].rename(
            columns={
                "probability": "base_probability",
                "signal": "base_portfolio_signal",
            }
        )
        merged = candidate.merge(
            base,
            on=["date", "corridor", "fold_test_year", "target"],
            how="inner",
            validate="one_to_one",
        )
        merged["loss_delta"] = (
            merged["probability"] - merged["target"]
        ) ** 2 - (merged["base_probability"] - merged["target"]) ** 2
        bootstrap = stratified_month_bootstrap(merged, repetitions, rng)
        base_brier = float(np.mean((merged["base_probability"] - merged["target"]) ** 2))
        candidate_brier = float(np.mean((merged["probability"] - merged["target"]) ** 2))
        challenger_portfolio_signal = merged["challenger_portfolio_signal"].astype(bool)
        base_portfolio_signal = merged["base_portfolio_signal"].astype(bool)
        signal_union = int((challenger_portfolio_signal | base_portfolio_signal).sum())
        signal_intersection = int((challenger_portfolio_signal & base_portfolio_signal).sum())
        row = {
            "comparison_id": comparison_id,
            "candidate_id": candidate_id,
            "base_id": base_id,
            "rows": len(merged),
            "candidate_brier": candidate_brier,
            "base_brier": base_brier,
            "relative_brier_improvement": 1.0 - candidate_brier / base_brier,
            "challenger_portfolio_signal_count": int(challenger_portfolio_signal.sum()),
            "base_portfolio_signal_count": int(base_portfolio_signal.sum()),
            "portfolio_signal_policy_jaccard": (
                signal_intersection / signal_union if signal_union else math.nan
            ),
            "years_improved": int(
                sum(group["loss_delta"].mean() < 0 for _, group in merged.groupby("fold_test_year"))
            ),
            "fold_corridor_cells_improved": int(
                sum(
                    group["loss_delta"].mean() < 0
                    for _, group in merged.groupby(["fold_test_year", "corridor"])
                )
            ),
            **bootstrap,
        }
        p_values[comparison_id] = bootstrap["one_sided_block_signflip_p_improvement"]
        rows.append(row)
    adjusted = holm_adjust(p_values)
    for row in rows:
        row["holm_p_brier_improvement"] = adjusted[row["comparison_id"]]
    return pd.DataFrame(rows).sort_values("relative_brier_improvement", ascending=False)


def block_length_sensitivity(
    oot: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    primary = oot[oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)]
    comparisons = [
        ("cny_basis_vs_base", "hgb_plus_cnyrub_basis", "hgb_base"),
        ("cny_ret1_vs_base", "hgb_plus_cnyrub_ret1", "hgb_base"),
        ("cny_basis_ret1_vs_basis", "hgb_plus_cnyrub_basis_ret1", "hgb_plus_cnyrub_basis"),
        ("direct_basis_vs_base", "hgb_plus_direct_corridor_basis", "hgb_base"),
        ("contract_vs_cny_basis", "hgb_all_contract_market", "hgb_plus_cnyrub_basis"),
        ("full_vs_cny_basis", "hgb_full_research", "hgb_plus_cnyrub_basis"),
    ]
    repetitions = min(repetitions, 5_000)
    rng = np.random.default_rng(SEED + 303)
    rows: list[dict[str, Any]] = []
    for comparison_id, candidate_id, base_id in comparisons:
        candidate = primary[primary["config_id"].eq(candidate_id)][
            ["date", "corridor", "fold_test_year", "target", "probability"]
        ]
        base = primary[primary["config_id"].eq(base_id)][
            ["date", "corridor", "fold_test_year", "target", "probability"]
        ].rename(columns={"probability": "base_probability"})
        paired = candidate.merge(
            base,
            on=["date", "corridor", "fold_test_year", "target"],
            how="inner",
            validate="one_to_one",
        )
        paired["loss_delta"] = (
            (paired["probability"] - paired["target"]) ** 2
            - (paired["base_probability"] - paired["target"]) ** 2
        )
        by_date = paired.groupby(["fold_test_year", "date"], as_index=False)["loss_delta"].mean()
        observed = float(by_date["loss_delta"].mean())
        for block_length in (10, 20, 40):
            blocks_by_year: dict[int, list[np.ndarray]] = {}
            for year, year_frame in by_date.groupby("fold_test_year"):
                values = year_frame.sort_values("date")["loss_delta"].to_numpy(float)
                blocks_by_year[int(year)] = [
                    values[start : start + block_length]
                    for start in range(0, len(values), block_length)
                ]
            estimates = np.empty(repetitions)
            for repetition in range(repetitions):
                samples: list[np.ndarray] = []
                for blocks in blocks_by_year.values():
                    choices = rng.integers(0, len(blocks), size=len(blocks))
                    samples.extend(blocks[index] for index in choices)
                estimates[repetition] = np.concatenate(samples).mean()
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "candidate_id": candidate_id,
                    "base_id": base_id,
                    "block_length_decision_dates": block_length,
                    "repetitions": repetitions,
                    "decision_dates": len(by_date),
                    "observed_delta_brier": observed,
                    "ci95_low": float(np.quantile(estimates, 0.025)),
                    "ci95_high": float(np.quantile(estimates, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def choose_one(audit: pd.DataFrame, source_class: str) -> tuple[str, str] | None:
    eligible = audit[
        audit["fast_operational_candidate"]
        & audit["research_source_class"].eq(source_class)
        & ~audit["config_id"].isin(["logit_base", "hgb_base"])
    ]
    if eligible.empty:
        return None
    best = eligible.sort_values("relative_brier_improvement", ascending=False).iloc[0]
    status = (
        "passes_exploratory_gates_shadow_only"
        if bool(best["eligible_for_prospective_shadow"])
        else "diagnostic_best_paired_delta_no_gate_pass"
    )
    return str(best["config_id"]), status


def choose_diagnostic_candidates(audit: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for source_class in ("public_source_research", "contract_required_market_data"):
        chosen = choose_one(audit, source_class)
        if chosen is not None:
            config_id, status = chosen
            result[source_class] = {"config_id": config_id, "status": status}
    return result


def choose_prospective_shadow_candidates(
    audit: pd.DataFrame,
    policy_audit: pd.DataFrame,
    corridor_audit: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    required_corridor_columns = {
        "config_id",
        "corridor",
        "passes_exploratory_corridor_policy_screen",
    }
    required_policy_columns = {
        "config_id",
        "passes_corridor_candidate_policy_uncertainty_gate",
    }
    if (
        policy_audit.empty
        or corridor_audit.empty
        or not required_policy_columns.issubset(policy_audit.columns)
        or not required_corridor_columns.issubset(corridor_audit.columns)
    ):
        return {}
    pooled_candidate_pass = set(
        policy_audit.loc[
            policy_audit["passes_corridor_candidate_policy_uncertainty_gate"],
            "config_id",
        ]
    )
    corridor_pass_counts = (
        corridor_audit[corridor_audit["passes_exploratory_corridor_policy_screen"]]
        .groupby("config_id")["corridor"]
        .nunique()
    )
    corridor_pass = set(corridor_pass_counts[corridor_pass_counts.ge(2)].index)
    eligible = audit[
        audit["eligible_for_prospective_shadow"]
        & audit["config_id"].isin(pooled_candidate_pass)
        & audit["config_id"].isin(corridor_pass)
    ]
    result: dict[str, dict[str, str]] = {}
    for source_class, group in eligible.groupby("research_source_class"):
        best = group.sort_values("relative_brier_improvement", ascending=False).iloc[0]
        result[str(source_class)] = {
            "config_id": str(best["config_id"]),
            "status": "eligible_for_new_prospective_shadow_only",
        }
    return result


def select_with_rolling_quantile(
    core: Any,
    frame: pd.DataFrame,
    scores: np.ndarray,
    quantile: float,
    initial_histories: dict[str, list[float]] | None = None,
    initial_last_session: dict[str, int] | None = None,
) -> tuple[np.ndarray, dict[str, list[float]], dict[str, int]]:
    """Select one best corridor using only prior portfolio-best scores."""
    histories = {key: list(values) for key, values in (initial_histories or {}).items()}
    last_sessions = dict(initial_last_session or {})
    key = "__portfolio__"
    history = histories.setdefault(key, [])
    last_session = last_sessions.get(key, -10_000)
    selected: list[int] = []
    best_candidates = core.portfolio_best_candidates(frame, scores)
    for row in best_candidates.itertuples(index=False):
        row_index = int(row.row_index)
        score = float(row.score)
        session = int(row.session_ordinal)
        reference = history[-ADAPTIVE_SCORE_WINDOW:]
        if len(reference) >= ADAPTIVE_MIN_HISTORY:
            threshold = float(np.quantile(reference, quantile, method="higher"))
            if score >= threshold and session - last_session > ADAPTIVE_COOLDOWN_SESSIONS:
                selected.append(row_index)
                last_session = session
        history.append(score)
    histories[key] = history[-ADAPTIVE_SCORE_WINDOW:]
    last_sessions[key] = last_session
    return np.asarray(selected, dtype=int), histories, last_sessions


def choose_adaptive_quantile(
    core: Any,
    validation_history: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[float, float, dict[str, list[float]], dict[str, int]]:
    choices: list[tuple[float, float, float]] = []
    for quantile in ADAPTIVE_QUANTILE_GRID:
        selected, _, _ = select_with_rolling_quantile(
            core, validation_history, scores, float(quantile)
        )
        frequency = core.mean_frequency(validation_history, selected)
        choices.append((abs(frequency - core.TARGET_SIGNALS_PER_WEEK), -float(quantile), frequency))
    _, negative_quantile, frequency = min(choices)
    quantile = -negative_quantile
    _, histories, last_sessions = select_with_rolling_quantile(
        core, validation_history, scores, quantile
    )
    return quantile, frequency, histories, last_sessions


def mark_candidate_layer_unavailable(frame: pd.DataFrame) -> pd.DataFrame:
    """Mask candidate-level metrics for policies that only define a portfolio stream."""
    result = frame.copy()
    for column in result.columns:
        if "candidate" in column and column != "candidate_layer_available":
            result[column] = np.nan
    result["candidate_layer_available"] = False
    return result


def run_adaptive_fold(
    core: Any,
    experiment: Any,
    horizon: int,
    test_year: int,
    panel: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    train, validation, test = core.split_for_year(panel, horizon, test_year)
    validation_history = panel[
        (panel["date"] >= pd.Timestamp(test_year - 1, 1, 1))
        & (panel["date"] < pd.Timestamp(test_year, 1, 1))
    ].copy()
    features = core.FEATURE_GROUPS[experiment.feature_set]
    model = core.make_model(experiment.model_kind, features)
    model.fit(train[features + ["corridor"]], train["target"].astype(int))
    validation_raw = model.predict_proba(validation[features + ["corridor"]])[:, 1]
    calibrator = core.fit_platt_calibrator(validation_raw, validation["target"])
    validation_history_raw = model.predict_proba(validation_history[features + ["corridor"]])[:, 1]
    validation_scores = core.apply_platt(calibrator, validation_history_raw)
    raw_probabilities = model.predict_proba(test[features + ["corridor"]])[:, 1]
    probabilities = core.apply_platt(calibrator, raw_probabilities)
    quantile, validation_frequency, histories, last_sessions = choose_adaptive_quantile(
        core,
        validation_history,
        validation_scores,
    )
    selected, _, _ = select_with_rolling_quantile(
        core,
        test,
        probabilities,
        quantile,
        initial_histories=histories,
        initial_last_session=last_sessions,
    )
    adaptive = core.Experiment(
        f"{experiment.config_id}__adaptive_q126",
        experiment.model_kind,
        experiment.feature_set,
    )
    rows = core.metric_rows(
        adaptive,
        horizon,
        test_year,
        train,
        validation,
        test,
        selected,
        np.asarray([], dtype=int),
        probabilities,
        quantile,
        math.nan,
        validation_frequency,
        math.nan,
        False,
    )
    for row in rows:
        for key in list(row):
            if "candidate" in key and key != "candidate_layer_available":
                row[key] = math.nan
        row["selection_policy"] = "rolling_portfolio_best_score_quantile"
        row["score_window"] = ADAPTIVE_SCORE_WINDOW
        row["minimum_score_history"] = ADAPTIVE_MIN_HISTORY
        row["posthoc_policy"] = True
        row["candidate_layer_available"] = False
    output = test[["date", "corridor", "target", "forward_bps", "symmetric_bps", "regret_bps"]].copy()
    output["candidate_signal"] = pd.array([pd.NA] * len(output), dtype="boolean")
    output["signal"] = output.index.isin(selected)
    output["probability"] = probabilities
    output["raw_probability"] = raw_probabilities
    output["calibration_method"] = calibrator.method
    output["calibration_status"] = calibrator.status
    output["platt_intercept"] = calibrator.intercept
    output["platt_slope"] = calibrator.slope
    output["raw_validation_brier"] = calibrator.raw_validation_brier
    output["attempted_platt_validation_brier"] = (
        calibrator.attempted_platt_validation_brier
    )
    output["applied_validation_brier"] = calibrator.applied_validation_brier
    output["calibrated_validation_brier"] = calibrator.calibrated_validation_brier
    output["fold_test_year"] = test_year
    output["config_id"] = adaptive.config_id
    output["horizon_cbr_rows_pub_proxy"] = horizon
    output["selection_policy"] = "rolling_portfolio_best_score_quantile"
    output["candidate_layer_available"] = False
    return rows, output


def run_adaptive_matrix(
    core: Any,
    panel: pd.DataFrame,
    experiments: Iterable[Any],
    years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    targeted = core.add_target(panel, PRIMARY_HORIZON)
    for experiment in experiments:
        for year in years:
            fold_rows, fold_predictions = run_adaptive_fold(
                core,
                experiment,
                PRIMARY_HORIZON,
                year,
                targeted,
            )
            rows.extend(fold_rows)
            fold_predictions["model_kind"] = experiment.model_kind
            fold_predictions["feature_set"] = experiment.feature_set
            predictions.append(fold_predictions)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def joint_date_shift_attribution(
    core: Any,
    panel: pd.DataFrame,
    experiments: Iterable[Any],
    years: Iterable[int],
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Post-hoc family placebo preserving the five-corridor daily structure."""
    base = set(groups["base"])
    families = {
        "cnyrub_spot": sorted(set(groups["plus_cnyrub_spot"]) - base),
        "cnyrub_basis": sorted(set(groups["plus_cnyrub_basis"]) - base),
        "cnyrub_fixing_increment": sorted(
            set(groups["plus_cnyrub_spot_fixing"]) - set(groups["plus_cnyrub_spot"])
        ),
        "rub_risk": sorted(set(groups["plus_rub_risk_market"]) - base),
        "rub_funding": sorted(set(groups["plus_rub_funding_market"]) - base),
        "kztrub": sorted(set(groups["plus_kztrub_market"]) - base),
        "direct_corridor": sorted(set(groups["plus_direct_corridor_market"]) - base),
        "direct_corridor_basis": sorted(set(groups["plus_direct_corridor_basis"]) - base),
        "public_fast": sorted(set(groups["all_public_fast"]) - base),
    }
    rows: list[dict[str, Any]] = []
    targeted = core.add_target(panel, PRIMARY_HORIZON)
    for experiment in experiments:
        features = core.FEATURE_GROUPS[experiment.feature_set]
        model_columns = features + ["corridor"]
        for year in years:
            train, validation, test = core.split_for_year(targeted, PRIMARY_HORIZON, year)
            model = core.make_model(experiment.model_kind, features)
            model.fit(train[model_columns], train["target"].astype(int))
            validation_raw = model.predict_proba(validation[model_columns])[:, 1]
            calibrator = core.fit_platt_calibrator(validation_raw, validation["target"])
            base_raw = model.predict_proba(test[model_columns])[:, 1]
            base_probability = core.apply_platt(calibrator, base_raw)
            base_probability_by_row = pd.Series(base_probability, index=test.index)
            lookup = test.set_index(["date", "corridor"])
            dates = np.asarray(sorted(test["date"].unique()))
            for family, family_features in families.items():
                present = [feature for feature in family_features if feature in features]
                if not present:
                    continue
                for shift_sessions in (20, 60, 120):
                    if shift_sessions >= len(dates):
                        continue
                    target_dates = dates[shift_sessions:]
                    source_dates = dates[:-shift_sessions]
                    source_by_date = dict(zip(target_dates, source_dates))
                    shifted_test = test[test["date"].isin(target_dates)].copy()
                    source_index = pd.MultiIndex.from_arrays(
                        [shifted_test["date"].map(source_by_date), shifted_test["corridor"]],
                        names=["date", "corridor"],
                    )
                    shifted = shifted_test[model_columns].copy()
                    shifted[present] = lookup.loc[source_index, present].to_numpy()
                    shifted_raw = model.predict_proba(shifted)[:, 1]
                    shifted_probability = core.apply_platt(calibrator, shifted_raw)
                    labels = shifted_test["target"].astype(int).to_numpy()
                    base_subset = base_probability_by_row.loc[shifted_test.index].to_numpy()
                    base_brier = float(np.mean((base_subset - labels) ** 2))
                    shifted_brier = float(np.mean((shifted_probability - labels) ** 2))
                    rows.append(
                        {
                            "config_id": experiment.config_id,
                            "fold_test_year": year,
                            "feature_family": family,
                            "shift_sessions": shift_sessions,
                            "feature_count": len(present),
                            "base_brier": base_brier,
                            "shifted_brier": shifted_brier,
                            "brier_increase_after_joint_shift": shifted_brier - base_brier,
                            "posthoc_attribution": True,
                            "preserves_corridor_date_structure": True,
                            "non_circular_past_only_shift": True,
                            "dropped_initial_decision_dates": shift_sessions,
                        }
                    )
    return pd.DataFrame(rows)


def merge_predictive_and_policy(core: Any, fold_metrics: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    aggregate = core.aggregate_metrics(fold_metrics)
    predictive = predictive_metrics(oot)
    return aggregate.merge(
        predictive,
        on=["config_id", "model_kind", "feature_set", "horizon_cbr_rows_pub_proxy"],
        validate="one_to_one",
    )


def timing_sensitivity(
    repo_root: Path,
    data_dir: Path,
    core: Any,
    selected: Iterable[Any],
) -> pd.DataFrame:
    rows = []
    for profile in PROFILES.values():
        panel = build_feature_panel(repo_root, data_dir, core, profile)
        for experiment in selected:
            metrics, oot, _ = run_matrix(
                core,
                panel,
                [experiment],
                [PRIMARY_HORIZON],
                DEVELOPMENT_YEARS,
            )
            combined = merge_predictive_and_policy(core, metrics, oot).iloc[0].to_dict()
            combined["availability_profile"] = profile.name
            rows.append(combined)
    return pd.DataFrame(rows).sort_values("availability_profile")


def moex_lag_ladder(
    repo_root: Path,
    data_dir: Path,
    core: Any,
    experiments: Iterable[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lag_days in (0, 1, 2, 3, 5):
        profile = replace(PROFILES["primary"], name=f"moex_lag_{lag_days}", moex_days=lag_days)
        panel = build_feature_panel(repo_root, data_dir, core, profile)
        metrics, oot, _ = run_matrix(
            core,
            panel,
            experiments,
            [PRIMARY_HORIZON],
            DEVELOPMENT_YEARS,
        )
        combined = merge_predictive_and_policy(core, metrics, oot)
        combined["moex_calendar_lag_days"] = lag_days
        rows.extend(combined.to_dict("records"))
    return pd.DataFrame(rows)


def retrained_source_delay_placebo(
    core: Any,
    panel: pd.DataFrame,
    experiments: Iterable[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        incremental = sorted(set(core.FEATURE_GROUPS[experiment.feature_set]) - set(core.FEATURE_GROUPS["base"]))
        for delay_rows in (0, 20, 60, 120):
            delayed = panel.copy()
            if delay_rows:
                delayed[incremental] = delayed.groupby("corridor")[incremental].shift(delay_rows)
            metrics, oot, _ = run_matrix(
                core,
                delayed,
                [experiment],
                [PRIMARY_HORIZON],
                DEVELOPMENT_YEARS,
            )
            combined = merge_predictive_and_policy(core, metrics, oot).iloc[0].to_dict()
            combined["source_delay_cbr_rows"] = delay_rows
            combined["retrained_after_delay"] = True
            combined["no_circular_wrap"] = True
            rows.append(combined)
    return pd.DataFrame(rows)


def materiality_target_sensitivity(
    core: Any,
    panel: pd.DataFrame,
    experiments: Iterable[Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for materiality_bps in (0.0, 10.0, 25.0, 50.0):
        targeted = core.add_target(panel, PRIMARY_HORIZON, materiality_bps=materiality_bps)
        metric_rows: list[dict[str, Any]] = []
        oot_frames: list[pd.DataFrame] = []
        for experiment in experiments:
            for year in DEVELOPMENT_YEARS:
                fold_rows, oot, _ = core.run_fold(
                    experiment,
                    PRIMARY_HORIZON,
                    year,
                    targeted,
                )
                metric_rows.extend(fold_rows)
                oot["model_kind"] = experiment.model_kind
                oot["feature_set"] = experiment.feature_set
                oot_frames.append(oot)
        combined = merge_predictive_and_policy(
            core,
            pd.DataFrame(metric_rows),
            pd.concat(oot_frames, ignore_index=True),
        )
        combined["minimum_future_advantage_bps"] = materiality_bps
        rows.extend(combined.to_dict("records"))
    return pd.DataFrame(rows)


def policy_month_bootstrap(
    frame: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    signal_column: str = "signal",
) -> dict[str, float]:
    work = frame.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    blocks = [group for _, group in work.groupby("month")]

    def estimate(sample: pd.DataFrame) -> tuple[float, float, float, float, float]:
        sample = sample.copy()
        cell_keys = ["fold_test_year", "corridor"]
        by_cell = sample.groupby(cell_keys)
        sample["cell_target_rate"] = by_cell["target"].transform("mean")
        sample["cell_forward_mean"] = by_cell["forward_bps"].transform("mean")
        sample["cell_symmetric_mean"] = by_cell["symmetric_bps"].transform("mean")
        sample["cell_regret_mean"] = by_cell["regret_bps"].transform("mean")
        signal = sample[sample[signal_column]]
        if signal.empty:
            return math.nan, math.nan, math.nan, math.nan, math.nan
        expected_hits = float(signal["cell_target_rate"].sum())
        lift = float(signal["target"].sum() / expected_hits) if expected_hits else math.nan
        hit_delta_pp = float(100.0 * (signal["target"] - signal["cell_target_rate"]).mean())
        forward_delta = float((signal["forward_bps"] - signal["cell_forward_mean"]).mean())
        symmetric_delta = float(
            (signal["symmetric_bps"] - signal["cell_symmetric_mean"]).mean()
        )
        regret_improvement = float((signal["cell_regret_mean"] - signal["regret_bps"]).mean())
        return lift, hit_delta_pp, forward_delta, symmetric_delta, regret_improvement

    observed = estimate(work)
    estimates = np.empty((repetitions, 5))
    for repetition in range(repetitions):
        choices = rng.integers(0, len(blocks), size=len(blocks))
        sample = pd.concat([blocks[index] for index in choices], ignore_index=True)
        estimates[repetition] = estimate(sample)
    names = (
        "cell_standardized_lift",
        "cell_standardized_hit_delta_pp",
        "cell_standardized_forward_bps_delta",
        "cell_standardized_symmetric_bps_delta",
        "cell_standardized_regret_bps_improvement",
    )
    result: dict[str, float] = {"development_oot_month_blocks": float(len(blocks))}
    for index, name in enumerate(names):
        finite = estimates[:, index][np.isfinite(estimates[:, index])]
        result[name] = observed[index]
        result[f"{name}_ci95_low"] = float(np.quantile(finite, 0.025))
        result[f"{name}_ci95_high"] = float(np.quantile(finite, 0.975))
        result[f"{name}_ci99_low"] = float(np.quantile(finite, 0.005))
        result[f"{name}_ci99_high"] = float(np.quantile(finite, 0.995))
    return result


def development_policy_audit(
    development_oot: pd.DataFrame,
    family_audit: pd.DataFrame,
    repetitions: int,
    diagnostic_tracks: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    gate_candidate_ids = set(
        family_audit.loc[
            family_audit["eligible_for_policy_uncertainty_audit"], "config_id"
        ].astype(str)
    )
    diagnostic_ids = {
        str(selection["config_id"])
        for selection in (diagnostic_tracks or {}).values()
    }
    candidate_ids = sorted(gate_candidate_ids | diagnostic_ids)
    if not candidate_ids:
        return pd.DataFrame(
            [
                {
                    "status": "no_candidate_passed_feature_family_gates",
                    "config_id": "",
                    "audit_scope": "none",
                    "passes_policy_uncertainty_gate": False,
                    "passes_corridor_candidate_policy_uncertainty_gate": False,
                    "passes_synthetic_portfolio_uncertainty_screen": False,
                    "synthetic_portfolio_is_promotion_gate": False,
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for offset, config_id in enumerate(candidate_ids):
        candidate = development_oot[
            development_oot["config_id"].eq(config_id)
            & development_oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)
        ].copy()
        policy = policy_month_bootstrap(
            candidate,
            repetitions,
            np.random.default_rng(SEED + 1_000 + offset),
        )
        candidate_policy = policy_month_bootstrap(
            candidate,
            repetitions,
            np.random.default_rng(SEED + 2_000 + offset),
            signal_column="candidate_signal",
        )
        signal_mix = candidate[candidate["signal"]]["corridor"].value_counts(normalize=True)
        synthetic_predicates = {
            "minimum_point_lift": policy["cell_standardized_lift"] >= MIN_POINT_LIFT,
            "lift_ci95_low_above_one": policy["cell_standardized_lift_ci95_low"] > 1.0,
            "hit_delta_ci95_low_above_zero": (
                policy["cell_standardized_hit_delta_pp_ci95_low"] > 0.0
            ),
            "symmetric_bps_ci95_low_above_zero": (
                policy["cell_standardized_symmetric_bps_delta_ci95_low"] > 0.0
            ),
            "forward_point_nonnegative": policy["cell_standardized_forward_bps_delta"] >= 0.0,
            "forward_ci95_low_above_minus_25": (
                policy["cell_standardized_forward_bps_delta_ci95_low"] > -25.0
            ),
        }
        corridor_candidate_predicates = {
            "minimum_point_lift": candidate_policy["cell_standardized_lift"] >= MIN_POINT_LIFT,
            "lift_ci95_low_above_one": (
                candidate_policy["cell_standardized_lift_ci95_low"] > 1.0
            ),
            "hit_delta_ci95_low_above_zero": (
                candidate_policy["cell_standardized_hit_delta_pp_ci95_low"] > 0.0
            ),
            "symmetric_bps_ci95_low_above_zero": (
                candidate_policy["cell_standardized_symmetric_bps_delta_ci95_low"] > 0.0
            ),
            "forward_point_nonnegative": (
                candidate_policy["cell_standardized_forward_bps_delta"] >= 0.0
            ),
            "forward_ci95_low_above_minus_25": (
                candidate_policy["cell_standardized_forward_bps_delta_ci95_low"] > -25.0
            ),
        }
        synthetic_predicates = {
            name: bool(passed) for name, passed in synthetic_predicates.items()
        }
        corridor_candidate_predicates = {
            name: bool(passed) for name, passed in corridor_candidate_predicates.items()
        }
        synthetic_portfolio_pass = bool(all(synthetic_predicates.values()))
        corridor_candidate_pass = bool(all(corridor_candidate_predicates.values()))
        rows.append(
            {
                "status": "evaluated",
                "config_id": config_id,
                "audit_scope": (
                    "promotion_gate_candidate"
                    if config_id in gate_candidate_ids
                    else "diagnostic_track_only"
                ),
                "signal_dates": int(candidate.loc[candidate["signal"], "date"].nunique()),
                "signal_corridor_mix": json.dumps(signal_mix.round(6).to_dict(), sort_keys=True),
                **policy,
                **{
                    f"candidate_{name}": value
                    for name, value in candidate_policy.items()
                },
                "passes_policy_uncertainty_gate": corridor_candidate_pass,
                "passes_corridor_candidate_policy_uncertainty_gate": corridor_candidate_pass,
                "corridor_candidate_gate_predicates": json.dumps(
                    corridor_candidate_predicates, sort_keys=True
                ),
                "corridor_candidate_gate_failures": json.dumps(
                    [
                        name
                        for name, passed in corridor_candidate_predicates.items()
                        if not passed
                    ]
                ),
                "passes_synthetic_portfolio_uncertainty_screen": synthetic_portfolio_pass,
                "synthetic_portfolio_screen_predicates": json.dumps(
                    synthetic_predicates, sort_keys=True
                ),
                "synthetic_portfolio_is_promotion_gate": False,
            }
        )
    return pd.DataFrame(rows)


def corridor_candidate_uncertainty_audit(
    development_oot: pd.DataFrame,
    diagnostic_tracks: dict[str, dict[str, str]],
    repetitions: int,
) -> pd.DataFrame:
    """Keep corridor policy signals separate from the all-five portfolio scenario."""
    rows: list[dict[str, Any]] = []
    for track_offset, (track_name, selection) in enumerate(diagnostic_tracks.items()):
        candidate = development_oot[
            development_oot["config_id"].eq(selection["config_id"])
            & development_oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)
        ]
        for corridor_offset, (corridor, group) in enumerate(candidate.groupby("corridor")):
            policy = policy_month_bootstrap(
                group,
                repetitions,
                np.random.default_rng(
                    SEED + 3_000 + 100 * track_offset + corridor_offset
                ),
                signal_column="candidate_signal",
            )
            fold_counts = group.groupby("fold_test_year")["candidate_signal"].sum()
            weekly_fulfillment: list[float] = []
            silent_week_shares: list[float] = []
            for _, fold in group.groupby("fold_test_year"):
                periods = fold["date"].dt.to_period("W-SUN")
                candidate_weeks = pd.period_range(periods.min(), periods.max(), freq="W-SUN")
                first_date = fold["date"].min().normalize()
                last_date = fold["date"].max().normalize()
                all_weeks = pd.PeriodIndex(
                    [
                        week
                        for week in candidate_weeks
                        if week.start_time.normalize() >= first_date
                        and week.end_time.normalize() <= last_date
                    ],
                    freq="W-SUN",
                )
                if all_weeks.empty:
                    all_weeks = candidate_weeks
                counts = (
                    fold.loc[fold["candidate_signal"], "date"]
                    .dt.to_period("W-SUN")
                    .value_counts()
                    .reindex(all_weeks, fill_value=0)
                )
                weekly_fulfillment.append(float(counts.between(1, 2).mean()))
                silent_week_shares.append(float(counts.eq(0).mean()))
            signal_count = int(group["candidate_signal"].sum())
            simultaneous_lift_low = policy["cell_standardized_lift_ci99_low"]
            simultaneous_hit_delta_low = policy[
                "cell_standardized_hit_delta_pp_ci99_low"
            ]
            simultaneous_symmetric_low = policy[
                "cell_standardized_symmetric_bps_delta_ci99_low"
            ]
            gate_predicates = {
                "minimum_signal_count": signal_count >= 20,
                "minimum_active_folds": fold_counts.gt(0).sum() >= 3,
                "maximum_top_fold_share": (
                    signal_count > 0
                    and float(fold_counts.max()) / signal_count <= MAX_TOP_FOLD_SIGNAL_SHARE
                ),
                "minimum_weekly_cadence_fulfillment": (
                    min(weekly_fulfillment) >= MIN_WEEKLY_CADENCE_FULFILLMENT
                ),
                "minimum_point_lift": policy["cell_standardized_lift"] >= MIN_POINT_LIFT,
                "simultaneous_lift_lower_above_one": simultaneous_lift_low > 1.0,
                "simultaneous_hit_delta_lower_above_zero": simultaneous_hit_delta_low > 0.0,
                "simultaneous_symmetric_bps_lower_above_zero": simultaneous_symmetric_low > 0.0,
                "forward_point_nonnegative": policy["cell_standardized_forward_bps_delta"] >= 0.0,
                "simultaneous_forward_bps_lower_above_minus_25": (
                    policy["cell_standardized_forward_bps_delta_ci99_low"] > -25.0
                ),
            }
            gate_predicates = {
                name: bool(passed) for name, passed in gate_predicates.items()
            }
            rows.append(
                {
                    "track": track_name,
                    "config_id": selection["config_id"],
                    "corridor": corridor,
                    "signal_count": signal_count,
                    "active_fold_count": int(fold_counts.gt(0).sum()),
                    "top_fold_signal_share": (
                        float(fold_counts.max()) / signal_count
                        if signal_count
                        else math.nan
                    ),
                    "minimum_fold_weekly_1_to_2_fulfillment": min(weekly_fulfillment),
                    "mean_silent_week_share": float(np.mean(silent_week_shares)),
                    **policy,
                    "bonferroni_five_corridor_simultaneous95_lift_low": simultaneous_lift_low,
                    "bonferroni_five_corridor_simultaneous95_hit_delta_pp_low": (
                        simultaneous_hit_delta_low
                    ),
                    "bonferroni_five_corridor_simultaneous95_symmetric_bps_delta_low": (
                        simultaneous_symmetric_low
                    ),
                    "gate_predicates": json.dumps(gate_predicates, sort_keys=True),
                    "gate_failures": json.dumps(
                        [name for name, passed in gate_predicates.items() if not passed]
                    ),
                    "passes_exploratory_corridor_policy_screen": bool(
                        all(gate_predicates.values())
                    ),
                    "status": "retrospective_exploratory_not_prospective_confirmation",
                }
            )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        [
            {
                "status": "no_diagnostic_track_available",
                "config_id": "",
                "corridor": "",
                "passes_exploratory_corridor_policy_screen": False,
            }
        ]
    )


def diagnostic_2026_audit(
    diagnostic_oot: pd.DataFrame,
    diagnostic_tracks: dict[str, dict[str, str]],
    family_audit: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2026)
    rows: list[dict[str, Any]] = []
    for track_name, selection in diagnostic_tracks.items():
        config_id = selection["config_id"]
        candidate = diagnostic_oot[diagnostic_oot["config_id"].eq(config_id)].copy()
        audit_row = family_audit[family_audit["config_id"].eq(config_id)].iloc[0]
        comparison_base_id = str(audit_row["comparison_base_id"])
        base = diagnostic_oot[diagnostic_oot["config_id"].eq(comparison_base_id)][
            ["date", "corridor", "fold_test_year", "target", "probability"]
        ].rename(columns={"probability": "base_probability"})
        paired = candidate.merge(
            base,
            on=["date", "corridor", "fold_test_year", "target"],
            how="inner",
            validate="one_to_one",
        )
        paired["loss_delta"] = (
            paired["probability"] - paired["target"]
        ) ** 2 - (paired["base_probability"] - paired["target"]) ** 2
        brier = stratified_month_bootstrap(paired, repetitions, rng)
        policy = policy_month_bootstrap(candidate, repetitions, rng)
        rows.append(
            {
                "track": track_name,
                "config_id": config_id,
                "development_selection_status": selection["status"],
                "comparison_base_id": comparison_base_id,
                "paired_prediction_rows": len(paired),
                "already_inspected_partial_year_diagnostic": True,
                **brier,
                **policy,
            }
        )
    return pd.DataFrame(rows)


def attach_source_lineage(predictions: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    lineage_columns = [
        column
        for column in panel
        if column.endswith(("_observation_date", "_available_date", "_age_days"))
    ]
    lineage = panel[["date", "corridor", *lineage_columns]].drop_duplicates(["date", "corridor"])
    return predictions.merge(
        lineage,
        on=["date", "corridor"],
        how="left",
        validate="many_to_one",
    )


def counterfactual_wait_cost_curve(
    oot: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = oot[oot["signal"]].copy()
    rate_lookup: dict[str, tuple[list[pd.Timestamp], np.ndarray, dict[pd.Timestamp, int]]] = {}
    for corridor, group in panel.sort_values("date").groupby("corridor"):
        dates = group["date"].tolist()
        rates = group["rub_per_unit"].to_numpy(float)
        rate_lookup[str(corridor)] = (dates, rates, {date: index for index, date in enumerate(dates)})
    rows: list[dict[str, Any]] = []
    for signal in signals.itertuples(index=False):
        dates, rates, positions = rate_lookup[str(signal.corridor)]
        position = positions.get(signal.date)
        if position is None:
            continue
        for wait_rows in (1, 2, 3, 5):
            if position + wait_rows >= len(dates):
                continue
            waiting_bps = (rates[position + wait_rows] / rates[position] - 1.0) * 10_000.0
            rows.append(
                {
                    "config_id": signal.config_id,
                    "fold_test_year": signal.fold_test_year,
                    "signal_date": signal.date,
                    "corridor": signal.corridor,
                    "wait_cbr_rows": wait_rows,
                    "future_date": dates[position + wait_rows],
                    "counterfactual_official_rate_change_while_waiting_bps": waiting_bps,
                    "is_fast_slow_confirmation_pair": False,
                }
            )
    events = pd.DataFrame(rows)
    if events.empty:
        return events, pd.DataFrame()
    summary = events.groupby(["config_id", "fold_test_year", "wait_cbr_rows"])[
        "counterfactual_official_rate_change_while_waiting_bps"
    ].agg(["count", "mean", "median"])
    summary = summary.reset_index().rename(
        columns={"count": "signal_count", "mean": "mean_bps", "median": "median_bps"}
    )
    return events, summary


def fast_slow_confirmation_audit(
    oot: pd.DataFrame,
    panel: pd.DataFrame,
    pairs: Iterable[tuple[str, str, str]],
    maximum_wait_rows: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair a fast market model trigger with a later official-history trigger."""
    primary = oot[oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)].copy()
    rate_lookup: dict[str, tuple[list[pd.Timestamp], np.ndarray, dict[pd.Timestamp, int]]] = {}
    for corridor, group in panel.sort_values("date").groupby("corridor"):
        dates = group["date"].tolist()
        rates = group["rub_per_unit"].to_numpy(float)
        rate_lookup[str(corridor)] = (dates, rates, {date: index for index, date in enumerate(dates)})
    rows: list[dict[str, Any]] = []
    for comparison_id, fast_id, slow_id in pairs:
        fast = primary[primary["config_id"].eq(fast_id) & primary["signal"]]
        slow = primary[primary["config_id"].eq(slow_id) & primary["signal"]]
        slow_by_cell = {
            (int(year), str(corridor)): group.sort_values("date")
            for (year, corridor), group in slow.groupby(["fold_test_year", "corridor"])
        }
        for event in fast.itertuples(index=False):
            dates, rates, positions = rate_lookup[str(event.corridor)]
            fast_position = positions.get(event.date)
            if fast_position is None:
                continue
            candidates = slow_by_cell.get((int(event.fold_test_year), str(event.corridor)))
            confirmation = None
            if candidates is not None:
                future = candidates[candidates["date"] >= event.date]
                for candidate in future.itertuples(index=False):
                    slow_position = positions.get(candidate.date)
                    if slow_position is None:
                        continue
                    if slow_position - fast_position <= maximum_wait_rows:
                        confirmation = (candidate, slow_position)
                    break
            row = {
                "comparison_id": comparison_id,
                "fast_config_id": fast_id,
                "slow_config_id": slow_id,
                "fold_test_year": int(event.fold_test_year),
                "corridor": str(event.corridor),
                "fast_signal_date": event.date,
                "fast_target_h5": int(event.target),
                "confirmed_within_maximum_wait": confirmation is not None,
                "maximum_wait_cbr_rows": maximum_wait_rows,
            }
            if confirmation is not None:
                candidate, slow_position = confirmation
                wait_rows = slow_position - fast_position
                row.update(
                    {
                        "slow_confirmation_date": candidate.date,
                        "wait_cbr_rows": wait_rows,
                        "slow_target_h5": int(candidate.target),
                        "official_rate_waiting_cost_bps": (
                            rates[slow_position] / rates[fast_position] - 1.0
                        )
                        * 10_000.0,
                    }
                )
            rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        return events, pd.DataFrame()
    summaries: list[dict[str, Any]] = []
    for keys, group in events.groupby(
        ["comparison_id", "fast_config_id", "slow_config_id", "fold_test_year"]
    ):
        confirmed = group[group["confirmed_within_maximum_wait"]]
        summaries.append(
            {
                "comparison_id": keys[0],
                "fast_config_id": keys[1],
                "slow_config_id": keys[2],
                "fold_test_year": keys[3],
                "fast_signal_count": len(group),
                "confirmation_count": len(confirmed),
                "confirmation_rate": len(confirmed) / len(group),
                "fast_h5_hit_rate": float(group["fast_target_h5"].mean()),
                "slow_confirmed_h5_hit_rate": (
                    float(confirmed["slow_target_h5"].mean()) if not confirmed.empty else math.nan
                ),
                "median_wait_cbr_rows": (
                    float(confirmed["wait_cbr_rows"].median()) if not confirmed.empty else math.nan
                ),
                "median_official_rate_waiting_cost_bps": (
                    float(confirmed["official_rate_waiting_cost_bps"].median())
                    if not confirmed.empty
                    else math.nan
                ),
                "mean_official_rate_waiting_cost_bps": (
                    float(confirmed["official_rate_waiting_cost_bps"].mean())
                    if not confirmed.empty
                    else math.nan
                ),
                "interpretation": (
                    "exploratory trigger-pairing; slow trigger is an official-history model, "
                    "not an independently identified causal confirmation"
                ),
            }
        )
    return events, pd.DataFrame(summaries)


def data_coverage(panel: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    dates = panel["date"]
    for group_name, features in groups.items():
        if group_name == "base":
            continue
        for feature in sorted(set(features) - set(groups["base"])):
            values = panel[feature]
            rows.append(
                {
                    "feature_set": group_name,
                    "feature": feature,
                    "non_null_rate": float(values.notna().mean()),
                    "first_non_null": str(dates[values.notna()].min().date()) if values.notna().any() else None,
                    "last_non_null": str(dates[values.notna()].max().date()) if values.notna().any() else None,
                    "unique_values": int(values.nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.run_tier == "final" and args.bootstrap_reps < MIN_FINAL_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            "final runs require at least "
            f"{MIN_FINAL_BOOTSTRAP_REPETITIONS} bootstrap repetitions"
        )
    repo_root = args.repo_root.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix a new run with existing artifacts: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    code_paths = [
        Path(__file__).resolve(),
        repo_root / "final_solution" / "training" / "core_experiment.py",
        repo_root / "final_solution" / "data_pipeline" / "fetch_open_data.py",
        repo_root / "final_solution" / "tests" / "test_training.py",
        repo_root / "final_solution" / "requirements-ml.txt",
    ]
    input_paths = [
        repo_root / "final_solution" / "data" / "cbr_daily.csv",
        data_dir.parent / "data_manifest.json",
        *sorted(data_dir.glob("*.csv")),
    ]
    initial_code_hashes = {
        str(path.relative_to(repo_root)): sha256(path) for path in code_paths
    }
    initial_input_hashes = {
        str(path.relative_to(repo_root)): sha256(path) for path in input_paths
    }
    core = load_core(repo_root)
    if core.COOLDOWN_SESSIONS != ADAPTIVE_COOLDOWN_SESSIONS:
        raise ValueError("adaptive and fixed-threshold cooldown settings diverged")

    groups = feature_groups(core)
    core.FEATURE_GROUPS.update(groups)
    experiments = experiment_specs(core)
    experiment_by_id = {experiment.config_id: experiment for experiment in experiments}
    primary_panel = build_feature_panel(repo_root, data_dir, core, PROFILES["primary"])
    coverage = data_coverage(primary_panel, groups)

    primary_metrics_raw, primary_oot, primary_coefficients = run_matrix(
        core,
        primary_panel,
        experiments,
        [PRIMARY_HORIZON],
        DEVELOPMENT_YEARS,
    )
    horizon_sensitivity_ids = {
        "logit_base",
        "hgb_base",
        "hgb_plus_cnyrub_basis",
        "hgb_plus_direct_corridor_basis",
        "hgb_all_public_fast",
        "hgb_all_contract_market",
        "hgb_full_research",
    }
    secondary_metrics_raw, secondary_oot, secondary_coefficients = run_matrix(
        core,
        primary_panel,
        [experiment for experiment in experiments if experiment.config_id in horizon_sensitivity_ids],
        [horizon for horizon in HORIZONS if horizon != PRIMARY_HORIZON],
        DEVELOPMENT_YEARS,
    )
    development_metrics_raw = pd.concat(
        [primary_metrics_raw, secondary_metrics_raw], ignore_index=True
    )
    development_oot = pd.concat([primary_oot, secondary_oot], ignore_index=True)
    coefficients = pd.concat(
        [primary_coefficients, secondary_coefficients], ignore_index=True
    )
    development = merge_predictive_and_policy(core, development_metrics_raw, development_oot)
    family_audit = paired_feature_audit(development_oot, development, args.bootstrap_reps)
    benchmark_audit = benchmark_comparison_audit(development_oot, args.bootstrap_reps)
    block_sensitivity = block_length_sensitivity(development_oot, args.bootstrap_reps)
    diagnostic_tracks = choose_diagnostic_candidates(family_audit)
    policy_audit = development_policy_audit(
        development_oot,
        family_audit,
        args.bootstrap_reps,
        diagnostic_tracks,
    )
    gate_candidate_tracks = {
        str(config_id): {"config_id": str(config_id)}
        for config_id in family_audit.loc[
            family_audit["eligible_for_policy_uncertainty_audit"], "config_id"
        ]
    }
    corridor_candidate_audit = corridor_candidate_uncertainty_audit(
        development_oot,
        gate_candidate_tracks,
        args.bootstrap_reps,
    )
    diagnostic_corridor_candidate_audit = corridor_candidate_uncertainty_audit(
        development_oot,
        diagnostic_tracks,
        args.bootstrap_reps,
    )
    prospective_shadow_tracks = choose_prospective_shadow_candidates(
        family_audit,
        policy_audit,
        corridor_candidate_audit,
    )
    if args.run_tier == "smoke":
        prospective_shadow_tracks = {}

    diagnostic_ids = ["logit_base", "hgb_base"]
    for track in diagnostic_tracks.values():
        selected_id = track["config_id"]
        selected_comparison_base = str(
            family_audit.loc[family_audit["config_id"].eq(selected_id), "comparison_base_id"].iloc[0]
        )
        if selected_comparison_base not in diagnostic_ids:
            diagnostic_ids.append(selected_comparison_base)
        if selected_id not in diagnostic_ids:
            diagnostic_ids.append(selected_id)
    diagnostic_experiments = [experiment_by_id[config_id] for config_id in diagnostic_ids]
    diagnostic_metrics_raw, diagnostic_oot, diagnostic_coefficients = run_matrix(
        core,
        primary_panel,
        diagnostic_experiments,
        [PRIMARY_HORIZON],
        [DIAGNOSTIC_YEAR],
    )
    diagnostic = merge_predictive_and_policy(core, diagnostic_metrics_raw, diagnostic_oot)
    diagnostic_candidate_experiments = [
        experiment_by_id[track["config_id"]] for track in diagnostic_tracks.values()
    ]
    timing = timing_sensitivity(repo_root, data_dir, core, diagnostic_candidate_experiments)
    lag_ladder_experiments = [
        experiment_by_id[config_id]
        for config_id in (
            "hgb_plus_cnyrub_basis",
            "hgb_plus_cnyrub_ret1",
            "hgb_plus_direct_corridor_basis",
            "hgb_all_contract_market",
        )
    ]
    lag_ladder = moex_lag_ladder(
        repo_root,
        data_dir,
        core,
        lag_ladder_experiments,
    )
    retrained_delay = retrained_source_delay_placebo(
        core,
        primary_panel,
        [
            experiment_by_id["hgb_plus_cnyrub_basis"],
            experiment_by_id["hgb_plus_direct_corridor_basis"],
        ],
    )
    materiality_sensitivity = materiality_target_sensitivity(
        core,
        primary_panel,
        [
            experiment_by_id["hgb_base"],
            experiment_by_id["hgb_plus_cnyrub_basis"],
            experiment_by_id["hgb_plus_direct_corridor_basis"],
            experiment_by_id["hgb_all_contract_market"],
        ],
    )
    diagnostic_audit = diagnostic_2026_audit(
        diagnostic_oot,
        diagnostic_tracks,
        family_audit,
        args.bootstrap_reps,
    )
    adaptive_development_raw, adaptive_development_oot = run_adaptive_matrix(
        core,
        primary_panel,
        diagnostic_candidate_experiments,
        DEVELOPMENT_YEARS,
    )
    adaptive_diagnostic_raw, adaptive_diagnostic_oot = run_adaptive_matrix(
        core,
        primary_panel,
        diagnostic_candidate_experiments,
        [DIAGNOSTIC_YEAR],
    )
    adaptive_development = merge_predictive_and_policy(
        core,
        adaptive_development_raw,
        adaptive_development_oot,
    )
    adaptive_development = mark_candidate_layer_unavailable(adaptive_development)
    adaptive_diagnostic = merge_predictive_and_policy(
        core,
        adaptive_diagnostic_raw,
        adaptive_diagnostic_oot,
    )
    adaptive_diagnostic = mark_candidate_layer_unavailable(adaptive_diagnostic)
    attribution = joint_date_shift_attribution(
        core,
        primary_panel,
        diagnostic_candidate_experiments,
        [*DEVELOPMENT_YEARS, DIAGNOSTIC_YEAR],
        groups,
    )
    calibration = pd.concat(
        [
            calibration_diagnostics(development_oot),
            calibration_diagnostics(development_oot, "raw_probability"),
            calibration_diagnostics(diagnostic_oot),
            calibration_diagnostics(diagnostic_oot, "raw_probability"),
        ],
        ignore_index=True,
    )
    target_correlations, target_date_summary = target_dependence_diagnostics(primary_panel, core)
    univariate = univariate_rank_diagnostics(
        primary_panel,
        core,
        [
            "moex_cnyrub_ret1",
            "moex_cny_close_minus_fixing_same_session",
            "moex_kztrub_ret1",
            "direct_corridor_market_ret1",
            "direct_corridor_market_minus_official_log_basis",
            "eia_brent_ret5",
            "fed_broad_usd_ret5",
            "ecb_ciss_fx_chg5",
            "ru_business_climate_chg1m",
            "kz_bai_economy_chg1m_x_kzt",
            "kz_cpi_mom_accel1m_x_kzt",
            "kz_expected_inflation_12m_pct_chg1m_x_kzt",
        ],
    )
    diagnostic_candidate_ids = {track["config_id"] for track in diagnostic_tracks.values()}
    waiting_events, waiting_summary = counterfactual_wait_cost_curve(
        pd.concat(
            [
                development_oot[
                    development_oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)
                    & development_oot["config_id"].isin(diagnostic_candidate_ids)
                ],
                diagnostic_oot[diagnostic_oot["config_id"].isin(diagnostic_candidate_ids)],
            ],
            ignore_index=True,
        ),
        primary_panel,
    )
    fast_slow_events, fast_slow_summary = fast_slow_confirmation_audit(
        development_oot,
        primary_panel,
        [
            ("cny_basis_fast_vs_official_history", "hgb_plus_cnyrub_basis", "hgb_base"),
            ("public_fast_vs_official_history", "hgb_all_public_fast", "hgb_base"),
        ],
    )

    development_predictions = attach_source_lineage(
        development_oot[development_oot["horizon_cbr_rows_pub_proxy"].eq(PRIMARY_HORIZON)],
        primary_panel,
    )
    diagnostic_predictions = attach_source_lineage(diagnostic_oot, primary_panel)

    development.to_csv(output_dir / "development_metrics.csv", index=False)
    development_metrics_raw.to_csv(output_dir / "development_fold_corridor_metrics.csv", index=False)
    predictive_cell_metrics(development_oot).to_csv(
        output_dir / "development_predictive_fold_corridor_metrics.csv", index=False
    )
    development_predictions.to_csv(output_dir / "development_h5_predictions.csv", index=False)
    family_audit.to_csv(output_dir / "feature_family_audit_h5.csv", index=False)
    benchmark_audit.to_csv(output_dir / "benchmark_comparison_audit_h5.csv", index=False)
    block_sensitivity.to_csv(output_dir / "block_length_sensitivity_h5.csv", index=False)
    policy_audit.to_csv(output_dir / "development_policy_uncertainty_audit_h5.csv", index=False)
    corridor_candidate_audit.to_csv(
        output_dir / "development_corridor_candidate_uncertainty_h5.csv",
        index=False,
    )
    diagnostic_corridor_candidate_audit.to_csv(
        output_dir / "diagnostic_corridor_candidate_uncertainty_h5.csv",
        index=False,
    )
    diagnostic.to_csv(output_dir / "diagnostic_2026_metrics.csv", index=False)
    diagnostic_metrics_raw.to_csv(output_dir / "diagnostic_2026_fold_corridor_metrics.csv", index=False)
    predictive_cell_metrics(diagnostic_oot).to_csv(
        output_dir / "diagnostic_2026_predictive_fold_corridor_metrics.csv", index=False
    )
    diagnostic_predictions.to_csv(output_dir / "diagnostic_2026_predictions.csv", index=False)
    diagnostic_audit.to_csv(output_dir / "diagnostic_2026_audit.csv", index=False)
    timing.to_csv(output_dir / "availability_timing_sensitivity.csv", index=False)
    lag_ladder.to_csv(output_dir / "moex_lag_ladder_h5.csv", index=False)
    retrained_delay.to_csv(output_dir / "retrained_source_delay_placebo_h5.csv", index=False)
    materiality_sensitivity.to_csv(output_dir / "target_materiality_sensitivity_h5.csv", index=False)
    adaptive_development.to_csv(output_dir / "posthoc_adaptive_development_metrics.csv", index=False)
    adaptive_diagnostic.to_csv(output_dir / "posthoc_adaptive_2026_metrics.csv", index=False)
    pd.concat([adaptive_development_raw, adaptive_diagnostic_raw], ignore_index=True).to_csv(
        output_dir / "posthoc_adaptive_fold_corridor_metrics.csv",
        index=False,
    )
    pd.concat([adaptive_development_oot, adaptive_diagnostic_oot], ignore_index=True).to_csv(
        output_dir / "posthoc_adaptive_predictions.csv",
        index=False,
    )
    attribution.to_csv(output_dir / "posthoc_joint_date_shift_attribution.csv", index=False)
    calibration.to_csv(output_dir / "calibration_diagnostics.csv", index=False)
    target_correlations.to_csv(output_dir / "target_pairwise_correlations_h5.csv", index=False)
    target_date_summary.to_csv(output_dir / "target_date_dependence_summary_h5.csv", index=False)
    univariate.to_csv(output_dir / "posthoc_univariate_rank_diagnostics_h5.csv", index=False)
    waiting_events.to_csv(output_dir / "counterfactual_wait_cost_events.csv", index=False)
    waiting_summary.to_csv(output_dir / "counterfactual_wait_cost_summary.csv", index=False)
    fast_slow_events.to_csv(output_dir / "fast_slow_confirmation_events.csv", index=False)
    fast_slow_summary.to_csv(output_dir / "fast_slow_confirmation_summary.csv", index=False)
    coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    pd.concat([coefficients, diagnostic_coefficients], ignore_index=True).to_csv(
        output_dir / "feature_coefficients.csv",
        index=False,
    )

    selection = {
        "run_tier": args.run_tier,
        "inferential_status": (
            "final_retrospective_exploratory"
            if args.run_tier == "final"
            else "smoke_not_for_selection"
        ),
        "minimum_final_bootstrap_repetitions": MIN_FINAL_BOOTSTRAP_REPETITIONS,
        "primary_horizon_cbr_rows_pub_proxy": PRIMARY_HORIZON,
        "development_years": DEVELOPMENT_YEARS,
        "diagnostic_year_already_inspected": DIAGNOSTIC_YEAR,
        "diagnostic_tracks": diagnostic_tracks,
        "prospective_shadow_tracks": prospective_shadow_tracks,
        "production_promoted_tracks": [],
        "selection_rule": {
            "minimum_relative_brier_improvement": MIN_RELATIVE_BRIER_IMPROVEMENT,
            "minimum_improved_years": MIN_IMPROVED_YEARS,
            "minimum_improved_fold_corridor_cells": MIN_IMPROVED_CELLS,
            "minimum_point_cell_standardized_lift": MIN_POINT_LIFT,
            "minimum_fold_lift_strictly_above": 1.0,
            "pooled_corridor_candidate_lift_month_block_ci95_lower_strictly_above": 1.0,
            "pooled_corridor_candidate_hit_delta_pp_ci95_lower_strictly_above": 0.0,
            "pooled_corridor_candidate_symmetric_bps_ci95_lower_strictly_above": 0.0,
            "pooled_corridor_candidate_forward_point_minimum_bps": 0.0,
            "pooled_corridor_candidate_forward_ci95_lower_strictly_above_bps": -25.0,
            "brier_month_block_ci_upper_below_zero": True,
            "holm_adjusted_p_below": 0.05,
            "positive_brier_skill_vs_prior_year_required": True,
            "per_signal_model_explanation_required": True,
            "source_lineage_and_policy_reproducibility_required": True,
            "candidate_cadence_range_per_corridor_week": CANDIDATE_CADENCE_RANGE,
            "minimum_weekly_candidate_cadence_fulfillment": MIN_WEEKLY_CADENCE_FULFILLMENT,
            "maximum_top_fold_signal_share": MAX_TOP_FOLD_SIGNAL_SHARE,
            "minimum_passing_corridors": 2,
            "minimum_signals_per_passing_corridor": 20,
            "minimum_active_folds_per_passing_corridor": 3,
            "corridor_simultaneous95_implemented_as_two_sided_99pct_interval": True,
            "corridor_lift_simultaneous_lower_strictly_above": 1.0,
            "corridor_hit_delta_pp_simultaneous_lower_strictly_above": 0.0,
            "corridor_symmetric_bps_simultaneous_lower_strictly_above": 0.0,
            "corridor_forward_point_minimum_bps": 0.0,
            "corridor_forward_simultaneous_lower_strictly_above_bps": -25.0,
            "synthetic_all_five_market_portfolio_cadence_range_per_week_diagnostic_only": (
                PORTFOLIO_CADENCE_RANGE
            ),
            "corridor_candidate_forward_delta_vs_base_nonnegative": True,
            "fast_machine_readable_source_required": True,
            "statistical_unit": "decision_date; five correlated corridor rows collapsed",
            "threshold_objective": (
                "independently per corridor on purged validation: cadence band first, then weighted_error_3 "
                "with FP:FN=3:1"
            ),
            "offline_level_names": {
                "candidate_signal": "corridor policy signal after threshold and corridor cooldown",
                "signal": "synthetic all-five market portfolio after shared cooldown; not a client estimate",
                "delivered_contact": "not observed; requires client eligibility and the bank CRM budget",
            },
            "synthetic_portfolio_tie_break_rule": core.PORTFOLIO_TIE_BREAK_RULE,
            "synthetic_portfolio_is_promotion_gate": False,
            "prediction_suppression_ledger": [
                "raw_threshold_candidate",
                "corridor_cooldown_suppressed",
                "candidate_signal",
                "portfolio_daily_rank_suppressed",
                "portfolio_shared_cooldown_suppressed",
                "signal",
            ],
        },
        "protocol_note": (
            "This entire study is retrospective and exploratory: multiple horizons, feature families and 2026 "
            "diagnostics were inspected while the protocol evolved. Holm correction covers only the displayed "
            "h=5 family and is a screen, not a confirmatory significance claim. Any candidate must be frozen and "
            "evaluated in a new prospective shadow period before promotion."
        ),
        "availability_profiles": {name: asdict(profile) for name, profile in PROFILES.items()},
        "posthoc_adaptive_policy": {
            "reason": "designed after observing 2026 fixed-threshold cadence drift; diagnostic only",
            "selection_uses_labels": False,
            "score_window_portfolio_best_dates": ADAPTIVE_SCORE_WINDOW,
            "minimum_history_dates": ADAPTIVE_MIN_HISTORY,
            "cooldown_sessions": ADAPTIVE_COOLDOWN_SESSIONS,
            "quantile_is_selected_on_previous_year_scores_for_target_cadence": True,
        },
        "probability_calibration": {
            "method": (
                "positive-slope Platt calibration on the purged prior-year validation fold; "
                "explicit identity fallback for a single class, non-monotone fitted slope or no validation Brier gain"
            ),
            "raw_and_calibrated_probabilities_saved": True,
            "calibration_parameters_never_fit_on_test": True,
            "ranking_inversion_forbidden": True,
        },
        "warnings": [
            "CBR date is an effective date, not a proven publication timestamp.",
            "World Bank current workbook is not an immutable real-time vintage.",
            "MOEX/KASE-style market-data use still needs legal approval for production.",
            "2026 is partial, already inspected, and never treated as confirmation.",
            "Month-block sign-flip p-values assume symmetric block effects and are secondary to confidence intervals.",
            "The rolling-quantile cadence policy is a post-hoc operational robustness analysis.",
            "The post-hoc adaptive policy is portfolio-only and has no comparable corridor candidate layer.",
            "The all-five market portfolio is an upper-bound scenario, not a per-client CRM simulation.",
            "The synthetic portfolio uses a deterministic hash tie-break and is diagnostic, never a promotion gate.",
            "Counterfactual wait and fast/slow trigger-pair deltas are not executable P&L and exclude spread, fees and app quote slippage.",
            "The slow trigger in the fast/slow audit is an official-history model, not an independently identified causal confirmation.",
            "Current downloaded histories are latest-revised snapshots, not immutable point-in-time vintages.",
        ],
    }
    write_json(output_dir / "selection.json", selection)
    expected_outputs = {
        "availability_timing_sensitivity.csv",
        "benchmark_comparison_audit_h5.csv",
        "block_length_sensitivity_h5.csv",
        "calibration_diagnostics.csv",
        "counterfactual_wait_cost_events.csv",
        "counterfactual_wait_cost_summary.csv",
        "development_fold_corridor_metrics.csv",
        "development_h5_predictions.csv",
        "development_metrics.csv",
        "development_policy_uncertainty_audit_h5.csv",
        "development_corridor_candidate_uncertainty_h5.csv",
        "diagnostic_corridor_candidate_uncertainty_h5.csv",
        "development_predictive_fold_corridor_metrics.csv",
        "diagnostic_2026_audit.csv",
        "diagnostic_2026_fold_corridor_metrics.csv",
        "diagnostic_2026_metrics.csv",
        "diagnostic_2026_predictions.csv",
        "diagnostic_2026_predictive_fold_corridor_metrics.csv",
        "fast_slow_confirmation_events.csv",
        "fast_slow_confirmation_summary.csv",
        "feature_coefficients.csv",
        "feature_coverage.csv",
        "feature_family_audit_h5.csv",
        "moex_lag_ladder_h5.csv",
        "posthoc_adaptive_2026_metrics.csv",
        "posthoc_adaptive_development_metrics.csv",
        "posthoc_adaptive_fold_corridor_metrics.csv",
        "posthoc_adaptive_predictions.csv",
        "posthoc_joint_date_shift_attribution.csv",
        "posthoc_univariate_rank_diagnostics_h5.csv",
        "retrained_source_delay_placebo_h5.csv",
        "selection.json",
        "target_date_dependence_summary_h5.csv",
        "target_materiality_sensitivity_h5.csv",
        "target_pairwise_correlations_h5.csv",
    }
    actual_outputs = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_outputs != expected_outputs:
        missing = sorted(expected_outputs - actual_outputs)
        unexpected = sorted(actual_outputs - expected_outputs)
        raise RuntimeError(
            f"output contract mismatch; missing={missing}, unexpected={unexpected}"
        )
    result_paths = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "experiment_manifest.json"
    )
    final_code_hashes = {
        str(path.relative_to(repo_root)): sha256(path) for path in code_paths
    }
    final_input_hashes = {
        str(path.relative_to(repo_root)): sha256(path) for path in input_paths
    }
    if final_code_hashes != initial_code_hashes or final_input_hashes != initial_input_hashes:
        raise RuntimeError(
            "code or inputs changed during execution; refusing to publish a success manifest"
        )
    manifest = {
        "seed": SEED,
        "run_tier": args.run_tier,
        "inferential_status": selection["inferential_status"],
        "bootstrap_repetitions": args.bootstrap_reps,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "code": initial_code_hashes,
        "inputs": initial_input_hashes,
        "outputs": {str(path.relative_to(output_dir)): sha256(path) for path in result_paths},
        "selection": selection,
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    write_json(
        output_dir / "_SUCCESS.json",
        {
            "status": (
                "complete" if args.run_tier == "final" else "smoke_complete_not_for_selection"
            ),
            "run_tier": args.run_tier,
            "experiment_manifest_sha256": sha256(output_dir / "experiment_manifest.json"),
        },
    )
    print(
        json.dumps(
            {
                "development_rows": len(development),
                "diagnostic_tracks": diagnostic_tracks,
                "prospective_shadow_tracks": prospective_shadow_tracks,
                "diagnostic_2026_rows": len(diagnostic),
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
