#!/usr/bin/env python3
"""Independent, offline ML/statistical audit for AlphaTransfer.

The script intentionally does not fetch data and does not import project code.  It
uses only the tracked snapshots so that the review remains independent from the
implementation under review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20_260_903
PORTFOLIO_TIE_BREAK_RULE = "sha256(seed|date|corridor), lowest digest wins"
HORIZONS = (1, 3, 5, 10, 20)
TEST_YEARS = (2023, 2024, 2025, 2026)
TRAIN_WINDOW_YEARS = 2
COOLDOWN_SESSIONS = 3
TARGET_CANDIDATE_SIGNALS_PER_CORRIDOR_WEEK = 1.0
CANDIDATE_CADENCE_BAND = (1.0, 2.0)
FALSE_POSITIVE_COST = 3.0
# Backward-compatible alias for downstream research helpers.  This target is for
# the corridor-level candidate stream, not the final client communication load.
TARGET_SIGNALS_PER_WEEK = TARGET_CANDIDATE_SIGNALS_PER_CORRIDOR_WEEK
THRESHOLD_GRID_SIZE = 151


CORE_FEATURES = [
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "ret60",
    "pr20",
    "pr60",
    "pr120",
    "pr252",
]
LEVEL_FEATURES = [f"{prefix}{window}" for prefix in ("dmin", "dmax") for window in (20, 60, 120, 252)]
VOL_FEATURES = ["vol5", "vol20", "vol60", "volratio"]
CALENDAR_FEATURES = ["dow", "dom", "month"]
COMMON_FEATURES = ["common_ret1", "common_ret5", "resid1", "resid5"]

FEATURE_GROUPS = {
    "core10": CORE_FEATURES,
    "core_levels": CORE_FEATURES + LEVEL_FEATURES,
    "core_vol": CORE_FEATURES + VOL_FEATURES,
    "core_vol_calendar": CORE_FEATURES + VOL_FEATURES + CALENDAR_FEATURES,
    "full": CORE_FEATURES
    + LEVEL_FEATURES
    + VOL_FEATURES
    + CALENDAR_FEATURES
    + COMMON_FEATURES
    + ["accel", "reversal1"],
}


@dataclass(frozen=True)
class Experiment:
    config_id: str
    model_kind: str
    feature_set: str


@dataclass(frozen=True)
class ProbabilityCalibrator:
    model: LogisticRegression | None
    method: str
    status: str
    intercept: float = math.nan
    slope: float = math.nan
    raw_validation_brier: float = math.nan
    attempted_platt_validation_brier: float = math.nan
    applied_validation_brier: float = math.nan
    calibrated_validation_brier: float = math.nan


EXPERIMENTS = (
    Experiment("weekly_first", "weekly_first", "none"),
    Experiment("down3", "down3", "none"),
    Experiment("level_pr120", "level_score", "pr120"),
    Experiment("logit_core10", "logistic", "core10"),
    Experiment("logit_core_levels", "logistic", "core_levels"),
    Experiment("logit_core_vol", "logistic", "core_vol"),
    Experiment("logit_core_vol_calendar", "logistic", "core_vol_calendar"),
    Experiment("logit_full", "logistic", "full"),
    Experiment("hgb_core_vol", "hist_gradient_boosting", "core_vol"),
)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=script_path.parent)
    parser.add_argument("--bootstrap-reps", type=int, default=20_000)
    parser.add_argument("--null-reps", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_rank(values: np.ndarray) -> float:
    current = values[-1]
    return float((np.sum(values < current) + 0.5 * np.sum(np.isclose(values, current))) / len(values))


def build_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "corridor", "rub_per_unit"}
    if missing := required - set(panel.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if panel.duplicated(["corridor", "date"]).any():
        raise ValueError("duplicate corridor/date rows")
    if (panel["rub_per_unit"] <= 0).any():
        raise ValueError("non-positive rates")

    parts: list[pd.DataFrame] = []
    for _, group in panel.sort_values(["corridor", "date"]).groupby("corridor"):
        group = group.copy()
        log_rate = np.log(group["rub_per_unit"])
        for lag in (1, 2, 3, 5, 10, 20, 60):
            group[f"ret{lag}"] = log_rate.diff(lag)
        for window in (5, 20, 60):
            group[f"vol{window}"] = group["ret1"].rolling(window).std()
        for window in (20, 60, 120, 252):
            rolling = group["rub_per_unit"].rolling(window, min_periods=window)
            group[f"pr{window}"] = rolling.apply(percentile_rank, raw=True)
            group[f"dmin{window}"] = np.log(group["rub_per_unit"] / rolling.min())
            group[f"dmax{window}"] = np.log(group["rub_per_unit"] / rolling.max())
        group["volratio"] = group["vol5"] / group["vol60"]
        group["accel"] = group["ret3"] - 0.3 * group["ret10"]
        group["reversal1"] = group["ret1"] - group["ret1"].shift(1)
        group["down3_candidate"] = (
            group["ret1"].lt(0) & group["ret1"].shift(1).lt(0) & group["ret1"].shift(2).lt(0)
        )
        group["session_ordinal"] = np.arange(len(group))
        group["dow"] = group["date"].dt.weekday
        group["dom"] = group["date"].dt.day
        group["month"] = group["date"].dt.month
        parts.append(group)

    result = pd.concat(parts).sort_values(["date", "corridor"]).reset_index(drop=True)
    result["common_ret1"] = result.groupby("date")["ret1"].transform("median")
    result["common_ret5"] = result.groupby("date")["ret5"].transform("median")
    result["resid1"] = result["ret1"] - result["common_ret1"]
    result["resid5"] = result["ret5"] - result["common_ret5"]
    return result


def add_target(
    panel: pd.DataFrame,
    horizon: int,
    materiality_bps: float = 0.0,
) -> pd.DataFrame:
    result = panel.copy()
    for name in ("target", "forward_bps", "symmetric_bps", "regret_bps"):
        result[name] = np.nan
    for _, raw_indexes in result.groupby("corridor").groups.items():
        indexes = np.asarray(sorted(raw_indexes))
        rates = result.loc[indexes, "rub_per_unit"].to_numpy(float)
        target_values = np.full(len(indexes), np.nan)
        forward_values = np.full(len(indexes), np.nan)
        symmetric_values = np.full(len(indexes), np.nan)
        regret_values = np.full(len(indexes), np.nan)
        for position in range(len(indexes) - horizon):
            future = rates[position + 1 : position + horizon + 1]
            current = rates[position]
            minimum_advantage_bps = (future.min() / current - 1.0) * 10_000.0
            target_values[position] = float(minimum_advantage_bps + 1e-12 >= materiality_bps)
            forward_values[position] = (future.mean() / current - 1.0) * 10_000.0
            regret_values[position] = (current / min(current, future.min()) - 1.0) * 10_000.0
            if position >= horizon:
                surrounding = rates[position - horizon : position + horizon + 1]
                symmetric_values[position] = (surrounding.mean() / current - 1.0) * 10_000.0
        result.loc[indexes, ["target", "forward_bps", "symmetric_bps", "regret_bps"]] = np.column_stack(
            [target_values, forward_values, symmetric_values, regret_values]
        )
    return result


def purge_tail(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Remove labels whose forward window can cross the next split boundary."""
    to_drop = frame.sort_values("date").groupby("corridor").tail(horizon).index
    return frame.drop(to_drop).copy()


def split_for_year(panel: pd.DataFrame, horizon: int, test_year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = panel[panel["target"].notna()]
    train_start = pd.Timestamp(test_year - TRAIN_WINDOW_YEARS - 1, 1, 1)
    validation_start = pd.Timestamp(test_year - 1, 1, 1)
    test_start = pd.Timestamp(test_year, 1, 1)
    test_end = pd.Timestamp(test_year + 1, 1, 1)
    train = eligible[(eligible["date"] >= train_start) & (eligible["date"] < validation_start)]
    validation = eligible[(eligible["date"] >= validation_start) & (eligible["date"] < test_start)]
    test = eligible[(eligible["date"] >= test_start) & (eligible["date"] < test_end)].copy()
    if test_end <= panel["date"].max():
        test = purge_tail(test, horizon)
    return purge_tail(train, horizon), purge_tail(validation, horizon), test


def make_model(kind: str, features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            ),
            ("corridor", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["corridor"]),
        ]
    )
    if kind == "logistic":
        classifier = LogisticRegression(C=0.1, max_iter=2_000)
    elif kind in {
        "hist_gradient_boosting",
        "hist_gradient_boosting_stump",
        "hist_gradient_boosting_leaf100",
    }:
        max_depth = 1 if kind == "hist_gradient_boosting_stump" else 2
        min_samples_leaf = 100 if kind != "hist_gradient_boosting" else 40
        classifier = HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=max_depth,
            learning_rate=0.05,
            l2_regularization=2.0,
            min_samples_leaf=min_samples_leaf,
            random_state=SEED,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def fit_platt_calibrator(probabilities: np.ndarray, labels: pd.Series) -> ProbabilityCalibrator:
    label_values = labels.astype(int).to_numpy()
    raw_validation_brier = float(np.mean((probabilities - label_values) ** 2))
    if labels.nunique() < 2:
        return ProbabilityCalibrator(
            model=None,
            method="identity",
            status="identity_fallback_single_class_validation",
            raw_validation_brier=raw_validation_brier,
            attempted_platt_validation_brier=math.nan,
            applied_validation_brier=raw_validation_brier,
            calibrated_validation_brier=raw_validation_brier,
        )
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibrator.fit(logits, label_values)
    intercept = float(calibrator.intercept_[0])
    slope = float(calibrator.coef_[0, 0])
    calibrated = calibrator.predict_proba(logits)[:, 1]
    calibrated_validation_brier = float(np.mean((calibrated - label_values) ** 2))
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0.0:
        return ProbabilityCalibrator(
            model=None,
            method="identity",
            status="identity_fallback_non_monotone_platt",
            intercept=intercept,
            slope=slope,
            raw_validation_brier=raw_validation_brier,
            attempted_platt_validation_brier=calibrated_validation_brier,
            applied_validation_brier=raw_validation_brier,
            calibrated_validation_brier=raw_validation_brier,
        )
    if calibrated_validation_brier >= raw_validation_brier - 1e-12:
        return ProbabilityCalibrator(
            model=None,
            method="identity",
            status="identity_fallback_no_validation_brier_gain",
            intercept=intercept,
            slope=slope,
            raw_validation_brier=raw_validation_brier,
            attempted_platt_validation_brier=calibrated_validation_brier,
            applied_validation_brier=raw_validation_brier,
            calibrated_validation_brier=raw_validation_brier,
        )
    return ProbabilityCalibrator(
        model=calibrator,
        method="prior_year_monotone_platt",
        status="fitted",
        intercept=intercept,
        slope=slope,
        raw_validation_brier=raw_validation_brier,
        attempted_platt_validation_brier=calibrated_validation_brier,
        applied_validation_brier=calibrated_validation_brier,
        calibrated_validation_brier=calibrated_validation_brier,
    )


def apply_platt(calibrator: ProbabilityCalibrator, probabilities: np.ndarray) -> np.ndarray:
    if calibrator.model is None:
        return probabilities
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.model.predict_proba(logits)[:, 1]


def select_with_cooldown(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    initial_last_session: dict[str, int] | None = None,
) -> np.ndarray:
    best_candidates = portfolio_best_candidates(frame, scores)
    return select_best_candidates(best_candidates, threshold, initial_last_session)


def select_per_corridor_with_cooldown(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float | dict[str, float],
    initial_last_session: dict[str, int] | None = None,
) -> np.ndarray:
    """Build the diagnostic candidate stream before the shared client budget."""
    score_by_index = pd.Series(scores, index=frame.index)
    last_sessions = dict(initial_last_session or {})
    selected: list[int] = []
    for corridor, group in frame.sort_values(["corridor", "date"]).groupby("corridor", sort=False):
        indexes = group.index.to_numpy(dtype=int)
        sessions = group["session_ordinal"].to_numpy(dtype=int)
        group_scores = score_by_index.loc[indexes].to_numpy(dtype=float)
        last_session = last_sessions.get(str(corridor), -10_000)
        corridor_threshold = (
            float(threshold[str(corridor)])
            if isinstance(threshold, dict)
            else float(threshold)
        )
        for position in range(len(indexes)):
            score = group_scores[position]
            session = sessions[position]
            if (
                np.isfinite(score)
                and score >= corridor_threshold
                and session - last_session > COOLDOWN_SESSIONS
            ):
                selected.append(int(indexes[position]))
                last_session = int(session)
        last_sessions[str(corridor)] = last_session
    if not selected:
        return np.asarray([], dtype=int)
    return (
        frame.loc[selected]
        .sort_values(["date", "corridor"])
        .index.to_numpy(dtype=int)
    )


def select_portfolio_from_candidates(
    frame: pd.DataFrame,
    scores: np.ndarray,
    candidate_selected: Iterable[int],
    initial_last_session: dict[str, int] | None = None,
) -> np.ndarray:
    """Apply one shared client contact stream to already eligible candidates."""
    candidate_index = pd.Index(candidate_selected, dtype=int)
    if candidate_index.empty:
        return np.asarray([], dtype=int)
    score_by_index = pd.Series(scores, index=frame.index)
    candidate_frame = frame.loc[candidate_index]
    candidate_scores = score_by_index.loc[candidate_index].to_numpy(dtype=float)
    best_candidates = portfolio_best_candidates(candidate_frame, candidate_scores)
    return select_best_candidates(best_candidates, -math.inf, initial_last_session)


def portfolio_tie_priority(date: pd.Timestamp, corridor: str) -> int:
    """Return a deterministic, corridor-neutral priority for an exact score tie."""
    payload = f"{SEED}|{pd.Timestamp(date).date().isoformat()}|{corridor}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def portfolio_best_candidates(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    work = frame[["date", "corridor", "session_ordinal"]].copy()
    work["row_index"] = frame.index
    work["score"] = scores
    work = work[work["score"].notna()]
    if work.empty:
        return work.assign(tie_count=pd.Series(dtype=int), tie_rule=PORTFOLIO_TIE_BREAK_RULE)
    maximum_score = work.groupby("date")["score"].transform("max")
    leaders = work[work["score"].eq(maximum_score)].copy()
    leaders["tie_count"] = leaders.groupby("date")["score"].transform("size")
    leaders["tie_priority"] = [
        portfolio_tie_priority(date, str(corridor))
        for date, corridor in zip(leaders["date"], leaders["corridor"])
    ]
    leaders["tie_rule"] = PORTFOLIO_TIE_BREAK_RULE
    return (
        leaders.sort_values(["date", "tie_priority", "corridor", "row_index"])
        .drop_duplicates("date", keep="first")
        .sort_values("date")
    )


def select_best_candidates(
    best_candidates: pd.DataFrame,
    threshold: float,
    initial_last_session: dict[str, int] | None = None,
) -> np.ndarray:
    selected: list[int] = []
    last_session = (initial_last_session or {}).get("__portfolio__", -10_000)
    for row in best_candidates.itertuples(index=False):
        row_index = int(row.row_index)
        score = float(row.score)
        session = int(row.session_ordinal)
        if score >= threshold and session - last_session > COOLDOWN_SESSIONS:
            selected.append(row_index)
            last_session = session
    return np.asarray(selected, dtype=int)


def selection_state(
    frame: pd.DataFrame,
    selected: Iterable[int],
    initial_last_session: dict[str, int] | None = None,
) -> dict[str, int]:
    state = dict(initial_last_session or {})
    selected_set = set(selected)
    chosen_sessions = [
        int(frame.loc[index, "session_ordinal"])
        for index in selected_set
        if index in frame.index
    ]
    if chosen_sessions:
        state["__portfolio__"] = max(chosen_sessions)
    return state


def corridor_selection_state(
    frame: pd.DataFrame,
    selected: Iterable[int],
    initial_last_session: dict[str, int] | None = None,
) -> dict[str, int]:
    state = dict(initial_last_session or {})
    for index in selected:
        if index not in frame.index:
            continue
        corridor = str(frame.loc[index, "corridor"])
        session = int(frame.loc[index, "session_ordinal"])
        state[corridor] = max(state.get(corridor, -10_000), session)
    return state


def weekly_first(frame: pd.DataFrame) -> np.ndarray:
    first_date_rows = portfolio_best_candidates(frame, np.zeros(len(frame)))
    weeks = first_date_rows["date"].dt.to_period("W-SUN")
    return first_date_rows.index[~weeks.duplicated()].to_numpy(dtype=int)


def weekly_first_per_corridor(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(["corridor", "date"])
    work = ordered.assign(week=ordered["date"].dt.to_period("W-SUN"))
    return work.drop_duplicates(["corridor", "week"], keep="first").index.to_numpy(dtype=int)


def down3(frame: pd.DataFrame, initial_last_session: dict[str, int] | None = None) -> np.ndarray:
    scores = frame["down3_candidate"].astype(float).to_numpy()
    return select_with_cooldown(frame, scores, 1.0, initial_last_session)


def mean_frequency(frame: pd.DataFrame, selected: Iterable[int]) -> float:
    selected_set = set(selected)
    span_days = max(1, (frame["date"].max() - frame["date"].min()).days + 1)
    return float(sum(index in selected_set for index in frame.index) / span_days * 7.0)


def mean_corridor_frequency(frame: pd.DataFrame, selected: Iterable[int]) -> float:
    selected_set = set(selected)
    frequencies: list[float] = []
    for _, group in frame.groupby("corridor"):
        span_days = max(1, (group["date"].max() - group["date"].min()).days + 1)
        count = sum(index in selected_set for index in group.index)
        frequencies.append(count / span_days * 7.0)
    return float(np.mean(frequencies)) if frequencies else math.nan


def candidate_weighted_error(
    frame: pd.DataFrame,
    selected: Iterable[int],
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> float:
    selected_mask = frame.index.isin(set(selected))
    labels = frame["target"].astype(bool).to_numpy()
    false_positives = int((selected_mask & ~labels).sum())
    false_negatives = int((~selected_mask & labels).sum())
    return float((false_positive_cost * false_positives + false_negatives) / len(frame))


def cadence_diagnostics(frame: pd.DataFrame, selected: Iterable[int]) -> dict[str, float]:
    selected_set = set(selected)
    weeks = frame["date"].dt.to_period("W-SUN")
    candidate_weeks = pd.period_range(weeks.min(), weeks.max(), freq="W-SUN")
    first_date = frame["date"].min().normalize()
    last_date = frame["date"].max().normalize()
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
    selected_dates = frame.loc[frame.index.isin(selected_set), "date"].sort_values()
    weekly_counts = (
        selected_dates.dt.to_period("W-SUN")
        .value_counts()
        .reindex(all_weeks, fill_value=0)
    )
    boundary_dates = pd.DatetimeIndex(
        [frame["date"].min(), *selected_dates.tolist(), frame["date"].max()]
    )
    gaps = np.asarray(
        [
            (right - left) / pd.Timedelta(days=1)
            for left, right in zip(boundary_dates[:-1], boundary_dates[1:])
        ],
        dtype=float,
    )
    return {
        "eligible_weeks": float(len(all_weeks)),
        "partial_boundary_weeks_excluded": float(len(candidate_weeks) - len(all_weeks)),
        "silent_week_share": float(weekly_counts.eq(0).mean()),
        "weeks_with_1_to_2_signals_share": float(weekly_counts.between(1, 2).mean()),
        "weeks_above_2_signals_share": float(weekly_counts.gt(2).mean()),
        "maximum_calendar_gap_days_including_boundaries": (
            float(gaps.max()) if len(gaps) else 0.0
        ),
        "median_calendar_gap_days_including_boundaries": (
            float(np.median(gaps)) if len(gaps) else 0.0
        ),
        "p90_calendar_gap_days_including_boundaries": (
            float(np.quantile(gaps, 0.90)) if len(gaps) else 0.0
        ),
    }


def _choose_single_corridor_frequency_threshold(
    validation: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[float, float, float]:
    score_values = np.asarray(scores, dtype=float)
    finite = score_values[np.isfinite(score_values)]
    if not len(finite):
        raise ValueError("no finite validation scores")
    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, THRESHOLD_GRID_SIZE)))
    ordered = validation[["corridor", "date", "session_ordinal", "target"]].copy()
    ordered["score"] = pd.Series(score_values, index=validation.index)
    ordered = ordered.sort_values(["corridor", "date"])
    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for _, group in ordered.groupby("corridor", sort=False):
        span_days = max(1, (group["date"].max() - group["date"].min()).days + 1)
        prepared.append(
            (
                group["session_ordinal"].to_numpy(dtype=int),
                group["score"].to_numpy(dtype=float),
                group["target"].astype(bool).to_numpy(),
                float(span_days),
            )
        )
    total_positives = int(validation["target"].astype(bool).sum())
    choices: list[tuple[float, ...]] = []
    for threshold in candidates:
        selected_count = 0
        selected_hits = 0
        corridor_frequencies: list[float] = []
        for sessions, group_scores, labels, span_days in prepared:
            last_session = -10_000
            corridor_count = 0
            for position in range(len(sessions)):
                score = group_scores[position]
                session = sessions[position]
                if (
                    np.isfinite(score)
                    and score >= threshold
                    and session - last_session > COOLDOWN_SESSIONS
                ):
                    corridor_count += 1
                    selected_hits += int(labels[position])
                    last_session = int(session)
            selected_count += corridor_count
            corridor_frequencies.append(corridor_count / span_days * 7.0)
        frequency = float(np.mean(corridor_frequencies))
        false_positives = selected_count - selected_hits
        false_negatives = total_positives - selected_hits
        weighted_error = float(
            (FALSE_POSITIVE_COST * false_positives + false_negatives) / len(validation)
        )
        below = max(0.0, CANDIDATE_CADENCE_BAND[0] - frequency)
        above = max(0.0, frequency - CANDIDATE_CADENCE_BAND[1])
        cadence_distance = below + above
        # Inside the preregistered cadence band, optimize the validation-only
        # asymmetric loss.  Ties prefer the lower communication load.
        choices.append(
            (
                float(cadence_distance > 0.0),
                cadence_distance,
                weighted_error,
                frequency,
                -float(threshold),
                float(threshold),
                frequency,
                weighted_error,
            )
        )
    *_, threshold, frequency, weighted_error = min(choices)
    return threshold, frequency, weighted_error


def choose_frequency_threshold(
    validation: pd.DataFrame,
    scores: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    """Tune an independent threshold for each corridor on the purged validation fold."""
    score_by_index = pd.Series(np.asarray(scores, dtype=float), index=validation.index)
    thresholds: dict[str, float] = {}
    for corridor, group in validation.groupby("corridor", sort=True):
        threshold, _, _ = _choose_single_corridor_frequency_threshold(
            group,
            score_by_index.loc[group.index].to_numpy(dtype=float),
        )
        thresholds[str(corridor)] = threshold
    selected = select_per_corridor_with_cooldown(validation, scores, thresholds)
    return (
        thresholds,
        mean_corridor_frequency(validation, selected),
        candidate_weighted_error(validation, selected),
    )


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return centre - half, centre + half


def corridor_candidate_diagnostics(
    frame: pd.DataFrame,
    selected: Iterable[int],
) -> dict[str, dict[str, float]]:
    selected_set = set(selected)
    result: dict[str, dict[str, float]] = {}
    for corridor, group in frame.groupby("corridor"):
        corridor_selected = [index for index in group.index if index in selected_set]
        span_days = max(1, (group["date"].max() - group["date"].min()).days + 1)
        cadence = cadence_diagnostics(group, corridor_selected)
        result[str(corridor)] = {
            "signals_per_week": len(corridor_selected) / span_days * 7.0,
            "weighted_error_3": candidate_weighted_error(group, corridor_selected),
            "weeks_with_1_to_2_signals_share": cadence[
                "weeks_with_1_to_2_signals_share"
            ],
            "silent_week_share": cadence["silent_week_share"],
        }
    return result


def metric_rows(
    experiment: Experiment,
    horizon: int,
    test_year: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    selected: np.ndarray,
    candidate_selected: np.ndarray,
    probabilities: np.ndarray | None,
    threshold: float | dict[str, float] | None,
    validation_candidate_frequency: float,
    validation_portfolio_frequency: float,
    validation_weighted_error_3: float,
    threshold_selected_with_labels: bool,
    validation_corridor_diagnostics: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    selected_set = set(selected)
    candidate_selected_set = set(candidate_selected)
    portfolio_span_days = max(1, (test["date"].max() - test["date"].min()).days + 1)
    portfolio_frequency = len(selected_set) / portfolio_span_days * 7.0
    portfolio_cadence = cadence_diagnostics(test, selected)
    rows: list[dict[str, object]] = []
    for corridor, group in test.groupby("corridor"):
        corridor_validation = (validation_corridor_diagnostics or {}).get(
            str(corridor), {}
        )
        validation_corridor = validation[validation["corridor"].eq(corridor)]
        prior_prevalence = float(validation_corridor["target"].mean())
        signal = group[group.index.map(selected_set.__contains__)]
        candidate_signal = group[group.index.map(candidate_selected_set.__contains__)]
        eligible_count = len(group)
        baseline_hits = int(group["target"].sum())
        signal_count = len(signal)
        signal_hits = int(signal["target"].sum())
        candidate_signal_count = len(candidate_signal)
        candidate_signal_hits = int(candidate_signal["target"].sum())
        candidate_false_positives = candidate_signal_count - candidate_signal_hits
        candidate_false_negatives = baseline_hits - candidate_signal_hits
        candidate_cadence = cadence_diagnostics(group, candidate_selected)
        span_days = max(1, (group["date"].max() - group["date"].min()).days + 1)
        hit_rate = signal_hits / signal_count if signal_count else math.nan
        baseline_rate = baseline_hits / eligible_count
        low, high = wilson(signal_hits, signal_count)
        row_probability = None
        model_brier = prior_brier = test_climatology_brier = math.nan
        if probabilities is not None:
            positions = test.index.get_indexer(group.index)
            row_probability = probabilities[positions]
            labels = group["target"].to_numpy(float)
            model_brier = float(np.mean((row_probability - labels) ** 2))
            prior_brier = float(np.mean((prior_prevalence - labels) ** 2))
            test_climatology_brier = float(np.mean((baseline_rate - labels) ** 2))
        rows.append(
            {
                "config_id": experiment.config_id,
                "model_kind": experiment.model_kind,
                "feature_set": experiment.feature_set,
                "horizon_cbr_rows_pub_proxy": horizon,
                "fold_test_year": test_year,
                "corridor": corridor,
                "train_start": train["date"].min().date().isoformat(),
                "train_end_inclusive": train["date"].max().date().isoformat(),
                "validation_start": validation["date"].min().date().isoformat(),
                "validation_end_inclusive_after_purge": validation["date"].max().date().isoformat(),
                "test_start": group["date"].min().date().isoformat(),
                "test_end_inclusive": group["date"].max().date().isoformat(),
                "purge_cbr_rows_pub_proxy": horizon,
                "threshold": (
                    threshold.get(str(corridor)) if isinstance(threshold, dict) else threshold
                ),
                "threshold_scope": (
                    "per_corridor" if isinstance(threshold, dict) else "shared_or_rule"
                ),
                "threshold_selected_with_labels": threshold_selected_with_labels,
                "validation_candidate_mean_signals_per_corridor_week": validation_candidate_frequency,
                "validation_portfolio_signals_per_week": validation_portfolio_frequency,
                "validation_candidate_weighted_error_3": validation_weighted_error_3,
                "validation_corridor_candidate_signals_per_week": corridor_validation.get(
                    "signals_per_week", math.nan
                ),
                "validation_corridor_candidate_weighted_error_3": corridor_validation.get(
                    "weighted_error_3", math.nan
                ),
                "validation_corridor_candidate_weeks_with_1_to_2_signals_share": (
                    corridor_validation.get("weeks_with_1_to_2_signals_share", math.nan)
                ),
                "validation_corridor_candidate_silent_week_share": corridor_validation.get(
                    "silent_week_share", math.nan
                ),
                "eligible_count": eligible_count,
                "baseline_hits": baseline_hits,
                "baseline_hit_rate": baseline_rate,
                "signal_count": signal_count,
                "signal_hits": signal_hits,
                "hit_rate": hit_rate,
                "hit_naive_iid_wilson95_low": low,
                "hit_naive_iid_wilson95_high": high,
                "hit_interval_note": (
                    "diagnostic_only_iid_interval; use date-grouped month-block uncertainty for claims"
                ),
                "lift": hit_rate / baseline_rate if signal_count and baseline_rate else math.nan,
                "signals_per_week": signal_count / span_days * 7.0,
                "candidate_signal_count": candidate_signal_count,
                "candidate_signal_hits": candidate_signal_hits,
                "candidate_hit_rate": candidate_signal["target"].mean(),
                "candidate_lift": (
                    candidate_signal_hits / candidate_signal_count / baseline_rate
                    if candidate_signal_count and baseline_rate
                    else math.nan
                ),
                "candidate_false_positives": candidate_false_positives,
                "candidate_false_negatives": candidate_false_negatives,
                "candidate_weighted_error_1": (
                    candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_2": (
                    2 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_3": (
                    3 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_5": (
                    5 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_signals_per_corridor_week": len(candidate_signal) / span_days * 7.0,
                "candidate_silent_week_share": candidate_cadence["silent_week_share"],
                "candidate_weeks_with_1_to_2_signals_share": candidate_cadence[
                    "weeks_with_1_to_2_signals_share"
                ],
                "candidate_weeks_above_2_signals_share": candidate_cadence[
                    "weeks_above_2_signals_share"
                ],
                "candidate_maximum_calendar_gap_days": candidate_cadence[
                    "maximum_calendar_gap_days_including_boundaries"
                ],
                "portfolio_signals_per_week": portfolio_frequency,
                "portfolio_silent_week_share": portfolio_cadence["silent_week_share"],
                "portfolio_weeks_with_1_to_2_signals_share": portfolio_cadence[
                    "weeks_with_1_to_2_signals_share"
                ],
                "portfolio_weeks_above_2_signals_share": portfolio_cadence[
                    "weeks_above_2_signals_share"
                ],
                "portfolio_maximum_calendar_gap_days": portfolio_cadence[
                    "maximum_calendar_gap_days_including_boundaries"
                ],
                "candidate_signal_symmetric_bps": candidate_signal["symmetric_bps"].mean(),
                "candidate_signal_forward_bps": candidate_signal["forward_bps"].mean(),
                "candidate_signal_regret_bps": candidate_signal["regret_bps"].mean(),
                "signal_symmetric_bps": signal["symmetric_bps"].mean(),
                "baseline_symmetric_bps": group["symmetric_bps"].mean(),
                "signal_forward_bps": signal["forward_bps"].mean(),
                "baseline_forward_bps": group["forward_bps"].mean(),
                "signal_regret_bps": signal["regret_bps"].mean(),
                "baseline_regret_bps": group["regret_bps"].mean(),
                "prior_year_prevalence": prior_prevalence,
                "model_brier": model_brier,
                "prior_year_prevalence_brier": prior_brier,
                "oracle_test_climatology_brier": test_climatology_brier,
            }
        )
    return rows


def run_fold(
    experiment: Experiment,
    horizon: int,
    test_year: int,
    panel: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    train, validation, test = split_for_year(panel, horizon, test_year)
    validation_history = panel[
        (panel["date"] >= pd.Timestamp(test_year - 1, 1, 1))
        & (panel["date"] < pd.Timestamp(test_year, 1, 1))
    ].copy()
    probabilities: np.ndarray | None = None
    raw_probabilities: np.ndarray | None = None
    policy_scores = np.full(len(test), np.nan)
    calibration_method = "none"
    calibration_status = "not_applicable"
    platt_intercept = platt_slope = math.nan
    raw_validation_brier = attempted_platt_validation_brier = math.nan
    applied_validation_brier = calibrated_validation_brier = math.nan
    validation_candidate_details: dict[str, dict[str, float]] = {}
    coefficient_rows: list[dict[str, object]] = []

    if experiment.model_kind in {
        "logistic",
        "hist_gradient_boosting",
        "hist_gradient_boosting_stump",
        "hist_gradient_boosting_leaf100",
    }:
        features = FEATURE_GROUPS[experiment.feature_set]
        model = make_model(experiment.model_kind, features)
        model.fit(train[features + ["corridor"]], train["target"].astype(int))
        validation_raw = model.predict_proba(validation[features + ["corridor"]])[:, 1]
        calibrator = fit_platt_calibrator(validation_raw, validation["target"])
        calibration_method = calibrator.method
        calibration_status = calibrator.status
        platt_intercept = calibrator.intercept
        platt_slope = calibrator.slope
        raw_validation_brier = calibrator.raw_validation_brier
        attempted_platt_validation_brier = calibrator.attempted_platt_validation_brier
        applied_validation_brier = calibrator.applied_validation_brier
        calibrated_validation_brier = calibrator.calibrated_validation_brier
        validation_history_raw = model.predict_proba(validation_history[features + ["corridor"]])[:, 1]
        validation_history_scores = apply_platt(calibrator, validation_history_raw)
        validation_scores = apply_platt(calibrator, validation_raw)
        raw_probabilities = model.predict_proba(test[features + ["corridor"]])[:, 1]
        probabilities = apply_platt(calibrator, raw_probabilities)
        policy_scores = probabilities
        # Labels in the purged tail are unavailable, but scores/cadence are PIT-safe.
        threshold, validation_candidate_frequency, validation_weighted_error_3 = choose_frequency_threshold(
            validation, validation_scores
        )
        validation_tuning_candidates = select_per_corridor_with_cooldown(
            validation, validation_scores, threshold
        )
        validation_candidate_details = corridor_candidate_diagnostics(
            validation, validation_tuning_candidates
        )
        validation_history_candidates = select_per_corridor_with_cooldown(
            validation_history, validation_history_scores, threshold
        )
        candidate_initial_state = corridor_selection_state(
            validation_history, validation_history_candidates
        )
        validation_history_selected = select_portfolio_from_candidates(
            validation_history, validation_history_scores, validation_history_candidates
        )
        portfolio_initial_state = selection_state(validation_history, validation_history_selected)
        candidate_selected = select_per_corridor_with_cooldown(
            test, probabilities, threshold, candidate_initial_state
        )
        selected = select_portfolio_from_candidates(
            test, probabilities, candidate_selected, portfolio_initial_state
        )
        validation_portfolio_frequency = mean_frequency(
            validation_history, validation_history_selected
        )
        threshold_selected_with_labels = True
        if experiment.model_kind == "logistic":
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            coefficients = model.named_steps["classifier"].coef_[0]
            coefficient_rows = [
                {
                    "config_id": experiment.config_id,
                    "horizon_cbr_rows_pub_proxy": horizon,
                    "fold_test_year": test_year,
                    "feature": name,
                    "standardized_coefficient": value,
                }
                for name, value in zip(feature_names, coefficients)
            ]
    elif experiment.model_kind == "level_score":
        validation_history_scores = (1.0 - validation_history["pr120"]).fillna(-np.inf).to_numpy()
        validation_scores = (1.0 - validation["pr120"]).fillna(-np.inf).to_numpy()
        test_scores = (1.0 - test["pr120"]).fillna(-np.inf).to_numpy()
        policy_scores = test_scores
        threshold, validation_candidate_frequency, validation_weighted_error_3 = choose_frequency_threshold(
            validation, validation_scores
        )
        validation_tuning_candidates = select_per_corridor_with_cooldown(
            validation, validation_scores, threshold
        )
        validation_candidate_details = corridor_candidate_diagnostics(
            validation, validation_tuning_candidates
        )
        validation_history_candidates = select_per_corridor_with_cooldown(
            validation_history, validation_history_scores, threshold
        )
        candidate_initial_state = corridor_selection_state(
            validation_history, validation_history_candidates
        )
        validation_history_selected = select_portfolio_from_candidates(
            validation_history, validation_history_scores, validation_history_candidates
        )
        portfolio_initial_state = selection_state(validation_history, validation_history_selected)
        candidate_selected = select_per_corridor_with_cooldown(
            test, test_scores, threshold, candidate_initial_state
        )
        selected = select_portfolio_from_candidates(
            test, test_scores, candidate_selected, portfolio_initial_state
        )
        validation_portfolio_frequency = mean_frequency(
            validation_history, validation_history_selected
        )
        threshold_selected_with_labels = True
    elif experiment.model_kind == "weekly_first":
        policy_scores = np.zeros(len(test))
        validation_history_candidates = weekly_first_per_corridor(validation_history)
        validation_history_selected = weekly_first(validation_history)
        candidate_selected = weekly_first_per_corridor(test)
        selected = weekly_first(test)
        threshold = None
        validation_candidate_frequency = mean_corridor_frequency(
            validation_history, validation_history_candidates
        )
        validation_weighted_error_3 = candidate_weighted_error(
            validation_history, validation_history_candidates
        )
        validation_candidate_details = corridor_candidate_diagnostics(
            validation_history, validation_history_candidates
        )
        validation_portfolio_frequency = mean_frequency(
            validation_history, validation_history_selected
        )
        threshold_selected_with_labels = False
    elif experiment.model_kind == "down3":
        validation_history_scores = validation_history["down3_candidate"].astype(float).to_numpy()
        test_scores = test["down3_candidate"].astype(float).to_numpy()
        policy_scores = test_scores
        validation_history_candidates = select_per_corridor_with_cooldown(
            validation_history, validation_history_scores, 1.0
        )
        candidate_initial_state = corridor_selection_state(
            validation_history, validation_history_candidates
        )
        validation_history_selected = select_portfolio_from_candidates(
            validation_history, validation_history_scores, validation_history_candidates
        )
        portfolio_initial_state = selection_state(validation_history, validation_history_selected)
        candidate_selected = select_per_corridor_with_cooldown(
            test, test_scores, 1.0, candidate_initial_state
        )
        selected = select_portfolio_from_candidates(
            test, test_scores, candidate_selected, portfolio_initial_state
        )
        threshold = 1.0
        validation_candidate_frequency = mean_corridor_frequency(
            validation_history, validation_history_candidates
        )
        validation_weighted_error_3 = candidate_weighted_error(
            validation_history, validation_history_candidates
        )
        validation_candidate_details = corridor_candidate_diagnostics(
            validation_history, validation_history_candidates
        )
        validation_portfolio_frequency = mean_frequency(
            validation_history, validation_history_selected
        )
        threshold_selected_with_labels = False
    else:
        raise ValueError(experiment.model_kind)

    test_output = test[["date", "corridor", "target", "forward_bps", "symmetric_bps", "regret_bps"]].copy()
    test_output["candidate_signal"] = test_output.index.isin(candidate_selected)
    test_output["signal"] = test_output.index.isin(selected)
    score_by_index = pd.Series(policy_scores, index=test.index)
    raw_threshold_candidate = pd.Series(False, index=test.index, dtype="boolean")
    if threshold is None:
        raw_threshold_candidate[:] = pd.NA
    else:
        for corridor, group in test.groupby("corridor"):
            corridor_threshold = (
                float(threshold[str(corridor)])
                if isinstance(threshold, dict)
                else float(threshold)
            )
            raw_threshold_candidate.loc[group.index] = (
                score_by_index.loc[group.index].ge(corridor_threshold)
                & score_by_index.loc[group.index].notna()
            )
    test_output["raw_threshold_candidate"] = raw_threshold_candidate
    test_output["corridor_cooldown_suppressed"] = test_output[
        "raw_threshold_candidate"
    ] & ~test_output["candidate_signal"]
    test_output["probability"] = probabilities if probabilities is not None else np.nan
    test_output["raw_probability"] = raw_probabilities if raw_probabilities is not None else np.nan
    test_output["calibration_method"] = calibration_method
    test_output["calibration_status"] = calibration_status
    test_output["platt_intercept"] = platt_intercept
    test_output["platt_slope"] = platt_slope
    test_output["raw_validation_brier"] = raw_validation_brier
    test_output["attempted_platt_validation_brier"] = attempted_platt_validation_brier
    test_output["applied_validation_brier"] = applied_validation_brier
    test_output["calibrated_validation_brier"] = calibrated_validation_brier
    candidate_mask = test.index.isin(candidate_selected)
    candidate_frame = test.loc[candidate_mask]
    candidate_scores = pd.Series(policy_scores, index=test.index).loc[candidate_frame.index]
    best_candidates = portfolio_best_candidates(
        candidate_frame,
        candidate_scores.to_numpy(dtype=float),
    )
    tie_count_by_row = best_candidates.set_index("row_index")["tie_count"]
    daily_top_indexes = set(best_candidates["row_index"].astype(int))
    test_output["portfolio_daily_rank_suppressed"] = (
        test_output["candidate_signal"] & ~test_output.index.isin(daily_top_indexes)
    )
    test_output["portfolio_shared_cooldown_suppressed"] = (
        test_output.index.isin(daily_top_indexes) & ~test_output["signal"]
    )
    test_output["portfolio_top_score_tie_count"] = (
        test_output.index.to_series().map(tie_count_by_row).fillna(0).astype(int)
    )
    test_output["portfolio_tie_break_rule"] = PORTFOLIO_TIE_BREAK_RULE
    test_output["fold_test_year"] = test_year
    test_output["config_id"] = experiment.config_id
    test_output["horizon_cbr_rows_pub_proxy"] = horizon
    rows = metric_rows(
        experiment,
        horizon,
        test_year,
        train,
        validation,
        test,
        selected,
        candidate_selected,
        probabilities,
        threshold,
        validation_candidate_frequency,
        validation_portfolio_frequency,
        validation_weighted_error_3,
        threshold_selected_with_labels,
        validation_candidate_details,
    )
    return rows, test_output, coefficient_rows


def aggregate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["config_id", "model_kind", "feature_set", "horizon_cbr_rows_pub_proxy"]
    for key, group in fold_metrics.groupby(keys, dropna=False):
        config_id, model_kind, feature_set, horizon = key
        eligible_count = int(group["eligible_count"].sum())
        baseline_hits = int(group["baseline_hits"].sum())
        signal_count = int(group["signal_count"].sum())
        signal_hits = int(group["signal_hits"].sum())
        candidate_signal_count = int(group["candidate_signal_count"].sum())
        candidate_signal_hits = int(group["candidate_signal_hits"].sum())
        candidate_false_positives = int(group["candidate_false_positives"].sum())
        candidate_false_negatives = int(group["candidate_false_negatives"].sum())
        baseline_rate = baseline_hits / eligible_count
        hit_rate = signal_hits / signal_count if signal_count else math.nan
        fold_counts = group.groupby("fold_test_year")[["eligible_count", "baseline_hits", "signal_count", "signal_hits"]].sum()
        fold_expected_hits = group.assign(
            expected_hits=group["signal_count"] * group["baseline_hit_rate"]
        ).groupby("fold_test_year")["expected_hits"].sum()
        fold_lifts = fold_counts["signal_hits"] / fold_expected_hits
        active_fold_count = int(fold_counts["signal_count"].gt(0).sum())
        candidate_fold_counts = group.groupby("fold_test_year")[
            ["candidate_signal_count", "candidate_signal_hits"]
        ].sum()
        candidate_fold_expected_hits = group.assign(
            expected_hits=group["candidate_signal_count"] * group["baseline_hit_rate"]
        ).groupby("fold_test_year")["expected_hits"].sum()
        candidate_fold_lifts = (
            candidate_fold_counts["candidate_signal_hits"] / candidate_fold_expected_hits
        )
        active_candidate_fold_count = int(
            candidate_fold_counts["candidate_signal_count"].gt(0).sum()
        )
        corridor_counts = group.groupby("corridor")[["eligible_count", "baseline_hits", "signal_count", "signal_hits"]].sum()
        corridor_expected_hits = group.assign(
            expected_hits=group["signal_count"] * group["baseline_hit_rate"]
        ).groupby("corridor")["expected_hits"].sum()
        corridor_lifts = corridor_counts["signal_hits"] / corridor_expected_hits
        active_corridor_count = int(corridor_counts["signal_count"].gt(0).sum())
        candidate_corridor_counts = group.groupby("corridor")[
            ["candidate_signal_count", "candidate_signal_hits"]
        ].sum()
        candidate_corridor_expected_hits = group.assign(
            expected_hits=group["candidate_signal_count"] * group["baseline_hit_rate"]
        ).groupby("corridor")["expected_hits"].sum()
        candidate_corridor_lifts = (
            candidate_corridor_counts["candidate_signal_hits"]
            / candidate_corridor_expected_hits
        )
        active_candidate_corridor_count = int(
            candidate_corridor_counts["candidate_signal_count"].gt(0).sum()
        )

        def weighted(column: str, weights: str) -> float:
            valid = group[column].notna() & group[weights].gt(0)
            return float(np.average(group.loc[valid, column], weights=group.loc[valid, weights])) if valid.any() else math.nan

        signal_symmetric = weighted("signal_symmetric_bps", "signal_count")
        baseline_symmetric = weighted("baseline_symmetric_bps", "eligible_count")
        signal_forward = weighted("signal_forward_bps", "signal_count")
        baseline_forward = weighted("baseline_forward_bps", "eligible_count")
        signal_regret = weighted("signal_regret_bps", "signal_count")
        candidate_signal_symmetric = weighted(
            "candidate_signal_symmetric_bps", "candidate_signal_count"
        )
        candidate_signal_forward = weighted(
            "candidate_signal_forward_bps", "candidate_signal_count"
        )
        candidate_signal_regret = weighted(
            "candidate_signal_regret_bps", "candidate_signal_count"
        )
        baseline_regret = weighted("baseline_regret_bps", "eligible_count")
        expected_signal_hits = float((group["signal_count"] * group["baseline_hit_rate"]).sum())
        expected_candidate_hits = float(
            (group["candidate_signal_count"] * group["baseline_hit_rate"]).sum()
        )
        standardized_baseline_rate = expected_signal_hits / signal_count if signal_count else math.nan
        standardized_lift = signal_hits / expected_signal_hits if expected_signal_hits else math.nan
        candidate_standardized_lift = (
            candidate_signal_hits / expected_candidate_hits
            if expected_candidate_hits
            else math.nan
        )
        signal_weighted_symmetric = weighted("baseline_symmetric_bps", "signal_count")
        signal_weighted_forward = weighted("baseline_forward_bps", "signal_count")
        signal_weighted_regret = weighted("baseline_regret_bps", "signal_count")
        candidate_weighted_symmetric = weighted(
            "baseline_symmetric_bps", "candidate_signal_count"
        )
        candidate_weighted_forward = weighted(
            "baseline_forward_bps", "candidate_signal_count"
        )
        candidate_weighted_regret = weighted(
            "baseline_regret_bps", "candidate_signal_count"
        )
        model_brier = weighted("model_brier", "eligible_count")
        prior_brier = weighted("prior_year_prevalence_brier", "eligible_count")
        rows.append(
            {
                "config_id": config_id,
                "model_kind": model_kind,
                "feature_set": feature_set,
                "horizon_cbr_rows_pub_proxy": horizon,
                "eligible_count": eligible_count,
                "baseline_hits": baseline_hits,
                "baseline_hit_rate": baseline_rate,
                "signal_count": signal_count,
                "signal_hits": signal_hits,
                "hit_rate": hit_rate,
                "candidate_signal_count": candidate_signal_count,
                "candidate_signal_hits": candidate_signal_hits,
                "candidate_hit_rate": (
                    candidate_signal_hits / candidate_signal_count
                    if candidate_signal_count
                    else math.nan
                ),
                "candidate_expected_hits_cell_standardized": expected_candidate_hits,
                "candidate_cell_standardized_lift": candidate_standardized_lift,
                "candidate_false_positives": candidate_false_positives,
                "candidate_false_negatives": candidate_false_negatives,
                "candidate_weighted_error_1": (
                    candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_2": (
                    2 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_3": (
                    3 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "candidate_weighted_error_5": (
                    5 * candidate_false_positives + candidate_false_negatives
                ) / eligible_count,
                "lift": hit_rate / baseline_rate if signal_count and baseline_rate else math.nan,
                "expected_signal_hits_cell_standardized": expected_signal_hits,
                "signal_weighted_baseline_hit_rate": standardized_baseline_rate,
                "cell_standardized_lift": standardized_lift,
                "mean_signals_per_week": float(
                    group.groupby("fold_test_year")["portfolio_signals_per_week"].first().mean()
                ),
                "mean_portfolio_signals_per_week": float(
                    group.groupby("fold_test_year")["portfolio_signals_per_week"].first().mean()
                ),
                "mean_candidate_signals_per_corridor_week": float(
                    group["candidate_signals_per_corridor_week"].mean()
                ),
                "minimum_fold_corridor_candidate_signals_per_week": float(
                    group["candidate_signals_per_corridor_week"].min()
                ),
                "maximum_fold_corridor_candidate_signals_per_week": float(
                    group["candidate_signals_per_corridor_week"].max()
                ),
                "mean_candidate_silent_week_share": float(
                    group["candidate_silent_week_share"].mean()
                ),
                "minimum_candidate_weeks_with_1_to_2_signals_share": float(
                    group["candidate_weeks_with_1_to_2_signals_share"].min()
                ),
                "mean_candidate_weeks_with_1_to_2_signals_share": float(
                    group["candidate_weeks_with_1_to_2_signals_share"].mean()
                ),
                "maximum_candidate_calendar_gap_days": float(
                    group["candidate_maximum_calendar_gap_days"].max()
                ),
                "candidate_top_fold_signal_share": (
                    float(candidate_fold_counts["candidate_signal_count"].max())
                    / candidate_signal_count
                    if candidate_signal_count
                    else math.nan
                ),
                "mean_portfolio_silent_week_share": float(
                    group.groupby("fold_test_year")["portfolio_silent_week_share"].first().mean()
                ),
                "minimum_portfolio_weeks_with_1_to_2_signals_share": float(
                    group.groupby("fold_test_year")[
                        "portfolio_weeks_with_1_to_2_signals_share"
                    ].first().min()
                ),
                "mean_portfolio_weeks_with_1_to_2_signals_share": float(
                    group.groupby("fold_test_year")[
                        "portfolio_weeks_with_1_to_2_signals_share"
                    ].first().mean()
                ),
                "maximum_portfolio_calendar_gap_days": float(
                    group["portfolio_maximum_calendar_gap_days"].max()
                ),
                "portfolio_top_fold_signal_share": (
                    float(fold_counts["signal_count"].max()) / signal_count
                    if signal_count
                    else math.nan
                ),
                "min_fold_lift": fold_lifts.min() if active_fold_count == len(fold_lifts) else math.nan,
                "folds_lift_ge_1": int((fold_lifts >= 1.0).sum()),
                "active_fold_count": active_fold_count,
                "fold_count": len(fold_lifts),
                "candidate_min_fold_lift": (
                    candidate_fold_lifts.min()
                    if active_candidate_fold_count == len(candidate_fold_lifts)
                    else math.nan
                ),
                "candidate_folds_lift_ge_1": int((candidate_fold_lifts >= 1.0).sum()),
                "candidate_active_fold_count": active_candidate_fold_count,
                "min_corridor_lift": (
                    corridor_lifts.min() if active_corridor_count == len(corridor_lifts) else math.nan
                ),
                "corridors_lift_ge_1": int((corridor_lifts >= 1.0).sum()),
                "active_corridor_count": active_corridor_count,
                "corridor_count": len(corridor_lifts),
                "candidate_min_corridor_lift": (
                    candidate_corridor_lifts.min()
                    if active_candidate_corridor_count == len(candidate_corridor_lifts)
                    else math.nan
                ),
                "candidate_corridors_lift_ge_1": int(
                    (candidate_corridor_lifts >= 1.0).sum()
                ),
                "candidate_active_corridor_count": active_candidate_corridor_count,
                "positive_fold_corridor_cells": int((group["lift"] >= 1.0).sum()),
                "active_fold_corridor_cells": int(group["signal_count"].gt(0).sum()),
                "fold_corridor_cell_count": len(group),
                "signal_symmetric_bps": signal_symmetric,
                "baseline_symmetric_bps": baseline_symmetric,
                "symmetric_bps_delta": signal_symmetric - baseline_symmetric,
                "cell_standardized_symmetric_bps_delta": signal_symmetric - signal_weighted_symmetric,
                "signal_forward_bps": signal_forward,
                "baseline_forward_bps": baseline_forward,
                "forward_bps_delta": signal_forward - baseline_forward,
                "cell_standardized_forward_bps_delta": signal_forward - signal_weighted_forward,
                "signal_regret_bps": signal_regret,
                "baseline_regret_bps": baseline_regret,
                "regret_bps_improvement": baseline_regret - signal_regret,
                "cell_standardized_regret_bps_improvement": signal_weighted_regret - signal_regret,
                "candidate_signal_symmetric_bps": candidate_signal_symmetric,
                "candidate_cell_standardized_symmetric_bps_delta": (
                    candidate_signal_symmetric - candidate_weighted_symmetric
                ),
                "candidate_signal_forward_bps": candidate_signal_forward,
                "candidate_cell_standardized_forward_bps_delta": (
                    candidate_signal_forward - candidate_weighted_forward
                ),
                "candidate_signal_regret_bps": candidate_signal_regret,
                "candidate_cell_standardized_regret_bps_improvement": (
                    candidate_weighted_regret - candidate_signal_regret
                ),
                "model_brier": model_brier,
                "prior_year_prevalence_brier": prior_brier,
                "brier_skill_vs_prior_year": 1.0 - model_brier / prior_brier if prior_brier and np.isfinite(model_brier) else math.nan,
                "exploratory_post_selection": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_cbr_rows_pub_proxy", "config_id"])


def add_coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    keys = ["config_id", "horizon_cbr_rows_pub_proxy", "feature"]
    summary = coefficients.groupby(keys)["standardized_coefficient"].agg(["mean", "std"]).reset_index()
    signs = coefficients.groupby(keys)["standardized_coefficient"].apply(
        lambda values: int((np.sign(values) == np.sign(values.mean())).sum())
    )
    summary["sign_agreement_folds"] = signs.to_numpy()
    summary = summary.rename(columns={"mean": "coefficient_mean", "std": "coefficient_std"})
    return coefficients.merge(summary, on=keys, how="left").sort_values(keys + ["fold_test_year"])


def calibration_bins(oot: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, float]:
    predicted = oot["probability"]
    if predicted.isna().all():
        return pd.DataFrame(), math.nan
    categories = pd.qcut(predicted, bins, duplicates="drop")
    result = oot.assign(calibration_bin=categories).groupby("calibration_bin", observed=True).agg(
        count=("target", "size"), mean_predicted=("probability", "mean"), observed_rate=("target", "mean")
    )
    result = result.reset_index()
    result["calibration_bin"] = result["calibration_bin"].astype(str)
    ece = float(np.average(np.abs(result["mean_predicted"] - result["observed_rate"]), weights=result["count"]))
    return result, ece


def month_block_bootstrap(
    oot: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    signal_column: str = "signal",
) -> dict[str, list[float]]:
    data = oot.assign(month=oot["date"].dt.to_period("M").astype(str)).copy()
    data["cell"] = data["fold_test_year"].astype(str) + ":" + data["corridor"]
    signal = data[signal_column].astype(float)
    data["eligible_count"] = 1.0
    data["signal_count"] = signal
    data["signal_hits"] = signal * data["target"]
    data["signal_forward"] = signal * data["forward_bps"]
    data["signal_symmetric"] = signal * data["symmetric_bps"]
    data["signal_regret"] = signal * data["regret_bps"]
    value_columns = [
        "eligible_count",
        "target",
        "signal_count",
        "signal_hits",
        "forward_bps",
        "signal_forward",
        "symmetric_bps",
        "signal_symmetric",
        "regret_bps",
        "signal_regret",
    ]
    grouped = data.groupby(["month", "cell"])[value_columns].sum()
    months = sorted(data["month"].unique())
    cells = sorted(data["cell"].unique())

    def blocks(column: str) -> np.ndarray:
        return grouped[column].unstack(fill_value=0.0).reindex(index=months, columns=cells, fill_value=0.0).to_numpy()

    weights = rng.multinomial(len(months), np.full(len(months), 1.0 / len(months)), size=repetitions)
    eligible = weights @ blocks("eligible_count")
    baseline_hits = weights @ blocks("target")
    signal_count = weights @ blocks("signal_count")
    signal_hits = weights @ blocks("signal_hits")
    baseline_forward = weights @ blocks("forward_bps")
    signal_forward = weights @ blocks("signal_forward")
    baseline_symmetric = weights @ blocks("symmetric_bps")
    signal_symmetric = weights @ blocks("signal_symmetric")
    baseline_regret = weights @ blocks("regret_bps")
    signal_regret = weights @ blocks("signal_regret")
    cell_hit_rate = np.divide(baseline_hits, eligible, out=np.zeros_like(baseline_hits), where=eligible > 0)
    expected_signal_hits = (signal_count * cell_hit_rate).sum(axis=1)
    total_signal_count = signal_count.sum(axis=1)
    cell_forward_mean = np.divide(baseline_forward, eligible, out=np.zeros_like(baseline_forward), where=eligible > 0)
    cell_symmetric_mean = np.divide(
        baseline_symmetric, eligible, out=np.zeros_like(baseline_symmetric), where=eligible > 0
    )
    cell_regret_mean = np.divide(baseline_regret, eligible, out=np.zeros_like(baseline_regret), where=eligible > 0)
    total_eligible = eligible.sum(axis=1)
    total_baseline_hits = baseline_hits.sum(axis=1)
    total_signal_hits = signal_hits.sum(axis=1)
    metrics = {
        "signal_forward_bps": signal_forward.sum(axis=1) / total_signal_count,
        "signal_symmetric_bps": signal_symmetric.sum(axis=1) / total_signal_count,
        "signal_regret_bps": signal_regret.sum(axis=1) / total_signal_count,
        "lift": total_signal_hits / expected_signal_hits,
        "absolute_delta_pp": 100.0 * (total_signal_hits - expected_signal_hits) / total_signal_count,
        "forward_bps_delta": (signal_forward - signal_count * cell_forward_mean).sum(axis=1) / total_signal_count,
        "symmetric_bps_delta": (signal_symmetric - signal_count * cell_symmetric_mean).sum(axis=1)
        / total_signal_count,
        "regret_bps_improvement": (signal_count * cell_regret_mean - signal_regret).sum(axis=1)
        / total_signal_count,
        "exposure_lift": (total_signal_hits / total_signal_count) / (total_baseline_hits / total_eligible),
        "exposure_absolute_delta_pp": 100.0
        * (total_signal_hits / total_signal_count - total_baseline_hits / total_eligible),
        "exposure_forward_bps_delta": signal_forward.sum(axis=1) / total_signal_count
        - baseline_forward.sum(axis=1) / total_eligible,
        "exposure_symmetric_bps_delta": signal_symmetric.sum(axis=1) / total_signal_count
        - baseline_symmetric.sum(axis=1) / total_eligible,
        "exposure_regret_bps_improvement": baseline_regret.sum(axis=1) / total_eligible
        - signal_regret.sum(axis=1) / total_signal_count,
    }
    return {
        name: np.quantile(values[np.isfinite(values)], [0.025, 0.5, 0.975]).tolist()
        for name, values in metrics.items()
    }


def circular_shift_null(
    oot: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    signal_column: str = "signal",
) -> tuple[dict[str, float], dict[str, float]]:
    prepared_oot = oot.assign(_policy_signal=oot[signal_column].astype(bool))
    matrices = []
    for _, fold in prepared_oot.groupby("fold_test_year"):
        matrix = {
            column: fold.pivot(index="date", columns="corridor", values=column).sort_index().to_numpy(float)
            for column in ("_policy_signal", "target", "forward_bps", "symmetric_bps", "regret_bps")
        }
        matrix["signal"] = matrix.pop("_policy_signal")
        matrix["cell_target_rate"] = matrix["target"].mean(axis=0)
        matrix["cell_forward_mean"] = matrix["forward_bps"].mean(axis=0)
        matrix["cell_symmetric_mean"] = matrix["symmetric_bps"].mean(axis=0)
        matrix["cell_regret_mean"] = matrix["regret_bps"].mean(axis=0)
        matrices.append(matrix)
    base_count = len(oot)
    base_hits = float(oot["target"].sum())
    base_forward = float(oot["forward_bps"].mean())
    base_symmetric = float(oot["symmetric_bps"].mean())
    base_regret = float(oot["regret_bps"].mean())
    annotated = prepared_oot.copy()
    cell = annotated.groupby(["fold_test_year", "corridor"])
    annotated["cell_target_rate"] = cell["target"].transform("mean")
    annotated["cell_forward_mean"] = cell["forward_bps"].transform("mean")
    annotated["cell_symmetric_mean"] = cell["symmetric_bps"].transform("mean")
    annotated["cell_regret_mean"] = cell["regret_bps"].transform("mean")
    signal = annotated[annotated["_policy_signal"]]
    observed = {
        "lift": float(signal["target"].sum() / signal["cell_target_rate"].sum()),
        "forward_bps_delta": float((signal["forward_bps"] - signal["cell_forward_mean"]).mean()),
        "symmetric_bps_delta": float((signal["symmetric_bps"] - signal["cell_symmetric_mean"]).mean()),
        "regret_bps_improvement": float((signal["cell_regret_mean"] - signal["regret_bps"]).mean()),
        "exposure_lift": float(signal["target"].mean() / oot["target"].mean()),
        "exposure_forward_bps_delta": float(signal["forward_bps"].mean() - base_forward),
        "exposure_symmetric_bps_delta": float(signal["symmetric_bps"].mean() - base_symmetric),
        "exposure_regret_bps_improvement": float(base_regret - signal["regret_bps"].mean()),
    }
    null = {name: np.empty(repetitions) for name in observed}
    for repetition in range(repetitions):
        signal_count = signal_hits = expected_signal_hits = 0.0
        signal_forward = signal_symmetric = signal_regret = 0.0
        standardized_forward = standardized_symmetric = standardized_regret = 0.0
        for matrix in matrices:
            shift = int(rng.integers(0, len(matrix["signal"])))
            shifted = np.roll(matrix["signal"], shift, axis=0).astype(bool)
            signal_count += shifted.sum()
            signal_hits += matrix["target"][shifted].sum()
            expected_signal_hits += (shifted * matrix["cell_target_rate"][None, :]).sum()
            signal_forward += matrix["forward_bps"][shifted].sum()
            signal_symmetric += matrix["symmetric_bps"][shifted].sum()
            signal_regret += matrix["regret_bps"][shifted].sum()
            standardized_forward += (
                matrix["forward_bps"] - matrix["cell_forward_mean"][None, :]
            )[shifted].sum()
            standardized_symmetric += (
                matrix["symmetric_bps"] - matrix["cell_symmetric_mean"][None, :]
            )[shifted].sum()
            standardized_regret += (
                matrix["cell_regret_mean"][None, :] - matrix["regret_bps"]
            )[shifted].sum()
        null["lift"][repetition] = signal_hits / expected_signal_hits
        null["forward_bps_delta"][repetition] = standardized_forward / signal_count
        null["symmetric_bps_delta"][repetition] = standardized_symmetric / signal_count
        null["regret_bps_improvement"][repetition] = standardized_regret / signal_count
        null["exposure_lift"][repetition] = (signal_hits / signal_count) / (base_hits / base_count)
        null["exposure_forward_bps_delta"][repetition] = signal_forward / signal_count - base_forward
        null["exposure_symmetric_bps_delta"][repetition] = signal_symmetric / signal_count - base_symmetric
        null["exposure_regret_bps_improvement"][repetition] = base_regret - signal_regret / signal_count
    p_values = {
        name: float((1 + np.sum(values >= observed[name] - 1e-12)) / (repetitions + 1))
        for name, values in null.items()
    }
    null_95 = {name: float(np.quantile(values, 0.95)) for name, values in null.items()}
    return {"observed": observed, "p_value_one_sided": p_values}, null_95


def tolerance_sensitivity(oot: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for tolerance in (0, 25, 50, 75, 100):
        labels = oot["regret_bps"] <= tolerance + 1e-9
        signal = oot["signal"].to_numpy(bool)
        signal_rate = float(labels[signal].mean())
        baseline_rate = float(labels.mean())
        cell_rate = labels.groupby([oot["fold_test_year"], oot["corridor"]]).transform("mean")
        expected_signal_hits = float(cell_rate[signal].sum())
        signal_hits = int(labels[signal].sum())
        rows.append(
            {
                "tolerance_bps": tolerance,
                "signal_hit_rate": signal_rate,
                "baseline_hit_rate": baseline_rate,
                "lift": signal_rate / baseline_rate,
                "cell_standardized_baseline_hit_rate": expected_signal_hits / signal.sum(),
                "cell_standardized_lift": signal_hits / expected_signal_hits,
            }
        )
    return rows


def corrected_saved_kzt(repo_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    features = pd.read_csv(repo_root / "data/kzt_v0/features.csv", parse_dates=["date"])
    backtest = json.loads((repo_root / "data/kzt_v0/backtest.json").read_text(encoding="utf-8"))
    events = [event for event in backtest["events"] if event.get("policy_eligible")]
    sources = (("CBR", "cbr_rate", "cbr_is_fresh"), ("NBK", "nbk_rate", "nbk_is_fresh"), ("MOEX", "moex_close", "moex_is_fresh"))
    rows = []
    for source, rate_column, fresh_column in sources:
        rates = features[rate_column].to_numpy(float)
        for scenario in ("favorable_now", "window_closing"):
            for horizon in HORIZONS:
                test_indexes: set[int] = set()
                for fold in backtest["folds"]:
                    mask = (features["date"] >= fold["validation_end"]) & (features["date"] < fold["test_end"])
                    fold_indexes = list(features.index[mask])
                    test_indexes.update(fold_indexes[:-horizon])
                eligible = [
                    index
                    for index in sorted(test_indexes)
                    if features.loc[index, fresh_column] == 1
                ]
                if scenario == "favorable_now":
                    label = lambda index: rates[index] <= rates[index + 1 : index + horizon + 1].min() + 1e-15
                else:
                    label = lambda index: rates[index + horizon] > rates[index] + 1e-15
                selected = [
                    int(event["index"])
                    for event in events
                    if event["scenario"] == scenario and int(event["index"]) in eligible
                ]
                baseline_hits = sum(label(index) for index in eligible)
                signal_hits = sum(label(index) for index in selected)
                baseline_rate = baseline_hits / len(eligible)
                hit_rate = signal_hits / len(selected) if selected else math.nan
                low, high = wilson(signal_hits, len(selected))
                forward = [
                    (rates[index + 1 : index + horizon + 1].mean() / rates[index] - 1.0) * 10_000.0
                    for index in selected
                ]
                rows.append(
                    {
                        "source": source,
                        "scenario": scenario,
                        "horizon_moex_rows": horizon,
                        "eligible_count": len(eligible),
                        "baseline_hits": baseline_hits,
                        "baseline_hit_rate": baseline_rate,
                        "signal_count": len(selected),
                        "signal_hits": signal_hits,
                        "hit_rate": hit_rate,
                        "hit_naive_iid_wilson95_low": low,
                        "hit_naive_iid_wilson95_high": high,
                        "hit_interval_note": (
                            "diagnostic_only_iid_interval; serial dependence is not represented"
                        ),
                        "lift": hit_rate / baseline_rate if selected and baseline_rate else math.nan,
                        "signal_forward_bps": float(np.mean(forward)) if forward else math.nan,
                        "label_definition": (
                            "current<=min(next_h_MOEX_rows)"
                            if scenario == "favorable_now"
                            else "endpoint_rate_t_plus_h_MOEX_rows>current"
                        ),
                        "eligibility": (
                            "saved_OOT_test_purged_by_fold_and_source_fresh_at_T; "
                            "future source values can be forward-filled"
                        ),
                    }
                )
    start = pd.Timestamp(backtest["folds"][0]["validation_end"])
    end = pd.Timestamp(backtest["folds"][-1]["test_end"])
    event_dates = pd.to_datetime([event["date"] for event in events])
    week_index = pd.period_range(start=start, end=end - pd.Timedelta(days=1), freq="W-SUN")
    event_weeks = pd.Series(event_dates).dt.to_period("W-SUN")
    frequency = {
        "test_span_days": int((end - start).days),
        "eligible_saved_events": len(events),
        "signals_per_week": len(events) / (end - start).days * 7.0,
        "week_count": len(week_index),
        "active_week_count": int(event_weeks.nunique()),
        "silent_week_share": 1.0 - event_weeks.nunique() / len(week_index),
        "first_signal_delay_days": int((event_dates.min() - start).days),
        "folds_without_any_candidate": int(sum(not fold["events"] for fold in backtest["folds"])),
        "fold_count": len(backtest["folds"]),
    }
    return pd.DataFrame(rows), frequency


def detect_down3(rates: np.ndarray) -> list[int]:
    updated = np.r_[False, np.abs(np.diff(rates)) > 1e-15]
    changed = np.where(updated)[0]
    raw = []
    for position in range(2, len(changed)):
        indexes = changed[position - 2 : position + 1]
        if all(rates[index] < rates[index - 1] for index in indexes):
            raw.append(int(indexes[-1]))
    selected: list[int] = []
    for index in raw:
        if not selected or index - selected[-1] > COOLDOWN_SESSIONS:
            selected.append(index)
    return selected


def source_robustness(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cbr = pd.read_csv(repo_root / "data/cbr_daily.csv", parse_dates=["date"])
    oxr = pd.read_csv(repo_root / "data/open_exchange_rates/rub_cis_daily.csv", parse_dates=["date"])
    quotes = sorted(cbr["corridor"].unique())
    profile_rows = []
    down3_rows = []
    for quote in quotes:
        aligned = cbr[cbr["corridor"] == quote][["date", "rub_per_unit"]].merge(
            oxr[oxr["quote"] == quote][["date", "rub_per_quote"]], on="date"
        )
        aligned = aligned.sort_values("date").reset_index(drop=True)
        cbr_rates = aligned["rub_per_unit"].to_numpy(float)
        oxr_rates = aligned["rub_per_quote"].to_numpy(float)
        basis = (oxr_rates / cbr_rates - 1.0) * 10_000.0
        cbr_signals = detect_down3(cbr_rates)
        oxr_signals = detect_down3(oxr_rates)
        exact_overlap = len(set(cbr_signals) & set(oxr_signals))
        union = len(set(cbr_signals) | set(oxr_signals))
        near_overlap = sum(any(abs(left - right) <= 1 for right in oxr_signals) for left in cbr_signals)
        profile_rows.append(
            {
                "corridor": quote,
                "aligned_rows": len(aligned),
                "level_correlation": aligned["rub_per_unit"].corr(aligned["rub_per_quote"]),
                "log_return_correlation": pd.Series(np.diff(np.log(cbr_rates))).corr(pd.Series(np.diff(np.log(oxr_rates)))),
                "basis_median_bps": float(np.median(basis)),
                "basis_mad_bps": float(np.median(np.abs(basis - np.median(basis)))),
                "basis_p95_abs_bps": float(np.quantile(np.abs(basis), 0.95)),
                "basis_max_abs_bps": float(np.max(np.abs(basis))),
                "cbr_down3_signals": len(cbr_signals),
                "oxr_down3_signals": len(oxr_signals),
                "exact_signal_jaccard": exact_overlap / union if union else math.nan,
                "cbr_signal_with_oxr_within_one_session_share": near_overlap / len(cbr_signals) if cbr_signals else math.nan,
            }
        )
        rates_by_source = {"CBR": cbr_rates, "OXR": oxr_rates}
        signals_by_source = {"CBR": cbr_signals, "OXR": oxr_signals}
        for horizon in (5, 20):
            eligible = list(range(max(3, horizon), len(aligned) - horizon))
            for trigger_source, selected_raw in signals_by_source.items():
                selected = [index for index in selected_raw if index in set(eligible)]
                for truth_source, truth_rates in rates_by_source.items():
                    labels = [truth_rates[index] <= truth_rates[index + 1 : index + horizon + 1].min() + 1e-15 for index in eligible]
                    signal_labels = [truth_rates[index] <= truth_rates[index + 1 : index + horizon + 1].min() + 1e-15 for index in selected]
                    baseline_rate = float(np.mean(labels))
                    hit_rate = float(np.mean(signal_labels))
                    forward = [
                        (truth_rates[index + 1 : index + horizon + 1].mean() / truth_rates[index] - 1.0) * 10_000.0
                        for index in selected
                    ]
                    down3_rows.append(
                        {
                            "corridor": quote,
                            "horizon_aligned_rows": horizon,
                            "trigger_source": trigger_source,
                            "truth_source": truth_source,
                            "eligible_count": len(eligible),
                            "signal_count": len(selected),
                            "hit_rate": hit_rate,
                            "baseline_hit_rate": baseline_rate,
                            "lift": hit_rate / baseline_rate,
                            "signal_forward_bps": float(np.mean(forward)),
                        }
                    )
    return pd.DataFrame(profile_rows), pd.DataFrame(down3_rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cbr_path = repo_root / "data/cbr_daily.csv"
    oxr_path = repo_root / "data/open_exchange_rates/rub_cis_daily.csv"
    panel_base = build_panel(cbr_path)

    fold_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    oot_by_configuration: dict[tuple[str, int], pd.DataFrame] = {}
    for horizon in HORIZONS:
        panel = add_target(panel_base, horizon)
        for experiment in EXPERIMENTS:
            experiment_oot = []
            for test_year in TEST_YEARS:
                metrics, oot, coefficients = run_fold(experiment, horizon, test_year, panel)
                fold_rows.extend(metrics)
                coefficient_rows.extend(coefficients)
                experiment_oot.append(oot)
            oot_by_configuration[(experiment.config_id, horizon)] = pd.concat(experiment_oot, ignore_index=True)

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["horizon_cbr_rows_pub_proxy", "config_id", "fold_test_year", "corridor"]
    )
    aggregate = aggregate_metrics(fold_metrics)
    learned = aggregate[
        aggregate["model_kind"].isin(
            [
                "logistic",
                "hist_gradient_boosting",
                "hist_gradient_boosting_stump",
                "hist_gradient_boosting_leaf100",
            ]
        )
    ]
    frequency_eligible = learned[
        learned["minimum_fold_corridor_candidate_signals_per_week"].ge(1.0)
        & learned["maximum_fold_corridor_candidate_signals_per_week"].le(2.0)
        & learned["minimum_candidate_weeks_with_1_to_2_signals_share"].ge(0.90)
        & learned["candidate_top_fold_signal_share"].le(0.50)
    ]
    ranking_pool = frequency_eligible if not frequency_eligible.empty else learned
    ranking_lift_leader = ranking_pool.sort_values(
        ["candidate_cell_standardized_lift", "config_id"], ascending=[False, True]
    ).iloc[0]
    economically_screened = frequency_eligible[
        frequency_eligible["candidate_signal_symmetric_bps"].gt(0)
        & frequency_eligible["candidate_signal_forward_bps"].ge(0)
    ]
    if economically_screened.empty:
        statistical_focus_row = ranking_lift_leader
        statistical_focus_rule = (
            "candidate_lift_leader_fallback_no_candidate_met_all_cadence_and_economic_screens"
        )
    else:
        statistical_focus_row = economically_screened.sort_values(
            [
                "candidate_cell_standardized_lift",
                "candidate_cell_standardized_forward_bps_delta",
                "config_id",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        statistical_focus_rule = "posthoc_economically_screened_candidate"
    statistical_focus_key = (
        str(statistical_focus_row["config_id"]),
        int(statistical_focus_row["horizon_cbr_rows_pub_proxy"]),
    )
    statistical_focus_oot = oot_by_configuration[statistical_focus_key]

    rng = np.random.default_rng(SEED)
    candidate_bootstrap = month_block_bootstrap(
        statistical_focus_oot,
        args.bootstrap_reps,
        rng,
        signal_column="candidate_signal",
    )
    portfolio_bootstrap = month_block_bootstrap(
        statistical_focus_oot,
        args.bootstrap_reps,
        np.random.default_rng(SEED + 1),
        signal_column="signal",
    )
    null_test, null_95 = circular_shift_null(
        statistical_focus_oot,
        args.null_reps,
        rng,
        signal_column="candidate_signal",
    )
    family_size = len(EXPERIMENTS) * len(HORIZONS)
    adjusted = {name: min(1.0, value * family_size) for name, value in null_test["p_value_one_sided"].items()}
    calibration, ece = calibration_bins(statistical_focus_oot)
    focus_probability = statistical_focus_oot["probability"].to_numpy(float)
    focus_labels = statistical_focus_oot["target"].to_numpy(float)
    brier = float(np.mean((focus_probability - focus_labels) ** 2))
    fold_prevalence = statistical_focus_oot.groupby("fold_test_year")["target"].transform("mean").to_numpy(float)
    oracle_fold_climatology_brier = float(np.mean((fold_prevalence - focus_labels) ** 2))

    coefficients = add_coefficient_stability(pd.DataFrame(coefficient_rows))
    corrected_kzt, kzt_frequency = corrected_saved_kzt(repo_root)
    source_profile, source_down3 = source_robustness(repo_root)
    statistics = {
        "status": "exploratory_post_selection_not_a_headline_claim",
        "primary_aggregate_estimand": (
            "corridor policy signal after threshold and corridor cooldown; fold×corridor standardization"
        ),
        "statistical_focus_configuration": {
            "role": statistical_focus_rule,
            "config_id": statistical_focus_key[0],
            "horizon_cbr_rows_pub_proxy": statistical_focus_key[1],
            "screening_rule": (
                "highest corridor-policy fold×corridor-standardized lift among learned configs meeting "
                "the per-corridor cadence/stability diagnostics and positive corridor-signal symmetric and "
                "forward bps point estimates; synthetic all-five portfolio metrics are diagnostic only"
            ),
            "screening_used_all_reported_OOT_results": True,
            "is_winner_or_confirmatory_claim": False,
        },
        "statistical_focus_absolute_signal_metrics": {
            "symmetric_bps": float(statistical_focus_row["candidate_signal_symmetric_bps"]),
            "forward_bps": float(statistical_focus_row["candidate_signal_forward_bps"]),
            "regret_bps": float(statistical_focus_row["candidate_signal_regret_bps"]),
        },
        "ranking_lift_leader": {
            "config_id": str(ranking_lift_leader["config_id"]),
            "horizon_cbr_rows_pub_proxy": int(ranking_lift_leader["horizon_cbr_rows_pub_proxy"]),
            "candidate_cell_standardized_lift": float(
                ranking_lift_leader["candidate_cell_standardized_lift"]
            ),
            "candidate_signal_symmetric_bps": float(
                ranking_lift_leader["candidate_signal_symmetric_bps"]
            ),
            "candidate_signal_forward_bps": float(
                ranking_lift_leader["candidate_signal_forward_bps"]
            ),
            "cell_standardized_symmetric_bps_delta": float(
                ranking_lift_leader["candidate_cell_standardized_symmetric_bps_delta"]
            ),
            "cell_standardized_forward_bps_delta": float(
                ranking_lift_leader["candidate_cell_standardized_forward_bps_delta"]
            ),
            "is_winner_or_confirmatory_claim": False,
        },
        "multiple_testing_family_size": family_size,
        "observed_and_null": null_test,
        "null_95_percentile": null_95,
        "bonferroni_adjusted_p_values": adjusted,
        "corridor_policy_month_block_bootstrap_95_interval": candidate_bootstrap,
        "synthetic_all_five_portfolio_month_block_bootstrap_95_interval": portfolio_bootstrap,
        "bootstrap_repetitions": args.bootstrap_reps,
        "null_repetitions": args.null_reps,
        "brier_score": brier,
        "oracle_same_fold_climatology_brier": oracle_fold_climatology_brier,
        "brier_skill_vs_oracle_same_fold_climatology": 1.0 - brier / oracle_fold_climatology_brier,
        "ece_10_equal_count_bins": ece,
        "tolerance_sensitivity": tolerance_sensitivity(statistical_focus_oot),
        "saved_kzt_frequency": kzt_frequency,
    }

    fold_metrics.to_csv(output_dir / "fold_corridor_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "feature_coefficients.csv", index=False)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
    corrected_kzt.to_csv(output_dir / "corrected_saved_kzt_metrics.csv", index=False)
    source_profile.to_csv(output_dir / "source_profile.csv", index=False)
    source_down3.to_csv(output_dir / "source_down3_crosscheck.csv", index=False)
    write_json(output_dir / "statistical_tests.json", statistics)
    output_paths = [
        output_dir / "fold_corridor_metrics.csv",
        output_dir / "aggregate_metrics.csv",
        output_dir / "feature_coefficients.csv",
        output_dir / "calibration_bins.csv",
        output_dir / "corrected_saved_kzt_metrics.csv",
        output_dir / "source_profile.csv",
        output_dir / "source_down3_crosscheck.csv",
        output_dir / "statistical_tests.json",
    ]
    script_path = Path(__file__).resolve()
    requirements_path = repo_root / "review_artifacts" / "quant_macro" / "requirements.txt"
    write_json(
        output_dir / "manifest.json",
        {
            "status": "exploratory_post_selection_not_a_headline_claim",
            "input_files": {
                str(cbr_path.relative_to(repo_root)): sha256(cbr_path),
                str(oxr_path.relative_to(repo_root)): sha256(oxr_path),
                "data/kzt_v0/features.csv": sha256(repo_root / "data/kzt_v0/features.csv"),
                "data/kzt_v0/backtest.json": sha256(repo_root / "data/kzt_v0/backtest.json"),
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "code_files": {
                str(script_path.relative_to(repo_root)): sha256(script_path),
                str(requirements_path.relative_to(repo_root)): sha256(requirements_path),
            },
            "output_files": {
                path.name: sha256(path)
                for path in output_paths
            },
            "seed": SEED,
            "arguments": {
                "bootstrap_repetitions": args.bootstrap_reps,
                "null_repetitions": args.null_reps,
                "output_directory": str(output_dir),
            },
            "corridors": sorted(panel_base["corridor"].unique()),
            "horizons_cbr_rows_pub_proxy": list(HORIZONS),
            "test_years": list(TEST_YEARS),
            "train_window_years": TRAIN_WINDOW_YEARS,
            "validation_window_years": 1,
            "purge_contract": (
                "train/validation and closed test-year tails purge h sessions; partial final year keeps every row "
                "whose target is globally resolved"
            ),
            "date_field_semantics": (
                "CBR effective-date records; ordered-record sequence is used as a publication-session proxy; "
                "actual published_at is unavailable"
            ),
            "target": "rub_per_unit[T] <= min(rub_per_unit[T+1:T+h])",
            "threshold_selection": (
                "purged prior-year validation only; first enforce 1-2 corridor policy signals/week, "
                "then minimize validation weighted_error_3; no test labels"
            ),
            "test_policy_state": (
                "corridor candidate cooldown and shared client cooldown are replayed separately on the complete "
                "prior-year history with the frozen fold model and threshold"
            ),
            "target_candidate_signals_per_corridor_week": TARGET_CANDIDATE_SIGNALS_PER_CORRIDOR_WEEK,
            "candidate_cadence_band_per_corridor_week": list(CANDIDATE_CADENCE_BAND),
            "threshold_false_positive_to_false_negative_cost": [FALSE_POSITIVE_COST, 1.0],
            "posthoc_candidate_frequency_band_per_corridor_week": [1.0, 2.0],
            "diagnostic_synthetic_all_five_portfolio_frequency_band_per_week": [0.8, 1.2],
            "posthoc_screening_scope": (
                f"learned configurations only; {len(frequency_eligible)} of {len(learned)} met the full "
                "per-corridor candidate cadence/stability screen; "
                f"{len(economically_screened)} also had positive absolute signal-date symmetric and forward "
                "point estimates; "
                "no configuration is declared a winner"
            ),
            "primary_aggregate_estimand": (
                "signal-count-weighted fold×corridor standardization; exposure-pooled metrics retained as sensitivity"
            ),
            "cooldown_cbr_rows_pub_proxy": COOLDOWN_SESSIONS,
            "experiments": [experiment.__dict__ for experiment in EXPERIMENTS],
            "feature_groups": FEATURE_GROUPS,
        },
    )
    print(f"Wrote artifacts to {output_dir}")
    print(
        "Post-hoc statistical focus (not a winner): "
        f"{statistical_focus_key[0]}, h_cbr_rows_pub_proxy={statistical_focus_key[1]}"
    )
    print(statistical_focus_row.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
