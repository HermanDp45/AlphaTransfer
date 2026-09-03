"""Small auditable classical-ML baseline with no third-party dependency.

The classifier is deliberately plain logistic regression: its inputs are the
explainable rule features, model coefficients are inspectable, and it can be
retrained on each walk-forward fold without an MLOps stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, sqrt
from statistics import mean
from typing import Iterable

from .cbr import Quote
from .engine import SignalConfig, indicator_rows


FEATURE_NAMES = ("decreasing_run", "lowness", "rebound_up", "seasonal_below_history", "volatility_bps")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-max(-35.0, min(35.0, value))))


def _vector(row: dict) -> list[float]:
    return [
        float(row["decreasing_run"]),
        1.0 - float(row["low_percentile"]),
        float(row["rebound_up"]),
        float(row["seasonal_below_history"]),
        float(row["volatility_bps"]),
    ]


@dataclass
class LocalMinimumLogit:
    """Probability that today's price is minimum in the local ±h window."""

    horizon: int = 5
    false_positive_cost: float = 3.0
    learning_rate: float = 0.12
    epochs: int = 350
    means: list[float] | None = None
    scales: list[float] | None = None
    weights: list[float] | None = None
    training_rows: int = 0

    @property
    def decision_threshold(self) -> float:
        # Expected-loss decision: predict positive when p*cost(FN) exceeds
        # (1-p)*cost(FP).  A false "transfer now" is deliberately expensive.
        return self.false_positive_cost / (self.false_positive_cost + 1.0)

    def fit(self, quotes: Iterable[Quote], train_end: date | str, config: SignalConfig = SignalConfig()) -> "LocalMinimumLogit":
        cutoff = train_end if isinstance(train_end, date) else date.fromisoformat(train_end)
        training_quotes = [q for q in quotes if q.date <= cutoff]
        features = indicator_rows(training_quotes, config, as_of=cutoff)
        series: dict[str, list[Quote]] = {}
        for q in training_quotes:
            series.setdefault(q.corridor, []).append(q)
        index = {c: {q.date: i for i, q in enumerate(sorted(values, key=lambda x: x.date))} for c, values in series.items()}
        ordered = {c: sorted(values, key=lambda x: x.date) for c, values in series.items()}
        x: list[list[float]] = []
        y: list[float] = []
        for row in features:
            current = ordered[row["corridor"]]
            i = index[row["corridor"]][row["date"]]
            # Labels are admitted only once every required future observation
            # has occurred before the training cutoff.
            if i < self.horizon or i + self.horizon >= len(current):
                continue
            p = current[i].rub_per_unit
            neighbourhood = [q.rub_per_unit for q in current[i - self.horizon : i + self.horizon + 1]]
            x.append(_vector(row))
            y.append(float(p <= min(neighbourhood)))
        if len(x) < 20 or not any(y) or all(y):
            raise ValueError("Not enough mixed, resolved observations to fit local-minimum model")
        self.means = [mean(col) for col in zip(*x)]
        self.scales = [max(sqrt(mean((v - m) ** 2 for v in col)), 1e-9) for col, m in zip(zip(*x), self.means)]
        z = [[(v - self.means[j]) / self.scales[j] for j, v in enumerate(row)] for row in x]
        self.weights = [0.0] * (len(FEATURE_NAMES) + 1)
        # Class balancing stops the rare local-min label being reduced to an
        # always-negative classifier; the asymmetric action threshold remains.
        positives, negatives = sum(y), len(y) - sum(y)
        pos_weight = negatives / positives
        for _ in range(self.epochs):
            gradient = [0.0] * len(self.weights)
            for row, target in zip(z, y):
                probability = _sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], row)))
                weight = pos_weight if target else 1.0
                error = (probability - target) * weight
                gradient[0] += error
                for j, value in enumerate(row):
                    gradient[j + 1] += error * value
            scale = self.learning_rate / len(z)
            self.weights = [w - scale * g for w, g in zip(self.weights, gradient)]
        self.training_rows = len(x)
        return self

    def predict_probability(self, feature: dict) -> float:
        if self.means is None or self.scales is None or self.weights is None:
            raise RuntimeError("fit must be called before predict_probability")
        vector = _vector(feature)
        z = [(v - self.means[j]) / self.scales[j] for j, v in enumerate(vector)]
        return _sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], z)))

    def metadata(self) -> dict:
        if self.weights is None:
            return {"model": "logistic_regression", "fitted": False}
        return {
            "model": "logistic_regression", "target": f"local minimum in ±{self.horizon} publication days",
            "features": list(FEATURE_NAMES), "standardized_coefficients": dict(zip(("intercept", *FEATURE_NAMES), self.weights)),
            "decision_threshold": self.decision_threshold, "false_positive_cost": self.false_positive_cost,
            "training_rows": self.training_rows,
        }


def ml_events_for_rows(rows: Iterable[dict], model: LocalMinimumLogit) -> list[dict]:
    """Score precomputed point-in-time feature rows with a frozen model.

    Backtests call this once per fold after calculating features through the
    fold end.  That is equivalent to calling :func:`ml_signal_at` on each
    date, while avoiding an expensive repeated walk over the same history.
    """
    events = []
    for row in rows:
        probability = model.predict_probability(row)
        if probability >= model.decision_threshold:
            events.append({
                "date": row["date"], "corridor": row["corridor"], "indicator": "ml_local_minimum",
                "direction": "lower_rub_per_unit", "strength": round(probability, 4), "speed": "learned",
                "scenario": "favourable_now", "source_indicators": ",".join(name for name, active in (("momentum_down", row["momentum_down"]), ("low_level", row["low_level"]), ("seasonal_low", row["seasonal_below_history"])) if active) or "feature_model",
                "rub_per_unit": row["rub_per_unit"], "low_percentile": row["low_percentile"],
                "decreasing_run": row["decreasing_run"], "volatility_bps": row["volatility_bps"],
                "model_probability": round(probability, 6), "model_threshold": model.decision_threshold,
            })
    return events


def ml_signal_at(quotes: Iterable[Quote], as_of: date | str, model: LocalMinimumLogit, config: SignalConfig = SignalConfig()) -> list[dict]:
    """Score one date using a model fitted only on earlier resolved labels."""
    cutoff = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
    rows = [r for r in indicator_rows(quotes, config, as_of=cutoff) if r["date"] == cutoff]
    return ml_events_for_rows(rows, model)
