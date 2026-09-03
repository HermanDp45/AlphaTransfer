"""Small deterministic classifiers with inspectable parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from statistics import mean


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-max(-35.0, min(35.0, x))))


def _matrix(rows: list[dict], names: list[str]) -> list[list[float]]:
    return [[float(r[n]) for n in names] for r in rows]


@dataclass
class LogisticModel:
    feature_names: list[str]
    l2: float = 1.0
    epochs: int = 300
    learning_rate: float = 0.08
    means: list[float] | None = None
    scales: list[float] | None = None
    weights: list[float] | None = None

    def fit(self, rows: list[dict], labels: list[int]) -> "LogisticModel":
        if len(rows) != len(labels) or len(rows) < 20 or len(set(labels)) < 2:
            raise ValueError("model needs at least 20 mixed labelled rows")
        x = _matrix(rows, self.feature_names)
        cols = list(zip(*x))
        self.means = [mean(c) for c in cols]
        self.scales = [max(sqrt(mean((v - m) ** 2 for v in c)), 1e-9) for c, m in zip(cols, self.means)]
        z = [[(v - self.means[j]) / self.scales[j] for j, v in enumerate(row)] for row in x]
        self.weights = [0.0] * (len(self.feature_names) + 1)
        positives, negatives = sum(labels), len(labels) - sum(labels)
        positive_weight = negatives / max(positives, 1)
        for _ in range(self.epochs):
            grad = [0.0] * len(self.weights)
            for values, target in zip(z, labels):
                p = sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], values)))
                sample_weight = positive_weight if target else 1.0
                error = (p - target) * sample_weight
                grad[0] += error
                for j, value in enumerate(values):
                    grad[j + 1] += error * value
            for j in range(len(self.weights)):
                penalty = 0.0 if j == 0 else self.l2 * self.weights[j]
                self.weights[j] -= self.learning_rate * (grad[j] + penalty) / len(z)
        return self

    def predict(self, row: dict) -> float:
        if self.weights is None or self.means is None or self.scales is None:
            raise RuntimeError("model is not fitted")
        z = [(float(row[n]) - self.means[j]) / self.scales[j] for j, n in enumerate(self.feature_names)]
        return sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], z)))

    def explain(self, row: dict, limit: int = 5) -> list[dict]:
        if self.weights is None or self.means is None or self.scales is None:
            return []
        items = []
        for i, name in enumerate(self.feature_names):
            standardized = (float(row[name]) - self.means[i]) / self.scales[i]
            items.append({"feature": name, "value": round(float(row[name]), 6), "contribution": round(self.weights[i + 1] * standardized, 5)})
        return sorted(items, key=lambda x: abs(x["contribution"]), reverse=True)[:limit]

    def metadata(self) -> dict:
        return {"kind": "logistic_regression", "l2": self.l2, "features": self.feature_names,
                "coefficients": dict(zip(["intercept", *self.feature_names], self.weights or []))}


@dataclass
class _Stump:
    feature: str
    threshold: float
    left: float
    right: float


@dataclass
class ShallowBoostingModel:
    feature_names: list[str]
    rounds: int = 12
    learning_rate: float = 0.25
    base: float = 0.0
    stumps: list[_Stump] | None = None

    def fit(self, rows: list[dict], labels: list[int]) -> "ShallowBoostingModel":
        if len(rows) != len(labels) or len(rows) < 20 or len(set(labels)) < 2:
            raise ValueError("model needs at least 20 mixed labelled rows")
        prevalence = min(0.999, max(0.001, mean(labels)))
        self.base = log(prevalence / (1 - prevalence))
        scores = [self.base] * len(rows)
        self.stumps = []
        for _ in range(self.rounds):
            residuals = [y - sigmoid(score) for y, score in zip(labels, scores)]
            best: tuple[float, _Stump] | None = None
            for feature in self.feature_names:
                values = sorted(float(r[feature]) for r in rows)
                candidates = sorted({values[int((len(values) - 1) * q)] for q in (.2, .4, .6, .8)})
                for threshold in candidates:
                    left_i = [i for i, row in enumerate(rows) if float(row[feature]) <= threshold]
                    right_i = [i for i, row in enumerate(rows) if float(row[feature]) > threshold]
                    if not left_i or not right_i:
                        continue
                    left = mean(residuals[i] for i in left_i)
                    right = mean(residuals[i] for i in right_i)
                    left_set = set(left_i)
                    error = sum((residuals[i] - (left if i in left_set else right)) ** 2 for i in range(len(rows)))
                    stump = _Stump(feature, threshold, left, right)
                    if best is None or error < best[0]:
                        best = (error, stump)
            if best is None:
                break
            stump = best[1]
            self.stumps.append(stump)
            for i, row in enumerate(rows):
                scores[i] += self.learning_rate * (stump.left if float(row[stump.feature]) <= stump.threshold else stump.right)
        return self

    def predict(self, row: dict) -> float:
        score = self.base
        for stump in self.stumps or []:
            score += self.learning_rate * (stump.left if float(row[stump.feature]) <= stump.threshold else stump.right)
        return sigmoid(score)

    def explain(self, row: dict, limit: int = 5) -> list[dict]:
        contributions: dict[str, float] = {}
        for stump in self.stumps or []:
            value = self.learning_rate * (stump.left if float(row[stump.feature]) <= stump.threshold else stump.right)
            contributions[stump.feature] = contributions.get(stump.feature, 0.0) + value
        return [{"feature": name, "value": round(float(row[name]), 6), "contribution": round(value, 5)}
                for name, value in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:limit]]

    def metadata(self) -> dict:
        return {"kind": "shallow_gradient_boosting", "rounds": self.rounds, "learning_rate": self.learning_rate,
                "features": self.feature_names, "stumps": [s.__dict__ for s in (self.stumps or [])]}
