#!/usr/bin/env python3
"""Production runner for final models.

Provides a uniform CLI contract for the standalone TabM H3 and H5 bundles.

Supported actions:
- train: train/rebuild model artifacts
- infer: ensure-ready then run inference
- metrics: ensure-ready then export unified metrics report in CSV
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TABM_H3_DIR = REPO_ROOT / "final_solution" / "tabm_h3"
TABM_H5_DIR = REPO_ROOT / "final_solution" / "tabm_h5"
ROBUST_SELECTION_DIR = REPO_ROOT / "research_v4" / "robust_selection"
H5_RESEARCH_DIR = REPO_ROOT / "research_v4" / "h5_fullhistory"


@dataclass(frozen=True)
class MetricRow:
    model: str
    dataset: str
    period: str
    corridor_filter: str
    horizon: int
    threshold_policy: str
    threshold: float | None
    action: str = "metrics"
    hit_rate: float | None = None
    lift: float | None = None
    forward_delta_bps: float | None = None
    regret_bps: float | None = None
    signals_per_corridor_week: float | None = None
    mean_cell_week_coverage: float | None = None
    rows: float | None = None
    dates: float | None = None
    signals: float | None = None
    hits: float | None = None
    base_hit: float | None = None
    auc: float | None = None
    brier: float | None = None
    raw_brier: float | None = None
    log_loss: float | None = None
    silent_weeks: float | None = None
    week_cells: float | None = None
    max_silent_week_run: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _run_python(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.args[0]}): exit {result.returncode}")
    return result


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if pd.notna(value_f) else None


def _year_slice(df: pd.DataFrame, as_of_from: str | None, as_of_to: str | None, *, column: str = "year") -> pd.DataFrame:
    years = _period_years(as_of_from, as_of_to)
    if years is None:
        return df
    if column not in df:
        raise RuntimeError(f"metrics source missing column {column}")
    return df[df[column].isin(years)]


def _period_years(as_of_from: str | None, as_of_to: str | None) -> list[int] | None:
    if as_of_from is None and as_of_to is None:
        return None
    start = pd.Timestamp(as_of_from) if as_of_from else None
    end = pd.Timestamp(as_of_to) if as_of_to else None
    if start is None and end is not None:
        start = pd.Timestamp(f"{end.year}-01-01")
    if end is None and start is not None:
        end = pd.Timestamp(f"{start.year}-12-31")
    if start is None or end is None:
        raise RuntimeError("invalid period boundaries")
    if end < start:
        raise RuntimeError("--as-of-to must be >= --as-of-from")
    return list(range(int(start.year), int(end.year) + 1))




def _years_from_period(period_value: Any) -> list[int]:
    if period_value is None:
        return []
    value = str(period_value)
    if "-" in value:
        left, right = value.split("-", 1)
        return list(range(int(left), int(right) + 1))
    return [int(value)]


def _period_intersects(as_of_from: str | None, as_of_to: str | None, period_value: Any) -> bool:
    years = _years_from_period(period_value)
    if not years:
        return False
    target_start = int(as_of_from[:4]) if as_of_from else min(years)
    target_end = int(as_of_to[:4]) if as_of_to else max(years)
    return max(min(years), target_start) <= min(max(years), target_end)

def _scope_from_filter(corridor_filter: str) -> str | None:
    f = str(corridor_filter or "KZT").strip().upper()
    if f == "ALL5":
        return "ALL5"
    if f == "ALL":
        return None
    if f == "KZT":
        return "KZT"
    return f


def _normalize_corridor_filter(v: str) -> str:
    normalized = (v or "").strip().upper()
    if normalized not in {"ALL", "ALL5", "KZT"}:
        raise ValueError(f"unsupported corridor filter: {v}")
    return normalized


def _parse_period_label(as_of_from: str | None, as_of_to: str | None) -> str:
    if as_of_from and as_of_to:
        return f"{as_of_from}_{as_of_to}"
    if as_of_from:
        return f"{as_of_from}_"
    if as_of_to:
        return f"_{as_of_to}"
    return "all_years"


def _row_from_metrics_dict(model: str, dataset: str, period: str, corridor_filter: str, threshold_policy: str, threshold: float | None, row: pd.Series, horizon: int) -> dict[str, Any]:
    return MetricRow(
        model=model,
        dataset=dataset,
        period=str(period),
        corridor_filter=corridor_filter,
        horizon=int(horizon),
        threshold_policy=threshold_policy,
        threshold=_to_float(threshold),
        hit_rate=_to_float(row.get("hit_rate")),
        lift=_to_float(row.get("lift")),
        forward_delta_bps=_to_float(row.get("forward_delta_bps")),
        regret_bps=_to_float(row.get("regret_bps")),
        signals_per_corridor_week=_to_float(row.get("signals_per_corridor_week")),
        mean_cell_week_coverage=_to_float(row.get("mean_cell_week_coverage")),
        rows=_to_float(row.get("rows")),
        dates=_to_float(row.get("dates")),
        signals=_to_float(row.get("signals")),
        hits=_to_float(row.get("hits")),
        base_hit=_to_float(row.get("base_hit")),
        auc=_to_float(row.get("auc")),
        brier=_to_float(row.get("brier")),
        raw_brier=_to_float(row.get("raw_brier")),
        log_loss=_to_float(row.get("log_loss")),
        silent_weeks=_to_float(row.get("silent_weeks")),
        week_cells=_to_float(row.get("week_cells")),
        max_silent_week_run=_to_float(row.get("max_silent_week_run")),
    ).as_dict()


class ModelRunner(ABC):
    """Common interface for H3/H5 production runners."""

    def __init__(
        self,
        *,
        model: str,
        action: str,
        output_dir: Path | None,
        as_of: str | None = None,
        as_of_from: str | None = None,
        as_of_to: str | None = None,
        horizon: int = 3,
        corridor_filter: str = "KZT",
        force: bool = False,
        seed: int | None = None,
        skip_data_prefetch: bool = False,
    ) -> None:
        self.model = model
        self.action = action
        self.output_dir = output_dir
        self.as_of = as_of
        self.as_of_from = as_of_from
        self.as_of_to = as_of_to
        self.horizon = int(horizon)
        self.corridor_filter = _normalize_corridor_filter(corridor_filter)
        self.force = force
        self.seed = seed
        self.skip_data_prefetch = skip_data_prefetch

    @abstractmethod
    def artifacts(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def _default_output_dir(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def ensure_data(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self, *, force: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def infer(self, *, force: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def metrics(self, *, force: bool = False) -> pd.DataFrame:
        raise NotImplementedError

    def _artifacts_ok(self) -> bool:
        return all(path.is_file() for path in self.artifacts())

    def ensure_ready_for(self, action: str, *, force: bool) -> None:
        if action not in {"train", "infer", "metrics"}:
            raise ValueError(f"unsupported action {action}")
        if action == "train":
            if not self._artifacts_ok() or force:
                self.train(force=force)
            return
        if force or not self._artifacts_ok():
            self.train(force=bool(force))

    @property
    def output(self) -> Path:
        return self.output_dir or self._default_output_dir()

    def _metrics_out_path(self) -> Path:
        return self.output / "metrics.csv"

    def _write_metrics_receipt(self, rows: list[dict[str, Any]]) -> Path:
        payload = {
            "model": self.model,
            "action": "metrics",
            "status": "ok",
            "rows": int(len(rows)),
            "path": str(self._metrics_out_path()),
            "period": {
                "as_of_from": self.as_of_from,
                "as_of_to": self.as_of_to,
            },
            "horizon": self.horizon,
            "corridor_filter": self.corridor_filter,
        }
        _write_json(self.output / "metrics_receipt.json", payload)
        return self._metrics_out_path()


class H3Runner(ModelRunner):
    """Wrapper over shipped TabM H3 bundle in ``final_solution/tabm_h3``."""

    def __init__(
        self,
        *,
        bundle: Path | None = None,
        state_in: Path | None = None,
        state_out: Path | None = None,
        source_json: Path | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.bundle = bundle if bundle is not None else (TABM_H3_DIR / "bundle.json")
        if not self.bundle.exists():
            raise FileNotFoundError(f"missing bundle: {self.bundle}")
        self.bundle = self.bundle.resolve()
        self.state_in = state_in
        self.state_out = state_out
        self.source_json = source_json
        self.mode = mode
        super().__init__(model="h3", **kwargs)
        self._config = _read_json(self.bundle)
        self._source_receipt = self._load_source_receipt()

    def _load_source_receipt(self) -> list[dict[str, Any]]:
        path = self.bundle.parent / "source_receipt.json"
        if not path.exists():
            return []
        return _read_json(path)

    @staticmethod
    def _data_file_path(relative_or_abs: str) -> Path:
        p = Path(relative_or_abs)
        return p if p.is_absolute() else (TABM_H3_DIR / p).resolve()

    def artifacts(self) -> list[Path]:
        base = self.bundle.parent
        return [
            base / "model" / "weights.pt",
            base / "model" / "preprocess.joblib",
            base / "training_receipt.json",
            base / "bundle.json",
            base / "source_receipt.json",
        ]

    def _default_output_dir(self) -> Path:
        return self.bundle.parent / "output"

    def _bundle_sources(self) -> dict[str, Path]:
        return {
            name: self.bundle.parent / Path(rel_path)
            for name, rel_path in self._config.get("source_paths", {}).items()
        }

    def ensure_data(self) -> None:
        required = self._bundle_sources()
        missing = [name for name, src in required.items() if not src.is_file()]
        if not missing:
            return
        if not missing:
            return

        # try to restore deterministic source mapping from source receipt first
        restored: list[str] = []
        receipt_by_name = {
            item.get("name"): item
            for item in self._source_receipt
            if isinstance(item, dict) and item.get("name")
        }
        for name in missing:
            target = required[name]
            info = receipt_by_name.get(name)
            if not isinstance(info, dict):
                continue
            source_hint = info.get("source")
            if not source_hint:
                continue
            source = self.bundle.parent.parent / source_hint
            if source.exists() and not target.exists():
                _ensure_dir(target.parent)
                shutil.copy2(source, target)
                restored.append(f"{name}:{source} -> {target}")
        missing = [name for name in required if not required[name].is_file()]

        # known local fallbacks for historical source artifacts
        fallback_map = {
            "cbr": REPO_ROOT / "research_v4" / "h3_finalization" / "audit" / "latest_cbr.csv",
            "oxr": REPO_ROOT / "research_v4" / "oxr2010_bank" / "input_oxr_snapshot.csv",
            "halyk": REPO_ROOT / "research_v4" / "liquidity" / "halyk_sell_daily.csv",
            "treasury_t10yie": REPO_ROOT / "research_v3" / "external_data" / "normalized" / "treasury_t10yie.csv",
            "treasury_t5yifr": REPO_ROOT / "research_v3" / "external_data" / "normalized" / "treasury_t5yifr.csv",
            "moex_cny_close": REPO_ROOT / "final_solution" / "data" / "normalized" / "moex_cnyrub_tom.csv",
            "moex_cny_fixing": REPO_ROOT / "final_solution" / "data" / "normalized" / "moex_cny_fixing.csv",
        }
        for name in list(missing):
            fallback = fallback_map.get(name)
            if fallback is not None and fallback.exists():
                target = required[name]
                _ensure_dir(target.parent)
                shutil.copy2(fallback, target)
                restored.append(f"{name}:{fallback} -> {target}")
                missing.remove(name)

        if missing and not self.skip_data_prefetch:
            # attempt deterministic refresh of normalized market data
            _run_python(
                [
                    sys.executable,
                    str(REPO_ROOT / "final_solution" / "data_pipeline" / "fetch_open_data.py"),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--output-dir",
                    str(REPO_ROOT / "final_solution" / "data"),
                ]
            )
            # fill possible MOEX sources directly from normalized pipeline
            for name in list(missing):
                if name in {"moex_cny_close", "moex_cny_fixing"}:
                    source = {
                        "moex_cny_close": REPO_ROOT / "final_solution" / "data" / "normalized" / "moex_cnyrub_tom.csv",
                        "moex_cny_fixing": REPO_ROOT / "final_solution" / "data" / "normalized" / "moex_cny_fixing.csv",
                    }[name]
                    if source.exists():
                        target = required[name]
                        shutil.copy2(source, target)
                        restored.append(f"{name}:pipeline {source} -> {target}")
                        if name in missing:
                            missing.remove(name)

        if missing:
            details = ", ".join(missing)
            raise FileNotFoundError(
                "Missing H3 source files required by bundle: "
                + details
                + (". Rebuild manually or run fetch script with proper offline cache."),
            )

        # verify checksums only if declared
        expected = self._config.get("source_sha256", {})
        mismatched = []
        for name, path in required.items():
            expected_sha = expected.get(name)
            if expected_sha and path.is_file():
                actual = _sha256(path)
                if actual != expected_sha:
                    mismatched.append((name, str(path), expected_sha, actual))
        if mismatched and not self.skip_data_prefetch:
            details = ", ".join(f"{name}: expected={exp} actual={act}" for name, _, exp, act in mismatched)
            raise RuntimeError(f"H3 source checksum mismatch: {details}")
        if restored:
            print("[h3] restored source files: " + "; ".join(restored))

        # validate that all manifest sources are available before infer/train
        if not self._artifacts_ok():
            return

    def train(self, *, force: bool = False) -> dict[str, Any]:
        if self._artifacts_ok() and not force:
            return {
                "model": "h3",
                "status": "ready",
                "action": "train",
                "message": "artifacts already available",
            }

        if importlib.util.find_spec("pyarrow") is None:
            raise RuntimeError("pyarrow is required for H3 retraining (train script reads pickle/serialized features). Install dependencies from final_solution/tabm_h3/requirements.txt")

        script = REPO_ROOT / "research_v4" / "h3_finalization" / "package.py"
        if not script.exists():
            raise RuntimeError(f"Cannot rebuild H3 artifacts: missing script {script}")
        self.ensure_data()
        _run_python([sys.executable, str(script)], cwd=REPO_ROOT)
        still_missing = [path for path in self.artifacts() if not path.is_file()]
        if still_missing:
            raise RuntimeError("H3 packaging finished but required artifacts are still missing")
        _ensure_dir(self.output)
        _write_json(
            self.output / "train_receipt.json",
            {
                "model": "h3",
                "status": "trained",
                "artifacts": [str(p.relative_to(REPO_ROOT)) for p in self.artifacts()],
                "force": bool(force),
            },
        )
        return {
            "model": "h3",
            "status": "trained",
            "action": "train",
            "artifacts": [str(p.relative_to(REPO_ROOT)) for p in self.artifacts()],
        }

    def infer(self, *, force: bool = False) -> dict[str, Any]:
        self.ensure_data()
        self.ensure_ready_for("infer", force=force)
        from final_solution.tabm_h3 import predict as h3_predict

        config, _ = h3_predict.load_config(self.bundle)
        as_of = self.as_of or config.get("default_as_of", config.get("model_cutoff"))
        mode = self.mode or config.get("metadata", {}).get("default_mode", "historical_smoke")

        output = self.output
        _ensure_dir(output)
        result = h3_predict.run(
            self.bundle,
            output,
            as_of,
            mode=mode,
            state_in=self.state_in,
            state_out=self.state_out,
            source_config=self.source_json,
        )

        # ensure decision artifact is colocated for parity
        return {
            **result,
            "model": "h3",
            "action": "infer",
            "output_dir": str(output),
            "train_horizon": int(config.get("train_horizon", self.horizon)),
        }

    def metrics(self, *, force: bool = False) -> pd.DataFrame:
        self.ensure_data()
        self.ensure_ready_for("metrics", force=force)

        by_year = pd.read_csv(TABM_H3_DIR / "evaluation" / "by_year.csv")
        summary = pd.read_csv(TABM_H3_DIR / "evaluation" / "summary.csv")

        policy = str(self._config.get("policy", {}).get("name", "rank80"))
        threshold = _to_float(self._config.get("policy", {}).get("threshold", float("nan")))

        mask_year = _year_slice(by_year, self.as_of_from, self.as_of_to)
        scope = _scope_from_filter(self.corridor_filter)
        filter_mask = pd.Series([True] * len(mask_year), index=mask_year.index)
        if scope and "evaluation_scope" in mask_year.columns:
            filter_mask &= mask_year["evaluation_scope"].astype(str).str.upper() == scope.upper()
        filter_mask &= mask_year["policy"].astype(str) == policy
        filter_mask &= mask_year["train_horizon"].fillna(self.horizon).astype(int) == self.horizon
        filter_mask &= mask_year["evaluation_horizon"].fillna(self.horizon).astype(int) == self.horizon

        rows_src = mask_year[filter_mask]
        if rows_src.empty:
            raise RuntimeError("No H3 by_year rows match requested filter")

        rows: list[dict[str, Any]] = []
        dataset_rel = str((TABM_H3_DIR / "evaluation" / "by_year.csv").relative_to(REPO_ROOT))
        for _, row in rows_src.iterrows():
            rows.append(
                _row_from_metrics_dict(
                    model="h3",
                    dataset=dataset_rel,
                    period=str(int(row["year"])),
                    corridor_filter=self.corridor_filter,
                    threshold_policy=policy,
                    threshold=threshold,
                    row=row,
                    horizon=self.horizon,
                )
            )

        if self.as_of_from is None and self.as_of_to is None:
            summary_filtered = summary.copy()
        else:
            summary_filtered = summary[
                [
                    _period_intersects(self.as_of_from, self.as_of_to, x)
                    for x in summary["period"].to_list()
                ]
            ]
        # keep only the requested scope; if missing scope in summary, keep all (h3 usually KZT)
        summary_mask = pd.Series([True] * len(summary_filtered), index=summary_filtered.index)
        if "evaluation_scope" in summary_filtered.columns and scope:
            summary_mask &= summary_filtered["evaluation_scope"].astype(str).str.upper() == scope.upper()
        summary_mask &= summary_filtered["policy"].astype(str) == policy
        summary_mask &= summary_filtered["train_horizon"].fillna(self.horizon).astype(int) == self.horizon
        summary_mask &= summary_filtered["evaluation_horizon"].fillna(self.horizon).astype(int) == self.horizon

        summary_selected = summary_filtered[summary_mask]
        if not summary_selected.empty:
            summary_dataset = str((TABM_H3_DIR / "evaluation" / "summary.csv").relative_to(REPO_ROOT))
            for _, row in summary_selected.iterrows():
                rows.append(
                    _row_from_metrics_dict(
                        model="h3",
                        dataset=summary_dataset,
                        period=str(row.get("period", _parse_period_label(self.as_of_from, self.as_of_to))),
                        corridor_filter=self.corridor_filter,
                        threshold_policy=policy,
                        threshold=threshold,
                        row=row,
                        horizon=self.horizon,
                    )
                )

        df = pd.DataFrame(rows)
        _ensure_dir(self.output)
        df.to_csv(self._metrics_out_path(), index=False)
        self._write_metrics_receipt(rows)
        df.attrs["path"] = str(self._metrics_out_path())
        return df


class H5Runner(ModelRunner):
    """Standalone H5 runtime backed by the selected full/120m recipe bundle."""

    def __init__(
        self,
        *,
        policy: str | None = None,
        bundle: Path | None = None,
        state_in: Path | None = None,
        state_out: Path | None = None,
        source_json: Path | None = None,
        mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.bundle = (bundle or (TABM_H5_DIR / "bundle.json")).resolve()
        self.requested_policy = policy
        self.state_in = state_in
        self.state_out = state_out
        self.source_json = source_json
        self.mode = mode
        super().__init__(model="h5", **kwargs)
        self.selection_path = TABM_H5_DIR / "evaluation" / "selection.json"
        self.predictions_path = TABM_H5_DIR / "evaluation" / "predictions.csv.gz"
        self.summary_path = TABM_H5_DIR / "evaluation" / "summary.csv"
        self.by_year_path = TABM_H5_DIR / "evaluation" / "by_year.csv"

    def _config(self) -> dict[str, Any]:
        if not self.bundle.is_file():
            raise FileNotFoundError(f"missing H5 bundle: {self.bundle}")
        return _read_json(self.bundle)

    @property
    def policy(self) -> str:
        selected = self._config().get("policy", {}).get("name", "rare65")
        if self.requested_policy is not None and self.requested_policy != selected:
            raise ValueError(
                f"Requested H5 policy {self.requested_policy!r} is not the packaged selected policy {selected!r}"
            )
        return str(selected)

    def artifacts(self) -> list[Path]:
        return [
            self.bundle,
            TABM_H5_DIR / "model" / "weights.pt",
            TABM_H5_DIR / "model" / "preprocess.joblib",
            TABM_H5_DIR / "model" / "model.json",
            TABM_H5_DIR / "initial_state.json",
            TABM_H5_DIR / "training_receipt.json",
            TABM_H5_DIR / "source_receipt.json",
            TABM_H5_DIR / "selection_receipt.json",
            TABM_H5_DIR / "feature_contract.json",
            TABM_H5_DIR / "policy.json",
            TABM_H5_DIR / "runtime_verification.json",
            TABM_H5_DIR / "policy_calibration_receipt.json",
            self.selection_path,
            self.predictions_path,
            self.summary_path,
            self.by_year_path,
        ]

    def _default_output_dir(self) -> Path:
        return TABM_H5_DIR / "output"

    def ensure_data(self) -> None:
        if not self.bundle.is_file():
            return
        config = self._config()
        required = {name: self.bundle.parent / path for name, path in config.get("source_paths", {}).items()}
        missing = [name for name, path in required.items() if not path.is_file()]
        if not missing:
            return
        if not self.skip_data_prefetch:
            # The two final models intentionally share the exact same point-in-time
            # source contract. Restore from the shipped H3 cache before attempting
            # to rebuild the selected bundle.
            h3 = TABM_H3_DIR / "data"
            for name in list(missing):
                source = h3 / Path(config["source_paths"][name]).name
                if source.is_file():
                    target = required[name]
                    _ensure_dir(target.parent)
                    shutil.copy2(source, target)
                    missing.remove(name)
        if missing:
            raise FileNotFoundError("Missing H5 source files: " + ", ".join(missing))

        expected = config.get("source_sha256", {})
        mismatched = [name for name, path in required.items() if expected.get(name) and _sha256(path) != expected[name]]
        if mismatched:
            raise RuntimeError("H5 source checksum mismatch: " + ", ".join(mismatched))

    @staticmethod
    def _research_bundle_ready(path: Path) -> bool:
        """Return true only when a final-fit bundle and its heavy artifacts exist.

        The JSON receipt is committed, while weights and preprocessing are
        intentionally ignored by Git.  Checking the receipt alone would make a
        clean clone skip the reproducible final refit and then fail packaging.
        """
        if not path.is_file():
            return False
        try:
            config = _read_json(path)
            model = config["model"]
            required = (
                path.parent / str(model["weights"]),
                path.parent / str(model["preprocessor"]),
            )
        except (KeyError, TypeError, ValueError, OSError):
            return False
        return all(artifact.is_file() for artifact in required)

    @staticmethod
    def _run_research_refit(*, force: bool) -> None:
        selection = H5_RESEARCH_DIR / "selection.json"
        # Selection is sealed rolling-OOT evidence. ``--force`` retrains the
        # selected production checkpoint; it must not silently rerun model
        # selection and change the public recipe.
        if not selection.is_file():
            experiment = next(
                (path for path in (H5_RESEARCH_DIR / "experiment.py", H5_RESEARCH_DIR / "run.py") if path.is_file()),
                None,
            )
            if experiment is None:
                raise RuntimeError(f"Cannot select H5 recipe: no experiment entrypoint under {H5_RESEARCH_DIR}")
            _run_python([sys.executable, str(experiment)], cwd=REPO_ROOT)
        # A final fit is distinct from annual OOT checkpoints and is required for
        # production inference. Let the research selection choose 120m/full history.
        final_fit = H5_RESEARCH_DIR / "final_fit.py"
        conventional_bundles = [
            H5_RESEARCH_DIR / "production_bundle" / "bundle.json",
            H5_RESEARCH_DIR / "final_fit" / "bundle.json",
        ]
        if force or not any(H5Runner._research_bundle_ready(path) for path in conventional_bundles):
            if not final_fit.is_file():
                raise RuntimeError(
                    f"H5 selection exists but final-fit entrypoint is missing: {final_fit}. "
                    "Annual OOT checkpoints cannot substitute for a production refit."
                )
            argv = [sys.executable, str(final_fit)]
            if force:
                argv.append("--force")
            _run_python(argv, cwd=REPO_ROOT)

    def train(self, *, force: bool = False) -> dict[str, Any]:
        if self._artifacts_ok() and not force:
            return {
                "model": "h5",
                "status": "ready",
                "action": "train",
                "message": "artifacts already available",
            }
        self._run_research_refit(force=force)
        _run_python([sys.executable, str(TABM_H5_DIR / "package.py")], cwd=REPO_ROOT)
        if not self._artifacts_ok():
            missing = [str(path) for path in self.artifacts() if not path.is_file()]
            raise RuntimeError("Failed to build standalone H5 artifacts: " + ", ".join(missing))
        self.ensure_data()
        _ensure_dir(self.output)
        _write_json(
            self.output / "train_receipt.json",
            {
                "model": "h5",
                "status": "trained",
                "policy": self.policy,
                "artifacts": [str(path.relative_to(REPO_ROOT)) for path in self.artifacts()],
                "force": bool(force),
                "action": "train",
            },
        )
        return {
            "model": "h5",
            "status": "trained",
            "action": "train",
            "policy": self.policy,
            "artifacts": [str(path.relative_to(REPO_ROOT)) for path in self.artifacts()],
        }

    def infer(self, *, force: bool = False) -> dict[str, Any]:
        self.ensure_ready_for("infer", force=force)
        self.ensure_data()
        from final_solution.tabm_h5 import predict as h5_predict

        config = self._config()
        as_of = self.as_of or config.get("default_as_of", config["model_cutoff"])
        mode = self.mode or config.get("metadata", {}).get("default_mode", "historical_smoke")
        return h5_predict.run(
            self.bundle,
            self.output,
            as_of,
            mode=mode,
            state_in=self.state_in,
            state_out=self.state_out,
            source_config=self.source_json,
        )

    def metrics(self, *, force: bool = False) -> pd.DataFrame:
        self.ensure_ready_for("metrics", force=force)
        self.ensure_data()
        by_year = pd.read_csv(self.by_year_path)
        summary = pd.read_csv(self.summary_path)

        by_year_mask = (
            by_year["policy"].astype(str).eq(self.policy)
            & by_year["train_horizon"].fillna(self.horizon).astype(int).eq(self.horizon)
            & by_year["evaluation_horizon"].fillna(self.horizon).astype(int).eq(self.horizon)
        )
        if "cohort" in by_year:
            by_year_mask &= by_year["cohort"].astype(str).eq("native_matured")
        filtered = by_year[by_year_mask]
        scope = _scope_from_filter(self.corridor_filter)
        if scope and "evaluation_scope" in filtered.columns:
            filtered = filtered[filtered["evaluation_scope"].astype(str).str.upper() == scope.upper()]
        filtered = _year_slice(filtered, self.as_of_from, self.as_of_to)
        if filtered.empty:
            raise RuntimeError("No H5 by_year rows match requested filter")

        if "config_id" in filtered:
            selected_config = _read_json(self.selection_path).get("selected_config_id")
            if selected_config:
                filtered = filtered[filtered["config_id"].astype(str).eq(str(selected_config))]
        if filtered.empty:
            raise RuntimeError("No H5 annual rows match the packaged selected config")
        dataset_by_year = str(self.by_year_path.relative_to(REPO_ROOT))
        dataset_summary = str(self.summary_path.relative_to(REPO_ROOT))
        rows: list[dict[str, Any]] = []
        for _, row in filtered.iterrows():
            rows.append(
                _row_from_metrics_dict(
                    model="h5",
                    dataset=dataset_by_year,
                    period=str(int(row["year"])),
                    corridor_filter=self.corridor_filter,
                    threshold_policy=self.policy,
                    threshold=None,
                    row=row,
                    horizon=self.horizon,
                )
            )

        summary_mask = (
            summary["policy"].astype(str).eq(self.policy)
            & summary["train_horizon"].fillna(self.horizon).astype(int).eq(self.horizon)
            & summary["evaluation_horizon"].fillna(self.horizon).astype(int).eq(self.horizon)
        )
        if "cohort" in summary:
            summary_mask &= summary["cohort"].astype(str).eq("native_matured")
        if self.as_of_from is not None or self.as_of_to is not None:
            summary_mask &= pd.Series(
                [_period_intersects(self.as_of_from, self.as_of_to, p) for p in summary["period"].to_list()],
                index=summary.index,
            )
        summary_filtered = summary[summary_mask]
        if scope and "evaluation_scope" in summary_filtered.columns:
            summary_filtered = summary_filtered[summary_filtered["evaluation_scope"].astype(str).str.upper() == scope.upper()]
        if "config_id" in summary_filtered:
            selected_config = _read_json(self.selection_path).get("selected_config_id")
            if selected_config:
                summary_filtered = summary_filtered[summary_filtered["config_id"].astype(str).eq(str(selected_config))]

        for _, row in summary_filtered.iterrows():
            rows.append(
                _row_from_metrics_dict(
                    model="h5",
                    dataset=dataset_summary,
                    period=str(row.get("period", _parse_period_label(self.as_of_from, self.as_of_to))),
                    corridor_filter=self.corridor_filter,
                    threshold_policy=self.policy,
                    threshold=None,
                    row=row,
                    horizon=self.horizon,
                )
            )

        df = pd.DataFrame(rows)
        _ensure_dir(self.output)
        df.to_csv(self._metrics_out_path(), index=False)
        self._write_metrics_receipt(rows)
        df.attrs["path"] = str(self._metrics_out_path())
        return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaTransfer final unified CLI")
    parser.add_argument("--model", choices=("h3", "h5"), default="h3")
    parser.add_argument("--action", choices=("train", "infer", "metrics"), default="infer")
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--as-of", dest="as_of", default=None)
    parser.add_argument("--as-of-from", dest="as_of_from", default=None)
    parser.add_argument("--as-of-to", dest="as_of_to", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--corridor-filter", choices=("all", "all5", "KZT"), default="KZT")
    parser.add_argument("--skip-data-prefetch", action="store_true", default=False)

    # H3-specific legacy pass-throughs
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--mode", choices=("operational", "historical_smoke"), default=None)
    parser.add_argument("--state-in", type=Path, default=None)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--sources", type=Path, default=None)

    # H5-specific
    parser.add_argument("--h5-policy", default="rare65")
    return parser


def execute(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    horizon = args.horizon if args.horizon is not None else (5 if args.model == "h5" else 3)

    if args.model == "h3":
        runner = H3Runner(
            bundle=args.bundle,
            output_dir=args.output_dir,
            action=args.action,
            as_of=args.as_of,
            as_of_from=args.as_of_from,
            as_of_to=args.as_of_to,
            horizon=horizon,
            corridor_filter=args.corridor_filter,
            force=args.force,
            seed=args.seed,
            skip_data_prefetch=args.skip_data_prefetch,
            state_in=args.state_in,
            state_out=args.state_out,
            source_json=args.sources,
            mode=args.mode,
        )
    else:
        runner = H5Runner(
            policy=args.h5_policy,
            bundle=args.bundle,
            output_dir=args.output_dir,
            action=args.action,
            as_of=args.as_of,
            as_of_from=args.as_of_from,
            as_of_to=args.as_of_to,
            horizon=horizon,
            corridor_filter=args.corridor_filter,
            force=args.force,
            seed=args.seed,
            skip_data_prefetch=args.skip_data_prefetch,
            state_in=args.state_in,
            state_out=args.state_out,
            source_json=args.sources,
            mode=args.mode,
        )

    if args.action == "train":
        return runner.train(force=args.force)
    if args.action == "infer":
        return runner.infer(force=args.force)
    if args.action == "metrics":
        return runner.metrics(force=args.force)
    raise RuntimeError(f"unsupported action {args.action}")


def main(argv: list[str] | None = None) -> int:
    result = execute(argv)
    if isinstance(result, pd.DataFrame):
        payload = {
            "status": "ok",
            "type": "metrics",
            "rows": int(len(result)),
            "path": str(result.attrs.get("path", "")),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    print(json.dumps({"status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
