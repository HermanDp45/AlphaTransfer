"""Command-line entry point for the reference solution."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess
import sys

from .config import PipelineConfig
from .pipeline import run_pipeline


def _parser(default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphatransfer-final",
        description="Reproducible AlphaTransfer model-to-product decision pipeline.",
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--client-context", type=Path)
    parser.add_argument(
        "--rebuild",
        choices=("none", "smoke", "final"),
        default="none",
        help="re-run the audited ML engine before producing product artifacts",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument(
        "--verify-full-bundle",
        action="store_true",
        help="run the strict 76-artifact and prediction-clock verifier",
    )
    return parser


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _rebuild(
    repo_root: Path,
    tier: str,
    repetitions: int,
) -> Path:
    if tier == "final" and repetitions < 10_000:
        raise ValueError("final rebuild requires at least 10,000 bootstrap repetitions")
    solution_root = repo_root / "final_solution"
    run_dir = solution_root / "work" / f"model_{tier}_{repetitions}"
    success = run_dir / "_SUCCESS.json"
    if success.is_file():
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete rebuild directory is not empty: {run_dir}. "
            "Move it aside and retry; the pipeline never deletes artifacts automatically."
        )
    script = solution_root / "training" / "train_and_evaluate.py"
    data_dir = solution_root / "data" / "normalized"
    _run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(run_dir),
            "--bootstrap-reps",
            str(repetitions),
            "--run-tier",
            tier,
        ],
        cwd=repo_root,
    )
    return run_dir


def _strict_verify(repo_root: Path, run_dir: Path) -> None:
    verifier = repo_root / "final_solution" / "training" / "verify_bundle.py"
    _run(
        [
            sys.executable,
            str(verifier),
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
        ],
        cwd=repo_root,
    )


def main(argv: list[str] | None = None) -> int:
    solution_root = Path(__file__).resolve().parents[1]
    repo_root = solution_root.parent
    args = _parser(solution_root / "config.toml").parse_args(argv)
    config = PipelineConfig.load(args.config.resolve(), repo_root)

    canonical = config.path("canonical_run_dir")
    if args.rebuild == "none":
        run_dir = canonical
    else:
        run_dir = _rebuild(
            repo_root,
            args.rebuild,
            args.bootstrap_reps,
        )
    if args.verify_full_bundle:
        if args.rebuild == "smoke":
            raise ValueError("strict final-bundle verification is unavailable for smoke runs")
        _strict_verify(repo_root, run_dir)

    predictions = args.predictions
    if predictions is None and args.rebuild != "none":
        rebuilt_predictions = run_dir / "development_h5_predictions.csv"
        predictions = rebuilt_predictions if rebuilt_predictions.is_file() else None

    as_of = args.as_of or date.fromisoformat(config.section("solution")["default_as_of"])
    output_dir = (args.output_dir or solution_root / "output").resolve()
    result = run_pipeline(
        config=config,
        run_dir=run_dir,
        output_dir=output_dir,
        as_of=as_of,
        predictions_path=predictions.resolve() if predictions else None,
        client_context_path=args.client_context.resolve() if args.client_context else None,
        verify_locked_inputs=args.rebuild == "none",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
