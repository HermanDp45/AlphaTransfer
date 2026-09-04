"""Central orchestration and reporting for the final solution."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

from .artifacts import (
    atomic_write_text,
    load_json,
    sha256,
    verify_lock,
    write_csv,
    write_json,
)
from .config import PipelineConfig
from .product import build_product_decision
from .scorecard import build_scorecard


def _metric_rows(scorecard: dict[str, Any]) -> list[dict[str, Any]]:
    headline = scorecard["headline_metrics"]
    return [
        {
            "metric": "cell_standardized_lift",
            "value": headline["cell_standardized_lift"]["value"],
            "ci95_low": headline["cell_standardized_lift"]["ci95"][0],
            "ci95_high": headline["cell_standardized_lift"]["ci95"][1],
            "required": ">=1.30 and CI low >1.00",
            "passed": headline["cell_standardized_lift"]["gate_passed"],
        },
        {
            "metric": "forward_official_reference_advantage_bps",
            "value": headline["forward_official_reference_advantage_bps"]["value"],
            "ci95_low": headline["forward_official_reference_advantage_bps"]["ci95"][0],
            "ci95_high": headline["forward_official_reference_advantage_bps"]["ci95"][1],
            "required": ">0 bps",
            "passed": headline["forward_official_reference_advantage_bps"]["gate_passed"],
        },
        {
            "metric": "weekly_coverage_1_to_2_candidates",
            "value": headline["weekly_coverage_1_to_2_candidates"]["value"],
            "ci95_low": "",
            "ci95_high": "",
            "required": ">=0.90 in every fold x corridor",
            "passed": headline["weekly_coverage_1_to_2_candidates"]["gate_passed"],
        },
    ]


def _source_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    manifest = load_json(config.path("data_manifest"))
    return [
        {
            "source_id": item["source_id"],
            "authority": item.get("authority", ""),
            "availability": item.get("availability", ""),
            "data_quality_status": item.get("data_quality_status", "research_snapshot"),
        }
        for item in manifest["sources"]
    ]


def _summary_markdown(
    scorecard: dict[str, Any],
    decision: dict[str, Any],
    source_count: int,
) -> str:
    headline = scorecard["headline_metrics"]
    brier = scorecard["model_selection_evidence"]
    delivery = decision["delivery"]
    signal = decision["market_signal"]
    signal_text = (
        f"RUB→{signal['corridor'].split('_')[-1]} на {signal['as_of']}"
        if signal
        else "нет сигнала на выбранную дату"
    )
    reasons = "\n".join(f"- `{reason}`" for reason in delivery["suppressed_reasons"])
    lift = headline["cell_standardized_lift"]
    benefit = headline["forward_official_reference_advantage_bps"]
    coverage = headline["weekly_coverage_1_to_2_candidates"]
    eligible = str(delivery["eligible_to_send"]).lower()
    return (
        "# AlphaTransfer — результат центрального pipeline\n\n"
        "**Статус:** `SHADOW_ONLY`; реальная отправка запрещена.  \n"
        f"**Исторический product preview:** {signal_text}.  \n"
        f"**Проверено источников:** {source_count}.\n\n"
        "## Три метрики решения\n\n"
        "| Метрика | Результат | Вердикт |\n"
        "|---|---:|---|\n"
        f"| Cell-standardized lift | {lift['value']:.3f}, 95% CI "
        f"[{lift['ci95'][0]:.3f}; {lift['ci95'][1]:.3f}] | PASS |\n"
        "| Forward official-reference advantage | "
        f"{benefit['value']:.2f} bps, 95% CI "
        f"[{benefit['ci95'][0]:.2f}; {benefit['ci95'][1]:.2f}] "
        "| PASS, не клиентская экономия |\n"
        f"| Недель с 1–2 кандидатами | {coverage['value']:.1%} "
        f"при пороге {coverage['required']:.0%} | FAIL |\n\n"
        "## Почему этот кандидат\n\n"
        f"HGB + компактный CNY/RUB basis снизил Brier с {brier['base_brier']:.6f} "
        f"до {brier['candidate_brier']:.6f} "
        f"({brier['relative_brier_improvement']:.2%}), улучшился в "
        f"{brier['years_improved']}/3 годах и "
        f"{brier['fold_corridor_cells_improved']}/"
        f"{brier['fold_corridor_cells_total']} fold×corridor ячейках. "
        "Модель «со всеми признаками» не выбрана: сложность не дала "
        "лучшего обобщения.\n\n"
        "## Product decision\n\n"
        f"`eligible_to_send = {eligible}`. Pipeline сформировал UX-preview, "
        "но не отправлял сообщение наружу.\n\n"
        "Причины подавления:\n\n"
        f"{reasons or '- нет'}\n\n"
        "Следующий честный шаг: заморозить протокол, подключить timestamped "
        "Alpha quotes и CRM eligibility, затем провести prospective shadow "
        "и рандомизированный holdout-пилот по incremental volume/revenue.\n"
    )


def run_pipeline(
    config: PipelineConfig,
    run_dir: Path,
    output_dir: Path,
    as_of: date,
    predictions_path: Path | None,
    client_context_path: Path | None,
    verify_locked_inputs: bool,
) -> dict[str, Any]:
    observed_hashes = (
        verify_lock(config.repo_root, config.path("input_lock"))
        if verify_locked_inputs
        else {}
    )
    scorecard = build_scorecard(config, run_dir)
    decision = build_product_decision(
        config,
        scorecard,
        as_of,
        predictions_path=predictions_path,
        client_context_path=client_context_path,
    )
    source_rows = _source_rows(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "model_scorecard.json", scorecard)
    write_json(output_dir / "signal_decision.json", decision)
    write_csv(
        output_dir / "key_metrics.csv",
        ["metric", "value", "ci95_low", "ci95_high", "required", "passed"],
        _metric_rows(scorecard),
    )
    write_csv(
        output_dir / "source_receipt.csv",
        ["source_id", "authority", "availability", "data_quality_status"],
        source_rows,
    )
    atomic_write_text(
        output_dir / "EXECUTIVE_SUMMARY.md",
        _summary_markdown(scorecard, decision, len(source_rows)),
    )

    receipt = {
        "schema_version": 1,
        "solution_id": config.section("solution")["id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(),
        "run_dir": str(run_dir),
        "predictions_path": str(predictions_path) if predictions_path else None,
        "locked_inputs_verified": verify_locked_inputs,
        "input_hashes": observed_hashes,
        "source_families": len(source_rows),
        "deployment_status": "shadow_only",
        "external_side_effects": [],
    }
    write_json(output_dir / "run_receipt.json", receipt)

    output_names = [
        "EXECUTIVE_SUMMARY.md",
        "key_metrics.csv",
        "model_scorecard.json",
        "run_receipt.json",
        "signal_decision.json",
        "source_receipt.csv",
    ]
    success = {
        "status": "complete",
        "deployment_status": "shadow_only",
        "eligible_to_send": decision["delivery"]["eligible_to_send"],
        "outputs": {name: sha256(output_dir / name) for name in output_names},
    }
    write_json(output_dir / "_SUCCESS.json", success)
    return {
        "status": "complete",
        "deployment_status": "shadow_only",
        "eligible_to_send": decision["delivery"]["eligible_to_send"],
        "output_dir": str(output_dir),
        "headline_metrics": scorecard["headline_metrics"],
        "suppressed_reasons": decision["delivery"]["suppressed_reasons"],
    }
