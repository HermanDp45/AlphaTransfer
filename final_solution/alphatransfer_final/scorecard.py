"""Build the compact model and product scorecard from the audited run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import as_bool, as_float, as_int, load_json, read_csv, select_one
from .config import PipelineConfig


def _selected_row(path: Path, config: PipelineConfig, label: str) -> dict[str, str]:
    solution = config.section("solution")
    return select_one(
        read_csv(path),
        {
            "config_id": solution["selected_config"],
        },
        label,
    )


def build_scorecard(config: PipelineConfig, run_dir: Path) -> dict[str, Any]:
    solution = config.section("solution")
    gates = config.section("gates")
    selected = solution["selected_config"]
    horizon = str(solution["primary_horizon"])

    metrics = select_one(
        read_csv(run_dir / "development_metrics.csv"),
        {"config_id": selected, "horizon_cbr_rows_pub_proxy": horizon},
        "development metric",
    )
    family = _selected_row(run_dir / "feature_family_audit_h5.csv", config, "family audit")
    uncertainty = _selected_row(
        run_dir / "development_policy_uncertainty_audit_h5.csv",
        config,
        "policy uncertainty audit",
    )
    lag_rows = [
        row
        for row in read_csv(run_dir / "moex_lag_ladder_h5.csv")
        if row.get("config_id") == selected
    ]
    lag_by_days = {int(float(row["moex_calendar_lag_days"])): row for row in lag_rows}
    selection = load_json(run_dir / "selection.json")

    lift = as_float(uncertainty, "candidate_cell_standardized_lift")
    lift_low = as_float(uncertainty, "candidate_cell_standardized_lift_ci95_low")
    forward_bps = as_float(uncertainty, "candidate_cell_standardized_forward_bps_delta")
    forward_low = as_float(
        uncertainty,
        "candidate_cell_standardized_forward_bps_delta_ci95_low",
    )
    weekly_coverage = as_float(family, "minimum_candidate_weeks_with_1_to_2_signals_share")
    relative_brier = as_float(family, "relative_brier_improvement")

    metric_gates = {
        "lift": lift >= float(gates["minimum_lift"])
        and lift_low > float(gates["minimum_lift_ci95_low"]),
        "forward_advantage": forward_bps > float(gates["minimum_forward_advantage_bps"]),
        "weekly_coverage": weekly_coverage >= float(gates["minimum_weekly_coverage"]),
        "brier": relative_brier >= float(gates["minimum_brier_improvement"]),
        "year_stability": as_int(family, "years_brier_improved")
        >= int(gates["minimum_improved_years"]),
        "cell_stability": as_int(family, "fold_corridor_cells_brier_improved")
        >= int(gates["minimum_improved_cells"]),
    }

    research_failures = [name for name, passed in metric_gates.items() if not passed]
    production_checks = {
        "prospective_shadow_complete": bool(selection.get("prospective_shadow_tracks")),
        "production_track_promoted": bool(selection.get("production_promoted_tracks")),
        "historical_publication_timestamps_verified": as_bool(
            family,
            "historical_point_in_time_verified",
        ),
        "production_data_rights_verified": as_bool(
            family,
            "production_data_rights_verified",
        ),
        "per_signal_explanation_available": as_bool(
            family,
            "per_signal_model_explanation_available",
        ),
        "source_lineage_and_policy_reproducible": as_bool(
            family,
            "source_lineage_and_policy_reproducible",
        ),
    }

    lag_diagnostic: dict[str, Any] = {}
    for lag in (1, 2):
        if lag in lag_by_days:
            row = lag_by_days[lag]
            lag_diagnostic[str(lag)] = {
                "model_brier": as_float(row, "model_brier"),
                "brier_skill_vs_prior_year": as_float(row, "brier_skill_vs_prior_year"),
                "candidate_lift": as_float(row, "candidate_cell_standardized_lift"),
            }

    return {
        "solution_id": solution["id"],
        "candidate": {
            "config_id": selected,
            "model": solution["selected_model"],
            "feature_set": "base + CNY/RUB market basis",
            "horizon": int(horizon),
            "development_years": solution["development_years"],
            "status": "offline_candidate_shadow_only",
        },
        "headline_metrics": {
            "cell_standardized_lift": {
                "value": lift,
                "ci95": [
                    lift_low,
                    as_float(uncertainty, "candidate_cell_standardized_lift_ci95_high"),
                ],
                "gate_passed": metric_gates["lift"],
            },
            "forward_official_reference_advantage_bps": {
                "value": forward_bps,
                "ci95": [
                    forward_low,
                    as_float(
                        uncertainty,
                        "candidate_cell_standardized_forward_bps_delta_ci95_high",
                    ),
                ],
                "gate_passed": metric_gates["forward_advantage"],
                "warning": "Historical official-reference proxy, not realized customer saving.",
            },
            "weekly_coverage_1_to_2_candidates": {
                "value": weekly_coverage,
                "required": float(gates["minimum_weekly_coverage"]),
                "mean_candidates_per_corridor_week": as_float(
                    metrics,
                    "mean_candidate_signals_per_corridor_week",
                ),
                "gate_passed": metric_gates["weekly_coverage"],
            },
        },
        "model_selection_evidence": {
            "base_brier": as_float(family, "base_brier"),
            "candidate_brier": as_float(family, "candidate_brier"),
            "relative_brier_improvement": relative_brier,
            "delta_brier_ci95": [
                as_float(family, "ci95_low"),
                as_float(family, "ci95_high"),
            ],
            "holm_adjusted_p": as_float(family, "holm_p_brier_improvement"),
            "years_improved": as_int(family, "years_brier_improved"),
            "fold_corridor_cells_improved": as_int(
                family,
                "fold_corridor_cells_brier_improved",
            ),
            "fold_corridor_cells_total": 15,
            "candidate_signals": as_int(metrics, "candidate_signal_count"),
            "candidate_hit_rate": as_float(metrics, "candidate_hit_rate"),
            "random_day_hit_rate": as_float(metrics, "baseline_hit_rate"),
        },
        "research_gates": metric_gates,
        "research_gate_failures": research_failures,
        "production_checks": production_checks,
        "production_ready": all(production_checks.values()) and not research_failures,
        "timing_diagnostic": {
            "moex_lag_days": lag_diagnostic,
            "conclusion": "D-1 close carries edge; a second day of lag removes Brier skill.",
        },
        "inferential_status": selection["inferential_status"],
        "protocol_note": selection["protocol_note"],
        "warnings": selection["warnings"],
    }
