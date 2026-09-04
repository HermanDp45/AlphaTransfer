"""Fail-closed conversion of a market candidate into a product decision."""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .artifacts import load_json, read_csv
from .config import PipelineConfig


CURRENCY_COPY = {
    "AMD": ("армянского драма", "Армению"),
    "KGS": ("кыргызского сома", "Кыргызстан"),
    "KZT": ("казахстанского тенге", "Казахстан"),
    "TJS": ("таджикского сомони", "Таджикистан"),
    "UZS": ("узбекского сума", "Узбекистан"),
}


def _truth(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _from_full_predictions(
    path: Path,
    config_id: str,
    horizon: int,
    as_of: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"date", "corridor", "probability", "candidate_signal", "signal"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Prediction file misses columns: {', '.join(sorted(missing))}")
        for row in reader:
            if (
                row.get("config_id") == config_id
                and row.get("horizon_cbr_rows_pub_proxy") == str(horizon)
                and row["date"] == as_of.isoformat()
            ):
                rows.append(
                    {
                        "date": row["date"],
                        "corridor": row["corridor"],
                        "probability": float(row["probability"]),
                        "candidate_signal": _truth(row["candidate_signal"]),
                        "portfolio_signal": _truth(row["signal"]),
                        "moex_observation_date": row.get("moex_cnyrub_observation_date", ""),
                        "moex_available_date": row.get("moex_cnyrub_available_date", ""),
                        "cbr_observation_date": row.get("cbr_fx_observation_date", ""),
                        "cbr_available_date": row.get("cbr_fx_available_date", ""),
                        "direct_market_age_days": row.get("direct_corridor_market_age_days", ""),
                    }
                )
    return rows


def load_candidates(
    config: PipelineConfig,
    as_of: date,
    predictions_path: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    solution = config.section("solution")
    if predictions_path is not None:
        if not predictions_path.is_file():
            raise FileNotFoundError(predictions_path)
        return (
            _from_full_predictions(
                predictions_path,
                solution["selected_config"],
                int(solution["primary_horizon"]),
                as_of,
            ),
            "full_historical_oot_predictions",
        )

    rows = read_csv(config.path("demo_candidates"))
    selected = []
    for row in rows:
        if row["date"] != as_of.isoformat():
            continue
        selected.append(
            {
                **row,
                "probability": float(row["probability"]),
                "candidate_signal": _truth(row["candidate_signal"]),
                "portfolio_signal": _truth(row["portfolio_signal"]),
            }
        )
    if not selected:
        raise RuntimeError(
            "The checked-in demo snapshot covers only the default date. "
            "Pass --predictions or use --rebuild for another --as-of date."
        )
    return selected, "checked_in_historical_demo_snapshot"


def _load_client_context(config: PipelineConfig, path: Path | None) -> dict[str, Any]:
    return load_json(path or config.path("default_client_context"))


def build_product_decision(
    config: PipelineConfig,
    scorecard: dict[str, Any],
    as_of: date,
    predictions_path: Path | None = None,
    client_context_path: Path | None = None,
) -> dict[str, Any]:
    candidates, candidate_source = load_candidates(config, as_of, predictions_path)
    context = _load_client_context(config, client_context_path)
    policy = config.section("policy")
    solution = config.section("solution")

    relevant = set(context.get("relevant_corridors", solution["corridors"]))
    relevant_candidates = [
        row
        for row in candidates
        if row["candidate_signal"] and row["corridor"] in relevant
    ]
    selected = max(
        relevant_candidates,
        key=lambda row: (row["probability"], row["portfolio_signal"]),
        default=None,
    )

    batch_hour, batch_minute = map(int, policy["batch_time_msk"].split(":"))
    msk = ZoneInfo("Europe/Moscow")
    generated_at = datetime.combine(
        as_of,
        time(batch_hour, batch_minute),
        tzinfo=msk,
    )
    expires_at = generated_at + timedelta(hours=int(policy["signal_ttl_hours"]))
    client_zone = ZoneInfo(context["timezone"])
    generated_at_local = generated_at.astimezone(client_zone)
    window_start = time.fromisoformat(policy["delivery_window_local_start"])
    window_end = time.fromisoformat(policy["delivery_window_local_end"])

    suppression: list[str] = []
    if selected is None:
        suppression.append("no_relevant_portfolio_signal_on_as_of_date")
    if context.get("urgent_transfer"):
        suppression.append("urgent_segment_excluded_from_proactive_contact")
    if int(context.get("marketing_contacts_last_7d", 0)) >= int(
        policy["max_marketing_contacts_7d"]
    ):
        suppression.append("shared_marketing_contact_budget_exhausted")
    if not window_start <= generated_at_local.time().replace(tzinfo=None) <= window_end:
        suppression.append("outside_client_local_delivery_window")
    for gate_name, passed in scorecard["research_gates"].items():
        if not passed:
            suppression.append(f"research_gate_failed:{gate_name}")
    if not scorecard["production_checks"]["prospective_shadow_complete"]:
        suppression.append("prospective_shadow_not_completed")
    if not scorecard["production_checks"]["production_data_rights_verified"]:
        suppression.append("production_market_data_rights_not_verified")
    if not scorecard["production_checks"]["historical_publication_timestamps_verified"]:
        suppression.append("publication_timestamps_not_verified")
    if not scorecard["production_checks"]["per_signal_explanation_available"]:
        suppression.append("per_signal_model_explanation_not_available")
    if not scorecard["production_checks"]["source_lineage_and_policy_reproducible"]:
        suppression.append("production_source_lineage_not_reproducible")
    if not scorecard["production_checks"]["production_track_promoted"]:
        suppression.append("model_not_production_promoted")

    alpha_quote = context.get("alpha_quote")
    if alpha_quote is None:
        suppression.append("alpha_executable_quote_not_provided")
    else:
        observed_at = datetime.fromisoformat(alpha_quote["observed_at"])
        if observed_at.tzinfo is None:
            raise ValueError("alpha_quote.observed_at must contain a timezone")
        if observed_at.astimezone(timezone.utc) < generated_at.astimezone(timezone.utc):
            suppression.append("alpha_quote_predates_signal")
        if observed_at.astimezone(timezone.utc) > expires_at.astimezone(timezone.utc):
            suppression.append("alpha_quote_observed_after_signal_expiry")
        if abs(float(alpha_quote.get("move_since_signal_bps", 0.0))) > float(
            policy["max_quote_move_bps"]
        ):
            suppression.append("market_moved_since_signal")

    preview: dict[str, Any] | None = None
    market_signal: dict[str, Any] | None = None
    if selected is not None:
        corridor = selected["corridor"]
        currency_name, country = CURRENCY_COPY[corridor]
        title = f"Курс {currency_name} в благоприятной зоне"
        body = (
            f"Сейчас курс {currency_name} в благоприятной зоне "
            "по исторической шкале. Если перевод уже планировался, "
            "проверьте сумму к получению в приложении."
        )
        market_signal = {
            "as_of": as_of.isoformat(),
            "corridor": f"RUB_{corridor}",
            "country": country,
            "score": selected["probability"],
            "scenario": "NOW_FAVORABLE",
            "data_as_of": {
                "moex_observation_date": selected.get("moex_observation_date"),
                "moex_available_date": selected.get("moex_available_date"),
                "cbr_observation_date": selected.get("cbr_observation_date"),
                "cbr_available_date": selected.get("cbr_available_date"),
            },
            "generated_at": generated_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "offline_target": "NOW hit over five effective CBR rows",
            "client_copy_contains_forecast": False,
            "client_selection_policy": (
                "highest score among candidate signals in the client's relevant corridors"
            ),
        }
        preview = {
            "label": "UX preview; not an authorized client contact",
            "push": {"title": title, "body": body},
            "disclaimer": (
                "Индикатор описывает текущий момент и не прогнозирует будущий курс."
            ),
            "tap_action": "Open existing transfer form with saved recipient; recheck Alpha quote.",
            "stale_state": (
                "Момент изменился. Показываем только актуальный курс "
                "и нейтральную шкалу."
            ),
        }

    return {
        "mode": "historical_product_preview",
        "as_of": as_of.isoformat(),
        "candidate_source": candidate_source,
        "client_context": {
            "client_id": context.get("client_id"),
            "timezone": context.get("timezone"),
            "relevant_corridors": sorted(relevant),
            "synthetic": str(context.get("client_id", "")).startswith("synthetic-"),
        },
        "market_signal": market_signal,
        "delivery": {
            "eligible_to_send": not suppression and scorecard["production_ready"],
            "suppressed_reasons": suppression,
            "local_delivery_window": [
                policy["delivery_window_local_start"],
                policy["delivery_window_local_end"],
            ],
            "candidate_generated_at_local": generated_at_local.isoformat(),
            "shared_marketing_contacts_limit_7d": policy["max_marketing_contacts_7d"],
            "external_message_sent": False,
        },
        "experience_preview": preview,
    }
