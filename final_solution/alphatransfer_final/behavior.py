"""Optional client-readiness research preview, using only available observations.

This module does not alter market probabilities, authorize delivery, or estimate
uplift. Synthetic inputs always remain explicitly labeled as simulation.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _result(status: str, reasons: list[str], *, source: str | None = None,
            ready: bool | None = None, simulation: bool = False) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "simulation": simulation,
        "preview_ready": ready,
        "production_eligible": False,
        "suppression_reasons": reasons,
        "fx_probability_modified": False,
        "causal_uplift_estimated": False,
        "effect_claim": "Not estimated; synthetic scenarios do not identify business uplift.",
        "next_gates": ["market_quality", "TTL", "timezone", "live_quote", "shared_CRM", "production_promotion"],
    }


def _calendar_date(value: Any, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"invalid_date:{field}")
    try:
        # as_of is a calendar-day API, not an intraday freshness check. Existing
        # product gates remain responsible for timezone-aware TTL/quote times.
        if "T" in value or " " in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid_date:{field}") from exc


def build_behavior_preview(context: dict[str, Any] | None, as_of: date) -> dict[str, Any]:
    """Evaluate optional ex-ante readiness; fail closed on absent/future data.

    ``context`` fields: source (bank_observed/synthetic/demo), known_at,
    historical_transfer_count, expected_transfer_date, available_balance_sufficient,
    recent_explicit_intent, last_completed_transfer, urgency_declared,
    allow_fx_notifications, remaining_shared_crm_slots_7d. Any observation field
    ending in _at or _through is validated; expected_transfer_date may be future
    because it denotes a forecast or user plan already known at known_at.

    Missing context returns disabled. Missing input/provenance returns unavailable.
    Malformed or future observations return rejected. ``preview_ready`` is only a
    research display field; production_eligible is False in every return state.
    """
    if context is None or context == {}:
        return _result("disabled", ["behavior_context_absent"])
    if not isinstance(context, dict):
        return _result("rejected", ["behavior_context_must_be_object"])
    if context.get("enabled") is False:
        return _result("disabled", ["behavior_preview_disabled"])
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        return _result("rejected", ["as_of_must_be_calendar_date"])
    source = context.get("source")
    simulation = source in ("synthetic", "demo") or context.get("synthetic") is True
    if source not in ("bank_observed", "synthetic", "demo"):
        return _result("unavailable", ["behavior_source_missing_or_unsupported"], simulation=simulation)
    if source == "bank_observed" and context.get("synthetic") is True:
        return _result("rejected", ["conflicting_synthetic_provenance"], source=source, simulation=True)
    if context.get("known_at") is None:
        return _result("unavailable", ["behavior_known_at_missing"], source=source, simulation=simulation)
    parsed: dict[str, date] = {}
    try:
        for field, value in context.items():
            is_observation = field.endswith("_at") or field.endswith("_through") or field in {
                "last_completed_transfer", "last_transfer_date", "last_income_date",
                "last_intent_date", "balance_date", "history_end_date", "observation_date",
            }
            if value is None or not is_observation:
                continue
            observed = _calendar_date(value, field)
            parsed[field] = observed
            if observed > as_of:
                return _result("rejected", [f"future_behavior_observation:{field}"], source=source, simulation=simulation)
        # A context snapshot cannot claim it already contained later observations.
        known_at=parsed['known_at']
        if any(d>known_at for field,d in parsed.items() if field!='known_at'):
            return _result("rejected", ["observation_after_context_known_at"], source=source, simulation=simulation)
        expected = context.get("expected_transfer_date")
        expected = _calendar_date(expected, "expected_transfer_date") if expected is not None else None
    except ValueError as exc:
        return _result("rejected", [str(exc)], source=source, simulation=simulation)
    # Unrecognized feature names must not create an accidental permissive gate.
    required = ("historical_transfer_count", "available_balance_sufficient", "recent_explicit_intent",
                "urgency_declared", "allow_fx_notifications", "remaining_shared_crm_slots_7d")
    missing=[key for key in required if key not in context or context[key] is None]
    if missing:
        return _result("unavailable", [f"behavior_field_missing:{key}" for key in missing], source=source, simulation=simulation)
    boolean_fields=("available_balance_sufficient", "recent_explicit_intent", "urgency_declared", "allow_fx_notifications")
    invalid=[key for key in boolean_fields if not isinstance(context[key],bool)]
    integer_fields=("historical_transfer_count", "remaining_shared_crm_slots_7d")
    invalid += [key for key in integer_fields if isinstance(context[key],bool) or not isinstance(context[key],int) or context[key]<0]
    if invalid:
        return _result("rejected", [f"invalid_behavior_field:{key}" for key in invalid], source=source, simulation=simulation)
    reasons=[]
    if not context['allow_fx_notifications']:
        reasons.append('fx_notifications_not_consented')
    if context['remaining_shared_crm_slots_7d']==0:
        reasons.append('shared_crm_cap')
    if context['urgency_declared']:
        reasons.append('urgent_transfer_use_organic_flow')
    if not context['available_balance_sufficient']:
        reasons.append('available_balance_not_confirmed')
    last=parsed.get('last_completed_transfer') or parsed.get('last_completed_transfer_at')
    if last is not None and (as_of-last).days<3:
        reasons.append('recent_transfer')
    phase=(context['historical_transfer_count']>=3 and expected is not None and 0<=(expected-as_of).days<=7)
    if not (phase or context['recent_explicit_intent']):
        reasons.append('readiness_not_supported')
    ready=not reasons
    result=_result('simulation' if simulation else ('ready' if ready else 'suppressed'), reasons,
                   source=source, ready=ready, simulation=simulation)
    result['research_thresholds']={'minimum_history_events':3,'forecast_window_days':7,'post_transfer_suppression_days':3}
    result['timing_resolution']='calendar_day; intraday availability and freshness require existing product checks'
    return result
