"""Scenario rules and delivery controls, separate from probabilities."""

from __future__ import annotations

from datetime import date, timedelta


def explainable_indicator(row: dict) -> list[str]:
    reasons = []
    if float(row.get("percentile_20", 1)) <= 0.25:
        reasons.append("cbr_in_bottom_quartile_20")
    if float(row.get("decreasing_run", 0)) >= 2:
        reasons.append("cbr_falling_streak")
    if float(row.get("return_3", 0)) < 0:
        reasons.append("cbr_3_observation_momentum_down")
    return reasons


def raw_candidates(rows: list[dict], probabilities: list[float], threshold: float, rebound_bps: float, closing_lookback: int) -> list[dict]:
    candidates: list[dict] = []
    for i, (row, probability) in enumerate(zip(rows, probabilities)):
        reasons = explainable_indicator(row)
        if probability >= threshold and reasons:
            candidates.append({"index": i, "date": row["date"], "scenario": "favorable_now", "confidence": probability,
                               "evidence": reasons, "rate": float(row["cbr_rate"])})
        start = max(0, i - closing_lookback)
        strong = [(j, probabilities[j]) for j in range(start, i) if probabilities[j] >= threshold and explainable_indicator(rows[j])]
        if strong:
            prior_i, prior_p = max(strong, key=lambda x: x[1])
            rebound = (float(row["cbr_rate"]) / float(rows[prior_i]["cbr_rate"]) - 1) * 10_000
            if rebound >= rebound_bps:
                candidates.append({"index": i, "date": row["date"], "scenario": "window_closing",
                                   "confidence": min(0.999, max(probability, prior_p) + min(rebound / 10_000, .1)),
                                   "evidence": [f"strong_candidate_on_{rows[prior_i]['date']}", f"rebound_{rebound:.1f}_bps"],
                                   "rate": float(row["cbr_rate"])})
    return candidates


def apply_policy(candidates: list[dict], cooldown_sessions: int, cap_7d: int, scenario_utility: dict[str, float] | None = None) -> list[dict]:
    scenario_utility = scenario_utility or {}
    grouped: dict[int, list[dict]] = {}
    for item in candidates:
        grouped.setdefault(int(item["index"]), []).append(item)
    chosen: list[dict] = []
    last_index = -10_000
    for index in sorted(grouped):
        options = grouped[index]
        winner = max(options, key=lambda x: (scenario_utility.get(x["scenario"], 0.0), x["confidence"]))
        for loser in options:
            if loser is not winner:
                chosen.append({**loser, "policy_eligible": False, "suppressed_reason": "scenario_conflict"})
        if index - last_index <= cooldown_sessions:
            winner = {**winner, "policy_eligible": False, "suppressed_reason": "cooldown"}
        else:
            cutoff = date.fromisoformat(winner["date"]) - timedelta(days=6)
            recent = [x for x in chosen if x.get("policy_eligible") and date.fromisoformat(x["date"]) >= cutoff]
            if len(recent) >= cap_7d:
                winner = {**winner, "policy_eligible": False, "suppressed_reason": "weekly_cap"}
            else:
                winner = {**winner, "policy_eligible": True, "suppressed_reason": None}
                last_index = index
        chosen.append(winner)
    return sorted(chosen, key=lambda x: (int(x["index"]), bool(x.get("policy_eligible"))))


def public_signal(candidate: dict, row: dict, explanations: list[dict], oot_lift: float, min_lift: float) -> dict:
    eligible = bool(candidate.get("policy_eligible")) and oot_lift >= min_lift
    reason = candidate.get("suppressed_reason")
    if candidate.get("policy_eligible") and oot_lift < min_lift:
        reason = "oot_lift_below_gate"
    return {
        "as_of": row["date"], "corridor": "RUB_KZT", "scenario": candidate["scenario"],
        "confidence": round(float(candidate["confidence"]), 4), "eligible_to_send": eligible,
        "rate_snapshot": {k: round(float(row[k]), 8) for k in ("cbr_rate", "nbk_rate", "moex_close", "usd_cross_rate", "cny_cross_rate")},
        "source_freshness": {"cbr": {"age_days": int(row["cbr_age_days"]), "is_fresh": bool(row["cbr_is_fresh"])},
                             "nbk": {"age_days": int(row["nbk_age_days"]), "is_fresh": bool(row["nbk_is_fresh"])},
                             "moex": {"age_days": 0, "is_fresh": True}},
        "evidence": candidate["evidence"] + explanations,
        "suppressed_reason": reason,
    }
