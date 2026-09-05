"""Pure research preview of a client-level gate; sends nothing.

Thresholds below are scenario assumptions. They must be fitted/validated on
consented real bank histories before promotion. FX probability is never changed.
"""
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class BehaviorContext:
    as_of: date
    historical_transfer_count: int
    expected_transfer_date: date | None
    available_balance_sufficient: bool | None
    recent_explicit_intent: bool
    last_completed_transfer: date | None
    urgency_declared: bool
    allow_fx_notifications: bool
    remaining_shared_crm_slots_7d: int
    known_at: date


def preview_readiness_gate(context: BehaviorContext) -> dict:
    """Keep UX for organic visits available; gate only the research push candidate."""
    c=context
    if c.known_at>c.as_of:
        raise ValueError('behavior context is unavailable as of decision date')
    reasons=[]
    if not c.allow_fx_notifications: reasons.append('fx_notifications_not_consented')
    if c.remaining_shared_crm_slots_7d<=0: reasons.append('shared_crm_cap')
    if c.urgency_declared: reasons.append('urgent_transfer_use_organic_flow')
    if c.available_balance_sufficient is not True: reasons.append('available_balance_not_confirmed')
    if c.last_completed_transfer and (c.as_of-c.last_completed_transfer).days<3:
        reasons.append('recent_transfer')
    # Explicit intent is a client action observed already, never a future label.
    phase=(c.historical_transfer_count>=3 and c.expected_transfer_date is not None
           and 0<=(c.expected_transfer_date-c.as_of).days<=7)
    if not (phase or c.recent_explicit_intent):reasons.append('readiness_not_supported')
    return dict(status='RESEARCH_PREVIEW_ONLY',eligible=not reasons,suppression_reasons=reasons,
                fx_probability_modified=False,causal_uplift_claimed=False,
                next_gate='existing TTL/timezone/live-quote/global-CRM checks')
