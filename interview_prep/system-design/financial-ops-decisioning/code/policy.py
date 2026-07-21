"""Deterministic policy checks: the rulebook, run in code, not by the model.

The decision here is knowable, so it is a rule, not a judgment call. Every check is exact,
cheap, and reproducible, which is what an auditor needs. Each check returns a finding tagged
with the action it forces:

    "deny"   a hard policy violation (a non-reimbursable category)
    "route"  send to a human for sign-off (high-impact, out of policy, low confidence,
             a suspected duplicate, or a currency that needs conversion)

The overall decision resolves by precedence: any deny wins, then any route, otherwise approve.
Thresholds below are illustrative. Set yours from your own policy and track the false-approve
and false-deny rates separately, because they cost very different amounts (see the case study).
"""
from llm import REQUIRED_FIELDS
from audit import is_duplicate

HIGH_IMPACT_USD = 10000                       # at or above this, a human signs off, always
CONFIDENCE_FLOOR = 0.75                        # below this, extraction is too thin to decide on
NON_REIMBURSABLE = {"alcohol", "gift-card", "gift card", "personal", "entertainment"}
CATEGORY_LIMIT = {                             # per-category soft ceilings (route above them)
    "meals": 100,
    "travel": 2000,
    "office": 500,
    "software": 5000,
    "hardware": 5000,
}


def evaluate(fields: dict):
    """Return (decision, findings). decision in {approve, deny, human_review}."""
    findings = []
    amount = fields.get("amount")
    category = (fields.get("category") or "").strip().lower()
    confidence = fields.get("_confidence", 0.0)

    missing = [f for f in REQUIRED_FIELDS if fields.get(f) in (None, "", [])]
    if missing:
        findings.append(("incomplete", f"missing required fields: {', '.join(missing)}", "route"))
    if confidence < CONFIDENCE_FLOOR:
        findings.append(("low_confidence",
                         f"extraction confidence {confidence} below floor {CONFIDENCE_FLOOR}", "route"))

    if category in NON_REIMBURSABLE:
        findings.append(("non_reimbursable", f"category '{category}' is not reimbursable", "deny"))

    invoice_id = fields.get("invoice_id")
    if invoice_id and is_duplicate(invoice_id):
        findings.append(("duplicate", f"invoice {invoice_id} was already processed", "route"))

    currency = (fields.get("currency") or "USD").upper()
    if currency != "USD":
        findings.append(("fx", f"non-USD currency {currency} needs conversion sign-off", "route"))

    if isinstance(amount, (int, float)):
        if amount >= HIGH_IMPACT_USD:
            findings.append(("high_impact",
                             f"amount {amount} at or above high-impact threshold {HIGH_IMPACT_USD}", "route"))
        limit = CATEGORY_LIMIT.get(category)
        if limit is not None and amount > limit:
            findings.append(("over_limit", f"amount {amount} over {category} limit {limit}", "route"))

    actions = {f[2] for f in findings}
    if "deny" in actions:
        decision = "deny"
    elif "route" in actions:
        decision = "human_review"
    else:
        decision = "approve"
    return decision, findings
