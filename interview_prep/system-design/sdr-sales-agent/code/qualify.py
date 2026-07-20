"""Qualification: a scored, explainable decision, not a yes/no black box.

The score is a transparent scorecard over the enriched signals. Two parts, both drawn from
the ideal customer profile (ICP):

  fit    : does this account look like the ones you win (size, industry, seniority)?
  intent : is there evidence they are in-market right now (inbound action, funding, hiring)?

Every point added carries a reason code that names the signal it came from, so a human can
read WHY a lead scored the way it did, an auditor can trace it, and you can evaluate the
score against how these leads actually converted. In production you derive the weights from
your own closed-won history rather than hand-setting them; the weights here are illustrative.
"""

# ICP thresholds and weights (illustrative; derive yours from closed-won data).
MIN_EMPLOYEES = 200
ICP_INDUSTRIES = {"b2b-saas", "fintech", "healthtech", "devtools"}
SENIOR_TITLES = ("chief", "vp", "vice president", "head", "director", "founder", "cto", "ceo")

QUALIFIED_AT = 60   # >= this: hand to drafting
NURTURE_AT = 35     # in [NURTURE_AT, QUALIFIED_AT): keep warm, no outreach yet
                    # < NURTURE_AT: disqualify


def _signal(signals, field):
    for s in signals:
        if s["field"] == field:
            return s
    return None


def qualify(signals) -> dict:
    """Return {score, disposition, reasons:[{code, points, evidence_id, detail}]}."""
    reasons = []

    emp = _signal(signals, "employees")
    if emp and isinstance(emp["value"], int) and emp["value"] >= MIN_EMPLOYEES:
        reasons.append({"code": "fit.size", "points": 25, "evidence_id": emp["id"],
                        "detail": f"{emp['value']} employees, at or above ICP floor"})

    ind = _signal(signals, "industry")
    if ind and str(ind["value"]).lower() in ICP_INDUSTRIES:
        reasons.append({"code": "fit.industry", "points": 20, "evidence_id": ind["id"],
                        "detail": f"industry {ind['value']} is in the ICP"})

    title = _signal(signals, "title")
    if title and any(t in str(title["value"]).lower() for t in SENIOR_TITLES):
        reasons.append({"code": "fit.seniority", "points": 20, "evidence_id": title["id"],
                        "detail": f"title '{title['value']}' has buying influence"})

    action = _signal(signals, "inbound_action")
    if action:
        reasons.append({"code": "intent.inbound", "points": 25, "evidence_id": action["id"],
                        "detail": f"recent inbound action: {action['value']}"})

    funding = _signal(signals, "funding")
    if funding and funding.get("verified"):
        reasons.append({"code": "intent.funding", "points": 10, "evidence_id": funding["id"],
                        "detail": f"funding signal: {funding['value']}"})

    score = sum(r["points"] for r in reasons)
    if score >= QUALIFIED_AT:
        disposition = "qualified"
    elif score >= NURTURE_AT:
        disposition = "nurture"
    else:
        disposition = "disqualified"
    return {"score": score, "disposition": disposition, "reasons": reasons}
