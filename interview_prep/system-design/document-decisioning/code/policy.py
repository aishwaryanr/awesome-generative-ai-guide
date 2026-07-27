"""The underwriting policy: the deterministic half of the decision.

Some rules are hard and non-negotiable, so they belong in code where they run the same way
every time and are trivial to audit: appetite limits, over-insurance, prior-claims ceilings,
and the delegated-authority limit above which a human must sign off. The model is only
consulted for the borderline judgment the rules deliberately leave open (see llm.py).

Change these constants to change the policy. They are the knobs an underwriting manager owns.
"""

AUTO_BIND_LIMIT = 500_000   # coverage above this exceeds delegated authority -> human sign-off
MAX_PRIOR_CLAIMS = 2        # more than this is outside appetite -> decline
OLD_CONSTRUCTION_YEAR = 1940  # older than this is a judgment call -> flag for the model


def _decline(reason):
    return {"outcome": "decline", "reason": reason, "flags": []}


def _refer(reason):
    return {"outcome": "refer", "reason": reason, "flags": []}


def evaluate(fields: dict) -> dict:
    """Apply the deterministic policy to extracted fields.

    Returns {"outcome": "decline" | "refer" | "clean", "reason": str, "flags": [str, ...]}.
    "clean" means the rules found no reason to decline or escalate, and the model makes the
    final approve-or-refer call on any borderline flags.
    """
    v = {name: data["value"] for name, data in fields.items()}
    coverage = v.get("requested_coverage") or 0
    value = v.get("property_value") or 0
    claims = v.get("prior_claims") or 0
    year = v.get("year_built") or 9999

    # Hard declines: non-negotiable, checked first.
    if str(v.get("flood_zone", "")).lower() == "yes" and \
            str(v.get("flood_endorsement", "")).lower() != "yes":
        return _decline("property in a flood zone without a flood endorsement (outside appetite)")
    if coverage > value:
        return _decline(f"requested coverage {coverage} exceeds property value {value} (over-insurance)")
    if claims > MAX_PRIOR_CLAIMS:
        return _decline(f"prior claims {claims} above the maximum of {MAX_PRIOR_CLAIMS}")

    # High-impact: within appetite but above delegated authority -> a human must sign off.
    if coverage > AUTO_BIND_LIMIT:
        return _refer(f"requested coverage {coverage} above the auto-bind authority of {AUTO_BIND_LIMIT}")

    # Borderline: the rules pass but leave a judgment call for the model.
    flags = []
    if year < OLD_CONSTRUCTION_YEAR:
        flags.append(f"construction predates {OLD_CONSTRUCTION_YEAR}, so age-related risk needs judgment")

    return {"outcome": "clean", "reason": "within policy and delegated authority", "flags": flags}
