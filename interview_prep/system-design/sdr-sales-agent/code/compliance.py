"""The brand and compliance guardrail: the real-time check that runs before any human sees
a draft, and the reason a wrong action stays impossible rather than merely unlikely.

It enforces three things, in order of severity:

  1. Suppression / opt-out. If the contact has opted out, no outreach is allowed, full stop.
     Emailing a suppressed contact is both a trust failure and a regulatory one (CAN-SPAM
     requires honoring opt-outs; GDPR requires a lawful basis). This is an absolute block.

  2. Faithfulness. Every personalized claim in the draft must map to a VERIFIED signal.
     A claim backed only by an unverified or stale signal is a fabricated claim about the
     prospect, the worst failure for this system, so the draft is blocked.

  3. Required elements. A commercial message must identify itself as outreach, carry a valid
     physical postal address, and offer a clear opt-out. These are code-checkable.

Guardrails must be fast and reliable before they are sophisticated, so all three checks are
deterministic code. Tone and brand-voice quality are judged separately (an LLM judge in the
eval layer), off the live path.
"""

REQUIRED_FOOTER_MARKERS = ("unsubscribe", "opt out", "opt-out")


def check(draft: dict, signals, opt_out: bool) -> dict:
    """Return {"ok": bool, "failures": [...], "checks": {...}}."""
    failures = []

    # 1. Suppression / opt-out is absolute.
    if opt_out:
        failures.append("contact has opted out; outreach is suppressed")

    # 2. Faithfulness: each claim must map to a verified signal.
    verified_ids = {s["id"] for s in signals if s.get("verified")}
    for c in draft.get("claims", []):
        if c.get("evidence_id") not in verified_ids:
            failures.append(f"unfaithful claim without a verified signal: \"{c.get('text')}\"")

    # 3. Required compliance elements in the message body.
    body = (draft.get("body") or "").lower()
    has_optout = any(m in body for m in REQUIRED_FOOTER_MARKERS)
    has_address = "postal" in body or "st," in body or "street" in body or "suite" in body
    if not has_optout:
        failures.append("missing a clear opt-out mechanism")
    if not has_address:
        failures.append("missing a valid physical postal address")

    return {
        "ok": not failures,
        "failures": failures,
        "checks": {"opt_out": opt_out, "verified_claims": len(verified_ids)},
    }


def compliant_footer() -> str:
    """A footer that satisfies the code-checkable CAN-SPAM elements. Address is illustrative."""
    return ("You are receiving this because you engaged with LevelUp Labs. "
            "Reply STOP or click unsubscribe to opt out. "
            "LevelUp Labs, 500 Market Street, Suite 200, San Francisco, CA 94105.")
