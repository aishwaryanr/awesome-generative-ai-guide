"""Lead enrichment: the tool that turns a bare inbound lead into a set of signals.

In production this is a set of tool calls: your CRM (Salesforce, HubSpot) for the record
and its engagement history, plus enrichment providers (Apollo, Clearbit, ZoomInfo style)
for firmographic, contact, and intent data. Here it is a small in-memory table keyed by
email domain, so the whole system runs offline with no external services.

Every signal carries its provenance: a `source` and an `as_of` date, and a `verified` flag.
That provenance is load-bearing. The message drafter is only allowed to reference a signal
that is verified, and the compliance guardrail uses the same flag to catch any claim about a
prospect that is not backed by a real, current signal. A stale or rumored signal is the
seed of a fabricated claim, so it is marked verified=False and never grounds a sentence.
"""

# Each signal: {id, kind, field, value, source, as_of, verified}
#   kind is one of: firmographic | contact | intent | suppression
# The suppression signal (opt_out) is authoritative: if present and True, no outreach is allowed.

_DB = {
    # A strong, well-qualified inbound lead with verified signals.
    "acme.io": {
        "opt_out": False,
        "signals": [
            {"id": "s1", "kind": "firmographic", "field": "employees", "value": 1200,
             "source": "crm:account", "as_of": "2026-07-01", "verified": True},
            {"id": "s2", "kind": "firmographic", "field": "industry", "value": "b2b-saas",
             "source": "crm:account", "as_of": "2026-07-01", "verified": True},
            {"id": "s3", "kind": "contact", "field": "title", "value": "VP Engineering",
             "source": "enrichment:apollo", "as_of": "2026-06-28", "verified": True},
            {"id": "s4", "kind": "intent", "field": "inbound_action", "value": "requested a demo",
             "source": "crm:activity", "as_of": "2026-07-14", "verified": True},
            {"id": "s5", "kind": "intent", "field": "funding", "value": "closed a Series C in June 2026",
             "source": "enrichment:news", "as_of": "2026-06-20", "verified": True},
        ],
    },
    # A qualified-fit lead whose ONLY personalization hook is an UNVERIFIED, rumored signal.
    # The drafter may reach for it; the faithfulness guardrail is what stops it becoming a
    # fabricated claim in the email.
    "startup.dev": {
        "opt_out": False,
        "signals": [
            {"id": "s1", "kind": "firmographic", "field": "employees", "value": 400,
             "source": "crm:account", "as_of": "2026-07-02", "verified": True},
            {"id": "s2", "kind": "firmographic", "field": "industry", "value": "b2b-saas",
             "source": "crm:account", "as_of": "2026-07-02", "verified": True},
            {"id": "s3", "kind": "contact", "field": "title", "value": "Head of Data",
             "source": "enrichment:apollo", "as_of": "2026-06-15", "verified": True},
            {"id": "s4", "kind": "intent", "field": "inbound_action", "value": "downloaded the evals guide",
             "source": "crm:activity", "as_of": "2026-07-13", "verified": True},
            # Rumored, single-blog sourced, and unconfirmed: NOT safe to state as fact.
            {"id": "s5", "kind": "intent", "field": "funding", "value": "rumored to be raising a round",
             "source": "enrichment:blog-rumor", "as_of": "2026-05-01", "verified": False},
        ],
    },
    # A contact who has opted out. Authoritative suppression: no outreach may be drafted or sent.
    "bigco.com": {
        "opt_out": True,
        "signals": [
            {"id": "s1", "kind": "firmographic", "field": "employees", "value": 8000,
             "source": "crm:account", "as_of": "2026-07-01", "verified": True},
            {"id": "s2", "kind": "contact", "field": "title", "value": "Director of Platform",
             "source": "enrichment:apollo", "as_of": "2026-06-30", "verified": True},
            {"id": "s3", "kind": "suppression", "field": "opt_out", "value": True,
             "source": "crm:consent", "as_of": "2026-05-10", "verified": True},
        ],
    },
}

# Signals for a lead we cannot match to any company: a personal mailbox, no firmographics.
_UNKNOWN = {
    "opt_out": False,
    "signals": [
        {"id": "s1", "kind": "contact", "field": "title", "value": "unknown",
         "source": "enrichment:apollo", "as_of": "2026-07-10", "verified": True},
    ],
}


def domain_of(email: str) -> str:
    return email.split("@", 1)[1].strip().lower() if "@" in email else ""


def enrich(email: str) -> dict:
    """Return {"opt_out": bool, "signals": [...]} for a lead, from CRM plus enrichment.

    Personal-mailbox domains resolve to no company, which is what makes an unqualified lead
    look unqualified rather than crashing the pipeline.
    """
    domain = domain_of(email)
    if domain in {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", ""}:
        return dict(_UNKNOWN)
    return _DB.get(domain, dict(_UNKNOWN))
