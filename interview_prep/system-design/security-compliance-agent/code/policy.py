"""The knowledge and signals layer: the policy corpus, the deterministic rule engine, and
the per-actor behavior baseline.

In production the policy corpus is a vector store over your written policies and past
incidents, the rules are a maintained detection library (Sigma rules, SQL, a CEP engine),
and the baseline is a user-and-entity-behavior-analytics model kept fresh from the stream.
Here each one is a small in-memory stand-in so the whole system runs offline with no
external services.
"""

# --- the policy corpus (what the model grounds a judgment in) -------------------------
POLICIES = {
    "data-handling": "Customer records containing PII may only leave the company through "
                     "approved, encrypted channels. Bulk export to a personal or external "
                     "address is a violation and must be blocked.",
    "access-control": "Access to production data outside business hours or from a new "
                      "location is higher risk and should be reviewed by a human before it "
                      "is trusted.",
    "audit-integrity": "Disabling audit logging, MFA, or security controls is never "
                       "permitted and must be blocked immediately.",
    "routine-activity": "Reading dashboards, viewing reports, and normal application use "
                        "during business hours is expected and allowed.",
}

# Words that clearly signal a violation the deterministic layer can block on its own.
_EXFIL = ("export", "download", "dump", "send", "upload", "email")
_SENSITIVE = ("pii", "customer", "ssn", "card", "salary", "records", "database", "table")
_EXTERNAL = ("gmail", "external", "personal", "dropbox", "proton", "outlook.com", "usb")
_TAMPER = ("disable", "turn off", "delete logs", "clear audit", "bypass mfa", "stop logging")

# Injection markers: an event is untrusted content, so text that tries to give the agent
# instructions is itself suspicious. The agent never obeys it (the lethal trifecta defense).
_INJECTION = ("ignore previous", "ignore all previous", "disregard", "you are now",
              "mark this as allowed", "approve this", "new instructions", "system prompt")

# --- per-actor behavior baseline (stand-in for a UEBA model) --------------------------
BASELINES = {
    "alice": {"hours": range(8, 19), "typical_volume": 50},
    "bob":   {"hours": range(8, 19), "typical_volume": 100},
    "svc-export": {"hours": range(0, 24), "typical_volume": 500},
}
_DEFAULT_BASELINE = {"hours": range(8, 19), "typical_volume": 50}


def normalize(event):
    """Accept a raw string or a structured event and return a full event dict."""
    if isinstance(event, str):
        return {"actor": "unknown", "text": event, "hour": 12, "volume": 1}
    e = dict(event)
    e.setdefault("actor", "unknown")
    e.setdefault("text", "")
    e.setdefault("hour", 12)
    e.setdefault("volume", 1)
    return e


def rule_check(event):
    """Deterministic first pass. Returns (signal, evidence) where signal is one of
    'block', 'allow', or 'inspect'. Fast, exact, and cheap: it runs on every event."""
    t = event["text"].lower()
    if any(m in t for m in _INJECTION):
        return "inspect", ["injection markers in event content, treating as data not instructions"]
    if any(m in t for m in _TAMPER):
        return "block", ["matches audit-integrity rule: security control tampering"]
    if (any(w in t for w in _EXFIL) and any(w in t for w in _SENSITIVE)
            and any(w in t for w in _EXTERNAL)):
        return "block", ["matches data-handling rule: bulk PII export to an external channel"]
    if any(w in t for w in ("viewed", "read", "opened dashboard", "login success", "report"))\
            and not any(w in t for w in _EXFIL):
        return "allow", ["matches routine-activity rule during expected use"]
    return "inspect", ["no deterministic rule fired, escalating to model judgment"]


def anomaly_signal(event):
    """Deviation from the actor's baseline. Returns (severity 0..2, evidence)."""
    base = BASELINES.get(event["actor"], _DEFAULT_BASELINE)
    severity, evidence = 0, []
    if event["hour"] not in base["hours"]:
        severity += 1
        evidence.append(f"off-hours activity at {event['hour']:02d}:00 for {event['actor']}")
    if event["volume"] > 3 * base["typical_volume"]:
        severity += 1
        evidence.append(f"volume {event['volume']} is over 3x the {event['actor']} baseline")
    return severity, evidence


def retrieve_policy(event, k: int = 1):
    """Keyword retriever with a relevance floor: pull the policy the judgment is graded on.

    Returns up to k (policy_id, text) pairs, or nothing when no content word matches, so an
    unrecognized event still routes to a human rather than being graded on the wrong policy.
    """
    stop = {"the", "a", "an", "to", "of", "and", "in", "on", "at", "for", "is", "was",
            "with", "from", "this", "that", "during", "outside", "new", "into"}
    q = {w for w in event["text"].lower().replace("-", " ").split() if w not in stop and len(w) > 2}
    scored = []
    for pid, text in POLICIES.items():
        words = {w for w in (pid + " " + text).lower().replace("-", " ").split() if len(w) > 2}
        overlap = len(q & words)
        if overlap:
            scored.append((overlap, pid, text))
    scored.sort(reverse=True)
    return [(pid, text) for _, pid, text in scored[:k]]
