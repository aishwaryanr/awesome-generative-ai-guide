"""The model layer, provider-agnostic: draft a personalized outreach message.

The drafter is the fallible generator. Its one job is to write a short, personalized message
that references ONLY real signals about the prospect, and to attach to every personalized
claim the id of the signal it came from. The compliance guardrail (compliance.py) checks
those claims against the VERIFIED signal set and blocks the draft if any claim is not backed.

Two implementations behind one function:

- a deterministic offline drafter so the whole graph runs with no API key, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model
  and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not
  change the graph to change the model.

Selecting a model (any one of these):
    export SDR_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export SDR_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export SDR_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If SDR_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
"""
import json
import os

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    model = os.environ.get("SDR_AGENT_MODEL")
    provider = os.environ.get("SDR_AGENT_PROVIDER")
    if not model:
        for key, m, p in _AUTODETECT:
            if os.environ.get(key):
                model, provider = m, p
                break
    if not model:
        return None
    try:
        from langchain.chat_models import init_chat_model
        return init_chat_model(model, model_provider=provider) if provider else init_chat_model(model)
    except Exception:
        return None


def draft(lead: dict, signals) -> dict:
    """Return {subject, body, claims:[{text, evidence_id}]}. Provider-agnostic.

    `claims` is the audit trail the guardrail verifies: each personalized sentence is paired
    with the id of the signal that backs it.
    """
    model = _get_model()
    if model is not None:
        try:
            return _draft_with_model(model, lead, signals)
        except Exception:
            pass  # fall back to the offline drafter if the provider call fails
    return _draft_offline(lead, signals)


def _by_field(signals, field):
    for s in signals:
        if s["field"] == field:
            return s
    return None


# --- offline deterministic drafter (no provider) --------------------------------------

def _draft_offline(lead: dict, signals) -> dict:
    name = lead.get("name", "there").split()[0]
    title = _by_field(signals, "title")
    action = _by_field(signals, "inbound_action")
    funding = _by_field(signals, "funding")

    lines, claims = [f"Hi {name},"], []

    if action:
        lines.append(f"I saw that you {action['value']} recently. It is exactly the kind of "
                     "thing our team helps with.")
        claims.append({"text": f"you {action['value']}", "evidence_id": action["id"]})

    # The drafter reaches for a funding mention whenever a funding signal exists. When that
    # signal is unverified (a rumor), this is exactly the sentence the guardrail must catch.
    if funding:
        lines.append(f"Also, congratulations on the news that you {funding['value']}.")
        claims.append({"text": f"you {funding['value']}", "evidence_id": funding["id"]})

    role = f"as {title['value']}" if title else "on your team"
    lines.append(
        f"Given your work {role}, I would love to show you how teams like yours cut manual "
        "review time. Open to a 15-minute call next week?"
    )
    lines.append("Best,\nThe LevelUp Sales Team")

    return {"subject": "A quick idea for your team", "body": "\n\n".join(lines), "claims": claims}


# --- real path: any provider via init_chat_model --------------------------------------

_DRAFT_SYSTEM = (
    "You are a sales development rep drafting a short outreach email. Rules:\n"
    "1. Reference ONLY facts present in the SIGNALS list. Never invent a claim about the prospect.\n"
    "2. For every personalized sentence, attach the id of the signal that supports it.\n"
    "3. Keep it under 120 words, plain and specific, no hype.\n"
    "Reply with ONLY a JSON object: "
    '{"subject": "...", "body": "...", "claims": [{"text": "...", "evidence_id": "s1"}]}'
)


def _draft_with_model(model, lead: dict, signals) -> dict:
    sig = "\n".join(f"- {s['id']} [{s['kind']}] {s['field']}: {s['value']} "
                    f"(source {s['source']}, as_of {s['as_of']})" for s in signals)
    user = f"PROSPECT: {lead.get('name', '')}\nSIGNALS:\n{sig}"
    resp = model.invoke([("system", _DRAFT_SYSTEM), ("human", user)])
    content = (getattr(resp, "content", "") or "").strip()
    start, end = content.find("{"), content.rfind("}")
    data = json.loads(content[start:end + 1])
    data.setdefault("claims", [])
    data.setdefault("subject", "A quick idea for your team")
    return data
