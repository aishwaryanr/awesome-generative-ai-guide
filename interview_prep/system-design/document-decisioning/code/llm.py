"""The model layer, provider-agnostic: the judgment half of the decision.

The deterministic policy (policy.py) has already handled every hard rule and every
high-impact case before the model is called. What is left is the borderline approve-or-refer
judgment the rules deliberately leave open. So the model chooses between exactly two options,
APPROVE or REFER, and the worst a wrong call can do is send a clean file to a human. It can
never approve something the policy declined, and it can never sign off above authority.

Two implementations behind one function:

- a deterministic FAKE policy so the whole graph runs offline with no API key, which is what
  makes this example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model
  and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not
  change the graph to change the model.

Selecting a model (any one of these):
    export UW_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export UW_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export UW_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If UW_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or
langchain-google-genai.
"""
import os

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def model_id() -> str:
    """The model recorded in the audit trail, so every decision names the model that made it."""
    model = os.environ.get("UW_AGENT_MODEL")
    if model:
        return model
    for key, m, _ in _AUTODETECT:
        if os.environ.get(key):
            return m
    return "offline-deterministic"


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline policy."""
    model = os.environ.get("UW_AGENT_MODEL")
    provider = os.environ.get("UW_AGENT_PROVIDER")
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


def decide(fields: dict, flags) -> dict:
    """Return the borderline judgment: {"decision": "approve" | "refer", "reason": ...}."""
    model = _get_model()
    if model is not None:
        try:
            return _decide_with_model(model, fields, flags)
        except Exception:
            pass  # fall back to the offline policy if the provider call fails
    return _decide_offline(flags)


# --- offline deterministic policy (no provider) ---------------------------------------

def _decide_offline(flags) -> dict:
    if flags:
        return {"decision": "refer",
                "reason": "borderline factors need underwriter judgment: " + "; ".join(flags)}
    return {"decision": "approve",
            "reason": "meets underwriting policy within delegated authority"}


# --- real path: any provider via init_chat_model --------------------------------------

_DECISION_SYSTEM = (
    "You are an insurance underwriting assistant making the final call on a submission whose "
    "hard rules have ALREADY passed a deterministic policy check. Decide between exactly two "
    "options and reply with EXACTLY one line, nothing else:\n"
    "APPROVE: <one-sentence reason grounded in the fields>\n"
    "REFER: <one-sentence reason a human underwriter should review this>\n"
    "You may never approve when borderline risk factors are unresolved; when in doubt, REFER. "
    "Base the reason only on the fields and flags provided. Do not invent facts."
)


def _decide_with_model(model, fields, flags) -> dict:
    f = "\n".join(f"- {k}: {d['value']} (confidence {d['confidence']})" for k, d in fields.items())
    fl = "\n".join(f"- {x}" for x in flags) or "(none)"
    user = f"FIELDS:\n{f}\n\nBORDERLINE FLAGS:\n{fl}"
    resp = model.invoke([("system", _DECISION_SYSTEM), ("human", user)])
    content = getattr(resp, "content", "") or ""
    line = content.strip().splitlines()[0].strip() if content.strip() else ""
    up = line.upper()
    if up.startswith("APPROVE"):
        reason = line.split(":", 1)[1].strip() if ":" in line else "meets policy within authority"
        return {"decision": "approve", "reason": reason}
    reason = line.split(":", 1)[1].strip() if ":" in line else "flagged for human review"
    return {"decision": "refer", "reason": reason}
