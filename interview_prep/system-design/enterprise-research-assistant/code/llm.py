"""The model layer, provider-agnostic, plus the offline planner and decision policy.

A research assistant makes two kinds of model call: it plans (break a question into the searches
that will answer it) and it decides the next step (search again, answer with citations, escalate,
or route a high-impact action to a human). This file gives you both behind plain functions, with
two implementations each:

- a deterministic offline policy so the whole graph runs with no API key, which is what makes this
  example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model and a
  key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not change the graph
  to change the model.

Selecting a model (any one of these):
    export RESEARCH_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export RESEARCH_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export RESEARCH_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If RESEARCH_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or langchain-google-genai.
"""
import os
import re
from typing import List, Optional

# High-impact intents: composing to a shared audience or reaching outside the company. These are
# gated to a human rather than executed by the agent.
ACTION_INTENT = ("post ", "publish", "announce", "send to", "email ", "share to", "broadcast",
                 "message everyone", "all-hands", "all hands", "all-company", "all company")

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline policy."""
    model = os.environ.get("RESEARCH_AGENT_MODEL")
    provider = os.environ.get("RESEARCH_AGENT_PROVIDER")
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


# --- planning: break a question into the searches that answer it ----------------------

def plan(question: str) -> List[str]:
    """Return an ordered list of sub-queries. Multi-part questions become multiple searches, which
    is what drives the multi-hop retrieval loop. Provider-agnostic: tries the model, falls back."""
    model = _get_model()
    if model is not None:
        try:
            out = _plan_with_model(model, question)
            if out:
                return out
        except Exception:
            pass
    return _plan_offline(question)


def _plan_offline(question: str) -> List[str]:
    # Split on a coordinating "and" that joins two asks. One clause becomes one search.
    parts = re.split(r",?\s+and\s+", question.strip(), flags=re.IGNORECASE)
    parts = [p.strip(" ?.") for p in parts if len(p.strip(" ?.")) > 3]
    return parts if len(parts) > 1 else [question]


# --- deciding the next step -----------------------------------------------------------

def decide(question: str, retrieved, pending: List[str]):
    """Return the next action: search, answer, escalate, or request_action. Provider-agnostic."""
    model = _get_model()
    if model is not None:
        try:
            return _decide_with_model(model, question, retrieved, pending)
        except Exception:
            pass
    return _decide_offline(question, retrieved, pending)


def _decide_offline(question: str, retrieved, pending: List[str]):
    if pending:  # still have planned searches to run: keep gathering (the multi-hop loop)
        return {"action": "search", "query": pending[0]}
    if any(t in question.lower() for t in ACTION_INTENT):
        # A high-impact action was requested. Research is done; hand the action to a human.
        return {"action": "request_action", "reason": "high-impact action needs human approval"}
    if retrieved:
        return {"action": "answer"}
    return {"action": "escalate", "reason": "no readable source answers this"}


# --- real path: any provider via init_chat_model --------------------------------------

_PLAN_SYSTEM = (
    "You plan research over internal company sources. Break the question into 1 to 4 short search "
    "queries, one per line, no numbering. If the question is a single ask, return it as one line."
)

_DECIDE_SYSTEM = (
    "You are an enterprise research assistant. Given the question, the context retrieved so far, "
    "and any remaining planned searches, reply with EXACTLY one line in one of these forms:\n"
    "SEARCH: <query>\n"
    "ANSWER\n"
    "ACTION: <the high-impact action being requested>\n"
    "ESCALATE\n"
    "Rules: keep searching while planned searches remain or the context cannot answer the "
    "question. Answer only from the provided context. If nothing readable supports an answer, "
    "reply ESCALATE. Any request to post, publish, email, or broadcast to a shared audience must "
    "reply ACTION so a human approves it. Never invent facts or sources."
)


def _plan_with_model(model, question: str) -> List[str]:
    resp = model.invoke([("system", _PLAN_SYSTEM), ("human", question)])
    content = getattr(resp, "content", "") or ""
    lines = [ln.strip(" -*0123456789.").strip() for ln in content.splitlines() if ln.strip()]
    return [ln for ln in lines if len(ln) > 3][:4]


def _decide_with_model(model, question: str, retrieved, pending: List[str]):
    ctx = "\n".join(f"- {d} ({s}): {t}" for d, t, s in (retrieved or [])) or "(none)"
    pend = "; ".join(pending) if pending else "(none)"
    user = f"QUESTION: {question}\n\nCONTEXT SO FAR:\n{ctx}\n\nREMAINING PLANNED SEARCHES: {pend}"
    resp = model.invoke([("system", _DECIDE_SYSTEM), ("human", user)])
    line = ((getattr(resp, "content", "") or "").strip().splitlines() or [""])[0].strip()
    up = line.upper()
    if up.startswith("SEARCH:"):
        return {"action": "search", "query": line.split(":", 1)[1].strip() or (pending[0] if pending else question)}
    if up.startswith("ACTION:"):
        return {"action": "request_action", "reason": line.split(":", 1)[1].strip() or "high-impact action"}
    if up.startswith("ANSWER"):
        return {"action": "answer"}
    return {"action": "escalate", "reason": "model could not ground an answer"}
