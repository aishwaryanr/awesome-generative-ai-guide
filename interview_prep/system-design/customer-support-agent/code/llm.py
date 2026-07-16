"""The model layer, provider-agnostic.

The agent needs the model to make one decision per step: answer now, call a tool, or
escalate. This file gives you two implementations behind one function:

- a deterministic FAKE policy so the whole graph runs offline with no API key, which is
  what makes this example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a
  model and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do
  not change the graph to change the model.

Selecting a model (any one of these):
    export SUPPORT_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export SUPPORT_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export SUPPORT_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If SUPPORT_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or
langchain-google-genai.
"""
import os

REFUND_INTENT = ("refund", "cancel", "money back", "chargeback")  # high-impact: needs a human
TICKET_INTENT = ("damaged", "broken", "replace", "return item", "wrong item")
ORDER_INTENT = ("order", "track", "where is my", "delivery status", "shipment")

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline policy."""
    model = os.environ.get("SUPPORT_AGENT_MODEL")
    provider = os.environ.get("SUPPORT_AGENT_PROVIDER")
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


def decide(question: str, context, tool_result):
    """Return the next action: {"action": "answer"|"tool"|"escalate", ...}. Provider-agnostic."""
    model = _get_model()
    if model is not None:
        try:
            return _decide_with_model(model, question, context, tool_result)
        except Exception:
            pass  # fall back to the offline policy if the provider call fails
    return _decide_offline(question, context, tool_result)


# --- offline deterministic policy (no provider) ---------------------------------------

def _decide_offline(question: str, context, tool_result):
    q = question.lower()
    if tool_result is not None:
        return {"action": "answer", "answer": _answer_from_tool(tool_result)}
    if any(t in q for t in REFUND_INTENT):
        return {"action": "escalate", "reason": "refund or cancellation requires human approval"}
    if any(t in q for t in TICKET_INTENT):
        return {"action": "tool", "tool": "create_ticket",
                "args": {"summary": question[:80], "category": "return-or-damage"}}
    if any(t in q for t in ORDER_INTENT):
        return {"action": "tool", "tool": "order_lookup", "args": {"order_id": _extract_order_id(q)}}
    if context:
        return {"action": "answer", "answer": _answer_from_context(context)}
    return {"action": "escalate", "reason": "no grounded answer available"}


def _answer_from_context(context):
    doc_id, text = context[0]
    return f"{text} (source: {doc_id})"


def _answer_from_tool(tool_result):
    if tool_result.get("tool") == "order_lookup":
        return (f"Your order {tool_result['order_id']} is {tool_result['status']}, "
                f"estimated delivery {tool_result['eta']}.")
    if tool_result.get("tool") == "create_ticket":
        return (f"I have opened support ticket {tool_result['ticket_id']} for you. "
                f"Our team will follow up by email within 1 business day.")
    return "Done."


def _extract_order_id(q: str):
    for tok in q.replace("#", " ").split():
        tok = tok.strip("?.,!:;()[]")
        if tok.isdigit() and len(tok) >= 4:
            return tok
    return "unknown"


# --- real path: any provider via init_chat_model --------------------------------------

_DECISION_SYSTEM = (
    "You are a customer support agent. Decide the next step and reply with EXACTLY one line, "
    "in one of these forms, and nothing else:\n"
    "ANSWER: <a grounded answer that uses only the context or tool result, and names the source>\n"
    "TOOL: order_lookup | <order_id>\n"
    "TOOL: create_ticket\n"
    "ESCALATE\n"
    "Answer only from the provided context or tool result. If you cannot ground it, reply ESCALATE. "
    "High-impact actions like refunds or cancellations must always reply ESCALATE for human approval. "
    "Never invent policy or order details."
)


def _decide_with_model(model, question, context, tool_result):
    ctx = "\n".join(f"- {d}: {t}" for d, t in (context or [])) or "(none)"
    tr = f"\nTOOL RESULT: {tool_result}" if tool_result else ""
    user = f"CONTEXT:\n{ctx}{tr}\n\nQUESTION: {question}"
    resp = model.invoke([("system", _DECISION_SYSTEM), ("human", user)])
    line = (getattr(resp, "content", "") or "").strip().splitlines()[0].strip() if getattr(resp, "content", "") else ""
    up = line.upper()
    if up.startswith("ANSWER:"):
        return {"action": "answer", "answer": line.split(":", 1)[1].strip()}
    if up.startswith("TOOL: ORDER_LOOKUP"):
        oid = line.split("|", 1)[1].strip() if "|" in line else _extract_order_id(question.lower())
        return {"action": "tool", "tool": "order_lookup", "args": {"order_id": oid}}
    if up.startswith("TOOL: CREATE_TICKET"):
        return {"action": "tool", "tool": "create_ticket",
                "args": {"summary": question[:80], "category": "return-or-damage"}}
    return {"action": "escalate", "reason": "model could not ground an answer"}
