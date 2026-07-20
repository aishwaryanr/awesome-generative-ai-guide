"""The model layer, provider-agnostic.

The harness owns the control flow (read, test, edit, open a PR, or escalate). The model owns
one creative decision: given the failing tests and what has already been tried, what edit to
make next. This file gives you two implementations behind one function, propose_edit:

- a deterministic OFFLINE policy so the whole harness runs with no API key, which is what
  makes this example runnable and testable in CI, and
- a real path that works with ANY provider through LangChain's init_chat_model. Set a model
  and a key and it uses OpenAI, Anthropic, Gemini, or any supported provider. You do not
  change the harness to change the model.

Selecting a model (any one of these):
    export CODING_AGENT_MODEL="gpt-4o-mini"        + OPENAI_API_KEY
    export CODING_AGENT_MODEL="claude-sonnet-5"    + ANTHROPIC_API_KEY
    export CODING_AGENT_MODEL="gemini-2.0-flash"   + GOOGLE_API_KEY
If CODING_AGENT_MODEL is unset, the provider is auto-detected from whichever key is present.
Install the matching integration: langchain-openai, langchain-anthropic, or
langchain-google-genai.
"""
import os
from typing import List, Optional

# The operators the coder may try in `return a <op> b`. A real model proposes a full diff; here
# the search space is small and closed so the offline run is deterministic and the loop is honest:
# it uses the test feedback to pick the next candidate, exactly as an agent iterates on failures.
CANDIDATE_OPERATORS = ["+", "*", "//", "%"]

# Auto-detect: (env key that must be present) -> (default model, provider)
_AUTODETECT = [
    ("OPENAI_API_KEY", "gpt-4o-mini", "openai"),
    ("ANTHROPIC_API_KEY", "claude-sonnet-5", "anthropic"),
    ("GOOGLE_API_KEY", "gemini-2.0-flash", "google_genai"),
    ("GEMINI_API_KEY", "gemini-2.0-flash", "google_genai"),
]


def _get_model():
    """Return a provider-agnostic chat model, or None to run the offline policy."""
    model = os.environ.get("CODING_AGENT_MODEL")
    provider = os.environ.get("CODING_AGENT_PROVIDER")
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


def propose_edit(task: dict, source: str, last_test: Optional[dict], tried: List[str],
                 current_op: str) -> Optional[str]:
    """Return the next operator to try, or None to give up. Provider-agnostic.

    tried is the list of operators already attempted, so the model never repeats a known
    failure. current_op is the operator in the file right now, which is also off the table.
    """
    off_limits = set(tried) | {current_op}
    model = _get_model()
    if model is not None:
        try:
            choice = _propose_with_model(model, task, source, last_test, off_limits)
            if choice in CANDIDATE_OPERATORS and choice not in off_limits:
                return choice
        except Exception:
            pass  # fall back to the offline policy if the provider call fails
    return _propose_offline(off_limits)


# --- offline deterministic policy (no provider) ---------------------------------------

def _propose_offline(off_limits: set) -> Optional[str]:
    """Pick the next untried operator. None when the search space is exhausted."""
    for op in CANDIDATE_OPERATORS:
        if op not in off_limits:
            return op
    return None


# --- real path: any provider via init_chat_model --------------------------------------

_SYSTEM = (
    "You are a coding agent fixing a Python function so its tests pass. The function body is "
    "`return a <op> b`. Reply with EXACTLY one line and nothing else, in one of these forms:\n"
    "EDIT: <op>   where <op> is one of + * // %\n"
    "GIVEUP       when no operator can make the tests pass\n"
    "Use the failing test output to choose. Never repeat an operator that already failed."
)


def _propose_with_model(model, task, source, last_test, off_limits) -> Optional[str]:
    failures = (last_test or {}).get("failures", [])
    fail_text = "\n".join(f"- {f['call']} returned {f['actual']!r}, expected {f['expected']!r}"
                          for f in failures) or "(no test run yet)"
    tried_text = ", ".join(sorted(off_limits)) or "(none)"
    user = (f"TASK: {task['text']}\n\nCURRENT FILE:\n{source}\n\n"
            f"FAILING TESTS:\n{fail_text}\n\nALREADY TRIED (do not repeat): {tried_text}")
    resp = model.invoke([("system", _SYSTEM), ("human", user)])
    content = getattr(resp, "content", "") or ""
    line = content.strip().splitlines()[0].strip() if content.strip() else ""
    if line.upper().startswith("EDIT:"):
        return line.split(":", 1)[1].strip()
    return None
