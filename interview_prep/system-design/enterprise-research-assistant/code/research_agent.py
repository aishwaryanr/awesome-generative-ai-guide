"""An enterprise research assistant as a LangGraph state machine.

This is example code for the case study. It shows the load-bearing parts of the design:
permission-scoped retrieval (the user's identity is applied at retrieval time, so a source they
cannot access never enters the context), a bounded multi-hop research loop (the agent runs its own
follow-up searches), answers that carry citations, a guardrail that escalates when an answer is not
grounded or when a source tries to inject instructions, and a human gate on any high-impact action.
It is deliberately small and runs offline (see llm.py). The path to real scale (a vector store,
hybrid retrieval, a reranker, connectors, freshness, Arize observability, evaluation gates) is
described in the case study writeup, not coded here.

    START -> plan -> retrieve -> agent --+--> retrieve   (multi-hop loop, bounded by MAX_STEPS)
                                         |
                                         +--> compose --> guardrail --> END
                                         |                     \\-> escalate -> END
                                         +--> human_gate -> END   (high-impact action: awaits approval)
                                         +--> escalate -> END      (nothing readable answers it)

Run:  python run.py
"""
from typing import TypedDict, Optional, List, Tuple

from langgraph.graph import StateGraph, START, END

from kb import retrieve, CORPUS, can_read
from llm import plan, decide

MAX_STEPS = 4  # cap the research loop so a confused agent cannot search forever

# A default identity for callers that do not pass one: a regular employee with no special roles.
DEFAULT_USER = {"id": "u_employee", "roles": {"employee"}}

_INJECTION_MARKERS = ("ignore previous", "ignore all previous", "disregard all",
                      "disregard previous", "note to assistant")


class State(TypedDict, total=False):
    question: str
    user: dict
    pending: List[str]                 # planned searches not yet run
    retrieved: List[Tuple[str, str, str]]  # (doc_id, text, system), permission-filtered
    decision: Optional[dict]
    steps: int
    answer: Optional[str]
    citations: List[str]
    escalate: bool
    awaiting_approval: bool
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_plan(state: State):
    subqueries = plan(state["question"])
    return {"pending": subqueries, "retrieved": [], "steps": 0,
            "trace": _trace(state, f"plan: {len(subqueries)} search(es) -> {subqueries}")}


def n_retrieve(state: State):
    pending = list(state.get("pending", []))
    query = pending.pop(0) if pending else state["question"]
    user = state.get("user", DEFAULT_USER)
    hits = retrieve(query, user)
    # merge new hits into the running context, de-duplicated by doc_id
    seen = {d for d, _, _ in state.get("retrieved", [])}
    merged = state.get("retrieved", []) + [h for h in hits if h[0] not in seen]
    got = ", ".join(d for d, _, _ in hits) or "nothing readable"
    return {"pending": pending, "retrieved": merged, "steps": state.get("steps", 0) + 1,
            "trace": _trace(state, f"retrieve[{query[:40]}]: {got}")}


def n_agent(state: State):
    d = decide(state["question"], state.get("retrieved"), state.get("pending", []))
    label = d["action"] + (f":{d.get('query')}" if d.get("query") else "")
    return {"decision": d, "trace": _trace(state, f"agent -> {label}")}


def n_compose(state: State):
    # Build a cited answer: every retrieved passage is stitched in with its source, so each claim
    # traces back to a document the user was allowed to read.
    retrieved = state.get("retrieved", [])
    parts = [f"{text} (source: {doc_id}, {system})" for doc_id, text, system in retrieved]
    citations = [doc_id for doc_id, _, _ in retrieved]
    return {"answer": " ".join(parts), "citations": citations,
            "trace": _trace(state, f"compose: {len(citations)} citation(s)")}


def n_guardrail(state: State):
    retrieved = state.get("retrieved", [])
    citations = state.get("citations", [])
    user = state.get("user", DEFAULT_USER)
    retrieved_ids = {d for d, _, _ in retrieved}

    grounded = bool(citations) and all(c in retrieved_ids for c in citations)
    injected = any(any(m in text.lower() for m in _INJECTION_MARKERS) for _, text, _ in retrieved)
    # Belt and suspenders: reconfirm every cited source is still readable by this user.
    leaked = any(not can_read(user, CORPUS.get(c, {"acl": set()})) for c in citations)

    if injected or not grounded or leaked:
        why = ("prompt injection in a retrieved source" if injected else
               "a citation the user cannot read" if leaked else "answer not grounded")
        return {"escalate": True, "trace": _trace(state, f"guardrail: FAIL ({why})")}
    return {"trace": _trace(state, "guardrail: pass")}


def n_human_gate(state: State):
    reason = state.get("decision", {}).get("reason", "high-impact action")
    return {"awaiting_approval": True,
            "answer": ("I have prepared a draft and routed it for human approval before anything "
                       f"is posted or sent ({reason})."),
            "trace": _trace(state, "human_gate: routed for approval, not executed")}


def n_escalate(state: State):
    reason = state.get("decision", {}).get("reason", "unresolved")
    return {"escalate": True,
            "answer": "I could not answer this from sources you have access to, so I am handing "
                      "it to a human who can help.",
            "trace": _trace(state, f"escalate: handed to human ({reason})")}


def route_agent(state: State):
    d = state["decision"]
    if d["action"] == "search" and state.get("steps", 0) < MAX_STEPS:
        return "retrieve"
    if d["action"] == "request_action":
        return "human_gate"
    if d["action"] == "escalate":
        return "escalate"
    return "compose"


def route_guardrail(state: State):
    return "escalate" if state.get("escalate") else END


def build():
    g = StateGraph(State)
    for name, fn in [("plan", n_plan), ("retrieve", n_retrieve), ("agent", n_agent),
                     ("compose", n_compose), ("guardrail", n_guardrail),
                     ("human_gate", n_human_gate), ("escalate", n_escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_agent,
                            {"retrieve": "retrieve", "compose": "compose",
                             "human_gate": "human_gate", "escalate": "escalate"})
    g.add_edge("compose", "guardrail")
    g.add_conditional_edges("guardrail", route_guardrail, {"escalate": "escalate", END: END})
    g.add_edge("human_gate", END)
    g.add_edge("escalate", END)
    return g.compile()


APP = build()


def answer(question: str, user: Optional[dict] = None) -> State:
    """Run one question through the agent as a given user, and return the final state."""
    return APP.invoke({"question": question, "user": user or DEFAULT_USER, "trace": []})
