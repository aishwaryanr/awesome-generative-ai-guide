"""A customer support agent as a LangGraph state machine.

This is example code for the case study: retrieval to ground answers, a bounded agent loop
that can call tools, a guardrail that refuses or escalates when an answer is not grounded,
and a human-escalation path (high-impact requests like refunds escalate for approval). It
is deliberately small and runs offline (see llm.py). The path to real scale and production
(vector search, caching, Arize observability, evaluation gates) is described in the case
study writeup, not coded here.

    START -> retrieve -> agent --+--> order_lookup --+
                                 |                    |  (loop, bounded by MAX_STEPS)
                                 +--> create_ticket --+
                                 |         back to agent
                                 +--> guardrail --> END
                                 |         \\-> escalate -> END
                                 +--> escalate -> END   (out of scope, or refund needs a human)

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from kb import retrieve
from llm import decide

MAX_STEPS = 3  # cap the loop so a confused agent cannot spin forever


class State(TypedDict, total=False):
    question: str
    context: List
    tool_result: Optional[dict]
    decision: Optional[dict]
    steps: int
    answer: Optional[str]
    escalate: bool
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_retrieve(state: State):
    ctx = retrieve(state["question"])
    return {"context": ctx, "steps": 0, "tool_result": None,
            "trace": _trace(state, f"retrieve: {len(ctx)} doc(s)")}


def n_agent(state: State):
    d = decide(state["question"], state.get("context"), state.get("tool_result"))
    label = d["action"] + (f":{d.get('tool')}" if d.get("tool") else "")
    upd = {"decision": d, "trace": _trace(state, f"agent -> {label}")}
    if d["action"] == "answer":
        upd["answer"] = d["answer"]
    return upd


def n_order_lookup(state: State):
    args = state["decision"].get("args", {})
    res = {"tool": "order_lookup", "order_id": args.get("order_id", "unknown"),
           "status": "in transit", "eta": "2026-07-18"}
    return {"tool_result": res, "steps": state.get("steps", 0) + 1,
            "trace": _trace(state, "tool: order_lookup")}


def n_create_ticket(state: State):
    args = state["decision"].get("args", {})
    res = {"tool": "create_ticket", "ticket_id": "T-10432", "summary": args.get("summary", "")}
    return {"tool_result": res, "steps": state.get("steps", 0) + 1,
            "trace": _trace(state, "tool: create_ticket")}


def n_guardrail(state: State):
    ctx = state.get("context") or []
    grounded = bool(ctx or state.get("tool_result"))
    injected = any(("ignore previous" in t.lower() or "disregard all" in t.lower())
                   for _, t in ctx)
    if injected or not grounded:
        why = "prompt injection in retrieved content" if injected else "answer not grounded"
        return {"escalate": True, "trace": _trace(state, f"guardrail: FAIL ({why})")}
    return {"trace": _trace(state, "guardrail: pass")}


def n_escalate(state: State):
    return {"answer": "I am escalating this to a human specialist who will reach out shortly.",
            "escalate": True, "trace": _trace(state, "escalate: handed to human")}


def route_agent(state: State):
    d = state["decision"]
    if d["action"] == "tool" and state.get("steps", 0) < MAX_STEPS:
        return d["tool"]
    if d["action"] == "escalate":
        return "escalate"
    return "guardrail"


def route_guardrail(state: State):
    return "escalate" if state.get("escalate") else END


def build():
    g = StateGraph(State)
    for name, fn in [("retrieve", n_retrieve), ("agent", n_agent),
                     ("order_lookup", n_order_lookup), ("create_ticket", n_create_ticket),
                     ("guardrail", n_guardrail), ("escalate", n_escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_agent,
                            {"order_lookup": "order_lookup", "create_ticket": "create_ticket",
                             "guardrail": "guardrail", "escalate": "escalate"})
    g.add_edge("order_lookup", "agent")
    g.add_edge("create_ticket", "agent")
    g.add_conditional_edges("guardrail", route_guardrail, {"escalate": "escalate", END: END})
    g.add_edge("escalate", END)
    return g.compile()


APP = build()


def answer(question: str) -> State:
    """Run one question through the agent and return the final state."""
    return APP.invoke({"question": question, "trace": []})
