"""A document decisioning agent as a LangGraph state machine.

This is example code for the case study: extract fields from a (mock) submission, verify the
extraction (confidence floor, required fields, PII already masked), apply the deterministic
underwriting policy, let the model make the borderline judgment, and write an immutable audit
record. Low-confidence, missing-field, high-impact, and borderline cases route to a human
underwriter. It is deliberately small and runs offline (see llm.py). Real scale and production
(a layout-aware parsing service, a vector-backed policy library, WORM audit storage, Arize
observability, evaluation gates) are described in the case study, not coded here.

    START -> extract -> verify --(low conf / missing field)--> human --> audit -> END
                          |
                        (ok)
                          v
                        policy --(decline)------------------------------> audit -> END
                          |    (above authority)-----------> human -----> audit -> END
                        (clean)
                          v
                        decide (model) --(approve)---------------------> audit -> END
                                         (refer)-----------> human ----> audit -> END

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from documents import extract, REQUIRED_FIELDS, CONF_FLOOR
from policy import evaluate
from llm import decide, model_id
from audit import record


class State(TypedDict, total=False):
    document: str
    extracted: dict
    needs_human: bool
    human_reason: Optional[str]
    policy: Optional[dict]
    decision: Optional[str]
    reason: Optional[str]
    referred: bool
    audit: Optional[dict]
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_extract(state: State):
    ex = extract(state["document"])
    return {"extracted": ex,
            "trace": _trace(state, f"extract: {len(ex['fields'])} field(s), "
                                   f"{len(ex['pii_masked'])} PII value(s) masked")}


def n_verify(state: State):
    fields = state["extracted"]["fields"]
    missing = [f for f in REQUIRED_FIELDS if f not in fields]
    low = [f for f, d in fields.items() if d["confidence"] < CONF_FLOOR]
    if missing or low:
        bits = []
        if missing:
            bits.append("missing " + ", ".join(missing))
        if low:
            bits.append("low-confidence extraction on " + ", ".join(low))
        why = "; ".join(bits)
        return {"needs_human": True, "human_reason": why,
                "trace": _trace(state, f"verify: FAIL ({why})")}
    return {"needs_human": False,
            "trace": _trace(state, "verify: all required fields present and above the confidence floor")}


def n_policy(state: State):
    pol = evaluate(state["extracted"]["fields"])
    upd = {"policy": pol, "trace": _trace(state, f"policy: {pol['outcome']} ({pol['reason']})")}
    if pol["outcome"] == "decline":
        upd["decision"] = "decline"
        upd["reason"] = pol["reason"]
    return upd


def n_decide(state: State):
    d = decide(state["extracted"]["fields"], state["policy"].get("flags", []))
    return {"decision": d["decision"], "reason": d["reason"],
            "trace": _trace(state, f"decide (model judgment) -> {d['decision']}")}


def n_human(state: State):
    reason = state.get("human_reason") or state.get("reason") \
        or (state.get("policy") or {}).get("reason") or "flagged for review"
    return {"decision": "refer", "referred": True,
            "reason": f"routed to a human underwriter: {reason}",
            "trace": _trace(state, "refer: routed to a human underwriter")}


def n_audit(state: State):
    fields = {name: d["value"] for name, d in state["extracted"]["fields"].items()}
    confidences = [d["confidence"] for d in state["extracted"]["fields"].values()]
    rec = record({
        "applicant": fields.get("applicant_name", "unknown"),
        "decision": state.get("decision"),
        "referred": state.get("referred", False),
        "reason": state.get("reason"),
        "policy_outcome": (state.get("policy") or {}).get("outcome", "n/a"),
        "fields": fields,
        "pii_masked": state["extracted"]["pii_masked"],
        "min_confidence": min(confidences) if confidences else 0.0,
        "model_version": model_id(),
    })
    return {"audit": rec,
            "trace": _trace(state, f"audit: record #{rec['seq']} written (hash {rec['hash'][:8]})")}


def route_verify(state: State):
    return "human" if state.get("needs_human") else "policy"


def route_policy(state: State):
    outcome = state["policy"]["outcome"]
    if outcome == "decline":
        return "audit"
    if outcome == "refer":
        return "human"
    return "decide"


def route_decide(state: State):
    return "human" if state["decision"] == "refer" else "audit"


def build():
    g = StateGraph(State)
    for name, fn in [("extract", n_extract), ("verify", n_verify), ("policy", n_policy),
                     ("decide", n_decide), ("human", n_human), ("audit", n_audit)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract")
    g.add_edge("extract", "verify")
    g.add_conditional_edges("verify", route_verify, {"human": "human", "policy": "policy"})
    g.add_conditional_edges("policy", route_policy,
                            {"audit": "audit", "human": "human", "decide": "decide"})
    g.add_conditional_edges("decide", route_decide, {"human": "human", "audit": "audit"})
    g.add_edge("human", "audit")
    g.add_edge("audit", END)
    return g.compile()


APP = build()


def decide_document(document: str) -> State:
    """Run one submission through the agent and return the final state."""
    return APP.invoke({"document": document, "trace": [], "referred": False})
