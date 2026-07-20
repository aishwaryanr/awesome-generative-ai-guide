"""A security and compliance screening agent as a LangGraph state machine.

This is example code for the case study. Each event from the stream runs through a layered
detector: a deterministic rule pass first (cheap, exact, runs on everything), then policy
retrieval and a model judgment for the ambiguous middle, with an anomaly signal from the
actor baseline folded in. The output is one verdict per event, allow / flag / block, with an
explanation, and every decision is written to a tamper-evident audit trail. Flagged events
route to human review, and blocking is reserved for a strong, policy-named signal so false
positives stay low.

    START -> ingest -> rules --+--(block)--> enforce ---------> audit -> END
                               |--(allow)-------------------->  audit -> END
                               +--(inspect)--> retrieve -> anomaly -> judge -> decide
                                                                                 |
                                        +--(allow)-----------------------------> audit -> END
                                        +--(flag)--> human_review -------------> audit -> END
                                        +--(block)--> enforce -----------------> audit -> END

The path to real scale and production (streaming ingestion, an append-only ledger, Arize
observability, evaluation gates) is described in the case study writeup, not coded here.

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from policy import normalize, rule_check, anomaly_signal, retrieve_policy
from llm import judge
from audit import AuditLog

AUDIT = AuditLog()  # shared append-only trail, so the hash chain spans every screened event


class State(TypedDict, total=False):
    event: dict
    policy: List
    severity: int
    evidence: List[str]
    verdict: Optional[str]
    reason: Optional[str]
    trace: List[str]
    audit_entry: Optional[dict]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_ingest(state: State):
    e = normalize(state["event"])
    return {"event": e, "evidence": [], "trace": _trace(state, f"ingest: {e['actor']}")}


def n_rules(state: State):
    signal, evidence = rule_check(state["event"])
    upd = {"evidence": state.get("evidence", []) + evidence,
           "trace": _trace(state, f"rules -> {signal}")}
    if signal == "block":
        upd["verdict"] = "block"
        upd["reason"] = evidence[0]
    elif signal == "allow":
        upd["verdict"] = "allow"
        upd["reason"] = evidence[0]
    return upd


def n_retrieve(state: State):
    pol = retrieve_policy(state["event"])
    return {"policy": pol, "trace": _trace(state, f"retrieve: {len(pol)} policy")}


def n_anomaly(state: State):
    severity, evidence = anomaly_signal(state["event"])
    return {"severity": severity, "evidence": state.get("evidence", []) + evidence,
            "trace": _trace(state, f"anomaly: severity {severity}")}


def n_judge(state: State):
    d = judge(state["event"], state.get("policy"), state.get("severity", 0))
    # Guardrail: never block on an ungrounded judgment. If no policy matched, the strongest
    # the agent may do on its own is flag for a human. Enforcement needs a named policy.
    verdict = d["verdict"]
    if verdict == "block" and not state.get("policy"):
        verdict = "flag"
    return {"verdict": verdict, "reason": d["reason"],
            "trace": _trace(state, f"judge -> {verdict}")}


def n_enforce(state: State):
    return {"trace": _trace(state, "enforce: block applied")}


def n_human_review(state: State):
    return {"trace": _trace(state, "route: sent to human review queue")}


def n_audit(state: State):
    entry = AUDIT.record(state["event"], state.get("verdict"), state.get("reason"),
                         state.get("evidence", []))
    return {"audit_entry": entry, "trace": _trace(state, f"audit: entry #{entry['seq']} logged")}


def route_rules(state: State):
    v = state.get("verdict")
    if v == "block":
        return "enforce"
    if v == "allow":
        return "audit"
    return "retrieve"


def route_decide(state: State):
    v = state.get("verdict")
    if v == "block":
        return "enforce"
    if v == "flag":
        return "human_review"
    return "audit"


def build():
    g = StateGraph(State)
    for name, fn in [("ingest", n_ingest), ("rules", n_rules), ("retrieve", n_retrieve),
                     ("anomaly", n_anomaly), ("judge", n_judge), ("enforce", n_enforce),
                     ("human_review", n_human_review), ("audit", n_audit)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "rules")
    g.add_conditional_edges("rules", route_rules,
                            {"enforce": "enforce", "audit": "audit", "retrieve": "retrieve"})
    g.add_edge("retrieve", "anomaly")
    g.add_edge("anomaly", "judge")
    g.add_conditional_edges("judge", route_decide,
                            {"enforce": "enforce", "human_review": "human_review", "audit": "audit"})
    g.add_edge("enforce", "audit")
    g.add_edge("human_review", "audit")
    g.add_edge("audit", END)
    return g.compile()


APP = build()


def screen(event) -> State:
    """Run one event through the screener and return the final state."""
    return APP.invoke({"event": event, "trace": []})
