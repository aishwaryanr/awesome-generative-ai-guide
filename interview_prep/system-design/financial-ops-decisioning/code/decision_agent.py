"""A financial-operations decisioning agent as a LangGraph state machine.

This is example code for the case study: extract fields from a (mock) invoice or expense,
apply deterministic policy checks, decide approve / deny / route-to-human, and write an
immutable audit record for every case. High-impact or out-of-policy or low-confidence cases
escalate to a human for sign-off. It is deliberately small and runs offline (see llm.py). The
path to real scale and production (a layout-aware extractor, a real rules engine, a WORM audit
store, an approval queue, Arize observability, evaluation gates) is described in the case study
writeup, not coded here.

    START -> extract -> policy -> decide --+--> approve -------+
                                           +--> deny ----------+--> audit -> END
                                           +--> human_review --+

The decision is deterministic (policy.py); the model only turns the document into fields, so
the same input always yields the same decision and the same audit trail.

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from llm import extract_fields
from policy import evaluate
from audit import write_record


class State(TypedDict, total=False):
    document: str
    fields: dict
    decision: Optional[str]
    findings: List
    record: Optional[dict]
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_extract(state: State):
    fields = extract_fields(state["document"])
    found = [f for f in ("vendor", "invoice_id", "amount", "date", "category") if fields.get(f) not in (None, "", [])]
    return {"fields": fields,
            "trace": _trace(state, f"extract: {len(found)} field(s), confidence {fields.get('_confidence')}")}


def n_policy(state: State):
    decision, findings = evaluate(state["fields"])
    reasons = [f"{code}: {msg}" for code, msg, _ in findings]
    return {"decision": decision, "findings": findings,
            "trace": _trace(state, f"policy: {len(findings)} finding(s) -> {decision}") if reasons
            else _trace(state, "policy: clean -> approve")}


def n_approve(state: State):
    return {"trace": _trace(state, "approve: within policy, auto-approved")}


def n_deny(state: State):
    return {"trace": _trace(state, "deny: hard policy violation")}


def n_human_review(state: State):
    return {"trace": _trace(state, "human_review: queued for sign-off")}


def n_audit(state: State):
    fields = state["fields"]
    reasons = [f"{code}: {msg}" for code, msg, _ in state.get("findings", [])] or ["clean: no findings"]
    record = write_record(fields.get("invoice_id"), state["decision"], reasons, fields)
    return {"record": record,
            "trace": _trace(state, f"audit: record #{record['seq']} written, hash {record['hash'][:12]}...")}


def route_decision(state: State):
    return {"approve": "approve", "deny": "deny", "human_review": "human_review"}[state["decision"]]


def build():
    g = StateGraph(State)
    for name, fn in [("extract", n_extract), ("policy", n_policy),
                     ("approve", n_approve), ("deny", n_deny), ("human_review", n_human_review),
                     ("audit", n_audit)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract")
    g.add_edge("extract", "policy")
    g.add_conditional_edges("policy", route_decision,
                            {"approve": "approve", "deny": "deny", "human_review": "human_review"})
    for name in ("approve", "deny", "human_review"):
        g.add_edge(name, "audit")
    g.add_edge("audit", END)
    return g.compile()


APP = build()


def process(document: str) -> State:
    """Run one document through the workflow and return the final state."""
    return APP.invoke({"document": document, "trace": []})
