"""An SDR (sales development) agent as a LangGraph state machine.

Given a mock inbound lead, the graph enriches it with signals, qualifies it against the ICP
with an explainable score, drafts a personalized message grounded only in real signals, runs
a brand and compliance guardrail over the draft, and routes every passing draft to a human
approval queue. Nothing is ever sent autonomously.

    START -> ingest -> enrich --+--> (opted out) --> suppressed -> END
                                |
                                +--> qualify --+--> (not qualified) --> disqualify -> END
                                               |
                                               +--> (qualified) --> draft --> guardrail --+
                                                                                          |
                                                              (fail) --> blocked -> END <-+
                                                                                          |
                                                              (pass) --> human_approval --+--> END

    Opt-out is an early hard gate: a suppressed contact is never enriched into a draft. The
    guardrail re-checks suppression before any send, so consent is enforced in two places.

It runs offline (see llm.py and enrichment.py). The path to real scale, deliverability, and
production observability is described in the case study writeup, not coded here.

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from enrichment import enrich, domain_of
from qualify import qualify
from llm import draft as draft_message
from compliance import check as compliance_check, compliant_footer


class State(TypedDict, total=False):
    lead: dict
    signals: List
    opt_out: bool
    qualification: dict
    draft: Optional[dict]
    guardrail: Optional[dict]
    status: str          # queued_for_approval | blocked | disqualified | suppressed
    sent: bool           # always False: a human sends, never the agent
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_ingest(state: State):
    lead = state["lead"]
    return {"sent": False, "status": "in_progress",
            "trace": _trace(state, f"ingest: {lead.get('email')} ({domain_of(lead.get('email',''))})")}


def n_enrich(state: State):
    data = enrich(state["lead"].get("email", ""))
    sigs = data["signals"]
    return {"signals": sigs, "opt_out": data["opt_out"],
            "trace": _trace(state, f"enrich: {len(sigs)} signal(s), opt_out={data['opt_out']}")}


def n_qualify(state: State):
    q = qualify(state.get("signals") or [])
    return {"qualification": q,
            "trace": _trace(state, f"qualify: score={q['score']} -> {q['disposition']}")}


def n_draft(state: State):
    d = draft_message(state["lead"], state.get("signals") or [])
    d["body"] = d["body"] + "\n\n" + compliant_footer()   # harness guarantees the CAN-SPAM elements
    return {"draft": d, "trace": _trace(state, f"draft: {len(d.get('claims', []))} grounded claim(s)")}


def n_guardrail(state: State):
    g = compliance_check(state["draft"], state.get("signals") or [], state.get("opt_out", False))
    label = "pass" if g["ok"] else "FAIL (" + "; ".join(g["failures"]) + ")"
    return {"guardrail": g, "trace": _trace(state, f"guardrail: {label}")}


def n_human_approval(state: State):
    return {"status": "queued_for_approval",
            "trace": _trace(state, "human_approval: queued for a human to review and send")}


def n_disqualify(state: State):
    q = state.get("qualification", {})
    return {"status": "disqualified",
            "trace": _trace(state, f"disqualify: {q.get('disposition')} at score {q.get('score')}")}


def n_blocked(state: State):
    return {"status": "blocked",
            "trace": _trace(state, "blocked: guardrail refused the draft, no send")}


def n_suppressed(state: State):
    return {"status": "suppressed",
            "trace": _trace(state, "suppressed: contact opted out, no enrichment or outreach")}


def route_suppression(state: State):
    return "suppressed" if state.get("opt_out") else "qualify"


def route_qualify(state: State):
    return "draft" if state["qualification"]["disposition"] == "qualified" else "disqualify"


def route_guardrail(state: State):
    return "human_approval" if state["guardrail"]["ok"] else "blocked"


def build():
    g = StateGraph(State)
    for name, fn in [("ingest", n_ingest), ("enrich", n_enrich), ("qualify", n_qualify),
                     ("draft", n_draft), ("guardrail", n_guardrail),
                     ("human_approval", n_human_approval), ("disqualify", n_disqualify),
                     ("blocked", n_blocked), ("suppressed", n_suppressed)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "enrich")
    g.add_conditional_edges("enrich", route_suppression,
                            {"suppressed": "suppressed", "qualify": "qualify"})
    g.add_conditional_edges("qualify", route_qualify,
                            {"draft": "draft", "disqualify": "disqualify"})
    g.add_edge("draft", "guardrail")
    g.add_conditional_edges("guardrail", route_guardrail,
                            {"human_approval": "human_approval", "blocked": "blocked"})
    for terminal in ("human_approval", "disqualify", "blocked", "suppressed"):
        g.add_edge(terminal, END)
    return g.compile()


APP = build()


def run_lead(lead: dict) -> State:
    """Run one inbound lead through the agent and return the final state."""
    return APP.invoke({"lead": lead, "trace": []})
