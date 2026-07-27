"""An analytics copilot (text to SQL) as a LangGraph state machine.

This is example code for the case study: link the question to the right tables, generate a
read-only SQL query, block anything that is not a single SELECT, execute it in a read-only
sandbox, rewrite the query when execution errors (self-correction), and answer from the rows.
When the question maps to no table, or the result is empty, the agent abstains instead of
returning a confident wrong number. It runs offline (see llm.py), so it needs no API key.

    START -> schema_link --+--> refuse -> END            (out of scope: links to no table)
                           |
                           +--> generate -> guardrail --+--> refuse -> END   (not read-only)
                                   ^                     |
                                   |                     +--> execute --+--> answer -> END
                                   +----- self-correct ---- (error) ----+--> refuse -> END
                                                                        (empty result: abstain)

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from db import execute
from guardrails import is_read_only
from llm import link_schema, generate_sql, synthesize_answer

MAX_REPAIRS = 2  # cap the self-correction loop so a broken query cannot retry forever


class State(TypedDict, total=False):
    question: str
    tables: List[str]
    sql: Optional[str]
    columns: List[str]
    rows: List
    error: Optional[str]
    repairs: int
    read_only: Optional[bool]
    answer: Optional[str]
    refused: bool
    reason: Optional[str]
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_schema_link(state: State):
    tables = link_schema(state["question"])
    upd = {"tables": tables, "repairs": 0, "error": None,
           "trace": _trace(state, f"schema_link: {tables or 'no tables (out of scope)'}")}
    if not tables:
        upd["reason"] = "out of scope: the question does not map to any table in the warehouse"
    return upd


def n_generate(state: State):
    sql = generate_sql(state["question"], state.get("tables", []), state.get("error"))
    tag = "regenerate (self-correct)" if state.get("error") else "generate"
    return {"sql": sql, "trace": _trace(state, f"{tag}: {sql}")}


def n_guardrail(state: State):
    ok, reason = is_read_only(state["sql"])
    upd = {"read_only": ok, "trace": _trace(state, f"guardrail: {'pass' if ok else 'REFUSE ' + reason}")}
    if not ok:
        upd["reason"] = f"blocked by the read-only guardrail: {reason}"
    return upd


def n_execute(state: State):
    try:
        columns, rows = execute(state["sql"])
    except Exception as e:  # sqlite3.Error and friends
        return {"error": str(e), "repairs": state.get("repairs", 0) + 1,
                "reason": f"the query failed to execute: {e}",
                "trace": _trace(state, f"execute: ERROR {e}")}
    upd = {"columns": columns, "rows": rows, "error": None,
           "trace": _trace(state, f"execute: {len(rows)} row(s)")}
    if not rows:
        upd["reason"] = "the query returned no rows, so no number answers the question"
    return upd


def n_answer(state: State):
    ans = synthesize_answer(state["question"], state["columns"], state["rows"])
    return {"answer": ans, "trace": _trace(state, f"answer: {ans}")}


def n_refuse(state: State):
    reason = state.get("reason", "the request could not be answered safely")
    return {"refused": True,
            "answer": f"I did not answer this. {reason}. A human can take it from here.",
            "trace": _trace(state, "refuse: abstained")}


def route_link(state: State):
    return "generate" if state.get("tables") else "refuse"


def route_guardrail(state: State):
    return "execute" if state.get("read_only") else "refuse"


def route_execute(state: State):
    if state.get("error"):
        return "generate" if state.get("repairs", 0) < MAX_REPAIRS else "refuse"
    return "answer" if state.get("rows") else "refuse"


def build():
    g = StateGraph(State)
    for name, fn in [("schema_link", n_schema_link), ("generate", n_generate),
                     ("guardrail", n_guardrail), ("execute", n_execute),
                     ("answer", n_answer), ("refuse", n_refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "schema_link")
    g.add_conditional_edges("schema_link", route_link, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "guardrail")
    g.add_conditional_edges("guardrail", route_guardrail, {"execute": "execute", "refuse": "refuse"})
    g.add_conditional_edges("execute", route_execute,
                            {"generate": "generate", "answer": "answer", "refuse": "refuse"})
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    return g.compile()


APP = build()


def answer(question: str) -> State:
    """Run one question through the copilot and return the final state."""
    return APP.invoke({"question": question, "trace": []})
