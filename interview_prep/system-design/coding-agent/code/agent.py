"""A coding agent as a LangGraph state machine: the harness around the model.

This is example code for the case study. It is the loop and scaffolding that turn a model into
an agent that can edit a repo and prove its work: read a file, run the test suite, propose an
edit, run the tests again, and repeat until the tests are green or a budget is hit. It shows
the load-bearing parts in miniature:

  - a BOUNDED loop (a hard step cap, so a stuck agent cannot spin forever)
  - a VERIFIER (the test suite is the ground truth; the harness only trusts green)
  - COMPACTION (the running history is summarized into a compact note so it stays small)
  - a GUARDRAIL (repeated or exhausted failure stops the loop and escalates to a human)

The model owns one decision (what edit to try next, in llm.py). The harness owns everything
else. It runs offline with a deterministic policy, so it needs no API key to try.

    START -> plan -> agent --read--> read_file --------> agent
                             --test--> run_tests -> compact -> agent
                             --edit--> edit_file -> run_tests -> compact -> agent   (loop, bounded)
                             --open_pr--> open_pr ------> END   (tests are green)
                             --escalate-> escalate ------> END   (budget hit or fixes exhausted)
"""
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

import sandbox
from llm import propose_edit

MAX_STEPS = 6          # cap the loop so a stuck agent cannot edit forever
HISTORY_KEEP = 3       # how many recent history lines to keep verbatim before compacting


class State(TypedDict, total=False):
    task: dict                 # {text, file, func, cases}
    repo: Dict[str, str]       # the working copy of the repository
    read: bool                 # has the agent read the file yet
    last_test: Optional[dict]  # the most recent test result (the verifier's verdict)
    tried: List[str]           # operators already attempted, so we never repeat a failure
    steps: int                 # edits made so far, checked against MAX_STEPS
    history: List[str]         # running log of steps (gets compacted)
    notes: str                 # compacted summary of older history (externalized state)
    plan: List[str]            # a small todo list the harness decomposes the task into
    decision: dict             # the harness's chosen next action
    pr: Optional[dict]
    escalate: bool
    trace: List[str]


def _log(state, msg):
    return state.get("history", []) + [msg]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


# --- the plan node: decompose the task into a small todo list before working ----------
# Real harnesses (Codex plan mode, Claude Code todos) draft a plan for a longer task and
# track it as they go. Here the plan is short and fixed, enough to show where it lives.

def n_plan(state: State):
    task = state["task"]
    plan = [f"locate {task['func']}() in {task['file']}",
            "run the tests to see the failure",
            f"edit {task['func']}() and re-run until green",
            "open a PR, or escalate if it cannot be made green"]
    return {"plan": plan, "trace": _trace(state, f"plan -> {len(plan)} steps")}


# --- the agent node: decide the next action (control flow lives in the harness) --------

def n_agent(state: State):
    task, repo = state["task"], state["repo"]
    last = state.get("last_test")

    if not state.get("read"):
        decision = {"action": "read"}
    elif last is None:
        decision = {"action": "test"}                       # baseline: see what fails first
    elif last.get("passed"):
        decision = {"action": "open_pr"}                    # verifier is green: ship it
    elif state.get("steps", 0) >= MAX_STEPS:
        decision = {"action": "escalate", "why": "step budget exhausted"}
    else:
        current_op = sandbox.current_operator(repo, task["file"], task["func"])
        op = propose_edit(task, sandbox.read_file(repo, task["file"]), last,
                          state.get("tried", []), current_op)
        if op is None:
            decision = {"action": "escalate", "why": "no remaining fix to try"}
        else:
            decision = {"action": "edit", "operator": op}

    label = decision["action"] + (f":{decision.get('operator')}" if decision.get("operator") else "")
    return {"decision": decision, "trace": _trace(state, f"agent -> {label}")}


# --- tool nodes: the harness executes what the agent asked for -------------------------

def n_read(state: State):
    src = sandbox.read_file(state["repo"], state["task"]["file"])
    return {"read": True, "history": _log(state, f"read {state['task']['file']} ({len(src)} chars)"),
            "trace": _trace(state, "tool: read_file")}


def n_edit(state: State):
    task = state["task"]
    op = state["decision"]["operator"]
    repo, msg = sandbox.edit_file(state["repo"], task["file"], task["func"], op)
    return {"repo": repo, "tried": state.get("tried", []) + [op],
            "steps": state.get("steps", 0) + 1,
            "history": _log(state, f"edit: {msg}"),
            "trace": _trace(state, f"tool: edit_file ({op})")}


def n_test(state: State):
    task = state["task"]
    result = sandbox.run_tests(state["repo"], task["file"], task["cases"])
    if result["passed"]:
        line = "test: PASS (all cases green)"
    else:
        f = result["failures"][0]
        line = f"test: FAIL {f['call']} -> {f['actual']!r}, expected {f['expected']!r}"
    return {"last_test": result, "history": _log(state, line),
            "trace": _trace(state, "tool: run_tests -> " + ("pass" if result["passed"] else "fail"))}


def n_compact(state: State):
    """Compaction: fold older history into a short note so the context stays small.

    Real harnesses summarize a long transcript into a compact brief and continue from it. Here
    we keep the last few lines verbatim and roll everything older into `notes`, which is the
    same move: survive a long task by carrying a summary instead of the whole history.
    """
    history = state.get("history", [])
    if len(history) <= HISTORY_KEEP:
        return {}
    old, recent = history[:-HISTORY_KEEP], history[-HISTORY_KEEP:]
    tried = state.get("tried", [])
    note = (f"[compacted {len(old)} earlier steps] operators tried so far: "
            f"{', '.join(tried) or 'none'}.")
    return {"history": recent, "notes": note,
            "trace": _trace(state, f"compact: folded {len(old)} step(s) into notes")}


def n_open_pr(state: State):
    task = state["task"]
    pr = sandbox.open_pr(state["repo"], task["file"], f"Fix {task['func']}(): {task['text']}")
    return {"pr": pr, "escalate": False,
            "trace": _trace(state, f"open_pr -> {pr['pr']}")}


def n_escalate(state: State):
    why = state["decision"].get("why", "unresolved")
    return {"escalate": True, "pr": None,
            "trace": _trace(state, f"escalate: handed to a human ({why})")}


# --- routing --------------------------------------------------------------------------

def route_agent(state: State):
    return state["decision"]["action"]


def build():
    g = StateGraph(State)
    for name, fn in [("plan", n_plan), ("agent", n_agent), ("read", n_read), ("edit", n_edit),
                     ("test", n_test), ("compact", n_compact),
                     ("open_pr", n_open_pr), ("escalate", n_escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "plan")
    g.add_edge("plan", "agent")
    g.add_conditional_edges("agent", route_agent,
                            {"read": "read", "test": "test", "edit": "edit",
                             "open_pr": "open_pr", "escalate": "escalate"})
    g.add_edge("read", "agent")
    g.add_edge("edit", "test")          # every edit is followed by the verifier
    g.add_edge("test", "compact")       # then compaction keeps the history small
    g.add_edge("compact", "agent")
    g.add_edge("open_pr", END)
    g.add_edge("escalate", END)
    # a recursion ceiling well above MAX_STEPS, so the loop is bounded by our cap, not the framework
    return g.compile()


APP = build()


def solve(task: dict) -> State:
    """Run one coding task through the harness and return the final state."""
    init: State = {"task": task, "repo": sandbox.fresh_repo(), "read": False,
                   "last_test": None, "tried": [], "steps": 0,
                   "history": [], "notes": "", "plan": [], "trace": []}
    return APP.invoke(init, {"recursion_limit": 100})
