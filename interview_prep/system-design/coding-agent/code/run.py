"""Run the coding agent, either over the built-in scenarios or on your own task.

    python run.py                 # run all scenarios and self-check (doubles as a test)
    python run.py "fix add"       # run one task against the sandbox repo

Runs offline with the deterministic policy in llm.py. Set a model and a provider key (see
README.md) to route the edit decision through a real model instead.
"""
import sys

from agent import solve

# Each task edits the sandbox repo (calculator.py) and is verified by its own test cases.
FIX_ADD = {
    "text": "add() returns the wrong value; make its tests pass",
    "file": "calculator.py", "func": "add",
    "cases": [("add(2, 3)", 5), ("add(0, 0)", 0), ("add(-1, 1)", 0)],
}
IMPOSSIBLE = {
    "text": "make add(2, 3) return 7",  # no operator swap yields 7: the agent must give up
    "file": "calculator.py", "func": "add",
    "cases": [("add(2, 3)", 7)],
}

SCENARIOS = [
    (FIX_ADD,    "reaches a passing test suite, then opens a PR"),
    (IMPOSSIBLE, "cannot be satisfied, so the agent escalates to a human"),
]


def show(task, note=None):
    s = solve(task)
    print(f"\nTASK: {task['text']}" + (f"\n   ({note})" if note else ""))
    print(f"   plan:         {len(s.get('plan', []))} steps")
    print(f"   tests passed: {bool(s.get('last_test') and s['last_test'].get('passed'))}")
    print(f"   escalated:    {s.get('escalate', False)}")
    print(f"   pr:           {(s.get('pr') or {}).get('pr')}")
    print(f"   notes:        {s.get('notes') or '(none)'}")
    print(f"   trace:        {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # a task description was passed on the command line
        show({"text": " ".join(sys.argv[1:]), "file": "calculator.py", "func": "add",
              "cases": [("add(2, 3)", 5), ("add(0, 0)", 0)]})
        return

    for task, note in SCENARIOS:
        show(task, note)

    # assertions (this is the test)
    green = solve(FIX_ADD)
    assert green.get("last_test", {}).get("passed"), "fix path: tests should end green"
    assert green.get("pr") and not green.get("escalate"), "fix path: should open a PR, not escalate"
    assert green.get("steps", 0) <= 6, "fix path: should stay within the step budget"

    stuck = solve(IMPOSSIBLE)
    assert stuck.get("escalate"), "impossible path: should escalate to a human"
    assert not stuck.get("pr"), "impossible path: should not open a PR"
    assert not stuck.get("last_test", {}).get("passed"), "impossible path: tests never go green"

    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
