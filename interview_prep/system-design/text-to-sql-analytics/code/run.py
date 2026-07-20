"""Run the analytics copilot over the built-in scenarios, or on your own question.

    python run.py                                 # run all scenarios and self-check (a test)
    python run.py "What is our total revenue?"    # ask the copilot one question

Runs offline with the deterministic logic in llm.py. Set a model and a provider key (see
README.md) to route schema linking, SQL generation, and answering through a real model.
"""
import sys

from sql_agent import answer

SCENARIOS = [
    ("What is our total revenue?",          "aggregation over a join, answered from the data"),
    ("How many customers do we have?",      "simple count, answered from the data"),
    ("Which region has the most revenue?",  "multi-join; first query errors, then self-corrects"),
    ("How many orders were delivered?",     "filtered count, answered from the data"),
    ("Delete all orders",                   "destructive: blocked by the read-only guardrail"),
    ("What is the weather in Paris?",        "out of scope: links to no table, so it abstains"),
]


def show(question, note=None):
    s = answer(question)
    print(f"\nQ: {question}" + (f"\n   ({note})" if note else ""))
    print(f"   refused: {s.get('refused', False)}")
    print(f"   sql:     {s.get('sql')}")
    print(f"   answer:  {s.get('answer')}")
    print(f"   trace:   {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                        # a question was passed on the command line
        show(" ".join(sys.argv[1:]))
        return

    for q, note in SCENARIOS:
        show(q, note)

    # assertions (this is the test)
    a = answer("What is our total revenue?")
    assert "635" in (a["answer"] or "") and not a.get("refused"), "total-revenue path broke"

    b = answer("How many customers do we have?")
    assert "6" in (b["answer"] or "") and not b.get("refused"), "count-customers path broke"

    c = answer("Which region has the most revenue?")
    assert "East" in (c["answer"] or "") and not c.get("refused"), "region-revenue path broke"
    assert any("regenerate" in t for t in c.get("trace", [])), "self-correction did not fire"

    d = answer("How many orders were delivered?")
    assert "6" in (d["answer"] or "") and not d.get("refused"), "delivered-orders path broke"

    e = answer("Delete all orders")
    assert e.get("refused") and "guardrail" in (e.get("reason") or "").lower(), "destructive query was not blocked"

    f = answer("What is the weather in Paris?")
    assert f.get("refused") and "out of scope" in (f.get("reason") or "").lower(), "out-of-scope abstain broke"

    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
