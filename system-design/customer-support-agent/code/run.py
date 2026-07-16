"""Run the support agent, either over the built-in scenarios or on your own question.

    python run.py                              # run all scenarios and self-check (doubles as a test)
    python run.py "Where is my order 5012?"    # ask the agent one question

Runs offline with the deterministic policy in llm.py. Set a model and a provider key (see
README.md) to route decisions through a real model instead.
"""
import sys

from support_agent import answer

SCENARIOS = [
    ("What is your return policy?",              "info question, answered from the knowledge base"),
    ("Where is my order 10432?",                 "action, resolved with the order_lookup tool"),
    ("My item arrived damaged, I need a replacement", "action, resolved by opening a ticket"),
    ("I want a refund for order 10432",          "high-impact, escalated for human approval"),
    ("What is the meaning of life?",             "out of scope, escalated to a human"),
]


def show(question, note=None):
    s = answer(question)
    print(f"\nQ: {question}" + (f"\n   ({note})" if note else ""))
    print(f"   escalated: {s.get('escalate', False)}")
    print(f"   answer:    {s.get('answer')}")
    print(f"   trace:     {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # a question was passed on the command line
        show(" ".join(sys.argv[1:]))
        return

    for q, note in SCENARIOS:
        show(q, note)

    # assertions (this is the test)
    a = answer("What is your return policy?")
    assert "30 days" in (a["answer"] or "") and not a.get("escalate"), "return-policy path broke"
    b = answer("Where is my order 10432?")
    assert "10432" in (b["answer"] or "") and "in transit" in (b["answer"] or "") and not b.get("escalate"), "order-lookup path broke"
    c = answer("My item arrived damaged, I need a replacement")
    assert "ticket" in (c["answer"] or "").lower() and not c.get("escalate"), "ticket path broke"
    r = answer("I want a refund for order 10432")
    assert r.get("escalate"), "refund should escalate for human approval"
    d = answer("What is the meaning of life?")
    assert d.get("escalate"), "escalation path broke"
    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
