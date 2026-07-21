"""Run the voice support agent over the built-in scenarios or on your own utterance.

    python run.py                              # run all scenarios and self-check (doubles as a test)
    python run.py "Where is my order 5012?"    # run one voice turn from the command line

Runs offline with the deterministic policy in llm.py. Set a model and a provider key (see
README.md) to route decisions through a real model instead. Audio is out of scope for this
demo: a real system wraps this with streaming ASR in front and streaming TTS behind.
"""
import sys

from turn_graph import handle_turn


# (transcript, barge_in, note). barge_in is an utterance the caller speaks WHILE the agent
# is replying, which cuts the turn short and re-plans.
SCENARIOS = [
    ("What is your return policy?", None,
     "info question, answered from the knowledge base"),
    ("Where is my order 10432?", None,
     "action, resolved with the order_lookup tool mid-turn"),
    ("My item arrived damaged, I need a replacement", None,
     "action, resolved by opening a ticket"),
    ("I want a refund for order 10432", None,
     "high-impact, handed off to a human"),
    ("What is the meaning of life?", None,
     "out of scope, handed off to a human"),
    ("Where is my order 10432?", "actually cancel that order",
     "BARGE-IN: caller interrupts mid-answer, turn is cut short and re-planned, then handed off"),
]


def show(transcript, barge_in=None, note=None):
    s = handle_turn(transcript, barge_in)
    print(f"\nCALLER: {transcript}" + (f"\n  [barge-in]: {barge_in}" if barge_in else "")
          + (f"\n  ({note})" if note else ""))
    print(f"   interrupted: {s.get('interrupted', False)}")
    print(f"   escalated:   {s.get('escalate', False)}")
    print(f"   reply:       {s.get('reply')}")
    print(f"   latency:     {s.get('latency_ms', 0)} ms  "
          + " + ".join(f"{stage} {ms}" for stage, ms in s.get("budget", [])))
    print(f"   trace:       {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # an utterance was passed on the command line
        show(" ".join(sys.argv[1:]))
        return

    for transcript, barge_in, note in SCENARIOS:
        show(transcript, barge_in, note)

    # assertions (this is the test)
    a = handle_turn("What is your return policy?")
    assert "30 days" in (a["reply"] or "") and not a.get("escalate"), "return-policy path broke"

    b = handle_turn("Where is my order 10432?")
    assert "10432" in (b["reply"] or "") and "in transit" in (b["reply"] or "") \
        and not b.get("escalate"), "order-lookup path broke"

    c = handle_turn("My item arrived damaged, I need a replacement")
    assert "ticket" in (c["reply"] or "").lower() and not c.get("escalate"), "ticket path broke"

    r = handle_turn("I want a refund for order 10432")
    assert r.get("escalate"), "refund should hand off to a human"

    d = handle_turn("What is the meaning of life?")
    assert d.get("escalate"), "out-of-scope handoff path broke"

    # barge-in: the caller interrupts the order answer to cancel, which re-plans to a handoff.
    e = handle_turn("Where is my order 10432?", barge_in="actually cancel that order")
    assert e.get("interrupted"), "barge-in should cut the turn short"
    assert e.get("escalate"), "the re-planned cancellation should hand off to a human"
    assert e.get("latency_ms", 0) > 0, "a voice turn should charge a latency budget"

    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
