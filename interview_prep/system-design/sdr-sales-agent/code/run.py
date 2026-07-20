"""Run the SDR agent, either over the built-in scenarios or on your own inbound lead.

    python run.py                       # run all scenarios and self-check (doubles as a test)
    python run.py "we want a demo"      # run one lead (uses a demo contact) with your note

Runs offline with the deterministic drafter in llm.py. Set a model and a provider key (see
README.md) to route drafting through a real model instead.
"""
import sys

from sdr_agent import run_lead

SCENARIOS = [
    ({"email": "priya@acme.io", "name": "Priya Nair"},
     "strong fit, verified signals: qualified, grounded draft, queued for a human"),
    ({"email": "sam@startup.dev", "name": "Sam Okafor"},
     "qualified, but the draft leans on an unverified rumor: guardrail blocks the fabrication"),
    ({"email": "jordan@bigco.com", "name": "Jordan Lee"},
     "contact has opted out: early suppression gate blocks all outreach"),
    ({"email": "alex@gmail.com", "name": "Alex Kim"},
     "personal mailbox, no company signals: not qualified, no outreach"),
]


def show(lead, note=None):
    s = run_lead(lead)
    print(f"\nLEAD: {lead.get('name')} <{lead.get('email')}>" + (f"\n   ({note})" if note else ""))
    q = s.get("qualification", {})
    print(f"   qualification: score={q.get('score')} disposition={q.get('disposition')}")
    print(f"   status:        {s.get('status')}  (sent={s.get('sent')})")
    g = s.get("guardrail")
    if g:
        print(f"   guardrail:     {'pass' if g['ok'] else 'FAIL -> ' + '; '.join(g['failures'])}")
    if s.get("status") == "queued_for_approval":
        print("   draft (for human review):")
        for line in s["draft"]["body"].splitlines():
            print(f"       {line}")
    print(f"   trace:         {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # a note was passed on the command line
        show({"email": "priya@acme.io", "name": "Priya Nair", "note": " ".join(sys.argv[1:])})
        return

    for lead, note in SCENARIOS:
        show(lead, note)

    # assertions (this is the test)
    a = run_lead({"email": "priya@acme.io", "name": "Priya Nair"})
    assert a["qualification"]["disposition"] == "qualified", "acme should qualify"
    assert a["status"] == "queued_for_approval", "qualified, grounded draft should reach human approval"
    assert a["guardrail"]["ok"], "verified-signal draft should pass the guardrail"
    assert a["sent"] is False, "the agent must never send; a human does"

    b = run_lead({"email": "sam@startup.dev", "name": "Sam Okafor"})
    assert b["qualification"]["disposition"] == "qualified", "startup.dev should qualify on fit"
    assert b["status"] == "blocked", "a draft citing an unverified rumor must be blocked"
    assert any("unfaithful" in f for f in b["guardrail"]["failures"]), "fabrication must be the reason"

    c = run_lead({"email": "jordan@bigco.com", "name": "Jordan Lee"})
    assert c["status"] == "suppressed", "an opted-out contact must be suppressed"
    assert c.get("draft") is None, "a suppressed contact is never drafted to"
    assert c["sent"] is False, "nothing goes to a suppressed contact"

    d = run_lead({"email": "alex@gmail.com", "name": "Alex Kim"})
    assert d["qualification"]["disposition"] != "qualified", "a no-company lead should not qualify"
    assert d["status"] == "disqualified", "unqualified leads get no outreach"
    assert d.get("draft") is None, "no draft is written for an unqualified lead"

    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
