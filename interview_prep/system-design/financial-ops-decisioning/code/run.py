"""Run the decisioning agent, either over the built-in scenarios or on your own document.

    python run.py                                          # run all scenarios and self-check (a test)
    python run.py "Vendor: Acme | Invoice: INV-9 | Amount: USD 90 | Date: 2026-07-10 | Category: software"

Runs offline with the deterministic parser and rulebook. Set a model and a provider key (see
README.md) to route field extraction through a real model instead. Documents are labelled text
(Vendor: ... | Invoice: INV-... | Amount: USD ... | Date: YYYY-MM-DD | Category: ...); a real
model reads freeform invoices and expenses.
"""
import sys

from decision_agent import process
from audit import AUDIT_LOG, verify_chain, reset

SCENARIOS = [
    ("Vendor: Acme Cloud | Invoice: INV-1001 | Amount: USD 1250.00 | Date: 2026-07-10 | Category: software",
     "within policy, auto-approved"),
    ("Vendor: Bar One | Invoice: INV-1002 | Amount: USD 90.00 | Date: 2026-07-11 | Category: alcohol",
     "non-reimbursable category, denied"),
    ("Vendor: BigIron | Invoice: INV-1003 | Amount: USD 42000.00 | Date: 2026-07-09 | Category: hardware",
     "high-impact amount, routed to a human for sign-off"),
    ("Vendor: City Cafe | Invoice: INV-1004 | Amount: USD 450.00 | Date: 2026-07-08 | Category: meals",
     "over the per-category limit, routed to a human"),
    ("Reimburse USD 240 for a team lunch",
     "thin document, low confidence, routed to a human"),
    ("Vendor: Acme Cloud | Invoice: INV-1001 | Amount: USD 1250.00 | Date: 2026-07-10 | Category: software",
     "duplicate invoice id, routed to a human"),
]


def show(document, note=None):
    s = process(document)
    f = s["fields"]
    print(f"\nDOC: {document[:78]}" + (f"\n   ({note})" if note else ""))
    print(f"   extracted: vendor={f.get('vendor')!r} amount={f.get('amount')!r} "
          f"category={f.get('category')!r} confidence={f.get('_confidence')}")
    print(f"   decision:  {s.get('decision')}")
    print(f"   reasons:   {[r for _, r, _ in s.get('findings', [])] or ['clean']}")
    print(f"   trace:     {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # a document was passed on the command line
        show(" ".join(sys.argv[1:]))
        return

    reset()
    for doc, note in SCENARIOS:
        show(doc, note)

    # assertions (this is the test): each path, plus the audit trail integrity
    reset()

    a = process("Vendor: Acme | Invoice: INV-2001 | Amount: USD 1200 | Date: 2026-07-10 | Category: software")
    assert a["decision"] == "approve", "clean invoice should auto-approve"

    b = process("Vendor: Bar | Invoice: INV-2002 | Amount: USD 80 | Date: 2026-07-10 | Category: alcohol")
    assert b["decision"] == "deny", "non-reimbursable category should deny"

    c = process("Vendor: BigCo | Invoice: INV-2003 | Amount: USD 25000 | Date: 2026-07-10 | Category: hardware")
    assert c["decision"] == "human_review", "high-impact amount should route to a human"

    d = process("Vendor: Cafe | Invoice: INV-2004 | Amount: USD 450 | Date: 2026-07-10 | Category: meals")
    assert d["decision"] == "human_review", "over-limit amount should route to a human"

    e = process("reimburse USD 50 for coffee")
    assert e["decision"] == "human_review", "thin low-confidence document should route to a human"

    f1 = process("Vendor: Acme | Invoice: INV-2005 | Amount: USD 900 | Date: 2026-07-10 | Category: software")
    assert f1["decision"] == "approve", "first submission should approve"
    f2 = process("Vendor: Acme | Invoice: INV-2005 | Amount: USD 900 | Date: 2026-07-10 | Category: software")
    assert f2["decision"] == "human_review", "duplicate invoice id should route to a human"

    assert all(s.get("record") for s in (a, b, c, d, e, f1, f2)), "every decision must write an audit record"
    assert len(AUDIT_LOG) == 7, f"expected 7 audit records, got {len(AUDIT_LOG)}"
    assert verify_chain(), "audit hash chain must verify"

    print("\nAll scenario checks passed. Audit chain verified over", len(AUDIT_LOG), "records.")


if __name__ == "__main__":
    main()
