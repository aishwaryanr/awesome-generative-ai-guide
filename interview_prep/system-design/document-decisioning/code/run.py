"""Run the underwriting agent, either over the built-in submissions or on your own document.

    python run.py                        # run all scenarios and self-check (doubles as a test)
    python run.py "applicant_name: Jane Okafor; property_value: 420000; requested_coverage: 300000; year_built: 1998; prior_claims: 0; construction: masonry; flood_zone: no"

Runs offline with the deterministic policy in llm.py. Set a model and a provider key (see
README.md) to route the borderline judgment through a real model instead.
"""
import json
import sys

from documents import SAMPLES
from underwriting_agent import decide_document
from audit import AUDIT_LOG, verify_chain

SCENARIOS = [
    ("clean-approve",          "clean file, within authority: approved"),
    ("flood-decline",          "flood zone, no endorsement: hard decline"),
    ("over-insurance-decline", "coverage above property value: hard decline"),
    ("high-impact-refer",      "coverage above auto-bind authority: human sign-off"),
    ("low-confidence-refer",   "poor scan, low extraction confidence: human review"),
    ("missing-field-refer",    "required field missing: human review"),
    ("old-construction-refer", "borderline age, model judgment: human review"),
]


def show(document, note=None):
    s = decide_document(document)
    rec = s.get("audit", {})
    print(f"\nDOC: {document[:76]}{'...' if len(document) > 76 else ''}" + (f"\n     ({note})" if note else ""))
    print(f"   decision:  {s.get('decision')}   referred: {s.get('referred', False)}")
    print(f"   reason:    {s.get('reason')}")
    print(f"   audit:     record #{rec.get('seq')} hash {str(rec.get('hash'))[:12]} model {rec.get('model_version')}")
    print(f"   trace:     {' | '.join(s.get('trace', []))}")
    return s


def main():
    if len(sys.argv) > 1:                       # a document was passed on the command line
        show(" ".join(sys.argv[1:]))
        return

    results = {name: show(SAMPLES[name], note) for name, note in SCENARIOS}

    # assertions (this is the test): every path is exercised and pinned
    assert results["clean-approve"]["decision"] == "approve" and \
        not results["clean-approve"]["referred"], "approve path broke"
    assert results["flood-decline"]["decision"] == "decline", "flood decline path broke"
    assert results["over-insurance-decline"]["decision"] == "decline", "over-insurance decline path broke"
    assert results["high-impact-refer"]["decision"] == "refer" and \
        results["high-impact-refer"]["referred"], "high-impact human sign-off path broke"
    assert results["low-confidence-refer"]["decision"] == "refer" and \
        results["low-confidence-refer"]["referred"], "low-confidence referral path broke"
    assert results["missing-field-refer"]["decision"] == "refer", "missing-field referral path broke"
    assert results["old-construction-refer"]["decision"] == "refer", "model-judgment referral path broke"

    # low-confidence file really was below the floor
    assert results["low-confidence-refer"]["audit"]["min_confidence"] < 0.85, "confidence floor not enforced"

    # PII never appears in the clear anywhere in the audit trail, and is masked where present
    dump = json.dumps(AUDIT_LOG)
    assert "123-45-6789" not in dump and "222-33-4444" not in dump, "raw PII leaked into the audit trail"
    assert results["clean-approve"]["audit"]["pii_masked"].get("ssn") == "***-**-6789", "PII masking broke"

    # one immutable audit record per decision, and the hash chain is intact
    assert len(AUDIT_LOG) == len(SCENARIOS), "every decision must write exactly one audit record"
    assert verify_chain(), "audit hash chain is broken"

    # tamper check: altering a past record must break the chain
    AUDIT_LOG[0]["decision"] = "tampered"
    assert not verify_chain(), "tampering with a record should break the chain"
    AUDIT_LOG[0]["decision"] = "approve"  # restore for a clean exit
    assert verify_chain(), "restore failed"

    print("\nAudit trail: {} immutable records, hash chain verified.".format(len(AUDIT_LOG)))
    print("Sample audit record (PII masked):")
    print(json.dumps({k: results["clean-approve"]["audit"][k] for k in
                      ("seq", "applicant", "decision", "referred", "policy_outcome",
                       "min_confidence", "pii_masked", "model_version")}, indent=2))
    print("\nAll scenario checks passed.")


if __name__ == "__main__":
    main()
