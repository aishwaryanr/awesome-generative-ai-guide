"""Run the clinical scribe over the built-in scenarios, or on your own transcript.

    python run.py                                   # run all scenarios and self-check (a test)
    python run.py "Patient: I have a sore throat."  # run one transcript through the pipeline

Runs offline with the deterministic policy in llm.py. Set a model and a provider key (see
README.md) to route extraction and the faithfulness check through a real model instead.

ALL DATA BELOW IS SYNTHETIC. There is no real protected health information here, and there
must never be.
"""
import sys

from scribe import run_scribe, sign_note, SECTIONS

# A synthetic outpatient encounter. No real patient, no real PHI.
CLEAN_TRANSCRIPT = """\
Patient: I have had pain in my right knee for about two weeks.
Patient: It hurts most when I climb stairs and it feels stiff in the morning.
Patient: I have not had any fever and there was no injury that I remember.
Clinician: Your temperature is 98.6 degrees and your blood pressure is 124 over 78.
Clinician: On exam there is mild swelling over the right knee and it is tender to touch.
Clinician: Your range of motion is slightly reduced and there is no instability.
Clinician: This looks consistent with early osteoarthritis of the right knee.
Clinician: I recommend ice, rest, and ibuprofen 400 mg as needed for pain.
Clinician: Let us order a knee x-ray and follow up in three weeks.
"""

# A note "drafted elsewhere" (imagine a vendor model) for the SAME knee encounter, but it has
# hallucinated a diagnosis and a prescription that were never discussed. This is the single most
# dangerous failure for a clinical scribe, and the faithfulness check must catch it.
FABRICATED_DRAFT = {
    "S": ["I have had pain in my right knee for about two weeks."],
    "O": ["On exam there is mild swelling over the right knee and it is tender to touch."],
    "A": ["This looks consistent with early osteoarthritis of the right knee.",
          "Acute otitis media of the right ear."],
    "P": ["Let us order a knee x-ray and follow up in three weeks.",
          "Start amoxicillin 500 mg twice daily for the ear infection."],
}

# A very short encounter, to show the note is still held for review and never auto-finalized.
SHORT_TRANSCRIPT = """\
Patient: I would like to renew my walking-shoe recommendation, nothing else is bothering me.
Clinician: Everything looks stable today, keep up the daily walks.
"""


def show_note(state, title):
    print(f"\n=== {title} ===")
    for s in SECTIONS:
        items = state["soap"].get(s, [])
        print(f"  {s}: " + ("; ".join(items) if items else "(none)"))
    if state.get("flags"):
        print("  FAITHFULNESS FLAGS (unsupported by transcript):")
        for f in state["flags"]:
            print(f"    [{f['section']}] {f['statement']}  <- unsupported: {', '.join(f['unsupported'])}")
    print(f"  status: {state['status']}  |  review_required: {state['review_required']}  |  finalized: {state.get('finalized')}")
    print(f"  trace: {' | '.join(state['trace'])}")


def scenarios():
    # 1. Clean transcript: extract a SOAP note, nothing unsupported, held for clinician review.
    clean = run_scribe(CLEAN_TRANSCRIPT)
    show_note(clean, "1. Clean encounter -> draft SOAP note")
    signed = sign_note(clean, clinician="Dr. Rao (synthetic)")
    print(f"  after sign-off: status={signed['status']} finalized={signed['finalized']} signed_by={signed['signed_by']}")

    # 2. A drafted note with a fabricated diagnosis and prescription: the check must catch it,
    #    and sign-off must be refused until it is corrected.
    caught = run_scribe(CLEAN_TRANSCRIPT, draft=FABRICATED_DRAFT)
    show_note(caught, "2. Fabricated diagnosis + drug -> caught before sign-off")
    refused = sign_note(caught, clinician="Dr. Rao (synthetic)")
    print(f"  after sign attempt: status={refused['status']} finalized={refused['finalized']}")

    # 3. A short encounter: still a draft, still requires review, never auto-final.
    short = run_scribe(SHORT_TRANSCRIPT)
    show_note(short, "3. Short encounter -> still requires clinician review")

    # --- assertions (this is the test) ------------------------------------------------
    assert all(clean["soap"][s] for s in ("S", "O", "A", "P")), "clean note missing a SOAP section"
    assert clean["flags"] == [], "clean note should have no unsupported statements"
    assert clean["status"] == "draft_ready_for_clinician_review", "clean note status wrong"
    assert clean["finalized"] is False, "the pipeline must never finalize a note on its own"
    assert signed["finalized"] is True and signed["signed_by"], "clinician sign-off should finalize a clean note"

    fab_statements = " ".join(f["statement"].lower() for f in caught["flags"])
    assert caught["flags"], "the fabricated statements were not caught"
    assert "amoxicillin" in fab_statements, "the fabricated prescription was not flagged"
    assert "otitis" in fab_statements, "the fabricated diagnosis was not flagged"
    assert caught["status"] == "blocked_unsupported_statements", "fabricated note should be blocked"
    assert refused["finalized"] is False, "a note with open flags must not be signable"

    assert short["review_required"] is True and short["finalized"] is False, "short note must still be reviewed"

    print("\nAll scenario checks passed.")


def main():
    if len(sys.argv) > 1:  # a transcript was passed on the command line
        state = run_scribe(" ".join(sys.argv[1:]))
        show_note(state, "your transcript")
        return
    scenarios()


if __name__ == "__main__":
    main()
