"""A clinical documentation assistant as a LangGraph state machine.

This is example code for the case study: ingest an encounter transcript, extract a structured
SOAP note, run a faithfulness check that flags any statement the transcript does not support,
and hold the note as a DRAFT that a clinician must review and sign before it is final. It is
deliberately small and runs offline (see llm.py). The path to real scale and production
(streaming ASR, terminology grounding, EHR write-back, Arize observability, evaluation gates)
is described in the case study writeup, not coded here.

    START -> ingest -> extract -> faithfulness -> review_gate -> END
                          |                             |
              (offline: grounded extraction)   flags? -> blocked, needs correction
                                               clean?  -> draft ready for clinician review

The graph NEVER marks a note final. Finalization happens only in sign_note(), which represents
a human clinician reviewing and signing. A note with open faithfulness flags cannot be signed.

ALL DATA IS SYNTHETIC. No real protected health information is used anywhere.

Run:  python run.py
"""
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END

from llm import extract_soap, check_faithfulness, split_utterances, SECTIONS


class State(TypedDict, total=False):
    transcript: str
    draft: Optional[dict]        # a note produced elsewhere (e.g. a vendor model) to verify
    soap: dict
    flags: List[dict]
    status: str
    review_required: bool
    finalized: bool
    signed_by: Optional[str]
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def n_ingest(state: State):
    # In production this is streaming speech-to-text with speaker diarization. Here the
    # transcript arrives as text, so ingest just records that the encounter was received.
    n = len(split_utterances(state["transcript"]))
    return {"trace": _trace(state, f"ingest: {n} utterance(s)")}


def n_extract(state: State):
    # If a draft note was supplied, verify THAT note (a common deployment: a vendor model
    # writes the note, and your pipeline is the safety check). Otherwise extract one here.
    if state.get("draft") is not None:
        soap = {s: list(state["draft"].get(s, [])) for s in SECTIONS}
        return {"soap": soap, "trace": _trace(state, "extract: using supplied draft note")}
    soap = extract_soap(state["transcript"])
    counts = ", ".join(f"{s}={len(soap[s])}" for s in SECTIONS)
    return {"soap": soap, "trace": _trace(state, f"extract: SOAP drafted ({counts})")}


def n_faithfulness(state: State):
    flags = check_faithfulness(state["transcript"], state["soap"])
    return {"flags": flags, "trace": _trace(state, f"faithfulness: {len(flags)} unsupported statement(s)")}


def n_review_gate(state: State):
    flags = state.get("flags", [])
    if flags:
        status = "blocked_unsupported_statements"
    else:
        status = "draft_ready_for_clinician_review"
    # A clinician always reviews. The pipeline never produces a final note on its own.
    return {"status": status, "review_required": True, "finalized": False,
            "signed_by": None, "trace": _trace(state, f"review_gate: {status}")}


def build():
    g = StateGraph(State)
    for name, fn in [("ingest", n_ingest), ("extract", n_extract),
                     ("faithfulness", n_faithfulness), ("review_gate", n_review_gate)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "faithfulness")
    g.add_edge("faithfulness", "review_gate")
    g.add_edge("review_gate", END)
    return g.compile()


APP = build()


def run_scribe(transcript: str, draft: Optional[dict] = None) -> State:
    """Run one encounter through the pipeline and return the DRAFT note state.

    The result is always a draft that requires clinician review. Pass `draft` to verify a note
    that was produced elsewhere instead of extracting a fresh one.
    """
    init: State = {"transcript": transcript, "trace": []}
    if draft is not None:
        init["draft"] = draft
    return APP.invoke(init)


def sign_note(state: State, clinician: str) -> State:
    """Human sign-off. This is the only path to a final note, and it refuses open flags.

    A clinician reviews the draft and signs. Signing is blocked while any statement is
    unsupported, so a fabricated symptom or dose can never reach a signed record unreviewed.
    """
    if not clinician:
        raise ValueError("a clinician identity is required to sign a note")
    if state.get("flags"):
        return {**state, "status": "unsigned_needs_correction", "finalized": False,
                "signed_by": None,
                "trace": _trace(state, f"sign_note: REFUSED, {len(state['flags'])} unsupported statement(s) must be corrected first")}
    return {**state, "status": "final_signed", "finalized": True, "signed_by": clinician,
            "trace": _trace(state, f"sign_note: reviewed and signed by {clinician}")}
