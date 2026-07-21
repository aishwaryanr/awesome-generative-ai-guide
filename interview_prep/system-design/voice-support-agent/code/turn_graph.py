"""One voice turn, modeled as a LangGraph state machine.

Audio is out of scope for a tiny runnable demo, so this models the CONVERSATION control
flow of a voice turn as text. The caller's speech arrives already transcribed, the agent
retrieves or calls a tool and forms a reply, and the reply text stands in for the audio a
real system would stream back through text-to-speech (TTS). A real system wraps this graph
with streaming ASR (speech to text) in front and streaming TTS behind, or replaces both
with a single speech-to-speech model. The control-flow decisions modeled here stay the same.

Two things make this a VOICE turn rather than a chat turn:

1. A per-node latency budget. Every node adds its stage cost to `latency_ms`, because in a
   phone call the wall clock is the product: dead air past roughly a second reads as a
   dropped call. The offline costs below are illustrative placeholders, not measurements.

2. Barge-in (interruption). A caller can start talking while the agent is still speaking.
   When a barge-in utterance is present, the harness cuts the in-progress turn short and
   re-plans against what the caller just said. In a real system a duplex audio pipeline
   watches for barge-in continuously during TTS; here we check at the speak boundary.

    START -> asr -> retrieve -> agent --+--> order_lookup --+
                                        |                    |  (loop, bounded by MAX_STEPS)
                                        +--> create_ticket --+
                                        |         back to agent
                                        +--> speak_gate --+--> barge-in? cut short -> retrieve
                                        |                 +--> respond (TTS) -> END
                                        +--> escalate -> END   (out of scope, or needs a human)

Run:  python run.py
"""
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from kb import retrieve
from llm import decide

MAX_STEPS = 3  # cap the tool loop so a confused agent cannot spin forever and burn the call

# Illustrative per-stage latency in milliseconds. These are placeholders to show WHERE the
# budget goes on a voice turn, not benchmarks. The real numbers come from your own traces.
LATENCY_MS = {
    "asr_endpoint": 300,       # streaming ASR plus endpoint detection (deciding the caller stopped)
    "retrieve": 200,           # hybrid retrieval over the help center
    "model_first_token": 350,  # model first token, the start of the spoken reply
    "tool": 400,               # a tool round trip mid-turn (order lookup, ticket)
    "tts_first_audio": 150,    # text-to-speech first audio out
    "barge_in_react": 120,     # stop speaking and re-plan after an interruption
}


class Turn(TypedDict, total=False):
    transcript: str            # what streaming ASR produced for the caller's utterance
    barge_in: Optional[str]    # an utterance that arrives WHILE the agent is speaking
    interrupted: bool          # set True once a barge-in cut a turn short
    context: List
    tool_result: Optional[dict]
    decision: Optional[dict]
    steps: int
    reply: Optional[str]       # the text the agent would speak (stands in for TTS audio)
    escalate: bool
    latency_ms: int
    budget: List               # list of (stage, ms) for the writeup mapping
    trace: List[str]


def _trace(state, msg):
    return state.get("trace", []) + [msg]


def _tick(state, stage):
    """Charge a stage against the latency budget and record it."""
    ms = LATENCY_MS[stage]
    return (state.get("latency_ms", 0) + ms, state.get("budget", []) + [(stage, ms)])


def n_asr(state: Turn):
    """Streaming ASR plus endpointing. The transcript is already in state; here we charge the
    cost of turning streamed audio into a final transcript and detecting the caller stopped."""
    lat, bud = _tick(state, "asr_endpoint")
    return {"latency_ms": lat, "budget": bud, "steps": 0, "tool_result": None,
            "trace": _trace(state, f"asr+endpoint: transcript finalized ({LATENCY_MS['asr_endpoint']} ms)")}


def n_retrieve(state: Turn):
    ctx = retrieve(state["transcript"])
    lat, bud = _tick(state, "retrieve")
    return {"context": ctx, "latency_ms": lat, "budget": bud,
            "trace": _trace(state, f"retrieve: {len(ctx)} doc(s) ({LATENCY_MS['retrieve']} ms)")}


def n_agent(state: Turn):
    d = decide(state["transcript"], state.get("context"), state.get("tool_result"))
    lat, bud = _tick(state, "model_first_token")
    label = d["action"] + (f":{d.get('tool')}" if d.get("tool") else "")
    upd = {"decision": d, "latency_ms": lat, "budget": bud,
           "trace": _trace(state, f"agent -> {label} ({LATENCY_MS['model_first_token']} ms)")}
    if d["action"] == "answer":
        upd["reply"] = d["answer"]
    return upd


def n_order_lookup(state: Turn):
    # A tool call mid-conversation risks dead air, so a real agent speaks a short filler
    # ("let me pull that up") while the call runs. We note that here.
    args = state["decision"].get("args", {})
    res = {"tool": "order_lookup", "order_id": args.get("order_id", "unknown"),
           "status": "in transit", "eta": "2026-07-18"}
    lat, bud = _tick(state, "tool")
    return {"tool_result": res, "steps": state.get("steps", 0) + 1, "latency_ms": lat, "budget": bud,
            "trace": _trace(state, f"tool: order_lookup (filler spoken to avoid dead air, {LATENCY_MS['tool']} ms)")}


def n_create_ticket(state: Turn):
    args = state["decision"].get("args", {})
    res = {"tool": "create_ticket", "ticket_id": "T-10432", "summary": args.get("summary", "")}
    lat, bud = _tick(state, "tool")
    return {"tool_result": res, "steps": state.get("steps", 0) + 1, "latency_ms": lat, "budget": bud,
            "trace": _trace(state, f"tool: create_ticket ({LATENCY_MS['tool']} ms)")}


def n_speak_gate(state: Turn):
    """The moment before the agent speaks. If the caller barged in while we were forming the
    reply, cut the turn short and re-plan against what they just said. Handle a barge-in once,
    so clearing it here keeps the loop bounded."""
    barge = state.get("barge_in")
    if barge:
        lat, bud = _tick(state, "barge_in_react")
        return {"interrupted": True, "transcript": barge, "barge_in": None,
                "context": None, "tool_result": None, "decision": None, "reply": None, "steps": 0,
                "latency_ms": lat, "budget": bud,
                "trace": _trace(state, f"barge-in: caller interrupted, cut short and re-planning ({LATENCY_MS['barge_in_react']} ms)")}
    return {"trace": _trace(state, "speak-gate: no interruption, proceeding to TTS")}


def n_respond(state: Turn):
    """Start streaming the reply as speech (TTS). We charge time-to-first-audio here."""
    lat, bud = _tick(state, "tts_first_audio")
    return {"latency_ms": lat, "budget": bud,
            "trace": _trace(state, f"tts: streaming reply as audio ({LATENCY_MS['tts_first_audio']} ms)")}


def n_escalate(state: Turn):
    return {"reply": "I am connecting you to a human specialist now. Please stay on the line.",
            "escalate": True, "trace": _trace(state, "escalate: warm handoff to a human")}


def route_agent(state: Turn):
    d = state["decision"]
    if d["action"] == "tool" and state.get("steps", 0) < MAX_STEPS:
        return d["tool"]
    if d["action"] == "escalate":
        return "escalate"
    return "speak_gate"


def route_speak_gate(state: Turn):
    return "retrieve" if state.get("interrupted") and state.get("transcript") and state.get("reply") is None \
        else "respond"


def build():
    g = StateGraph(Turn)
    for name, fn in [("asr", n_asr), ("retrieve", n_retrieve), ("agent", n_agent),
                     ("order_lookup", n_order_lookup), ("create_ticket", n_create_ticket),
                     ("speak_gate", n_speak_gate), ("respond", n_respond), ("escalate", n_escalate)]:
        g.add_node(name, fn)
    g.add_edge(START, "asr")
    g.add_edge("asr", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_agent,
                            {"order_lookup": "order_lookup", "create_ticket": "create_ticket",
                             "speak_gate": "speak_gate", "escalate": "escalate"})
    g.add_edge("order_lookup", "agent")
    g.add_edge("create_ticket", "agent")
    g.add_conditional_edges("speak_gate", route_speak_gate, {"retrieve": "retrieve", "respond": "respond"})
    g.add_edge("respond", END)
    g.add_edge("escalate", END)
    return g.compile()


APP = build()


def handle_turn(transcript: str, barge_in: Optional[str] = None) -> Turn:
    """Run one voice turn and return the final state.

    transcript: what the caller said (already transcribed by ASR).
    barge_in:   an optional utterance the caller speaks WHILE the agent is replying, which
                cuts the turn short and re-plans. Pass None for a clean, uninterrupted turn.
    """
    return APP.invoke({"transcript": transcript, "barge_in": barge_in, "trace": []})
