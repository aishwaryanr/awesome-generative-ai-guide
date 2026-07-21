# Runnable example: the real-time voice support agent

A small, readable LangGraph implementation of the control flow designed in the [case study](../README.md). Audio is out of scope for a tiny runnable demo, so this models the **conversation control flow of one voice turn as text**: the caller's speech arrives already transcribed, the agent retrieves or calls a tool and forms a reply, and the reply text stands in for the audio a real system would stream back. Two things make it a voice turn rather than a chat turn: a **per-node latency budget** (every stage charges milliseconds, because on a phone call the wall clock is the product) and **barge-in** (the caller can interrupt mid-reply, which cuts the turn short and re-plans). A wrong or high-impact request hands off to a human. It is example code rather than a scaled system. Streaming ASR, streaming TTS, telephony, and production observability are described in the case study rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                              # run all scenarios and self-check
python run.py "Where is my order 5012?"    # run one voice turn from the command line
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 6 scenarios through the graph (a policy question answered from the knowledge base, an order lookup via a tool mid-turn, a ticket creation, a refund that hands off to a human, an out-of-scope question that hands off, and a **barge-in** where the caller interrupts the order answer to cancel, so the turn is cut short and re-planned into a handoff) and asserts the expected path and the latency budget for each, so it doubles as a test. Pass an utterance in quotes to run a single turn.

## Where to start reading

Read the files in this order:
1. `turn_graph.py` the LangGraph state machine: the nodes, the edges, the per-node latency budget, and the barge-in re-plan. Start here to see the shape of one voice turn.
2. `kb.py` a tiny in-memory knowledge base and keyword retriever with a relevance floor (it stands in for a vector store; the floor is what makes out-of-scope calls hand off).
3. `llm.py` the decision layer: a deterministic offline policy, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
4. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the turn together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same control flow (streaming transcript in, retrieve or tool, a bounded loop, a barge-in re-plan, a latency budget, and a human handoff) in the SDK you use, for example the OpenAI Agents SDK and its voice pipeline, the Anthropic SDK, LiveKit Agents, Pipecat, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Wrap it with real audio

This demo starts from a transcript and ends at reply text on purpose, so the control-flow decisions stay easy to read and test. A real deployment adds the audio layer around it in one of two shapes:

- **A pipeline:** streaming ASR (speech to text) in front, streaming TTS (text to speech) behind, with a voice-activity detector and an endpointing or turn-detection model deciding when the caller stopped. This graph is the middle of that pipeline.
- **A single speech-to-speech model** (audio in, audio out) that folds ASR, the model, and TTS into one call for lower latency. The decision nodes here become the model's tool-calling and handoff behavior.

Either way, barge-in is watched continuously during playback by a duplex audio path, rather than only at the speak boundary as modeled here. The case study covers both shapes and the latency budget in detail.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export VOICE_AGENT_MODEL="gpt-realtime" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export VOICE_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export VOICE_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `VOICE_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider.

## Production observability with Arize

This example prints a `trace` and a `latency` budget for each turn. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (grounding, refusal correctness, tool-call correctness) and latency monitors over live traffic and alert on drift. On a voice agent the latency spans are first-class, because time-to-first-audio is the number the caller feels. See the production and ops sections of the [case study](../README.md) for how this fits the design.
