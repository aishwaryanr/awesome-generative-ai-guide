# Runnable example: the clinical scribe

A small, readable LangGraph implementation of the scribe designed in the [case study](../README.md). It implements the core pipeline: ingest a synthetic encounter transcript, extract a structured SOAP note (Subjective, Objective, Assessment, Plan), run a faithfulness check that flags any statement the transcript does not support, and hold the note as a draft that a clinician must review and sign before it is final. It is example code rather than a scaled system. The path to real scale, terminology grounding, EHR write-back, and production observability is described in the case study follow-ups rather than coded here.

All data in this example is synthetic. There is no real protected health information anywhere, and there must never be. Do not point this at a real transcript that contains patient data.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                                   # run all scenarios and self-check
python run.py "Patient: I have a sore throat."  # run one transcript through the pipeline
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 3 scenarios through the graph (a clean encounter that produces a reviewable draft, a note carrying a fabricated diagnosis and prescription that the faithfulness check catches before sign-off, and a short encounter that still requires review) and asserts the expected path for each, so it doubles as a test. Pass a transcript in quotes to run the pipeline on your own synthetic input.

## Where to start reading

Read the files in this order:
1. `scribe.py` the LangGraph state machine: ingest, extract, faithfulness, review gate, and the `sign_note` sign-off step. Start here to see the shape of the whole pipeline and why the graph never marks a note final on its own.
2. `llm.py` the two clinical operations: `extract_soap` turns a transcript into a SOAP note, and `check_faithfulness` flags any statement the transcript does not support. Each has a deterministic offline policy and a provider-agnostic real path through LangChain's `init_chat_model`.
3. `run.py` the scenarios, the assertions, and the command-line entry point, including the synthetic transcript and the fabricated draft that the faithfulness check catches.

## Use a different framework

LangGraph is one choice for wiring the pipeline together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same pipeline (transcript ingest, SOAP extraction, a faithfulness check, and a clinician sign-off gate) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export CLINICAL_SCRIBE_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export CLINICAL_SCRIBE_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export CLINICAL_SCRIBE_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `CLINICAL_SCRIBE_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the scribe runs on the deterministic offline policy, so the graph and the tests never depend on a provider. When a real model is connected, extraction becomes a summarizing model that can paraphrase, and the faithfulness check becomes an LLM judge that reads each statement against the transcript. That is exactly the setup the faithfulness check is built for: a model that can invent, checked by a step that only trusts the source.

## Production observability with Arize

This example prints a `trace` for each run. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, extraction, and faithfulness check becomes a span, then run online evals (faithfulness on a sample, omission rate, terminology-mapping accuracy) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
