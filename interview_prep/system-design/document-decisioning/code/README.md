# Runnable example: the document decisioning agent

A small, readable LangGraph implementation of the underwriting agent designed in the [case study](../README.md). It implements the core architecture: extract fields from a (mock) submission, verify the extraction against a confidence floor, apply a deterministic underwriting policy, let the model make the borderline judgment, route low-confidence or high-impact cases to a human, and write an immutable audit record for every decision. It is example code rather than a scaled system. The path to a real layout-aware parsing service, WORM audit storage, and production observability is described in the case study rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                        # run all scenarios and self-check
python run.py "applicant_name: Jane Okafor; property_value: 420000; requested_coverage: 300000; year_built: 1998; prior_claims: 0; construction: masonry; flood_zone: no"
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 7 submissions through the graph (a clean approval, two hard declines, a high-impact referral for human sign-off, a low-confidence scan, a missing required field, and a borderline judgment) and asserts the expected decision for each, so it doubles as a test. It also checks that no raw PII reaches the audit trail and that the audit hash chain is tamper-evident. Pass a document in quotes (a `key: value; key: value` string) to decide one submission directly.

## Where to start reading

Read the files in this order:
1. `underwriting_agent.py` the LangGraph state machine: the nodes, the edges, and where each path routes to a human. Start here to see the shape of the whole agent.
2. `documents.py` the mock submissions and the field extractor. It stands in for a layout-aware parser plus an extraction model, and it returns typed fields with a per-field confidence and PII already masked.
3. `policy.py` the deterministic underwriting policy: the hard rules and the delegated-authority limit, with the borderline judgment left for the model.
4. `llm.py` the decision layer: a deterministic offline policy, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
5. `audit.py` the append-only, hash-chained audit trail, with a chain verifier.
6. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (extraction, a verification gate, a deterministic policy, a bounded model judgment, human referral, and an audit trail) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export UW_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export UW_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export UW_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `UW_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider. The model only makes the borderline approve-or-refer judgment: hard declines and high-impact referrals are decided by the deterministic policy before the model is ever called, so the worst a wrong model call does is send a clean file to a human.

## Production observability with Arize

This example prints a `trace` for each run. In production you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (extraction field accuracy, false-approve and false-decline rates, referral correctness) over live traffic and alert on drift. See the production and ops section and the follow-ups in the [case study](../README.md) for how this fits the design.
