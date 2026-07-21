# Runnable example: the security and compliance screening agent

A small, readable LangGraph implementation of the agent designed in the [case study](../README.md). It implements the core architecture: a layered detector (deterministic rules first, then policy retrieval and a model judgment, with an anomaly signal from the actor baseline), one verdict per event of allow, flag, or block with an explanation, a human-review route for flagged events, and a tamper-evident audit trail over every decision. It is example code rather than a scaled system. The path to real streaming ingestion, an append-only ledger, optimization, and production observability is described in the case study follow-ups rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                                                    # run all scenarios and self-check
python run.py "bob exported the customer table to gmail at 3am"  # screen one event
python run.py "what is the data handling policy?"                # ask a policy question
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 5 events through the graph (a routine event that is allowed, a PII export and a control-tampering event that are blocked by deterministic rules, an off-baseline access that is flagged for human review, and an injection attempt in the event payload that is never obeyed) and asserts the expected verdict for each, that every decision carries an explanation and an audit entry, and that the audit chain verifies. So it doubles as a test. Pass an event in quotes to screen it directly, or a question to look up the policy.

## Where to start reading

Read the files in this order:
1. `compliance_agent.py` the LangGraph state machine: the nodes, the edges, and how an event flows from ingest through the layered detector to a verdict and the audit trail. Start here to see the shape of the whole agent.
2. `policy.py` the knowledge and signals layer: the policy corpus with a keyword retriever and a relevance floor, the deterministic rule engine, and the per-actor behavior baseline that produces the anomaly signal.
3. `llm.py` the judgment layer: a deterministic offline policy, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`. It grades an event against the retrieved policy and treats the event text as untrusted data.
4. `audit.py` the append-only, hash-chained audit trail, and the `verify` that recomputes the chain to detect tampering.
5. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (a layered detector, a verdict with an explanation, a human-review route, and a tamper-evident audit trail) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export COMPLIANCE_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export COMPLIANCE_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export COMPLIANCE_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `COMPLIANCE_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider. The deterministic rules and the injection guardrail run the same way regardless of provider, so the enforcement decision never rests on the model alone.

## Production observability with Arize

This example prints a `trace` for each event. In production you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, rule check, and model call becomes a span, then run online evals (false-positive rate, explanation quality, injection resistance) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
