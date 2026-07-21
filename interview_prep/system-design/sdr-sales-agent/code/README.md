# Runnable example: the SDR sales-development agent

A small, readable LangGraph implementation of the agent designed in the [case study](../README.md). It implements the core architecture: enrich an inbound lead through a tool, qualify it with an explainable score, draft a personalized message grounded only in real signals, run a brand and compliance guardrail over the draft, and route every passing draft to a human for approval. Nothing is ever sent by the agent. It is example code rather than a scaled system. The path to real scale, deliverability, and production observability is described in the case study follow-ups rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                       # run all scenarios and self-check
python run.py "we want a demo"      # run one lead (a demo contact) with your note
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 4 inbound leads through the graph and asserts the expected path for each, so it doubles as a test:

- a strong-fit lead with verified signals: it qualifies, gets a grounded draft, and is queued for a human to review and send.
- a qualified lead whose draft leans on an unverified rumor: the guardrail catches the fabricated claim and blocks it.
- a contact who has opted out: the early suppression gate stops all outreach before any draft is written.
- a personal-mailbox lead with no company signals: it does not qualify, so no outreach is drafted.

Pass a note in quotes to run a single lead directly.

## Where to start reading

Read the files in this order:

1. `sdr_agent.py` the LangGraph state machine: the nodes, the edges, the early suppression gate, the qualify-then-draft branch, and the guardrail-then-human-approval branch. Start here to see the shape of the whole agent.
2. `enrichment.py` the enrichment tool: a small in-memory stand-in for your CRM plus enrichment providers. Every signal carries a source, an as-of date, and a verified flag, which is what lets generation cite only real signals.
3. `qualify.py` the scored, explainable qualification. Each point added carries a reason code that names the signal it came from.
4. `llm.py` the drafter: a deterministic offline drafter, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`. Each personalized sentence is paired with the id of the signal that backs it.
5. `compliance.py` the brand and compliance guardrail: suppression, faithfulness (every claim maps to a verified signal), and the code-checkable CAN-SPAM elements.
6. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (enrichment, an explainable qualification score, grounded drafting, a compliance guardrail, and human approval before send) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export SDR_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export SDR_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export SDR_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `SDR_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline drafter, so the graph and the tests never depend on a provider. The compliance guardrail is deterministic code either way, so a stronger model buys better copy without loosening the safety checks.

## Production observability with Arize

This example prints a `trace` for each run. In production you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (faithfulness, qualification correctness, compliance-element coverage) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
