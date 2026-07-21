# Runnable example: the financial-ops decisioning agent

A small, readable LangGraph implementation of the agent designed in the [case study](../README.md). It implements the core architecture: extract fields from a mock invoice or expense, apply deterministic policy checks, decide approve / deny / route-to-human, and write an immutable audit record for every case, with high-impact and low-confidence cases escalating to a human for sign-off. It is example code rather than a scaled system. The path to real scale, a real rules engine, a WORM audit store, an approval queue, and production observability is described in the case study follow-ups rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                              # run all scenarios and self-check
python run.py "Vendor: Acme | Invoice: INV-9 | Amount: USD 90 | Date: 2026-07-10 | Category: software"
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 6 scenarios through the graph (a clean invoice that auto-approves, a non-reimbursable category that is denied, a high-impact amount routed to a human, an over-limit amount routed to a human, a thin low-confidence document routed to a human, and a duplicate invoice id routed to a human), asserts the expected decision for each, and verifies the audit hash chain, so it doubles as a test. Pass a document in quotes to decide one case directly. Documents are labelled text (`Vendor: ... | Invoice: INV-... | Amount: USD ... | Date: YYYY-MM-DD | Category: ...`); a real model reads freeform invoices and expenses.

## Where to start reading

Read the files in this order:
1. `decision_agent.py` the LangGraph state machine: the nodes, the edges, and the approve / deny / human-review branch. Start here to see the shape of the whole workflow.
2. `llm.py` the extraction layer: a deterministic offline parser, plus a provider-agnostic real path that turns a document into typed fields through any model via LangChain's `init_chat_model`. It also computes the extraction confidence that routes thin documents to a human.
3. `policy.py` the deterministic rulebook: completeness and confidence, non-reimbursable categories, duplicates, currency, and amount thresholds, resolved into approve / deny / route-to-human.
4. `audit.py` the append-only, hash-chained audit trail (and the processed-id set that catches duplicates), with `verify_chain` to prove the log was not edited.
5. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the workflow together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (extraction, a deterministic policy step, an approve / deny / route branch, a human sign-off gate, and an immutable audit record) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model` for the extraction step, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export FINOPS_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export FINOPS_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export FINOPS_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `FINOPS_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline parser, so the graph and the tests never depend on a provider. The policy decision stays deterministic either way, because the rules are knowable and you want them exact and auditable.

## Production observability with Arize

This example prints a `trace` for each run. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit: instrument the LangGraph app with OpenInference so every node, extraction, and decision becomes a span, then run online evals (extraction-field accuracy, false-approve and false-deny rates, escalation correctness) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
