# Runnable example: the analytics copilot (text to SQL)

A small, readable LangGraph implementation of the copilot designed in the [case study](../README.md). It implements the core architecture: schema linking to pick the right tables, read-only SQL generation, a guardrail that blocks anything that is not a single SELECT, execution in a read-only sandbox, execution-guided self-correction when a query errors, and abstention when the question is out of scope or returns no rows. It is deliberately small example code. The path to real scale, optimization, and production observability lives in the case study follow-ups.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                                 # run all scenarios and self-check
python run.py "What is our total revenue?"    # ask the copilot your own question
```

It runs fully offline, so no API key is needed, and the database is an in-memory SQLite built at import time (`sqlite3` is part of the standard library). With no argument, `run.py` sends 6 scenarios through the graph (an aggregation, a count, a multi-join that errors and self-corrects, a filtered count, a destructive query that the guardrail blocks, and an out-of-scope question that abstains) and asserts the expected path for each, so it doubles as a test. Pass a question in quotes to ask the copilot directly.

## Where to start reading

Read the files in this order:
1. `sql_agent.py` the LangGraph state machine: schema link, generate, guardrail, execute, self-correct, answer, refuse. Start here to see the shape of the whole copilot.
2. `db.py` the tiny in-memory warehouse: a small star schema, seed rows, and the read-only executor. The executor pins `PRAGMA query_only = ON`, so it is a genuine read-only sandbox.
3. `guardrails.py` the static read-only check that runs before execution: allow one SELECT, refuse everything else.
4. `llm.py` the model layer: schema linking, SQL generation with execution-guided self-correction, and answer synthesis, each with a deterministic offline version and a provider-agnostic real path through LangChain's `init_chat_model`.
5. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the copilot together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (schema linking, read-only SQL generation, a guardrail, a read-only executor, and a self-correction loop) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export SQL_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export SQL_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export SQL_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `SQL_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the copilot runs on the deterministic offline logic, so the graph and the tests never depend on a provider. The guardrail and the read-only sandbox apply to model-written SQL exactly as they do to the offline SQL, which is the point: the model proposes a query, and the system decides whether it is safe to run.

## Production observability with Arize

This example prints a `trace` for each run. In production you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (execution success, guardrail-block rate, self-correction rate, abstention rate) over live traffic and alert on drift. See the production and evaluation sections in the [case study](../README.md) for how this fits the design.
