# Runnable example: the enterprise research assistant

A small, readable LangGraph implementation of the assistant designed in the [case study](../README.md). It implements the load-bearing parts of the architecture: permission-scoped retrieval (a user's identity is applied at retrieval time, so a source they cannot access never enters the context), a bounded multi-hop research loop (the agent runs its own follow-up searches), answers that carry citations, a guardrail that escalates when an answer is not grounded or when a source tries to inject instructions, and a human gate on any high-impact action. It is example code rather than a scaled system. The path to real scale (a vector store, hybrid dense-plus-sparse retrieval, a reranker, connectors, freshness, and production observability) is described in the case study follow-ups rather than coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                                  # run all scenarios and self-check
python run.py "What is our PTO policy?"        # ask the agent your own question (as a regular employee)
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 7 scenarios through the graph (a single-source cited answer, a multi-hop answer stitched from two searches, a permission block where a regular employee is refused an HR-only source, the same question answered for an HR user who is allowed to read it, a high-impact action routed to a human before anything is posted, a retrieved source that tries to inject instructions and is escalated, and an out-of-scope question that escalates) and asserts the expected path for each, so it doubles as a test. Pass a question in quotes to ask the agent directly.

## Where to start reading

Read the files in this order:
1. `research_agent.py` the LangGraph state machine: the nodes, the edges, the bounded multi-hop research loop, the citation guardrail, and the human gate. Start here to see the shape of the whole agent.
2. `kb.py` a tiny multi-source corpus where every document carries a source system and an access-control list, plus a permission-scoped keyword retriever with a relevance floor. It stands in for connectors and a vector store; the permission filter runs before scoring, which is what stops a user from seeing a source they cannot access.
3. `llm.py` the planning and decision layer: a deterministic offline policy that breaks a question into searches and picks the next step, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
4. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (permission-scoped retrieval, a bounded multi-hop loop, cited answers, a grounding guardrail, and a human gate) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export RESEARCH_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export RESEARCH_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export RESEARCH_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `RESEARCH_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider.

## Production observability with Arize

This example prints a `trace` for each run. In production you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, search, and model call becomes a span, then run online evals (citation faithfulness, permission-leak checks, retrieval hit rate) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
