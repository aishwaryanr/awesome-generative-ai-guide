# Runnable example: the customer support agent

A small, readable LangGraph implementation of the agent designed in the [case study](../README.md). It implements the core architecture: retrieval to ground answers, a bounded agent loop that can call tools, a guardrail that refuses or escalates when an answer is not grounded, and a human handoff. It is example code, not a scaled system. The path to real scale, optimization, and production observability is described in the case study follow-ups, not coded here.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                              # run all scenarios and self-check
python run.py "Where is my order 5012?"    # ask the agent your own question
```

It runs fully offline, so no API key is needed. With no argument, `run.py` sends 5 scenarios through the graph (an answered policy question, an order lookup via a tool, a ticket creation, a refund that escalates for human approval, and an out-of-scope question that escalates) and asserts the expected path for each, so it doubles as a test. Pass a question in quotes to ask the agent directly.

## Where to start reading

Read the files in this order:
1. `support_agent.py` the LangGraph state machine: the nodes, the edges, and the bounded agent loop. Start here to see the shape of the whole agent.
2. `kb.py` a tiny in-memory knowledge base and keyword retriever with a relevance floor (it stands in for a vector store; the floor is what makes out-of-scope questions escalate).
3. `llm.py` the decision layer: a deterministic offline policy, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
4. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (retrieval, a bounded tool loop, a guardrail, and escalation) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export SUPPORT_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export SUPPORT_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export SUPPORT_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `SUPPORT_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider.

## Production observability with Arize

This example prints a `trace` for each run. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (faithfulness, refusal correctness, tool-call correctness) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
