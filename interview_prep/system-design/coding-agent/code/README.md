# Runnable example: the coding agent

A small, readable LangGraph implementation of the harness designed in the [case study](../README.md). It is the loop and scaffolding that turn a model into an agent that can edit a repository and prove its work: draft a short plan, read a file, run the test suite, propose a targeted edit, run the tests again, and repeat until the tests are green or a budget is hit. It shows the load-bearing parts in miniature: a plan/todo step, a bounded loop, a targeted edit tool (a scoped change to one function in place of a full-file rewrite, the idea behind Codex's `apply_patch` and a string-replace editor), the test suite as the verifier, a compaction step that keeps the running history small, and a guardrail that escalates to a human when fixes are exhausted or the budget runs out. It is example code rather than a scaled system. The path to a real repository, a real sandbox, and production observability is described in the case study rather than coded here.

For a fuller reference implementation to read alongside this one, study Hugging Face's [tau](https://github.com/huggingface/tau), a small readable terminal coding agent whose three layers map almost 1:1 onto these files: `tau_ai` (its provider abstraction) onto `llm.py`, `tau_agent` (its loop and harness) onto `agent.py`, and `tau_coding` (its file and shell tools and sessions) onto `sandbox.py` and `run.py`.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python run.py                 # run all scenarios and self-check
python run.py "fix add"       # run one task against the sandbox repo
```

It runs fully offline, so no API key is needed, and it ships with small built-in sample data so it works out of the box. With no argument, `run.py` sends 2 scenarios through the harness (a fixable bug that reaches a passing test suite and opens a PR, and an impossible request that the agent cannot satisfy and escalates to a human) and asserts the expected path for each, so it doubles as a test. Pass a task in quotes to run it directly.

## Where to start reading

Read the files in this order:
1. `agent.py` the LangGraph state machine: the plan step, the harness loop, the nodes, the edges, the bounded step cap, the compaction node, and the escalation guardrail. Start here to see the shape of the whole agent.
2. `sandbox.py` the tiny in-repo sandbox: the repository (one small file with a bug), the typed tools (read, targeted edit, run-tests, open-PR), and the test runner that acts as the verifier.
3. `llm.py` the model layer: a deterministic offline policy for the one creative decision (what edit to try next), plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
4. `run.py` the scenarios, the assertions, and the command-line entry point.

The division of labor is the point: the harness owns the control flow (when to read, test, edit, ship, or escalate), and the model owns one decision inside it (the next edit). That is the split a real coding agent makes.

## Use a different framework

LangGraph is one choice for wiring the harness together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same harness (a bounded read-edit-test loop, a verifier, compaction, and escalation) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export CODING_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export CODING_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export CODING_AGENT_MODEL="gemini-2.0-flash" GOOGLE_API_KEY="..."

python run.py
```

If `CODING_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the harness and the tests never depend on a provider.

## Production observability with Arize

This example prints a `trace` for each run. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **Arize** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (patch validity, test pass rate, edits per task, escalation correctness) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
