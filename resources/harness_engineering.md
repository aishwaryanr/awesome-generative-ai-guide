# Harness Engineering 101

How to build the layer that turns a model into an agent.

A large language model, on its own, does very little. You give it text and it returns text. It keeps no memory of an earlier request, has no way to act on the world, and cannot run itself a second time. The systems built on these models do all of those things: a coding agent reads your files, edits them, runs the tests, and tries again when they fail; a research agent searches, reads, and returns a cited report. The model at the center is the same limited function. The difference is the machinery built around it.

That machinery is the harness. This guide teaches the concepts and the judgment of building it. It is theory first, because in the AI era you rarely hand-roll a harness from bare metal. You take a framework and engineer its layers for your domain.

> This is the free 101 guide. The full [Harness Engineering 101 handbook](https://levelup-labs.ai/) and the LevelUp Labs Maven course go deeper, with a complete worked implementation.

---

## 1. What is a harness

A large language model is a function. Text goes in, text comes out. It has no memory of previous calls, no ability to act on the world, and no way to run itself more than once. Each call is independent.

```
        text in  ------->  [ MODEL ]  ------->  text out
   no memory   .   no loop   .   no tools   .   no hands
```

The harness is everything you wrap around that function to turn it into an agent. Concretely, it is 5 kinds of machinery:

- **The loop.** Code that calls the model, does something with the answer, and calls it again, until the job is done.
- **Context assembly.** What information you place in front of the model on each step: instructions, history, retrieved documents, tool results.
- **Tools.** The hands. Functions the model can invoke to search, read, write, run code, or call an API, plus the code that executes them.
- **Memory.** What the system keeps across steps and sessions, so the agent survives longer than a single context window.
- **Gates.** The checks that decide what the agent may do, and that verify its work before it counts as done.

```
   +----------------- THE HARNESS -------------------+
   |   loop    context    tools    memory    gates   |
   |          +---------------------------+          |
   |          |         THE MODEL         |          |
   |          |     text in -> text out   |          |
   |          +---------------------------+          |
   +-------------------------------------------------+
              agent  =  model  +  harness
```

The sentence to carry through everything: **agent = model + harness. If you are not the model, you are the harness.** The model you can buy. The harness you build. That is where almost all of the interesting, hard, and valuable work in building agents happens.

### The maturity ladder

Harness engineering is the third rung on a ladder the field has climbed for years.

```
   rung 3   HARNESS ENGINEERING   everything wrapping the model
   rung 2   CONTEXT ENGINEERING   which tokens go in the window
   rung 1   PROMPT ENGINEERING    the exact words you send
```

Prompt engineering was the craft of the words. Context engineering widened the lens to the whole window: the instruction, the examples, the retrieved documents, the history, and the order they arrive in. Harness engineering widens it once more, to everything around the window: the loop, the tools, the memory, and the gates. Each rung contains the ones below it.

*Go deeper:* LangChain, [The Anatomy of an Agent Harness](https://blog.langchain.com/the-anatomy-of-an-agent-harness/). Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents).

### The harness is a layer you build into every agent

When beginners hear "build your own harness," they hear "build your own Cursor." That is not what it means. The harness is a layer inside an agent: the runtime part that is not your task-specific prompt and logic. It is the piece you could lift out and reuse for a different task. Cursor and Claude Code are that layer built so generally it works for almost any coding task, and then sold. They are productized harnesses. Your harness is the same layer, engineered for your one domain. "Build your own harness" means engineer that layer deliberately instead of accepting whatever the framework gives you by default.

### The thread that runs through everything: survival

On a short task, the harness barely shows. Ask an agent one quick question with one tool call and almost nothing can go wrong. The trouble starts when the agent runs long and acts on its own. Over many steps its context fills with old tool results and stale reasoning, and its answers drift. The field calls this **context rot**. Every hard part of the harness is, at bottom, about surviving length and autonomy. Context rot is the disease. Memory, compaction, and clearing are the treatment. The loop is what runs long enough to need them.

### The five-part stack

```
   OBSERVABILITY   see what the agent did
   GATES           verify + decide what is allowed
   TOOLS           the hands: act on the world
   CONTEXT+MEMORY  what the model sees and keeps
   THE LOOP        run the model, again and again
   THE MODEL       text in -> text out
```

You do not build all 5 layers every time. How many you build, and how heavy each is, depends on how much autonomy you grant the agent.

---

## 2. How the discipline arrived

Harness engineering became a named practice in 2026, when models got good enough that the bottleneck moved from the model to everything around it. The posture shifted with it: humans moved from *in* the loop, inspecting every output, to *on* the loop, designing the environment the agent runs in. The thesis line the field settled on: the model is a commodity, the harness is the moat.

*Go deeper:* Mitchell Hashimoto, [My AI Adoption Journey](https://mitchellh.com/writing/my-ai-adoption-journey) (the origin of "engineering the harness"). OpenAI's harness-engineering post (Ryan Lopopolo) and Martin Fowler's "Harness Engineering" give the practice its scale story and software-engineering standing.

---

## 3. How much harness do you need

This is the decision that governs everything else, so make it before you build.

**The Agency-Control tradeoff, from the build side:** the more autonomy you grant, the more harness you must build. A deterministic workflow barely needs one, because the code path is the control. A fully autonomous agent needs a heavy one.

```
   harness
   weight  |                               * autonomous agent
           |                        *
           |                *
           |        *
           |  * deterministic workflow
           +---------------------------------------> autonomy granted
```

**The two halves of a harness.** Feed-forward is what you give the agent up front: principles, conventions, docs, skills. Feedback is what runs after it acts: static analysis, tests, running the app, review agents. Each half can be deterministic (linters, structural tests, cheap and exact) or inference-based (a review agent, flexible and costly).

**Size the build with a risk read.** Risk = probability × impact × detectability. A mistake that is likely, damaging, and hard to notice earns a heavy gate. A mistake that is rare, cheap, and obvious does not. You have to be this tall to reduce supervision.

**The warrant test.** The question is never "does the off-the-shelf harness fit." It is "does my domain make a normally-invisible component load-bearing enough that I must engineer that layer." That promotion is the reason to build. When no component is promoted, accept the framework defaults and engineer no extra layer at all.

*Go deeper:* Birgitta Böckeler (Thoughtworks), [The Engineering of AI Agents](https://www.youtube.com/watch?v=_R83pFpUWyM), for feed-forward/feedback, the CPU-vs-inference split, and the risk read.

---

## 4. The core components

### The agent loop

The loop is the backbone everything hangs off. It is a call to the model, do something with the answer, call it again, until a stop condition.

```
   plan  ->  act (tool call)  ->  observe (result back into context)  ->  plan ...
```

A framework runs this loop for you. What you own is the stop conditions and the step budgets that keep a runaway loop from spinning forever. Autonomy lives here: the loop is where the agent decides its own next action.

*Go deeper:* Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), on workflows versus agents (an agent is a model using tools in a loop).

### Context engineering

Everything the agent cannot see is invisible to it, so what you put in the window each step decides the quality of every step. The window holds the system prompt, the tools, the history, retrieved documents, memory, and the response forming.

Two failure modes appear at length. **Context rot:** coherence degrades as the window fills with stale content. **Context anxiety:** the model rushes as it nears the limit. A bigger window is not a free win, because relevant tokens still get lost in the middle of a long one.

There are 3 ways to shrink a context, and it matters which you use:

- **Clear:** prune finished tool results. Safe, lossless.
- **Compact:** summarize the history. Lossy, so compaction does not equal coherence.
- **Offload:** move state out of the window into a file or store, and pull it back when needed.

*Go deeper:* Anthropic on effective context engineering, and the Latent Space episode [Extreme Harness Engineering](https://www.latent.space/p/harness-eng) on treating context as code.

### Memory and long-running state

Memory is how an agent survives past a single context window. Think of each new session as an engineer arriving for a shift with no memory of the last one. The harness is the handoff notes: progress files, feature checklists, and durable state written outside the window. Models respect structured state (JSON breadcrumbs) more than loose prose.

Forgetting is a feature. A memory store that only grows gets slow and confused, so design what to drop, not only what to keep.

*Go deeper:* Anthropic, [Build Agents That Run for Hours](https://www.youtube.com/watch?v=mR-WAvEPRwE), on file-system state and multi-session coherence.

### Tool design

A tool is a schema plus an executor plus a returned result. The agent reads every token you return, so tool design is token design.

- Prefer search-first tools over list-all tools.
- Name tools with a domain prefix so the agent picks the right one.
- Shape responses: return concise results by default, detail on request.
- Wrap verbose commands to surface only what matters, usually the failures.
- Tell the agent when not to call a tool. More tools is not a better agent.

There is a real tension between CLIs and MCP. MCP standardizes integration but forces every tool's schema into context. Token-efficient CLI wrappers are lighter but bespoke. It is a tradeoff to weigh per domain, not a settled verdict.

*Go deeper:* Anthropic, [Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents).

### Skills and MCP

A **skill** bundles instructions plus code plus assets into a reusable capability the agent loads on demand, through progressive disclosure: it sees the skill exists, and loads the detail only when it needs it. **MCP**, the Model Context Protocol, is the standard way to connect an agent to tools and data, the common interface so you wire a capability once and reuse it. A capability can be a tool, a skill, or an MCP server; choose by how reusable and how heavy it is.

*Go deeper:* Anthropic, [Equipping Agents with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills).

---

## 5. Keeping it alive

### Verification and eval loops

Guardrails block in the moment; evals measure over time. You need both. Self-evaluation is a trap, because it is easier to critique than to generate. The **generator / evaluator / planner** pattern splits those roles into separate context windows, so a harsh standalone evaluator grades the builder's work.

```
   planner  --->  generator  --->  evaluator  --->  (pass? ship : back to generator)
                       ^_________________________________|
```

The generator and evaluator agree on what "done" means as checkable criteria before work starts, because vague criteria give vague critiques. You can grade subjective quality if you write your opinion down as a weighted rubric and calibrate it with reference examples. Inside the loop, the agent runs tests, type-checks, and end-to-end checks on its own work before declaring done. The honest gap: verifying functionality is still largely unsolved.

The improvement flywheel: run, observe the failures, fix the environment, re-run. Fix the harness, not the diff. Pair every "do not do X" rule with an eval, so you can delete the rule when a new model makes it obsolete.

*Go deeper:* Anthropic, [Build Agents That Run for Hours](https://www.youtube.com/watch?v=mR-WAvEPRwE), for the generator/evaluator/planner pattern, negotiated contracts, and rubrics.

### Guardrails, permissions, and sandboxing

Gates decide what the agent may do. There are input guardrails, output guardrails, and action guardrails. The live threat is prompt injection, and the sharpest way to see the risk is the lethal trifecta: untrusted content, access to private data, and the ability to communicate externally. Any 2 are manageable; all 3 together is dangerous.

```
   untrusted content  +  private data access  +  external comms  =  high risk
```

Prefer allowlists over blocklists, and a review-before-act mode over blanket auto-approval. Encode good behavior durably: put the rule in a check whose error message teaches the agent the right move. Sandbox execution and think in blast radius, even locally.

### Sub-agents and orchestration

Split work across agents for context isolation, specialization, or parallelism. The cost is coordination overhead, compounding failure, and harder debugging. Run 1 agent first, and add a second only with evidence. One sharp design call: do not hand a critic sub-agent the generator's traces. Judge the output, and let the generator reflect on its own work.

### Observability and agent-legible software

You cannot fix what you cannot see. Traces capture every prompt, tool call, and result. Reading traces is the debugging loop: find where the agent's judgment diverged from yours, then tune the prompt or the environment. This is increasingly the highest-leverage layer. Alongside it, write for the agent as a reader: clear errors, self-describing tools, an AGENTS.md, and non-text rendered as text so the agent can perceive it.

*Go deeper:* Latent Space, [Extreme Harness Engineering](https://www.latent.space/p/harness-eng), on observability-first, agent-legible software.

---

## 6. When to build your own

Apply the warrant. Walk your domain and ask which single component a constraint promotes into the load-bearing wall:

- Repo migration promotes context and verification.
- Regulated support promotes memory, permission gating, and audit.
- Deep research promotes sub-agents and per-claim verification.
- Large-dataset jobs promote durable state, retries, and cost governance.

Build the harness on-policy, native to the model's real output, so it stays in-distribution and does not fight future model progress. Harnesses are living. Adapt and delete pieces as the model improves. A harness that ages out was not wrong; it was right for the old model, and the frontier moved.

---

## 7. Building it in practice

In the AI era you do not build the harness from bare metal. You take a harness framework and engineer its layers for your domain, which is what "build your own harness" means here. Every concept in this guide maps to a framework feature: the loop is the compiled graph, compaction is summarization middleware, memory is the store, tools are typed functions, skills are skills, gates are permissions and interrupts, sub-agents are the sub-agent roster, observability is tracing. You decide which one or two layers your domain forces you to engineer, and accept sensible defaults for the rest.

Frameworks and harnesses worth reading: **Deep Agents** (LangChain/LangGraph) and the **Claude Agent SDK** as productized harnesses you engineer, and **[TAU](https://github.com/huggingface/tau)** (Hugging Face) as a small, readable teaching harness whose architecture already maps to the 5 layers above.

---

## Where to go next

- Follow the [Harness Engineering path](../paths/harness-engineering.md) for a curated sequence through this material.
- Study the components hands-on in the [Agentic AI Crash Course](../free_courses/agentic_ai_crash_course/README.md), and evaluation in [AI Evals for Everyone](../free_courses/ai_evals_for_everyone/README.md).
- The living [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) collection for per-topic sourcing.
- The full **Harness Engineering 101 handbook** and the **LevelUp Labs Maven course** at [levelup-labs.ai](https://levelup-labs.ai/) for the complete treatment, including a full worked implementation.
