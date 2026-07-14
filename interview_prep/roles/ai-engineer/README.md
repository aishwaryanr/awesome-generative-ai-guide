# AI Engineer: Interview Prep

Deep, self-contained prep for the **AI Engineer** role: the person who builds LLM-powered products. This folder is designed to be the only resource you need. Work through it in order, or jump to the round you are about to face.

**Files in this folder:**

1. [README.md](README.md): what the role is, who hires for it, and the full interview map (this file).
2. [rounds.md](rounds.md): every round in the loop, what each tests, what "good" looks like, and example prompts.
3. [questions.md](questions.md): a 60+ question bank with model answers, grouped by theme. The heart of this folder.
4. [resources.md](resources.md): the best free reading, talks, papers, and repos, verified and grouped by topic.
5. [courses.md](courses.md): the best free courses, starting with this repository's own.
6. [prep-plan.md](prep-plan.md): a day-by-day plan (3-week track plus a 1-week crunch).

---

## What the job actually is

An AI Engineer takes a foundation model that someone else trained and turns it into a product feature that works, ships, and holds up in production. You rarely train models from scratch. Instead you spend your time on the layer above the model:

- **Prompting and context engineering:** deciding what goes into the context window (instructions, retrieved documents, tool results, memory, few-shot examples) and what stays out. In 2025-2026 this is treated as the primary lever, often ahead of clever prompt wording.
- **Retrieval (RAG):** grounding the model in your data with chunking, embeddings, vector and hybrid search, reranking, and increasingly agentic retrieval.
- **Agents and tool use:** giving the model tools, memory, and a loop so it can take actions and iterate toward a goal, including function calling, the Model Context Protocol (MCP), and multi-agent orchestration.
- **Evaluation:** building labeled eval sets, LLM-as-judge rubrics, retrieval metrics, and production monitoring so quality is measured, not asserted.
- **Deployment:** cost, latency, caching, streaming, guardrails, observability, and safe rollout behind fallbacks.

The defining shift versus a classic ML role: whiteboard algorithm puzzles (reverse a linked list, implement BFS) are largely gone from AI-specific loops. They are replaced by real engineering problems, how to chunk documents, how to evaluate an LLM's output, how to stop an agent from looping forever, how to defend against prompt injection. Roughly 75% of a modern AI Engineer technical loop is RAG, agents, evals, and LLM-powered system design; classic ML (trees, CNNs, gradient descent) is maybe 25% and often skipped entirely at product-focused startups.

## Who hires for it, and at what levels

Three buyer types, with different emphases:

- **AI-native startups** (the product *is* an agent or an LLM app): move fast, ship, own a feature end to end. They weight hands-on building and a strong take-home or portfolio heavily. Titles: AI Engineer, Founding AI Engineer, Member of Technical Staff.
- **Big-tech AI teams** (applied AI / applied ML orgs inside large companies): more structured loops, a real coding round, deeper system design, and a bar on fundamentals. Titles: AI Engineer, Applied AI Engineer, ML Engineer (LLM), Software Engineer, ML.
- **Enterprises** (adopting AI into existing products and workflows): weight retrieval-at-scale, access control, security, compliance, and integration with legacy systems. Titles: AI Engineer, GenAI Engineer, LLM Engineer, AI Solutions Engineer.

Leveling roughly tracks scope:

- **Junior / L3:** implement well-specified components (a retriever, a prompt, an eval script) with guidance.
- **Mid / L4:** own a feature end to end, make and defend architecture tradeoffs, build the eval that gates it.
- **Senior / L5+:** design systems across teams, set the eval and safety bar, mentor, and reason about cost and reliability at scale. Senior loops add a deeper system-design round and sharper "how does this break" probing.

## How this role differs from adjacent roles

- **vs. ML / Fine-tuning Engineer:** they train and post-train models (LoRA, SFT, DPO, RLHF, GRPO, quantization). You mostly consume models and engineer the system around them. Overlap exists, but if the loop centers on training-loop internals, that is the ML role. See the [interview prep hub](../../README.md) for the other role tracks.
- **vs. Applied Scientist / Research Engineer:** they invent methods and design evals for brand-new capabilities, with more research depth. You optimize for shipping a reliable product. AI Engineer loops are lighter on paper-reproduction, heavier on production judgment.
- **vs. AI Product Manager:** they decide what to build and why; you decide how and build it. You still need the PM's cost-quality-latency intuition to defend tradeoffs.
- **vs. Solutions Architect / Enterprise AI:** heavy overlap at enterprises. The architect leans toward integration, access control, and rollout across an org; the AI Engineer leans toward writing the code that runs it. Enterprise AI Engineer loops pull in the architect's concerns.
- **vs. classic Software Engineer:** you own the non-deterministic layer. The hardest part is not the code, it is making a probabilistic system behave reliably and proving it does.

## The end-to-end interview loop

A typical full loop, in order. Not every company runs every round; startups compress, big-tech expands. Full detail per round is in [rounds.md](rounds.md).

| # | Round | Length | What it centers on |
|---|-------|--------|--------------------|
| 1 | Recruiter screen | 30 min | Fit, motivation, level, comp, project story |
| 2 | Technical / coding | 45-60 min | Python plus practical LLM coding (call an API, parse structured output, write a small RAG or tool loop) |
| 3 | Take-home (sometimes, replaces or adds to #2) | 2-7 days | Build a small working RAG app or agent with evals, then defend it |
| 4 | LLM / ML system design | 60 min | Design a RAG or agentic system; reason about evals, cost, latency, failure modes |
| 5 | Behavioral | 45-60 min | Shipping AI features, handling ambiguity, production incidents, collaboration |
| 6 | Hiring manager | 45-60 min | Judgment, ownership, tradeoff thinking, team fit, and your questions for them |

Two throughlines the whole loop is really testing:

1. **Do you have taste?** Can you pick the simplest approach that clears the bar (prompt before RAG before fine-tune before agent), and say why.
2. **Can you prove it works?** Evaluation is repeatedly cited as the single biggest skill gap in AI Engineer candidates. The people who can build *and* measure stand out.

## Where to go next in this repo (grounding)

Use these to shore up any topic. Paths are relative to this folder.

- **Topics:** [Foundations](../../../topics/foundations.md) - [Prompting](../../../topics/prompting.md) - [RAG](../../../topics/rag.md) - [Agents](../../../topics/agents.md) - [Evaluation](../../../topics/evaluation.md) - [Fine-tuning](../../../topics/fine-tuning.md) - [Production](../../../topics/production.md) - [Safety and Security](../../../topics/safety-security.md) - [Multimodal](../../../topics/multimodal.md)
- **Journeys:** [Build](../../../journeys/build.md) - [Use](../../../journeys/use.md) - [Understand](../../../journeys/understand.md)
- **Paths:** [Harness Engineering](../../../paths/harness-engineering.md) - [Agent Builder](../../../paths/agent-builder.md)
- **Core question banks:** [60 GenAI Interview Questions](../../60_gen_ai_questions.md) - [Role-Based Interview Prep](../../README.md)
- **Free courses (all, by topic):** [courses.md](../../../courses.md)

---

Back to [Role-Based Interview Prep](../../README.md) - [repository index](../../../README.md).
