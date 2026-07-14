# AI Product Manager: Interview Prep

A deep, self-contained prep folder for the **AI Product Manager (AI PM)** role. If you work through the 6 files here, you should not need another resource for this loop.

- **[rounds.md](rounds.md)**: every interview round, what it tests, how it is run, what good looks like, common mistakes, and realistic prompts.
- **[questions.md](questions.md)**: a 45-question bank with concise model answers, grouped by theme.
- **[resources.md](resources.md)**: the best free reading, talks, papers, and repos, all links verified.
- **[courses.md](courses.md)**: the best free courses, starting with this repository's own.
- **[prep-plan.md](prep-plan.md)**: a 4-week plan and a 1-week crunch plan, sequenced day by day.

---

## What the job actually is

An AI Product Manager owns products or features whose core behavior is powered by a model (an LLM, an agent, a recommender, a vision or speech model). The work is ordinary product management plus 4 things that a normal PM rarely has to reason about:

1. **Capability judgment.** Deciding whether a model should be anywhere near the problem, and which class of model (rules, classic ML, or an LLM or agent) fits. The strongest AI PMs start one step earlier than classic product sense: before "what should we build," they ask "should a model touch this at all." Boring, deterministic solutions often win.
2. **Probabilistic UX.** The output is non-deterministic and sometimes confidently wrong. You design the experience *around* wrongness: confidence gates, graceful fallbacks to a human, citations or abstention, undo, and edit-in-place.
3. **Evaluation and metrics.** You own an eval harness, not just a dashboard. You separate offline eval sets and regression suites from online production signals, and you watch for silent model degradation even when the north-star metric rises.
4. **Cost, latency, and risk tradeoffs.** Token cost per call, p95 latency, and unit economics change what you can ship. Responsible AI (fairness, privacy, hallucination, agent permissions, EU AI Act readiness) is threaded through every decision, not bolted on at the end.

The role sits on top of the same skill tree the rest of this repo teaches. Ground yourself in the [Foundations](../../../topics/foundations.md), [RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production](../../../topics/production.md), and [Safety and Security](../../../topics/safety-security.md) topic pages, and follow the [Use AI](../../../journeys/use.md) and [Understand AI](../../../journeys/understand.md) journeys for the conceptual grounding a PM needs without going full engineer.

## Who hires for it, and at what levels

- **Frontier labs and AI-native companies** (model providers, agent platforms, AI-first startups). Loops here go deepest on capability judgment, evals, and agent design, and increasingly include a live prototype round (build a small demo with a tool like Cursor or v0). Example public loops: OpenAI, Sierra (agent PM).
- **Big tech shipping AI features** (Microsoft, Google, Meta, Amazon, and similar). These add a dedicated AI product-sense round to an otherwise standard PM loop. Meta, for example, runs a product-sense round that shifts into building or critiquing an AI feature. Levels map to the normal PM ladder (roughly IC PM, Senior PM, Group PM or Principal, then Director).
- **Enterprises adding AI to existing products** (banks, healthcare, SaaS, retail). Here responsible AI, compliance, build-versus-buy, and cross-functional trust with data science matter most.

Titles you will see for the same work: AI PM, GenAI PM, ML PM, Agent PM, AI Platform PM, Applied AI PM, and Technical PM (AI). Seniority raises the bar on strategy (portfolio, build-versus-buy, moats) and on ambiguity, not on how many facts you memorize.

## How this differs from adjacent roles

| Role | Primary object | What the AI PM does differently |
|------|----------------|-------------------------------|
| **Traditional PM** | A feature and its funnel | Adds capability judgment, probabilistic UX, evals, and cost/latency/risk tradeoffs on top of normal PM. |
| **Data / ML PM (predictive)** | A model in a pipeline | Overlaps heavily; AI PM leans toward generative and agentic products, prompts-as-spec, and RAG rather than only classification and ranking. |
| **AI / LLM Engineer** | The system's code | The PM decides what to build and why, sets eval targets and launch gates, and owns the tradeoffs; the engineer builds it. See the [AI/LLM Engineer track](../../role_based_prep.md). |
| **Applied Scientist** | Model quality and research | The scientist pushes the metric; the PM decides which metric matters to the user and when good enough ships. |
| **AI Program / TPM** | Delivery and coordination | The TPM drives execution across teams; the AI PM owns the what and the why and the success definition. |
| **Designer (AI UX)** | The interaction | Partners closely; the PM owns tradeoffs and metrics, the designer owns the interaction craft. |

## The end-to-end interview loop

A typical 2025-2026 AI PM loop runs 5 to 7 conversations. Not every company runs every round, but the shape is stable:

```
1. Recruiter screen ........ archetype fit, one AI product you truly owned, comp and logistics
2. AI product sense ........ design an AI feature end to end; grade the "should we even use AI" instinct
3. Analytical / metrics .... eval harness, offline vs online, unit economics, north-star vs guardrails
4. AI technical literacy ... what models and agents can and cannot do; RAG, evals, cost, latency tradeoffs
5. Behavioral / leadership . ambiguity, influence, killing an AI feature, explaining AI to non-experts
6. Cross-functional / exec . a real engineer and data scientist in the room; build-vs-buy; roadmap
7. (Increasingly) prototype  build a small working demo; graded on judgment, not code
```

The **evaluation and technical-depth round is the one that most predicts the hire.** It was bolted onto loops over the last 18 months and is where AI-washed candidates fall apart. The [rounds.md](rounds.md) file breaks down each of these in detail.

## How to use this folder

1. Skim this README and [rounds.md](rounds.md) to build a mental map of the loop.
2. Work [questions.md](questions.md) until you can answer each theme cold, in your own words, with a specific example.
3. Close gaps with the topic pages and [resources.md](resources.md) / [courses.md](courses.md).
4. Follow [prep-plan.md](prep-plan.md) to sequence everything into a dated path.

Related repo starting points: the [60 GenAI Interview Questions](../../60_gen_ai_questions.md), the [Role-Based Interview Prep](../../role_based_prep.md) tracks, the [GenAI](../../../resources/genai_roadmap.md), [Agents](../../../resources/agents_roadmap.md), and [RAG](../../../resources/RAG_roadmap.md) roadmaps, and the [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md) for current context.
