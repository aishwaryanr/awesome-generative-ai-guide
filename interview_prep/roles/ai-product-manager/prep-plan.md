# AI Product Manager: Prep Plan

Two dated plans that sequence the [rounds](rounds.md), [questions](questions.md), [resources](resources.md), and [courses](courses.md) into a path. Pick the 4-week plan if you have runway, or the 1-week crunch if the loop is close. Both assume you already have general PM fundamentals and are adding the AI-specific layer.

How to use it: each day has a **learn** block and a **practice** block. Do the practice out loud or in writing, because the interview is spoken. Keep a running portfolio of 4 stories (one you shipped and measured, one you killed, one where you pushed back on AI, one degradation or fairness incident) and refine it all the way through.

---

## Timeline

```text
4-week plan. Adds the AI-specific layer on top of general PM fundamentals.

  Week 1 |== Foundations + capability judgment ====|
  Week 2 |== Evaluation, metrics, and cost =========|
  Week 3 |== Agents, technical literacy, responsible AI ==|
  Week 4 |== Strategy, execution, behavioral =======| --> LOOP
```

## 4-Week Plan (about 1 to 2 hours per day)

### Week 1: Foundations and capability judgment

- **Day 1.** Read the [README](README.md) and [rounds.md](rounds.md) end to end. Map the loop and note which rounds scare you most. Skim the [GenAI roadmap](../../../resources/genai_roadmap.md).
- **Day 2.** Learn: [Foundations topic](../../../topics/foundations.md) and start [Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners) (lessons 1 to 4). Practice: the [Capability judgment and product sense](questions.md#1-capability-judgment-and-product-sense) questions out loud.
- **Day 3.** Learn: [Prompting](../../../topics/prompting.md) and [RAG](../../../topics/rag.md) topics. Practice: the [Model and architecture literacy](questions.md#2-model-and-architecture-literacy) questions. Explain RAG and embeddings to an imaginary non-technical exec in 60 seconds each.
- **Day 4.** Learn: [People + AI Guidebook](https://pair.withgoogle.com/guidebook/) chapters on mental models, feedback, and errors. Practice: the [Capability judgment and product sense](questions.md#1-capability-judgment-and-product-sense) questions (product sense and trust).
- **Day 5.** Learn: [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/) and [A practical guide to building agents (OpenAI)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf). Practice: sketch the architecture of an AI feature you know on paper.
- **Weekend.** First full mock: an AI product-sense prompt from [rounds.md](rounds.md) round 2, timed at 45 minutes, self-recorded. Draft 2 of your 4 portfolio stories.

### Week 2: Evaluation, metrics, and cost

- **Day 6-7.** Learn: [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) (this is the highest-leverage material for the whole loop). Practice: the [Evaluation and metrics](questions.md#3-evaluation-and-metrics) questions.
- **Day 8.** Learn: [Evidently AI LLM guide](https://www.evidentlyai.com/llm-guide) and [OpenAI Evals guide](https://platform.openai.com/docs/guides/evals). Practice: the [Evaluation and metrics](questions.md#3-evaluation-and-metrics) questions (LLM-as-judge, agent metrics, offline vs online).
- **Day 9.** Learn: cost and latency. Reread rounds.md round 3 and the [Production topic](../../../topics/production.md). Practice: the [Cost, latency, and unit economics](questions.md#4-cost-latency-and-unit-economics) questions (unit economics, latency, model sizing). Do the arithmetic on a real cost-per-call example.
- **Day 10.** Learn: [AI Evaluation 2025 research table](../../../research_updates/ai_evaluation_2025_table.md) and [RAG research table](../../../research_updates/rag_research_table.md). Practice: the RAG diagnosis question (in [Model and architecture literacy](questions.md#2-model-and-architecture-literacy)) and the silent-degradation question (in [Evaluation and metrics](questions.md#3-evaluation-and-metrics)) as full 3-minute answers.
- **Weekend.** Second mock: the metrics and technical-depth round, 45 minutes, with a friend playing a data scientist who keeps asking "how did you measure that." Draft your remaining 2 portfolio stories.

### Week 3: Agents, technical literacy, and responsible AI

- **Day 11-12.** Learn: [Agents topic](../../../topics/agents.md), [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents), and [Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/README.md). Practice: the agent questions in [Model and architecture literacy](questions.md#2-model-and-architecture-literacy) and [Responsible AI, safety, and agents](questions.md#5-responsible-ai-safety-and-agents).
- **Day 13.** Learn: [Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [MCP intro](https://modelcontextprotocol.io/docs/getting-started/intro). Practice: the context-engineering and MCP questions in [Model and architecture literacy](questions.md#2-model-and-architecture-literacy).
- **Day 14.** Learn: [Safety and Security topic](../../../topics/safety-security.md) and [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Practice: the [Responsible AI, safety, and agents](questions.md#5-responsible-ai-safety-and-agents) questions.
- **Day 15.** Learn: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) and the [EU AI Act explorer](https://artificialintelligenceact.eu/). Practice: the [Responsible AI, safety, and agents](questions.md#5-responsible-ai-safety-and-agents) questions (responsible AI as launch gates).
- **Day 16.** Learn: [Fine-tuning topic](../../../topics/fine-tuning.md) and [Fine-tuning 101](../../../resources/fine_tuning_101.md). Practice: the fine-tuning questions in [Model and architecture literacy](questions.md#2-model-and-architecture-literacy) (build vs fine-tune vs custom).
- **Weekend.** Third mock: the AI technical-literacy round with someone technical, then the guardrails-for-an-agent design question live.

### Week 4: Strategy, execution, behavioral, and integration

- **Day 17.** Learn: [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md) and [The AI PM Roadmap 2026](https://www.productcompass.pm/p/ai-product-manager-roadmap-2026). Practice: the [Business and strategy](questions.md#6-business-and-strategy) questions (build vs buy, pricing, moats, roadmap, ROI).
- **Day 18.** Learn: rounds.md round 6 (cross-functional) and the [company interview guides](resources.md) for your target (Microsoft, OpenAI, or Sierra). Practice: the probabilistic-PRD question in [Execution and behavioral](questions.md#7-execution-and-behavioral) and a build-vs-buy decision live.
- **Day 19.** Behavioral day. Practice: the [Execution and behavioral](questions.md#7-execution-and-behavioral) questions plus the rounds.md round 5 prompts. Tighten all 4 portfolio stories into STAR with real numbers. Make sure you have a graveyard story.
- **Day 20.** Optional prototype round: build a small demo of one feature with an AI builder tool (Cursor, v0, or similar), narrating tradeoffs. See rounds.md round 7.
- **Day 21.** Full loop simulation: product sense, metrics, technical, behavioral back to back. Note weak spots and reread only those questions and topics.
- **Ongoing.** Skim the [2026 papers folder](../../../research_updates/2026_papers) so you have one or two fresh references to mention.

## 1-Week Crunch Plan (about 2 to 3 hours per day)

For a loop that is days away. Triage hard: the metrics and technical rounds fail candidates most, so weight them.

- **Day 1.** Read [README](README.md) and [rounds.md](rounds.md). Work the [Capability judgment and product sense](questions.md#1-capability-judgment-and-product-sense) questions out loud. Draft your 4 portfolio stories.
- **Day 2.** Do the core of [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md). Work the [Evaluation and metrics](questions.md#3-evaluation-and-metrics) questions. This is the highest-value day.
- **Day 3.** Read [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and the [Agents topic](../../../topics/agents.md). Work the [Model and architecture literacy](questions.md#2-model-and-architecture-literacy) questions. Practice the two 60-second explainer answers (RAG, embeddings).
- **Day 4.** Cost and responsible AI. Work the [Cost, latency, and unit economics](questions.md#4-cost-latency-and-unit-economics) and [Responsible AI, safety, and agents](questions.md#5-responsible-ai-safety-and-agents) questions. Skim the [People + AI Guidebook](https://pair.withgoogle.com/guidebook/) errors and trust chapters.
- **Day 5.** Strategy and execution. Work the [Business and strategy](questions.md#6-business-and-strategy) and [Execution and behavioral](questions.md#7-execution-and-behavioral) questions. Read the [company interview guide](resources.md) for your target. Tighten STAR stories.
- **Day 6.** Two mocks: one AI product-sense round and one metrics round, timed, out loud, ideally with a friend drilling "how did you measure that."
- **Day 7.** Light review of only your weak spots. Reread the [rounds.md](rounds.md) "common mistakes" for each round. Rest so you are sharp.

## Readiness checklist (green before you walk in)

- I can decide whether a problem should use AI at all, and defend it.
- I can design an AI feature end to end including the unhappy path (fallback, confidence, citations, undo).
- I can describe an eval harness: offline set and regression suite versus online signals, and name specific metrics (recall@k, faithfulness, containment, escalation, edit rate).
- I can define hallucination, measure it, and name a launch gate.
- I can do the unit-economics arithmetic and reason about p95 latency.
- I can explain RAG, embeddings, agents, reasoning models, and MCP in plain language, and say when each fits.
- I can design guardrails for an agent (scoped permissions, human confirmation, limits, kill switch, red-team).
- I have 4 crisp portfolio stories with numbers, including one AI feature I killed.

When you can check all 8, you are ready. Back to the [README](README.md).
