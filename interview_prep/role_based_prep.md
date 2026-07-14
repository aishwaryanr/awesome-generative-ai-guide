# Role-Based Interview Prep

GenAI interviews look very different depending on the role. An LLM engineer gets grilled on RAG and agent design, an ML engineer on fine-tuning and serving, a PM on tradeoffs and product sense. This guide maps the common AI roles to what interviews actually test, and the material in this repository to prepare with.

**Start here, for every role:** the [60 GenAI Interview Questions](60_gen_ai_questions.md). Then follow the role below.

`Level 🟡→🔴 · Source ⭐ LevelUp Labs original / 🌐 External`

---

## 🏗️ AI / LLM Engineer

Builds LLM-powered applications: prompting, RAG, agents, evaluation, and shipping. The most common GenAI role today.

- **Tested on:** LLM fundamentals, prompting and context engineering, RAG design, agents and tool use, evaluation, and the system design of an LLM application.
- **Prepare with:** the [Build journey](../journeys/build.md) 101 to 301, and the [Retrieval and RAG](../topics/rag.md), [AI Agents](../topics/agents.md), [Prompting and Context](../topics/prompting.md), and [Evaluation and Observability](../topics/evaluation.md) topics.
- **Go deep:** the [Agent Builder](../paths/agent-builder.md) and [Harness Engineering](../paths/harness-engineering.md) paths.

## 🔧 ML / Fine-tuning Engineer

Trains, fine-tunes, and serves models, and owns the MLOps around them.

- **Tested on:** transformers and foundations, fine-tuning and post-training (SFT, LoRA, DPO, RLHF, GRPO), quantization and serving, LLMOps, and evaluation.
- **Prepare with:** the [LLM Foundations](../topics/foundations.md), [Fine-tuning and Post-training](../topics/fine-tuning.md), and [Production and LLMOps](../topics/production.md) topics.

## 🔬 Applied Scientist / Research Engineer

Designs methods, runs experiments, reads and implements papers, and builds evaluations.

- **Tested on:** deep foundations, reasoning models, the active research areas (agents, RAG, alignment, evaluation), reading a paper, and designing a rigorous eval.
- **Prepare with:** the [Understand journey](../journeys/understand.md), the living research tables ([RAG](../research_updates/rag_research_table.md), [AI evaluation](../research_updates/ai_evaluation_2025_table.md), [agentic search](../research_updates/agentic_search_retrieval_table.md)), the monthly [best papers](../research_updates/2026_papers), and the [State of AI report](../research_updates/state_of_ai_2025_report/README.md).

## 📋 AI Product Manager

Shapes AI products and balances capability, cost, risk, and user experience.

- **Tested on:** what LLMs and agents can and cannot do, tradeoffs (latency, cost, quality), evaluation and metrics, responsible AI, and product sense.
- **Prepare with:** the [Use journey](../journeys/use.md), the [AI Agents](../topics/agents.md) and [Evaluation and Observability](../topics/evaluation.md) topics (product evals), the [Safety and Security](../topics/safety-security.md) topic, and the [State of AI report](../research_updates/state_of_ai_2025_report/README.md).

## 🏛️ AI Solutions Architect / Enterprise AI

Designs and integrates AI systems for an organization.

- **Tested on:** system design of RAG and agents at scale, integration, security, cost, evaluation, and production reliability.
- **Prepare with:** [Build 301](../journeys/build.md#build-301-production-and-frontier), the [Production and LLMOps](../topics/production.md) and [Safety and Security](../topics/safety-security.md) topics, and [Securing Agentic AI Systems](../resources/securing_agentic_ai_systems.md).

---

## How to use this

1. Do the [60 GenAI Interview Questions](60_gen_ai_questions.md) as a baseline, whatever your role.
2. Work the topics for your role above, using the linked topic pages and journey levels.
3. Stay current on the research your interviewer will expect you to know, via the [monthly papers](../research_updates/2026_papers) and [research tables](../research_updates/).

> System-design interview drills (LLM search, a support agent, chat-with-your-data, and more) are planned as a future addition. Until then, use the system-design cues under [Build 301](../journeys/build.md#build-301-production-and-frontier).
