# AI Product Manager: Resources

Curated, free resources for AI PM prep. Every external link was checked and returned HTTP 200 at the time of writing. Start with this repository's own material (it is written for exactly this purpose), then branch out.

---

## Start here: in this repository

- **[Topic pages](../../../topics/foundations.md)**: [Foundations](../../../topics/foundations.md), [Prompting](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Fine-tuning](../../../topics/fine-tuning.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Multimodal](../../../topics/multimodal.md), [Production](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md).
- **[Journeys](../../../journeys/use.md)**: [Use AI](../../../journeys/use.md) and [Understand AI](../../../journeys/understand.md) give a PM the right depth without going full engineer; [Build AI](../../../journeys/build.md) if you want to go deeper.
- **[60 GenAI Interview Questions](../../60_gen_ai_questions.md)** and **[Role-Based Prep](../../README.md)**: the core question banks.
- **Roadmaps**: [GenAI roadmap](../../../resources/genai_roadmap.md), [Agents roadmap](../../../resources/agents_roadmap.md), [RAG roadmap](../../../resources/RAG_roadmap.md).
- **Guides**: [Agentic RAG 101](../../../resources/agentic_rag_101.md), [Fine-tuning 101](../../../resources/fine_tuning_101.md), [Agents 101](../../../resources/agents_101_guide.md), [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).
- **Research tables (current)**: [RAG research](../../../research_updates/rag_research_table.md), [AI Evaluation 2025](../../../research_updates/ai_evaluation_2025_table.md), [Agentic Search and Retrieval](../../../research_updates/agentic_search_retrieval_table.md).
- **[State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)**: the macro context an AI PM is expected to know.
- **Paths**: [Harness Engineering](../../../paths/harness-engineering.md) and [Agent Builder](../../../paths/agent-builder.md) for the agent-product depth interviewers probe.

## Interview-specific guides (AI PM loops at named companies)

- [Microsoft AI Product Manager interview guide (Exponent, 2026)](https://www.tryexponent.com/guides/microsoft-ai-product-manager-interview): rounds and sample questions.
- [Sierra Agent Product Manager interview guide (Exponent, 2026)](https://www.tryexponent.com/guides/sierra-agent-product-manager-pm-interview-guide): agent-PM specific, RAG, MCP, memory, agent metrics.
- [OpenAI Product Manager interview guide (Exponent, 2026)](https://www.tryexponent.com/guides/openai-product-manager-interview): frontier-lab PM loop.
- [Product Sense interview prep (Exponent, 2026)](https://www.tryexponent.com/blog/product-sense-interview): the classic product-sense frame the AI round builds on.
- [AI Product Manager interview questions (KORE1, 2026)](https://www.kore1.com/ai-product-manager-interview-questions-2026/): deep question set on evals, unit economics, guardrails, with strong-answer guidance.
- [AI Product Sense Guide (Aakash Gupta)](https://www.news.aakashg.com/p/ai-product-sense-guide): how to run the AI product-sense round.
- [The AI Product Manager Roadmap 2026 (Product Compass)](https://www.productcompass.pm/p/ai-product-manager-roadmap-2026): the skill map for the role.

## Core reading: what models and agents can and cannot do

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents): the reference on when to use a workflow versus an agent, and the common patterns. Essential for the technical-literacy and product-sense rounds.
- [How we built our multi-agent research system (Anthropic)](https://www.anthropic.com/engineering/built-multi-agent-research-system): multi-agent tradeoffs, cost, and evaluation from a shipped system.
- [Effective Context Engineering for AI Agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): the context-as-product-lever framing.
- [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) (OpenAI, 2025): the production architecture and orchestration (routing, guardrails, caching, observability) a PM should picture. Foundational anchor.
- [Emerging patterns in building GenAI products](https://martinfowler.com/articles/gen-ai-patterns/) (Martin Fowler, Thoughtworks): the enduring patterns and gotchas of shipping LLM products. Foundational anchor.
- [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/): the canonical reference-architecture diagram. Foundational anchor.
- [Model Context Protocol (intro)](https://modelcontextprotocol.io/docs/getting-started/intro) and the [MCP home](https://modelcontextprotocol.io/): the standard for connecting agents to tools and data.

## Evaluation and metrics

- [AI Evals for Everyone (this repo)](../../../free_courses/ai_evals_for_everyone/README.md): the single best starting point for the evaluation round.
- [AI Evaluation 2025 research table (this repo)](../../../research_updates/ai_evaluation_2025_table.md): current methods and benchmarks.
- [OpenAI Evals guide](https://platform.openai.com/docs/guides/evals): how offline eval harnesses are actually built.
- [Evidently AI: LLM evaluation guide](https://www.evidentlyai.com/llm-guide): practical, current guide to offline and online LLM evaluation and monitoring.
- [Measuring the value of AI agents (Google Cloud)](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents): the reliability, adoption, and business-value framing for agent metrics.

## Responsible AI, safety, and regulation

- [People + AI Guidebook (Google PAIR)](https://pair.withgoogle.com/guidebook/): the definitive free reference on designing human-centered AI UX (mental models, feedback, errors, trust). Read this before any product-sense round.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework): the govern, map, measure, manage structure interviewers expect you to know.
- [EU AI Act explorer](https://artificialintelligenceact.eu/): risk tiers and obligations that shape launch gates.
- [Securing Agentic AI Systems (this repo)](../../../resources/securing_agentic_ai_systems.md): prompt injection, permissions, and agent guardrails.

## Papers to know (recognize the idea, not memorize the math)

Foundational anchors that define the vocabulary interviewers use:

- [Attention Is All You Need (Transformer, 2017)](https://arxiv.org/abs/1706.03762)
- [Retrieval-Augmented Generation (RAG, 2020)](https://arxiv.org/abs/2005.11401)
- [Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903)
- [ReAct: Reasoning and Acting in LLMs (2022)](https://arxiv.org/abs/2210.03629)
- [Training Language Models to Follow Instructions (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155)
- [Constitutional AI (2022)](https://arxiv.org/abs/2212.08073)
- [Toolformer (2023)](https://arxiv.org/abs/2302.04761)
- [Training Compute-Optimal LLMs (Chinchilla scaling, 2022)](https://arxiv.org/abs/2203.15556)
- [A Survey on Evaluation of Large Language Models (2023)](https://arxiv.org/abs/2307.03109)

## Product strategy and staying current

- [Mind the Product](https://www.mindtheproduct.com/): general PM craft, with growing AI product coverage.
- [State of AI 2025 report (this repo)](../../../research_updates/state_of_ai_2025_report/README.md): the macro trends to reference.
- Follow the [2026 papers folder](../../../research_updates/2026_papers) and [survey papers](../../../research_updates/survey_papers.md) in this repo to keep current without doom-scrolling.

A note on sourcing: favor 2025-2026 material for anything about capability, cost, and agent patterns, since the field moves fast. The papers above are labeled as foundational anchors on purpose; know the idea, not the arithmetic.
