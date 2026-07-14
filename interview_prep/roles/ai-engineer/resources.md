# AI Engineer: Resources

Curated free resources to prepare, grouped by topic. Every external link was checked and returned HTTP 200 at the time of writing. Repo-internal links (this repository's own material) come first in each section because they are already organized for interview prep. Favor the 2025-2026 items; older entries are labeled as foundational anchors.

Jump to: [Start here (in this repo)](#start-here-in-this-repo) - [LLM fundamentals](#llm-fundamentals) - [Prompting and context engineering](#prompting-and-context-engineering) - [RAG](#retrieval-and-rag) - [Agents, tools, MCP](#agents-tools-and-mcp) - [Evaluation](#evaluation) - [Reasoning models and the field](#reasoning-models-and-the-state-of-the-field) - [Safety and security](#safety-and-security) - [Interview-specific](#interview-specific) - [Practice repos](#hands-on-practice-repos)

---

## Start here (in this repo)

This repository is built for exactly this prep. Ground yourself here first, then go external for depth.

- [Topic: Foundations](../../../topics/foundations.md), [Prompting](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md): each topic page is organized by journey and level with the best free material.
- [60 GenAI Interview Questions](../../60_gen_ai_questions.md): the repo's flagship question bank.
- [Role-Based Interview Prep](../../README.md): quick role tracks including AI / LLM Engineer.
- [GenAI Roadmap](../../../resources/genai_roadmap.md), [Agents Roadmap](../../../resources/agents_roadmap.md), [RAG Roadmap](../../../resources/RAG_roadmap.md): structured learning paths.
- [Journeys: Build](../../../journeys/build.md), [Use](../../../journeys/use.md), [Understand](../../../journeys/understand.md).
- [Paths: Harness Engineering](../../../paths/harness-engineering.md), [Agent Builder](../../../paths/agent-builder.md).

## LLM fundamentals

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017, foundational anchor): the transformer paper. Know the architecture and self-attention at a high level.
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020, foundational anchor): GPT-3, in-context learning, the few-shot idea.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) (updated through 2025): a broad reference to sweep for gaps in vocabulary and concepts.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) (2025): free, current walk through transformers, tokenization, and using and fine-tuning models.
- Repo topic page: [Foundations](../../../topics/foundations.md).

## Prompting and context engineering

- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) (Anthropic, 2025): the current, practical treatment of context as a managed budget. High-signal for this role.
- [Prompt Engineering Guide](https://www.promptingguide.ai/) (dair-ai, kept current): a thorough, free reference on prompting techniques.
- [Prompt Engineering whitepaper](https://www.kaggle.com/whitepaper-prompt-engineering) (Google via Kaggle, 2024-2025): concise, practitioner-focused overview of prompting patterns.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (2023, foundational anchor): why context position and length matter.
- Repo topic page: [Prompting](../../../topics/prompting.md).

## Retrieval and RAG

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020, foundational anchor): the original RAG paper.
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (updated through 2024-2025): naive, advanced, and modular RAG, a strong mental map.
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) (2023): retrieval with self-reflection, a bridge to agentic RAG.
- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) (2024): GraphRAG and why it helps on global and multi-hop questions.
- [A complete guide to RAG evaluation](https://www.evidentlyai.com/llm-guide/rag-evaluation) (Evidently, current): faithfulness, relevance, context precision/recall, and how to test them.
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216) (2024): a widely used open embedding model, useful for embedding-choice discussions.
- Repo materials: [RAG topic page](../../../topics/rag.md), [Agentic RAG 101](../../../resources/agentic_rag_101.md), [RAG Research Table](../../../research_updates/rag_research_table.md).

## Agents, tools, and MCP

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (Anthropic, 2024-2025): the reference on workflows versus agents and when to use each. Read this before any agent system-design round.
- [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) (OpenAI, 2025, PDF): a complementary practitioner guide.
- [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) (Anthropic, 2025): how to design tools an agent uses reliably.
- [Model Context Protocol: Introduction](https://modelcontextprotocol.io/introduction) (2024-2025): the MCP standard, what it is and why it matters.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022, foundational anchor): the reason-act-observe loop behind most agents.
- [Toolformer](https://arxiv.org/abs/2302.04761) (2023, foundational anchor): models learning to call tools.
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng, 2023): a clear synthesis of planning, memory, and tool use.
- [Agents](https://huyenchip.com/2025/01/07/agents.html) (Chip Huyen, 2025): a grounded, current overview of agent design.
- Repo materials: [Agents topic page](../../../topics/agents.md), [Agents 101 Guide](../../../resources/agents_101_guide.md), [Agentic Search and Retrieval Table](../../../research_updates/agentic_search_retrieval_table.md).

## Evaluation

- [LLM evaluation: a beginner's guide](https://www.evidentlyai.com/llm-guide/llm-evaluation) (Evidently, current): a solid grounding in LLM-as-judge, offline versus online, and metrics.
- [Ragas documentation](https://docs.ragas.io/en/stable/) (current): the RAG triad and metrics made concrete, with code.
- [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) (Lilian Weng, 2024): why models hallucinate and how to think about detection.
- [Common pitfalls when building generative AI applications](https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html) (Chip Huyen, 2025): evaluation and production pitfalls, exactly the judgment interviewers probe.
- Repo materials: [Evaluation topic page](../../../topics/evaluation.md), [AI Evaluation 2025 Table](../../../research_updates/ai_evaluation_2025_table.md).

## Reasoning models and the state of the field

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025): reinforcement learning for reasoning, the reference point for how modern reasoning models are trained.
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) (2022, foundational anchor): the origin of step-by-step reasoning.
- Repo material: [State of AI 2025 Report](../../../research_updates/state_of_ai_2025_report/README.md) and the [research tables](../../../research_updates/) for what is current.

## Safety and security

- [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022, foundational anchor): alignment via principles, useful vocabulary for responsible-AI questions.
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) (2022, foundational anchor): RLHF and instruction following.
- Repo material: [Safety and Security topic page](../../../topics/safety-security.md), [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).

## Interview-specific

- [AI Engineering Field Guide](https://github.com/alexeygrigorev/ai-engineering-field-guide) (2026): research into real AI engineering interview assignments, take-home challenges, and hiring practices. The take-home breakdown is especially useful.
- [AI Engineering Interview Questions](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions) (current): a broad Q and A cheat sheet to self-quiz against.

## Hands-on practice repos

Building is the best prep. These give you working code to learn from and extend.

- [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) (current): a large collection of RAG patterns as runnable notebooks.
- [GenAI Agents](https://github.com/NirDiamant/GenAI_Agents) (current): agent patterns and tutorials end to end.
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) and [OpenAI Cookbook](https://cookbook.openai.com/): official recipes for tool use, structured output, RAG, and evals.
- [Anthropic Courses](https://github.com/anthropics/courses): free hands-on courses (prompting, tool use, evals).
- [LangGraph](https://github.com/langchain-ai/langgraph): a widely used agent-orchestration framework; know at least one.
- [DeepEval](https://github.com/confident-ai/deepeval) and [Arize Phoenix](https://github.com/Arize-ai/phoenix): open evaluation and tracing tools worth knowing by name.
- Repo material: [60 AI Projects](../../../resources/60_ai_projects.md) and [GenAI Projects](../../../resources/gen_ai_projects.md) for project ideas.

---

Next: [courses](courses.md) and the [prep plan](prep-plan.md). Back to the [role README](README.md).
