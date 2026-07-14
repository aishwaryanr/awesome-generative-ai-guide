# FDE Prep Resources

Curated, free resources for the Forward-Deployed Engineer loop, grouped by topic. Every external link was verified to return HTTP 200. Start with this repository's own material, then use the external anchors to go deeper. Favor the 2025 and 2026 material; older items are labeled as foundational anchors.

---

## Start here: this repository

- [FDE README](README.md), [rounds](rounds.md), [questions](questions.md), [courses](courses.md), and [prep-plan](prep-plan.md) in this folder.
- [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and [Role-Based Interview Prep](../../role_based_prep.md) (the Solutions Architect and LLM Engineer tracks overlap most with FDE).
- Topic pages: [Foundations](../../../topics/foundations.md), [Prompting](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Fine-tuning](../../../topics/fine-tuning.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Multimodal](../../../topics/multimodal.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md).
- Journeys and paths: [Build AI](../../../journeys/build.md), [Use AI](../../../journeys/use.md), [Understand AI](../../../journeys/understand.md), [Harness Engineering](../../../paths/harness-engineering.md), [Agent Builder](../../../paths/agent-builder.md).
- Guides: [Agentic RAG 101](../../../resources/agentic_rag_101.md), [Agents 101](../../../resources/agents_101_guide.md), [Fine-tuning 101](../../../resources/fine_tuning_101.md), [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).
- Roadmaps: [GenAI Roadmap](../../../resources/genai_roadmap.md), [Agents Roadmap](../../../resources/agents_roadmap.md), [RAG Roadmap](../../../resources/RAG_roadmap.md).
- Current landscape: [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md), and the research tables on [RAG](../../../research_updates/rag_research_table.md), [AI evaluation](../../../research_updates/ai_evaluation_2025_table.md), and [agentic search and retrieval](../../../research_updates/agentic_search_retrieval_table.md).

---

## The FDE role and interview loop

- [Forward Deployed Engineer Interview: The Definitive 2026 Guide (Exponent)](https://www.tryexponent.com/blog/forward-deployed-engineer-interview-the-definitive-2026-guide-fde): the most complete free breakdown of the loop, round by round, with company differences.
- [Forward Deployed Engineer Interview Questions: 2026 Prep Guide (Perspective AI)](https://getperspective.ai/blog/forward-deployed-engineer-interview-questions-2026-prep-guide): rounds, pass rates, weights, and common red flags.
- [Palantir's Forward-Deployed Engineering Playbook (Perspective AI)](https://getperspective.ai/blog/palantir-forward-deployed-engineering-playbook-anthropic-openai-copying): what an FDE does day-to-day and how OpenAI and Anthropic adapted the model.
- [A Guide to Palantir Forward-Deployed Software Engineering Interviews (Palantir blog)](https://blog.palantir.com/a-guide-to-palantir-forward-deployed-software-engineering-interviews-9c6ba9e07a4c): the source on the decomposition round from the company that invented it. Foundational anchor.
- [Palantir Careers](https://www.palantir.com/careers/): read real FDE and FDSE job descriptions to mirror the language and requirements.

---

## Foundations (transformers, LLMs, the vocabulary)

- [The Illustrated Transformer, Jay Alammar](https://jalammar.github.io/illustrated-transformer/): the clearest visual explanation of attention. Foundational anchor (2018), still the best intro.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): the transformer paper. Foundational anchor (2017); know the idea, not every equation.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165): in-context learning and scaling. Foundational anchor (2020).
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774): capabilities and limitations framing. Anchor (2023).
- [llm-course, Maxime Labonne](https://github.com/mlabonne/llm-course): a free, well-maintained roadmap and notebooks across the LLM stack.

---

## Prompting and context engineering

- [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/): the comprehensive free reference, kept current.
- [Prompt Engineering Guide repo (DAIR.AI)](https://github.com/dair-ai/Prompt-Engineering-Guide): the source repo with techniques and papers.
- [Anthropic prompt engineering docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview): practical, production-oriented guidance.
- [OpenAI prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering): the vendor's own patterns.
- [Effective context engineering for AI agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): why what enters the context window matters more than wording. 2025.
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) and [Self-Consistency](https://arxiv.org/abs/2203.11171): foundational anchors on reasoning-style prompting.

---

## RAG

- [RAG paper (Lewis et al.)](https://arxiv.org/abs/2005.11401): the original retrieval-augmented generation paper. Foundational anchor (2020).
- [RAG_Techniques, Nir Diamant](https://github.com/NirDiamant/RAG_Techniques): a large, current, runnable collection of RAG patterns (chunking, reranking, hybrid search, evaluation).
- [Pinecone Learn](https://www.pinecone.io/learn/): clear articles on embeddings, vector search, and RAG design.
- [Weaviate blog](https://weaviate.io/blog): current write-ups on retrieval, hybrid search, and RAG evaluation.
- Repository deep-dives: [RAG topic](../../../topics/rag.md), [Agentic RAG 101](../../../resources/agentic_rag_101.md), [RAG research table](../../../research_updates/rag_research_table.md).

---

## Agents and MCP

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents): the reference on when to use agents versus workflows and how to keep them simple. 2024, still current.
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629): the reason-act-observe loop behind most agents. Foundational anchor (2022).
- [Model Context Protocol: introduction](https://modelcontextprotocol.io/introduction) and [getting started](https://modelcontextprotocol.io/docs/getting-started/intro): the open standard for wiring agents to tools and data.
- [Introducing MCP (Anthropic announcement)](https://www.anthropic.com/news/model-context-protocol): the why behind MCP.
- [GenAI_Agents, Nir Diamant](https://github.com/NirDiamant/GenAI_Agents): runnable agent patterns and tutorials, current.
- [awesome-ai-agents (E2B)](https://github.com/e2b-dev/awesome-ai-agents): a broad, maintained index of agent frameworks and projects.
- Repository deep-dives: [Agents topic](../../../topics/agents.md), [Agents 101 guide](../../../resources/agents_101_guide.md), [agentic search and retrieval table](../../../research_updates/agentic_search_retrieval_table.md).

---

## Evaluation

- [Evidently AI LLM evaluation guide](https://www.evidentlyai.com/llm-guide): a thorough, current free guide to evaluating LLM and RAG systems.
- Repository: [Evaluation topic](../../../topics/evaluation.md) and the [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md) for current methods and metrics.
- See also the [courses page](courses.md) for the repository's own AI Evals for Everyone course.

---

## Deployment, cost, latency, and reliability

- [LLM Numbers Every Developer Should Know (Ray)](https://github.com/ray-project/llm-numbers): quick mental math for cost, latency, and context budgets.
- [Google SRE Book](https://sre.google/sre-book/table-of-contents/): the reference on reliability, rollouts, monitoring, and incident response. Foundational anchor; the operational mindset the design and reliability rounds reward.
- [Google Cloud SRE Reliability Site](https://www.usenix.org/conference/srecon): SREcon talk archives on running systems in production. (Use the linked talks; the operational thinking transfers directly to FDE deployment questions.)
- [The Pragmatic Engineer newsletter](https://newsletter.pragmaticengineer.com/): free posts on real engineering practice, stakeholder work, and shipping.
- Repository: [Production and LLMOps topic](../../../topics/production.md).

---

## Responsible AI and security

- [Anthropic: Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety): read before any Anthropic mission-alignment round. 2023 anchor, still the reference.
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/rsp): how a frontier lab governs deployment risk.
- Repository: [Safety and Security topic](../../../topics/safety-security.md) and [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).

---

## Coding practice

- [NeetCode](https://neetcode.io/): free structured coding practice. FDE coding is usually practical (parsing, integration, retries) rather than hard algorithms, but fluency helps under time pressure.
- [AI Engineer roadmap (roadmap.sh)](https://roadmap.sh/ai-engineer): a free skills map to find and close gaps.
- [OpenAI Cookbook](https://cookbook.openai.com/) and [OpenAI Cookbook repo](https://github.com/openai/openai-cookbook): runnable recipes for building with LLMs.
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook): runnable recipes and patterns for building with Claude.

---

## Landscape and staying current

- [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md) (repository): the current landscape in one place.
- [LangChain blog](https://blog.langchain.dev/): current write-ups on agent and RAG engineering.
- [Microsoft: Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners): a free, broad, lesson-based course you can skim for gaps.

Next: [courses.md](courses.md) for structured free courses, then [prep-plan.md](prep-plan.md) to sequence everything.
