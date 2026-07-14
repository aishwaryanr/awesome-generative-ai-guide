# Resources

Curated, free resources to prepare for the AI Strategist loop. Every external link was checked and returned HTTP 200 at the time of writing; if one rots, search the title. Grouped by what it prepares you for. Start with the repository's own material, which is built for exactly this.

---

## Start here (in this repository)

- [Foundations](../../../topics/foundations.md), [Prompting and context](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Fine-tuning](../../../topics/fine-tuning.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and security](../../../topics/safety-security.md), [Multimodal](../../../topics/multimodal.md): the concept map for the fluency round.
- [Use AI journey](../../../journeys/use.md), [Build with AI journey](../../../journeys/build.md), [Understand AI journey](../../../journeys/understand.md): pick your depth.
- [GenAI roadmap](../../../resources/genai_roadmap.md), [Agents roadmap](../../../resources/agents_roadmap.md), [RAG roadmap](../../../resources/RAG_roadmap.md): structured learning paths.
- [Agentic RAG 101](../../../resources/agentic_rag_101.md), [Agents 101 guide](../../../resources/agents_101_guide.md), [Fine-tuning 101](../../../resources/fine_tuning_101.md), [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md): practical guides.
- [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md): the market context you will be quizzed on.
- [RAG research table](../../../research_updates/rag_research_table.md), [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md), [Agentic search and retrieval table](../../../research_updates/agentic_search_retrieval_table.md): current, cited research.
- [60 GenAI interview questions](../../60_gen_ai_questions.md) and [role-based prep](../../role_based_prep.md): the question banks.
- Builder paths, to understand what your engineers do: [Harness engineering](../../../paths/harness-engineering.md), [Agent builder](../../../paths/agent-builder.md).

---

## AI fluency: LLMs, agents, RAG, reasoning

- [Andrej Karpathy: Intro to Large Language Models (1 hour talk)](https://www.youtube.com/watch?v=zjkBMFhNj_g): the single best plain-language grounding in how LLMs work.
- [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI): a longer, deeper follow-up on training and behavior.
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): the reference on when a workflow beats an agent and how to keep agents simple. Essential for the agents question.
- [Anthropic: Tips for building AI agents (video)](https://www.youtube.com/watch?v=LP5OCa20Zpg): short, practical, current.
- [OpenAI: A practical guide to building agents (PDF)](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf): a business-oriented framing of agents, tools, and guardrails.
- [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): why what enters the context window matters more than phrasing.
- [Chip Huyen: Agents](https://huyenchip.com/2025/01/07/agents.html): a clear, vendor-neutral breakdown of agent design and failure modes.
- [Chip Huyen: Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html): still-useful foundational anchor on the production gap.
- [Prompting Guide](https://www.promptingguide.ai/): a broad, current reference on prompting and context techniques.
- [Google: Prompt engineering resources](https://developers.google.com/machine-learning/resources/prompt-eng): concise official guidance.
- [Kaggle and Google: Prompt engineering whitepaper](https://www.kaggle.com/whitepaper-prompt-engineering): deeper, structured treatment.

## MCP, tools, and the current stack

- [Model Context Protocol: introduction](https://modelcontextprotocol.io/introduction): the open standard for connecting agents to tools and data.
- [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol): the why, in business terms.
- [Anthropic cookbook](https://github.com/anthropics/anthropic-cookbook): worked examples of RAG, tools, and evals, useful for understanding what your engineers actually build.

## Evaluation and quality

- [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md) (repo): current, cited evaluation research in one place.
- [RAG research table](../../../research_updates/rag_research_table.md) (repo): how RAG quality is measured and improved.
- The repo's [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course (see [courses.md](courses.md)) is your primary eval resource.

## Papers to be able to name and summarize

You do not need to read these end to end. You need a one-sentence summary of each and why it mattered.

- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): the original RAG paper. Foundational anchor.
- [Retrieval-Augmented Generation for LLMs: a survey (2023)](https://arxiv.org/abs/2312.10997): a map of the modern RAG landscape.
- [Chain-of-Thought prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903): why letting models reason step by step works. Foundational anchor.
- [ReAct: reasoning and acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): the reason-then-act loop behind agents. Foundational anchor.
- [GPT-4 technical report (2023)](https://arxiv.org/abs/2303.08774): capability and evaluation framing from a frontier model. Foundational anchor.
- [Llama 2 (2023)](https://arxiv.org/abs/2307.09288): the open-weight model that shaped the build-vs-buy conversation. Foundational anchor.

## Business, ROI, and the enterprise-AI reality

- [Stanford HAI: 2025 AI Index Report](https://hai.stanford.edu/ai-index/2025-ai-index-report): the definitive, citable data on adoption, cost, capability, and investment. Know a handful of its numbers.
- [Deloitte: State of Generative AI in the Enterprise](https://www.deloitte.com/us/en/what-we-do/capabilities/applied-artificial-intelligence/content/state-of-generative-ai-in-enterprise.html): survey data on adoption, ROI, and the change-management gap.
- [Anthropic Economic Index](https://www.anthropic.com/economic-index): real data on how AI is actually used across tasks and occupations.
- [a16z: 16 changes to the way enterprises are building with AI (2025)](https://a16z.com/ai-enterprise-2025/): current buyer-side patterns in enterprise AI.
- [a16z: How generative AI is transforming the enterprise (2024)](https://a16z.com/generative-ai-enterprise-2024/): the prior year's baseline, useful for showing the trend.
- [Ethan Mollick: One Useful Thing](https://www.oneusefulthing.org/): grounded, non-hype writing on how AI actually changes work and organizations.

## Risk, governance, and responsible AI

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework): the govern-map-measure-manage framework enterprises anchor to. Learn the four functions.
- [NIST AI RMF 1.0 (full PDF)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf): the document itself.
- [NIST AI RMF Knowledge Base](https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF): a navigable version with the playbook.
- [EU AI Act explorer](https://artificialintelligenceact.eu/): the risk tiers and obligations, in readable form.
- [European Commission: regulatory framework on AI](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai): the official overview and timeline.
- [OECD AI Principles](https://oecd.ai/en/ai-principles): the widely referenced international baseline.
- [OpenAI: Practices for governing agentic AI systems (PDF)](https://cdn.openai.com/papers/practices-for-governing-agentic-ai-systems.pdf): governance specific to agents.
- [Anthropic: Core views on AI safety](https://www.anthropic.com/index/core-views-on-ai-safety): a frontier lab's framing of the risk landscape.
- [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md) (repo): prompt injection, tool risk, and controls.

## Staying current (read weekly during prep)

- [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/): weekly, business-aware AI news.
- [Latent Space](https://www.latent.space/): deep interviews and analysis on the applied AI stack.
- [The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/): engineering-side context on how AI is built and shipped.

Next: **[courses.md](courses.md)**.
