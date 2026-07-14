# FDE Free Courses

The best free courses for the Forward-Deployed Engineer loop, grouped by topic. Start with this repository's own courses (they are self-contained and current), then use the verified external courses to go deeper. Every external link returns HTTP 200. Free only.

An FDE needs breadth over narrow depth: enough foundations to reason clearly, real fluency in RAG and agents, genuine evaluation skill, and the deployment and reliability mindset. Sequence accordingly.

---

## Start here: this repository's courses

- [Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/README.md): builds agent fundamentals (tools, memory, loops, orchestration) that the design and agent questions assume. Do this first if agents are your weak spot.
- [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md): builds evaluation skill, which is the single most-probed AI depth area for FDE roles at OpenAI and Anthropic ("how do you know it works?"). Do not skip this.
- [All free courses by topic (repository index)](../../../courses.md): the full catalog to find a course for any gap.
- Structured tracks: [Harness Engineering path](../../../paths/harness-engineering.md) and [Agent Builder path](../../../paths/agent-builder.md) sequence exactly what an AI FDE deploys.

---

## Foundations and the LLM stack

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1): free, hands-on course across transformers, tokenization, and using and adapting LLMs. Builds the base the foundations questions assume.
- [Generative AI for Beginners (Microsoft)](https://github.com/microsoft/generative-ai-for-beginners): a free, lesson-based course covering prompting, RAG, agents, and app-building. Good breadth if you are starting cold.
- [AI For Beginners (Microsoft)](https://microsoft.github.io/AI-For-Beginners/): a free curriculum for the broader ML and AI vocabulary. Foundational anchor; skim for gaps.
- [The Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/): free lectures on building LLM-powered applications end to end, close to the FDE build loop.

---

## Prompting and context engineering

- [ChatGPT Prompt Engineering for Developers (DeepLearning.AI)](https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/1/introduction): free short course on practical prompting patterns for building, not just chatting.
- [Anthropic prompt engineering interactive tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial): a free, hands-on, notebook-based course on prompting for production. Builds the reliable-prompting skill the coding and design rounds reward.
- [DeepLearning.AI short courses catalog](https://www.deeplearning.ai/short-courses/): browse the current free short-course list for prompting, RAG, agents, and evaluation.

---

## RAG

- [Anthropic Courses (GitHub)](https://github.com/anthropics/courses): free courses including retrieval and building with Claude, notebook-based and current.
- [RAG_Techniques (Nir Diamant)](https://github.com/NirDiamant/RAG_Techniques): a course-grade, runnable collection covering chunking, hybrid search, reranking, and RAG evaluation. Work through several notebooks; RAG is central to the FDE build and design rounds.
- Repository: the [RAG topic page](../../../topics/rag.md) and [Agentic RAG 101](../../../resources/agentic_rag_101.md) read as a compact course.

---

## Agents and MCP

- [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction): a free, structured, hands-on course on building agents from fundamentals through deployment. Pair with the repository's Agentic AI Crash Course.
- [Functions, Tools and Agents with LangChain (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/): free short course on tool use and function calling, the mechanics behind reliable agents.
- [AI Agents in LangGraph (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/): free short course on building agentic workflows with explicit control flow.
- [MCP: Build Rich-Context AI Apps with Anthropic (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/): free short course on the Model Context Protocol, directly relevant to integrating into a customer's tools and data.
- [GenAI_Agents (Nir Diamant)](https://github.com/NirDiamant/GenAI_Agents): runnable agent tutorials and patterns to practice building.
- Repository: [Agents topic](../../../topics/agents.md) and [Agents 101 guide](../../../resources/agents_101_guide.md).

---

## Evaluation

- [AI Evals for Everyone (this repository)](../../../free_courses/ai_evals_for_everyone/README.md): the primary course for the evaluation depth FDE loops probe hard. Learn to build a labeled set, validate a judge, and measure reliability.
- [Evaluating and Debugging Generative AI (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/): free short course on experiment tracking, evaluation, and debugging generative systems.
- Repository: [Evaluation topic](../../../topics/evaluation.md) and the [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md).

---

## Fine-tuning (know when, and roughly how)

- Repository: [Fine-tuning topic](../../../topics/fine-tuning.md) and [Fine-tuning 101 guide](../../../resources/fine_tuning_101.md). For FDE you mainly need to defend when to fine-tune versus RAG versus prompt, not to train models daily, so understanding beats hands-on depth here.
- [llm-course (Maxime Labonne)](https://github.com/mlabonne/llm-course): free notebooks on fine-tuning and quantization if you want hands-on depth.

---

## Deployment, reliability, and responsible AI

- [Google SRE Book](https://sre.google/sre-book/table-of-contents/): the free reference for the reliability, rollout, and incident-response mindset the design and reliability rounds reward. Foundational anchor; read the chapters on service level objectives, monitoring, and release engineering.
- Repository: [Production and LLMOps topic](../../../topics/production.md), [Safety and Security topic](../../../topics/safety-security.md), and [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).
- [Anthropic: Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) and [Responsible Scaling Policy](https://www.anthropic.com/rsp): required reading before an Anthropic mission-alignment round.

---

## Coding fluency

- [NeetCode](https://neetcode.io/): free structured practice for coding-under-time-pressure. FDE coding is practical, but fluency removes friction.
- [OpenAI Cookbook](https://cookbook.openai.com/) and [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook): work recipes to build muscle memory for LLM app code (retries, streaming, tool calls, RAG).

Next: [prep-plan.md](prep-plan.md) sequences these courses into a day-by-day path.
