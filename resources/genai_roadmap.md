# 5-Day LLM Foundations Roadmap

If you're feeling overwhelmed by the scattered knowledge about LLMs, this roadmap and curated resources from top sources are here to guide you. Dedicate 2 to 3 hours daily to work through the resources thoroughly, and by the fifth day you'll be ready to build your own LLM application. **This roadmap is designed for individuals with basic machine learning knowledge.** Once you've established your foundation, use this repository to dive into research papers, explore additional courses, and keep building.

![Applied_LLMs_(28).png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Applied_LLMs_(28).png)

> Updated for 2026. The way people build with LLMs has changed: most builders never fine-tune a model, they assemble prompting, context, retrieval, and tools into an agent, then evaluate it. So the core 5 days below take you from foundations to a working, evaluated agent. Fine-tuning is here too, as an optional advanced day for when you actually need it.

## Day 1: LLM and Reasoning Model Foundations

**Watch (about 1 hour):**

1. Intro to Large Language Models by Andrej Karpathy ([link](https://www.youtube.com/watch?v=zjkBMFhNj_g))
2. How Transformer LLMs Work by DeepLearning.AI ([link](https://www.deeplearning.ai/courses/how-transformer-llms-work))

**Read (about 1 hour):**

1. LLM Lingo: the common terms in plain language ([link](../resources/llm_lingo))
2. What are LLMs, by Amazon ([link](https://aws.amazon.com/what-is/large-language-model/))
3. **(Optional, go deeper)** Build a GPT from scratch, Neural Networks: Zero to Hero by Andrej Karpathy ([link](https://karpathy.ai/zero-to-hero.html))

---

## Day 2: Prompting and Context Engineering

Prompting is the fastest lever. In 2026 the bigger skill is context engineering: deciding what the model sees, not just how you phrase the ask.

**Watch (about 1 hour):**

1. Prompt Engineering Interactive Tutorial by Anthropic ([link](https://github.com/anthropics/courses/tree/master/prompt_engineering_interactive_tutorial))

**Read (about 1 hour):**

1. Introduction to prompt engineering ([link](https://www.promptingguide.ai/introduction))
2. Advanced prompt engineering techniques ([link](https://www.promptingguide.ai/techniques))

---

## Day 3: Retrieval Augmented Generation (RAG)

**Watch (about 1 hour):**

1. Retrieval Augmented Generation (RAG) by DeepLearning.AI ([link](https://www.deeplearning.ai/courses/retrieval-augmented-generation))

**Read (about 1 hour):**

1. Agentic RAG 101: retrieval combined with agentic control ([link](../resources/agentic_rag_101.md))
2. **(Optional, current research)** Most impactful RAG papers, updated regularly ([link](../research_updates/rag_research_table.md))

---

## Day 4: Agents and Tools

An agent is a model plus a harness: tools, memory, and a loop that lets it take actions, not just answer. This is how most real LLM applications are built now.

**Watch (about 1 hour):**

1. AI Agents Course, unit 0, by Hugging Face ([link](https://huggingface.co/learn/agents-course/unit0/introduction))

**Read (about 1.5 hours):**

1. Agentic AI Crash Course: agents, tools, RAG, MCP, planning, memory, multi-agent ([link](../free_courses/agentic_ai_crash_course/README.md))
2. Introduction to the Model Context Protocol by Anthropic ([link](https://anthropic.skilljar.com/introduction-to-model-context-protocol))

---

## Day 5: Evaluation and Shipping

Building an agent is half the job. The other half is knowing whether it works, and catching it when it breaks in production. That is evaluation and observability.

**Watch (about 1 hour):**

1. Evaluating AI Agents by DeepLearning.AI and Arize ([link](https://www.deeplearning.ai/courses/evaluating-ai-agents))

**Read (about 1 hour):**

1. AI Evals for Everyone: model vs product evals, reference datasets, metrics, monitoring ([link](../free_courses/ai_evals_for_everyone/README.md))
2. **(Optional)** Securing Agentic AI Systems: what breaks once you ship ([link](../resources/securing_agentic_ai_systems.md))

---

## Optional Advanced Day: Fine-Tuning and Post-Training

Most builders reach for prompting, RAG, and agents long before they fine-tune. When you do need to change model behavior, this is the modern toolkit: supervised fine-tuning, preference optimization (DPO), and reinforcement learning on verifiable rewards (GRPO).

**Watch:** Post-training of LLMs by DeepLearning.AI ([link](https://www.deeplearning.ai/courses/post-training-of-llms))

**Read:** Fine-tuning 101 ([link](../resources/fine_tuning_101.md)) · PEFT methods by Hugging Face ([link](https://huggingface.co/blog/peft)) · the RLHF and post-training course by Nathan Lambert ([link](https://rlhfbook.com/course))

---

**Where to next:** browse [all free courses by topic](../courses.md), follow the monthly [best papers](../research_updates/2026_papers) and the [State of AI report](../research_updates/state_of_ai_2025_report/README.md), or pick a [learning path](../paths/agent-builder.md) and keep going.
