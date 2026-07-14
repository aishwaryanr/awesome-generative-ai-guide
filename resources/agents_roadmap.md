# 5-Day AI Agents Roadmap

Agents are how most real LLM applications get built now: a model wrapped in a harness of tools, memory, and a loop, so it can take actions and not just answer. This roadmap takes you from "what is an agent" to a working, evaluated agent in 5 days, using free resources and materials from this repository. Plan for about 2 to 3 hours a day.

![agent_roadmap_image.gif](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/agent_roadmap_image.gif)

> Updated for 2026. Agents changed more than any other area since 2024: reasoning models, the Model Context Protocol, agent harnesses, and real evaluation are now the core, so this roadmap is built around them.

## Day 1: What Is an Agent

The mental model first: an agent is a model plus a harness. Understand the loop before the tools.

**Watch (about 1 hour):**

1. AI Agents Course, unit 0, by Hugging Face ([link](https://huggingface.co/learn/agents-course/unit0/introduction))

**Read (about 1 hour):**

1. Agentic AI Crash Course, part 1: what are AI agents ([link](../free_courses/agentic_ai_crash_course/part1_what_are_ai_agents_anyway.md))
2. Agentic AI Crash Course, part 2: the 4 types of agentic systems ([link](../free_courses/agentic_ai_crash_course/part2_the_4_types_of_agentic_systems.md))
3. Agents 101 guide ([link](../resources/agents_101_guide.md))

---

## Day 2: Tools and Function Calling

Tools are what turn a model into an agent. Learn how tool use and function calling work.

**Watch (about 1 hour):**

1. Functions, Tools and Agents with LangChain by DeepLearning.AI ([link](https://learn.deeplearning.ai/functions-tools-agents-langchain))

**Read (about 1 hour):**

1. Agentic AI Crash Course, part 3: what are tools in AI ([link](../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md))

---

## Day 3: Context, Memory, and Planning

Long tasks need context management, memory, and planning. Reasoning models changed how agents plan.

**Read (about 1.5 hours):**

1. Agentic AI Crash Course, part 6: planning in agents and reasoning models ([link](../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md))
2. Agentic AI Crash Course, part 7: memory in agents ([link](../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md))
3. Agentic AI Crash Course, part 4: RAG and agentic RAG ([link](../free_courses/agentic_ai_crash_course/part4_what_is_rag_and_agentic.md))

---

## Day 4: MCP and Multi-Agent Systems

The Model Context Protocol standardized how agents connect to tools and data. Multi-agent systems let agents specialize and collaborate.

**Watch (about 1 hour):**

1. MCP Course by Hugging Face and Anthropic ([link](https://huggingface.co/learn/mcp-course/en/unit0/introduction))

**Read (about 1 hour):**

1. Agentic AI Crash Course, part 5: what is MCP and why care ([link](../free_courses/agentic_ai_crash_course/part5_what_is_mcp_and_why_care.md))
2. Agentic AI Crash Course, part 8: multi-agent systems ([link](../free_courses/agentic_ai_crash_course/part8_multi_agent_systems.md))

---

## Day 5: Evaluate and Ship Agents

Agents fail in ways tests can't predict. Evaluation, observability, and security are what separate a demo from a shipped system.

**Watch (about 1 hour):**

1. Evaluating AI Agents by DeepLearning.AI and Arize ([link](https://www.deeplearning.ai/courses/evaluating-ai-agents))

**Read (about 1.5 hours):**

1. AI Evals for Everyone ([link](../free_courses/ai_evals_for_everyone/README.md))
2. Agentic AI Crash Course, part 9: real-world agentic systems ([link](../free_courses/agentic_ai_crash_course/part9_real_world_agentic_systems.md))
3. Securing Agentic AI Systems ([link](../resources/securing_agentic_ai_systems.md))

---

## Keep going

- The living [Agentic Search and Retrieval research table](../research_updates/agentic_search_retrieval_table.md), updated regularly.
- Agentic AI Crash Course, part 10: lessons and what's ahead ([link](../free_courses/agentic_ai_crash_course/part10_ai_agent_lessons_whats_ahead.md)).
- The hands-on [Agentic AI notebooks](../resources/agentic_ai_course_lil/README.md) for action and planning autonomy.
- **(Foundational anchor)** LLM Powered Autonomous Agents by Lilian Weng ([link](https://lilianweng.github.io/posts/2023-06-23-agent/)), a 2023 classic that still holds up.
