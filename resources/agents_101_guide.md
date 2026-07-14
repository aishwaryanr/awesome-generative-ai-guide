# LLM Agents 101

![llm_guide.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/llm_guide.png)

## Introduction to LLM Agents

An LLM agent is a large language model given the ability to act, not just answer. The clearest way to think about it: **an agent is a model plus a harness.** The model is the reasoning core. The harness is everything you build around it so it can take real actions: tools, memory, and a loop that lets it try, observe the result, and try again until the task is done.

Imagine you're building an assistant that plans vacations. It can answer simple questions like "What's the weather in Paris next week?" from a single call. But a real request looks like "Plan a 10-day Europe trip next summer with historic landmarks, local food, and a $3000 budget." That needs planning, budgeting, and looking things up across many sources. An agent handles it by using the model to reason and plan, calling tools to search flights, hotels, and attractions, and keeping track of the budget and preferences in memory across many steps. The model is the brain, the harness is what lets it get the job done.

## What's Inside an Agent: The Harness

A widely used way to break down the harness is into four parts.

![Screenshot 2024-04-07 at 2.53.23 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_2.53.23_PM.png)

Image Source: [Introduction to LLM Agents, Nvidia](https://developer.nvidia.com/blog/introduction-to-llm-agents/)

1. **Agent core (the brain):** the central decision loop. It holds the agent's goal, decides which tool to use and when, pulls in relevant memory, and often carries a persona or set of operating rules. With reasoning models (the o-series, DeepSeek-R1 and successors), a lot of the planning that used to be hand-built now happens inside the model's own chain of thought.
2. **Memory:** where the agent keeps state. Short-term memory holds the current task's working context; long-term memory holds facts and history across sessions, usually retrieved by semantic similarity plus signals like recency and importance. Managing what goes into the context window, and what stays out, is a core skill now often called context engineering.
3. **Tools:** the actions the agent can take. Tools range from web search and code execution to retrieval (RAG) and any API. The **Model Context Protocol (MCP)** has become the common standard for connecting an agent to tools and data sources, so you wire a capability once and reuse it across agents.
4. **Planning:** how the agent breaks a hard task into steps, critiques its own work, and decides what to do next. Task decomposition and reflection are the staples; reasoning models made this dramatically more capable.

An earlier survey framed the same idea as brain, perception, and action. It is another useful lens on the same structure.

![Screenshot 2024-04-07 at 2.39.12 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_2.39.12_PM.png)

Image Source: [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)

## Multi-Agent Systems

A single agent works in isolation. A multi-agent system splits a problem across several specialized agents that collaborate, on the principle of division of labor: give each agent a focused role and let them coordinate. This can improve both efficiency and quality on complex tasks, and it is how many production systems now handle work that is too big for one context window.

![Screenshot 2024-04-07 at 3.03.48 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_3.03.48_PM.png)

Image source: [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)

Interactions are usually **cooperative** (agents share information and build on each other's work, either freely or in a defined order) or **adversarial** (agents debate or critique each other to sharpen the answer). Multi-agent systems add real cost and coordination overhead, so reach for them when a single agent genuinely can't hold the task, not by default.

## Agents in the Real World

The 2023 wave of task-loop demos (BabyAGI and friends) proved the idea. What actually shipped in 2025 and 2026 looks different, and it is built on strong agent harnesses:

- **Coding agents:** Claude Code, Codex, and Cursor are agent harnesses that plan, edit files, run code, and verify, driving real software work.
- **Computer-use and browser agents:** agents that operate a screen or browser to complete tasks across apps.
- **Deep research agents:** agents that plan a research question, search and read across the web, and synthesize a cited report.
- **Enterprise workflow agents:** customer support, operations, and analysis agents running in production, watched by observability and evaluation.

The pattern behind all of them is the same: a capable model, a well-designed harness, and a loop that survives long, autonomous tasks.

## Evaluating Agents

Agents fail in ways unit tests can't predict, so evaluation is its own discipline. A useful principle: **evaluate the whole system, not just the model.** A better base model does not fix a broken harness, a missing tool, or context the agent never sees.

Modern agent evaluation combines task benchmarks (for example tau-bench and similar multi-turn, tool-use suites) with evals you build around how your own system breaks, plus production observability to catch failures live. Useful dimensions to score include:

- **Utility:** does it complete the task, and how efficiently (success rate, cost, steps).
- **Reliability and robustness:** does it hold up under messy inputs and adversarial cases.
- **Safety and trustworthiness:** does it stay within guardrails, avoid harmful actions, and behave predictably when given real-world autonomy.

For a full treatment, see the [AI Evals for Everyone](../free_courses/ai_evals_for_everyone/README.md) course, and [Securing Agentic AI Systems](../resources/securing_agentic_ai_systems.md) for what breaks once agents can act.

## Build Your Own Agent

Now that you understand how agents work, here are strong, current resources to build one:

1. [AI Agents Course](https://huggingface.co/learn/agents-course/unit0/introduction) by Hugging Face, hands-on and certified.
2. The [Agentic AI Crash Course](../free_courses/agentic_ai_crash_course/README.md) in this repository: agents, tools, RAG, MCP, planning, memory, and multi-agent, in 10 parts.
3. [Functions, Tools and Agents with LangChain](https://learn.deeplearning.ai/functions-tools-agents-langchain) by DeepLearning.AI.
4. [Introduction to the Model Context Protocol](https://anthropic.skilljar.com/introduction-to-model-context-protocol) by Anthropic.
5. Follow the [5-Day AI Agents Roadmap](../resources/agents_roadmap.md) for a structured path.

## References

1. [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)
2. [Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/pdf/2402.01680.pdf)
3. [Introduction to LLM Agents](https://developer.nvidia.com/blog/introduction-to-llm-agents/), Nvidia
4. The living [Agentic Search and Retrieval research table](../research_updates/agentic_search_retrieval_table.md) in this repository, updated regularly.
