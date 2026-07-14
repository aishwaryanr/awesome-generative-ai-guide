# Harness Engineering

**Who it's for:** builders who want to go past *using* an agent to *assembling* one.

The idea in one line: **an agent is a model plus a harness.** The model reasons; the harness is everything you build around it so it can act, prompting, context, tools, memory, retrieval, the loop, and verification. This path takes you from using pre-built harnesses to assembling and evaluating your own.

Start with the **[Harness Engineering 101 guide](../resources/harness_engineering.md)** for the full theory (what a harness is, the maturity ladder, the agency-control tradeoff, context rot, verification loops, the lethal trifecta), then use this path to go deeper.

`Level 🟡→🔴 · Source ⭐ LevelUp Labs original / 🌐 External`

---

## 1. The mental model: model + harness

- **[Agentic AI Crash Course, part 1](../free_courses/agentic_ai_crash_course/part1_what_are_ai_agents_anyway.md)** and **[part 2](../free_courses/agentic_ai_crash_course/part2_the_4_types_of_agentic_systems.md)** ⭐ 📖: what an agent is, and the 4 types of agentic systems.
- **[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)** 🌐 📖 by Anthropic: the canonical read on when to use a simple workflow versus a full agent, and the common patterns.

## 2. Use a harness (operator skill)

Learn the leading harnesses before you build one.

- **[Claude Code in Action](https://anthropic.skilljar.com/claude-code-in-action)** 🌐 🎥 by Anthropic Academy: agentic coding, explore-plan-code-commit.
- **[Learn Cursor](https://cursor.com/learn)** 🌐 🎥 by Cursor.
- **[OpenAI Codex Essentials](https://www.freecodecamp.org/news/openai-codex-essentials-ai-assisted-agentic-development-course/)** 🌐 🎥 by freeCodeCamp.
- **[Claude Code Best Practices](https://www.anthropic.com/engineering/claude-code-best-practices)** 🌐 📖 and the **[Claude Code docs](https://docs.claude.com/en/docs/claude-code/overview)** 🌐 📖 by Anthropic.

## 3. Assemble a harness (builder skill)

The pieces of a harness, one at a time. All from the [Agentic AI Crash Course](../free_courses/agentic_ai_crash_course/README.md).

- **Tools and function calling** → [part 3](../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md) ⭐
- **Retrieval as a tool** → [part 4](../free_courses/agentic_ai_crash_course/part4_what_is_rag_and_agentic.md) ⭐ and [Agentic RAG 101](../resources/agentic_rag_101.md) ⭐
- **Planning and reasoning models** → [part 6](../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md) ⭐
- **Memory** → [part 7](../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md) ⭐
- **MCP** → [part 5](../free_courses/agentic_ai_crash_course/part5_what_is_mcp_and_why_care.md) ⭐ and Anthropic's **[Introduction to MCP](https://anthropic.skilljar.com/introduction-to-model-context-protocol)** 🌐 🎥
- **Multi-agent and sub-agents** → [part 8](../free_courses/agentic_ai_crash_course/part8_multi_agent_systems.md) ⭐

## 4. Evaluate and ship the harness

A harness is only as good as your ability to tell it is working, and to catch it when it breaks.

- **[AI Evals for Everyone](../free_courses/ai_evals_for_everyone/README.md)** ⭐ 📖 (certified): model vs product evals, reference datasets, monitoring.
- **[Securing Agentic AI Systems](../resources/securing_agentic_ai_systems.md)** ⭐ 📖: what breaks once the agent can act.
- **[Agentic AI Crash Course, part 9](../free_courses/agentic_ai_crash_course/part9_real_world_agentic_systems.md)** and **[part 10](../free_courses/agentic_ai_crash_course/part10_ai_agent_lessons_whats_ahead.md)** ⭐: real-world systems and where this is heading.

---

**You can now:** use an agent harness fluently, assemble your own from tools, memory, context, and a loop, and evaluate it. Keep going with the [AI Agents topic](../topics/agents.md), the [Agent Builder path](agent-builder.md), and the living [Agentic Search and Retrieval research table](../research_updates/agentic_search_retrieval_table.md).
