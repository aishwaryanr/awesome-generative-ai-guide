# AI Engineer: Prep Plan

Two structured plans that sequence the rounds, questions, resources, and courses in this folder into a day-by-day path. Use the 3-week track if you have runway; use the 1-week crunch if the onsite is close. Both are built around one principle for this role: **build one real thing and evaluate it**, because that single project answers most coding, take-home, system-design, and behavioral questions at once.

Jump to: [The anchor project](#the-anchor-project) - [3-week track](#3-week-track) - [1-week crunch](#1-week-crunch) - [Day-of checklist](#day-of-checklist)

---

## Timeline

```text
3-week track (10-15 hrs/week). Everything hangs off one anchor project you build and evaluate.

  Week 1 |== Fundamentals + build the anchor project skeleton ==|
  Week 2 |== Agents, evaluation: add a tool step + an eval set ==|
  Week 3 |== System design, timed mocks, polish + demo =========| --> ONSITE
```

## The anchor project

Before either plan, commit to building one production-shaped project and finishing it. The highest-signal choice is a **RAG app with evaluation and a small agent layer**:

- Ingest a real document set, chunk and embed it, and answer questions with citations.
- Refuse ("I do not have that information") when the answer is not retrievable.
- Add an eval set of 20 to 30 hand-labeled questions (answerable, partially answerable, unanswerable) and a script that reports retrieval hit rate, faithfulness, and answer relevance.
- Add one tool-using step (a search or lookup tool) with an iteration cap and error handling.
- Wrap it: a one-command run, a Dockerfile, and a README stating your assumptions, tradeoffs, and known failure modes.

You will reference this project in the behavioral and hiring-manager rounds, reuse it as your take-home baseline, and mine it for system-design answers. Build it once, use it everywhere.

---

## 3-week track

Assumes roughly 10 to 15 hours per week. Read [rounds.md](rounds.md) once up front so you know what each week is aiming at.

### Week 1: Fundamentals and the anchor project skeleton

- **Day 1:** Read [README.md](README.md) and [rounds.md](rounds.md). Skim the [Foundations topic page](../../../topics/foundations.md). Answer [questions.md](questions.md) 1 to 9 (LLM fundamentals) out loud; note what you cannot explain.
- **Day 2:** Prompting and context engineering: questions 10 to 16, plus [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and the [Prompt Engineering Guide](https://www.promptingguide.ai/). Practice getting reliable structured output from an API.
- **Day 3:** Stand up the anchor project skeleton: ingestion, chunking, embedding, a basic retrieve-then-generate loop with citations. No evals yet.
- **Day 4:** RAG mechanics: questions 17 to 24. Read the [RAG survey](https://arxiv.org/abs/2312.10997) intro and the [RAG evaluation guide](https://www.evidentlyai.com/llm-guide/rag-evaluation). Add the refusal path to your project.
- **Day 5:** RAG at scale and failure modes: questions 25 to 29. Add hybrid retrieval or a reranker to the project and note the quality change.
- **Weekend:** Work the [Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/README.md) parts 1 to 5. Rest one day.

### Week 2: Agents, evaluation, and depth

- **Day 6:** Agents and tool use: questions 30 to 39. Read [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (workflows versus agents) and [Model Context Protocol](https://modelcontextprotocol.io/introduction). Finish the Agentic AI Crash Course (parts 6 to 10).
- **Day 7:** Add the tool-using step to your anchor project with an iteration cap and error handling. Keep it simple; resist over-engineering.
- **Day 8:** Evaluation: questions 40 to 46. Work through [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and the [Ragas docs](https://docs.ragas.io/en/stable/).
- **Day 9:** Build the eval set and script for your project (retrieval hit rate, faithfulness, answer relevance, refusal rate). This is the differentiator; spend real time here.
- **Day 10:** Reasoning models and cost/latency: questions 47 to 55. Read [Agents](https://huyenchip.com/2025/01/07/agents.html) and [Common pitfalls](https://huyenchip.com/2025/01/16/ai-engineering-pitfalls.html). Add caching and a token-budget trimmer to your project.
- **Weekend:** Safety and security: questions 56 to 61, plus the [Safety topic page](../../../topics/safety-security.md) and [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Add basic input/output guardrails. Rest one day.

### Week 3: System design, mocks, and polish

- **Day 11:** System design: questions 62 to 66. Re-read the system-design section of [rounds.md](rounds.md). Do one design out loud on a whiteboard, timed to 45 minutes.
- **Day 12:** Two more timed system-design drills (enterprise RAG with access control, and an agent with a latency budget). Record yourself; check you started from requirements and quantified cost and latency.
- **Day 13:** Business and judgment: questions 67 to 71. Prepare 5 behavioral stories in situation-task-action-result form, at least one about an AI system failing in production, drawn from your anchor project and past work.
- **Day 14:** Coding round practice: implement, from scratch and timed, a chunk-retrieve-cite function, a structured-extraction function with validation, and a capped tool loop. Handle every unhappy path.
- **Day 15:** Polish the anchor project README and demo. Do a full mock loop with a friend or out loud: coding, system design, behavioral.
- **Day 16:** Review your weak areas from the mocks. Re-answer any [questions.md](questions.md) items you fumbled. Prepare questions for the hiring manager.
- **Buffer:** Skim [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and the [State of AI 2025 Report](../../../research_updates/state_of_ai_2025_report/README.md) to catch anything current you missed.

---

## 1-week crunch

Assumes the onsite is days away and you can commit 4 to 6 focused hours per day. Prioritize ruthlessly: breadth of answers plus one demonstrable project beats deep study of any single topic.

- **Day 1:** Read [README.md](README.md) and [rounds.md](rounds.md). Work [questions.md](questions.md) 1 to 29 (fundamentals, prompting, RAG) out loud. If you have no project to show, start a minimal RAG app today: ingest, retrieve, cite, refuse.
- **Day 2:** [questions.md](questions.md) 30 to 46 (agents, tool use, evaluation). Read [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) and skim the [RAG evaluation guide](https://www.evidentlyai.com/llm-guide/rag-evaluation). Add a tiny eval (even 10 labeled questions) to your app.
- **Day 3:** [questions.md](questions.md) 47 to 61 (reasoning models, cost/latency, safety). Read [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Add caching, an iteration cap, and one guardrail to your app.
- **Day 4:** System design: questions 62 to 66. Do two timed 45-minute designs out loud (a support agent, and enterprise RAG with access control). Force yourself to start from requirements and put numbers on cost and latency.
- **Day 5:** Behavioral and judgment: questions 67 to 71. Write and rehearse 4 to 5 stories, one about an AI failure in production. Finalize your project demo and a clear README.
- **Day 6:** Full mock loop out loud: one coding problem (chunk-retrieve-cite or structured extraction with validation), one system design, one behavioral. Fix the biggest gap the mock exposes.
- **Day 7 (light):** Re-skim your weak [questions.md](questions.md) sections and the [60 GenAI Interview Questions](../../60_gen_ai_questions.md). Prepare questions for the hiring manager. Rest.

---

## Day-of checklist

- Can you state, in one sentence each, when to use prompting vs RAG vs fine-tuning vs an agent? (Questions 17, 30, 67.)
- Can you describe your anchor project's architecture and one tradeoff you would change?
- Can you name the RAG triad and how you would evaluate a RAG system? (Questions 23, 24.)
- Do you have a number ready for a rough cost per query and a latency budget split? (Questions 50, 51.)
- Do you have one production-failure story with a real lesson?
- For coding: default to structured output, handle the empty-retrieval and malformed-output cases, and cap every loop.
- For system design: requirements first, simplest approach that clears the bar, evaluation and monitoring as part of the design, then failure modes.
- Have 3 informed questions ready for the hiring manager about their roadmap, quality bar, and how they ship.

---

Back to the [role README](README.md). Question bank: [questions.md](questions.md).
