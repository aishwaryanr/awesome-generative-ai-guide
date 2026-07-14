# FDE Prep Plan

Concrete, day-by-day plans that sequence the [rounds](rounds.md), [questions](questions.md), [resources](resources.md), and [courses](courses.md) into a path. Two versions: a 3-week plan (about 2 to 3 hours per weekday) and a 1-week crunch plan.

**Guiding principle.** The FDE loop is roughly half technical and half judgment, and most candidates over-prepare coding and under-prepare the case, customer-simulation, and evaluation rounds. Weight your time toward the parts that decide the outcome: the ambiguous case round (highest weight, lowest pass rate), the customer simulation, and AI evaluation depth. Every session should end with something spoken aloud or written, not just read.

**Before you start.** Read the [README](README.md) and [rounds](rounds.md) once end to end so you know the shape of what you are preparing for. Pull the actual job description for your target company from [Palantir Careers](https://www.palantir.com/careers/) or the company's site, and note which rounds they run (see the company-specific shape section in [rounds.md](rounds.md)).

---

## Timeline

```text
3-week plan (about 2-3 hrs per weekday).

  Week 1 |== Foundations + technical fluency ===========|
  Week 2 |== Agents, evaluation, and system design ======|
  Week 3 |== Judgment, communication, behavioral, mocks ==| --> ONSITE
```

## 3-week plan

### Week 1: foundations and technical fluency

**Day 1: Map the role and self-assess.**
Read [README](README.md) and [rounds](rounds.md). Read the [Exponent 2026 FDE guide](https://www.tryexponent.com/blog/forward-deployed-engineer-interview-the-definitive-2026-guide-fde) and the [Perspective AI FDE guide](https://getperspective.ai/blog/forward-deployed-engineer-interview-questions-2026-prep-guide). Rate yourself 1 to 5 on each of the 10 [question themes](questions.md); the lowest two get extra time this week.

**Day 2: Foundations.**
Work the [Foundations topic](../../../topics/foundations.md). If shaky, read [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/). Answer the [GenAI foundations questions](questions.md#2-genai-foundations) out loud, then refine.

**Day 3: Prompting and context engineering.**
Do part of [ChatGPT Prompt Engineering for Developers](https://learn.deeplearning.ai/courses/chatgpt-prompt-eng/lesson/1/introduction) and read [Anthropic on context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Skim the [Prompt Engineering Guide](https://www.promptingguide.ai/).

**Day 4: RAG, part 1.**
Read the [RAG topic](../../../topics/rag.md) and [Agentic RAG 101](../../../resources/agentic_rag_101.md). Answer the [Retrieval and RAG questions](questions.md#3-retrieval-and-rag) aloud.

**Day 5: RAG, part 2 (hands-on).**
Build a small RAG pipeline over a folder of documents (chunk, embed, retrieve, answer, cite), following a notebook from [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques). Be able to defend your chunking choice with a retrieval metric.

**Weekend: coding warm-up.**
Do 4 to 6 practical problems: parse messy CSV or JSON, a rate limiter, exponential backoff with jitter, a small CLI. Use [NeetCode](https://neetcode.io/) only to remove rust, not to grind algorithms. Narrate aloud as if an interviewer were watching.

### Week 2: agents, evaluation, and system design

**Day 6: Agents fundamentals.**
Start the [Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/README.md) and read [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). Answer the agent fundamentals questions in the [Agents and MCP](questions.md#4-agents-and-mcp) theme aloud.

**Day 7: MCP and tool design.**
Read the [MCP introduction](https://modelcontextprotocol.io/introduction) and skim the [DeepLearning.AI MCP course](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/). Answer the MCP questions in the [Agents and MCP](questions.md#4-agents-and-mcp) theme. Sketch how you would expose one of a customer's systems as an MCP server.

**Day 8: Evaluation.**
Work through [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and read the [Evidently AI LLM guide](https://www.evidentlyai.com/llm-guide). Answer the [Evaluation questions](questions.md#5-evaluation) aloud. This is high-yield; give it real time.

**Day 9: Deployment, cost, and reliability.**
Read the [Production topic](../../../topics/production.md) and skim [LLM Numbers](https://github.com/ray-project/llm-numbers) and the relevant [Google SRE Book](https://sre.google/sre-book/table-of-contents/) chapters. Answer the [Deployment, cost, and reliability questions](questions.md#6-deployment-cost-and-reliability) aloud.

**Day 10: System design drills.**
Do 2 full system-design walkthroughs on a whiteboard, out loud, 45 minutes each, from the [rounds.md](rounds.md) design prompts (private VPC HIPAA RAG; 12 fragmented data sources; agent evaluation harness). Force yourself to start with an MVP walking skeleton and name trade-offs.

**Weekend: responsible AI and security, plus review.**
Read the [Safety and Security topic](../../../topics/safety-security.md) and [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). If targeting Anthropic, read [Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety) and the [Responsible Scaling Policy](https://www.anthropic.com/rsp). Answer the [Responsible AI and security questions](questions.md#7-responsible-ai-and-security). Re-answer any question you fumbled earlier in the week.

### Week 3: judgment, communication, behavioral, and mocks

**Day 11: The decomposition round, framework.**
Study the 5-step framework in [rounds.md](rounds.md) and read the [Palantir FDSE interview guide](https://blog.palantir.com/a-guide-to-palantir-forward-deployed-software-engineering-interviews-9c6ba9e07a4c). Do 1 full ambiguous case aloud from the [Case and decomposition](questions.md#9-case-and-decomposition-judgment) questions, narrating all 5 steps.

**Day 12: Decomposition, reps.**
Do 2 more full cases aloud (another from the [Case and decomposition](questions.md#9-case-and-decomposition-judgment) questions and one design-case prompt from [rounds.md](rounds.md)). Time yourself to 45 minutes. Record yourself and check: did you clarify before solving, surface assumptions, sequence by risk and value, propose a thin MVP, and name failure modes?

**Day 13: Customer simulation.**
Run the role-play scenarios in the [Customer communication and business](questions.md#10-customer-communication-and-business) questions. Have a friend play the frustrated or non-technical customer if possible. Drill: diagnose before prescribing, acknowledge before pushing back, offer options with trade-offs, never over-promise.

**Day 14: Behavioral stories.**
Write and rehearse 6 to 8 STAR stories covering the list in [rounds.md](rounds.md): end-to-end ownership, difficult stakeholder, reversed decision, alignment without authority, tight deadline with incomplete info, a real failure, a cross-customer pattern, saying no and holding the line. 60 to 90 seconds each, "I" not "we." Prepare a specific, evidence-backed "why this company."

**Day 15: Full mock loop.**
Simulate a compressed loop in one sitting: 1 coding problem, 1 system design, 1 decomposition case, 1 customer simulation, 3 behavioral questions. Get feedback or self-review against the "what good looks like" and "common mistakes" in [rounds.md](rounds.md).

**Weekend: targeted repair and rest.**
Re-drill only your weakest 2 rounds. Re-read the company-specific shape notes for your target. Skim your fumbled [questions.md](questions.md) answers once more. Rest the day before the interview; do not cram.

---

## 1-week crunch plan

For when the loop is days away. About 3 to 4 hours per day, weighted hard toward judgment rounds and evaluation.

- **Day 1: Orient and self-assess.** Read [README](README.md), [rounds](rounds.md), and both FDE guides ([Exponent](https://www.tryexponent.com/blog/forward-deployed-engineer-interview-the-definitive-2026-guide-fde), [Perspective AI](https://getperspective.ai/blog/forward-deployed-engineer-interview-questions-2026-prep-guide)). Read all of [questions.md](questions.md) once, marking weak answers.
- **Day 2: AI depth.** Speed-run the [RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), and [Evaluation](../../../topics/evaluation.md) topic pages plus [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents). Answer the [Retrieval and RAG](questions.md#3-retrieval-and-rag), [Agents and MCP](questions.md#4-agents-and-mcp), and [Evaluation](questions.md#5-evaluation) questions aloud. Do not skip evaluation.
- **Day 3: Coding and system design.** 3 practical coding problems (parse messy data, rate limiter, backoff), narrated aloud. Then 2 system-design walkthroughs from [rounds.md](rounds.md), starting each with an MVP skeleton and naming trade-offs.
- **Day 4: Decomposition cases.** The whole day on the signature round: study the 5-step framework, then do 3 full cases aloud from the [Case and decomposition](questions.md#9-case-and-decomposition-judgment) questions, recording and reviewing against "clarify before solving."
- **Day 5: Customer simulation and reliability.** Run all the role-plays in the [Customer communication and business](questions.md#10-customer-communication-and-business) questions. Then read the [Deployment, cost, and reliability](questions.md#6-deployment-cost-and-reliability) and [Responsible AI and security](questions.md#7-responsible-ai-and-security) questions and answer the deployment, cost, and responsible-AI ones aloud.
- **Day 6: Behavioral and company fit.** Write and rehearse 6 STAR stories, "I" not "we," 60 to 90 seconds each. Prepare a specific "why this company" grounded in the company's real work. If targeting Anthropic, read [Core Views on AI Safety](https://www.anthropic.com/news/core-views-on-ai-safety).
- **Day 7: One full mock, then rest.** Compressed mock loop (coding, design, case, simulation, behavioral), self-reviewed against [rounds.md](rounds.md). Light review of weak spots. Rest before the interview.

---

## The night before, and the interview itself

- Re-read the "common mistakes" and "what good looks like" bullets for each round in [rounds.md](rounds.md). They are the rubric.
- Internalize the one rule that fails the most candidates: **clarify and scope before you solve**, especially in the decomposition and design rounds. Silence and solution-jumping both lose points; narrate continuously.
- Use "I" not "we." Own your decisions and their trade-offs.
- Never over-promise in the customer simulation. Ownership language plus honest options beats confident guarantees.
- Have 2 or 3 sharp, specific questions ready for each interviewer about their customers and deployments; it signals genuine interest, which Palantir and Anthropic screen for.

Back to the [FDE README](README.md).
