# AI Engineer: The Interview Rounds

Every round an AI Engineer loop runs, in the order you usually hit them. For each: what it tests, how it is run, what "good" looks like, common mistakes, and realistic example prompts. Study the round you are about to face, then drill the matching themes in [questions.md](questions.md).

Jump to: [Recruiter screen](#1-recruiter-screen) - [Technical / coding](#2-technical--coding-round) - [Take-home](#3-take-home) - [System design](#4-llm--ml-system-design) - [Behavioral](#5-behavioral) - [Hiring manager](#6-hiring-manager)

---

## 1. Recruiter screen

**What it tests.** Basic fit: are you real, are you at the right level, do your motivation and comp expectations line up, and can you tell your project story in plain language. No deep technical grilling, but a vague or buzzword-heavy answer here can end the loop.

**How it is run.** 25-35 minutes by phone or video with a recruiter or sourcer. They walk your resume, ask what you have built, why this company, your timeline, comp range, and work authorization. They are scoring for signal they can pass to the hiring manager.

**What good looks like.**
- A 60-90 second story of one LLM system you shipped: the problem, what you built, and one concrete result or lesson.
- You can say plainly what an AI Engineer does and why you want *this* role (not just "AI is exciting").
- A comp range and timeline given without flinching.

**Common mistakes.**
- Reciting a tech stack instead of an outcome ("I used LangChain and Pinecone" with no problem or result).
- Being unable to name a single thing that went wrong and what you did about it.
- Over- or under-leveling yourself so the recruiter routes you to the wrong loop.

**Example prompts.**
- "Walk me through a recent project where you used LLMs. What was your specific contribution?"
- "Why do you want to work on applied AI here rather than at a big lab?"
- "What are you looking for in your next role, and what is your comp range?"
- "How much of your recent work has been hands-on building versus research or analysis?"

---

## 2. Technical / coding round

**What it tests.** Practical Python plus hands-on LLM coding. This is not LeetCode. It is: can you call a model API, parse and validate structured output, wire up a small retrieval or tool-use loop, and reason about correctness, cost, and latency while you type. Some big-tech loops still include one moderate data-structures problem, so keep basic Python (dicts, sets, string parsing, recursion) sharp, but the center of gravity is LLM plumbing.

**How it is run.** 45-60 minutes, live, in a shared editor or a notebook, sometimes with internet and real API access, sometimes with a stubbed or mocked client. You are expected to think aloud, handle edge cases (empty retrieval, malformed JSON, an API error), and write code that a colleague could read.

**What good looks like.**
- You clarify the contract first: inputs, expected output shape, what "correct" means, and failure behavior.
- You reach for structured output (JSON schema or tool-calling) instead of regex-parsing free text, and you validate it.
- You handle the unhappy path: retries with backoff, a fallback when retrieval is empty, a cap on loop iterations, an "I don't know" path.
- You mention cost and latency naturally (batching, caching a stable prefix, picking a smaller model for a sub-step).
- Clean, testable code, and you write or sketch at least one test.

**Common mistakes.**
- Parsing model output with brittle string slicing instead of asking for structured output.
- No timeout, retry, or error handling on API calls.
- An agent loop with no maximum iteration cap.
- Ignoring the empty-retrieval and unanswerable cases entirely.
- Silence: not narrating tradeoffs as you go.

**Example prompts.**
- "Given a list of documents, write a function that chunks them, embeds the chunks (client provided), retrieves the top k for a query, and returns an answer with citations. Handle the case where nothing relevant is found."
- "Implement a function that asks the model to extract `{name, date, amount}` from an invoice string and returns validated JSON. What do you do when the model returns malformed output?"
- "Write a minimal tool-use loop: the model can call `search(query)` and `finish(answer)`. Cap it at 5 steps and handle a tool error."
- "Here is a prompt that sometimes returns prose and sometimes JSON. Make its output reliable."
- "Implement a simple semantic cache so repeated near-duplicate queries don't hit the model twice."
- "Write a token-budget trimmer that fits a system prompt, chat history, and retrieved chunks into a context limit, dropping the least important content first."

---

## 3. Take-home

Common at startups; sometimes replaces the live coding round, sometimes adds to it. The single most predictive round for this role, because it mirrors the actual job.

**What it tests.** End-to-end building judgment. RAG systems are the most common assignment (40%+ of take-homes). Can you ingest data, build retrieval, ground answers with citations, refuse when the answer is not in the data, *and* measure quality? Evaluation is where most candidates are thin and where you can stand out.

**How it is run.** Asynchronous, 2-7 day deadline, though most reviewers expect 4-8 focused hours of work, not a magnum opus. You submit code plus a short writeup, and there is usually a follow-up call where you demo it and defend every decision. Deployment maturity (Docker, a simple CI check, basic monitoring) reads far better than a raw notebook.

**What good looks like.**
- A working system that does the core task and correctly says "I don't have that information" when the answer is not retrievable.
- An eval set (even 15-30 hand-labeled questions split into answerable, partially answerable, and unanswerable) plus a script that reports faithfulness, answer relevance, and retrieval hit rate.
- A README that states your assumptions, your architecture, what you would do with more time, and the known failure modes. This writeup often matters more than the last 10% of polish.
- Sensible defaults with reasons: your chunk size, your k, your model choice, why you rerank or not.
- Clean git history, a one-command run, and no secrets committed.

**Common mistakes.**
- Shipping a demo with zero evaluation. This is the most common reason strong-looking take-homes get rejected.
- Over-engineering (a 4-agent framework for a task a single retrieval call would solve).
- No handling of the unanswerable case, so the system confidently hallucinates.
- A writeup that lists what you built but never names a tradeoff or a limitation.
- Burning all the time on a fancy UI and none on correctness or evals.

**Example prompts.**
- "Build a RAG chatbot over this set of PDFs. It must answer with citations and refuse when the answer is not in the documents. Include an evaluation of its quality."
- "Given this support-ticket dataset, build a system that drafts a reply grounded in our help docs. Show how you would measure whether the drafts are good."
- "Build an agent that answers questions about a company's public filings using a provided search tool. Report where it fails."
- "Here is a flaky prompt pipeline. Improve its reliability and prove the improvement with an eval."

---

## 4. LLM / ML system design

**What it tests.** Can you architect an LLM-powered system out loud and defend the tradeoffs. The interviewer wants to hear you reason about retrieval versus fine-tuning, the agency-versus-control tradeoff, evaluation as release infrastructure, and the cost-latency-quality triangle. Seniority shows in how you handle scale, failure modes, and "how does this break."

**How it is run.** 45-60 minutes, whiteboard or shared doc. Open-ended prompt, you drive. Strong interviewers deliberately add constraints mid-round (10x the traffic, a strict latency budget, a new compliance rule, a jump in hallucination reports) to see you adapt.

**What good looks like.**
- You start with requirements: accuracy bar, latency budget, cost ceiling, scale, safety and escalation policy. You do not jump to a vector DB in the first 30 seconds.
- You sketch the pipeline (ingestion and freshness, chunking, embedding, retrieval, rerank, prompt assembly, generation, guardrails, response) and justify each choice.
- You name the cheapest thing that clears the bar first, then add complexity only when a requirement forces it.
- You treat evaluation as a first-class part of the design: what you measure offline, what you monitor online, and how the eval gates a rollout.
- You quantify: rough token counts, an approximate cost per query, a latency budget split across stages, a caching strategy.
- You volunteer failure modes (irrelevant retrieval, stale data, prompt injection from retrieved content, an agent looping) and their mitigations.

**Common mistakes.**
- Jumping to a solution before you know the requirements.
- Treating "add RAG" or "make it an agent" as the answer without justifying it.
- No evaluation or monitoring story.
- Hand-waving cost and latency instead of putting numbers on them.
- Designing for infinite scale when the prompt implies a small internal tool, or vice versa.
- Ignoring access control and freshness at enterprise scale, where the data layer, not the model, is the real bottleneck.

**Example prompts.**
- "Design a customer support agent for an enterprise. It reads our knowledge base, can look up orders and file tickets, and escalates to a human when unsure. Walk me through it."
- "Design a RAG system to answer employee questions over 10 million internal documents with per-user access control. How do you keep it fresh, fast, and cheap?"
- "Design an insurance-claims agent that ingests a claim and outputs an approve / deny / needs-review decision. How do you control cost and add guardrails?"
- "Design the evaluation system for an LLM feature that is already in production and getting hallucination complaints. What do you build?"
- "Design a code-assistant that answers questions about a large private codebase. Latency budget is 2 seconds."
- "We are getting one reasoning-model call per request and latency is too high. Redesign to cut p95 latency in half without tanking quality."

---

## 5. Behavioral

**What it tests.** How you actually work: ownership, dealing with ambiguity and non-deterministic systems, handling a production incident, collaborating with PMs and researchers, and disagreeing well. AI-specific twist: they probe how you handle the messiness unique to LLM products (a model that regresses after a provider update, a hallucination that reached a user, a demo that worked but the eval said ship-no).

**How it is run.** 45-60 minutes, STAR-style ("tell me about a time..."). Often one round, sometimes folded into the hiring-manager round. Answers should be specific, first-person, and honest about what went wrong.

**What good looks like.**
- Concrete stories with your specific actions and a measurable or clear outcome, using situation-task-action-result structure without sounding scripted.
- At least one real failure with a genuine lesson, ideally about an AI system misbehaving in production.
- Evidence you make decisions with data (an eval, a metric) rather than vibes, and that you can say "the demo looked great but the numbers said no."
- Clear collaboration: how you turned a fuzzy PM request into a shippable spec, or aligned with a researcher on an eval.

**Common mistakes.**
- Vague "we" stories where your own contribution is invisible.
- A fake weakness or a failure with no real lesson.
- Blaming the model, the data, or a teammate for everything.
- No story that involves measuring or monitoring, which for this role reads as "does not evaluate."

**Example prompts.**
- "Tell me about an AI feature you shipped that did not work as expected in production. What happened and what did you do?"
- "Describe a time you had to say no to shipping something that looked impressive in a demo."
- "Tell me about a disagreement with a PM or researcher over an AI feature. How did you resolve it?"
- "Give an example of a time you had very ambiguous requirements. How did you make progress?"
- "Tell me about the hardest debugging problem you have had with an LLM system."

---

## 6. Hiring manager

**What it tests.** The manager's own read on judgment, ownership, and team fit, plus a synthesis of everything above. Expect a mix: a lighter technical or design discussion, deeper behavioral and motivation questions, a "what would you do in your first 90 days" style probe, and real time for your questions (which are scored).

**How it is run.** 45-60 minutes with the person you would report to. More conversational than the earlier rounds. They are deciding whether they want you on the team and whether you would raise the bar.

**What good looks like.**
- You connect your experience to *their* specific problems (you researched the product and can talk about it concretely).
- You show taste: you can articulate when *not* to use AI, when a simpler approach wins, and how you would de-risk an ambiguous project.
- Thoughtful questions about the roadmap, the eval and quality bar, how the team ships, and what success looks like at 6 months.
- Honest calibration about what you are strong at and what you would need to ramp on.

**Common mistakes.**
- No informed questions, or only comp-and-perks questions.
- Overclaiming and getting caught when the manager probes a detail.
- Not being able to say why this company over the others you are interviewing at.
- Treating it as a pure Q&A instead of a two-way conversation.

**Example prompts.**
- "What would you focus on in your first 90 days here?"
- "Tell me about a technical decision you now think was wrong. What would you do differently?"
- "How do you decide whether an AI feature is good enough to ship?"
- "What kind of team and manager do you do your best work with?"
- "What questions do you have for me?"

---

## Round-to-theme map

Drill the matching sections in [questions.md](questions.md) before each round:

- Coding round: LLM fundamentals, prompting and structured output, RAG mechanics, agents and tool use.
- Take-home: RAG, evaluation, production and cost/latency.
- System design: RAG at scale, agents, evaluation, production, safety and security, cost/latency.
- Behavioral and hiring manager: the judgment and business/strategy questions at the end of the bank.

---

Next: the full [question bank](questions.md). Back to the [role README](README.md).
