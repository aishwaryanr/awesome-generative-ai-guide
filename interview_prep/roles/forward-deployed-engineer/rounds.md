# FDE Interview Rounds, Round by Round

Every round an FDE loop runs, with what it tests, how it is run, what "good" looks like, the common mistakes, and realistic example prompts. Order and naming vary by company; see the company notes at the end.

A loop typically has 5 to 8 stages over 3 to 6 weeks. Roughly half the loop is technical (coding, design, AI depth) and half is judgment (case, customer simulation, behavioral). Prepare both halves.

---

## The interview loop

```text
 [1] Recruiter -> [2] Hiring mgr -> [3] Coding -> [4] System design -> [5] Ambiguous case -> [6] Client sim -> [7] Values -> OFFER
     fit            motivation &     build          architecture         DECOMPOSE a vague     customer         behavioral
                    scope            fast           under constraints    problem (signature)   communication
```

Round 5, decomposing a vague business problem, is the signature round for this role.

## 1. Recruiter screen

**What it tests:** Motivation, role fit, communication baseline, and logistics (timeline, location, compensation range). Whether you understand what an FDE actually does.

**How it is run:** About 30 minutes, video or phone, with a recruiter. Conversational. Roughly an 80 percent pass rate; it filters out obvious mismatches.

**What good looks like:**
- A crisp, specific reason you want FDE work, not generic enthusiasm for the company. Name the customer-facing, own-the-outcome nature of the role.
- Evidence you know the company's customers and recent deployments.
- A clear, short walk-through of your background aimed at customer-facing engineering.

**Common mistakes:**
- Generic "I love the mission" answers with nothing specific.
- Describing yourself as a pure backend or research engineer with no interest in customers.
- Not having a coherent one-line answer to "what does an FDE do day-to-day?"

**Example prompts:**
- "Walk me through your background in 3 minutes."
- "Why FDE specifically, and not a standard software engineering role?"
- "What do you think an FDE does day-to-day?"
- "Which of our customers or use cases are you most interested in, and why?"
- "What are you looking for in your next role?"

---

## 2. Hiring-manager screen

**What it tests:** Depth of your past work, ownership, and technical judgment. Whether you led or followed. Whether you can tell a technical story clearly.

**How it is run:** 45 to 60 minutes with the hiring manager, often before or after the coding round. A deep dive on one or two projects plus judgment questions.

**What good looks like:**
- "I" language that makes your individual contribution unmistakable, even on team projects.
- A project story with real constraints, trade-offs you chose, and what you would do differently.
- Judgment about sequencing: why you built X first and deferred Y.

**Common mistakes:**
- "We" language that hides what you personally did.
- A tour of technologies with no decisions or trade-offs.
- No honest reflection on what went wrong.

**Example prompts:**
- "Tell me about the most technically challenging thing you have shipped end to end."
- "Walk me through a deployment or launch that did not go well. What did you do?"
- "How did you decide what to build first on that project?"
- "Tell me about a time you owned something from a vague ask all the way to production."
- "Where did you disagree with a stakeholder, and how did it resolve?"

---

## 3. Coding round

**What it tests:** Practical engineering: parsing and transforming messy real-world data, integrating flaky external systems, writing clean and tested code, and narrating your thinking. This is deliberately not LeetCode-style algorithm trivia at most FDE-hiring companies; it mimics real deployment work.

**How it is run:** 45 to 60 minutes in a shared editor (CoderPad, CodePair, CodeSignal-style) or as a take-home. Roughly 55 to 60 percent pass. You may be asked to run and debug code, not just write it.

**What good looks like:**
- Ask clarifying questions about inputs, edge cases, and the environment before coding.
- Clean, readable, tested code over a clever "optimal" solution.
- Continuous narration of your reasoning and trade-offs.
- Handling the ugly cases: malformed rows, timeouts, retries, partial failures.

**Common mistakes:**
- Jumping straight to code without clarifying the environment or constraints.
- Silence while problem-solving.
- Optimizing for algorithmic elegance while ignoring robustness, error handling, or customer constraints.

**Example prompts:**
- "Parse a messy CSV or JSON file with inconsistent quoting and missing fields, and produce clean structured records."
- "Implement a rate limiter that supports both per-user and global limits."
- "Implement exponential backoff with jitter for a flaky third-party API, with a retry cap."
- "Build a small RAG pipeline over a folder of documents: chunk, embed, retrieve, answer, and defend your chunking choice."
- "Ingest a folder of PDFs and produce a JSON index of the entities found."
- "Write SQL to find customers with a return rate above 30 percent in the last quarter, then explain how you would speed it up."

---

## 4. System design / architecture round

**What it tests:** Real-world deployment thinking. Data flow and trust boundaries, identity and access, observability, failure modes and rollback, and the cost vs. latency vs. complexity trade-off. For AI FDE roles, this is where RAG, agents, and evaluation architecture come in.

**How it is run:** About 60 minutes, whiteboard or shared diagram. You drive; the interviewer probes trade-offs and pushes on failure modes.

**What good looks like:**
- Start from requirements: accuracy bar, latency budget, data sensitivity, scale, and the escalation or fallback policy.
- Propose a thin walking-skeleton MVP first, then iterate and harden.
- Name trade-offs explicitly and pick, rather than hedging.
- Cover identity and permissions, observability (what to log, alert, and dashboard), and rollback.
- Treat evaluation as release infrastructure, not an afterthought.

**Common mistakes:**
- Designing the perfect system with no MVP and no sequencing.
- Ignoring the customer's real constraints (their cloud, their compliance, their data quality).
- Forgetting permissions, monitoring, cost, or how you would roll back.
- Hand-waving on how you would know the system actually works.

**Example prompts:**
- "Design a private, VPC-deployed RAG system for a healthcare customer with HIPAA constraints over 50 million documents."
- "Design ingestion and transformation for 12 fragmented retail data sources with no clean schema, feeding a forecasting model."
- "Design an evaluation harness for an AI agent that reroutes shipments, targeting 99 percent on-time delivery."
- "A naive RAG endpoint returns in 1.5 seconds. Get it under 100 milliseconds. What do you change?"
- "Design how you would version, A/B test, and roll back prompts in production."
- "Design observability for an agent system: what do you log, alert on, and put on a dashboard?"

---

## 5. Ambiguous case / decomposition round (the signature round)

**What it tests:** Judgment under uncertainty. Can you take a vague, real-world business problem, clarify the actual goal, surface assumptions and missing information, decompose it into solvable pieces, sequence them by risk and value, and propose a thin end-to-end first slice? This is the highest-weight round (about 30 percent) and the lowest pass rate (about 40 percent). Palantir made it famous with its open-ended "decomp" interview.

**How it is run:** 45 to 60 minutes. A customer hands you a fuzzy problem with no single correct answer. You are expected to think out loud the entire time and lead the conversation. There is no code; there is structure.

**What good looks like, narrate all five explicitly:**
1. **Clarify the actual goal and success metric.** What does the customer really want, and how will we know we succeeded?
2. **Identify stakeholders and what "done" looks like** for each.
3. **Map available inputs and their data quality**, and name what is missing.
4. **Decompose into solvable subproblems** and sequence them, saying why this order (de-risk the riskiest unknown first, or deliver visible value first).
5. **Propose a thin walking-skeleton MVP end to end**, then say how you would deepen it. Surface failure modes and the risks you are knowingly accepting.

**Common mistakes:**
- Jumping to a solution before scoping. This is the number-one instant rejection.
- Treating ambiguity as a complaint ("this is underspecified") rather than as the work.
- Going silent to think. Narrate.
- Never naming assumptions or the risks you are accepting.

**Example prompts:**
- "A logistics customer says their ops team cannot trust the dashboard. Figure out what is wrong and propose a plan. Go."
- "A major city wants to reduce 911 response times. They have call data, traffic data, and ambulance GPS. You have 60 minutes."
- "A regional bank wants to unify fraud detection across three legacy acquired systems with inconsistent labels. Scope the first 90 days."
- "A logistics firm wants an AI agent to handle shipment rerouting. They have SAP data, real-time weather APIs, and 500 warehouse managers. How do you build it?"
- "An insurer wants LLM-powered claim summarization across 30 million claims under state-by-state regulation. Where do you start?"

---

## 6. Customer / client simulation round

**What it tests:** Communication under pressure and relationship judgment. Can you deliver bad news, push back on a bad request without losing the customer, explain technical limits to non-technical people, and unblock yourself politically? The interviewer role-plays the customer, sometimes frustrated or non-technical.

**How it is run:** About 45 minutes of role-play. The interviewer stays in character. Startups often fold this into other rounds instead of running it standalone.

**What good looks like:**
- Diagnose before you prescribe: ask what is actually going wrong before proposing a fix.
- Acknowledge what the customer is right about before pushing back.
- Offer explicit options with trade-offs rather than a single take-it-or-leave-it.
- Use ownership language ("I will get this to you by Friday") and never over-promise.
- Stay calm; separate the person's frustration from the technical facts.

**Common mistakes:**
- Over-promising to make the tension go away.
- Getting defensive or matching the customer's frustration.
- Explaining with jargon a VP will not follow.
- Caving on data governance or security to please the customer.

**Example prompts (interviewer in character):**
- "Your deployment slipped 3 weeks. I am the customer's CTO. Tell me."
- "I want a feature that would compromise our data governance. Talk me out of it, or do it."
- "Explain to me, a non-technical VP, why your RAG system cannot guarantee 100 percent accuracy."
- "My security team will not give you production credentials. Now what?"
- "I think we should use the architecture my team already picked. Convince me otherwise, or go along."

---

## 7. Behavioral / values round

**What it tests:** Ownership, ambiguity tolerance, cross-functional collaboration, conflict resolution, growth, and (at the labs) mission alignment. Whether you operate well without clear direction or formal authority.

**How it is run:** 45 to 60 minutes, or sprinkled across other rounds. Structured questions about your past. At Anthropic and Palantir, expect serious screening on why you want their specific mission.

**What good looks like:**
- 6 to 8 prepared STAR stories (Situation, Task, Action, Result), 60 to 90 seconds each spoken aloud.
- Specific, individual ("I") contributions with measurable results.
- Honest failure reflection with a concrete lesson applied later.
- Evidence of driving alignment without authority and holding a line under pressure.

**Common mistakes:**
- Vague stories with no result or no clear personal role.
- No genuine failure story, or a fake weakness.
- Generic mission answers at companies that screen alignment hard.

**Prepare stories covering:**
- Owning a project end to end from a vague ask to production.
- Handling a difficult stakeholder or customer.
- Reversing a bad technical decision you had made or endorsed.
- Driving alignment across functions without formal authority.
- Shipping under a tight deadline with incomplete information.
- A real failure and what you changed afterward.
- Spotting a pattern across customers and changing a team process.
- Saying "no" to a customer and holding the line.

**Example prompts:**
- "Tell me about a time you collaborated across functions on an ambiguous problem."
- "Describe a project that failed. What did you learn and change?"
- "Tell me about disagreeing with a technical decision and how you handled it."
- "Give an example of driving impact without any formal authority."
- "Why this company, and why now?" (Read the company's public writing first.)

---

## Company-specific shape

- **Palantir (FDSE):** Recruiter, a Karat-style coding screen, then an onsite with coding, system design, the open-ended decomposition interview, behavioral, and a hiring-manager final. About 4 weeks. The decomp round is the differentiator; study Palantir's own guidance on navigating open-ended questions. Screens hard for genuine interest in its customer problems.
- **OpenAI:** Recruiter, a substantial take-home (reported around 5 hours), a take-home walkthrough plus technical deep dive, then an onsite of 3 to 4 hours (hiring manager, technical, design or case). About 3 weeks. Weights case studies, customer empathy, and business judgment heavily (roughly half), and probes evaluation depth: "How do you know your AI system actually works?"
- **Anthropic (Applied AI Engineer):** Recruiter, take-home, hiring-manager screen, a practical coding assessment (roughly 90 minutes, CodeSignal-style, not LeetCode), technical interviews, and a behavioral or mission-alignment round. 4 to 6 weeks. Read the Core Views on AI Safety and the Responsible Scaling Policy before the alignment round.
- **Databricks:** Emphasizes Spark, SQL, data modeling, RAG over enterprise data, and MLflow, often with a notebook workshop alongside the customer.
- **Scale AI:** Defense and government focus, messy data unification, and security-adjacent questions.
- **ElevenLabs and AI startups:** A tight loop, often with no standalone behavioral round; a case study is central. Show execution velocity and end-to-end ownership.

Next: drill the [question bank](questions.md).
