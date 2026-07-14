# The Rounds

Every round an AI Strategist loop can run, what it tests, how it is conducted, what "good" looks like, the common ways candidates fail, and realistic prompts. Map these to your buyer type (consultancy, enterprise, AI-forward company) using [README.md](README.md).

Jump to: [Recruiter screen](#1-recruiter--hiring-manager-screen) - [AI landscape / fluency](#2-ai-landscape-and-technology-fluency-round) - [Case round](#3-case-round-the-heart-of-the-loop) - [Technical decomposition](#4-technical-decomposition--light-system-design-round) - [Stakeholder / customer simulation](#5-stakeholder-communication--customer-simulation-round) - [Behavioral](#6-behavioral-round) - [Take-home](#7-take-home-variant) - [Debrief and questions to ask](#8-your-questions-and-the-debrief)

---

## 1. Recruiter / hiring-manager screen

**What it tests.** Basic fit, motivation, whether you can talk about AI in business terms without either hand-waving or drowning the recruiter in jargon, seniority calibration, and comp alignment. The hiring-manager version goes deeper on your actual track record with AI initiatives.

**How it runs.** 30 to 45 minutes, phone or video. Expect a "walk me through your background," a "why this role and why now," a plain-language AI question or two, and logistics. The manager version adds a real example of an AI initiative you influenced and what the business outcome was.

**What good looks like.** A crisp 90-second narrative that ends on business outcomes ("we cut handle time 30% and I owned the value case"), one sharp point of view on where enterprise AI is heading, and a specific reason you want this seat. You translate fluently: you can say "we used retrieval so answers stay current and cite sources, instead of retraining" without lecturing.

**Common mistakes.** Reciting a resume chronologically with no outcomes. Over-indexing on tools you have used instead of decisions you have driven. Being unable to name a single number from past work. Sounding like a generalist who added "AI" to a title.

**Example prompts.**
- "Walk me through a time you helped an organization decide where to apply AI first. How did you choose?"
- "In one or two sentences, how would you explain the difference between generative AI and traditional automation to a CFO?"
- "What is one thing about the current AI wave that most executives get wrong?"
- "Why a strategist role and not a product or engineering one?"

---

## 2. AI landscape and technology-fluency round

**What it tests.** Whether your fluency is real and current. Can you reason about the 2025-2026 stack, make correct build-vs-buy-vs-fine-tune-vs-RAG calls, know why evals matter, understand agents and their failure surface, and speak accurately about cost, latency, reasoning models, MCP, and context engineering. This round protects the firm from strategists who sign off on bad technical plans.

**How it runs.** 45 to 60 minutes of live Q&A, often with an engineer or senior technologist. Rapid-fire concept checks, then follow-ups that push for depth ("okay, but when would that break?"). No coding, but you must reason like someone who has watched real systems fail.

**What good looks like.** You give the correct one-paragraph answer, then volunteer the trade-off and the failure mode without being asked. You default to the simplest thing that works (prompt, then RAG, then fine-tune, then agent) and can justify moving up the ladder. You use current terms correctly: you do not confuse fine-tuning with RAG, you know an agent is a loop with tools and memory, you know evals split into retrieval and generation quality, you can estimate why a long context or an extra tool call costs money and time.

**Common mistakes.** Buzzword salad with no mechanism underneath. Recommending fine-tuning for fresh facts. Treating "agent" as a synonym for "LLM." Claiming a system is done with no eval story. Ignoring cost and latency entirely. Talking about 2023-era capabilities as if nothing has changed.

**Example prompts.**
- "A client wants their policy documents answerable by a chatbot. RAG, fine-tuning, or long context. Walk me through the decision."
- "When is an agent worth the extra cost and failure surface, and when is a fixed workflow better?"
- "How would you evaluate whether a customer-support AI is actually good enough to ship? What would you measure?"
- "What do reasoning models change about which problems are now worth attempting?"
- "What is MCP and why would an enterprise care?"
- "A vendor demo looks perfect. What three questions expose whether it will survive production?"

The full concept bank with model answers is **[questions.md](questions.md)**.

---

## 3. Case round (the heart of the loop)

**What it tests.** The whole job in 60 minutes: structured thinking, business and ROI reasoning, current AI fluency applied under pressure, prioritization, build-vs-buy, risk, and adoption. Consultancies weight this heaviest. In 2026 an AI-implementation case shows up in roughly 1 in 3 first-round MBB interviews, and firms like McKinsey have piloted formats where you use their internal AI tool live and are graded on how you prompt it, challenge it, and integrate its output.

**How it runs.** An interviewer gives you a company and a prompt ("Should this insurer adopt AI, and where first?"). You lead: clarify, structure, work through it out loud, do some quantification, and land a recommendation with risks and next steps. Interviewer-led firms will interrupt and push; candidate-led firms let you drive. Expect a couple of curveballs mid-case (a new constraint, a skeptical board member, a cost surprise).

**A structure that works** (announce it, then walk it):
1. **Clarify and scope.** What is the business goal and the metric that matters. What is the time and budget horizon. What is the data and tech reality. Who are the stakeholders. Do not solve before you scope; jumping to a solution is the single most common rejection reason.
2. **Frame the opportunity space.** Map candidate use cases across functions. A clean cut: revenue, cost, risk, and experience; or by function (support, sales, ops, finance, engineering).
3. **Prioritize with an explicit rubric.** Score each use case on business impact, technical feasibility, data readiness, time to production, and risk/compliance. Produce a ranked shortlist of 3 to 5, not a laundry list. Leading enterprises fund fewer initiatives and get more return.
4. **Recommend build vs buy vs partner** per priority use case, with reasons: buy for commodity capability and speed, build for genuine differentiation with data moat and engineering depth, partner to share risk on the hard middle.
5. **Model the value.** Baseline first, then expected lift, then cost to build and run, then payback. Order of magnitude is fine; showing the logic beats false precision.
6. **Name the risks and governance.** Data, security, model quality, hallucination, regulatory, and reputational. Attach controls (human-in-the-loop, evals, monitoring, a governance tier).
7. **Sequence and adopt.** Quick win to prove value, then platform investment, then the transformational bet. Budget for change management (a common rule of thumb is 20 to 30 percent of the AI investment) or the pilot dies in "pilot purgatory."
8. **Recommendation.** Lead with the answer, back it with 3 reasons, state the biggest risk and the first two weeks of action.

**What good looks like.** You lead confidently, structure before you dive, quantify without hiding behind spreadsheets, make defensible build-vs-buy calls, weave in real AI fluency (you know a support-triage RAG deflection use case is more shippable than an autonomous underwriting agent), name failure modes proactively, and end with a clear, sequenced recommendation. If given an AI tool to use live, you prompt it well, sanity-check its output, and keep ownership of the reasoning.

**Common mistakes.** No structure, or a memorized framework forced onto a bad fit. Solving before scoping. A use-case list with no prioritization. Ignoring data readiness and change management. Recommending the most advanced technology instead of the most valuable and shippable one. No numbers, or fake-precise numbers. Forgetting risk and governance entirely. Running out of time with no recommendation.

**Example prompts.**
- "A regional bank's CEO asks where to deploy AI in the next 12 months. Build the strategy."
- "A B2B SaaS company wants to cut support costs 30% with AI. Walk me through your plan, including whether to build or buy."
- "A hospital network wants to use generative AI. Where would you start, and where would you refuse to?"
- "A retailer has run 15 AI pilots and scaled zero. Diagnose why and give them a plan to change it."
- "A manufacturer has a data team but no AI wins. Should they build an internal platform or buy point solutions?"
- "Our own firm wants to embed AI in how we deliver client work. Design the roadmap."

---

## 4. Technical decomposition / light system-design round

**What it tests.** Mostly in AI-forward, boutique, and forward-deployed loops. Can you take one real workflow and lay out a thin, buildable, governed path to production. Not "design Twitter"; more "wrap a model around this messy client process and make it survive contact with production, compliance, and users." This is the biggest filter in forward-deployed loops.

**How it runs.** 45 to 60 minutes. You are handed a workflow and asked to decompose it. Narrate every step: clarify the problem, identify stakeholders and success metrics, map inputs (data, systems, freshness, auth), decompose into solvable sub-problems with sequencing, then propose a thin "walking skeleton" MVP before optimizing.

**What good looks like.** You scope before designing. You start with a walking-skeleton MVP, then iterate. You cover data flow, trust boundaries, auth (SSO/SAML, VPC), observability, evals, failure modes, and rollback. You state trade-offs between cost, latency, complexity, and maintainability explicitly. You treat it as a business-constrained system, not a pure optimization. You know where a human stays in the loop and how you would measure quality before and after launch.

**Common mistakes.** Jumping to a perfect production architecture. Ignoring compliance and governance. Treating it as pure technical optimization. No eval or monitoring story. Over-engineering an agent where a retrieval step plus a form would do.

**Example prompts.**
- "A claims team spends hours reading PDFs to extract 12 fields. Design the thinnest AI system that helps, and how you would prove it works."
- "Design a document-Q&A assistant over a regulated knowledge base. Walk me from data to production, including how you evaluate and monitor it."
- "Take a sales team's manual RFP-response process. Where does AI fit, what is the MVP, and what are the failure modes?"
- "A client wants an 'autonomous' agent for order processing. Talk me through what you would actually build first and why."

---

## 5. Stakeholder communication / customer-simulation round

**What it tests.** Can you carry a recommendation into a room that pushes back. Communication, executive presence, translating technical trade-offs for non-technical people, delivering bad news, de-escalating, and holding a line without becoming defensive. Behavioral and communication signals often carry close to half the overall hiring decision for this role, and in applied-AI loops a customer-conversation round quietly rejects a large share of candidates who cleared the technical bar.

**How it runs.** 45 minutes. An interviewer role-plays a stakeholder: a skeptical CFO, a frustrated business owner, a non-technical board member, or an over-eager executive who wants "an agent for everything." You present a solution, defend a trade-off, deliver a delay or a cost increase, or talk someone down from a bad idea.

**What good looks like.** You ask diagnostic questions before pitching. You acknowledge what the stakeholder is right about before you push back. You use ownership language ("I will get you the pilot results by end of month") rather than deflection. You offer options with explicit trade-offs. You translate: no unexplained jargon, analogies that land ("an agent is like a capable new hire who can use our tools but needs guardrails and review"). You never promise what you cannot deliver, and you stay calm under heat.

**Common mistakes.** Pitching before understanding the concern. Getting defensive under criticism. Making commitments without knowing the constraints. Drowning a non-technical person in jargon. Caving instantly when challenged, or the opposite, refusing to acknowledge a valid point. Over-promising ROI or timelines.

**Example prompts.**
- "I am the CFO. You are asking for 2 million dollars and I have seen AI budgets vanish before. Convince me, and I will push back."
- "Tell the head of a department that the AI tool their team loves is not accurate enough to ship yet."
- "The CEO read that competitors have 'autonomous agents' and wants one in 90 days. Manage that conversation."
- "A client is furious that the pilot missed its accuracy target. De-escalate and set the next step."
- "Explain to a non-technical board why you recommend buying a vendor tool instead of building in-house, without using jargon."

---

## 6. Behavioral round

**What it tests.** Leadership, influence without authority, comfort with ambiguity, judgment, resilience through a failed or stalled initiative, and ethics. For a role whose top-requested skills are leadership, communication, change management, and stakeholder management, this round matters more than candidates expect.

**How it runs.** 45 minutes, STAR-style ("tell me about a time"). Expect at least one failure story and at least one ethics or responsible-AI probe. Senior loops push on driving transformation and building consensus across resistant functions.

**What good looks like.** Specific stories with a real situation, your actual actions, and a measured outcome, ideally a number. You show you can move an organization that does not report to you: how you built the coalition, handled the skeptic, and sequenced the change. Your failure story shows genuine learning, not a humblebrag. On ethics you show you would slow down or say no when the risk warrants it.

**Common mistakes.** Vague, hypothetical, or team-credit-only answers with no "I." A failure story that is secretly a brag. No metrics. No evidence you have driven adoption, only that you wrote a deck. Dodging the ethics question or treating governance as red tape.

**Example prompts.**
- "Tell me about an AI or analytics initiative that failed or stalled. What happened and what did you learn?"
- "Describe getting a resistant leader or team to adopt something new. How did you do it without authority?"
- "Tell me about a time the data or the technology could not support the ambition. How did you reset expectations?"
- "When have you recommended against doing something with AI, or against doing it at all?"
- "Walk me through a high-stakes recommendation you made under real uncertainty."

---

## 7. Take-home variant

Some companies (especially boutiques and AI-forward firms) replace or supplement the live case with a take-home: a mini AI strategy or opportunity assessment for a described company, delivered as a short deck or memo, sometimes with a 30-minute readout.

**What good looks like.** An executive summary that leads with the recommendation. A scored, prioritized shortlist (not a dump). A build-vs-buy call per priority with reasoning. A simple, transparent ROI model with stated assumptions. A risk and governance section. A phased roadmap with a change-management line item. Clean, skimmable, and honest about uncertainty. Treat it exactly like real client work: senior candidates are distinguished by shaping responsible-AI framing and orchestrating transformation, not by technical depth alone.

**Common mistakes.** A 40-slide data dump. No prioritization. ROI with hidden or absurd assumptions. Recommending the fanciest tech. No risk or adoption plan. Missing the "what would I do in the first two weeks" that shows you can start.

**Example prompt.** "Here is a one-page description of a mid-market logistics company. Produce a 6 to 8 slide AI strategy and be ready to present it for 30 minutes."

---

## 8. Your questions and the debrief

Strong candidates use their own questions to demonstrate the role. Good ones to ask:

- "How do you currently decide which AI initiatives get funded, and how do you measure whether they worked?"
- "What is the biggest barrier to AI adoption here: data, talent, governance, or executive alignment?"
- "How mature is your AI governance, and who owns it?"
- "Where has an AI effort here stalled, and what would you want a strategist to do differently?"
- "How does this role split between advising and actually driving the change?"

Next: **[questions.md](questions.md)**.
