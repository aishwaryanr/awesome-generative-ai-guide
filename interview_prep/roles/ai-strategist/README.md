# AI Strategist: Interview Prep

A deep, self-contained prep folder for the **AI Strategist** role: the person who advises an organization on where AI creates value, how to sequence adoption, whether to build or buy, what it will cost and return, what could go wrong, and how to bring the org along. This folder is meant to be the only resource you need to walk into the loop prepared.

Read the files in order:

1. **[README.md](README.md)** (this file): what the job is, who hires, levels, adjacent roles, and the full interview loop.
2. **[rounds.md](rounds.md)**: every round you will face, what each tests, what "good" looks like, and example prompts.
3. **[questions.md](questions.md)**: a 40-question bank with model answers, covering AI fluency and business strategy.
4. **[resources.md](resources.md)**: verified free reading, talks, papers, and repos.
5. **[courses.md](courses.md)**: the best free courses, starting with this repository's own.
6. **[prep-plan.md](prep-plan.md)**: a day-by-day 3-week plan plus a 1-week crunch version.

---

## What the job actually is

An AI Strategist sits between the technology and the business. The core deliverable is a **defensible plan**: which AI use cases to fund, in what order, at what cost, with what expected return, under what risk and governance constraints, and how the organization will actually adopt them. You are judged on judgment, not on shipping code.

The concrete artifacts an AI Strategist produces look like this:

- An **AI opportunity map**: a scored, ranked backlog of use cases across functions (support, sales, ops, finance, engineering), each tied to a business metric.
- A **roadmap**: a sequenced plan (quick wins, then platform investments, then transformational bets) with owners and milestones.
- **Build vs buy vs partner** recommendations for each priority use case, with the reasoning made explicit.
- A **business case / ROI model**: baseline, expected lift, cost to build and run, payback period, and how it will be re-measured.
- A **risk and governance plan**: data, security, model, and regulatory risk, mapped to controls (often referencing the NIST AI Risk Management Framework, the EU AI Act, or ISO/IEC 42001).
- A **change and adoption plan**: sponsorship, training, incentives, and the operating-model changes that decide whether a pilot ever reaches production.

The reason this role exists is a hard, well-documented fact: most enterprise AI spending does not convert to value. Multiple 2025 and 2026 analyses (MIT, RAND, McKinsey, Deloitte, Gartner) put the share of AI pilots that fail to reach production or hit their business case at roughly 80 to 95 percent, and the failures are rarely about the model. They are about picking the wrong use case, no baseline, no change management, absent governance, and executive sponsorship that evaporates after the demo. An AI Strategist is hired to be the person who does not let that happen.

To do that well you need genuine **technology fluency**. You do not have to train models, but you must know, in current terms, what large language models can and cannot do, when to use RAG versus fine-tuning versus a well-engineered prompt, what an agent is and when it is worth the added cost and failure surface, how evals work and why they are non-negotiable, what reasoning models change, what MCP and context engineering are, and how cost and latency actually behave. Interviews test this fluency directly because a strategist who cannot smell a bad technical plan cannot protect a budget.

---

## Who hires, and at what level

Three buyer types hire AI Strategists, and the flavor of the loop shifts with each:

- **Consultancies** (McKinsey, BCG, Bain, Deloitte, Accenture, EY, KPMG, plus boutiques). They test structured case-cracking. In 2026, an AI-implementation case appears in roughly 1 in 3 first-round MBB interviews, and some firms (for example McKinsey's Lilli pilot) now have you use an internal AI tool live and grade how you prompt it, challenge its output, and fold it into your recommendation.
- **Enterprises** standing up an AI function (banks, insurers, retailers, healthcare, telecom). They test whether you can prioritize against their P&L, navigate their data and compliance reality, and drive adoption through existing org structures. Change management, stakeholder management, and governance weigh heavily here.
- **AI-forward companies and AI labs** (OpenAI, Anthropic, Google, Palantir, Databricks, and the vendor/consulting layer around them). This is where the strategist blurs into the **forward-deployed** and **applied AI** archetype: you go into a client's environment, find the highest-value workflow, and drive it to production. Job postings for forward-deployed roles jumped more than 800 percent in 2025. These loops add a hands-on technical decomposition round and a live customer-simulation round.

Levels, from a 2026 analysis of 1,859 AI-strategy postings:

| Level | Typical experience | What you own | Rough US median base |
|-------|--------------------|--------------|----------------------|
| Junior (about 3% of postings) | 4+ years | Execute deployment, enable adoption | ~$139K |
| Mid (about 26%) | 6+ years | Own a roadmap and cross-functional delivery | ~$190K |
| Senior / Principal (about 71%) | 10+ years | Enterprise vision, transformation, responsible-AI framework | ~$230K (80th pct to ~$310K) |

The most-requested competencies in those postings were not coding. They were team leadership (32%), communication (22%), cross-functional collaboration (22%), change management (21%), and stakeholder management (about 20%). Generative-AI and ML knowledge showed up as required in only 12 to 15 percent, but that understates reality: the fluency is assumed at senior levels and is the fastest way to fail a technical round if you lack it.

---

## How this role differs from adjacent roles

- **vs AI Product Manager**: The PM owns one product's roadmap, backlog, and shipping cadence. The Strategist owns the portfolio and the org-level plan: which bets to make, in what order, build vs buy, and how the enterprise changes. The PM answers "what do we build next in this product"; the Strategist answers "should we build this at all, and what is the plan across the company."
- **vs AI / ML Engineer**: The engineer builds and ships the system. The Strategist decides whether it should be built, by whom, and what it must return. You need to understand the engineer's world well enough to pressure-test their plan, not to write it.
- **vs Management Consultant (generalist)**: Same structured-case muscles, but the Strategist must carry real 2025-2026 AI fluency. A generalist framework with no technical substance now fails the AI-implementation case.
- **vs Forward-Deployed / Applied AI Engineer**: The FDE lives in the client's codebase and ships. The Strategist may sit above or alongside them, but in AI-lab and boutique loops the two roles converge, and you will get a technical decomposition round.
- **vs Chief AI Officer / Head of AI**: The same work at a permanent, accountable, budget-owning altitude. Senior Strategist interviews are a proving ground for this seat.

---

## The end-to-end interview loop

A representative loop, in order. Not every company runs all six; consultancies lean on the case, enterprises on stakeholder and behavioral, AI-forward companies add the technical decomposition and customer simulation. Full detail per round is in **[rounds.md](rounds.md)**.

1. **Recruiter / hiring-manager screen** (30 to 45 min): fit, motivation, AI narrative, comp. Can you talk about AI value in plain business language.
2. **AI landscape and technology-fluency round** (45 to 60 min): live Q&A on the current stack: LLMs, RAG, fine-tuning, agents, evals, reasoning models, MCP, context engineering, cost and latency, responsible AI.
3. **Case round** (60 min live, or a multi-day take-home): build an AI strategy for a given company or function. The heart of the loop.
4. **Technical decomposition / light system-design round** (AI-forward and boutique loops): take one workflow and lay out a thin, buildable, governed path to production.
5. **Stakeholder-communication / customer-simulation round** (45 min): present a recommendation, defend a trade-off, deliver bad news, de-escalate a skeptical or non-technical stakeholder.
6. **Behavioral round** (45 min): leadership, ambiguity, influence without authority, a failed or stalled AI initiative, ethics.

Some loops fold 2 and 4 together, or fold 5 into the case debrief. Read the pattern of your specific company from the buyer type above.

---

## Grounding in this repository

Use these to build real fluency, not just talking points:

- **Foundations and the map**: [Foundations](../../../topics/foundations.md), [Prompting and context](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Fine-tuning](../../../topics/fine-tuning.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Multimodal](../../../topics/multimodal.md), [Production and LLMOps](../../../topics/production.md), [Safety and security](../../../topics/safety-security.md).
- **Journeys** (pick your depth): [Use AI](../../../journeys/use.md), [Build with AI](../../../journeys/build.md), [Understand AI](../../../journeys/understand.md).
- **Market context you will be quizzed on**: [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md).
- **The current research tables**: [RAG](../../../research_updates/rag_research_table.md), [AI evaluation 2025](../../../research_updates/ai_evaluation_2025_table.md), [Agentic search and retrieval](../../../research_updates/agentic_search_retrieval_table.md).
- **Roadmaps and guides**: [GenAI roadmap](../../../resources/genai_roadmap.md), [Agents roadmap](../../../resources/agents_roadmap.md), [RAG roadmap](../../../resources/RAG_roadmap.md), [Agentic RAG 101](../../../resources/agentic_rag_101.md), [Agents 101](../../../resources/agents_101_guide.md), [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md).
- **The question banks**: [60 GenAI interview questions](../../60_gen_ai_questions.md) and [role-based prep](../../README.md).
- **Builder paths** (to understand what your engineers actually do): [Harness engineering](../../../paths/harness-engineering.md), [Agent builder](../../../paths/agent-builder.md).

Next: **[rounds.md](rounds.md)**.
