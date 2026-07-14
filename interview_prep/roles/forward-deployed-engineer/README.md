# Forward-Deployed Engineer (FDE) Interview Prep

A deep, self-contained prep folder for the Forward-Deployed Engineer role. If you read only one thing, read this page, then work through the other 5 files in order.

**In this folder**
- [README.md](README.md): what the role is, who hires for it, how it differs from adjacent roles, and the full interview loop (you are here).
- [rounds.md](rounds.md): every interview round, what it tests, how it runs, what "good" looks like, and example prompts.
- [questions.md](questions.md): a 40+ question bank with concise model answers, grouped by theme.
- [resources.md](resources.md): the best free reading, talks, and papers, all links verified.
- [courses.md](courses.md): the best free courses, starting with this repository's own.
- [prep-plan.md](prep-plan.md): a day-by-day 3-week plan plus a 1-week crunch plan.

---

## What the job actually is

A Forward-Deployed Engineer is a customer-facing engineer who deploys, integrates, and customizes software (increasingly AI systems) directly inside a client's environment. You embed with the customer, learn their domain, ship working code fast, and own the full loop from a vague business problem to a running system that people trust.

The role was popularized by Palantir, where an FDE (or FDSE, Forward-Deployed Software Engineer) embeds at a customer site, models the customer's domain into an ontology, and ships a working transform, dashboard, or workflow in the first week rather than a slide deck. In 2025 and 2026 the same model spread across the AI industry: OpenAI, Anthropic, Databricks, Scale AI, ElevenLabs, and many AI startups now hire FDEs (Anthropic calls the closest version "Applied AI Engineer") to put large language models, agents, and custom pipelines into production at Fortune 500s and regulated enterprises.

The defining traits, drawn from how Palantir describes the role and how the AI labs adapted it:
- **Ship on day one.** You deliver production software, not implementation roadmaps or consultant decks. A working prototype in week one beats a perfect plan in month six.
- **Own the full value loop.** From discovery and scoping through integration, evaluation, rollout, and support. You are the single throat to choke for whether the deployment works.
- **Discovery is engineering.** Expect to spend a large share of the week (reports of 30 to 40 percent at the AI labs) in structured customer conversations, mapping their domain and data before and while you build.
- **Engineer-diplomat, not sales-engineer.** The common framing: an FDE is an engineering hire who happens to be customer-facing; a solutions engineer is a sales hire who happens to be technical. You write real production code, at the customer, and hold the technical line.

For AI-specific FDE roles, the work centers on retrieval-augmented generation over the customer's data, agents that call the customer's tools and systems, evaluation harnesses that prove the system works, and deployment inside the customer's cloud (often a private VPC) with their identity, security, and compliance constraints.

---

## Who hires for it, and at what levels

| Company | Role name | What they emphasize |
|---|---|---|
| Palantir | Forward-Deployed Software Engineer (FDSE), Forward-Deployed Engineer | Open-ended decomposition, ontology modeling, embedded delivery, government and enterprise |
| OpenAI | Forward-Deployed Engineer | Production AI systems, evaluation depth, agent frameworks, business judgment weighted heavily |
| Anthropic | Applied AI Engineer / Forward-Deployed Engineer | Safety, evals, reliable Claude deployments in regulated environments, mission alignment |
| Databricks | Customer-facing / AI Engineer | Spark, SQL, lakehouse, RAG over enterprise data, notebooks with the customer |
| Scale AI | Forward-Deployed Engineer | Defense and government, messy data unification, security-adjacent work |
| ElevenLabs and AI startups | FDE / Solutions Engineer | Speed, scrappiness, end-to-end ownership, tight loops |

**Levels.** FDE roles run from mid-level (roughly 3 to 5 years of experience) through staff and principal. Junior FDE hiring exists but is rarer, because the role leans hard on judgment, communication, and independent ownership. Titles vary: "Forward-Deployed Engineer," "Applied AI Engineer," "Solutions Engineer," "Customer Engineer," "Deployment Strategist" (the non-coding sibling at Palantir). Compensation at the AI labs is frequently equity-heavy (Anthropic reportedly weights offers toward equity units).

---

## How the FDE role differs from adjacent roles

- **vs. Software Engineer (SWE):** A SWE builds the core product for all customers from headquarters. An FDE builds the customer-specific integration, at the customer, and owns whether that one deployment succeeds. FDEs are judged on customer outcomes, not just code.
- **vs. Solutions Engineer / Sales Engineer:** A sales engineer supports the sale and does demos and light technical scoping. An FDE writes production code and owns delivery after the sale. The FDE holds the technical line even against the customer when needed.
- **vs. AI / LLM Engineer:** An LLM engineer may build the same RAG or agent systems, but usually inside their own product and team. An FDE does it inside the customer's environment, under the customer's constraints, with the customer watching, and must communicate every trade-off to non-technical stakeholders.
- **vs. Consultant:** A consultant delivers analysis and recommendations. An FDE delivers running software. "Ship on day one" is the line that separates them.
- **vs. Applied Scientist / Research Engineer:** Applied scientists push model capability and research. FDEs push deployed capability into a specific business, and are measured on reliability and adoption rather than benchmark gains.

The interview loop reflects this: it is roughly half technical (coding, system design, AI depth) and half judgment (ambiguous case decomposition, customer simulation, behavioral). Candidates who prepare only algorithms fail the half that decides the outcome.

---

## The end-to-end interview loop

A typical FDE loop runs 5 to 8 stages over 3 to 6 weeks. Exact order varies by company, but the shape is consistent. Full detail per round is in [rounds.md](rounds.md).

1. **Recruiter screen** (about 30 min): motivation, role fit, why FDE over SWE, communication baseline.
2. **Hiring-manager screen** (45 to 60 min): depth of past work, ownership, technical judgment, individual contribution.
3. **Coding round** (45 to 60 min, live or take-home): practical engineering on realistic problems (parse messy data, build a small RAG pipeline, rate limiter, retries with backoff). Clean and communicated beats clever.
4. **System design / architecture round** (about 60 min): deploy a real AI system under real constraints (private VPC, HIPAA, identity, cost vs. latency, failure modes, rollback).
5. **Ambiguous case / decomposition round** (45 to 60 min): the signature round. A vague enterprise problem you must scope, decompose, and sequence under uncertainty. Highest weight (about 30 percent), lowest pass rate (about 40 percent). Jumping to a solution before scoping is the single most-cited instant rejection.
6. **Customer / client simulation round** (about 45 min): the interviewer role-plays a frustrated or non-technical customer. Tests communication under pressure, pushing back without losing the relationship, and never over-promising.
7. **Behavioral / values round** (45 to 60 min): ownership, ambiguity tolerance, conflict, mission alignment. Prepare 6 to 8 STAR stories. Use "I," not "we."

Company shape notes: OpenAI runs a faster loop (roughly 3 weeks) with a substantial take-home and heavy weight on evaluation and business judgment. Palantir centers the open-ended decomposition round and screens hard for genuine interest in its customer problems. Anthropic screens mission alignment seriously (read its safety writing) and runs practical, non-LeetCode coding. Startups compress everything into a tight loop and prize execution velocity.

---

## Ground yourself with these repository resources

Build the technical base the loop assumes, using this repository:

- **Topic pages:** [Foundations](../../../topics/foundations.md), [Prompting](../../../topics/prompting.md), [RAG](../../../topics/rag.md), [Fine-tuning](../../../topics/fine-tuning.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Multimodal](../../../topics/multimodal.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md).
- **Journeys:** [Build AI](../../../journeys/build.md) is the primary track for this role; [Use AI](../../../journeys/use.md) and [Understand AI](../../../journeys/understand.md) fill gaps.
- **Paths:** [Harness Engineering](../../../paths/harness-engineering.md) and [Agent Builder](../../../paths/agent-builder.md) map directly to what an AI FDE deploys.
- **Question banks:** [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and [Role-Based Interview Prep](../../README.md) (the Solutions Architect and LLM Engineer tracks overlap most).
- **Guides and roadmaps:** [GenAI Roadmap](../../../resources/genai_roadmap.md), [Agents Roadmap](../../../resources/agents_roadmap.md), [RAG Roadmap](../../../resources/RAG_roadmap.md), [Agentic RAG 101](../../../resources/agentic_rag_101.md), [Agents 101](../../../resources/agents_101_guide.md), [Fine-tuning 101](../../../resources/fine_tuning_101.md), [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).
- **Current landscape:** the [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md) and the research tables on [RAG](../../../research_updates/rag_research_table.md), [AI evaluation](../../../research_updates/ai_evaluation_2025_table.md), and [agentic search and retrieval](../../../research_updates/agentic_search_retrieval_table.md).

Next: work through [rounds.md](rounds.md).
