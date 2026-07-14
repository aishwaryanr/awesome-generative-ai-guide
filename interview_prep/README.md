# Role-Based Interview Prep

This section is a set of self-contained prep folders, one for each major AI role. Generative AI interviews no longer test a single generic skill set. What an AI Engineer is grilled on looks very different from what an AI Product Manager, a Forward-Deployed Engineer, or an AI Strategist is asked. So the prep here is organized by the job you are actually interviewing for.

## How to use this section

1. **Pick your role** from the list below and open its folder.
2. **Work that folder end to end.** Each one is built to be the only resource you need for that loop: read the overview, study the rounds, drill the question bank, then follow the prep plan through to interview day.
3. **Layer in the shared fundamentals.** Every role assumes a working grasp of the core generative AI concepts. Use the shared material below to fill gaps or to refresh before a technical screen.

```text
Pick your role, then work its folder end to end:

  Overview  ->  Rounds  ->  Question bank  ->  Resources + Courses  ->  Prep plan  ->  interview day
  what the      the loop,    drill the          go deep where            day-by-day
  job is        each round   answered Q&A       you are thin             plan
```

## Pick your role

- **[AI Engineer](roles/ai-engineer/README.md)**: builds LLM-powered product features that ship and hold up in production. Prompting, context engineering, RAG, agents, evals, and reliability.
- **[AI Product Manager](roles/ai-product-manager/README.md)**: owns products whose core behavior comes from a model. Capability judgment, probabilistic UX, evaluation, and cost, latency, and risk tradeoffs.
- **[Forward-Deployed Engineer](roles/forward-deployed-engineer/README.md)**: customer-facing engineer who embeds with a client, learns their domain, and ships working AI systems fast, from a vague business problem to a running product people trust.
- **[AI Strategist](roles/ai-strategist/README.md)**: advises an organization on where AI creates value, how to sequence adoption, whether to build or buy, and how to manage cost, return, risk, and governance.

Not sure which is you? A quick comparison:

| Role | You mainly | Signature round | Heaviest topics |
|---|---|---|---|
| [AI Engineer](roles/ai-engineer/README.md) | build LLM features that ship | take-home + system design | RAG, agents, evaluation, system design |
| [AI Product Manager](roles/ai-product-manager/README.md) | decide what to build and why | product sense (design a feature) | capability judgment, metrics, tradeoffs |
| [Forward-Deployed Engineer](roles/forward-deployed-engineer/README.md) | build and deploy at the customer | ambiguous-case decomposition | full-stack build, communication, ambiguity |
| [AI Strategist](roles/ai-strategist/README.md) | advise on where AI creates value | the strategy case | landscape fluency, ROI, governance |

## Shared fundamentals for everyone

No matter the role, every loop expects fluency in the core ideas. Start with the question bank, then use the topic pages and courses to go deeper where you are thin.

- **[60 GenAI Interview Questions](60_gen_ai_questions.md)**: the broad, role-agnostic question bank covering the concepts that come up in nearly every AI interview. Read this first.

**Repository topics** (concept-by-concept reference reading):

- [Foundations](../topics/foundations.md): how large language models work, tokens, embeddings, transformers, and the vocabulary the rest depends on.
- [Prompting](../topics/prompting.md): prompt and context engineering, the primary lever in modern LLM products.
- [RAG](../topics/rag.md): retrieval-augmented generation, chunking, indexing, and grounding answers in your own data.
- [Agents](../topics/agents.md): tool use, planning, memory, and multi-step autonomous systems.
- [Evaluation](../topics/evaluation.md): building eval sets, LLM-as-judge, regression suites, and catching silent degradation.
- [Fine-tuning](../topics/fine-tuning.md): when to fine-tune versus prompt, and the methods for adapting a base model.
- [Production](../topics/production.md): serving, latency, cost, monitoring, and everything between a demo and a live system.
- [Safety and security](../topics/safety-security.md): prompt injection, guardrails, privacy, and responsible AI.
- [Multimodal](../topics/multimodal.md): models that work across text, images, audio, and video.

**Courses**:

- [All free courses, by topic](../courses.md): the full catalog of free courses in this guide, grouped by topic, starting with the repository's own material.

## What each role folder contains

Every role folder follows the same structure, so once you know one you know them all:

- **Overview**: what the job really is, who hires for it, how it differs from adjacent roles, and the full interview map.
- **Rounds**: every round in the loop, what each tests, what good looks like, and example prompts.
- **Question bank with answers**: a deep, themed set of questions with model answers, the heart of each folder.
- **Resources**: the best free reading, talks, papers, and repos, verified and grouped.
- **Courses**: the best free courses for that role, starting with this repository's own.
- **Prep plan**: a day-by-day plan, usually a multi-week track plus a one-week crunch version.
