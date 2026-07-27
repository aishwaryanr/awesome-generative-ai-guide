# AI System Design Case Studies

Worked AI system design interviews: a realistic question, a full answer, then the follow-ups an interviewer actually asks. Each case is built on one 5-layer spine, grounded in Problem-First design, backed by real production data, and it ships with runnable, provider-agnostic code you can execute.

This is the system design round for AI Engineers and Forward-Deployed Engineers, and a deep reference for anyone designing production AI systems.

## The spine: how we think about AI system design

A quick note before we get started. This is how we look at every one of these problems, and it is worth saying why up front. When you sit down to design an AI system, you need a structure to hang your reasoning on, or you jump straight to a vector database or a model name and skip the decisions that actually matter. These 5 layers are that structure. Most AI systems can be reasoned about as some combination of them, which is why we lead with the spine before touching the specific problem.

This is not the only way to do it, and it is not a universal template. It is a starting point. You decide which layers are load-bearing for the problem in front of you and go deep there, and as systems get more complex you will reach past this spine into more innovative approaches.

- **Layer 1, the model.** The model at the center. Which model, why, and how you handle its non-determinism.
- **Layer 2, the wrapping layer, or the architecture.** Knowledge (retrieval), tools, and memory composed around the model. This is where most of the design lives.
- **Layer 3, evals and guardrails.** How you know any of it is good, and how you stop the bad in real time. Offline sets that gate releases, and online checks that act as guardrails on live traffic.
- **Layer 4, production and ops.** The loop that makes it real: scale, latency, cost, reliability, observability.
- **Layer 5, optimization.** Where you make a working system better and take on harder problems: model routing, caching, advanced retrieval, and multi-agent when one agent is not enough.

Composing these layers into one coherent system is what we mean by system design. Each case study below walks these layers for its own problem, going deepest where that problem is hardest.

```
                          +-------------+
                          |  1  MODEL   |          the model, the brain
                          +------+------+
                                 |
      +------------- 2  WRAPPING LAYER (ARCHITECTURE) --------------+
      |    [ Knowledge / RAG ]    [ Tools ]    [ Memory ]          |   the architecture
      +---------------------------- + -----------------------------+   around the model
                                 |
                   +-------------+--------------+
                   |                            |
      3  EVALS & GUARDRAILS         4  PRODUCTION & OPS
      (is it good? stop the bad)   (keep it alive at scale)
                   |                            |
      +----------------------- 5  OPTIMIZATION --------------------------+
      |   make it better: routing, caching, advanced retrieval,          |
      |   multi-agent when one agent is not enough                       |
      +-----------------------------------------------------------------+

      system design = composing all of the above into one coherent system
```

## Cases

Ten worked case studies, each foregrounding a different part of the spine.

- **[Customer support agent](customer-support-agent/README.md)**: answers from a help center, looks up orders and opens tickets through tools, escalates to a human. Tools and escalation.
- **[Enterprise research assistant](enterprise-research-assistant/README.md)**: answers across many internal sources with citations and per-user permissions. Deep multi-source retrieval and multi-agent research.
- **[Text-to-SQL analytics copilot](text-to-sql-analytics/README.md)**: answers business questions over a warehouse in natural language. Schema-aware SQL, safe execution, and correctness evals.
- **[Financial and operations decisioning](financial-ops-decisioning/README.md)**: runs a repetitive finance workflow (invoice or expense) with policy checks, an audit trail, and human sign-off. Evals, guardrails, and audit.
- **[Security and compliance agent](security-compliance-agent/README.md)**: monitors activity streams for policy violations in real time. Real-time guardrails with a low false-positive budget.
- **[Coding agent](coding-agent/README.md)**: takes a task, edits a repo, runs tests, opens a pull request. The harness, context and compaction, verification loops, long-horizon autonomy, and multi-agent.
- **[Document decisioning](document-decisioning/README.md)**: reads submitted underwriting documents, extracts, applies policy, decides or routes, with an audit trail. Extraction plus compliance.
- **[Real-time voice support agent](voice-support-agent/README.md)**: a phone-line agent with streamed audio and interruptions. Latency, streaming, and turn-taking.
- **[SDR sales-development agent](sdr-sales-agent/README.md)**: qualifies leads and drafts personalized outreach. CRM tools, brand and compliance guardrails, human approval before send.
- **[Clinical scribe](clinical-scribe/README.md)**: turns an encounter transcript into a structured note, on synthetic data only. Extraction and faithfulness safety.

## How to read these

Start with the spine above, then open any case. Each one recaps the spine, scopes the problem in Problem-First terms, walks the layers going deepest where that case is hardest, and takes the follow-ups an interviewer would ask. You can also point your coding agent at a case and have it walk you through interactively, or reimplement it in the SDK you use (see the callout at the top of each case).
