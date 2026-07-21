# Enterprise Research Assistant: the PM interview

## The interview question

> "Employees lose hours hunting across our internal tools for answers. You are the PM. Should we build an internal AI research assistant, what does success look like, and how do you ship it without leaking data or eroding trust in its answers?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk it, and how you make the calls when the model is probabilistic and sometimes wrong. The technical design (permission-scoped retrieval, the multi-hop search loop, evals, the harness) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/enterprise-research-assistant` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, and the number that must not break.
- **4 Experience.** The probabilistic UX: trust, transparency, citations, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, pilot, staged release, go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to an employee, an engineer, and a CISO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. The answer an employee needs already exists somewhere: a policy in the wiki, a decision buried in a ticket, an owner named in the code, a thread in chat. Finding it by hand means searching several tools, reading a stack of documents, and interrupting the 2 colleagues who happen to know. As an illustrative scoping assumption, a knowledge worker loses on the order of a few hours a week to this, and the experts who hold the context in their heads lose more, because they are the ones getting pinged all day. Two users have pain: the employee who waits, and the expert interrupted to answer a question the corpus already contains.

Write the outcome before the system: return a correct, cited answer stitched together only from the sources this specific person is allowed to see, so employees stop hunting and experts stop being human search engines. Narrow the intervention until it hurts: retrieve across the connected sources under the asking user's permissions, run the follow-up searches a hard question needs, answer with citations, and hand off cleanly when nothing readable supports an answer. An assistant that also takes actions, writes to systems, and automates workflows in version 1 is the over-scoping trap. Keep it read-and-answer.

The clarifying questions you need to ask a stakeholder: what is the cost of a wrong answer in the workflows people will actually use this for; how are permissions modeled in each source system, and how fast do they change; what share of questions truly span several sources; is a citation mandatory on every answer; what are the compliance constraints on the most sensitive sources; and what is the promise about reaching a human when the assistant cannot answer.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to answer grounded questions from an internal corpus, cite the source, and abstain when nothing supports an answer. They are far less reliable when you let them roam an entire corpus without a permission filter, or take actions off the back of what they read. So the design keeps the assistant read-only, grounds every claim in retrieved text, and enforces each user's permissions at retrieval time. Match the ambition to what the model does reliably, gate the consequential parts, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume question type on a well-understood, lower-sensitivity corpus, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a real fork, and here the buy side is crowded. Enterprise search and assistant vendors (for example Glean, Microsoft 365 Copilot, Guru, Coveo) already carry connectors to the common source systems and, crucially, a permission-aware retrieval layer, which is the hardest part to get right. Buying gets you live faster and inherits their work on access control. Building in-house gives you control over behavior, the corpus, cost, and the experience, and it compounds when the way your company finds knowledge is itself a differentiator. Decide on a few axes: how differentiated the experience needs to be, how sensitive the data and the permission model are, token cost against control, time to market against the data moat you would build, vendor lock-in, and whether you have the team to own an evaluation and monitoring loop for the life of the product. A common answer is to buy to learn fast on a low-sensitivity corpus, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

One capability question sits underneath both build and buy: how much compute a hard research question is worth. A broad question that fans out across many independent sources is exactly where spending more tokens buys real quality, through a multi-agent research design. Treat that as a product decision about quality against cost rather than a mechanism: you pay for the fan-out on the valuable, genuinely broad questions and keep the cheap single-pass path for the quick lookups that are the majority. The multi-agent follow-up below works the tradeoff.

> **Real data: [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).** A lead agent with parallel subagents beat a single agent by about 90.2% on Anthropic's internal research eval, while using about 15x the tokens of an ordinary chat. Read at the product level, that is the quality-versus-cost dial for research: fan-out buys real capability on broad questions and costs real money, so you spend it where the question is worth it and stay on the cheap path everywhere else. Treat the figures as a point-in-time reading on one eval and track cost per answered question on your own traffic.

**What is shifting under you.** Two retrieval changes are worth tracking as the product owner, and both complement the grounded pipeline rather than replace it. Agentic retrieval lets the assistant run its own iterative searches and decide what to fetch next, which lifts quality on the hard multi-source questions ([agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)). Long-context models let you fit whole documents into the prompt, which simplifies the build for a small, self-contained corpus ([long context](https://ai.google.dev/gemini-api/docs/long-context)). At enterprise scale the retrieval pipeline stays the primary path, because hundreds of thousands of permissioned items will not fit one window and cost, latency, precision, and freshness still favor retrieving the few right sources under each user's access. Treat both as levers you reach for on the questions and corners that earn them, and keep the pipeline as the default.

### 3 Success metrics

Pick the minimum set that tells you the truth, and name the number that must not break.

- **Adoption.** Weekly active users as a share of the employees the assistant is meant to serve, and repeat use. A research assistant that people try once and abandon has failed no matter how good a single answer looks.
- **Time saved per question.** The core business outcome: answered in seconds instead of a multi-tool hunt. Triangulate self-reported time saved with a measured drop in duplicate questions reaching human experts.
- **Citation faithfulness.** The trust metric. Every claim traces to a source the user can open, and the cited source genuinely supports the claim. This is what makes an answer usable in an enterprise, and it erodes fastest when it slips.
- **Permission-leak rate.** The hard ceiling, and the number that must stay near zero. A single source shown to someone who should not see it can sink the project on its own, so this is measured on every request rather than sampled.
- **Abstention and escalation rate.** Too high means the assistant is not helping; too low means it is answering questions it should hand off, which hides failures behind confident prose.
- **Cost per answered question.** The unit economics finance will ask about, and where the multi-agent fan-out shows up.

Separate leading signals (grounded-answer rate, retrieval hit rate) from lagging outcomes (adoption, sustained time saved). Pair the business outcome (adoption, time saved) with a quality metric beneath it (citation faithfulness, permission-leak rate) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to watch is coverage bought by guessing: an assistant that drives its answer rate up by responding to everything, including the questions it should abstain on, looks helpful on a dashboard and quietly feeds wrong answers into real decisions. Answer rate is only healthy when it rises alongside faithfulness, with the leak rate flat at zero.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core. Design it so a wrong answer is rare, visible, and cheap to recover from. This has an established design playbook: help the employee form an accurate picture of what the assistant can do, make its uncertainty legible, and give a fast path to correct it or reach a person ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Ground and cite, inline.** Answers quote the real source and link it, so the reader can follow every claim back and verify it. In an enterprise an uncited answer is unusable, because the reader has no way to trust it.
- **Gate on confidence and hand off gracefully.** When retrieval clears no relevance floor, the assistant abstains and points to the person or team who owns that knowledge, or falls back to raw search results the user can open, rather than a dead end ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). The permission boundary hides inside this: when the only relevant sources sit outside the user's permissions, the readable set is empty, it falls below the floor, and the assistant abstains cleanly rather than hinting at what it could not show.
- **Set expectations.** Make it clear this answers from internal sources under the reader's own access, and what it can and cannot help with, so the employee's mental model matches what the system actually does ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)).
- **Make correction cheap.** Let the employee rephrase, flag a wrong or stale answer, or report that a citation points to something they cannot open, and route that feedback back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and tone.** A few seconds is fine for a simple lookup, and a harder multi-source question can take longer if you stream the answer so it starts appearing quickly. The voice stays plain and cites rather than sounds authoritative.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of real questions, including the same question asked by users with different permissions so the leak checks have something to catch, and treat it as the release gate: a change ships only if citation faithfulness, answer correctness, and the permission-leak ceiling hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a data leak across a permission boundary: an employee shown a source, or an answer built on a source, they had no right to see, whether that is compensation data, an unannounced reorg, or deal documents. A confident wrong answer erodes trust and is recoverable; a cross-permission leak is a security and legal event that can end the program. So permissions are enforced at retrieval time, before any content reaches the model, and the permission-leak check is a cheap deterministic guardrail that runs on every request. The other guardrails (grounding every claim, a relevance floor that abstains when unsure, treating every retrieved document as untrusted data to cite rather than an instruction to follow) are product-safety features, because they cap the worst case at a clean handoff rather than a wrong answer or a leaked source.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates, and expand one team at a time.

- **Shadow first.** Run the assistant against real questions without surfacing its answers, and compare against the eval bar, including the permission checks.
- **Pilot one team.** Turn it on for a single team over a well-understood, lower-sensitivity corpus, read every transcript, and watch the metrics from section 3.
- **Expand team by team** behind go and no-go gates on adoption, citation faithfulness, and the permission-leak ceiling. Each new team brings its own source systems and its own permission model, so you re-verify the leak checks on that team's data before you widen access, rather than assuming the last team's clearance carries over.
- **Monitor and keep a human path.** Sample live traffic, watch for drift and for citation faithfulness slipping, and always leave a path to the human who owns the knowledge.

Have a rollback you can trigger in minutes, and one pull-back rule that is not negotiable: a confirmed cross-permission leak pauses the rollout immediately, because that is the failure the whole design exists to prevent.

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with adoption paired with time saved and citation faithfulness, gate on the permission-leak ceiling, and refuse to celebrate an answer rate that climbs because the assistant stopped abstaining. Prompts sent and sessions measure activity rather than value; report reliability, because an assistant that is right most of the time can still leak or fabricate a citation often enough to lose trust.

**"The assistant shows someone a document they should not see. Walk me through your response."** Treat it as a security incident rather than a bug. Contain it first, pausing the affected path or the rollout, then trace the leak to the step that broke, usually a stale permission sync or the retrieval filter, add the exact case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern that should hold the rollout. The audit trail of which sources built each answer and what the user's access was is what makes this traceable.

**"Engineering says 6 months. Scope an MVP."** Cut to one team, 1 or 2 well-permissioned source systems, single-pass retrieve-cite-or-abstain with no multi-hop loop and no actions. Prove adoption, citation faithfulness, and a flat leak rate there before adding more sources, the follow-up-search loop, and the multi-agent fan-out. The follow-ups add capability the way versions would, without betting the launch on the hardest, most sensitive slice.

**"When do you pay for multi-agent research?"** Frame it as quality against cost rather than a mechanism. A broad, high-value question that genuinely fans out across many independent sources is worth the roughly order-of-magnitude token spend, because parallel subagents buy real quality there ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). A quick single-source lookup stays on the cheap path. Let the traces show when a class of question has outgrown one agent, and route by that rather than defaulting everything to the expensive design.

**"Build or buy?"** Enterprise search vendors get you live fast and, more importantly, carry a permission-aware retrieval layer, which is the hardest and highest-stakes part to build. Buy to learn fast on a low-sensitivity corpus, and build in-house when the experience, the corpus, or the permission model is differentiating or too sensitive to hand to a vendor. Decide on differentiation, data and permission sensitivity, and whether you can own an evaluation loop for the product's life.

**"Leadership wants it to answer everything."** Push back with coverage against trust. An assistant that answers every question, including the ones it should hand off, drives a good-looking answer rate while quietly feeding wrong answers into decisions. Abstention is a product feature: the relevance floor and the permission boundary cap the worst case at a clean handoff, and that is what keeps the answers people do get worth trusting.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf), and Glean on [permissions-aware enterprise AI](https://www.glean.com/perspectives/security-permissions-aware-ai).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (permission-scoped retrieval, the multi-hop loop, evals, the harness).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
</content>
</invoke>
