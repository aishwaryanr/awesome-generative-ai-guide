# Coding Agent: the PM interview

## The interview question

> "Leadership wants AI to make engineering faster. You are the PM. Should we roll out a coding agent, what does success look like, and how do you measure real productivity without shipping bugs or losing developer trust?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you measure real developer productivity, and how you make the calls when the model is probabilistic and sometimes ships a wrong change. The technical design (the harness, tools, the verification loop, evals) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/coding-agent` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, developer trust, and the human fallback move from edge cases to the center of the product.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the harness, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, the number that must not break, and the vanity metric you refuse to celebrate.
- **4 Experience.** The probabilistic UX: trust, transparency, the diff you review, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, pilot, staged go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a developer, an engineering lead, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. Engineers spend a large share of the week on well-specified, low-novelty changes: fixing a failing test, wiring a small feature behind an existing pattern, bumping a dependency and repairing what breaks, closing a lint or type error. Each is minutes to hours of context-loading and mechanical edits, and each interrupts deeper work. Three users have pain: the developer buried in the mechanical middle, the reviewer who has to check whatever gets produced, and the org that wants more throughput without more headcount.

Write the outcome before the system: turn a well-specified task into a correct, reviewed pull request that passes the existing test suite, so a human reviews a diff instead of writing it. Narrow the intervention until it hurts: take a task plus a repository plus a test suite, work in a sandbox, edit files, run the tests, iterate until they pass, and open a pull request for human review, and stop there. An agent that ships to production unattended in version 1 is the over-scoping trap.

The clarifying questions you need to ask a stakeholder: what is the cost of a merged wrong change; what share of tasks are truly well-specified; what is the current cycle-time and review-burden baseline; how good is the test suite that will act as the verifier; what may the agent do without a human; and what is the promise about a human approving every merge.

> **Real anchor: the uplift is measured, and it concentrates.** In a controlled trial, developers given an AI pair-programmer completed a real task about **55% faster** than the control group ([GitHub, quantifying the impact](https://github.blog/news-insights/research/research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness/)). The gain concentrates in well-specified, lower-complexity work, which is precisely the slice this product scopes to. Match the ambition to where the measured value is, rather than to the demo.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to make a well-specified, localized change that a test suite can verify, and to iterate on test feedback until the suite is green. They are far less reliable running unbounded autonomy on large, ambiguous changes, which is why the design keeps the merge behind a human and the agent inside a sandbox. Match the ambition to what the model does reliably, gate the consequential step (the merge) behind human oversight, and let evaluation tell you where that line sits for your repository rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume, well-specified workflow in a well-tested area of the codebase, where a wrong change is caught by the suite and cheap to revert, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a real fork. Off-the-shelf coding assistants (for example GitHub Copilot, Cursor, Codex, Claude Code) get you live faster and ship their own harness, sandbox, and guardrails. Building in-house gives you control over behavior, your repository conventions, data sensitivity, and cost, and it compounds when your codebase or workflow is unusual enough that a general tool leaves value on the table. Decide on a few axes: how differentiated the workflow needs to be, how sensitive your code and secrets are, token cost against control, time to market against the data moat you would build, vendor lock-in, and whether you have the team to own an evaluation and monitoring loop for the life of the product. A common answer is to buy an off-the-shelf assistant to learn fast, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, name the number that must not break, and name the vanity metric you refuse to chase. Developer productivity is famously easy to measure badly, so lead with the frameworks that got it right: [DORA](https://dora.dev/guides/dora-metrics-four-keys/) tracks delivery through lead time for changes and change-failure rate, and the [SPACE framework](https://www.microsoft.com/en-us/research/publication/the-space-of-developer-productivity-theres-more-to-it-than-you-think/) makes the point that productivity is multidimensional and that raw activity counts (lines of code, commits) fail to capture it.

- **Task success.** Tests green and the change actually does what was asked. The core outcome.
- **Cycle time.** Lead time from task to a merged, reviewed change (DORA). The speed leadership actually cares about, measured end to end rather than at the keyboard.
- **Change-failure rate.** The share of merged changes that break something (DORA). This is the number that must stay near zero, and it can sink the project on its own.
- **Review burden.** Reviewer time per pull request and reviewer-accept rate, so speed is never bought by flooding reviewers.
- **Developer-reported friction.** The satisfaction and trust signal from SPACE, gathered from the engineers who live with the agent. A tool developers quietly route around is failing even when the dashboard looks fine.
- **Cost per resolved task.** The unit economics finance will ask about.
- **Escalation rate.** Too high means the agent is not helping; too low can mean it is forcing marginal changes through.

Separate leading signals (iterations-to-green, reviewer-accept rate) from lagging outcomes (cycle time, change-failure rate). Pair the business outcome (cycle time, throughput) with a quality metric beneath it (change-failure rate, developer-reported friction) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to refuse is raw output volume: lines of code merged, pull requests opened, or a high suggestion-acceptance rate. Those climb when the agent is genuinely productive and climb just as fast when it floods reviewers with plausible-looking changes that break things, so they measure activity while telling you nothing about value.

> **Cautionary number: perceived speed is not real productivity.** In a 2025 randomized study, experienced open-source developers using early-2025 AI tools took about **19% longer** to finish real tasks, even though they believed the tools had made them roughly **20% faster** ([METR](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)). The PM lesson is to measure delivery outcomes on real work rather than self-reported speedup, because the two can point in opposite directions.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core. Design it so a wrong change is rare, visible, and cheap to reject. This has an established design playbook: help the developer form an accurate picture of what the agent can do, make its uncertainty legible, and give a fast path to correct it or take over ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **The diff is the unit of trust.** The agent proposes a reviewable pull request a human accepts or rejects, so a wrong change is caught before it merges and costs a review comment rather than an incident. Scoped, readable diffs beat large rewrites for exactly this reason.
- **Carry the evidence.** Every change ships with its test result, so the reviewer and the agent both trust green, and the pull request description explains what changed and why and flags what the agent was unsure about, so the reviewer's mental model matches what the system actually did ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)).
- **Gate on the verifier and escalate gracefully.** When the agent cannot make the suite green or the change reaches outside its scope, it stops and hands to a human with its notes and no dead end ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). Mandatory human review before every merge is a product decision made up front rather than a fallback bolted on later.
- **Make correction cheap.** Let the developer reject, comment, or re-scope the task in one step, and route that feedback back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and cadence.** The work is asynchronous, so a task taking minutes is fine, and the experience should optimize the clarity of the diff and the reliability of the result over raw speed.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled reference dataset of real tasks (a repository snapshot, a task description, and the tests that should end green) across localized fixes, multi-file changes, tasks that should be refused, and adversarial cases, and treat it as the release gate: a change ships only if resolution rate, reliability across repeated runs, and a suite-wide regression ceiling hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product. Reliability hides in the tail: an agent that fixes a bug 3 times out of 4 and silently breaks it the fourth is unshippable on the strength of its average, so report reliability across repeated runs (on the order of pass^k, roughly the probability it succeeds every time) alongside average success.

Name the one failure that ends the project, and design against it first. Here it is a merged incorrect or insecure change reaching production: a refactor that passes the tests it touched while breaking something elsewhere, or a change that leaks a secret or obeys an instruction injected through repository content. The guardrails (the test suite as the verifier, a suite-wide regression gate, a scope check on the diff, a least-privilege sandbox with no production credentials, and mandatory human review of every merge) are product-safety features rather than engineering details, because they cap the worst case at a needless escalation rather than a bad merge. The production harnesses expose the sandbox and the approval policy as an explicit dial, so you grant the narrowest autonomy a task needs ([OpenAI Codex, approvals and security](https://developers.openai.com/codex/agent-approvals-security)).

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, team by team, with gates.

- **Shadow first.** Run the agent on real tasks and open issues without opening pull requests, and compare its would-be changes against the eval bar.
- **Pilot one team.** Turn it on for one team or one well-tested area of the codebase, read every pull request, and watch cycle time, review burden, and change-failure rate.
- **Stage the rollout** team by team behind go and no-go gates on change-failure rate, reviewer-accept rate, and developer-reported friction, with a rollback you can trigger in minutes by turning the agent off for a team. The test suite and mandatory code review are the standing guardrails on every merge.
- **Know when to pull back.** If change-failure rate rises, if review burden grows faster than throughput (a sign reviewers are rubber-stamping), or if developer trust drops, pause and diagnose before widening. Measure real delivery, so aggressive rollout never quietly trades a bug spike or reviewer fatigue for a throughput headline.

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with cycle time paired with change-failure rate, add review burden and developer-reported friction, and refuse to celebrate lines of code, pull-request counts, or a high acceptance rate, because those measure activity. Measure delivery outcomes on real work rather than self-reported speedup, since developers can feel faster while finishing slower.

**"The agent merges a change that breaks production. Walk me through your response."** Contain the impact first (revert the change), trace the failure to the step that broke (a thin test, an over-broad diff, a missed regression), add the case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern that should pause the rollout. The audit trail and the human review path are what make this recoverable.

**"Engineering says 6 months. Scope an MVP."** Cut to localized, test-covered fixes on one team, pull-request-only with a human merge and no autonomous production changes. Prove cycle time and change-failure rate there before adding multi-file changes and refactors. The follow-ups add capability the way versions would, without betting the launch on the hardest slice.

**"How do you choose the autonomy level?"** Match autonomy to blast radius. Let the agent open a pull request on test-covered localized fixes, require a human review before every merge, and keep large refactors and infrastructure changes behind tighter oversight. Move the line outward only when evaluation on real tasks earns it.

**"Leadership frames this as making engineering faster. How do you set the goal?"** Frame it as throughput and focus: offload the mechanical middle so engineers spend more time on design and the hard problems, with change-failure rate and developer trust as guardrails. The METR result is the cautionary tale for a pure speed framing, because perceived speed and real delivery can diverge.

**"Build or buy?"** Decide on differentiation, code and secret sensitivity, and whether you can own an evaluation loop for the product's life. Buy an off-the-shelf assistant to learn fast, then build the parts that become your advantage.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf).
- **Measuring developer productivity:** the [DORA four keys](https://dora.dev/guides/dora-metrics-four-keys/), the [SPACE framework of developer productivity](https://www.microsoft.com/en-us/research/publication/the-space-of-developer-productivity-theres-more-to-it-than-you-think/), and the [GitHub Copilot controlled trial](https://github.blog/news-insights/research/research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness/).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **The cautionary arc:** the [METR 2025 developer-productivity study](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (the harness, tools, the verification loop, evals).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
</content>
</invoke>
