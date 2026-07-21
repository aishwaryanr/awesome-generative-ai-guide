# Document Decisioning for Insurance Underwriting: the PM interview

## The interview question

> "Underwriting takes weeks and we are losing deals to slow quotes. You are the PM. Should we automate the document decision, what does success look like, and how do you satisfy regulators while cutting turnaround?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk a decision that binds real money, and how you make the calls when the model is probabilistic and a wrong decision reaches a regulator. The technical design (extraction, deterministic policy, evals, the audit trail) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/document-decisioning` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product. In a regulated decision like underwriting, one more thing moves to the center: every decision has to be explainable and reconstructable long after it is made.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without binding a risk it should have declined. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, the number that must not break, and the anti-metric that catches a gamed dashboard.
- **4 Experience.** The probabilistic UX: confidence, transparency, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Refer-only first, staged release behind approval, go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a broker, an underwriter, a compliance officer, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. A new-business submission arrives as a pile of documents: an application form, a broker email, a loss run (the history of past claims), inspection reports, financial statements, often scanned and in different formats. An underwriter reads and keys these facts by hand, and a submission can sit in a queue for weeks before anyone touches it. Weeks of turnaround is where deals are lost: the broker places the risk with whoever quotes first. Two users have pain: the broker and their client waiting on a quote, and the underwriter buried in routine document reading who has less time for the genuinely hard risks.

Write the outcome before the system: compress the read-and-decide cycle on the routine submissions so an underwriter spends their time on the risks that need judgment, measured by the share of submissions decided straight through without a human keying facts, the accuracy of the extracted facts, the turnaround time, and, above all, the rate of wrong decisions, tracked separately for wrong approvals and wrong declines because they cost different things. Narrow the intervention until it hurts: read the documents, extract a defined set of underwriting facts with a confidence on each, apply the written policy, and produce one of 3 outcomes, approve, decline, or refer to a human. An agent that tries to bind coverage on its own, handle new business and renewals and claims at once, and price the risk in version 1 is the over-scoping trap.

The clarifying questions you need to ask a stakeholder: what is the cost of a wrong approval versus a wrong decline; what document types arrive and how many are scanned versus digital; which decisions may the agent make on its own and above what value a human must sign off; how current must the underwriting policy be; and what does the regulator require you to prove after the fact. Their answers set every later tradeoff.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to read facts out of a messy document with a confidence on each, and to apply written rules once the facts are clean. They are far less reliable running unbounded autonomy on a high-impact action, which is why the design keeps the bind behind deterministic rules and a human. Split the work by how much guarantee it needs: the hard rules (appetite limits, authority limits, over-insurance checks) run as deterministic code so they behave the same way every time, and the model is reserved for the borderline approve-or-refer judgment on cases the rules already cleared. Match the ambition to what the model does reliably, gate the consequential action behind human oversight, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, use deterministic code where behavior must be guaranteed, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume, in-appetite workflow where the facts are cleanest, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a real fork, and the regulated setting raises the stakes on it. Off-the-shelf document-intelligence and underwriting-automation platforms get you live faster and carry their own extraction models and connectors. Building in-house gives you control over the exact fact schema, the policy code, and the audit trail, and it compounds when underwriting judgment is core to your book. Decide on a few axes: how differentiated the decision needs to be, how sensitive the data and the action are (submissions carry names, government IDs, and financials, and the action binds money), token cost against control, time to market against the data moat you would build from your own labeled submissions, vendor lock-in, and whether you can own an evaluation and monitoring loop for the life of the product in a domain where you have to defend every decision. A common answer is to buy the commodity extraction to learn fast, then build the parts that become differentiating, the policy logic and the audit layer, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, and name the number that must not break.

- **Straight-through-processing rate.** The share of submissions decided without a human keying facts. The core business outcome.
- **Turnaround time.** Parse-to-decision time against the old baseline, the number the broker feels and the reason deals were being lost.
- **False-approve rate and false-decline rate, tracked separately.** A single "decision accuracy" number hides the whole problem, because a wrong approval and a wrong decline cost different things. A wrong approval binds a risk policy would decline and puts money on a claim you should never have taken. A wrong decline is lost business, and a pattern of them is a fair-treatment exposure.
- **Incorrect-bind rate.** The hard ceiling. Binding a risk the policy declines is the number that must stay near zero, and it can sink the project on its own.
- **Per-field extraction accuracy.** The whole decision rests on the facts being read correctly, so this is watched as a leading signal beneath the outcome.
- **Referral rate and cost per decided submission.** Watch both ends: too many referrals means the automation is barely saving work, and too few means it is over-automating and hiding failures. Cost per decided submission is the unit economics finance will ask about.

Separate leading signals (extraction accuracy, referral precision) from lagging outcomes (turnaround, straight-through rate). Pair the business outcome (straight-through rate, turnaround) with a quality metric beneath it (per-field accuracy, the false-approve and false-decline rates) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The number that must not break is the false-approve rate. The anti-metric to watch is straight-through rate bought at the cost of correct binds: an agent that hits a throughput target by approving borderline risks, or by referring so aggressively that the automation delivers no real speed, looks fine on a dashboard and quietly costs the book.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core, and here the primary user is the underwriter who owns the decision, with the broker and their client downstream. Design it so a wrong read is rare, visible, and cheap to recover from. This has an established design playbook: help the user form an accurate picture of what the agent can do, make its uncertainty legible, and give a fast path to correct it or reach a person ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Show the evidence for every fact.** Each extracted field links back to the document and page it came from and carries a confidence, so an underwriter and an auditor can both trust it and check it.
- **Gate on confidence and refer gracefully.** When a field falls below a confidence floor, is out of range, or conflicts with another document, the submission routes to a human with the facts already laid out and no dead end ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). Above the delegated-authority limit, a human signs off before coverage binds. Human sign-off on the high-impact decision is a product decision made up front rather than a fallback bolted on later.
- **Set expectations.** Make it clear which facts the agent read, which rule fired, and why a case was referred, so the underwriter's mental model matches what the system actually does ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)), and the handoff feels like a prepared file rather than a wall.
- **Make correction cheap.** Let the underwriter fix a misread field in one step, and route that correction back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency.** The old baseline was weeks, so a decision within minutes is transformative, which frees the design to spend time on careful extraction and verification rather than shaving seconds.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled reference set of real submissions across clean approvals, clear declines, borderline referrals, low-quality scans, and adversarial documents, and treat it as the release gate: a change ships only if per-field extraction accuracy, the false-approve ceiling, the false-decline rate, and referral precision hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product. Report reliability alongside the average: an agent that reads a submission right most of the time yet wrong 1 in 4 falls short of shippable, so you gate on consistency across repeated runs, and the confidence floor turns the shaky cases into referrals.

Name the one failure that ends the project, and design against it first. Here it is a wrong bind, approving a risk the policy would decline, or a decision the business cannot explain and reconstruct when a regulator asks 6 months later. The guardrails (a confidence floor on every extracted field, deterministic hard rules the model cannot override, human sign-off above authority, and an immutable audit trail behind every decision) are product-safety features rather than engineering details, because they cap the worst case at an unnecessary referral rather than a wrongful bind.

Fairness and bias sit inside this layer, because they are a failure mode with its own cost. Watch the false-decline rate across segments, since a pattern of wrong declines on a protected-adjacent group is both lost revenue and a discrimination exposure, and keep the decision explainable so the reason for every decline is a rule or a stated fact rather than an opaque score. The regulated setting sets the floor here: the [EU AI Act](https://artificialintelligenceact.eu/annex/3/) classifies AI used for risk assessment and pricing in life and health insurance as high-risk, which carries explicit obligations for human oversight, record-keeping, and the ability to override an automated output. The audit trail and the sign-off are how you meet them, and frameworks like the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) map the same practices onto US deployments.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates, and start where a wrong decision is impossible.

- **Shadow first.** Run the agent alongside underwriters on real submissions without acting on its output, and compare its extraction and its proposed decision against the eval bar.
- **Refer-only next.** Turn it on to read, extract, verify, and assemble a prepared case file, then refer every submission to a human who decides. This proves per-field accuracy and the turnaround win with zero decision risk, because no bind is ever automated.
- **Automated decisions behind approval.** Once refer-only clears the bar, enable approve and decline on the cleanest in-appetite, in-authority slice, with human sign-off on anything above authority, and stage the rollout behind go and no-go gates on the straight-through rate, the false-decline rate, and the false-approve ceiling, with a rollback you can trigger in minutes.
- **Monitor and keep the human path.** Sample live decisions, watch for drift as new document formats and new brokers arrive, and pull back the moment the false-approve rate or a false-decline pattern crosses its threshold.

> **Real arc: PwC and Anthropic, 2025.** In their expanded enterprise alliance, PwC reported a live deployment where agentic AI compressed insurance underwriting cycles from roughly ten weeks to ten days, in their words "opening lines of business that were not previously economically viable," and Anthropic's CEO put the same number plainly. The PM lesson is in PwC's own framing: this is a domain where accuracy and reliability are non-negotiable, so the speed is only worth anything if the extraction is right and every decision can be defended later. Set the false-approve ceiling and the human sign-off on day one, so a faster cycle never quietly binds a risk it should have declined. [[PwC and Anthropic](https://www.pwc.com/us/en/about-us/newsroom/press-releases/anthropic-pwc-expand-alliance-agentic-enterprise.html)]

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with the straight-through rate and turnaround paired with per-field accuracy, gate on the false-approve ceiling, and watch the false-decline rate as closely as the false-approve rate. Refuse to celebrate a straight-through number that comes from approving borderline risks to hit throughput, or from referring so much that the automation delivers no real speed. Report reliability across repeated runs, since an agent that is right most of the time can still bind wrong often enough to sink the book.

**"The agent binds a risk it should have declined. Walk me through your response."** Contain the exposure first, halt the automated path and flag the bind for review while you assess whether it is reversible. Pull the immutable audit record and trace the failure to the step that broke, an extraction error, a hard rule that did not fire, or a model judgment that should have referred. Add the case to the eval set so it cannot regress, then decide whether it was a one-off or a pattern that should pause the automated decisions and drop back to refer-only. The audit trail and the human path are what make this recoverable.

**"Engineering says 6 months. Scope an MVP."** Ship refer-only on one line of business with the cleanest documents: read, extract with confidence, verify, and hand the underwriter a prepared file, with no autonomous decisions at all. Prove per-field accuracy and the turnaround win there before enabling approve and decline on the in-appetite slice. The later stages add autonomy the way versions would, without betting the launch on the hardest decision.

**"How do you choose the autonomy level?"** Match autonomy to the cost of being wrong and the reversibility of the action. Auto-decide clean, in-appetite, in-authority submissions where the facts read cleanly, let the model choose only between approve and refer so it can never bind a risk the rules declined, and require a human for anything above the authority limit or with a shaky read. Move the line outward only when evaluation on live traffic earns it.

**"A regulator asks you to prove why a specific application was declined 6 months ago. Can you, and how do you keep the decision fair?"** Yes, and the design is what makes the answer yes. Pull the immutable, hash-chained audit record and it reconstructs the whole decision: the source documents, the exact fields and their confidences, the model and policy versions in force that day, the rule that fired or the facts the model weighed, and the human who signed off if it was above authority. Fairness rides on the same design: because the reason for every decline is a stated rule or fact, you can audit decline patterns across segments and watch the false-decline rate for a fair-treatment problem before it becomes a regulatory one.

**"Build or buy?"** Decide on differentiation, data and action sensitivity, and whether you can own an evaluation and audit loop for the product's life in a regulated domain. Buy the commodity document extraction to learn fast, then build the parts that become your advantage and your compliance surface, the policy logic and the immutable audit trail.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **Regulated decisioning:** the [EU AI Act high-risk list](https://artificialintelligenceact.eu/annex/3/) and the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), and the [PwC and Anthropic underwriting deployment](https://www.pwc.com/us/en/about-us/newsroom/press-releases/anthropic-pwc-expand-alliance-agentic-enterprise.html) for the ten-weeks-to-ten-days outcome.

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (extraction, deterministic policy, evals, the audit trail).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
