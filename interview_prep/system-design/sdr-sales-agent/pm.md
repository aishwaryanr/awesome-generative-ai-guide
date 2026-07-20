# SDR (Sales-Development) Agent: the PM interview

## The interview question

> "Reps spend more time researching than selling. You are the PM. Should we deploy an AI SDR to qualify and draft outreach, what does success look like, and how do you protect the brand and stay compliant?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk it, and how you make the calls when the model is probabilistic and sometimes wrong. The technical design (enrichment tools, the qualification score, evals, the brand-and-compliance guardrail) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/sdr-sales-agent` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product. For an SDR agent that writes to real prospects under your brand, that center is where the money and the risk both sit.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust or the brand. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, the number that must not break, and the anti-metric.
- **4 Experience.** The probabilistic UX: grounding, transparency, brand voice, and the human handoff before any send.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, then a pilot on one segment, staged go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a customer, an engineer, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. A sales-development rep works inbound leads (a form fill, a demo request, a content download) into qualified conversations. A rep handles on the order of a few dozen to a few hundred leads a week and spends several minutes researching each one across the CRM and the web before writing a first-touch email. Speed to that first touch is a large share of whether a lead ever converts, so the good leads go cold in the backlog while the rep grinds through research. Two users have pain: the rep buried in repetitive research, and the prospect whose interest cools before anyone reaches out.

Write the outcome before the system: for every inbound lead, produce a qualification decision the rep trusts and, for the leads worth pursuing, a personalized draft ready for a human to approve and send. Narrow the intervention until it hurts: enrich the lead, score it against your ideal customer, and draft one grounded first-touch message, staying well short of an autonomous machine that emails on its own. The agent proposes and a human disposes. An SDR agent that also runs the whole sequence, handles objections, and manages the deal in version 1 is the over-scoping trap.

The clarifying questions you need to ask a stakeholder: what is your ideal customer profile and how is a lead scored today; what counts as a qualified lead; which outreach regulations apply and in which countries; may the agent ever send autonomously or is human approval required; what channels (email, LinkedIn, both); how fresh is the CRM and enrichment data; and what is the one failure the business will not tolerate. That last answer, for almost every company, is a fabricated claim about a prospect or a compliance violation, and it shapes the rest of the design.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to enrich a lead, score it against a clear scorecard, and draft a personalized message from verified signals with a human on the send. They are far less reliable running unbounded autonomy that fires email at prospects on their own, which is why the design keeps the send as a human action. Match the ambition to what the model does reliably, gate the consequential step behind human oversight, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume repetitive workflow with a bounded cost of being wrong, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)). Inbound qualification and first-touch drafting fit, because a human reviews every message before it leaves.

Build or buy is a real fork, and the market is crowded. Sales-engagement platforms (Outreach, Salesloft) and enrichment and data tools (Apollo, Clay, ZoomInfo) already run sequencing, deliverability, and suppression, and a wave of AI-SDR products layer drafting on top. Buying gets you live faster and inherits their compliance plumbing and evals. Building in-house gives you control over qualification logic, brand voice, data, and cost, and it compounds when outbound is core to how you grow. Decide on a few axes: how differentiated the outreach needs to feel, how sensitive your CRM data and the compliance exposure are, token cost against control, time to market against the data moat you would build from your own closed-won history, vendor lock-in and who owns your sender reputation, and whether you have the team to own an evaluation and monitoring loop for the life of the product. A common answer is to buy to learn fast on a segment, then build the qualification and grounding that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, name the number that must not break, and name the anti-metric that catches gaming.

- **Qualified pipeline generated.** The dollars of qualified pipeline the agent sources that reps then work. The core business outcome the whole system is judged on.
- **Reply rate and positive-reply rate.** Reply rate is the share of sent messages that get any answer; positive-reply rate is the share that answer with interest. Positive-reply rate is the honest quality signal, because volume without positive replies is just noise in a prospect's inbox.
- **Qualification precision on a sample.** Of the leads the agent passes to a rep, how many were actually worth the rep's time, checked against what converted.
- **Deliverability and sender reputation.** Bounce rate, spam-complaint rate, and inbox placement. This is the plumbing every reply rate depends on, and it degrades quietly across the whole sending domain.
- **Fabricated-claim rate and compliance-violation rate.** The hard ceiling. A confident wrong claim about a prospect, or an email to a suppressed contact with no lawful basis, is the number that must stay near zero, and either one can sink the project on its own.
- **Cost per qualified lead.** The unit economics finance will ask about.

Separate leading signals (grounded-draft rate, deliverability, draft-acceptance rate) from lagging outcomes (positive replies, meetings booked, pipeline). Pair the business outcome (pipeline generated) with a quality metric beneath it (positive-reply rate, qualification precision) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to watch is volume bought at the cost of the brand: an agent that lifts sent-message count while spam complaints and unsubscribes climb looks busy on a dashboard and is burning your domain reputation and your brand in every inbox it reaches.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core, and here that experience has two audiences: the rep who approves the draft, and the prospect who receives it. Design it so a wrong or off-brand message is rare, caught before it sends, and cheap to correct. This has an established design playbook: help the rep form an accurate picture of what the agent can do, make its uncertainty legible, and keep a fast path to fix or reject ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Ground every claim and show its source.** Each personalized line in the draft carries the verified signal it came from, so a rep and an auditor can both see why the message says what it says. A claim with no verified signal behind it never reaches the draft.
- **Human approval is a core product surface.** Every draft lands in an approval queue where a person reviews and sends. The promise that no message reaches a prospect without a human is a product decision made up front rather than a safety net bolted on later, and it is what keeps a jailbroken or hallucinated draft from ever leaving the building ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)).
- **Set the rep's expectations.** Show the qualification score with its reason codes and the draft with its evidence, so the rep's mental model matches what the system actually did ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)). A score the rep can read builds trust faster than a bare number.
- **Make correction cheap and route it back.** Let the rep edit a line, reject a draft, or fix a qualification call in one step, and feed those edits and rejections back into evaluation, because a heavy rep edit is one of the strongest signals of a quality problem ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and voice.** A draft ready within minutes is fine here, and the voice should match your brand and read as written for this person rather than as a mass blast.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of real leads across strong-fit, borderline, poor-fit, opted-out, and adversarial (injected enrichment) cases, and treat it as the release gate: a change ships only if grounded-draft rate, qualification precision, and the fabrication and compliance ceilings hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a fabricated claim about a prospect (congratulating someone on funding they never raised or a role they do not hold) or a compliance violation (emailing a contact who opted out, or sending with no lawful basis and no opt-out). Either one destroys trust at brand scale and creates legal exposure, and the danger is that both scale with volume, so a single bad behavior becomes a brand-wide incident fast. The guardrails (grounding every claim in a verified signal, a claim-check before any human sees the draft, an authoritative suppression check, and human approval on every send) are product-safety features rather than engineering details, because they cap the worst case at a rejected draft rather than a wrongful send.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates, on one segment before the whole funnel.

- **Shadow first.** Run the agent alongside reps on real inbound without sending anything, and compare its qualification calls and its drafts against the eval bar and against what reps actually do.
- **Pilot one segment.** Turn it on for a single segment (one region, one product line, or one lead source), read every draft before it goes out, and watch the metrics from section 3 on a small, dedicated sending setup so a problem cannot touch your main domain.
- **Stage the rollout** behind go and no-go gates on pipeline, positive-reply rate, deliverability, and the fabrication and compliance ceilings, with a rollback you can trigger in minutes.
- **Monitor and keep the human on the send.** Sample live traffic, watch spam complaints and bounces for drift, and keep human approval on every message as you widen the segments.

> **Real arc: the payback is fast, and that is exactly why the guardrails come first.** Vendor and market benchmarks put the payback on an AI-SDR deployment at roughly 3 months (one widely cited figure is 3.2 months) against on the order of 8.7 months for a ramped human rep, because the agent generates pipeline in days rather than after a hiring-and-training cycle. Read these as directional vendor numbers rather than audited results. The PM lesson is that the speed comes from volume, and volume is precisely what turns one bad behavior, a fabricated claim or an email to someone who opted out, into a repeated, brand-wide incident. So you set the fabrication ceiling, the compliance ceiling, and the human-on-the-send guarantee on day one, and you prove them on one segment before you scale. [[UserGems](https://www.usergems.com/blog/are-ai-sdrs-worth-it)]

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with qualified pipeline paired with positive-reply rate, gate on the fabrication and compliance ceilings, and refuse to celebrate sent-message volume while spam complaints and unsubscribes climb. Report deliverability alongside reply rate, because an agent that emails more can lift raw replies while quietly degrading the domain everyone sends from.

**"The agent sends a prospect a message with a made-up claim about their company. Walk me through your response."** In this design it should be caught before send, so first confirm why the guardrail missed it: trace the failure to the step that broke (enrichment returned a stale field, grounding let an unverified signal through, or the claim-check has a gap). Contain it by pausing that segment, add the case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern. The evidence trail on every claim and the human on every send are what make this recoverable rather than a public incident.

**"Engineering says 6 months. Scope an MVP."** Cut to inbound qualification plus a drafted first-touch message, human approval on every send, one segment, one channel, one compliance profile. Prove qualification precision, positive-reply rate, and the fabrication and compliance ceilings there before adding channels, autonomy, or higher volume. The follow-ups add capability the way versions would, without betting the launch on the hardest slice.

**"How do you choose the autonomy level?"** Match autonomy to the cost of being wrong. Auto-enrich and auto-score, draft with grounding, and require a human on every send. Move the line outward (say, auto-sending low-risk follow-ups to warm, already-engaged contacts) only when evaluation on live traffic earns it, and keep the fabrication and compliance ceilings as the gate.

**"Leadership frames this as replacing the SDR team. How do you set the goal?"** Frame it as capacity and speed: take the research grind off reps so they spend their time selling and working the qualified pipeline the agent surfaces, with positive-reply rate, deliverability, and a human on every send as guardrails. A pure headcount framing pushes for volume, and volume without those guardrails is exactly what torches sender reputation and the brand.

**"Build or buy?"** Decide on differentiation, data and compliance sensitivity, who owns your sender reputation, and whether you can own an evaluation loop for the product's life. Buy to learn fast on a segment, then build the qualification and grounding that become your advantage.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **Outreach compliance:** the FTC, [CAN-SPAM Act: A Compliance Guide for Business](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business), and for the EU and UK the ICO on [legitimate interests](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/a-guide-to-lawful-basis/lawful-basis-for-processing/legitimate-interests/) and [GDPR Article 6](https://gdpr-info.eu/art-6-gdpr/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (enrichment tools, the qualification score, evals, the brand-and-compliance guardrail).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
</content>
</invoke>
