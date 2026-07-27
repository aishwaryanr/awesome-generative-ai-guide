# Clinical Documentation Assistant: the PM interview

## The interview question

> "Clinicians spend hours on notes and are burning out. You are the PM. Should we deploy an AI scribe, what does success look like, and how do you keep it safe enough for clinical use?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk it, and how you make the calls when the model is probabilistic and sometimes wrong. The technical design (the transcription-to-SOAP pipeline, the faithfulness check, evals, the harness) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome. As in the engineering version, every transcript here is synthetic, and the system is built and tested on mock encounters so no real patient data is ever in the loop.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/clinical-scribe` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product. In a clinical setting that shift is sharpest of all, because a made-up symptom or dose in a note is a safety event and a liability event at once.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, and the number that must not break.
- **4 Experience.** The probabilistic UX: trust, transparency, provenance, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, pilot, staged release, go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a clinician, an engineer, and a compliance officer in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to a clinical feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a health system owns. Clinicians spend a large share of the day on the electronic health record and desk work, much of it after hours as unpaid documentation time, and documentation burden is a leading driver of burnout. Two users carry the pain: the clinician who loses evenings to notes and attention to the screen instead of the patient, and the health system that pays for it in burnout, turnover, and throughput. There is a third stakeholder whose trust is the whole ballgame, the patient whose record has to be right.

Write the outcome before the system: draft a faithful, structured note from the encounter so the clinician reviews, edits, and signs it in a fraction of the time, cutting after-hours documentation and giving attention back to the room, while a hard rule keeps any unreviewed note out of the record. Narrow the intervention until it hurts: turn the transcript into a structured draft, flag anything the transcript does not support, and hand it to the clinician. Stop well short of auto-filing notes, placing orders, or making clinical decisions. A scribe that tries to code the visit for billing, place orders, and draft the note in version 1 is the over-scoping trap.

> **Real outlier: adoption is real, and it is uneven.** A study across 5 academic medical centers found clinicians using an ambient AI scribe saved about 16 minutes of documentation time and spent about 13 fewer minutes in the record for every 8 hours of patient care, enough to see roughly one extra patient every two weeks. The same reporting stressed that the time savings were modest and adoption inconsistent, and that clinicians needed support to get value from the tool. Relief of documentation burden is real, and it arrives unevenly. That is the gap between a demo and a product. [[STAT](https://www.statnews.com/2026/04/01/ai-ambient-scribes-modest-time-savings-clinical-documentation/)] [[AMA on time returned to clinicians](https://www.ama-assn.org/practice-management/digital-health/ai-scribes-save-15000-hours-and-restore-human-side-medicine)]

The clarifying questions you need to ask a stakeholder: what is the tolerance for a missed detail versus an invented one; what specialty and note format are we starting with; does the note write back into the electronic health record and through what interface; who signs and what does the sign-off legally mean; what are the privacy, retention, and regulatory constraints; and, for the build itself, the discipline that this is developed on synthetic encounters so no real protected health information is ever exposed. In this case study the system uses synthetic data only.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to transcribe a conversation and structure it into a draft note a clinician can edit. They are far less reliable at guaranteeing that every line is faithful to what was said, which is exactly why the design pairs a faithfulness check with mandatory clinician review and keeps the write to the record behind a human sign-off. Match the ambition to what the model does reliably, gate the consequential step (a note entering the chart) behind human oversight, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor one specialty with a high documentation load and a clean note format, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a real fork, and for a scribe it leans harder toward buy than most cases. Ambient documentation vendors (for example Abridge, DAX Copilot, Nabla, Suki, Ambience) get you live faster and carry their own accuracy work, electronic-health-record integrations, and compliance posture. Building in-house gives you control over behavior, data, and cost, and it compounds when documentation is core to your product. Decide on a few axes: how differentiated the experience needs to be, how sensitive the data and the regulatory exposure are, the cost and effort of the electronic-health-record integration, token cost against control, time to market against the data advantage you would build, vendor lock-in, and whether you can own an evaluation and monitoring loop for the life of the product, because faithfulness has to be measured continuously regardless of who built the model. A common answer is to buy to learn fast, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, and name the number that must not break.

- **Documentation time saved and time-to-sign.** Minutes returned to the clinician per encounter and per shift, including after-hours record time. The core business outcome.
- **Sustained adoption by specialty.** The share of eligible clinicians who keep using it after the novelty fades, read per specialty because the note format lands differently in each.
- **Clinician edit distance and acceptance rate.** How much rewriting the draft still needs before it is signed, which is quality as the clinician feels it.
- **Fabrication rate in the note.** The hard ceiling. An invented symptom, finding, or medication is the number that must stay near zero, and it can sink the project on its own.
- **Omission rate.** The quiet failure: a real symptom, medication, or instruction dropped from the note. It pulls against the fabrication rate, so you watch both.

Separate leading signals (faithfulness on a sample, unsupported-statement flags) from lagging outcomes (adoption, burnout, retention). Pair the business outcome (time saved, adoption) with a quality metric beneath it (fabrication rate, edit distance) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to watch is time saved bought at the cost of trust: a scribe that saves minutes because clinicians rubber-stamp the draft without truly reading it looks great on a dashboard and quietly ships fabrications into the record. Real review has to stay real, so pair every time-saved number with a fabrication and edit-rate check that would catch review theater.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core. Design it so a wrong line is rare, visible, and cheap to correct before anyone signs. This has an established design playbook: help the clinician form an accurate picture of what the scribe does, make its uncertainty legible, and give a fast path to fix it ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Ground every statement to the transcript.** Each line in the draft carries the span of the encounter that supports it, so a clinician (and later an auditor) can click any statement and see the words that justify it.
- **Flag the unsupported and make review the default.** When a statement is not supported by the transcript, the scribe surfaces it for review rather than burying it, and nothing reaches the record without a clinician sign-off ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). The guarantee that a human reviews and signs every note is a product decision made up front rather than a control bolted on later.
- **Set expectations.** Make it plain that the output is a draft the clinician owns and edits, so the clinician's mental model matches what the system actually does ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)), and the sign-off feels like the point of the workflow rather than a speed bump.
- **Make correction cheap.** Let the clinician edit a line in place, delete an unsupported statement, or fix a flipped negation in one step, and route that correction back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and tone.** The note should be ready when the visit ends, and the draft should read in the clinician's documentation style so editing is faster than composing from scratch.

The human handoff is the load-bearing wall here. The clinician sign-off is the required control that makes every other design choice safe, because it caps the worst case at a flagged line a clinician removes rather than a fabricated statement in a signed record.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of synthetic encounters (clean cases, cases with negations, cases with dropped details, and adversarial cases where a tempting fabrication is easy to make) and treat it as the release gate: a change ships only if the fabrication rate, the omission rate, and per-statement faithfulness hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a fabricated symptom or medication that reaches a signed note and drives care: a wrong dose, an invented complaint, a medication the patient never mentioned. That single event is a patient-safety incident, a liability exposure, and a regulatory problem at once, and it is the failure that gets a clinical AI pulled. The guardrails (a faithfulness check on every statement, an unsupported-statement flag that routes to review, mandatory sign-off, and an immutable audit trail of who reviewed what and when) are product-safety features rather than engineering details, because they cap the worst case at a flagged line a clinician deletes.

> **Real outlier: fabrication is not hypothetical.** An audit by Ontario's auditor general of AI scribe systems approved for provincial use found that about 45% of the systems fabricated information or suggested treatments never discussed, about 60% recorded medications different from what the clinician prescribed, and about 85% missed critical mental-health details raised in the conversation. Invented content, wrong doses, and dropped details are the three failure modes a scribe has to engineer against directly, which is why the fabrication ceiling and the human sign-off are the load-bearing product decisions. [[Global News on the Ontario auditor general report](https://globalnews.ca/news/11844349/ontario-auditor-general-ai-usage/)]

### 6 Rollout

Ship it the way you de-risk any high-stakes clinical launch, in stages with gates, one specialty at a time.

- **Shadow first.** Run the scribe on real encounters and generate drafts that stay out of the record, compare them against the clinician's own note and the eval bar, and read the fabrications and omissions before a single note is signed from the system.
- **Pilot a specialty.** Turn it on for one specialty with a clean note format, read every note, and watch the metrics from section 3, especially the fabrication rate and the edit distance.
- **Stage by specialty** behind go and no-go gates on time saved, adoption, and the fabrication ceiling, with a rollback plan you can trigger in minutes and a per-specialty eval set before each expansion.
- **Monitor and keep the sign-off mandatory.** Sample live notes for fabrication and omission drift, watch for review theater (drafts signed with near-zero edits and no reads), and keep the clinician sign-off as a hard requirement that cannot be switched off under time pressure.

Pull back when the fabrication rate drifts above its ceiling in a specialty, when clinicians start rubber-stamping instead of reviewing, or when a single fabricated statement reaches a signed note. Each of those is a gate failure that pauses the rollout while you trace it, add the case to the eval set, and fix the step that broke.

> **Real arc: adoption is real, fabrication is real, and both are uneven.** The 5-center study above shows the upside is genuine and modest, and it does not arrive on its own. The Ontario audit shows deployed systems fabricating content, recording wrong medications, and dropping critical details. The PM lesson is to set the fabrication ceiling and the mandatory-review guarantee on day one, and to expand by specialty behind those gates, so aggressive time-savings never quietly cost you a wrong record. [[STAT](https://www.statnews.com/2026/04/01/ai-ambient-scribes-modest-time-savings-clinical-documentation/)] [[Ontario auditor general](https://globalnews.ca/news/11844349/ontario-auditor-general-ai-usage/)]

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with documentation time saved paired with sustained adoption and clinician edit distance, gate on the fabrication ceiling, and refuse to celebrate time savings that come from clinicians skipping review. Report reliability, since a scribe that writes a faithful note most of the time and invents a dose occasionally is the failure that harms a patient, so pass^k on note quality matters more than the average.

**"The scribe puts a medication the patient never mentioned into a signed note. Walk me through your response."** Contain the patient impact first and correct the record through the proper clinical process, then trace the failure to the step that broke (transcription, extraction, or the faithfulness check), add the case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern that should pause the rollout in that specialty. The audit trail of who reviewed and signed, plus the mandatory sign-off, are what make this traceable and recoverable.

**"Engineering says 6 months. Scope an MVP."** Cut to one specialty, drafting the note from the transcript with unsupported statements flagged and mandatory clinician review, no write-back to the electronic health record and no orders. Prove time saved and a fabrication rate near zero there before adding terminology grounding, record reconciliation, and eventually a gated write-back. The follow-ups add capability the way versions would, without betting the launch on the hardest slice.

**"How do you choose the autonomy level?"** Match autonomy to the cost of being wrong. The scribe drafts and flags, the clinician reviews and signs, and the write to the record stays behind that sign-off. Auto-filing a note or placing an order is the high-consequence action you keep a human in front of, and you move the line outward only when evaluation on live traffic earns it, which for a signed clinical record is a high bar.

**"Leadership frames this as seeing more patients per hour. How do you set the goal?"** Frame it as burnout relief and attention returned to the room: give clinicians their evenings back and their focus back to the patient, with the fabrication ceiling and a guaranteed real review as guardrails. The Ontario audit is the cautionary tale for a pure throughput framing, because a scribe pushed to save time by cutting review is exactly how fabricated content reaches the record.

**"Build or buy?"** Decide on differentiation, data and regulatory sensitivity, the cost of the electronic-health-record integration, and whether you can own an evaluation loop for the product's life. Buy to learn fast against an ambient documentation vendor, then build the parts that become your advantage, and either way own the faithfulness measurement yourself.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **Ambient clinical documentation, adoption and safety:** STAT, [the 5-center ambient scribe study](https://www.statnews.com/2026/04/01/ai-ambient-scribes-modest-time-savings-clinical-documentation/), and [AMA on time returned to clinicians](https://www.ama-assn.org/practice-management/digital-health/ai-scribes-save-15000-hours-and-restore-human-side-medicine); Global News on [the Ontario auditor general report](https://globalnews.ca/news/11844349/ontario-auditor-general-ai-usage/); the [randomized trial of two scribes](https://pmc.ncbi.nlm.nih.gov/articles/PMC12265753/) and the [6-scribe competitive analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309782/); and TMLT on [why clinician review and sign-off are load-bearing](https://www.tmlt.org/resource/using-ai-medical-scribes-risk-management-considerations).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (the transcription-to-SOAP pipeline, the faithfulness check, evals, the harness).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
</content>
</invoke>
