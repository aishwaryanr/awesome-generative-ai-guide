# Real-Time Voice Support Agent: the PM interview

## The interview question

> "Phone wait times are hurting CSAT. You are the PM. Should we put an AI voice agent on the line, what does success look like, and how do you keep it from feeling like a worse IVR?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether an AI voice agent should ever pick up the phone, what good sounds like on a live call, how you de-risk it, and how you make the calls when the model is probabilistic, sometimes wrong, and every second of its thinking is silence the caller hears. The technical design (the real-time pipeline, streaming, endpointing, barge-in, evals) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/voice-support-agent` and have it quiz you on the product decisions, pressure-test your metrics and rollout plan, or role-play the interviewer pushing on latency and the human handoff.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product. On a voice product one more thing changes: the wall clock joins the core, because a caller cannot skim, and dead air on the line reads as a dropped call.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, the number that must not break, and the anti-metric.
- **4 Experience.** The probabilistic UX: the latency and turn-taking feel, trust, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, pilot, staged gates, go and no-go, and when to pull back.

Product management is composing these into a decision you can defend to a customer, an engineer, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. Callers wait on hold at peak, punch through touch-tone menus, and repeat themselves to each agent. Most calls are the same routine handful (order status, store hours, return policy, password reset), a few minutes each, and a human-handled call runs on the order of several dollars in fully loaded cost while an automated one is cents. Two users have pain: the caller stuck on hold, and the agent buried in repetitive calls who could be handling the hard and emotional ones. The pain that is specific to voice is time. A caller cannot scan a screen while they wait, so every second of silence while the system thinks feels like the line went dead, and a menu that loops a caller back to the start is the exact experience you are trying to end.

Write the outcome before the system: contain the routine calls end to end in natural spoken conversation, correct and grounded, fast enough to feel human, so people who need a person get one sooner. Narrow the intervention until it hurts: answer from the help center, take a few safe actions like an order lookup, confirm anything high-impact out loud, and hand off everything else. A voice agent that tries to run refunds, fraud holds, and account changes in version 1 is the over-scoping trap, and it is worse on the phone because there is no screen to catch a mistake before it lands.

The clarifying questions you need to ask a stakeholder: how long may the line be silent before it reads as broken; what share of calls are truly routine; what is the current containment and CSAT baseline; which actions may an agent take without a human; what is the promise to the caller about reaching a person; and are calls recorded, with what consent rules. Their answers set every later tradeoff.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models, wrapped in streaming speech recognition and synthesis, are reliable enough to answer grounded questions from your help center in natural speech and take a few low-risk actions with a human fallback. They are far less reliable running unbounded autonomy on high-impact spoken actions, which is why the design keeps refunds and cancellations behind a spoken confirmation and a person. Match the ambition to what the model does reliably, gate the consequential actions behind human oversight, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first call type, favor a high-volume routine intent with a low cost of being wrong, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a sharper fork on voice than it is on chat, because a voice platform carries the hard real-time plumbing for you. Off-the-shelf voice agents and platforms (for example Intercom Fin Voice, plus the voice stacks OpenAI and others expose through the [Realtime API](https://developers.openai.com/api/docs/guides/realtime) and [voice-agent tooling](https://developers.openai.com/api/docs/guides/voice-agents)) get you live faster and hand you telephony, streaming speech recognition and synthesis, turn detection, and barge-in, which are the pieces that take months to get right. Building in-house gives you control over behavior, data, latency, and cost, and it compounds when support is core to your product. Decide on a few axes: how differentiated the spoken experience needs to be, how sensitive your data and the actions are, token and per-minute cost against control, time to market against the data moat you would build, vendor lock-in, and whether you have the team to own an evaluation and monitoring loop for the life of the product. A common answer is to buy the real-time platform to learn fast, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, name the number that must not break, and name the anti-metric.

- **Containment rate.** The share of calls resolved without a human. The core business outcome.
- **CSAT on contained calls.** Quality as the caller feels it, so containment is never bought with frustration.
- **Time-to-first-audio.** How long the line is silent after the caller stops talking, the number the caller experiences on every turn. A voice product lives or dies here.
- **Trapped-or-wrong-action rate.** The hard ceiling: the share of calls where the caller gets stuck in a loop, cannot reach a person, or the agent takes a wrong spoken action like confirming a refund it should not. This is the number that must stay near zero, and it can sink the project on its own.
- **Escalation rate.** Too high means the agent is not helping; too low means it is over-containing and hiding failures.

The guardrail that must not break sits underneath all of these: a caller can always reach a person. Design it in on day one so it is a product promise rather than a fallback bolted on later. Separate leading signals (grounded-answer rate, time-to-first-audio, dead-air duration) from lagging outcomes (CSAT, repeat-call rate). Pair the business outcome (containment, cost per contained call) with a quality metric beneath it (CSAT, the trapped-or-wrong-action rate) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to watch is containment bought at the cost of trust: an agent that closes calls by refusing to escalate, or by keeping a frustrated caller circling its own menu, looks great on a dashboard and quietly recreates the IVR you were paid to replace.

### 4 The experience: designing for a model that is sometimes wrong, on a line that is always live

The product is the experience you build around a probabilistic core, and on voice the experience is half about being right and half about the rhythm of the conversation. Design it so a wrong answer is rare, visible, and cheap to recover from, and so the call feels like talking to a person rather than fighting a phone tree. This has an established design playbook: help the caller form an accurate picture of what the assistant can do, make its uncertainty legible, and give a fast path to correct it or reach a person ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Latency and turn-taking are the experience.** A natural human turn gap is a few hundred milliseconds, and silence past roughly a second reads as a dropped call, so the caller should hear the reply begin within a fraction of a second of finishing. When a lookup will take longer, the agent speaks a short filler ("let me pull that up") and keeps the line alive while the work runs, which is a real turn-taking tool rather than a nicety.
- **Barge-in.** Let the caller interrupt and have the agent stop promptly and listen, the way a person would, so someone can cut a long answer short or correct a wrong turn without waiting for the agent to finish.
- **Ground and confirm out loud.** Answers quote real policy, and because the caller cannot see a citation, the agent reads high-stakes details back ("refunding 40 dollars to the card ending 12, is that right") before it acts, so a misheard order number or amount surfaces before it becomes a wrong action ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)).
- **Escalate gracefully, and make the handoff feel like help.** When confidence is low, a request is high-impact, or the caller asks for a person, the agent hands off with the full call context and no dead end, so the caller never repeats themselves to the human. Set expectations at the top of the call so the caller's mental model matches what the assistant can actually do ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)).
- **Make correction cheap.** Let the caller rephrase, correct a wrong turn, or reach a person in one step, and route that signal back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of real calls that spans routine, ambiguous, action, interruption, and adversarial cases, ideally with real audio so speech recognition and synthesis are exercised rather than just text, and treat it as the release gate: a change ships only if grounded-answer rate, time-to-first-audio, and the trapped-or-wrong-action ceiling hold together. On voice you gate on three axes at once, since a fast wrong answer is still wrong and a correct answer that arrives after the caller hangs up resolved nothing. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a caller trapped by the agent: stuck in a loop, unable to reach a person, sitting through dead air while the agent thinks, or handed a confident wrong spoken action like a promised refund that will not come. Any one of those is the worse-IVR outcome the question warns about, and one viral clip of it can end the program. The guardrails (grounding, a relevance floor that hands off when the agent is unsure, a latency guardrail that escalates when the reply stalls, spoken confirmation and human approval on high-impact actions, and the always-available path to a person) are product-safety features rather than engineering details, because they cap the worst case at a quick handoff rather than a trapped caller or a wrongful action.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates, and sequence it by call type with the easiest intents first.

- **Shadow first.** Run the agent alongside humans on real calls without letting it speak to callers, and compare its would-be answers and its latency against the eval bar.
- **Pilot one intent.** Turn it on for a single routine call type (order status, say), read or listen to every transcript, and watch the metrics from section 3.
- **Stage by call type** behind go and no-go gates on containment, CSAT, time-to-first-audio, and the trapped-or-wrong-action ceiling, adding intents outward only as each earns it, with a rollback you can trigger in minutes.
- **Monitor and keep the human path wired in.** Sample live calls, watch for drift and for the voice-specific tells (talk-over, "hello, are you there", long silences, hangups mid-turn), and guarantee a caller can always reach a person.
- **Know when to pull back.** If the trapped-or-wrong-action rate crosses its ceiling, if CSAT on contained calls drops below the human baseline, or if latency balloons at peak, narrow the intents or roll back rather than pressing on.

> **Real arc: Klarna, 2024 to 2025.** Klarna's assistant did the work of about 700 agents, handled 2.3 million conversations in its first month, and cut resolution time from 11 minutes to under 2. Then in 2025 the company walked it back and rehired humans so a customer could always reach a person. The PM lesson for voice is to set the human-path guarantee and the CSAT guardrail on day one, so aggressive containment never quietly costs you trust. [[press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[the walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]

> **Real build: Intercom Fin Voice.** Building a production phone agent, Intercom chose the speech-to-text, language-model, text-to-speech pipeline over a single voice-to-voice model to keep control and observability. Simple queries returned in about 1 second, and queries that needed 3 to 4 seconds got a spoken filler so the line never went silent. Their text predecessor resolved about 56% of conversations on average, with some customers reaching 70 to 80%. The PM read: containment is real, a filler phrase is a genuine turn-taking product decision, and the seams you can inspect can matter more than a theoretical latency win. [[Intercom case study](https://www.zenml.io/llmops-database/building-a-production-voice-ai-agent-for-customer-support-in-100-days)]

## Follow-ups an interviewer asks

**"How do you measure success, and how do you keep it from being a worse IVR?"** Lead with containment paired with CSAT on contained calls, gate on time-to-first-audio and the trapped-or-wrong-action ceiling, and refuse to celebrate containment that comes from the agent trapping callers or declining to escalate. Report reliability, since an agent that is right most of the time can still strand enough callers to erode trust.

**"The agent gives a caller a wrong answer that costs money. Walk me through your response."** Contain the caller impact first, trace the failure to the step that broke (speech recognition mishearing a number, retrieval, grounding, or the action), add the case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern that should pause the rollout. The recorded call, the audit trail, and the human path are what make this recoverable.

**"Engineering says 6 months. Scope an MVP."** Cut to one routine intent, read-only answers with a clean handoff, no autonomous actions, on one call queue. Prove containment, CSAT, and time-to-first-audio there before adding tools and higher-risk actions. The follow-ups add capability the way versions would, without betting the launch on the hardest call type.

**"How do you choose the autonomy level for a spoken action?"** Match autonomy to the cost of being wrong and the fact that there is no screen to catch it. Auto-resolve grounded answers, confirm medium-risk actions out loud, and require spoken confirmation plus a human for high-impact actions like refunds. Move the line outward only when evaluation on live calls earns it.

**"Callers keep getting stuck. How do you find and fix it?"** Sample live calls on the voice-specific signals (talk-over, "are you there", long silences, hangups), listen to the ones your metrics missed, and let a human name the dimension you were not measuring (maybe the agent pauses a beat too long before a lookup, or an endpoint gap cuts callers off). That becomes a new metric back in the eval set, and the fix ships behind the gate.

**"Build or buy?"** On voice the plumbing tilts you toward buying a real-time platform to start, since telephony, streaming speech, turn detection, and barge-in are hard to build well. Decide on differentiation, data and action sensitivity, latency control, and whether you can own an evaluation loop for the product's life. Buy the platform to learn fast, then build the parts that become your advantage.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Voice platform and build-vs-buy:** OpenAI's [Realtime API guide](https://developers.openai.com/api/docs/guides/realtime) and [Voice agents guide](https://developers.openai.com/api/docs/guides/voice-agents) for the platform you would buy, and [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf) for how enterprises sequence and buy.
- **Metrics, turn-taking, and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), LiveKit on [turn detection, VAD, and endpointing](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection) for the latency feel a PM should understand, and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **The real arcs:** the [Intercom Fin Voice case study](https://www.zenml.io/llmops-database/building-a-production-voice-ai-agent-for-customer-support-in-100-days), the [Klarna press release](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/), and the [2025 walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (the real-time pipeline, streaming, endpointing, barge-in, evals, the harness).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
- Other case: [Customer Support Agent, the PM interview](../customer-support-agent/pm.md).
