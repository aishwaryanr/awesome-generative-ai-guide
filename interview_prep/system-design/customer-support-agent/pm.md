# Customer Support Agent: the PM interview

## The interview question

> "Our support team is underwater and leadership wants an AI agent to help. You are the PM. Should we build it, what does success look like, and how would you take it to production without torching customer trust?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk it, and how you make the calls when the model is probabilistic and sometimes wrong. The technical design (retrieval, tools, evals, the harness) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/customer-support-agent` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, and the number that must not break.
- **4 Experience.** The probabilistic UX: trust, transparency, tone, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Pilot, staged release, go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a customer, an engineer, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. Tier-1 agents handle roughly 60 tickets a shift, about 70% of them repetitive (order status, return policy, password reset), 4 to 8 minutes each, with a queue at peak. A human-handled ticket runs on the order of 5 to 8 dollars fully loaded, and slow first responses cost satisfaction and repeat contacts. Two users have pain: the customer who waits, and the agent buried in repetitive work.

Write the outcome before the system: deflect the repetitive 70% to an agent that resolves them end to end, correct and grounded, so humans handle the hard and emotional cases faster. Narrow the intervention until it hurts: answer from the help center, take a few safe actions, escalate everything else. A support agent that tries to do refunds, fraud, and onboarding in version 1 is the over-scoping trap.

The clarifying questions you need to ask a stakeholder: what is the cost of a wrong answer or a wrong action; what share of tickets are truly repetitive; what is the current containment and satisfaction baseline; what actions may an agent take without a human; and what is the promise to the customer about reaching a person.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models are reliable enough to answer grounded questions from your help center and take a few low-risk actions with a human fallback. They are far less reliable running unbounded autonomy on high-impact actions, which is why the design keeps refunds and cancellations behind a human. Match the ambition to what the model does reliably, gate the consequential actions behind human oversight, and let evaluation tell you where that line sits for your data rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume repetitive workflow with a low cost of being wrong, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

Build or buy is a real fork. Off-the-shelf support agents (for example Intercom Fin, Decagon, Sierra) get you live faster and carry their own evals and guardrails. Building in-house gives you control over behavior, data, and cost, and it compounds when support is core to your product. Decide on a few axes: how differentiated the experience needs to be, how sensitive your data and the actions are, token cost against control, time to market against the data moat you would build, vendor lock-in, and whether you have the team to own an evaluation and monitoring loop for the life of the product. A common answer is to buy to learn fast, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

**What is shifting under you.** Two retrieval changes are worth tracking, and both complement the grounded pipeline rather than replace it. Agentic retrieval lets the agent run its own follow-up searches on the harder ticket that needs more than one hop ([agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)). Long-context models let you fit more of the help center into the prompt, which simplifies the build for a small corpus ([long context](https://ai.google.dev/gemini-api/docs/long-context)). For the repetitive majority that a single article resolves, grounded single-shot retrieval stays the primary path, because it is cheaper and faster per request and keeps the answer precise. Treat both as levers you reach for where a question earns them, and keep the pipeline as the default.

### 3 Success metrics

Pick the minimum set that tells you the truth, and name the number that must not break.

- **Containment or deflection rate.** The share of tickets resolved without a human. The core business outcome.
- **Customer satisfaction on contained tickets.** Quality as the customer feels it, so deflection is never bought with frustration.
- **Incorrect-action rate.** The hard ceiling. A wrong refund or a hallucinated policy is the number that must stay near zero, and it can sink the project on its own.
- **Cost per resolved ticket.** The unit economics finance will ask about.
- **Escalation rate.** Too high means the agent is not helping; too low means it is over-deflecting and hiding failures.

Separate leading signals (grounded-answer rate, escalation precision) from lagging outcomes (satisfaction, retention). Pair the business outcome (containment, cost per resolved ticket) with a quality metric beneath it (satisfaction, incorrect-action rate) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). The anti-metric to watch is deflection bought at the cost of trust: an agent that closes tickets by refusing to escalate looks great on a dashboard and quietly burns customers.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core. Design it so a wrong answer is rare, visible, and cheap to recover from. This has an established design playbook: help the customer form an accurate picture of what the assistant can do, make its uncertainty legible, and give a fast path to correct it or reach a person ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Ground and cite.** Answers quote real policy and link the source, so a customer and an auditor can both trust them.
- **Gate on confidence and escalate gracefully.** When the agent's confidence is low or the request is high-impact, it abstains and hands to a human with full context and no dead end ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). The promise that a customer can always reach a person is a product decision made up front rather than a fallback bolted on later.
- **Set expectations.** Make it clear the customer is talking to an assistant and what it can help with, so the customer's mental model matches what the system actually does ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)), and make the handoff feel like help rather than a wall.
- **Make correction cheap.** Let the customer rephrase, correct a wrong turn, or ask for a person in one step, and route that feedback back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and tone.** A few seconds is fine for chat, and the voice should match your brand and de-escalate rather than over-apologize.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of real questions and action scenarios, and treat it as the release gate: a change ships only if grounded-answer rate, escalation precision, and the incorrect-action ceiling hold. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a confident wrong answer on a high-impact action, for example telling a customer they will be refunded when they will not, or issuing a refund a jailbreak talked the agent into. The guardrails (grounding, a relevance floor that escalates when unsure, human approval on refunds) are product-safety features rather than engineering details, because they cap the worst case at a needless escalation rather than a wrongful action.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates.

- **Shadow first.** Run the agent alongside humans on real tickets without sending its answers, and compare against the eval bar.
- **Pilot a slice.** Turn it on for one queue or one topic, read every transcript, and watch the metrics from section 3.
- **Stage the rollout** behind go and no-go gates on containment, satisfaction, and the incorrect-action ceiling, with a rollback plan you can trigger in minutes.
- **Monitor and keep a human path.** Sample live traffic, watch for drift, and guarantee a customer can always reach a person.

> **Real arc: Klarna, 2024 to 2025.** Klarna's assistant did the work of about 700 agents, handled 2.3 million conversations in its first month, and cut resolution time from 11 minutes to under 2. Then in 2025 the company walked it back and rehired humans so a customer could always reach a person. The PM lesson is to set the human-path guarantee and the satisfaction guardrail on day one, so aggressive deflection never quietly costs you trust. [[press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[the walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with containment paired with satisfaction on contained tickets, gate on the incorrect-action ceiling, and refuse to celebrate deflection that comes from the agent refusing to escalate. Report reliability, since an agent that is right most of the time can still fail often enough to erode trust.

**"The agent gives a customer a wrong answer that costs money. Walk me through your response."** Contain the customer impact first, trace the failure to the step that broke (retrieval, grounding, or an action), add the case to the eval set so it cannot regress, and decide whether it was a one-off or a pattern that should pause the rollout. The audit trail and the human path are what make this recoverable.

**"Engineering says 6 months. Scope an MVP."** Cut to the repetitive 70% on read-only answers with escalation, no autonomous actions, on one queue. Prove containment and satisfaction there before adding tools and higher-risk actions. The follow-ups add capability the way versions would, without betting the launch on the hardest slice.

**"How do you choose the autonomy level?"** Match autonomy to the cost of being wrong. Auto-resolve grounded answers, suggest-to-agent for medium-risk actions, and require a human for high-impact actions like refunds. Move the line outward only when evaluation on live traffic earns it.

**"Leadership frames this as headcount reduction. How do you set the goal?"** Frame it as capacity and speed: deflect repetitive work so humans handle the hard and emotional cases faster, with satisfaction and a guaranteed human path as guardrails. The Klarna arc is the cautionary tale for a pure headcount framing.

**"Build or buy?"** Decide on differentiation, data and action sensitivity, and whether you can own an evaluation loop for the product's life. Buy to learn fast, then build the parts that become your advantage.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf).
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).
- **The Klarna arc:** the [Klarna press release](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/) and the [2025 walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (retrieval, tools, evals, the harness).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
