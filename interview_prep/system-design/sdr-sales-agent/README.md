# Designing an SDR (Sales-Development) Agent

## The interview question

> "Design an SDR (sales development) agent that qualifies inbound leads and drafts personalized outreach. It reads a lead from the CRM, enriches it with external data through tools, scores it against your ideal customer, and writes a personalized first-touch message. Walk me through it."

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real-world benchmarks and deployment data, and it ships with [runnable, provider-agnostic LangGraph code](code/) you can execute.

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that the core component is non-deterministic, so evaluation moves from a final gate to the center of the design.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/sdr-sales-agent (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs, especially the enrichment tools and the brand-and-compliance guardrail. When we get to the code, run `code/run.py` and explain what each node in the graph does."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine

This case is built on the same 5-layer spine as the rest of the collection: **1 the model**, **2 the wrapping layer** (the architecture: knowledge, tools, and memory), **3 evals and guardrails**, **4 production and ops**, and **5 optimization** (where multi-agent lives). System design is the work of composing them into one coherent system. For how we use this spine and why it is a starting point rather than a rulebook, see [the spine](../README.md#the-spine-how-we-think-about-ai-system-design).

The rest of this case study walks these layers for this problem, going deepest where it is hardest.

---

## The answer

### Step 1: scope before you architect

Before reaching for an enrichment API, clarify the problem, because in AI the largest failures are designed in before a single prompt is written. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** A sales-development rep (SDR) is the person who works inbound leads (a form fill, a demo request, a content download) into qualified sales conversations. The repetitive core of the job is researching each lead across the CRM and the web, deciding whether it fits the ideal customer, and writing a first-touch email. A rep handles a few dozen to a few hundred leads a week, spends several minutes of research per lead, and the good leads go cold while they work through the backlog. Speed to the first touch is a large share of whether a lead ever converts.
- **Outcome, written before the system.** For every inbound lead, produce a qualification decision the rep trusts and, for the leads worth pursuing, a personalized draft ready for a human to approve and send. Measured by qualification accuracy against what the reps and the pipeline actually confirm, by the reply and meeting rates on sent messages, and by a hard ceiling on compliance violations and fabricated claims about a prospect.
- **The AI intervention, narrowed until it hurts.** Enrich and score the inbound lead, then draft one grounded first-touch message, staying well short of an autonomous outbound machine that sends on its own. The agent proposes; a human disposes.
- **System and safety.** An evaluation set that gates every release, a brand-and-compliance guardrail on every draft, [prompt-injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) defense on untrusted enrichment content, suppression and opt-out enforcement, human approval as a hard requirement before any send, full tracing, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): what is the ideal customer profile and how is a lead scored today; what counts as a qualified lead; which regulations apply to your outreach and in which countries; may the agent ever send autonomously or is human approval required; what channels (email, LinkedIn, both); how fresh is the CRM and enrichment data; what is the worst failure the business will not tolerate. This also avoids the traps that sink these projects: leading with "build an outbound agent that emails everyone" (solutioning in the problem statement), packing prospecting and sending and objection-handling and deal management into one system (over-scoping), and designing without a measurable owner for qualification quality.

> **Real outlier: the payback is fast, and that is exactly why guardrails matter.** Vendor and market benchmarks put the payback period on an AI-SDR deployment at around 3 months (one widely cited figure is 3.2 months) against roughly 8.7 months for a ramped human SDR, because the agent starts generating pipeline in days rather than after a hiring-and-training cycle. Read these as directional vendor and market numbers rather than audited results. The lesson for the design is that the speed comes from volume, and volume is precisely what turns one bad behavior, a fabricated claim or an email to someone who opted out, into a repeated, brand-wide incident. [[UserGems](https://www.usergems.com/blog/are-ai-sdrs-worth-it)]

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data, because every method in the wrapping layer below is chosen for a specific shape of data. Change an assumption and the method changes with it. For this case study, assume:

- **The lead arrives with intent, and the data lives in tools.** These are inbound leads, so there is already a signal (they requested a demo, downloaded a guide, filled a pricing form). The record lives in a CRM (Salesforce, HubSpot), and richer attributes come from enrichment providers (Apollo, Clearbit, ZoomInfo style) reached through tools. *This is why* Layer 2 is dominated by tools and integrations rather than by a document retrieval index, and why the tool contracts are the product.
- **Signals are structured, sourced, and perishable.** Company size, industry, title, tech stack, recent funding, hiring. Each has a source and a date, and some of it goes stale fast (a title changes when someone moves jobs). *This is why* every signal is stored with its provenance and a verified flag, and why re-enrichment on engagement matters more than a one-time pull.
- **Outreach is regulated.** In the United States, commercial email falls under [CAN-SPAM](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business). In the EU and UK, contacting a person is processing their personal data under the [GDPR](https://gdpr-info.eu/art-6-gdpr/). *This is why* a brand-and-compliance guardrail is mandatory rather than optional, and why suppression and opt-out state is authoritative.
- **A human approves before any send.** The agent drafts and queues; a person reviews and sends. *This is why* the approval queue is a first-class node in the graph, and autonomy stops at the draft.
- **Fabricating a claim about the prospect is the worst failure.** An email that congratulates someone on funding they never raised, or a role they do not hold, destroys trust and creates brand and legal risk. *This is why* generation is grounded strictly in verified signals, faithfulness is the highest-signal metric, and a claim-check runs before a human ever sees the draft.
- **Volume is batch-like, latency is forgiving.** Leads arrive in a steady stream and a draft that is ready within minutes is fine. *This is why* throughput, deliverability, and cost per lead matter more than sub-second latency, and why enrichment results are worth caching.
- **One language and one region to start, more later.** English and one regulatory regime at launch. *This is why* a single compliance profile is enough now, and multi-region and multilingual are later extensions (Follow-up 6).

Keep these in view as you read the wrapping layer: each choice below points back to one of them. If your leads are cold rather than inbound, your outreach is unregulated, or a human is not in the loop, revisit the assumption and pick a different method.

### Step 2: walk the layers for this system

**Layer 1, the model.** The answer here is a routing strategy paired with structured output. Enrichment parsing and qualification scoring are cheap, high-volume steps a small fast model handles well; drafting a message that has to sound human and stay grounded is where a stronger model earns its cost. The model is non-deterministic, so the same lead can produce different phrasings of a draft, which is fine, and different factual claims, which is not. You constrain that with structured output (the drafter returns each personalized claim paired with the id of the signal that backs it), with grounding, and with the guardrail in Layer 3 that verifies those claims before any human sees them. A useful primer on this shape of system is OpenAI's [practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) and Anthropic's [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents).

**Layer 2, the wrapping layer (the architecture).** This is where most of the design lives, so go deep here. It is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. For an SDR agent, the knowledge is the prospect's signals rather than a help-center corpus, the tools reach into the CRM and enrichment providers, and the whole layer decides whether the agent qualifies well and writes something true.

#### Knowledge: the lead-signal pipeline

The agent must ground every qualification decision and every sentence of the draft in real, sourced signals about this specific lead, or it invents a prospect that does not exist. That is a retrieval pipeline of a different shape from document RAG: the corpus is one lead assembled from many tools, and every stage is a decision whose right answer depends on your data. Each choice below points back to the [assumptions](#the-assumptions-we-make-about-the-data-and-the-use-case) above.

```
┌──────────────────────────────────────────────────────────┐
│ Lead-signal pipeline: one lead assembled from many tools │
└──────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────┐
  │                  Inbound lead                  │
  │  form fill · demo request · content download   │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │             1  Identity resolution             │
  │         match email + domain to a CRM          │
  │              account and contact               │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │               2  Enrich (tools)                │
  │        firmographic + contact + intent,        │
  │            each with source + as_of            │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │            3  Store with provenance            │
  │  every signal tagged: source, date, verified?  │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │            4  Retrieve for the task            │
  │ pull the signals that matter for this decision │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │         Qualify (scored, explainable)          │
  └────────────────────────────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────┐
  │                     Draft                      │
  │       grounded in verified signals only        │
  └────────────────────────────────────────────────┘
```

**1. Identity resolution.** Before you can enrich a lead you have to know who it is: match the inbound email and its domain to an existing CRM account and contact, or create one, so you do not enrich a duplicate or attach signals to the wrong record. A personal-mailbox domain (gmail, outlook) resolves to no company, which is itself a signal that the lead may not fit a business ideal customer. *Given our assumption* that leads are inbound with a real email, domain matching covers most of it. *What data decides it:* how much of your inbound uses corporate versus personal email, and how clean your CRM's dedup already is.

**2. Enrich through tools.** Enrichment is the step that turns a bare email into a picture of the company and the person, by calling data providers. It comes in three kinds, and each answers a different question:
- **Firmographic:** the company. Size, industry, revenue, location, tech stack. This tells you whether the account is the kind you sell to.
- **Contact:** the person. Title, seniority, department, tenure. This tells you whether the person influences the buying decision.
- **Intent:** timing. A recent demo request, a funding round, a hiring spike, a page they kept visiting. This tells you whether they are in-market right now.

Enrichment providers disagree and have gaps, especially outside their home region, so treat coverage and accuracy as measurable rather than assumed. *This is why* our assumption tagged every signal with a source. *What data decides it:* run two or three providers against a sample of leads you already know the truth about and measure field-level coverage (how often a field is filled) and accuracy (how often it is right), per region.

**3. Store with provenance.** Every signal is stored with where it came from, when, and whether it is verified. That provenance is load-bearing for this system in a way it is not for most: the drafter is only allowed to state a signal that is verified, and the guardrail uses the same flag to catch any claim that rests on a rumor or a stale field. A single-blog rumor of a funding round is a real signal for scoring intent softly, and an unsafe thing to state as fact in an email. *This is why* the assumption separated verified from unverified. *What data decides it:* how fast your fields go stale (titles change often, company size slowly), which sets how aggressively you re-enrich.

**4. Retrieve for the task.** Different steps need different signals: qualification wants firmographic and intent, drafting wants the one or two specific, verified hooks that make a message feel written for this person. Pulling only the relevant signals keeps the model's context tight and keeps a weak signal from dragging a decision. This is the same discipline as ranking and capping retrieved chunks in a document system, applied to a lead. If you also want the draft's claims about **your own product** to be grounded, retrieve those from a small product-and-proof knowledge base and hold the drafter to it the same way, which is ordinary [retrieval-augmented generation](https://arxiv.org/abs/2005.11401).

#### Qualification: a scored, explainable decision

Qualification is the decision of whether a lead is worth a rep's time, and the design principle is that it is a transparent score, so a rep trusts it, an auditor can trace it, and you can evaluate it. Two ideas set it up:

- **The [ideal customer profile (ICP)](https://en.wikipedia.org/wiki/Lead_scoring)** is the description of the accounts you win: the size, industries, and roles that convert. It is the yardstick the score measures against.
- **[Lead scoring](https://en.wikipedia.org/wiki/Lead_scoring)** turns that yardstick into points. A common shape splits the score into fit (does the account match the ICP) and intent (is there evidence they are in-market), which maps onto the old sales heuristic [BANT](https://blog.hubspot.com/sales/bant): budget, authority, need, and timing.

The move that matters is making the score explainable. Every point carries a reason code that names the signal it came from, so the output is a decision plus its evidence rather than a bare number.

```
  fit    size >= ICP floor         +25   (from signal s1: 1200 employees)
         industry in the ICP       +20   (from signal s2: b2b-saas)
         seniority / buying power  +20   (from signal s3: VP Engineering)
  intent recent inbound action     +25   (from signal s4: requested a demo)
         verified funding / hiring +10   (from signal s5: closed a Series C)
  -----------------------------------------------------------------------
  score  -> >= 60 qualified | 35-59 nurture | < 35 disqualify
  every point cites the signal that earned it; a human can read WHY
```

The thresholds and weights above are illustrative, so treat them as a starting scorecard and tune them on your own data. **What data decides it:** derive the weights from your closed-won history rather than hand-setting them. Ask which attributes actually predicted a lead becoming a customer, weight those, and check the score against outcomes. That turns qualification into something you can [calibrate](#trust-the-judge-then-close-the-discovery-loop) and measure (Layer 3) instead of a set of numbers someone guessed. A lead that scores in the middle is a nurture case or a human review rather than a forced yes or no.

#### Generation: personalized, and grounded so it never fabricates

The draft is a short first-touch message that reads as written for this person. The entire risk of this system concentrates here, because the failure that erodes trust fastest is a confident, specific, wrong claim about the prospect. You defend against it by construction rather than by hoping the model behaves:

- **Ground in verified signals only.** The drafter receives the verified signal set and is instructed to reference only those. A rumored or stale signal is available for soft scoring and is off-limits as a stated fact.
- **Make claims auditable.** The drafter returns each personalized sentence paired with the id of the signal it came from, so the check downstream is mechanical rather than a matter of opinion. This is structured output doing safety work.
- **Grade before send.** A claim-check verifies that every personalized claim maps to a verified signal, and strips or blocks any that does not. This is the same idea as grade-before-answer retrieval systems like [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884), applied to grade-before-send. The measure of it is [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/): does every statement trace to the evidence.

The result is that a fabricated claim is caught by the guardrail rather than delivered to a prospect, which is exactly what the runnable code demonstrates.

#### Tools (actions)

Tools are the agent's hands, and for an SDR agent they are the bulk of the design. Each is a typed, allowlisted contract rather than open-ended code execution.

```
  crm_lookup(email) -> {account, contact, history}        READ    low blast radius
  enrich_company(domain) -> {size, industry, tech}        READ    external provider, cache it
  enrich_contact(email) -> {title, seniority, tenure}     READ    external provider, cache it
  get_engagement_history(contact) -> [touches]            READ    prior emails, do not repeat
  check_suppression(contact) -> {opted_out, consent}      READ    compliance-critical, authoritative
  upsert_lead(account, contact, score) -> id              WRITE   idempotent, logged
  queue_for_approval(draft) -> approval_id                WRITE   a human reviews and sends
                                                          (there is no autonomous send tool)
```

The calls that matter: the **tool description is a prompt** (the model picks a tool from its name and doc, so vague descriptions cause wrong calls); **least privilege** (read tools are cheap to trust, the one write that reaches a person is gated behind human approval); **[idempotency](https://en.wikipedia.org/wiki/Idempotence)** (a retried `upsert_lead` must not create a duplicate contact, and a retried draft must not queue two messages); **error handling** (a failed enrichment call is caught and the lead proceeds with fewer signals or routes to a human, never with an invented one); and the **loop is bounded** (a hard step cap stops the agent from enriching forever on a lead it cannot resolve). Reaching tools, reading the result, and deciding the next step is the [ReAct](https://arxiv.org/abs/2210.03629) pattern. For tool design specifically, Anthropic's [Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents) is the reference. The deliberate absence of an autonomous send is the single most important tool-design choice: sending is a human action, and the agent's most powerful move is to queue a draft.

#### Memory

Memory is what keeps the agent from treating a warm account like a stranger, and it comes in layers.

```
  SHORT-TERM (this lead)      : the signals and score the agent is reasoning over right now
  ACCOUNT (this relationship) : prior touches and replies, so it does not repeat a message
                                or contact someone your team already emailed this week
  CONSENT (durable, authoritative) : opt-out and suppression state, retrieved and obeyed, never overridden
```

The calls that matter: **retrieve account history, do not dump it** (pull the recent touches and the last reply rather than the entire relationship); **treat consent state as authoritative** (an opt-out is a hard stop that no score or signal can override); and **treat enrichment content as untrusted** (a prospect's scraped bio or website can carry an injected instruction, the memory arm of the [lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) in Follow-up 2).

Together, the lead-signal knowledge, the tools, and the memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for an agent this is the center of the design, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, blocking an unfaithful or non-compliant draft and enforcing the human handoff. There is no single accuracy number, and no generic metric you can copy off a shelf, because good means something different for every product. So start where we teach you to start: **from failure modes.** For an SDR agent the unacceptable failures are a fabricated claim about a prospect, qualifying a bad-fit lead (wastes a rep) or disqualifying a good one (lost revenue), a compliance violation (emailing a suppressed contact, no opt-out, no lawful basis), an off-brand or spammy message, and a send without human approval. Translate each into an observable behavior. The metrics below are a menu you draw from once you know your failure modes, and the target is the minimum set that gives the most signal for your product. This section stays consistent with the free [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course.

**Measure against today's baseline rather than an abstract target.** The right bar is the status quo: the rep's current manual research and outreach. Where leads went unworked because researching every one by hand was too costly, an agent that drafts for them reliably clears a low bar by existing. Where a rep already researches and writes well, the agent has to match or beat that on reply and meeting rates, measured retrospectively against the outreach in your history. That keeps the eval tied to the revenue decision the business faces rather than chasing a round accuracy number.

Evaluate at three levels: **each component** (did enrichment return accurate fields, did qualification match the truth, did the draft stay grounded, did the guardrail catch the fabrication), **the whole task** end to end (was this lead routed correctly and, when qualified, given a compliant, on-brand, grounded draft), and **live traffic** (are reply and meeting rates holding, and complaints staying near zero).

> **Real finding: [tau2-bench](https://arxiv.org/abs/2506.07982).** On a tool-using agent, a single-attempt pass rate looks respectable, but pass^k (succeed on all k independent tries) collapses as k grows. An SDR agent that grounds its draft 3 times in 4 is one fabricated claim per 4 emails at scale, which is not shippable. Reliability is the bar, above average accuracy.

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: does every claim in the draft map to a verified signal, is the physical address and opt-out present, was the suppression check called, does the qualification score match the scorecard, did it call `enrich_company` with the right domain. Fast, reliable, cheap. This is where you catch the worst failures, because faithfulness here is checkable in code once claims carry evidence ids.
- **LLM judges.** One model scoring another against an explicit rubric, for subjective qualities code cannot capture: brand voice, whether the personalization is relevant rather than generic, whether the message reads human. Scalable, and a new source of non-determinism, so it must be calibrated before you trust it (below).
- **Human evaluation.** The gold standard you calibrate the other two against. Reps label qualification decisions against what actually converted, and sample drafts for quality. Too slow to run on everything, so you sample: calibration, edge cases, anything a prospect complained about.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of 30 to 50 labeled leads across strong-fit, borderline, poor-fit, opted-out, and adversarial (injected enrichment) cases. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Enrichment / signals** | missing fields, wrong data, stale | field coverage, field accuracy, staleness, source agreement | code-based against leads with known truth |
| **Qualification** | qualifies a bad fit, disqualifies a good one | precision, recall, calibration, agreement with rep judgment, borderline confusion | code-based / human against labeled dispositions |
| **Message generation** | fabricates a claim, generic, off-brand, unreadable | faithfulness / groundedness, personalization relevance, brand-voice adherence, readability | code-based (claims to signals) + LLM judge |
| **Compliance guardrail** | emails a suppressed contact, missing opt-out or address, no lawful basis | suppression-leak rate (must be zero), opt-out-element rate, address-present rate, lawful-basis coverage | code-based + adversarial suite |
| **Human approval** | drafts a human would never send | approval / acceptance rate, edit distance, time-to-approve | human on the queue |
| **End to end** | wrong route, or a bad draft reaches a person | routing accuracy, pass@1, **pass^k**, reply rate, meeting rate | scenario suite + judge + live |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: these are the metrics teams typically reach for, and your job is to choose wisely and instrument only the few that earn their place. A dashboard with 30 metrics and no owner tells you nothing. Pick the two or three per component that map to your real failure modes and drop the rest. For this product, **faithfulness is the highest-signal single metric**, because a fabricated claim is the failure you least want to ship, and suppression-leak rate is the compliance metric you hold at zero.

Report **pass^k** alongside the average, because an agent that grounds its draft on average but fabricates 1 time in 4 is one incident per 4 emails at volume (the tau2-bench finding above).

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem, or is it merely interesting?
- **Reliability:** human evaluation and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check is cheap, a fast judge call is medium, detailed human review is expensive.

High impact and low cost are the must-haves (the faithfulness claim-check, the compliance-element checks, the suppression check). High impact and high cost are strategic investments you run on a sample (a calibrated brand-voice judge, rep labeling of qualification). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the business, run inline so the system acts the moment they trip: a claim not backed by a verified signal, a contact who opted out, a missing compliance element, a draft trying to reach a person without approval. The action is immediate: block the draft, suppress the send, route to a human. Guardrails must be fast and reliable before sophisticated, so they are deterministic code here.
- **Improvement flywheel (offline, batch).** Everything else: reply-rate and meeting-rate trends, qualification precision over time, brand-voice drift, deliverability. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live traffic:

| Metric | Job | Why it matters |
|---|---|---|
| fabricated-claim rate | guardrail | the worst failure; the claim-check holds it at zero |
| suppression-leak rate | guardrail | an email to an opted-out contact is a legal and trust breach; must be zero |
| compliance-element coverage | guardrail | opt-out and address present on every draft, enforced inline |
| unapproved-send attempts | guardrail | confirms the human-in-the-loop gate holds |
| qualification precision on a sample | flywheel | are the leads you pass actually worth a rep's time |
| reply rate / positive-reply rate | flywheel | quality of the outreach as prospects react to it |
| meeting-booked rate | flywheel | the outcome the pipeline is measured on |
| spam-complaint / unsubscribe rate | flywheel | rising numbers mean the outreach is too aggressive or off-target |
| deliverability / bounce rate | flywheel | list and domain health, the plumbing reply rate depends on |
| draft acceptance / edit rate | flywheel | how often a human sends the draft as written, a proxy for quality |
| cost per qualified lead, tokens per lead | flywheel | unit economics, the number finance asks about |

Frame these as measures to track over time rather than fixed thresholds, because the right reply rate for your market and the right qualification cutoff for your funnel are things you learn from your own data.

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label a few hundred drafts, run the judge on the same set, measure agreement (percent agreement or Cohen's kappa), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the underlying model, and run the judge on a sampled percentage of live traffic to control cost.

Then run the discovery loop, because prospects and reps will surface failures your metrics were never built for. Sample live traffic on **signals** (a heavy rep edit before sending, a prospect replying "please remove me," a spam complaint, a negative reply, a bounce). When a signal keeps firing but your metrics look clean, that gap is the tell: a human reads those traces, names the quality dimension you were not measuring (a tone that reads as pushy, a personalization hook that lands as creepy), and it becomes a new metric added back into the reference dataset. Evaluation is never finished.

![Eval and discovery loop: pre-deployment validation, then production monitoring, with the reference set growing each cycle](../assets/eval_discovery_loop.png)

Instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. Reading traces is how you find the exact step where a qualification or a claim went wrong. Tooling worth knowing: Ragas and DeepEval for component metrics, promptfoo for CI gates, Arize Phoenix for tracing and online evals. *Deeper:* [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../../topics/evaluation.md).

**Layer 4, production and ops.** The loop that keeps it alive at scale: enrichment calls cached and rate-limited, deliverability managed (domain reputation, warmup, bounce handling), suppression lists enforced globally, retries that stay idempotent, and observability so every step is traceable. Detail is in Follow-ups 1 and 3.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: model routing (a small model for enrichment parsing and scoring, a stronger one for drafting), caching enrichment results and stable prompt prefixes, and multi-agent. For a routine single-lead flow a single agent is enough, and for high-value target accounts the design to reach for is a research and enrichment sub-agent, a drafting sub-agent, and an independent compliance-and-brand reviewer, which Follow-up 5 lays out.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (a fixed enrich-score-draft pipeline that always routes to a human), prove it with evals, and only hand the model more freedom (its own follow-up enrichment searches, a bounded loop, more channels) once the measurement says the constrained version is solid and its limits are what block you. You earn agency through evaluation before you grant it.

Composed, the layers give one architecture:

```
┌───────────────────────────────────────────────────────────────────┐
│ Observability: every node, tool, and model call is a span (Arize) │
└───────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────┐
  │             Inbound lead             │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │         Identity resolution          │
  │     match email + domain to CRM      │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │       Enrich (CRM + providers)       │
  │   firmographic · contact · intent    │
  │ each signal: source, date, verified? │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │           Suppression gate           │
  │   opted out? → STOP (no outreach)    │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │    Qualify (scored, explainable)     │
  │ not qualified → nurture / disqualify │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │                Draft                 │
  │  grounded in verified signals only,  │
  │  each claim carries its evidence id  │
  └──────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────┐
  │     Brand + compliance guardrail     │
  │ suppression recheck · faithfulness · │
  │ CAN-SPAM: opt-out + physical address │
  └───────────────────┬──────────────────┘
                ┌─────┴───────────────────┐
           pass ▼                         ▼ fail
  ┌───────────────────────────┐  ┌────────────────┐
  │   Human approval queue    │  │ Block / revise │
  │ a human reviews and sends │  └────────────────┘
  └───────────────────────────┘
```

Read it as the spine composed: the model (layer 1), wrapped in lead-signal knowledge, enrichment and CRM tools, and account memory (layer 2), gated by evals (layer 3) that run inline as guardrails and offline as a flywheel, run under production observability (layer 4), with optimization (layer 5) held to what this system actually needs, which is model routing and caching rather than a second agent. Composing exactly these pieces into one coherent system is the system design. The suppression gate and the compliance guardrail are what make human approval the safe default: the agent's most powerful action is to queue a grounded, compliant draft for a person, and a fabricated claim or an email to a suppressed contact stays blocked rather than merely unlikely.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this architecture (minus the scale pieces): enrichment through a tool, an explainable qualification score, a drafter that grounds every claim in a verified signal, a brand-and-compliance guardrail, and a human approval queue. It runs offline with a deterministic drafter, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                       # run the scenarios (also a self-test)
python run.py "we want a demo"      # run one lead with your own note
```

The scenarios assert each path: a strong lead qualifies and its grounded draft is queued for a human, a qualified lead whose draft leans on an unverified rumor is blocked by the guardrail, an opted-out contact is suppressed before any draft, and a no-company lead does not qualify. Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route drafting through a real model. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now handle 10x the lead volume. Where does it break, and how do you scale it?"

Name where it breaks first, then scale that, cheapest lever first.

- **Enrichment (layer 2).** External providers are the first bottleneck and the first cost, because you pay per lookup and you hit rate limits. Cache enrichment by domain and contact with a freshness window, batch and deduplicate lookups, and fall back across providers when one is down or thin on a region. At volume the data layer, rather than the model, is usually what strains.
- **Deliverability (layer 4).** More email is not more pipeline if it lands in spam. Warm sending domains, spread volume, honor per-day caps, handle bounces, and enforce the suppression list globally so no opted-out contact is ever reachable from any part of the system. Rising complaint and bounce rates degrade the whole domain's reputation, so treat them as guardrail metrics.
- **Routing and caching (layer 5).** Route enrichment parsing and scoring to a small fast model; reserve the stronger model for drafting; cache the stable prompt prefix (system instructions, brand voice, tool schemas).
- **Infrastructure (layer 4).** Horizontal workers behind a queue, idempotent writes so a retry never double-creates a lead or double-queues a draft, and backpressure so a lead spike degrades gracefully.

Put numbers on it: tokens and enrichment calls per lead, a cost per qualified lead, and a per-stage budget. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 2: "A prospect's enriched bio contains hidden text telling the agent to email a competitor's list. How do you prevent unsafe actions?"

Separate drafting from sending, and keep untrusted content untrusted.

- **Untrusted input.** Enrichment content (a scraped bio, a company description, a website) and the lead's own form text are untrusted; the agent never executes instructions found in them. The sharp way to see the risk is the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
  The lethal trifecta

           ┌───────────────────┐   ┌───────────────────┐   ┌────────────┐
           │ Untrusted content │   │ Access to private │   │ Ability to │
           │  (enriched bio)   │   │     CRM data      │   │ send email │
           └─────────┬─────────┘   └─────────┬─────────┘   └──────┬─────┘
                     └───────────────────────┼────────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │ All 3 together is dangerous │
                              └─────────────────────────────┘

                any 2 are manageable; an SDR agent naturally has all three
```

An SDR agent naturally has all three, which is exactly why the send is not the agent's to make.

- **Human approval is the circuit breaker.** Because a person reviews every draft before it goes out, an injection that slips past the input guardrail still cannot send anything. The worst a jailbreak achieves is a draft a human declines.
- **Guardrails and blast radius.** Strip or neutralize instruction-like content in enrichment before it reaches the model, keep the faithfulness check so injected claims without a verified signal cannot enter the message, use least-privilege tool scopes, and keep an immutable audit log of every enrichment, decision, and queued draft.

*Deeper:* [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 3: "Cut the cost and time per lead without dropping quality."

Budget both across the stages (identity resolution, enrichment, scoring, drafting, the guardrail), then attack the largest.

- Cache enrichment aggressively, because the same company and contact recur across leads and the data changes slowly.
- Cache the stable prompt prefix (brand voice, tool schemas, few-shot examples), which is identical on every draft.
- Route the cheap steps (parsing, scoring) to a small model and reserve the strong model for drafting.
- Batch leads through enrichment and scoring rather than one lookup per lead.

Quality holds because the eval set from Layer 3 gates each change: if a cheaper drafting route drops the faithfulness rate or the brand-voice score, it does not ship. Pair every efficiency change with the metric it could hurt.

*Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 4: "You are now sending into the EU. What changes, and how do you keep an audit trail?"

Two constraints promote normally-optional components into load-bearing walls: a **lawful basis** for processing personal data, and an **audit trail** for every decision and send.

- **Regulatory basis.** United States commercial email must follow [CAN-SPAM](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business): accurate headers, a truthful subject line, identification as an ad, a valid physical postal address, a working opt-out that you honor within 10 business days, and responsibility even for mail a vendor sends on your behalf. Each violating email carries a penalty of up to 53,088 dollars, which is why the compliance elements are code-checked on every draft. In the EU and UK, contacting a person is processing personal data under the [GDPR](https://gdpr-info.eu/art-6-gdpr/), commonly on the [legitimate-interests](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/a-guide-to-lawful-basis/lawful-basis-for-processing/legitimate-interests/) basis for B2B outreach, which the guidance ties to targeted, relevant contact and an easy opt-out rather than untargeted mass mail. The agent's grounded, ICP-scored personalization supports that relevance, and its enrichment provenance supports showing where each prospect's data came from.
- **Audit trail.** Add an immutable, queryable log of every lead: the signals and their sources, the qualification score and its reason codes, the draft and its evidence ids, the guardrail result, and who approved and sent. When a regulator or a prospect asks why they were contacted, the answer is a record rather than a guess.
- **Consent state is authoritative.** Suppression and opt-out live as durable state that no score or signal can override, checked at the suppression gate and again at the guardrail.

The pattern generalizes: a domain constraint is what forces you to engineer a layer you would otherwise accept as a framework default. *Deeper:* [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

For high-value target accounts, the design to recommend splits the work across agents: a research and enrichment sub-agent that gathers and verifies signals, a drafting sub-agent that writes in the brand voice, and an independent compliance-and-brand reviewer that sees only the finished draft. A routine inbound lead stays a single agent with tools, because the work is one coherent thread, and you let the account value show when the fan-out is worth it. Here is when multi-agent (layer 5, optimization) pays off and how the architecture extends.

**When multi-agent earns its place:**
- **The work decomposes into independent sub-tasks.** Deep account research across many sources, running in parallel, is the classic fit, and it is where a research agent pulls its weight.
- **Distinct sub-tasks need distinct context and skills.** Researching an account, scoring it, writing in the brand voice, and reviewing for brand and regulatory risk are genuinely different jobs. A dedicated compliance reviewer with only the policy and the draft in its context judges more sharply than a generalist juggling everything.
- **You want an independent reviewer.** A separate compliance-and-brand agent that never sees the writer's reasoning, only its output, is a stronger check than self-review, which is easier to pass than to earn.
- **The task is valuable enough to pay for it.** Multi-agent trades tokens for capability, which fits high-value target accounts more than high-volume routine leads.

**How you would extend this architecture.** Keep the single-agent design intact and add an orchestrator that runs specialists per lead, with the same suppression gate, guardrail, evals, human approval, and observability wrapping the whole system.

```
┌─────────────────────────────────────────┐
│ Observability: spans across every agent │
└─────────────────────────────────────────┘

                                        ┌──────────────────┐
                                        │       Lead       │
                                        └──────────────────┘
                                                  ▼
                                        ┌──────────────────┐
                                        │ Suppression gate │
                                        └──────────────────┘
                                                  ▼
                           ┌─────────────────────────────────────────────┐
                           │                Orchestrator                 │
                           │ per lead: research · score · write · review │
                           └──────────────────────┬──────────────────────┘
              ┌───────────────────────────────────┴───────────────────────────────────┐
              ▼                       ▼                       ▼                       ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │   Research agent   │  │   Qualify agent    │  │    Writer agent    │  │  Compliance agent  │
   │  enrich + verify   │  │    score vs ICP    │  │    grounded in     │  │  brand + CAN-SPAM  │
   │ signals, own tools │  │    reason codes    │  │  verified signals  │  │    / GDPR, sees    │
   │                    │  │                    │  │                    │  │   only the draft   │
   └──────────┬─────────┘  └──────────┬─────────┘  └──────────┬─────────┘  └──────────┬─────────┘
              └───────────────────────┴───────────┬───────────┴───────────────────────┘
                                                  ▼
                                    ┌───────────────────────────┐
                                    │   Human approval queue    │
                                    │ a human reviews and sends │
                                    └───────────────────────────┘
```

> **Real data: Anthropic's multi-agent research system.** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens, and token usage explained roughly 80% of the performance gap. The takeaway is to spend those tokens where task value and genuine parallelism justify them, so a fan-out research agent fits high-value accounts and a single agent fits routine inbound. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]

The honest rule: start with the single agent, instrument it, and let the traces tell you when account research or an independent compliance review has outgrown one agent. *Deeper:* [Agents](../../../topics/agents.md).

---

## Follow-up 6: "Add LinkedIn as a second channel, and make it work in other languages."

The architecture does not change; the data and the evaluation do. Each channel gets its own draft format and its own compliance profile (LinkedIn's terms and rate limits differ from email's CAN-SPAM rules), and each language gets labeled examples so faithfulness and brand voice are measured per language rather than assumed. Enrichment coverage and the suppression list apply across channels, so an opt-out in one channel suppresses the others. This is the recurring lesson: most "make it do X" follow-ups are answered in the data and eval layers, and the box diagram stays the same. *Deeper:* [Evaluation](../../../topics/evaluation.md).

---

## Follow-up 7: "A better model drops next quarter. How do you avoid a rewrite?"

Build the harness on-policy and keep it a layer rather than a fork. Keep the model behind the provider-agnostic interface (the runnable code already does this, any model via one call), pin behavior with the eval set rather than with brittle prompt hacks, and pair every "do not fabricate" rule with the faithfulness eval that enforces it, so you can lean on the model less as it gets better. When the new model lands, you swap it, re-run the eval gate, and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved. *Deeper:* [Agents](../../../topics/agents.md).

---

## Real-world reference points

Read these as directional vendor and market benchmarks rather than audited results. They still tell a consistent story.

- **Payback around 3 months.** One widely cited benchmark puts AI-SDR payback at 3.2 months against roughly 8.7 months for a ramped human SDR, because pipeline starts in days. Fast payback comes from volume, and volume is what makes guardrails non-negotiable. [[UserGems](https://www.usergems.com/blog/are-ai-sdrs-worth-it)]
- **Pipeline generation up 2 to 3x.** Early adopters report a 2 to 3x increase in pipeline generation, with more modest broad averages (around a 20% to 24% lift in qualified pipeline) and rare outliers far higher. The spread is the point: results depend on data quality and targeting, which is what the qualification and eval layers are for. [[Apollo](https://www.apollo.io/insights/how-do-revenue-leaders-calculate-expected-pipeline-from-an-ai-sdr-deployment)]
- **Qualification accuracy.** Automated lead scoring is reported in the 85% to 95% range against roughly 60% to 75% for manual qualification, which is the case for an explainable score you can calibrate against outcomes. [[UserGems](https://www.usergems.com/blog/are-ai-sdrs-worth-it)]
- **Provider signal.** Salesforce reported running its own Agentforce sales agents against its dormant-lead backlog and sourcing new pipeline from leads that were otherwise going untouched, an existence proof of the enrich-qualify-draft loop at scale. [[Salesforce](https://www.salesforce.com/news/stories/first-year-agentforce-customer-zero/)]
- **Compliance stakes.** Under CAN-SPAM each violating email carries a penalty of up to 53,088 dollars, and the sender stays responsible even for mail a vendor sends on its behalf, which is why the compliance elements are code-checked on every draft. [[FTC](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business)]
- **tau2-bench:** pass^k collapses as k grows; reliability is the shippable bar, above average accuracy. [[paper](https://arxiv.org/abs/2506.07982)]

---

## Research to know

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis 2020): the grounding pattern that keeps claims tied to evidence, applied here to lead signals and product proof.
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao 2022): reason and act in a loop, the shape of the enrichment agent.
- [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): systems that grade their own evidence before answering, the pattern behind grade-before-send.
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495): check the draft's claims before returning it, to cut fabrication.
- [tau2-bench](https://arxiv.org/abs/2506.07982): evaluating tool-using agents on multi-turn tasks with a reliability metric.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md)); [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../../resources/harness_engineering.md); Aishwarya's talks ([Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k) and [Generative AI in the Real World](https://www.youtube.com/watch?v=Ajiu8uyfSq0), both on O'Reilly) and her [YouTube channel](https://www.youtube.com/channel/UCf9CdAgj8AHmpMwyoe67w7w); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), [Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents), and [Multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system).
- FTC, [CAN-SPAM Act: A Compliance Guide for Business](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business); ICO, [Legitimate interests](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/a-guide-to-lawful-basis/lawful-basis-for-processing/legitimate-interests/); and [GDPR Article 6](https://gdpr-info.eu/art-6-gdpr/), the regulatory primary sources the compliance guardrail encodes.
- [Arize observability docs](https://arize.com/docs/) and [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/concepts/low_level/), the tracing loop and state-machine primitives the [code](code/) uses.

---

## Related in this repo

Topics: [RAG](../../../topics/rag.md) · [Agents](../../../topics/agents.md) · [Evaluation](../../../topics/evaluation.md) · [Production and LLMOps](../../../topics/production.md) · [Safety and Security](../../../topics/safety-security.md). Guides: [Harness Engineering](../../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Prepping to be asked this? See the [interview prep hub](../../README.md).
