# Designing a Financial-Operations Decisioning Agent

## The interview question

> "Design an agent that processes a repetitive financial-operations workflow end to end, for example invoice approval or expense audit: extract the facts, apply policy, decide or route, with a full audit trail and human sign-off on exceptions. Walk me through it."

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real-world benchmarks and deployment data, and it ships with [runnable, provider-agnostic LangGraph code](code/).

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that one core component is non-deterministic, so evaluation moves from a final gate to the center of the design. In a decisioning workflow that handles money, that shift is the whole game: a wrong decision is a real loss or a real compliance breach, so correctness and a defensible audit trail carry more weight than raw speed.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/financial-ops-decisioning (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs, especially the split between deterministic policy and model judgment, and how false-approve and false-deny are evaluated separately. When we get to the code, run `code/run.py` and explain what each node in the graph does and how the audit hash chain works."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine

This case is built on the same 5-layer spine as the rest of the collection: **1 the model**, **2 the wrapping layer** (the architecture: knowledge, tools, and memory), **3 evals and guardrails**, **4 production and ops**, and **5 optimization** (where multi-agent lives). System design is the work of composing them into one coherent system. For how we use this spine and why it is a starting point rather than a rulebook, see [the spine](../README.md#the-spine-how-we-think-about-ai-system-design).

The rest of this case study walks these layers for this problem, going deepest where it is hardest.

---

## The answer

### Step 1: scope before you architect

The first move is to clarify the problem, well before any vector database, because in AI the largest failures are designed in before a single prompt is written. That is doubly true here, where an over-eager auto-approval is money out the door and a missing audit record is a compliance finding. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** Accounts-payable and finance-operations analysts key invoices and expenses into an ERP, match them against purchase orders and policy, and approve or reject them one by one. Volume is high and most items are routine and low-value, which makes the work slow, repetitive, and error-prone under a month-end deadline. Industry cost estimates put a manually processed invoice on the order of 10 to 20 dollars, and automated processing closer to a couple of dollars.
- **Outcome, written before the system.** Auto-decide the clean, in-policy majority and route only genuine exceptions to a human, so analysts spend their time on judgment calls rather than data entry. Measured by straight-through-processing rate (share decided without a human), decision accuracy split into false-approve and false-deny rates, and audit completeness, with a hard ceiling on wrongful approvals.
- **The AI intervention, narrowed until it hurts.** Extract fields from the document, apply the written policy in deterministic code, decide approve, deny, or route-to-human, and record every decision. The agent proposes; a human disposes on anything high-impact or uncertain. It stays well short of an autonomous treasury that moves money on its own.
- **System and safety.** A labelled evaluation set that gates every release, an immutable audit trail on every decision, human approval gates on high-impact and low-confidence cases, defense against instructions smuggled inside a document, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): what is the dollar ceiling above which a human must sign off; what is the cost of a wrong approval versus a wrong denial; how much of the policy is written as hard rules versus analyst judgment; what document types and formats arrive; how clean is the reference data (vendor master, purchase orders, prior invoices); what are the retention and segregation-of-duties requirements; how fresh must the policy rulebook be. This also avoids the traps that sink these projects: leading with "build an agent that reads invoices" (solutioning in the problem statement), folding invoices, expenses, vendor onboarding, and fraud into one system (over-scoping), and designing without a measurable owner for the false-approve rate.

> **Real outlier: PwC and Anthropic, 2025.** In production deployments, PwC reported insurance **underwriting cycles compressed from about 10 weeks to about 10 days**, opening lines of business that were not previously economically viable, with clients across their agentic builds reporting delivery improvements of up to 70%. That is the shape of the win here: the value is in compressing a slow, judgment-heavy decisioning workflow, and the constraint is that these are regulated decisions, so the machinery around the model (policy, evidence, sign-off) is the product. It is Problem-First layer 2 (outcome) and layer 4 (safety) at industry scale. [[Anthropic](https://www.anthropic.com/news/pwc-expanded-partnership)] [[PwC](https://www.pwc.com/us/en/about-us/newsroom/press-releases/anthropic-pwc-expand-alliance-agentic-enterprise.html)]

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data, because every method in the wrapping layer below is chosen for a specific shape of data. Change an assumption and the method changes with it. For this case study, assume:

- **Inputs: structured plus semi-structured.** Some items arrive as clean structured records from an ERP or a supplier portal, and some arrive as semi-structured documents (PDF invoices, scanned receipts, freeform expense notes) where the same facts sit in different layouts. *This is why* extraction is the model's job (turning a messy document into typed fields), and why a layout-aware parser earns its place for the PDF and scan path while structured feeds skip extraction entirely.
- **An explicit policy rulebook exists.** The business already has written rules: spending limits per category, an approver threshold, non-reimbursable categories, required purchase-order matches, tax and duplicate checks. *This is why* the policy decision is deterministic code rather than a model judgment: the rule is knowable, so you encode it exactly and keep it auditable, and you reserve the model for the genuinely fuzzy calls.
- **Auditability is a hard requirement.** A regulator or an auditor must be able to see who decided what, when, and on what evidence, and be confident the record was not edited after the fact. *This is why* an immutable audit record on every decision is load-bearing here, promoted from a framework default into a wall you engineer deliberately.
- **High-impact steps are gated to a human.** Above a dollar threshold, or when the evidence is thin, a person signs off before anything posts. *This is why* the workflow has three outcomes (approve, deny, route-to-human) with route-to-human as the safe default, rather than a binary auto-decision.
- **Correctness and traceability matter more than latency.** A decision that lands in a few seconds or a few minutes is fine; a decision that is wrong or unexplainable is not. *This is why* you can afford verification passes, a second look at low-confidence extractions, and a human sign-off step, and why you optimize for the false-approve and false-deny rates rather than for p95 latency.
- **Volume: high and repetitive, with a long clean tail.** Thousands of items a day, most of them routine and in policy, a minority genuinely exceptional. *This is why* the design automates the clean majority and concentrates human attention on the exceptions, which is exactly where the analyst's judgment is worth paying for.

Keep these in view as you read the wrapping layer: each choice below points back to one of them. If your inputs are all clean structured feeds, the extraction layer thins out; if your policy is mostly analyst judgment rather than written rules, the balance shifts toward the model and the eval burden grows.

### Step 2: walk the layers for this system

**Layer 1, the model.** The model has one job in this system: read a semi-structured document and return typed fields (vendor, invoice id, amount, currency, date, category, line items). Because the same invoice can be phrased and laid out a hundred ways, this is exactly the kind of messy-input-to-clean-output task a model is good at, while the exact-rule policy decision is exactly the kind you keep away from it. You handle the model's non-determinism three ways: **structured output** so it returns typed JSON rather than prose ([schema-constrained decoding](https://platform.openai.com/docs/guides/structured-outputs) forces the shape, and provider [tool schemas](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) do the same), a **field-level confidence** signal so a shaky extraction routes to a human, and a **deterministic policy step downstream** so the decision itself never depends on a phrasing the model happened to pick. A routing strategy fits well: a cheap fast model reads the clean structured-ish documents, and a stronger multimodal model handles the messy scans, with the eval set deciding where the line sits.

**Layer 2, the wrapping layer (the architecture).** This is where most of the design lives, so go deep here. It is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. In a decisioning agent the knowledge is the policy and the reference data, the tools act on the ERP and the approval queue, and the memory is the audit trail itself.

**Knowledge: extraction, the policy rulebook, and reference data.** Two kinds of knowledge feed a decision. The first is the facts of the document, which you have to extract. The second is what the business rules and reference data say about those facts. Each choice below points back to the [assumptions](#the-assumptions-we-make-about-the-data-and-the-use-case) above.

```
┌────────────────────────────────────────────────────────────┐
│ Knowledge pipeline: extraction to a deterministic decision │
└────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │              document               │
  └─────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────┐
  │          1  Parse / ingest          │
  │            text + layout            │
  └─────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────┐
  │  2  Schema-constrained extraction   │
  │            typed fields             │
  └─────────────────────────────────────┘
                     ▼                    low?
  ┌─────────────────────────────────────┐    ┌──────────────────┐
  │         3  Field confidence         │───▶│ route to a human │
  └─────────────────────────────────────┘    │  thin evidence   │
                     ▼                       └──────────────────┘
  ┌─────────────────────────────────────┐
  │         4  Reference match          │
  │ vendor master · PO · prior invoices │
  │    enriched, deduplicated record    │
  └─────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────┐
  │       5  Deterministic policy       │
  └─────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────┐
  │   approve / deny / route-to-human   │
  └─────────────────────────────────────┘
```

**1. Parse and ingest.** A clean ERP feed is already structured, so it skips straight to the policy step. A PDF invoice or a scanned receipt is not: it is a layout with headings, tables of line items, and totals, and naive text extraction flattens that and loses which number is the subtotal and which is the tax. Parse structure-aware, and for PDFs and scans reach for a layout-aware extractor ([Docling](https://github.com/docling-project/docling), [LlamaParse](https://github.com/run-llama/llama_cloud_services), [Reducto](https://reducto.ai/)) rather than a raw text dump. Attach metadata to every extracted record (source, document type, received-at, entity, currency) because routing, retention, and reconciliation all ride on it. *Given our assumption* of mixed structured and semi-structured inputs, you build the extraction path for the messy documents and let the clean feeds bypass it. *What data decides it:* audit your messiest documents (multi-page invoices, faded scans, multi-column receipts) and measure how often parsing mangles the totals and line items.

**2. Structured extraction.** This is the load-bearing stage of the knowledge layer, because every downstream check reads the fields this step produces. The move is to make the model fill a schema rather than write prose: define the exact fields and types you need (vendor as a string, amount as a number, date as an ISO date, line items as a typed list) and use [schema-constrained decoding](https://platform.openai.com/docs/guides/structured-outputs) or a [tool signature](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) so the output is guaranteed to parse. A schema does two things: it prevents malformed output, and it makes each field individually checkable. For semi-structured documents, a few worked examples in the prompt (a couple of input-document-to-fields pairs) measurably lift accuracy. *Because we assumed the policy is applied to typed facts,* the schema is designed backward from the fields the policy needs, so extraction produces exactly the inputs the rules consume. *What data decides it:* the set of fields your policy actually reads, and a field-level accuracy measurement on a labelled set of documents you extracted by hand.

**3. Field confidence and verification.** Extraction is where silent errors enter, so this stage decides which extractions you trust enough to auto-decide and which you send to a human. The signal to reach for is a **field-level confidence** score, and the important lesson is that raw model token probabilities are a weak signal for this. A stronger approach cross-checks: extract the same document two ways and compare, verify that line items sum to the stated total, and confirm the extracted values appear in the source layout. When confidence on a field the policy needs is low, route the whole item to a human rather than deciding on a shaky number. This is the same idea as a relevance floor in retrieval: convert an uncertain read into a human hand-off instead of a confident mistake.

> **Real outlier: extraction confidence is hard, and it is the whole ballgame.** A 2026 study of LLM-based document field extraction reports frontier models **failing on about 26% of fields** on a 55-field invoice task, and that ordinary token-probability confidence separates good extractions from bad ones only weakly. A purpose-built confidence engine that combines a field-focused pass, a holistic pass, and layout signals reached about **0.928 ROC AUC and cut selective-prediction risk by roughly 70%** over the token-probability baseline. Extraction errors also cascade: current document-extraction benchmarks show OCR and parsing noise propagating straight into every decision built on top of the extracted text, so a wrong field becomes a wrong payment. The takeaway for this design: the metric that decides what auto-approves versus what a human reviews is a confidence signal you have to engineer, and getting it right is worth as much as the extractor itself. [[Beyond Logprobs](https://arxiv.org/abs/2606.24420)] [[OHRBench](https://arxiv.org/abs/2412.02592)] [[trustworthiness scoring for structured outputs](https://arxiv.org/abs/2603.18014)]

**4. Reference match.** A field on its own is not a decision. You match the extracted record against reference data: the vendor master (is this a known, approved vendor), the open purchase orders (does this invoice match a PO within tolerance), and the ledger of prior invoices (has this invoice id or this vendor-amount-date triple already been paid). The duplicate check is the highest-value one, because a duplicate invoice is a double payment, a common and expensive real-world loss. *Because we assumed high volume with a long clean tail,* most items match cleanly and pass through; the ones that fail to match are exactly the exceptions worth a human's time. *What data decides it:* the cleanliness and freshness of your vendor and PO data, and how you handle fuzzy vendor-name matches, which is one of the few places model judgment genuinely helps.

**5. The policy rulebook, applied in code.** Because the written policy is knowable, it lives in deterministic code and the model never has to judge it. Spending limits per category, the approver threshold, non-reimbursable categories, currency handling, tax math, and duplicate detection are all exact rules, and you want them exact: reproducible, testable, and readable by an auditor. A rules engine (even a plain, well-tested function, as in the [code](code/)) evaluates every applicable rule and produces findings, each tagged with the action it forces. This is the [business rules engine](https://en.wikipedia.org/wiki/Business_rules_engine) pattern: the logic lives in one inspectable place rather than scattered through prompts. The reason to keep this out of the model is precisely that you can prove what it will do.

**The split between deterministic rules and model judgment.** The central design call in this system is where the line sits, and the principle is simple: **prefer a rule wherever the rule is knowable, and reserve the model for the genuinely fuzzy calls.** A knowable rule (is the amount over the limit, is the category reimbursable, does the invoice id already exist) is cheaper, exact, reproducible, and auditable in code, so encode it. A fuzzy call (does this freeform business justification describe an allowable expense, is "Acme Corp." the same vendor as "ACME Corporation Ltd" in the master, does this receipt look altered) is where a model earns its place, and every such call carries its own confidence and routes to a human when it is unsure. The failure to avoid is handing the model a decision that policy already answers exactly, because then you have made a deterministic outcome non-deterministic and hard to defend.

```
┌───────────────────────────────────────────────────┐
│ The split: rule where knowable, model where fuzzy │
└───────────────────────────────────────────────────┘

                                  ┌──────────────────┐
                                  │    Each check    │
                                  └─────────┬────────┘
                         ┌──────────────────┴─────────────────┐
                         ▼                                    ▼
        ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
        │            Knowable             │  │              Fuzzy              │
        │    deterministic rule (code)    │  │   model judgment + confidence   │
        │ amount > limit? · duplicate id? │  │    justification allowable?     │
        │      tax math? · PO match?      │  │       vendor fuzzy-match?       │
        │                                 │  │          altered doc?           │
        └────────────────┬────────────────┘  └────────────────┬────────────────┘
                         └──────────────────┬─────────────────┘
                                            ▼
                              ┌───────────────────────────┐
                              │ low confidence on either, │
                              │     route to a human      │
                              └───────────────────────────┘
```

**Decision thresholds and routing.** The workflow has three outcomes, and routing among them is where the policy findings resolve into an action. **Approve** the clean, in-policy, high-confidence majority automatically. **Deny** the hard violations where the rule is unambiguous (a non-reimbursable category, a hard cap breached with no override). **Route-to-human** everything in between: an amount at or above the high-impact threshold, an item out of policy but plausibly legitimate, a low-confidence extraction, a suspected duplicate, or a currency that needs a conversion sign-off. Route-to-human is the safe default, so anything the system is unsure about lands with a person rather than a guess. The thresholds themselves (the dollar ceiling, the confidence floor) are levers you set from your own policy and tune against the eval set; treat any specific number as an example and track the false-approve and false-deny rates as you move it.

**Tools (actions).** Tools are the agent's hands, and here they touch systems of record, so each one is a typed, allowlisted contract with least privilege.

```
  ledger_lookup(invoice_id | vendor+amount+date) -> {seen, po_match}   READ   cheap to trust, catches duplicates
  erp_post(record, decision)                      -> posting_id        WRITE  idempotent, gated behind a decision
  enqueue_for_approval(record, findings)          -> queue_id          WRITE  the human hand-off, the safe default
  write_audit(record)                             -> record_hash       WRITE  append-only, every decision
```

The calls that matter: the **tool description is a prompt** (a vague description causes a wrong call); **least privilege** (read tools are cheap to trust, the ERP write is gated behind an explicit decision and never called on a routed item); **[idempotency](https://en.wikipedia.org/wiki/Idempotence)** (a retried `erp_post` must not pay an invoice twice, which is the same double-payment risk the duplicate check guards); **error handling** (a failed post is caught and retried or escalated, never assumed successful); and the **loop is bounded** (a hard step cap, since this workflow is short and any long loop is a sign something is wrong). This is the [ReAct](https://arxiv.org/abs/2210.03629) pattern kept deliberately tight: the agent has few tools and a short, mostly linear path, because a decisioning workflow wants less agency and more control than an open-ended assistant.

**Memory: the audit trail as durable state.** Memory here is less about conversation and more about a permanent, defensible record. It comes in layers.

```
  WORKING   (this item)      : the extracted fields, the reference matches, and the policy findings in play
  LEDGER    (all items)      : processed invoice ids and paid records, so a duplicate is caught before it pays twice
  AUDIT     (immutable)      : one record per decision (who/what/when/evidence), append-only, tamper-evident
```

The audit trail is the load-bearing piece, because our assumption made auditability a hard requirement. Make it **append-only** and **tamper-evident**: an [audit trail](https://en.wikipedia.org/wiki/Audit_trail) where each record carries a hash of the record before it forms a [hash chain](https://en.wikipedia.org/wiki/Hash_chain), so altering any past record breaks every hash after it and the tampering is detectable. In production this is a write-once store, an immutable table, or a ledger, rather than a mutable row you can quietly edit. The record captures who decided (agent or which human), what was decided, when, and on what evidence (the extracted fields and the policy findings), so the decision is reproducible from the record alone. The [runnable code](code/) implements exactly this hash chain, small enough to read in one sitting.

**Human approval gates.** The exceptions the system routes land in an approval queue where a person signs off before anything posts. This is where [separation of duties](https://en.wikipedia.org/wiki/Separation_of_duties) and the [four-eyes principle](https://en.wikipedia.org/wiki/Four_eyes_principle) live: the agent that prepares a decision is not the party that approves a high-impact one, which is a control auditors expect for money movement. The design goal for blast radius is that the worst outcome of any confusion, a bad extraction, or an adversarial document is a needless routing to a human, and a wrongful high-impact auto-approval stays impossible because the high-impact path always requires a person. Together, knowledge, tools, and memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for a money-moving agent this is the center of the design, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, blocking a post or firing a human hand-off the moment something is off. There is no single accuracy number here, and no generic metric you can copy off a shelf, because "good" means something specific for this workflow. So start where we teach you to start: **from failure modes.** Ask what could go wrong that would be unacceptable for this business (a wrongful approval that pays a bad invoice, a wrongful denial that blocks a legitimate one, a duplicate that pays twice, a missing or edited audit record, an extraction that transposes an amount), then translate each into an observable, measurable behavior. The metrics below are a menu you draw from once you know your failure modes.

**Measure against today's baseline rather than an abstract target.** The right bar is the status quo: the current manual invoice and expense review. Where volume was too high to review every item, so some went out unchecked, an agent that screens them reliably clears a low bar by existing. Where a reviewer already catches the bad invoices well, the agent has to match or beat that, measured retrospectively against the decisions your reviewers made on the same historical invoices. That keeps the eval tied to the money decision the business faces rather than chasing a round accuracy number.

Evaluate at three levels: **each component** (did extraction read the fields correctly, did the reference match catch the duplicate, did the policy apply the right rule), **the whole task** end to end (was the right decision reached and recorded), and **live decisions** (is it still right in production).

> **Real finding: [tau2-bench](https://arxiv.org/abs/2506.07982).** On a tool-using agent, a single-attempt pass rate looks respectable, but pass^k (succeed on all k independent tries) collapses as k grows. A decisioning agent that reaches the right call on a single try yet fails 1 run in 4 is not shippable when each failure is a mispaid or wrongly blocked invoice. Reliability is the bar, above average accuracy.

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: did the decision match the policy given the fields, does every decision have an audit record, does the hash chain verify, did extraction produce the exact amount. Fast, reliable, cheap. Use wherever "good" is objectively checkable, and compare against a labelled reference dataset here.
- **LLM judges.** One model scoring another against an explicit rubric, for the subjective calls: was a fuzzy vendor match reasonable, was a routed exception explained clearly, does a freeform justification plausibly fit policy. Scalable, and a new source of non-determinism, so calibrate before you trust it (below).
- **Human evaluation.** The gold standard you calibrate the other two against, and the party that actually signs off on exceptions. Too slow to run on all traffic, so you sample: calibration, edge cases, and every high-impact decision.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of labelled examples across clean-approve, hard-deny, high-impact, out-of-policy, duplicate, and messy-extraction cases. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Extraction** | wrong amount, misread vendor, dropped line item, bad date | field-level accuracy, total-reconciliation rate, extraction confidence calibration ([ROC AUC](https://en.wikipedia.org/wiki/Confusion_matrix) of the confidence signal) | code-based against hand-labelled documents |
| **Reference match** | misses a duplicate, wrong PO match, bad vendor match | duplicate-catch rate, PO-match precision, fuzzy-match accuracy | code-based, plus an LLM judge for fuzzy matches |
| **Policy decision** | applies the wrong rule, wrong threshold | rule-application accuracy vs the written policy | code-based against expected findings |
| **Decision outcome** | wrongful approve, wrongful deny | **false-approve rate and false-deny rate, reported separately**, plus routing accuracy | code-based against a labelled decision set |
| **Audit** | missing record, editable log, incomplete evidence | audit-completeness rate, hash-chain verification, evidence-sufficiency | code-based |
| **Escalation** | auto-decides what a human should see, or floods the human with clean items | escalation precision, escalation recall, straight-through-processing rate | code-based / LLM judge on a labelled should-route set |
| **End to end** | wrong decision reached and recorded | task success rate, pass@1, **pass^k**, decisions per human touch | scenario suite + judge |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: your job is to choose wisely and instrument only the few that earn their place. A dashboard with 30 metrics and no owner tells you nothing. Pick the two or three per component that map to your real failure modes and drop the rest.

**Measure false-approve and false-deny separately, because they cost very different amounts.** This is the single most important evaluation choice in the case. A blended "decision accuracy" hides the tradeoff that actually matters. A **false approve** pays a bad, duplicate, or out-of-policy invoice: real money out the door, and a compliance exposure. A **false deny** blocks a legitimate payment: a delayed vendor, a frustrated employee, and analyst rework, which is a cost but usually a smaller and more recoverable one. These are the two error types of a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix), and they trade off against each other: tighten the thresholds and confidence floor to drive false approves toward zero, and more legitimate items route to a human (higher false-deny-into-review, lower straight-through rate). Because the costs are asymmetric, you tune that tradeoff on purpose rather than chasing one accuracy number, and you set the false-approve ceiling first and let the straight-through rate be whatever that ceiling allows. Report both rates over time and treat any specific target as an example you track, since the right operating point is a business decision.

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem (the false-approve rate), or is it merely interesting?
- **Reliability:** human review and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check is cheap, a fast judge call is medium, detailed human review is expensive.

High impact and low cost are the must-haves (the false-approve rate, audit completeness, the hash-chain check, the escalation flags). High impact and high cost are strategic investments you run on a sample (a human relabelling of a slice of auto-approved items to catch silent false approves). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the business, run inline so the system can act the moment they trip: a decision that contradicts a hard policy rule, a post attempted on a routed item, a low-confidence extraction, a missing audit write. The action is immediate (block the post, route to a human). Guardrails must be fast and reliable before sophisticated.
- **Improvement flywheel (offline, batch).** Everything else: false-approve and false-deny trends, straight-through rate, extraction accuracy on a sample, drift as vendors and formats change. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live decisions:

| Metric | Job | Why it matters |
|---|---|---|
| false-approve rate | guardrail + flywheel | the hard ceiling from scoping; money out the door; must stay near zero |
| policy-contradiction / unsafe-post rate | guardrail | a decision that breaks a hard rule must be blocked in the moment |
| low-confidence extraction trigger | guardrail | fires the human hand-off before a shaky field becomes a decision |
| audit-write success + chain verification | guardrail | a decision without a verifiable record is a compliance breach |
| false-deny rate | flywheel | legitimate items wrongly blocked; vendor and analyst friction |
| straight-through-processing rate | flywheel | share decided without a human, the core efficiency outcome |
| escalation rate | flywheel | too high means it is not helping; too low means risky over-approval |
| duplicate-catch rate | flywheel | health of the double-payment defense |
| decisions per human touch, cost per decision | flywheel | unit economics, the number finance asks about |
| extraction field accuracy on a live sample | flywheel | catches format drift before it drives wrong decisions |

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label a few hundred examples, run the judge on the same set, measure agreement (percent agreement or [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the underlying model, and run the judge on a sampled percentage of live decisions to control cost.

Then run the discovery loop, because the world will always produce failures your metrics were never built for (a new invoice format, a novel duplicate pattern, a vendor gaming the tolerance). Sample live decisions on **signals** (analyst overrides of an auto-decision, reopened items, disputed payments, items that sat in review too long). When a signal keeps firing but your metrics look clean, that gap is the tell: a human reads those decisions, names the dimension you were not measuring, and it becomes a new metric added back into the reference dataset. Evaluation is never finished. You build for the failures you can anticipate, and you monitor to discover the ones you cannot.

![Eval and discovery loop: pre-deployment validation, then production monitoring, with the reference set growing each cycle](../assets/eval_discovery_loop.png)

Instrument the LangGraph app with OpenInference so every node, extraction, and decision becomes a span in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. Reading traces is how you find the exact step where a wrong decision was made, whether it was a misread field or a misapplied rule. Tooling worth knowing: Ragas and DeepEval for component metrics, promptfoo for CI gates, Arize Phoenix for tracing and online evals. *Deeper:* [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../../topics/evaluation.md).

**Layer 4, production and ops.** The loop that keeps it alive at scale: batch and streaming ingestion from the ERP and mailboxes, connection pooling to the systems of record, an approval queue with SLAs, retention on the audit store, and observability so every decision is traceable. Detail is in Follow-ups 1 and 3.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: confidence-based auto-approval (raise the share decided without a human as the extractor and the confidence signal improve), model routing (a cheap model for clean documents, a strong multimodal model for messy scans), caching the stable extraction prompt, and multi-agent. Confidence-based auto-approval and routing pay off first, and as finance operations spans invoices, expenses, and vendor onboarding, an orchestrator routing by document type to policy specialists plus an exception-investigation sub-agent is the design to reach for, which Follow-up 5 lays out. As scope grows, this is where the more innovative approaches enter.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (extract, apply the rules, route everything that is not a clean in-policy match to a human), prove it with evals, and only widen the auto-decision envelope (a higher confidence-based auto-approval share, model judgment on fuzzy matches) once the false-approve rate says the constrained version is solid. You earn agency through evaluation before you grant it. Every increase in what auto-decides is paid for with an eval that shows the false-approve rate held.

Composed, the layers give one architecture:

```
┌───────────────────────────────────────────────────────────────────────┐
│ Observability: every node, extraction, and decision is a span (Arize) │
└───────────────────────────────────────────────────────────────────────┘

                                          ┌──────────────────┐
                                          │     document     │
                                          │   feed or PDF    │
                                          └──────────────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │  Ingest / parse  │
                                          └──────────────────┘
                                                    ▼
                                         ┌─────────────────────┐
                                         │       Extract       │
                                         │ schema-constrained, │
                                         │ model, typed fields │
                                         └─────────────────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │ Field confidence │
                                          └──────────────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │ Reference match  │
                                          │    ledger, PO    │
                                          └──────────────────┘
                                                    ▼
                                        ┌──────────────────────┐
                                        │ Deterministic policy │
                                        │   rulebook in code   │
                                        └───────────┬──────────┘
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                          ▼                          ▼
             ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
             │        APPROVE        │  │         DENY          │  │    ROUTE-TO-HUMAN     │
             │       erp_post*       │  │        record         │  │ enqueue_for_approval, │
             │                       │  │                       │  │    human sign-off     │
             └───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘
                         └──────────────────────────┬──────────────────────────┘
                                                    ▼
                                      ┌───────────────────────────┐
                                      │  Immutable audit record   │
                                      │ append-only, hash-chained │
                                      └───────────────────────────┘
                                                    ▼
                                          ┌──────────────────┐
                                          │       END        │
                                          └──────────────────┘


  low confidence routes to route-to-human; erp_post* is gated behind an APPROVE decision, idempotent
```

Read it as the spine composed: the model (layer 1) extracts typed fields, wrapped in the policy rulebook, tools, and the audit memory (layer 2), gated by evals (layer 3) that run inline as guardrails on the false-approve and audit checks and offline as a flywheel on the error rates, run under production observability (layer 4), with optimization (layer 5) held to what this system actually needs, which is confidence-based auto-approval and model routing rather than a second agent. Composing exactly these pieces into one coherent system is the system design. The field-confidence floor and the deterministic policy are what make route-to-human the safe default: when extraction is shaky or the amount is high-impact, the item goes to a person with the full evidence rather than to an auto-decision.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this architecture (minus the scale pieces): extraction into typed fields, a deterministic policy rulebook, an approve / deny / route-to-human branch, human sign-off on high-impact and low-confidence cases, and an immutable, hash-chained audit record for every decision. It runs offline with a deterministic parser and rulebook, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                              # run the scenarios and self-check (also verifies the audit chain)
python run.py "Vendor: Acme | Invoice: INV-9 | Amount: USD 90 | Date: 2026-07-10 | Category: software"
```

Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route field extraction through a real model; the policy decision stays deterministic either way. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now handle 10x the volume. Where does it break, and how do you scale it?"

Name where it breaks first, then scale that, cheapest lever first.

- **The human-review queue (layer 4).** At 10x, the exceptions routed to people are the real bottleneck, ahead of any model cost. The cheapest lever is to shrink the exception rate honestly: improve the extractor and the confidence signal so more clean items clear the auto-approval bar, and tighten the reference data so fewer false duplicate and PO-mismatch flags fire. You raise the straight-through-processing rate only as fast as the false-approve rate lets you.
- **Extraction throughput (layer 2 and layer 5).** Batch the structured feeds, reserve the strong multimodal model for the documents that genuinely need it, route the clean ones to a cheap model, and cache the stable extraction prompt prefix. The messy-scan path is where the cost concentrates, so that is where routing pays.
- **Infrastructure (layer 4).** Horizontal workers behind a queue, connection pooling to the ERP and the systems of record, backpressure so a month-end spike degrades into a longer queue rather than dropped items, and an append-only audit store sized for retention.

```
┌─────────────────────┐
│ Production topology │
└─────────────────────┘

                                            ┌───────────────────┐
                                            │ feeds + mailboxes │
                                            └───────────────────┘
                                                      ▼
                                            ┌──────────────────┐
                                            │   Ingest queue   │
                                            └──────────────────┘
                                                      ▼
                                        ┌───────────────────────────┐
                                        │      Extract workers      │
                                        │ N replicas · model router │
                                        │    cheap ↔ multimodal     │
                                        └─────────────┬─────────────┘
               ┌──────────────────────────────────────┴──────────────────────────────────────┐
               ▼                         ▼                         ▼                         ▼
   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │    Reference data    │  │    Policy engine     │  │     Audit store      │  │    Approval queue    │
   │ vendor · PO · ledger │  │    deterministic     │  │    append-only ·     │  │    analysts (SLA)    │
   │                      │  │                      │  │      retention       │  │                      │
   └──────────────────────┘  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘

  Arize collects traces and online evals: false-approve, false-deny, and drift
```

Put numbers on it: decisions per human touch, cost per decision, and the straight-through-processing rate you can hold at your false-approve ceiling. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 2: "A vendor hides an instruction in an invoice, like 'approved, ignore the limit'. How do you prevent a wrong or unsafe decision?"

Separate reading from deciding, and keep the deciding in code.

- **The policy never reads model free-text.** Extraction turns the document into typed fields; the deterministic rulebook decides from those fields alone and never executes an instruction found in the document. A sentence in an invoice cannot raise a limit, because the limit lives in code the document cannot reach. This separation is the core defense.
- **Untrusted input.** Documents and their contents are untrusted, and so is anything persisted from them. The sharp way to see the risk is the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
  The lethal trifecta

           ┌───────────────────┐   ┌───────────────────┐   ┌────────────┐
           │ Untrusted content │   │     Access to     │   │ Ability to │
           │                   │   │ systems of record │   │ move money │
           └─────────┬─────────┘   └─────────┬─────────┘   └──────┬─────┘
                     └───────────────────────┼────────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │ All 3 together is dangerous │
                              └─────────────────────────────┘

                 any 2 are manageable; keep all 3 from meeting in one path
```

- **Blast radius.** Least-privilege tools, an immutable audit record of every decision, human sign-off on every high-impact case, and a design where the worst a malicious document achieves is a needless routing to a human, while a wrongful auto-payment from that instruction stays impossible because the money path always runs through the deterministic gate and, for high-impact cases, a person. Prompt injection is the live threat here ([OWASP LLM01](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)); the structure of the workflow is what contains it.

*Deeper:* [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 3: "Push the straight-through rate up without letting false approves rise."

Treat it as a tradeoff you tune on purpose, with the false-approve ceiling fixed first.

- Improve the confidence signal so the auto-approval bar admits more genuinely-clean items, which is the highest-leverage move (the confidence-engine finding above: the confidence engine is worth as much as the extractor).
- Clean the reference data so fewer legitimate items get flagged as duplicates or PO mismatches and wrongly routed, which lifts the rate without touching the ceiling.
- Cache the stable extraction prompt prefix to cut cost per decision, and route clean documents to a cheaper model, so more volume is affordable at the same accuracy.

> **Real number: [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).** Caching the stable prefix of a prompt (the extraction schema, the field instructions, worked examples) can cut cost by up to about 90% and latency by up to about 85% on the cached portion, per provider documentation. On a decisioning agent whose extraction schema and instructions are identical on every document, this is a high-leverage optimization.

The false-approve rate holds because the eval set from Layer 3 gates each change: if raising the auto-approval share pushes false approves above the ceiling, it does not ship. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 4: "New regulation: every payment over a threshold needs two approvers and 7 years of retention. How does the design change?"

That single constraint promotes two normally-optional controls into load-bearing walls: **segregation of duties** and **records retention**. Add a second, distinct human approver on the high-impact path ([four-eyes](https://en.wikipedia.org/wiki/Four_eyes_principle) / [separation of duties](https://en.wikipedia.org/wiki/Separation_of_duties)), so the party that prepares a decision is never the only party that approves it, and extend the audit store to a retention-managed, write-once medium that holds every record and its evidence for the required period. The decision logic and the extraction are unchanged. This is the general pattern: a domain constraint is what forces you to engineer a control you would otherwise accept as a framework default, which is the same reason the audit trail was load-bearing from the start here. *Deeper:* [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

For the realistic version of finance operations, where invoices, expense reports, and vendor onboarding each carry their own rulebook, the design to recommend is an orchestrator that routes by document type to policy specialists, with an exception-investigation sub-agent that takes over anything the linear path flags. You run a single tight workflow as one agent while the scope is one document type and the path is short and linear, and you let the traces show when a second workflow or an investigation task has arrived. Here is when multi-agent (layer 5, optimization) pays off and how the architecture extends.

**When multi-agent earns its place:**
- **Distinct sub-workflows need distinct policy, tools, and reference data.** Accounts-payable invoices, travel-and-expense reports, and vendor onboarding each have their own rulebook, their own systems of record, and their own document shapes. Holding all of them in one agent bloats the context and blurs the behavior, while specialists keep each policy tight and inspectable.
- **A genuinely different reasoning job appears.** Fraud investigation is not policy application: it hunts for patterns across many documents and vendors, which is a research-style task that fans out and fits a specialist agent, unlike the linear decisioning path.
- **The work decomposes into independent sub-tasks that can run in parallel.** When quality and throughput gains outweigh the extra tokens, parallelism pays.
- **The task is valuable enough to pay for it.** Multi-agent trades tokens for capability, which fits high-value regulated work.

**How you would extend this architecture.** Keep the single-agent decisioning design intact and add an orchestrator that classifies the document type and routes it to a policy sub-agent, one per workflow, each owning its rulebook, reference data, and tools. An exception-investigation sub-agent picks up anything a policy sub-agent flags, hunting patterns across vendors and documents the way a linear check cannot. The same extraction discipline, deterministic-policy gate, human sign-off, evals, immutable audit trail, and observability now wrap the whole system across agents.

```
┌─────────────────────────────────────────┐
│ Observability: spans across every agent │
└─────────────────────────────────────────┘

                                              ┌──────────────────┐
                                              │     document     │
                                              └──────────────────┘
                                                        ▼
                                              ┌──────────────────┐
                                              │  Ingest / parse  │
                                              └──────────────────┘
                                                        ▼
                                       ┌────────────────────────────────┐
                                       │          Orchestrator          │
                                       │ classify document type · route │
                                       └────────────────┬───────────────┘
               ┌────────────────────────────────────────┴───────────────────────────────────────┐
               ▼                          ▼                          ▼                          ▼
   ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
   │   AP-invoice policy   │  │  T&E-expense policy   │  │   Vendor-onboarding   │  │       Exception       │
   │    PO + invoice KB    │  │   expense-policy KB   │  │ policy: KYC / vendor  │  │     investigation     │
   │       erp_post*       │  │  reimbursement rules  │  │  master KB + checks   │  │ cross-vendor patterns │
   │                       │  │                       │  │                       │  │       read-only       │
   └───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘
               └──────────────────────────┴─────────────┬────────────┴──────────────────────────┘
                                                        ▼
                                            ┌──────────────────────┐
                                            │ Deterministic policy │
                                            │   + human sign-off   │
                                            └──────────────────────┘
                                                        ▼
                                           ┌────────────────────────┐
                                           │ Immutable audit record │
                                           └────────────────────────┘
                                                        ▼
                                              ┌──────────────────┐
                                              │       END        │
                                              └──────────────────┘


  every money-moving post still runs through the gate and a person
```

> **Real data: Anthropic's multi-agent research system.** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens, and token usage explained roughly 80% of the performance gap. The takeaway is to spend those tokens where task value and genuine parallelism justify them, and to stay single-agent where they do not. A linear decisioning workflow does not; a cross-vendor fraud investigation might. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]

The honest rule: start with the single agent, instrument it, and let the traces tell you when a sub-workflow or an investigation task has outgrown one agent. Multi-agent is where this design goes once finance operations is big enough to need it. *Deeper:* [Agents](../../../topics/agents.md).

---

## Follow-up 6: "Now handle a new document type and a second entity in another currency."

The architecture does not change; the extraction schema and the evaluation do. Add the new document type to the extraction schema with a few worked examples, extend the reference data and the rulebook for the second entity, and handle the currency in the deterministic policy (a non-base currency routes to a human for a conversion sign-off, or a rate table is applied and recorded in the audit evidence). The eval set grows a labelled slice per document type and per entity, so field accuracy and the false-approve and false-deny rates are measured per type rather than assumed. Extraction, the policy gate, the audit trail, and the human sign-off apply unchanged. This is the recurring lesson: most "make it handle X" follow-ups are answered in the data and eval layers, and the box diagram stays the same. *Deeper:* [Evaluation](../../../topics/evaluation.md).

---

## Follow-up 7: "A better model drops next quarter. How do you avoid a rewrite?"

Build the harness on-policy and keep it a layer rather than a fork. Keep the model behind the provider-agnostic interface (the runnable code already does this, any model via one call), pin behavior with the eval set (the false-approve and false-deny rates, extraction field accuracy) rather than with brittle prompt hacks, and pair every "do not do X" extraction rule with an eval so you can delete the rule when a better model makes it obsolete. Because the decision is deterministic, a stronger model changes only extraction quality, which means a swap can raise the auto-approval share safely: you swap the model, re-run the eval gate, confirm the false-approve rate held, and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved. *Deeper:* [Agents](../../../topics/agents.md).

---

## Real-world reference points

- **PwC and Anthropic (2025):** insurance underwriting cycles compressed from about 10 weeks to about 10 days in production, opening lines of business that were not previously viable, with delivery improvements of up to 70% across agentic builds. Compressing slow, regulated decisioning is the win; the machinery around the model is the product. [[Anthropic](https://www.anthropic.com/news/pwc-expanded-partnership)] [[PwC](https://www.pwc.com/us/en/about-us/newsroom/press-releases/anthropic-pwc-expand-alliance-agentic-enterprise.html)]
- **MIT Sloan and Stanford GSB (2025):** across hundreds of thousands of transactions at 79 companies, AI cut the average monthly financial close by about 7.5 days and raised general-ledger granularity by about 12%, with accountants using AI as a tool alongside their judgment. Automation compresses the close cycle by real margins when a human stays on the judgment calls. [[Stanford GSB](https://www.gsb.stanford.edu/insights/ai-reshaping-accounting-jobs-doing-boring-stuff)] [[MIT Sloan](https://mitsloan.mit.edu/ideas-made-to-matter/how-generative-ai-can-make-accountants-more-productive)] [[paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240924)]
- **Document field extraction (2026 study):** frontier models fail on about 26% of fields on a 55-field invoice task, and ordinary token-probability confidence flags the errors only weakly; a purpose-built confidence engine reached about 0.928 ROC AUC and cut selective-prediction risk by roughly 70%. Separately, current document-extraction benchmarks show OCR and parsing noise cascading into every downstream decision. The confidence signal that decides auto-approve versus human review is worth as much as the extractor. [[Beyond Logprobs](https://arxiv.org/abs/2606.24420)] [[OHRBench](https://arxiv.org/abs/2412.02592)]
- **Invoice automation economics:** manual invoice processing is often estimated in the low tens of dollars per invoice and automated processing at a fraction of that, with leading no-touch rates well above half of volume. The value is real, and it concentrates in auto-deciding the clean majority while routing exceptions.
- **tau2-bench:** pass^k collapses as k grows; reliability is the shippable bar for a money workflow, above average accuracy.

---

## Research to know

- [ReAct](https://arxiv.org/abs/2210.03629) (Yao 2022): reason and act in a loop, the shape of the agent, kept deliberately tight here.
- [OHRBench](https://arxiv.org/abs/2412.02592) (2024): OCR Hinders RAG, showing how document-extraction errors cascade into downstream retrieval and decisions, the extraction risk this system rests on.
- [Beyond Logprobs](https://arxiv.org/abs/2606.24420) and [trustworthiness scoring for structured outputs](https://arxiv.org/abs/2603.18014): confidence signals that decide what a machine can trust versus what a human must review.
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495) (Dhuliawala 2023): verify a draft before returning it, the idea behind cross-checking an extraction.
- [tau2-bench](https://arxiv.org/abs/2506.07982) (2024): evaluating tool-using agents on multi-turn tasks with a reliability metric.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md)); [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../../resources/harness_engineering.md); Aishwarya's talks ([Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k) and [Generative AI in the Real World](https://www.youtube.com/watch?v=Ajiu8uyfSq0), both on O'Reilly) and her [YouTube channel](https://www.youtube.com/channel/UCf9CdAgj8AHmpMwyoe67w7w); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- Anthropic and PwC, [expanded agentic-enterprise alliance](https://www.anthropic.com/news/pwc-expanded-partnership), and Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents).
- OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf), and [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs).
- Choi (Stanford GSB) and Xie (MIT Sloan), [Human + AI in Accounting: Early Evidence from the Field](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240924) (2025); Stanford GSB summary, [AI is reshaping accounting jobs by doing the boring stuff](https://www.gsb.stanford.edu/insights/ai-reshaping-accounting-jobs-doing-boring-stuff).
- [Arize observability docs](https://arize.com/docs/) and [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/), the tracing loop and state-machine primitives the [code](code/) uses.

---

## Related in this repo

Topics: [RAG](../../../topics/rag.md) · [Agents](../../../topics/agents.md) · [Evaluation](../../../topics/evaluation.md) · [Production and LLMOps](../../../topics/production.md) · [Safety and Security](../../../topics/safety-security.md). Guides: [Harness Engineering](../../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Prepping to be asked this? See the [interview prep hub](../../README.md).
