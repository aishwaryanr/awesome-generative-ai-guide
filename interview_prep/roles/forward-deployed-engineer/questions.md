# FDE Question Bank

A large, current question bank for the Forward-Deployed Engineer loop, grouped by theme, each with a concise model answer. Answers are meant to anchor your own, not to be recited. There are 45 questions across 9 themes.

Pair this with the repository's [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and the [Role-Based Interview Prep](../../role_based_prep.md) tracks.

Jump to: [Role and motivation](#1-role-and-motivation) · [GenAI foundations](#2-genai-foundations) · [RAG](#3-rag) · [Agents and MCP](#4-agents-and-mcp) · [Evaluation](#5-evaluation) · [Deployment, cost, and reliability](#6-deployment-cost-and-reliability) · [Responsible AI and security](#7-responsible-ai-and-security) · [Case and decomposition](#8-case-and-decomposition-judgment) · [Customer communication and business](#9-customer-communication-and-business)

---

## 1. Role and motivation

**1. What does a Forward-Deployed Engineer actually do, and why is it different from a standard SWE role?**
An FDE embeds with a customer, learns their domain, and ships working software inside the customer's environment, owning the outcome from a vague business problem to a running, trusted system. A standard SWE builds the core product for everyone from headquarters and is judged on the code; an FDE builds the customer-specific integration and is judged on whether that one deployment succeeds and gets adopted. The FDE spends a large share of the week in customer discovery, treating that discovery as engineering work. The line that separates FDE from consultant is "ship on day one": deliver running software, not a roadmap.

**2. Why FDE and not a pure backend or research role?**
Because the interesting problems live at the boundary between a capable system and a messy real organization, and I want to own that whole boundary, not just one layer of it. I like shipping something a real team uses within days and iterating with them in the room. I am energized by ambiguity and by translating between technical reality and business stakeholders, which is the core of the job. A pure backend or research role optimizes one dimension deeply; FDE optimizes end-to-end impact for a specific customer, which is what I want.

**3. How do you think about your first 30, 60, and 90 days on a new deployment?**
First 30 days: build trust and map the domain. Run many customer conversations, learn their data and workflows, ship one small but real thing to establish credibility and get access unblocked. Days 30 to 60: stand up a thin end-to-end system on the highest-value workflow, with evaluation and monitoring from the start, and get real users on it. Days 60 to 90: harden, expand to the next workflow, and route the patterns I am seeing back to the product team. The through-line is early credibility, then compounding value, then feedback to the core product.

**4. How do you decide what to build first for a new customer?**
Sequence by a combination of value and risk: find the workflow where a working slice creates visible value fast, and de-risk the biggest unknown early so I do not discover a blocker in month three. I prefer a thin end-to-end path over a deep single component, because an end-to-end skeleton surfaces integration and data problems that a polished component hides. I confirm the success metric with the customer before building, so we agree on what "working" means. If access, data quality, or a security approval is the real bottleneck, I sequence to unblock that first.

**5. A customer wants a feature you are confident will not solve their real problem. What do you do?**
I diagnose before I argue: ask what outcome they are actually chasing and what makes them believe this feature delivers it. I acknowledge what they are right about, then show, with their own data or a quick example, why the feature misses the real problem and what would hit it instead. I offer explicit options with trade-offs rather than a flat refusal, so they keep ownership of the decision. If they still want it and the cost is low, I may build a thin version to prove the point empirically while advancing the better path in parallel.

---

## 2. GenAI foundations

**6. Explain what a transformer is to a technical customer, at a useful level.**
A transformer is a neural network built around self-attention, which lets every token look at every other token and weigh how relevant each is when building its representation. Stacked attention and feed-forward layers turn input tokens into context-aware vectors, and a language model uses them to predict the next token. Attention is what lets the model handle long-range dependencies and run in parallel across a sequence, which is why it scaled where earlier recurrent models did not. For deployment purposes, the practical consequences are the context-window limit, the quadratic cost of attention in sequence length, and tokenization effects.

**7. What is the difference between pretraining, fine-tuning, and in-context learning?**
Pretraining learns general language and world knowledge from a huge corpus by next-token prediction; it is expensive and done once by the model provider. Fine-tuning continues training on a smaller task or domain dataset to change behavior, format, or style. In-context learning changes nothing in the weights: you steer the model at inference time with instructions and examples in the prompt. For a customer, in-context learning and retrieval are the cheap, fast, reversible levers; fine-tuning is the heavier lever for stable behavior at scale.

**8. When would you use RAG instead of fine-tuning to give a model new knowledge?**
Use RAG when the knowledge changes often, is large or proprietary, or needs citations, because retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, or style, or to bake in a stable skill, not for volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior. For most customer deployments RAG is the right first move because it is cheaper, faster to iterate, and easier to govern.

**9. What is context engineering, and why can it matter more than prompt wording?**
Context engineering is deciding what actually enters the model's context window: instructions, retrieved documents, tool results, memory, and what stays out. It matters because models degrade when the context is bloated, noisy, or poorly ordered, so getting the right minimal, well-structured information in usually beats clever phrasing. In deployment this is where most quality gains come from: better retrieval, tighter tool outputs, summarized history, and clear structure. Treat the context window as a scarce, carefully curated budget rather than a place to dump everything.

**10. How do reasoning models differ from standard LLMs, and how does that change how you use them?**
Reasoning models are trained, often with reinforcement learning on verifiable rewards, to spend test-time compute on an internal chain of thought before answering. They trade latency and cost for much stronger performance on math, code, and multi-step problems. They change how you prompt: give the goal and constraints and let the model do the step-by-step work, rather than hand-holding each step. In a deployment you route hard, high-stakes steps to a reasoning model and keep routine, latency-sensitive steps on a smaller fast model.

**11. Explain embeddings to a VP of operations who has never written code.**
An embedding turns a piece of text into a list of numbers that captures its meaning, so that things which mean similar things sit close together in that number space. That lets the system find relevant documents by meaning rather than exact keywords, so a search for "late shipments" can surface a report that says "delivery delays." It is the mechanism behind semantic search and behind giving the AI the right background documents before it answers. The practical payoff for you is that the system can pull the right context out of your data even when the words do not match exactly.

**12. What are the main knobs (temperature, top-p, max tokens, system prompt) and when do you touch them?**
Temperature and top-p control randomness: low values make outputs focused and repeatable, high values make them diverse and creative. For extraction, classification, or anything you evaluate, keep temperature low for consistency; for brainstorming, raise it. Max tokens caps output length and cost; the system prompt sets durable role, rules, and format. In a customer deployment I default to low temperature and a tightly specified system prompt, because reliability and reproducibility matter more than flourish.

---

## 3. RAG

**13. Walk me through a RAG pipeline end to end.**
Ingest and clean the source documents, split them into chunks, embed each chunk, and store the vectors with metadata in a vector index. At query time, embed the query, retrieve the top-k most similar chunks (often with a metadata filter and permission check), optionally rerank them, and assemble a prompt that puts the retrieved context alongside the question with an instruction to answer only from that context and cite sources. The model generates the answer, and you log the query, retrieved chunks, and answer for evaluation and debugging. Each stage is a place quality can break, so you evaluate retrieval and generation separately.

**14. How do you choose a chunking strategy, and how do you defend it to a skeptical customer?**
Chunk on natural structure first (sections, paragraphs, headings) rather than arbitrary character counts, size chunks to hold one coherent idea, and add overlap so answers that straddle a boundary are not cut in half. I attach metadata (source, section, date, permissions) so retrieval can filter and citations are precise. I defend it empirically: build a small labeled question set, measure whether the answer-bearing chunk is actually retrieved at k, and show the customer that number improving as I tune chunk size and overlap. The right strategy is the one the retrieval metrics on their data support, not a rule of thumb.

**15. How do you evaluate a RAG system?**
Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually retrieved (recall and precision at k). For generation, measure faithfulness (is the answer grounded in the retrieved context with no hallucination), answer relevance, and correctness, using a labeled question set and an LLM-as-judge with a clear rubric that you have validated against human labels. Then monitor the same signals in production. Separating the two lets you tell whether a bad answer came from bad retrieval or bad generation.

**16. Why might a model do worse with retrieved context than with none, and what does that tell you?**
Because irrelevant or conflicting retrieved text can distract or mislead the model, and long, noisy context degrades reasoning. It tells you retrieval quality and context construction matter as much as the model, and that more context is not automatically better. The fixes are better retrieval (filtering, reranking), tighter and shorter context, and instructing the model to ignore context that does not answer the question and to say when it lacks grounding. This is why you measure retrieval quality directly rather than assuming retrieval helps.

**17. How do you handle permissions and access control in enterprise RAG?**
Retrieval must respect per-user permissions end to end: store access metadata with each chunk and filter the vector search by what the requesting user is allowed to see, so a user can never retrieve a document they could not open directly. Do not rely on the model to withhold restricted content; enforce it in retrieval, before generation. Keep the index in sync with the source system's permissions, and log access for audit. The common failure is a shared index that leaks restricted documents into answers, so permission-aware retrieval is a hard requirement, not a feature.

**18. When is RAG the wrong tool?**
When the task needs behavior or style change rather than fresh knowledge (that is fine-tuning or prompting), when the knowledge is small and stable enough to just put in the prompt, or when the real need is an action rather than an answer (that is an agent with tools). RAG also struggles when questions require reasoning across many documents at once or aggregations that retrieval cannot assemble, where a structured query over a database is better. Match the tool to the need; do not reach for retrieval reflexively.

**19. How would you get a naive RAG endpoint from 1.5 seconds down under 100 milliseconds?**
First measure where the time goes: embedding the query, the vector search, reranking, and generation. Cache aggressively (embed and cache frequent queries, cache the stable prompt prefix), use a smaller and faster embedding and generation model where quality allows, and cap or drop a slow reranker. Reduce retrieved context so generation has fewer tokens to process, and stream the response so time-to-first-token is low even if full completion is longer. Sub-100ms end to end with a large generation step is often unrealistic, so I would set the honest budget with the customer and optimize time-to-first-token and perceived latency.

---

## 4. Agents and MCP

**20. What makes something an agent rather than a single LLM call, and when do you actually need one?**
An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow will do, since agents add cost, latency, and new failure modes. In deployments I start with the simplest thing that clears the bar and add agency only where the task genuinely requires it.

**21. What is the Model Context Protocol (MCP) and why does it matter for an FDE?**
MCP is an open standard for connecting an agent to tools and data sources through a common interface. It matters because you wire a capability once and reuse it across agents and harnesses, instead of building a custom integration for every tool and every app. For an FDE integrating into a customer's stack, MCP means their systems (a database, a ticketing system, an internal API) can be exposed as reusable servers that any compliant agent can call, which cuts integration time and standardizes auth and permissions. It turns bespoke glue code into a reusable, governable connector.

**22. How do you design tools for an agent so it uses them reliably?**
Give each tool a clear, single purpose, a descriptive name, and a schema with well-documented parameters, because the model chooses tools from their descriptions. Return concise, structured results rather than dumping raw payloads that bloat context. Make tools safe by construction: least privilege, validation, and idempotency where possible, so a repeated or mistaken call cannot do damage. Then evaluate tool selection and argument accuracy directly, because most agent failures are the model picking the wrong tool or passing bad arguments.

**23. How do you keep an agent's cost and latency under control?**
Cache the stable prefix of the context, keep the context minimal and well-ordered, and use cheaper or smaller models for sub-steps while reserving a reasoning model for the hard step. Cap tool calls and loop iterations so a stuck agent cannot spin forever, cache tool results, and give the model only as much reasoning budget as the step needs. Measure cost and latency per task, not just per call, because an agent multiplies calls. Set a hard budget and fail gracefully when it is hit.

**24. Why measure reliability (for example pass^k) for agents, not just average accuracy?**
Because an agent that succeeds on average but fails unpredictably is not shippable. Metrics like pass^k (does it succeed on all k independent attempts) capture consistency, which is what a long, autonomous task in production actually requires. A customer who sees the agent get the same task right 8 times and catastrophically wrong twice will not trust it, regardless of the average. So I report reliability and worst-case behavior, not just a mean score.

**25. How would you handle prompt injection in a customer-facing agent?**
Treat all retrieved and tool-returned content as untrusted input, because indirect injection hides instructions inside documents, web pages, or tool outputs the agent reads. Separate trusted instructions from untrusted data in the prompt structure, and never let the model's reading of a document silently escalate its permissions. Enforce least-privilege tools and require human approval for sensitive or irreversible actions, so even a successful injection cannot do serious damage. Add input and output guardrails and log actions for audit, and test with adversarial injected content before shipping.

**26. When would you build a multi-agent system versus a single agent, and what are the risks?**
Reach for multiple agents when the task decomposes into genuinely separate roles or when parallelism across independent subtasks buys real speed or clarity, for example a researcher agent feeding a writer agent. The risks are compounding: more coordination overhead, more places to fail, higher cost and latency, and harder debugging because errors propagate across agents. I default to a single well-designed agent with good tools and only add agents when a single one demonstrably cannot handle the coordination. Multi-agent is a real tool, but it is often reached for too early.

---

## 5. Evaluation

**27. How do you evaluate an LLM system beyond "it looks right in the demo"?**
Start from what success means as a checkable outcome, build a labeled evaluation set that covers realistic and adversarial cases, and choose metrics that isolate the capability rather than proxies. Combine automated scoring (exact checks where possible, an LLM-as-judge with a validated rubric where not) with periodic human review and real production feedback. Version the eval set and run it on every change, so quality is a number you defend, not a vibe. The demo shows the best case; the eval set shows the distribution, and the distribution is what ships.

**28. How do you build and trust an LLM-as-judge?**
Write a clear, specific rubric for what a good output is, then validate the judge against a set of human-labeled examples and measure agreement before trusting it at scale. Use a strong model as the judge, keep its task narrow (score one dimension at a time), and watch for known biases like preferring longer answers or its own style. Re-check agreement periodically and on edge cases, because a drifting or gameable judge quietly corrupts every downstream decision. The judge is itself a system you evaluate, not an oracle.

**29. What is reward hacking or eval gaming, and how do you guard against it?**
It is when a model optimizes the measured proxy rather than the real goal, for example exploiting a judge's biases or a benchmark's shortcuts. Guard against it with held-out and adversarial evals, multiple diverse judges or metrics, and by checking whether gains transfer to independent tasks rather than one benchmark. Be suspicious of a score that jumps without a plausible mechanism. The defense is triangulation: no single metric decides, and you verify improvements survive on data the system was not tuned against.

**30. A customer says the model is "getting worse." How do you investigate?**
First make it measurable: pin down which outputs, since when, and against what expectation, and reproduce the failure on specific inputs. Check what changed: a model or prompt version, the data or retrieval index, the input distribution (drift), or an integration upstream. Run the current system against a fixed evaluation set and compare to a past baseline to separate real regression from perception. Often the model is unchanged and the inputs drifted or an upstream source degraded, which is why a versioned eval set and monitoring are the tools that answer this quickly.

**31. How do you design an evaluation for a brand-new capability with no benchmark?**
Start from what success means as a checkable outcome, then build a small, high-quality labeled set that covers realistic and adversarial cases yourself, working with the customer's domain experts. Choose metrics that isolate the capability rather than convenient proxies, and validate any judge against human labels before scaling it. Treat the eval set as a versioned artifact you defend and grow as you discover new failure modes. A modest, honest, hand-built eval beats a large auto-generated one you cannot trust.

---

## 6. Deployment, cost, and reliability

**32. Compare hosted API versus self-hosted open-weight models for a regulated enterprise.**
A hosted API (a frontier model behind a vendor endpoint) gives you the strongest models, no infrastructure to run, and fast iteration, at the cost of sending data to a third party, per-token pricing, and dependence on their availability and retention terms. Self-hosting an open-weight model keeps data inside the customer's boundary and can be cheaper at high volume, but you own the GPUs, the ops, the scaling, and usually accept lower peak capability. For a regulated enterprise the deciding factors are data residency and compliance, the availability of a private or VPC deployment of the hosted model, volume economics, and the required capability. Many deployments land on a private VPC deployment of a hosted model, which keeps data in-boundary while retaining a frontier model.

**33. How do you reason about the cost, latency, and quality trade-off in a deployment?**
Treat it as a dial per step, not one global setting: a high-stakes or hard step may justify a slower, more expensive reasoning model, while a routine step should use a small fast one. Decide with evaluation on the customer's own data, because the right point is task-specific, and revisit it as models get cheaper. Use caching, shorter context, and batching to cut cost without cutting quality. Make the trade-off explicit to the customer as a business choice (this much accuracy costs this much latency and money) rather than hiding it.

**34. Design observability for a production LLM or agent system.**
Log the full interaction: input, retrieved context or tool calls, the prompt and model version, the output, latency, tokens, and cost, with a trace ID so you can reconstruct any single request. Track quality signals continuously (a rolling eval on sampled traffic, faithfulness and error rates), plus operational metrics (latency percentiles, error rates, throttling), and set alerts on the ones that matter. Dashboard the trends so drift and regressions show up before the customer complains. Add user feedback capture (thumbs, corrections, escalations) because it is the cheapest source of real failure cases.

**35. How do you roll out an AI system safely at a customer?**
Gate on an evaluation suite built around how the system breaks, start with a narrow scope and a human in the loop, and add monitoring and guardrails before widening. Expand only as the metrics hold, keep a fast rollback, and treat evals as release infrastructure rather than a one-time check. Roll out behind a fallback so a failure degrades to a safe default rather than breaking the workflow. The pattern is: small blast radius, human oversight, measured expansion, easy reversal.

**36. Debug a customer integration where a third-party API starts timing out intermittently.**
Reproduce and characterize first: is it a subset of requests, a time of day, a payload size, a region, and what does the latency distribution look like? Check the obvious external causes (rate limits and throttling, the provider's status page, DNS or network, auth token expiry) and add logging and a trace to see exactly where time is spent. Make the client resilient regardless of root cause: timeouts, retries with exponential backoff and jitter, a circuit breaker, and idempotency so retries are safe. Then work with the provider on the root cause while the resilience keeps the customer running.

**37. A data pipeline is producing duplicate records in the customer's warehouse. How do you isolate the cause?**
Quantify and localize first: how many duplicates, since when, and are they exact duplicates or near-duplicates, which points at different causes. Trace one duplicated record backward through each stage (source extract, transform, load) to find the stage where the count first doubles. Common causes are a non-idempotent load that reprocesses on retry, an at-least-once delivery source without dedup, a join fan-out, or overlapping backfill windows. Fix with an idempotent upsert on a stable key and dedup logic, then backfill-correct the existing duplicates.

**38. How do you version, A/B test, and roll back prompts in production?**
Treat prompts as versioned artifacts in source control with an identifier logged on every request, so you always know which prompt produced which output. Run a new prompt against your evaluation set before it goes near production, then A/B test it on a slice of live traffic while comparing quality, cost, and latency to the current version. Keep the previous version deployable so rollback is a config change, not a code deploy. The point is that a prompt change is a production change and deserves the same discipline as a code change.

---

## 7. Responsible AI and security

**39. How do you handle hallucination in a customer-facing product?**
Reduce it (grounding via RAG, constrained outputs, clear instructions to answer only from provided context), detect it (evaluation and monitoring, faithfulness checks, citation verification), and contain it (show sources, make correction easy, and route uncertain or high-stakes cases to a human). You manage the risk; you do not eliminate it, and you should say so plainly to the customer. Set the expectation that the system is a strong assistant with guardrails, not an infallible oracle, and design the workflow so a wrong answer is caught rather than acted on blindly.

**40. What are the main security concerns for an agentic system, and how do you mitigate them?**
Prompt injection (especially indirect injection from retrieved or tool content), over-broad tool permissions and privilege escalation, data exfiltration, and unsafe real-world actions taken with autonomy. Mitigate with least-privilege tools, input and output guardrails, human approval for sensitive or irreversible actions, and audit logging of every action. Separate trusted instructions from untrusted data, and constrain what any single tool can do so a compromise has a small blast radius. Test adversarially before shipping, because the attack surface is the content the agent reads, not just the user's prompt.

**41. How do you handle data privacy and compliance in a deployment?**
Keep sensitive data access least-privilege and permission-aware end to end, so retrieval and memory only ever surface what a user is allowed to see. Be deliberate about what leaves the customer's environment and what any vendor may retain, and prefer in-boundary or VPC deployment when residency or regulation demands it. Log for auditability, minimize what data you collect and keep, and design so you can honor deletion and access requests. Bring the customer's security and compliance teams in early, because their approval is often the real critical path.

**42. How do you talk to a customer about responsible AI and model limitations without either overselling or scaring them?**
Be concrete rather than adjectival: name the specific failure modes, roughly how often they occur on their data, and what each one would cost in their workflow. Frame quality as a distribution shown by evaluation, not a single impressive demo, and tie the risk to the specific decision the system drives. Pair every limitation with the mitigation and the human-oversight design that contains it. Customers trust honesty backed by evidence far more than confident promises, and it protects the relationship when something inevitably goes wrong.

---

## 8. Case and decomposition (judgment)

**43. "A logistics customer says their ops team cannot trust the dashboard. Figure out what is wrong and propose a plan." How do you approach it?**
I clarify the real goal first: what decision does ops make from this dashboard, and what would "trustworthy" mean to them, before assuming it is a data problem. I identify the stakeholders (ops users, the data team, whoever owns the sources) and map the inputs and their quality, naming what I do not yet know. Then I decompose the likely failure surface (stale or delayed data, wrong or ambiguous metric definitions, silent pipeline failures, a UI that hides caveats) and sequence to check the cheapest, most likely causes first. I would propose a thin first slice, for example one trusted metric with a freshness indicator and a clear definition, then expand, and I would name the risks I am accepting along the way. The interviewer is watching whether I scope before I solve and narrate throughout.

**44. Walk me through how you would scope an AI agent for automated shipment rerouting, given SAP data, weather APIs, and 500 warehouse managers.**
First the goal and success metric: what fraction of reroutes should be automated versus recommended, what is the cost of a wrong reroute, and what on-time target defines success. Stakeholders and "done": warehouse managers who must trust and sometimes override it, ops leadership who own the metric, and IT who own SAP access. Inputs and data quality: how fresh and reliable the SAP and weather feeds are, and what ground truth I have to evaluate reroute decisions. I decompose into detect-disruption, propose-reroute, and act, and I would sequence a recommendation-only MVP with a human approving each reroute, an evaluation harness measuring decision quality against outcomes, and full autonomy only where reliability (including worst-case, not just average) clears the bar. I surface the failure modes (bad reroute, feed outage, manager distrust) and design guardrails, human override, and monitoring around them.

**45. How do you decide, for an AI feature, between prompting, RAG, fine-tuning, and an agent?**
Match the tool to the need and start with the simplest option that clears the quality bar, because each step up adds cost and risk. Prompting and context for behavior you can specify, RAG for fresh or proprietary knowledge that needs citations, fine-tuning for stable behavior or style at scale (not for volatile facts), and an agent only when the task genuinely needs multiple steps and actions. Often the answer is a combination: RAG for knowledge plus light prompt or behavior tuning, with agency added only where required. I decide on the customer's own data with a quick evaluation rather than by preference, and I can defend the choice with the trade-offs at each step.

---

## 9. Customer communication and business

Short model responses for the role-play round. Delivery matters as much as content; keep them calm, specific, and ownership-forward.

**46. "Your deployment slipped 3 weeks. I am the CTO. Tell me."**
Lead with ownership and the facts, no burying: "I want to give you a straight update. The deployment will land 3 weeks later than we planned, and here is why and what I am doing about it." Name the specific cause (for example a data-access approval that took longer than expected), what is already unblocked, and the new date with the concrete steps to hit it. Offer a way to de-risk in the meantime, such as shipping a smaller working slice now so their team sees value while the full scope lands. Then invite their input on priorities, because they may want to trade scope for the original date.

**47. "Explain to me, a non-technical VP, why your RAG system cannot guarantee 100 percent accuracy."**
The system answers by finding the most relevant documents in your data and writing an answer from them, and both the finding and the writing are very good but not perfect, so a small fraction of answers will be wrong or incomplete. That is why we show the sources behind each answer, so your team can verify anything important in one click, and why we route uncertain or high-stakes cases to a person. The right way to think about it is a fast, well-supervised assistant that makes your experts far more productive, not an oracle we let run unchecked. We measure the error rate on your data and hold it to an agreed bar, and we design the workflow so a wrong answer is caught before it is acted on.

**48. "My security team will not give you production credentials. How do you unblock yourself?"**
That is a reasonable stance, and I would rather work within it than around it, so first I would ask their security team exactly what they need to be comfortable, because usually it is scope and auditability, not a flat no. In the meantime I can make progress without production access: work in a sandboxed or lower environment, use synthetic or de-identified data, and design against their access model so nothing is thrown away when approval comes. I would propose least-privilege, time-boxed, audited access scoped to only what the deployment needs, which is far easier to approve than broad credentials. And I would loop in their security team as partners early, because their sign-off is on the critical path and treating them as such speeds everything up.

**49. How do you push back on a customer's chosen architecture without damaging the relationship?**
Acknowledge what is right about their choice first, so it is clear I understand it rather than dismissing it. Then make the concern concrete and tied to their goal: "the risk with this approach is X, which would cost you Y in the outcome you care about," backed by evidence or a quick example rather than opinion. Offer an alternative with explicit trade-offs and let them own the decision, because my job is to give them the clearest possible picture, not to win. If they still choose their path and the risk is acceptable, I commit to it fully; if the risk is severe, I escalate it plainly and in writing.

**50. How do you spot a pattern across customers and turn it into product change?**
I keep notes across deployments on where customers hit the same friction, the same missing capability, or the same integration pain, and I look for the recurring shape rather than treating each as bespoke. When a pattern is clear, I write it up with the concrete customer evidence and the cost of not fixing it, and I route it to the product team as a prioritized, grounded request rather than an anecdote. Where I can, I build a reusable solution (a shared connector, a template, an internal tool) so the next FDE does not re-solve it. Being the channel from the field back to the core product is a core part of the job, not a side task.

---

Next: [resources.md](resources.md) to shore up any theme you fumbled, and [prep-plan.md](prep-plan.md) to sequence your practice.
