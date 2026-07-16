# Designing a Customer Support Agent

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real production data, and it ships with [runnable, provider-agnostic LangGraph code](code/) you can execute.

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that the core component is non-deterministic, so evaluation moves from a final gate to the center of the design.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/system-design/customer-support-agent (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs. When we get to the code, run `code/run.py` and explain what each node in the graph does."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine: 5 layers we use for this system design

A quick note before we get started. This is how we are going to look at the problem, and it is worth saying why up front. When you sit down to design an AI system, you need a structure to hang your reasoning on, or you jump straight to a vector database or a model name and skip the decisions that actually matter. These 5 layers are that structure. Most AI systems can be reasoned about as some combination of them, which is why we lead with the spine before touching the specific problem.

This is not the only way to do it, and it is not a universal template. It is a starting point. You decide which layers are load-bearing for the problem in front of you and go deep there, and as systems get more complex you will reach past this spine into more innovative approaches. For this case study, these 5 layers are our spine.

- **Layer 1, the model.** The model at the center. Which model, why, and how you handle its non-determinism.
- **Layer 2, the wrapping layer, or the architecture.** Knowledge (retrieval), tools, and memory composed around the model. This is where most of the design lives.
- **Layer 3, evals and guardrails.** How you know any of it is good, and how you stop the bad in real time. Offline sets that gate releases, and online checks that act as guardrails on live traffic.
- **Layer 4, production and ops.** The loop that makes it real: scale, latency, cost, reliability, observability.
- **Layer 5, optimization.** Where you make a working system better and take on harder problems: model routing, caching, advanced retrieval, and multi-agent when a single agent genuinely is not enough.

Composing these layers into one coherent system is what we mean by system design, and it is the whole of this writeup.

```
                          +-------------+
                          |  1  MODEL   |          the model, the brain
                          +------+------+
                                 |
      +------------- 2  WRAPPING LAYER (ARCHITECTURE) --------------+
      |    [ Knowledge / RAG ]    [ Tools ]    [ Memory ]          |   the architecture
      +---------------------------- + -----------------------------+   around the model
                                 |
                   +-------------+--------------+
                   |                            |
      3  EVALS & GUARDRAILS         4  PRODUCTION & OPS
      (is it good? stop the bad)   (keep it alive at scale)
                   |                            |
      +----------------------- 5  OPTIMIZATION --------------------------+
      |   make it better: routing, caching, advanced retrieval,          |
      |   multi-agent (only when a single agent is not enough)           |
      +-----------------------------------------------------------------+

      system design = composing all of the above into one coherent system
```

The rest of this case study walks these layers for a support agent, iterates the design, then takes the follow-ups an interviewer actually asks.

---

## The question

> "Design a customer support agent for an e-commerce company. It reads the help center and answers questions, can look up an order and open a support ticket through tools, and escalates to a human when it is unsure. Walk me through it."

---

## The answer

### Step 1: scope before you architect

The first move is not a vector database. It is clarifying the problem, because in AI the largest failures are designed in before a single prompt is written. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** Tier-1 support agents handle about 60 tickets a shift. Roughly 70% are repetitive (order status, return policy, password reset), 4 to 8 minutes each, with a 15-minute queue at peak. A human-handled ticket runs on the order of 5 to 8 dollars in fully loaded cost; an automated resolution is cents.
- **Outcome, written before the system.** Deflect the repetitive 70% to an agent that resolves them end to end, correct and grounded. Measured by containment rate, grounded-answer rate, and customer satisfaction on contained tickets, with a hard ceiling on incorrect actions.
- **The AI intervention, narrowed until it hurts.** Retrieve and answer over the help center, take a few safe actions through tools, escalate everything else, staying well short of an autonomous support platform.
- **System and safety.** An evaluation set that gates every release, input and output guardrails, [prompt-injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) defense, human escalation as the default when unsure, full tracing, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): what is the accuracy bar and the cost of a wrong action; what latency is acceptable; which actions may the agent take autonomously; what is the escalation policy; how fresh must the help center be; are there compliance constraints. This also avoids the traps that sink these projects: leading with "build a RAG agent" (solutioning in the problem statement), packing refunds and fraud and onboarding into one system (over-scoping), and designing without a measurable owner.

> **Real outlier: Klarna, February 2024.** Klarna's OpenAI-powered assistant did the work of about 700 full-time agents, handled 2.3 million conversations in its first month (about two-thirds of its chat volume), cut resolution time from 11 minutes to under 2, ran in 35+ languages across 23 markets, dropped repeat inquiries 25%, and was estimated at 40 million dollars of profit improvement. **Then the cautionary half:** by 2025 the CEO said the automation had "gone too far" and Klarna began rehiring humans so a customer could always reach a person. That is the whole design in one story: enormous deflection is real, and a human path is not optional. It is Problem-First layer 2 (outcome) and layer 4 (safety) at industry scale. [[Klarna press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[the walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data, because every method in the wrapping layer below is chosen for a specific shape of data. Change an assumption and the method changes with it. For this case study, assume:

- **Corpus: small and semi-structured.** A few hundred to a few thousand help-center articles in HTML and markdown, mostly prose with some tables (shipping fees, return windows) and step lists, updated occasionally (policies change monthly), not by the second. *This is why* a single vector index is plenty and freshness means re-indexing on edit rather than streaming.
- **Questions: short, and often carrying exact tokens.** One or two sentences. Many contain order ids, SKUs, tracking numbers, or error codes; many are paraphrases of the same intent ("can I send it back" means the return policy). *This is why* retrieval here is hybrid: dense for meaning and paraphrase, sparse for the exact ids.
- **Answers: usually one article.** A typical question is resolved by a single article or section; few need several documents stitched together. *This is why* single-shot retrieval is enough and multi-hop methods are not needed yet.
- **Volume and latency: chat-like.** Thousands of conversations a day, a response expected in a few seconds, and questions that repeat heavily. *This is why* there is a latency budget the reranker must fit inside, and why caching pays off.
- **Language: one to start.** Primarily English at launch. *This is why* a single embedder is fine now; multilingual is a later extension (Follow-up 6).
- **Actions and stakes.** A few safe actions (order lookup, ticket) and one high-impact one (refund), where a wrong action has real cost. *This is why* there are few typed tools, the refund is gated, and the relevance floor is conservative.

Keep these in view as you read the wrapping layer: each choice below points back to one of them. If your corpus is millions of documents, your questions are multi-hop, or your content is mostly tables, revisit the assumption and pick a different method.

### Step 2: walk the layers for this system

**Layer 1, the model.** The answer here is a routing strategy: cheap fast models handle the repetitive 70%, a stronger model handles the ambiguous cases, and a reasoning model is reserved for the rare hard escalation triage. The model is non-deterministic, so the same question can produce different phrasings. You handle that with structured output, grounding, and evaluation.

**Layer 2, the wrapping layer (the architecture).** This is where most of the design lives, so go deep here. It is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. In a support agent this layer is the product.

**Knowledge (retrieval).** The agent must ground every answer in your help center, or it invents policy. That is a retrieval pipeline, and every stage is a decision whose right answer depends on your data. Each choice below points back to the [assumptions](#the-assumptions-we-make-about-the-data-and-the-use-case) above. Here is the pipeline, then each stage: why it matters and what data tells you how to set it.

```
 help-center docs --1.PARSE/INGEST--> [structured text + metadata] --2.CHUNK--> [chunks]
                                                                                    |
                                                                              3.EMBED
                                                                                    v
 question --5.RETRIEVE (hybrid: dense + BM25)--> candidates --6.RERANK--> top few --> 7.RELEVANCE FLOOR
     ^                                                                                      |
     |                                                              below floor? retrieve nothing --> ESCALATE
     +------------------------------------ 4. VECTOR STORE / INDEX --------------------------+
```

**1. Parse and ingest.** Help-center content is HTML and markdown with real structure: headings, tables, ordered steps. Naive text extraction flattens that and the retriever loses the scent. Parse structure-aware, and for any PDFs or complex documents use a layout-aware parser ([Docling](https://github.com/docling-project/docling), [LlamaParse](https://github.com/run-llama/llama_cloud_services), [Reducto](https://reducto.ai/)) rather than a raw text dump. Attach metadata to every chunk (article id, section, `updated_at`, locale, product), because freshness, filtering, and access control all ride on it. *Given our assumption* that the corpus is mostly prose with a few tables, a structure-aware markdown parse covers most of it, and you reach for a layout parser only for the shipping and fee tables. *What data decides it:* audit your messiest documents (nested tables, multi-column layouts) and measure how often parsing mangles them.

**2. Chunk.** The chunk is the unit of retrieval, so its size is a real lever. Too big and the answer is buried among irrelevant text that also burns context budget ([Lost in the Middle](https://arxiv.org/abs/2307.03172)); too small and a sentence is severed from the context that gives it meaning. *Because we assumed one article usually answers a question,* chunk by structure (one section or heading per chunk) rather than by a fixed character count. The current high-leverage move is [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval): prepend a short model-generated blurb situating each chunk in its document before you embed it, which Anthropic reported cut retrieval failures by about 35%, and about 49% combined with reranking. *What data decides it:* your question and answer length distribution, and a chunk-size sweep scored on a retrieval-quality measure such as [recall@k](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29) (does the right chunk show up in the top results) against a labeled set. Pick the size where that score stops improving.

**3. Embed.** An [embedding](https://arxiv.org/abs/2004.04906) is a list of numbers that captures a piece of text's meaning, so two texts that mean similar things land close together in that number space. The embedding model you pick defines what "similar" means, so a weak or off-domain embedder caps everything downstream no matter how good your reranker is. Choose from the [MTEB retrieval leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (a public ranking of embedding models on retrieval tasks), favoring one trained near your domain and your languages; [Matryoshka embeddings](https://arxiv.org/abs/2205.13147) are a newer style that lets you shorten the number list later to trade a little accuracy for cheaper, faster search. *Because we assumed English at launch,* a single strong English embedder is enough now, and a multilingual one is the swap when Follow-up 6 arrives. *How you would choose:* run 2 or 3 candidate embedders on your own labeled question-to-document pairs and compare a retrieval-quality measure (recall@k is a common one), weighed against embedding latency, cost per million tokens, and index memory at your corpus size.

**4. Store and index.** The chunks and their embeddings live in a vector store, and the *index* is the structure inside it that makes similarity search fast enough to run on every question. The index choice is a trade-off between speed, memory, and accuracy. [HNSW](https://arxiv.org/abs/1603.09320) builds a navigable graph through the vectors and gives fast, high-recall search at the cost of more memory; [IVF-PQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) compresses the vectors into buckets, using less memory but needing more tuning to stay accurate. Managed stores (Pinecone, Weaviate, Qdrant, Turbopuffer) run this for you; [pgvector](https://github.com/pgvector/pgvector) adds vector search to Postgres and is often enough if you already run one. *Our assumption of a few thousand articles* puts us in the simplest regime, where one HNSW index or pgvector is plenty; billions of vectors would be a different world. *How you would choose:* corpus size, how many queries per second you expect, your accuracy target, and whether you need metadata filters (search within one product or locale) and per-tenant isolation (one customer's data never surfacing in another's search).

**5. Retrieve.** Retrieval is the step that, given a question, pulls the handful of chunks most likely to hold the answer. There are two ways to match a question to a chunk, and they fail in opposite places, which is why this stage matters most for our assumptions.

- **[Dense retrieval](https://arxiv.org/abs/2004.04906)** compares *meaning*. You turn the question and every chunk into embeddings (the number-lists from the last step) and pull the chunks whose numbers sit closest to the question's. This is what lets "can I send it back" find the return-policy article even though they share no words, so it shines on the paraphrases we assumed. Its weak spot is exact strings: an order id like `4021-XZ` or an error code carries no meaning to embed, so dense retrieval often slides right past it.
- **[Sparse retrieval](https://www.pinecone.io/learn/hybrid-search-intro/)** compares *words*. [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is the long-standing method here: a keyword-scoring formula that ranks a chunk by how many of the query's words it contains, weighting rarer words more heavily. It catches the exact order ids, SKUs, and error codes we assumed show up often, and needs no model to run. Its weak spot is the mirror image: it cannot tell that "send it back" and "return" mean the same thing.

Because our questions carry both meaning and exact ids, you want both methods, which is **hybrid retrieval**: run dense and sparse together and merge their two ranked lists into one. The common way to merge is [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), a simple rule that rewards a chunk for ranking highly in *either* list, so results both methods agree on rise to the top. To get started, run a BM25 index and a dense index side by side and fuse them with Reciprocal Rank Fusion, which most vector databases support directly.

When a question is too short or vague to retrieve well, three techniques rewrite it before searching, and you can add them one at a time as needed:
- **Query rewriting**: have the model turn a fragment into a fuller question ("broken" becomes "my item arrived damaged, what are my options") that retrieves better. Useful when users type terse fragments. [[explainer](https://arxiv.org/abs/2305.14283)]
- **Multi-query**: have the model produce several phrasings of the same question, retrieve for each, and combine the results, so one awkward wording does not sink the search. [[explainer](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)]
- **HyDE** (Hypothetical Document Embeddings): have the model draft a rough, made-up answer to the question, then retrieve the real chunks closest to that draft, on the idea that an answer resembles the target document more than the bare question does. You can use HyDE when questions and answers tend to be worded very differently. [[explainer](https://arxiv.org/abs/2212.10496)]

*How you would evaluate it:* bucket a sample of real questions by type and see where dense-only retrieval misses. If exact-id lookups are a meaningful share of those misses, the sparse arm is not optional. Watch a retrieval-quality measure (recall@k is a common one) as you tune, where the exact metric matters less than tracking the same one over time.

**6. Rerank.** First-stage retrieval is tuned to cast a wide net: get the right chunk somewhere in the top ~50, even if it is not ranked first. But the model only reads the few chunks you actually hand it, so the best ones need to be at the very top. A **reranker** does exactly that: it takes those ~50 candidates and reorders them so the most relevant land in the top few you pass to the model.

The reason a reranker beats the first-stage order comes down to *how* it reads. Dense retrieval embeds the question and each chunk **separately** and then compares the two number-lists, which is fast, because every chunk's embedding can be computed ahead of time, but coarse. A [**cross-encoder**](https://www.sbert.net/examples/applications/cross-encoder/README.html) reranker instead feeds the question and one chunk into a model **together**, so it can weigh how well that specific chunk answers that specific question and catch nuances the separate embeddings miss. It is slower, because it runs once per candidate at query time with nothing pre-computed, which is why you run it on ~50 candidates rather than the whole corpus. Off-the-shelf cross-encoders (Cohere Rerank, Voyage, the open bge-reranker) drop in with a single call, so getting started is mostly wiring rather than training.

*How you would evaluate it:* compare retrieval quality with and without the reranker on your own labeled set, and weigh any lift against the latency it adds at query time. Given our few-second budget, keep the reranker only if the improvement is worth the wait.

**7. Relevance floor and abstention.** This is the single most important safety control in the whole system, because it converts "I could not find this" into an escalation instead of a confident hallucination. Set a threshold on the reranker score, or add a lightweight grader that judges whether the retrieved context is sufficient (the idea behind [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): retrieve, grade, then answer, re-retrieve, or abstain). *Because we assumed a wrong action is costly,* set the floor conservatively so the agent escalates rather than guesses. *What data decides it:* sweep the threshold and plot escalation precision against answer coverage on a labeled should-answer versus should-escalate set, then pick the point that keeps incorrect answers under the ceiling you set in scoping.

**When to reach further.** [GraphRAG](https://arxiv.org/abs/2404.16130) for questions that hop across several documents, and agentic retrieval (the agent issues its own follow-up searches) when one query cannot get there. *We assumed a single article usually answers,* so single-shot hybrid retrieval is enough here; drop that assumption, with multi-hop questions or a product knowledge graph, and GraphRAG earns its place. The repo's [RAG topic page](../../topics/rag.md) collects the primary sources for each of these stages.

> **Real finding: [Lost in the Middle](https://arxiv.org/abs/2307.03172).** Accuracy drops when the relevant passage sits in the middle of a long context rather than at the ends. More retrieved text is not a free win, which is why you rerank and cap k instead of stuffing the window.

**Tools (actions).** Tools are the agent's hands. Each one is a typed, allowlisted contract rather than open-ended code execution.

```
  order_lookup(order_id) -> {status, eta}         READ   low blast radius, cheap to trust
  create_ticket(summary, category) -> ticket_id   WRITE  idempotent, logged
  issue_refund(order_id, amount)                  WRITE  high-impact -> human approval (Follow-up 2)
```

The calls that matter: the **tool description is a prompt** (the model picks a tool from its name and doc, so vague descriptions cause wrong calls); **least privilege** (read tools are cheap to trust, write tools are gated); **[idempotency](https://en.wikipedia.org/wiki/Idempotence)** (a retried `create_ticket` must not open two tickets); **error handling** (a failed tool call is caught and retried or escalated, never hallucinated over); and the **loop is bounded** (a hard step cap stops the agent from calling tools forever on an ambiguous result). This is the [ReAct](https://arxiv.org/abs/2210.03629) pattern: reason, act, observe, repeat, until an answer or an escalation.

**Memory.** Memory is what makes this a conversation instead of a series of unrelated questions, and it comes in layers.

```
  SHORT-TERM (this session)  : the running conversation, so "what about express?" resolves against the last turn
  WORKING   (this task)      : the retrieved context and tool results the agent is currently reasoning over
  LONG-TERM (this customer)  : past orders and tickets, RETRIEVED on demand, never stuffed wholesale into the prompt
```

The calls that matter: **what to keep and what to drop** (the context window is finite, so old turns are summarized rather than carried verbatim); **retrieve long-term memory, do not dump it** (pull the 2 relevant past tickets rather than the entire history); and **treat memory as untrusted** (anything persisted from a user or a document can smuggle an injected instruction into a later turn, the memory arm of the lethal trifecta in Follow-up 2).

Together, knowledge, tools, and memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for an agent this is the center of the design, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, blocking an ungrounded or unsafe answer and firing the human handoff the moment something is off. There is no single "accuracy" number, and no generic metric you can copy off a shelf, because "good" means something different for every product. So start where we teach you to start: **from failure modes.** Ask what could go wrong that would be unacceptable for this business (a wrong refund, a hallucinated policy, a missed escalation), then translate each into an observable, measurable behavior. The metrics below are a menu you draw from once you know your failure modes, and the target is the minimum set that gives the most signal for your product.

Evaluate at three levels: **each component** (did retrieval find the right doc, did the model ground its answer, did it call the right tool), **the whole task** end to end (was the ticket actually resolved), and **live traffic** (is it still good in production).

> **Real finding: [tau-bench](https://arxiv.org/abs/2406.12045).** On a tool-using agent, a single-attempt pass rate looks respectable, but pass^k (succeed on all k independent tries) collapses as k grows. A support agent that works 1 time in 1 and fails 1 time in 4 is not shippable. Reliability is the bar, above average accuracy.

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: did it call `order_lookup` with the right id, is the output valid, does the answer cite a real source, did it refuse. Fast, reliable, cheap. Use wherever "good" is objectively checkable, and compare against a reference dataset (golden answers) here.
- **LLM judges.** One model scoring another against an explicit rubric, for subjective qualities (faithfulness, tone, escalation appropriateness) that code cannot capture. Scalable, and a new source of non-determinism, so it must be calibrated before you trust it (below).
- **Human evaluation.** The gold standard you calibrate the other two against. Too slow and costly to run on all traffic, so you sample: calibration, edge cases, high-stakes interactions.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of 30 to 50 labeled examples across answerable, partial, unanswerable, action, and adversarial cases. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Retrieval** | misses the right article, returns junk | recall@k, precision@k, [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank), [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain), context relevance | code-based against labeled query to doc pairs |
| **Answer generation** | hallucinates, off-topic, incomplete, wrong citation | [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) / groundedness, answer relevance, correctness vs golden, completeness, citation accuracy, hallucination rate | LLM judge with rubric + golden answers |
| **Tool use** | wrong tool, wrong arguments, needless call | tool-selection accuracy, argument exact-match, tool-call success rate, unnecessary-call rate | code-based against the expected trace |
| **Safety** | executes injected instructions, leaks PII, takes an unsafe action | injection-resistance rate, PII-leak rate, unsafe-action rate, jailbreak pass rate | adversarial red-team suite |
| **Escalation** | deflects when it should hand off, or escalates the easy ones | escalation precision, escalation recall, false-deflection rate | LLM judge / code-based on a labeled should-escalate set |
| **End to end** | task not actually resolved | task success / resolution rate, pass@1, **pass^k**, turns-to-resolution | scenario suite + judge |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: these are the metrics teams typically reach for, and your job is to choose wisely and instrument only the few that earn their place. A dashboard with 30 metrics and no owner tells you nothing. Pick the two or three per component that map to your real failure modes and drop the rest.

Report **pass^k** alongside the average, because a support agent that succeeds 1 try in 1 yet fails 1 in 4 is not shippable (the tau-bench finding above). For this product, faithfulness is the highest-signal single metric: it measures hallucination directly, the failure that erodes trust fastest.

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem, or is it merely interesting?
- **Reliability:** human evaluation and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check is cheap, a fast judge call is medium, detailed human review is expensive.

High impact and low cost are the must-haves (safety filters, structure checks, escalation flags). High impact and high cost are strategic investments you run on a sample (a calibrated faithfulness judge). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the business, run inline so the system can act the moment they trip: an unsafe action, an ungrounded answer, low confidence. The action is immediate (escalate to a human, block the response). Guardrails must be fast and reliable before sophisticated.
- **Improvement flywheel (offline, batch).** Everything else: quality trends, faithfulness on a sample, CSAT, drift. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live traffic:

| Metric | Job | Why it matters |
|---|---|---|
| unsafe / incorrect-action rate | guardrail | the hard ceiling from scoping; must stay near zero |
| groundedness on a live sample | guardrail + flywheel | catches hallucination drift before users report it |
| low-confidence / uncertainty trigger | guardrail | fires the human handoff in real time |
| containment / deflection rate | flywheel | share of tickets resolved without a human, the core outcome |
| CSAT / thumbs-up rate | flywheel | quality as the customer actually feels it |
| escalation rate | flywheel | too high means it is not helping; too low means risky over-deflection (the Klarna walk-back) |
| p50 / p95 latency | flywheel | user experience, budgeted per stage |
| cost per resolved ticket, tokens per request | flywheel | unit economics, the number finance asks about |
| tool error rate | flywheel | health of the order and ticket dependencies |
| retrieval hit rate on live queries | flywheel | surfaces help-center coverage gaps |

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label a few hundred examples, run the judge on the same set, measure agreement (percent agreement or Cohen's kappa), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the underlying model, and run the judge on a sampled percentage of live traffic to control cost.

Then run the discovery loop, because users will always find failures your metrics were never built for. Sample live traffic on **signals** (thumbs-down, retries, rephrasing, explicit escalation requests, abandonment). When a signal keeps firing but your metrics look clean, that gap is the tell: a human reads those traces, names the quality dimension you were not measuring, and it becomes a new metric added back into the reference dataset. Evaluation is never finished. You build for the failures you can anticipate, and you monitor to discover the ones you cannot.

```
   user signals (thumbs-down, retries, escalations, abandonment)
        |
   sample + read traces --> metrics miss it? --> name new dimension
        ^                                              |
        |                                       new metric into
   reference dataset <--- offline gate (CI) <---  reference set
        |
   pass gate --> canary --> deploy --> guardrails + flywheel on live traffic ---+
        ^                                                                        |
        +------------------------ signals feed back --------------------------- +
```

Instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. Reading traces is how you find the exact step where the agent's judgment diverged from yours. Tooling worth knowing: Ragas and DeepEval for component metrics, promptfoo for CI gates, Arize Phoenix for tracing and online evals. *Deeper:* [AI Evals for Everyone](../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../topics/evaluation.md).

**Layer 4, production and ops.** The loop that keeps it alive at scale: vector search with freshness and access control, reliability, latency budgets, and observability so every step is traceable. Detail is in Follow-ups 1 and 3.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: model routing (a cheap model for the easy majority, a strong model for the rest), prompt and semantic caching, advanced retrieval, and multi-agent. For this system the optimizations that pay off are routing and caching; multi-agent is not needed, and Follow-up 5 shows why. As cases get more complex, this is where the more innovative approaches enter.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (fixed retrieve-then-answer, refuse when nothing clears the relevance floor), prove it with evals, and only hand the model more freedom (tools, a bounded loop, memory) once the measurement says the constrained version is solid and its limits are what block you. You earn agency through evaluation before you grant it. Every increase in autonomy is paid for with an eval that shows it helped.

Composed, the layers give one architecture:

```
   +---------------------- OBSERVABILITY: every node, tool, and model call is a span (Arize) ----------------------+
   |                                                                                                              |
 user --> INPUT GUARDRAIL --> RETRIEVE ---------> AGENT LOOP (bounded, ReAct) <------> TOOLS (allowlisted, typed)
          (injection, PII)    hybrid + rerank            |            ^                order_lookup / create_ticket
                              + relevance floor          |            |                / issue_refund*
                                                         v            |
                                              MEMORY (session + retrieved customer context)
                                                         |
                                                         v
                                    OUTPUT GUARDRAIL --> grounded & safe? --yes--> ANSWER + cited source
                                                                    \--no / low confidence / high-impact-->
                                                                                        ESCALATE --> human
                                              (* issue_refund requires human approval, Follow-up 2)
```

Read it as the spine composed: the model (layer 1), wrapped in retrieval, tools, and memory (layer 2), gated by evals (layer 3) that run inline as guardrails and offline as a flywheel, run under production observability (layer 4), with optimization (layer 5) held to what this system actually needs, which is model routing and caching rather than a second agent. Composing exactly these pieces into one coherent system is the system design. The relevance floor and the output guardrail are what make escalation the safe default: when the agent cannot ground an answer or the action is high-impact, it hands off to a human with full context rather than guessing.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this architecture (minus the scale pieces): retrieval with a relevance floor, a bounded agent loop with 2 tools, a grounding-and-injection guardrail, and escalation, where high-impact requests like refunds escalate for human approval. It runs offline with a deterministic policy, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                              # run the scenarios (also a self-test)
python run.py "Where is my order 5012?"    # ask the agent your own question
```

Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route decisions through a real model. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now handle 10x the traffic. Where does it break, and how do you scale it?"

Name where it breaks first, then scale that, cheapest lever first.

- **Retrieval (layer 2).** The keyword search becomes a vector store, or hybrid dense plus BM25, with a reranker. The index type is a real tradeoff: HNSW gives fast, high-recall search at higher memory; IVF is lighter but needs tuning. An ingestion pipeline keeps the index fresh as articles change, and retrieval respects per-user and per-tenant access control so a customer never sees another account's data. At scale, the data and retrieval layer is usually the bottleneck, ahead of the model.
- **Routing and caching (layer 5, optimization).** Route easy questions to a small fast model; reserve a larger one for hard cases; cache the stable prompt prefix; add a semantic cache for near-duplicate questions.
- **Infrastructure (layer 4, production and ops).** Horizontal scale behind a queue, connection pooling to tool services, backpressure so a spike degrades gracefully instead of failing.

```
 clients --> gateway --> [input guardrails] --> agent service (LangGraph, N replicas)
                                                    |
        +-------------------+---------------+-------+-------------+
   vector store        tool services     model router       memory store
   + reranker          (orders,tickets)  (small <-> large)  (conversation)
        |                                                        |
   ingestion / freshness                                   Arize (traces,
   + access control                                        online evals, alerts)
```

Put numbers on it: tokens per request, an approximate cost per resolved ticket, and a latency budget split across stages. *Deeper:* [Production and LLMOps](../../topics/production.md).

---

## Follow-up 2: "A customer tries to jailbreak it into issuing a refund it should not. How do you prevent wrong or unsafe actions?"

Separate answering from acting, and gate the acting.

- **Action guardrails.** High-impact tools (refunds, cancellations) require human approval or a hard policy check, never blanket autonomy. Allowlists over blocklists.
- **Untrusted input.** Retrieved documents and user messages are untrusted; the agent never executes instructions found in them. The sharp way to see the risk is the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
     untrusted content   +   access to private data   +   ability to act/communicate
              \                       |                            /
               \______________ any 2 are manageable ____________/
                          all 3 together is dangerous
```

- **Blast radius.** Least-privilege tool scopes, an immutable audit log of every action, and a design where the worst a jailbreak achieves is a needless escalation, while a wrongful refund stays impossible.

*Deeper:* [Securing Agentic AI Systems](../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../topics/safety-security.md).

---

## Follow-up 3: "Cut p95 latency in half without tanking quality."

Budget latency per stage (retrieval, model, tools) against a target such as p95 under 3 seconds, then attack the largest stage.

- Cache the stable prompt prefix, and add a semantic cache for repeated questions.
- Route to a smaller model for easy questions and sub-steps; reserve the large model for the hard path.
- Stream the answer so time-to-first-token stays low even when total time does not.
- Parallelize independent tool calls; trim tool outputs so the model reads fewer tokens.

> **Real number: [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).** Caching the stable prefix of a prompt (system instructions, tool schemas, long context) can cut cost by up to about 90% and latency by up to about 85% on the cached portion, per provider documentation. On a support agent whose system prompt and tool schemas are identical on every call, this is the single highest-leverage optimization.

Quality holds because the eval set from Layer 3 gates each change: if a faster route drops grounded-answer rate, it does not ship.

---

## Follow-up 4: "New compliance rule: every refund needs an audit trail. How does the design change?"

That single constraint promotes 2 normally-optional components into load-bearing walls: **audit logging** and **human approval** on the refund action. Add an immutable, queryable log of every refund decision (who, when, on what evidence) and route refunds through a human approval step. Nothing else in the architecture changes. This is the general pattern: a domain constraint is what forces you to engineer a layer you would otherwise accept as a framework default.

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

For this support flow, a single agent with tools is the right call: the conversation is one coherent thread, and a second agent would add coordination cost while the capability stays the same. As the scope grows, multi-agent (layer 5, optimization) becomes the natural next step, so it is worth knowing exactly when it earns its place and how you would extend this design to get there.

**When multi-agent earns its place:**
- **The work decomposes into independent sub-tasks that can run in parallel.** The quality and latency gains then outweigh the extra tokens. Research-style tasks that fan out across many sources are the classic fit.
- **Distinct sub-domains need distinct context and tools.** Billing, returns, technical troubleshooting, and account security each want their own knowledge, tools, and policies. Holding all of them in one agent bloats the context and blurs its behavior, while specialists keep each context tight and focused.
- **The context window is the bottleneck.** As the surface area grows, one generalist runs out of room, while specialists each stay well within budget.
- **The task is valuable enough to pay for it.** Multi-agent trades tokens for capability, which fits high-value work.

**How you would extend this architecture.** Keep the single-agent design intact and add an orchestrator (a supervisor that classifies intent and delegates) in front of specialist sub-agents. Each specialist owns its own tools, retrieval scope, and memory, and the same input and output guardrails, evals, escalation, and observability now wrap the whole system across agents.

```
   +------------------------ OBSERVABILITY: spans across every agent -------------------------+
   |                                                                                          |
 user --> INPUT GUARDRAIL --> ORCHESTRATOR (classify intent, delegate, then compose the reply)
                                        |
          +-----------------+----------+-----------+---------------------+
          v                 v                      v                     v
     RETURNS agent     BILLING agent         TECH-SUPPORT agent     ACCOUNT agent
     returns KB        billing KB             troubleshooting KB     auth-gated tools
     + return tools    + refund*              + device tools         + identity checks
          |                 |                      |                     |
          +-----------------+----------+-----------+---------------------+
                                       v
                          OUTPUT GUARDRAIL --> grounded & safe? --yes--> ANSWER
                                                        \--no / high-impact--> ESCALATE --> human
                              (* refund still requires human approval, Follow-up 2)
```

> **Real data: Anthropic's multi-agent research system.** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens, and token usage explained roughly 80% of the performance gap. The takeaway is to spend those tokens where task value and genuine parallelism justify them, and to stay single-agent where they do not. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]

The honest rule: start with the single agent, instrument it, and let the traces tell you when a sub-domain or a parallel workload has outgrown one agent. Multi-agent is where this design goes once the problem is big enough to need it.

---

## Follow-up 6: "Make it multilingual."

The architecture does not change; the data and the evaluation do. Retrieval needs per-language coverage of the help center or cross-lingual embeddings, and the eval set needs labeled questions per language so quality is measured per language rather than assumed. Escalation and guardrails apply unchanged. Klarna's assistant ran in 35+ languages on the same architecture. This is the recurring lesson: most "make it do X" follow-ups are answered in the data and eval layers, and the box diagram stays the same.

---

## Follow-up 7: "A better model drops next quarter. How do you avoid a rewrite?"

Build the harness on-policy and keep it a layer rather than a fork. Keep the model behind the provider-agnostic interface (the runnable code already does this, any model via one call), pin behavior with the eval set rather than with brittle prompt hacks, and pair every "do not do X" rule with an eval so you can delete the rule when a better model makes it obsolete. When the new model lands, you swap it, re-run the eval gate, and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved.

---

## Real-world reference points

- **Klarna (2024 to 2025):** 700 agents' worth of work, 2.3M chats in month 1 (about two-thirds of volume), 11 min to under 2 min, 35+ languages, 25% fewer repeat inquiries, ~40M dollars profit, then a public walk-back toward keeping humans reachable. Deflection is real; the human path is not optional. [[press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]
- **Anthropic multi-agent:** +90.2% on research, ~15x tokens, token use explains ~80% of variance; only worth it when task value beats token cost. [[blog](https://www.anthropic.com/engineering/multi-agent-research-system)]
- **Prompt caching:** up to ~90% cost and ~85% latency reduction on the cached portion; the highest-leverage support-agent optimization given identical system prompts.
- **tau-bench:** pass^k collapses as k grows; reliability is the shippable bar, above average accuracy.
- **Lost in the Middle:** long context buries relevant passages; retrieval needs a relevance floor and a reranker.

---

## Research to know

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis 2020): the grounding pattern the whole system rests on.
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao 2022): reason and act in a loop, the shape of the agent.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu 2023): why more context is not free.
- [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): retrieval that decides when to retrieve, re-retrieve, or abstain.
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495): check the draft before returning it, to cut hallucination.
- [tau-bench](https://arxiv.org/abs/2406.12045): evaluating tool-using agents on multi-turn tasks with a reliability metric.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../topics/rag.md), [Agents](../../topics/agents.md), [Evaluation](../../topics/evaluation.md), [Production and LLMOps](../../topics/production.md), [Safety and Security](../../topics/safety-security.md)); [AI Evals for Everyone](../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../resources/harness_engineering.md); Aishwarya's YouTube ([AI Engineering: A Realistic Roadmap for Beginners](https://www.youtube.com/watch?v=pAXbl1EBHJ8), [Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k), the [CC/CD talk](https://www.youtube.com/watch?v=z7T1pCxgvlA), the [full channel](https://www.youtube.com/@aishwaryanr4606)); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- Anthropic, [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and [Multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system).
- OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- Martin Fowler / Thoughtworks, [Emerging Patterns in Building GenAI Products](https://martinfowler.com/articles/gen-ai-patterns/).
- [Arize observability docs](https://arize.com/docs/) and [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/concepts/), the tracing loop and state-machine primitives the [code](code/) uses.

---

## Related in this repo

Topics: [RAG](../../topics/rag.md) · [Agents](../../topics/agents.md) · [Evaluation](../../topics/evaluation.md) · [Production and LLMOps](../../topics/production.md) · [Safety and Security](../../topics/safety-security.md). Guides: [Harness Engineering](../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../resources/securing_agentic_ai_systems.md). Prepping to be asked this? See the [interview prep hub](../../interview_prep/README.md).
