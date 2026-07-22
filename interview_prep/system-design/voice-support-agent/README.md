# Designing a Real-Time Voice Customer Support Agent

## The interview question

> "Design a real-time voice customer support agent for a phone line. A caller phones in, the agent answers in natural speech, grounds its answers in the help center, can look up an order through a tool, and hands off to a human when it is unsure. Callers interrupt, and the audio is streamed both ways. Walk me through it, and be specific about latency and turn-taking."

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real-world benchmarks and deployment data, and it ships with [runnable, provider-agnostic LangGraph code](code/) you can execute.

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that the core component is non-deterministic, so evaluation moves from a final gate to the center of the design. On a voice agent one more thing is new: the wall clock is part of the product, because a caller hears every millisecond of thinking as silence on the line.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/voice-support-agent (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs, going deepest on the real-time latency budget and turn-taking. When we get to the code, run `code/run.py` and explain what each node in the graph does, including the barge-in re-plan."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine

This case is built on the same 5-layer spine as the rest of the collection: **1 the model**, **2 the wrapping layer** (the architecture: knowledge, tools, and memory), **3 evals and guardrails**, **4 production and ops**, and **5 optimization** (where multi-agent lives). System design is the work of composing them into one coherent system. For how we use this spine and why it is a starting point rather than a rulebook, see [the spine](../README.md#the-spine-how-we-think-about-ai-system-design).

The rest of this case study walks these layers for this problem, going deepest where it is hardest.

---

## The answer

### Step 1: scope before you architect

The first move is clarifying the problem, before you reach for a vector database or the OpenAI Realtime API, because in AI the largest failures are designed in before a single prompt is written. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at LevelUp Labs: work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** Callers wait on hold, punch through touch-tone menus, and repeat themselves to each agent. Most calls are the same routine handful (order status, return policy, store hours, password reset), each a few minutes long, with a queue at peak. A human-handled call runs on the order of several dollars in fully loaded cost; an automated one is cents. The pain that is specific to voice is time: a caller cannot skim, so every second of silence while the system thinks feels like the line went dead.
- **Outcome, written before the system.** Contain the routine calls end to end in natural spoken conversation, correct and grounded, fast enough to feel human, with a hard ceiling on wrong actions and a clean handoff to a person for everything else. Measured by containment rate, grounded-answer rate, time-to-first-audio, and caller satisfaction on contained calls.
- **The AI intervention, narrowed until it hurts.** Answer over the help center, take a few safe actions through tools, confirm anything high-impact out loud, and hand off everything else, staying well short of an autonomous phone agent that can move money on its own.
- **System and safety.** An evaluation set that gates every release and covers latency and speech quality alongside correctness, real-time guardrails, [prompt-injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) defense, human handoff as the default when unsure, full call tracing, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): what is the latency bar in plain terms (how long may the line be silent before it reads as broken); how does the audio arrive (a telephony provider over [SIP](https://en.wikipedia.org/wiki/Voice_over_IP), or [WebRTC](https://en.wikipedia.org/wiki/WebRTC) from an app); what is the accuracy bar and the cost of a wrong action; which actions may the agent take on its own; what is the handoff policy; are calls recorded, and what are the consent and privacy rules. This also avoids the traps that sink these projects: leading with "use the Realtime API" (solutioning in the problem statement), packing refunds and fraud and account changes into one agent (over-scoping), and designing without a measurable owner for containment and latency.

> **Real outlier: Klarna, February 2024.** Klarna's OpenAI-powered assistant did the work of about 700 full-time agents, handled 2.3 million conversations in its first month (about two-thirds of its chat volume), cut resolution time from 11 minutes to under 2, ran in 35+ languages across 23 markets, dropped repeat inquiries 25%, and was estimated at 40 million dollars of profit improvement. **Then the cautionary half:** by 2025 the CEO said the automation had gone too far and Klarna began rehiring humans so a customer could always reach a person. That resolution-time collapse is the prize a voice agent chases, and the walk-back is the reason a human path stays wired in from day one. It is Problem-First layer 2 (outcome) and layer 4 (safety) at industry scale. [[Klarna press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[the walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data and the channel, because every method in the layers below is chosen for a specific shape of them. Change an assumption and the method changes with it. For this case study, assume:

- **Channel: streamed audio, both ways, in real time.** Audio arrives as a continuous stream over telephony or WebRTC, and your reply streams back the same way while you are still forming it. *This is why* streaming is the default at every stage and latency is the binding constraint that ranks above almost everything else.
- **Latency: sub-second to first audio is the bar.** Natural human turn gaps sit around 200 to 300 milliseconds, and dead air past roughly a second reads as a dropped call, so the caller should hear the reply begin within a fraction of a second of finishing. *This is why* every stage gets a latency budget, the model has to stream its first tokens, and you spend real effort on when the caller stopped talking.
- **Callers interrupt and speak in fragments.** People barge in over the agent, trail off, restart, and say "um." *This is why* you need voice-activity detection, endpointing, barge-in handling, and partial transcripts, none of which a text chatbot ever has to think about.
- **Transcripts are noisy.** Speech recognition mishears names, order numbers, and homophones, so the text your model reads is imperfect. *This is why* the agent confirms high-stakes details back to the caller out loud, and retrieval has to tolerate a wrong word or two.
- **Corpus: the same help center, plus live order data.** A few hundred to a few thousand help-center articles, updated occasionally, alongside per-caller order state reached through tools. *This is why* a single hybrid index plus a few typed tools is enough, and freshness means re-indexing on edit rather than streaming.
- **Actions and stakes, with no screen to fall back on.** A few safe actions (order lookup, ticket) and high-impact ones (refund, cancellation), and the caller cannot see a confirmation dialog, so a wrong spoken action is costly and hard to take back. *This is why* high-impact actions are gated behind a spoken confirmation and a human, and the handoff is the safe default.
- **Language: one to start.** Primarily English at launch. *This is why* a single speech-recognition model, a single voice, and a single embedder are fine now, and multilingual is a later extension (Follow-up 6).

Keep these in view as you read the layers: each choice below points back to one of them. If your callers never interrupt, your calls are asynchronous voicemails rather than live, or a wrong action costs nothing, revisit the assumption and pick a different method.

### Step 2: walk the layers for this system

**Layer 1, the model.** This is a load-bearing layer for a voice agent, because the model choice sets your latency floor before you design anything else, so go deep here. The first decision is the shape of the model itself. There are two, and they trade latency against control.

*Term first, so the rest reads cleanly.* **[ASR](https://en.wikipedia.org/wiki/Speech_recognition)** (automatic speech recognition, also called speech-to-text) turns the caller's audio into text. **[TTS](https://en.wikipedia.org/wiki/Speech_synthesis)** (text-to-speech, also called speech synthesis) turns the agent's text back into audio. The model in the middle is what decides what to say.

- **A pipeline: streaming ASR into a text model into streaming TTS.** Three components in a row: transcribe the caller, feed the text to a normal language model, speak the model's text back. Its strength is control and observability, because each stage produces an inspectable artifact (a transcript, a decision, a reply) that you can log, evaluate, and guardrail, and you can swap any one component without touching the others. Its cost is that latency stacks across the three stages, and the text hop discards tone, pacing, and emotion that were in the caller's voice.
- **A single speech-to-speech model: audio in, audio out.** One model takes the caller's audio and emits the agent's audio directly, folding ASR, reasoning, and TTS into a single call. This is what the [OpenAI Realtime API](https://developers.openai.com/api/docs/guides/realtime) exposes, and what research systems like [Moshi](https://arxiv.org/abs/2410.00037) demonstrate. Its strength is latency and naturalness: fewer hops, and the model keeps the prosody and can handle overlapping speech natively. Its cost is that the intermediate text is no longer a clean seam you can inspect and gate, and the approach is newer, so fewer places exist to insert a guardrail or a deterministic check.

*What data decides it:* your latency bar, how much you need per-stage observability and the ability to gate each seam, whether tone and emotion carry weight for your brand, and whether regulation requires you to keep an exact transcript of every turn. A high compliance and control need pulls toward the pipeline; a raw latency and naturalness need pulls toward speech-to-speech. Many teams start with the pipeline for the observability and move hybrid as speech-to-speech models mature.

> **Real outlier: Intercom Fin Voice.** Building a production phone agent, Intercom chose the speech-to-text, language-model, text-to-speech **pipeline** over a single voice-to-voice model, and said plainly they did it to prioritize control and observability over theoretical performance. Simple queries returned in about 1 second; queries that needed 3 to 4 seconds got a spoken filler ("let me look into this for you") so the line never went silent. Their text predecessor resolved about 56% of conversations on average across 5,000+ customers, with some reaching 70 to 80%. The lesson: on a real deployment, the seams you can inspect can matter more than the theoretical latency win, and a filler phrase is a real turn-taking tool. [[Intercom case study](https://www.zenml.io/llmops-database/building-a-production-voice-ai-agent-for-customer-support-in-100-days)]

Whichever shape you pick, the model is non-deterministic, so the same call can produce different phrasings. You handle that the same way as any agent (structured decisions, grounding, and evaluation), plus one voice-specific control: you evaluate the spoken output for quality rather than only the text, which Layer 3 covers.

Within either shape, model **routing** still applies: a cheap fast model handles the routine majority, and a stronger model is reserved for the ambiguous cases. On voice, routing is also a latency lever, because the fast model's quicker first token is what keeps the line alive.

**Layer 2, the wrapping layer (the architecture).** This is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. On a voice agent it looks much like a text support agent, with every choice pulled toward speed and toward tolerating noisy transcripts.

**Knowledge (retrieval).** The agent must ground every spoken answer in your help center, or it invents policy out loud, which is worse on a call because the caller cannot see a citation. That is a retrieval pipeline, and every stage is a decision whose right answer depends on your data. Each choice below points back to the [assumptions](#the-assumptions-we-make-about-the-data-and-the-use-case) above.

```
┌──────────────────────────────────────────────────────┐
│ Knowledge pipeline: retrieval with a relevance floor │
└──────────────────────────────────────────────────────┘

  ┌────────────────────────────┐
  │      Help-center docs      │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │       Parse / ingest       │
  │ structured text + metadata │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │           Chunk            │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │           Embed            │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │    Vector store / index    │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │          Retrieve          │
  │    hybrid: dense + BM25    │
  │      (transcript in)       │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │           Rerank           │
  │          top few           │
  └────────────────────────────┘
                 ▼
  ┌────────────────────────────┐
  │      Relevance floor       │
  └──────────────┬─────────────┘
            ┌────┴─────────────────┐
above floor ▼                      ▼ below floor
  ┌───────────────────┐  ┌───────────────────┐
  │ To agent / answer │  │ Retrieve nothing  │
  └───────────────────┘  │ Hand off to human │
                         └───────────────────┘
```

- **Parse and ingest, then chunk.** Parse the help center structure-aware (headings, tables, ordered steps) and attach metadata (article id, `updated_at`, product, locale) to every chunk, because freshness and filtering ride on it. Chunk by structure, one section per chunk, since a single article usually answers a routine call. The high-leverage move is [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval): prepend a short model-generated blurb situating each chunk in its document before embedding, which Anthropic reported cut retrieval failures by about 35%, and about 49% combined with reranking. *What data decides it:* a chunk-size sweep scored on a retrieval measure such as [recall@k](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29) (does the right chunk show up in the top results) against a labeled set.
- **Embed and index.** An [embedding](https://arxiv.org/abs/2004.04906) is a list of numbers that captures a piece of text's meaning, so two texts that mean similar things land close together in that number space. Pick an embedder from the [MTEB retrieval leaderboard](https://huggingface.co/spaces/mteb/leaderboard) trained near your domain, and store the vectors in an index that makes search fast. At a few thousand articles, one [HNSW](https://arxiv.org/abs/1603.09320) index or [pgvector](https://github.com/pgvector/pgvector) on Postgres is plenty. *What data decides it:* corpus size, queries per second, and whether you need metadata filters and per-tenant isolation.
- **Retrieve, hybrid, because transcripts carry both meaning and exact tokens.** [Dense retrieval](https://arxiv.org/abs/2004.04906) compares meaning, so "can I send it back" finds the return-policy article, which is what you want for spoken paraphrases. [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is a keyword-scoring formula that ranks a chunk by how many of the query's words it contains, weighting rarer words more, which catches an order number or a product code the caller reads out. Because a call carries both, run both and merge the two ranked lists with [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), a rule that rewards a chunk for ranking highly in either list. On voice you also lean on retrieval being robust to a wrong word, since ASR mishears; keeping the sparse arm helps when the dense arm is thrown by a garbled phrase.
- **Rerank, then a relevance floor.** A [reranker](https://www.sbert.net/examples/applications/cross-encoder/README.html) is a model that reorders the top candidates so the best land first; a [cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) does this by reading the question and each chunk together rather than separately, which is more accurate and slower, so you run it on a few dozen candidates. On voice you weigh the reranker's accuracy gain against the milliseconds it adds, because those milliseconds come straight out of the latency budget. The **relevance floor** is the single most important safety control: set a threshold (the idea behind [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884), which retrieve, grade, then answer, re-retrieve, or abstain) so that when nothing clears it the agent hands off to a human rather than inventing policy on the phone. *Because a wrong spoken action is costly,* set the floor conservatively.

> **Real finding: [Lost in the Middle](https://arxiv.org/abs/2307.03172).** Accuracy drops when the relevant passage sits in the middle of a long context rather than at the ends. More retrieved text is not a free win, which is why you rerank and cap how many chunks you pass, doubly so on voice where every extra token the model reads is latency the caller hears.

**Tools (actions).** Tools are the agent's hands. Each one is a typed, allowlisted contract rather than open-ended code execution.

```
  order_lookup(order_id) -> {status, eta}          READ    low blast radius, cheap to trust
  create_ticket(summary, category) -> ticket_id    WRITE   idempotent, logged
  issue_refund(order_id, amount)                   WRITE   high-impact -> spoken confirm + human (Follow-up 3)
```

The calls that matter are the same as any agent: the **tool description is a prompt** (a vague description causes wrong calls); **least privilege** (read tools are cheap to trust, write tools are gated); **[idempotency](https://en.wikipedia.org/wiki/Idempotence)** (a retried `create_ticket` must not open two tickets); **error handling** (a failed call is retried or handed off, never hallucinated over); and the **loop is bounded** by a hard step cap. This is the [ReAct](https://arxiv.org/abs/2210.03629) pattern: reason, act, observe, repeat. The voice-specific addition is **do not create dead air while a tool runs**: speak a short filler and stream it, or acknowledge the request, while the order lookup completes in the background, which is exactly the "let me look into this" move Intercom described.

**Memory.** Memory is what makes this a conversation instead of a series of unrelated questions, and it comes in layers.

```
  SHORT-TERM (this call)     : the running transcript, so "what about express?" resolves against the last turn
  WORKING   (this task)      : the retrieved context and tool results the agent is reasoning over right now
  LONG-TERM (this caller)    : past orders and tickets, RETRIEVED on demand, never read wholesale into the prompt
```

The calls that matter: **summarize old turns rather than carrying them verbatim** so the context stays small and the model stays fast; **retrieve long-term memory, do not dump it** (pull the 2 relevant past tickets rather than the whole history); and **treat memory as untrusted**, because anything persisted from a caller or a document can smuggle an injected instruction into a later turn (the memory arm of the lethal trifecta in Follow-up 3). Together, knowledge, tools, and memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for an agent this is the center of the design, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, cutting off an ungrounded or unsafe answer and firing the human handoff the moment something is off. There is no single accuracy number, because good means something different for every product, so start where we teach you to start: **from failure modes.** Ask what could go wrong that would be unacceptable (a wrong refund spoken and acted on, a hallucinated policy, a missed handoff, a reply that arrives so late the caller hangs up, a voice so garbled the caller cannot understand it), then translate each into an observable, measurable behavior. On a voice agent the failure modes span three axes at once: **is it correct, is it fast enough, and is the speech understandable.** You measure all three together, because a fast wrong answer is still wrong, and a correct answer that arrives after the caller has left resolved nothing.

**Measure against today's baseline rather than an abstract target.** The right bar is the status quo: the current IVR menu and the hold-and-wait phone experience. Where calls went unanswered or callers gave up on hold because staffing every line was too costly, a voice agent that handles them reliably clears a low bar by existing. Where a human line already resolves the call well, the agent has to match or beat that on resolution and caller experience, measured retrospectively against how those same calls went. That keeps the eval tied to the decision the business faces rather than chasing a round accuracy number.

Evaluate at three levels: **each component** (did ASR transcribe the caller correctly, did retrieval find the right doc, did the model ground its answer, did it call the right tool, did TTS produce clear speech), **the whole call** end to end (was the issue actually resolved), and **live traffic** (is it still good, and still fast, in production).

> **Real finding: [tau2-bench](https://arxiv.org/abs/2506.07982).** On a tool-using agent, a single-attempt pass rate looks respectable, but pass^k (succeed on all k independent tries) collapses as k grows. A voice agent that works 1 call in 1 and fails 1 in 4 is not shippable, and on the phone a failed call is a person stuck on hold. Reliability is the bar, above average accuracy.

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: did it call `order_lookup` with the right id, is the output valid, did it hand off when nothing cleared the floor, was time-to-first-audio under budget. Fast, reliable, cheap. Compare against a reference dataset (golden answers) here. **[Word error rate](https://en.wikipedia.org/wiki/Word_error_rate)** (WER, the share of words ASR gets wrong, checked by transcribing against a known script) is a code-based speech metric that belongs here.
- **LLM judges.** One model scoring another against an explicit rubric, for subjective qualities (faithfulness, tone, whether the handoff was appropriate) that code cannot capture. Scalable, and a new source of non-determinism, so it must be calibrated before you trust it.
- **Human evaluation.** The gold standard you calibrate the other two against. On voice, humans also score speech quality by **[mean opinion score](https://en.wikipedia.org/wiki/Mean_opinion_score)** (MOS, listeners rating naturalness on a 1 to 5 scale). Too slow to run on all traffic, so you sample: calibration, edge cases, high-stakes calls.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of labeled calls across answerable, partial, unanswerable, action, interruption, and adversarial cases, ideally with real audio so ASR and TTS are exercised rather than just text. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Speech in (ASR)** | mishears names, ids, homophones | word error rate, entity error rate on ids and names | code-based against a known script |
| **Retrieval** | misses the right article, returns junk | recall@k, precision@k, [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank), [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) | code-based against labeled transcript to doc pairs |
| **Answer generation** | hallucinates, off-topic, wrong citation | [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) / groundedness, correctness vs golden, completeness | LLM judge with rubric + golden answers |
| **Tool use** | wrong tool, wrong arguments, needless call | tool-selection accuracy, argument exact-match, tool-call success rate | code-based against the expected trace |
| **Turn-taking** | cuts the caller off, or leaves dead air, or talks over them | endpoint accuracy, false-barge-in rate, response-to-barge-in time, dead-air duration | code-based on labeled turn boundaries |
| **Speech out (TTS)** | robotic, garbled, mispronounces | mean opinion score, pronunciation accuracy on domain terms | human sample + MOS predictor |
| **Latency** | reply arrives too late | time-to-first-audio p50 / p95, end-to-end turn latency | code-based timing on the trace |
| **Safety** | executes injected instructions, takes an unsafe action | injection-resistance rate, unsafe-action rate | adversarial red-team suite |
| **End to end** | call not actually resolved | containment / resolution rate, pass@1, **pass^k**, turns-to-resolution | scenario suite + judge |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: pick the two or three per component that map to your real failure modes and drop the rest. A dashboard with 30 metrics and no owner tells you nothing. Report **pass^k** alongside the average (the tau2-bench finding), and for this product track faithfulness, time-to-first-audio, and containment as the three highest-signal numbers, one per axis: correct, fast, resolved.

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem, or is it merely interesting?
- **Reliability:** human evaluation and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check (WER, a latency timer) is cheap, a fast judge call is medium, detailed human MOS review is expensive.

High impact and low cost are the must-haves (the latency timer, the unsafe-action check, the handoff flag). High impact and high cost are strategic investments you run on a sample (a calibrated faithfulness judge, human MOS scoring). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the caller, run inline so the system can act the moment they trip: an unsafe action, an ungrounded answer, low confidence, a latency spike that means the reply is stalling. The action is immediate (hand off to a human, cut off a bad reply). Guardrails must be fast and reliable before sophisticated, and on voice they must fit inside the turn.
- **Improvement flywheel (offline, batch).** Everything else: quality trends, faithfulness on a sample, MOS, containment, drift. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live traffic:

| Metric | Job | Why it matters |
|---|---|---|
| unsafe / incorrect-action rate | guardrail | the hard ceiling from scoping; must stay near zero |
| groundedness on a live sample | guardrail + flywheel | catches hallucination drift before callers report it |
| low-confidence / uncertainty trigger | guardrail | fires the human handoff in real time |
| time-to-first-audio p50 / p95 | guardrail + flywheel | the number the caller feels; a spike means the line is going silent |
| barge-in response time | guardrail + flywheel | how fast the agent stops talking when interrupted |
| dead-air / talk-over rate | flywheel | turn-taking health, the tell that endpointing is mistuned |
| containment / resolution rate | flywheel | share of calls resolved without a human, the core outcome |
| caller satisfaction / hangup rate | flywheel | quality as the caller actually feels it |
| handoff rate | flywheel | too high means it is not helping; too low means risky over-containment (the Klarna walk-back) |
| word error rate on live audio | flywheel | surfaces ASR drift on accents and noisy lines |
| cost per contained call, tokens per turn | flywheel | unit economics, the number finance asks about |

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label examples, run the judge on the same set, measure agreement (percent agreement or Cohen's kappa), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the underlying model. Then run the discovery loop, because callers will always find failures your metrics were never built for. Sample live traffic on **signals** that are unique to voice: the caller talking over the agent, repeated "hello? are you there?", long silences, and hangups mid-turn. When a signal keeps firing but your metrics look clean, that gap is the tell: a human listens to those calls, names the quality dimension you were not measuring (maybe the agent pauses a beat too long before a tool result), and it becomes a new metric added back into the reference dataset. Evaluation is never finished.

![Eval and discovery loop: pre-deployment validation, then production monitoring, with the reference set growing each cycle](../assets/eval_discovery_loop.png)

Instrument the app with OpenInference so every node, tool call, model call, and latency span lands in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. On voice the latency spans are first-class, because time-to-first-audio is the metric the caller experiences directly. *Deeper:* [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../../topics/evaluation.md).

**Layer 4, production and ops.** This is the load-bearing layer for a voice agent, so it gets the deepest treatment. Everything here serves one goal: a natural spoken turn that begins within a fraction of a second and survives interruptions. It is 5 pieces: the real-time pipeline, streaming and partial results, endpointing, barge-in, and the latency budget that ties them together.

**The real-time pipeline.** Even with a single speech-to-speech model, it helps to see the full loop, because the same jobs exist inside that one model. Here it is as a pipeline, which is where the seams are visible.

```
┌───────────────────────────────────────────────────────┐
│ The real-time pipeline: one voice turn, seams visible │
└───────────────────────────────────────────────────────┘

  ┌────────────────────────────────────┐
  │      Caller audio (streamed)       │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │                VAD                 │
  │      is the caller speaking?       │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │           Streaming ASR            │
  │   interim transcripts (partial)    │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │    Endpointing / turn detection    │
  │         caller has stopped         │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │  Model: retrieve / tool / decide   │
  │        streams reply tokens        │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │           Streaming TTS            │
  │          agent audio out           │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │         Caller interrupts?         │
  │ barge-in: duck / stop TTS, re-plan │
  └────────────────────────────────────┘

  on barge-in the loop returns to the top: the agent stops speaking and re-plans against the new audio
```

**Streaming and partial results.** *Term first:* **partial (or interim) results** are the preliminary transcripts a streaming ASR emits while the caller is still talking, before the words are final. A streaming recognizer marks these `is_final: false` and only later commits a finalized transcript ([Deepgram interim results](https://developers.deepgram.com/docs/interim-results)). The reason this matters for latency is that it lets the rest of the system start early: the agent can begin retrieving, or narrow its intent, off the partial transcript rather than waiting for the caller to finish, so by the time the caller stops, work is already underway. Everything streams: audio in streams, ASR streams partials, the model streams its first tokens, and TTS streams its first audio, so the caller hears the reply begin before the full sentence exists.

**Endpointing: knowing when the caller stopped.** *Term first:* **[endpointing](https://developers.deepgram.com/docs/endpointing)** is deciding the moment the caller has finished their turn so the agent can take its turn. This is one of the highest-leverage tuning knobs on the whole system, because it sits on a sharp tradeoff. The classic approach is a silence timer on top of **[voice-activity detection](https://en.wikipedia.org/wiki/Voice_activity_detection)** (VAD, the component that decides whether a slice of audio contains speech or silence): wait for a set gap of silence, then treat the turn as over. Tune the gap too short and you cut the caller off mid-sentence, which is the fastest way to make an agent feel rude; tune it too long and every turn carries dead air. A silence timer also cannot tell a thinking pause from a finished thought.

The current high-leverage move is a **turn-detection model** that reads the words rather than just the silence. Semantic endpointing predicts whether the utterance is actually complete from its linguistic content, so "my order number is" holds the turn even across a long pause, while "where is my order" releases it fast. LiveKit's open-weight turn detector, for example, is a 135M-parameter transformer that adds this contextual signal on top of VAD and runs on a CPU alongside the agent ([LiveKit on turn detection](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection)). *What data decides it:* measure your dead-air duration and your cut-off rate on labeled turn boundaries, and tune the endpointing gap or swap in a semantic model until both sit where your callers tolerate them.

**Barge-in: handling interruptions.** *Term first:* **barge-in** is the caller starting to speak while the agent is still talking, and handling it means the agent stops promptly and listens. This is table stakes for a natural call, because people interrupt to correct, to add detail, or to cut a long answer short. The mechanism is a **duplex** audio path: the system keeps listening even while it is speaking, and the moment VAD detects the caller's voice, it ducks or stops the TTS playback and re-plans against what the caller just said. The runnable code models exactly this: a barge-in utterance cuts the in-progress turn short, clears the half-formed reply, and routes back to re-plan. One subtlety worth measuring is telling a real interruption from a backchannel like "uh-huh" or "right," which a caller says to signal they are listening rather than to take the floor; treating every backchannel as a full barge-in makes the agent flinch and stop constantly.

**The latency budget: where the time goes.** Put a number on every stage and add them up, because the caller experiences the sum. This is the single most useful artifact on a voice agent. The numbers below are illustrative examples for a pipeline; track your own from traces and drive down the largest stage.

```
┌─────────────────────────────────────────────────────────────────┐
│ The latency budget: where the time goes (illustrative pipeline) │
└─────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────┐
  │          Caller stops talking          │
  └────────────────────────────────────────┘
                       ▼
  ┌────────────────────────────────────────┐
  │    Endpointing decides turn is over    │
  │   ~200-500 ms · silence gap, tunable   │
  └────────────────────────────────────────┘
                       ▼
  ┌────────────────────────────────────────┐
  │              Finalize ASR              │
  │ ~0-100 ms · partials already streamed  │
  └────────────────────────────────────────┘
                       ▼
  ┌────────────────────────────────────────┐
  │      Retrieve + model first token      │
  │ ~300-500 ms · routing + streaming help │
  └────────────────────────────────────────┘
                       ▼
  ┌────────────────────────────────────────┐
  │          TTS first audio out           │
  │  ~100-200 ms · streaming, first chunk  │
  └────────────────────────────────────────┘
                       ▼
  ┌────────────────────────────────────────┐
  │      Caller hears the reply begin      │
  └────────────────────────────────────────┘

  time-to-first-audio target: under ~1 s
```

Two things fall out of this budget. First, endpointing is often the largest and most controllable slice, which is why a semantic turn detector earns its place. Second, the metric that matters is **time-to-first-audio**, the moment the caller hears the reply start, and analogous to time-to-first-token in a text stream; total turn time matters less, because a caller who hears the answer begin will wait through the rest. When a tool call would blow the budget, you speak a filler and run the tool in the background, keeping the line alive while the work completes.

> **Real outlier: [Moshi](https://arxiv.org/abs/2410.00037) (Kyutai, 2024).** A full-duplex speech-to-speech model that models the caller's stream and its own stream in parallel, so explicit speaker turns disappear and overlap is native. It reports a theoretical latency of 160 milliseconds and about 200 milliseconds in practice on a single GPU. That is the ceiling the architecture is chasing: near human-conversation turn gaps, from one model, with barge-in built into the design rather than bolted on. It is the strongest argument for where speech-to-speech is heading, weighed against the observability a pipeline still gives you today.

**Reliability, scale, and observability** round out the layer: streaming sessions are stateful and long-lived, so you scale concurrent calls rather than stateless requests (Follow-up 1); telephony brings its own concerns (jitter buffers, packet loss, the [Opus](https://en.wikipedia.org/wiki/Opus_(audio_format)) codec); and every call is traced end to end so you can replay the exact turn where the agent talked over someone. Detail is in Follow-ups 1 and 2.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: model routing (a cheap fast model for the routine majority, a stronger one for the rest, which on voice is also a latency lever), prompt and semantic caching, advanced retrieval, and multi-agent. Routing, caching, and shaving the latency budget pay off on the live turn, and multi-agent earns its place off that turn: post-call summarization, QA, and follow-up sub-agents run after the call where latency is free, which Follow-up 5 lays out.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (fixed retrieve-then-answer, a silence-timer endpoint, hand off when nothing clears the relevance floor), prove it with evals that include latency and speech quality, and only hand the model more freedom (tools, a bounded loop, a semantic turn detector, barge-in re-planning) once the measurement says the constrained version is solid and its limits are what block you. You earn agency through evaluation before you grant it.

Composed, the layers give one architecture:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Observability: every node, tool, model call, and latency span is traced (Arize) │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────┐
  │    Caller audio (streamed)     │
  │ VAD + streaming ASR · partials │
  └────────────────────────────────┘
                   ▼
  ┌────────────────────────────────┐
  │     Endpoint / turn detect     │
  │        caller stopped?         │
  └────────────────────────────────┘
                   ▼
  ┌────────────────────────────────┐
  │        Input guardrail         │
  │        injection · PII         │
  └────────────────────────────────┘
                   ▼
  ┌────────────────────────────────┐
  │            Retrieve            │
  │    hybrid + rerank · floor     │
  └────────────────────────────────┘
                   ▼                 act / observe
  ┌────────────────────────────────┐    ┌─────────────────────┐
  │           Agent loop           │    │ Tools (allowlisted) │
  │         bounded ReAct          │◀──▶│   order_lookup ·    │
  └────────────────────────────────┘    │   create_ticket ·   │
                   ▼                    │    issue_refund*    │
  ┌────────────────────────────────┐    └─────────────────────┘
  │             Memory             │
  │    call transcript + caller    │
  └────────────────────────────────┘
                   ▼
  ┌────────────────────────────────┐
  │        Output guardrail        │
  │  grounded, safe & in-budget?   │
  └────────────────┬───────────────┘
           ┌───────┴───────────┐
       yes ▼                   ▼ no / low conf / high-impact
  ┌─────────────────┐  ┌──────────────┐
  │  Streaming TTS  │  │   Hand off   │
  │ speak to caller │  │ human (warm) │
  └─────────────────┘  └──────────────┘

  barge-in: caller interrupts, duck or stop TTS, and re-plan against the new speech
  issue_refund* requires a spoken confirmation and human approval (Follow-up 3)
```

Read it as the spine composed: the model (layer 1), whether a pipeline or a single speech-to-speech model, wrapped in retrieval, tools, and memory (layer 2), gated by evals (layer 3) that run inline as guardrails and offline as a flywheel, all of it under production streaming and turn-taking with a latency budget (layer 4), with optimization (layer 5) held to what this system needs, which is routing, caching, and latency shaving rather than a second agent. The relevance floor, the output guardrail, and the latency guardrail are what make the human handoff the safe default: when the agent cannot ground an answer, the action is high-impact, or the reply is stalling, it hands off to a human with the full call context rather than guessing or leaving the line silent.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this control flow (minus the audio and scale pieces): it models one voice turn as text, with a per-node latency budget, retrieval with a relevance floor, a bounded agent loop with tools, a grounding guardrail, a barge-in re-plan that cuts a turn short, and a human handoff where high-impact requests like refunds and cancellations hand off. It runs offline with a deterministic policy, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                              # run the scenarios (also a self-test)
python run.py "Where is my order 5012?"    # run one voice turn from the command line
```

Audio is out of scope for a tiny demo, so a real system wraps this graph with streaming ASR in front and streaming TTS behind, or replaces both with a single speech-to-speech model. Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route decisions through a real model. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK and its voice pipeline, the Anthropic SDK, LiveKit Agents, Pipecat, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now handle 10x the concurrent calls. Where does it break, and how do you scale it?"

Name where it breaks first, then scale that, cheapest lever first. Voice scales differently from chat, because a call is a long-lived stateful stream rather than a stateless request.

- **The audio path (layer 4).** Each active call holds an open ASR stream, a TTS stream, and a model session for its whole duration, so you scale concurrent sessions rather than requests per second. ASR and TTS (or the speech-to-speech model) become the compute bottleneck ahead of the text model, because they run continuously per call. Pool and autoscale the speech services, and size for peak concurrency plus headroom, since a caller cannot be queued behind a spinner.
- **Retrieval (layer 2).** The keyword search becomes a hybrid dense plus BM25 store with a reranker, kept fresh by an ingestion pipeline and scoped by per-tenant access control so one caller never reaches another account's data.
- **Routing and caching (layer 5).** Route routine calls to a small fast model, reserve a larger one for hard cases, cache the stable prompt prefix, and add a semantic cache for the repeated routine questions that dominate a support line.
- **Infrastructure (layer 4).** Horizontal scale behind a session-aware load balancer, backpressure so a spike degrades gracefully (a brief hold message) instead of dropping calls, and graceful failover for the telephony leg.

Put numbers on it: concurrent calls per replica, cost per contained call, and a latency budget that still holds under load, because a p95 that balloons at peak is the failure the caller feels. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 2: "Cut time-to-first-audio without making it feel rushed."

Budget latency per stage (endpointing, ASR finalize, retrieve, model first token, TTS first audio), then attack the largest stage, which on voice is usually endpointing or the model's first token.

- **Tune or replace endpointing.** Move from a fixed silence timer to a semantic turn-detection model so the agent takes its turn the instant the caller is genuinely done, without shortening the gap so far that it cuts people off. This often buys the most, because the silence gap is dead time by construction.
- **Stream everything and act on partials.** Begin retrieval off the interim transcript, stream the model's first tokens into TTS, and start speaking the first chunk before the full reply exists, so time-to-first-audio stays low even when total turn time does not.
- **Route to a smaller model for the routine path,** and reserve the large model for the hard cases; the fast model's quicker first token is a direct latency win.
- **Consider a single speech-to-speech model** where the three-stage pipeline latency is the wall you keep hitting, weighing the latency gain against the per-stage observability you give up.
- **Fill tool time with speech.** When a tool call would blow the budget, speak a short filler and run the tool in the background rather than leaving silence.

> **Real number: [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).** Caching the stable prefix of a prompt (system instructions, tool schemas) can cut cost by up to about 90% and latency by up to about 85% on the cached portion, per provider documentation. On a voice agent whose system prompt and tool schemas are identical on every turn, this trims the model's first-token time, which is the slice the caller hears as the pause before the reply.

Quality and pacing hold because the eval set from Layer 3 gates each change: if a faster endpoint raises the cut-off rate, or a smaller model drops grounded-answer rate, it does not ship. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 3: "A caller tries to talk it into a refund it should not give. How do you prevent wrong or unsafe actions?"

Separate answering from acting, and gate the acting, with an extra wrinkle for voice: there is no screen, so the confirmation itself has to be spoken and logged.

- **Action guardrails.** High-impact tools (refunds, cancellations) require a spoken confirmation read back to the caller and human approval or a hard policy check, never blanket autonomy. Allowlists over blocklists.
- **Untrusted input.** The caller's speech and any retrieved document are untrusted, and the agent never executes instructions found in them. The sharp way to see the risk is the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
  The lethal trifecta

          ┌───────────────────┐   ┌──────────────┐   ┌───────────────────┐
          │ Untrusted content │   │  Access to   │   │    Ability to     │
          │                   │   │ private data │   │ act / communicate │
          └─────────┬─────────┘   └───────┬──────┘   └─────────┬─────────┘
                    └─────────────────────┼────────────────────┘
                                          ▼
                           ┌─────────────────────────────┐
                           │ All 3 together is dangerous │
                           └─────────────────────────────┘

              any 2 are manageable; keep all 3 from meeting in one path
```

- **Blast radius.** Least-privilege tool scopes, an immutable audit log of every action with the audio and transcript that justified it, and a design where the worst a manipulation achieves is a needless handoff, while a wrongful refund stays impossible without a human.

*Deeper:* [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 4: "New rule: every call must be recorded and auditable, with consent. How does the design change?"

That single constraint promotes 3 normally-optional components into load-bearing walls: **consent capture** at the start of the call, **call recording with a retained transcript**, and an **immutable audit trail** on every action. Capture consent as the first turn (and branch to a human or a recording-free path if the caller declines), store the audio and the finalized transcript with retention and access controls, redact sensitive fields (payment details, other personal data) from the stored transcript, and log every tool action against the segment that justified it. This is also where the pipeline architecture earns its keep, because its inspectable transcript is exactly what an audit needs, which can tip the Layer 1 choice on its own. Nothing else in the box diagram changes: a domain constraint is what forces you to engineer a layer you would otherwise accept as a framework default.

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

Keep the live turn a single agent. A voice turn lives and dies on latency, and a routing hop between agents in the middle of a sentence is the one cost the caller feels, so the real-time path stays one agent with tools. The place multi-agent (layer 5, optimization) genuinely helps is off that path: everything that happens after the caller hangs up, plus a warm handoff to a live specialist during the call. That is where you recommend it, because the latency budget no longer applies.

**When multi-agent earns its place (off the latency path):**
- **Post-call work is naturally parallel and off the clock.** Summarizing the call, filling the CRM, checking QA against policy, and drafting the follow-up are independent jobs that run after the caller hangs up, so the token cost buys quality with no latency penalty.
- **A live turn should hand off rather than fan out.** When a call needs a domain the agent does not own (a billing dispute, a security freeze), the win is a warm handoff to a specialist, human or a specialist agent, with the transcript and context passed along, instead of a mid-turn agent hop the caller waits through.
- **Different post-call jobs want different context and rules.** A QA reviewer scoring the call against policy, a summarizer, and a follow-up drafter each judge more sharply with only their own task in context.
- **The call is valuable enough to pay for it.** Off the latency path, multi-agent trades tokens for quality where the trade is clean.

**How you would extend this architecture.** Keep the real-time agent exactly as designed for latency, and add a background pipeline that fires when the call ends. An orchestrator dispatches specialist sub-agents in parallel: a summarizer writes the call notes, a QA sub-agent scores the transcript against policy and flags coaching moments, and a follow-up sub-agent drafts the email or the ticket for human approval. During the call, a warm-handoff step passes the transcript and context to a live specialist when the intent needs one. The same guardrails, evals, and observability wrap the background agents, and none of them sit on the caller's turn.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Observability: spans + latency on the live turn, spans on the background pipeline │
└───────────────────────────────────────────────────────────────────────────────────┘

                                            ┌─────────────────────────┐
                                            │ Live turn: single agent │
                                            │    latency-critical     │
                                            └─────────────────────────┘
                                                         ▼
                              ┌────────────────────────────────────────────────────┐
                              │   caller to ASR + endpoint to input guardrail to   │
                              │ agent (tools) to output guardrail to TTS to caller │
                              └────────────────────────────────────────────────────┘
                                                         ▼
                                               ┌──────────────────┐
                                               │    Call ends     │
                                               └──────────────────┘
                                                         ▼
                                               ┌──────────────────┐
                                               │    Transcript    │
                                               └──────────────────┘
                                                         ▼
                                               ┌──────────────────┐
                                               │   Orchestrator   │
                                               │     dispatch     │
                                               └─────────┬────────┘
                             ┌───────────────────────────┴───────────────────────────┐
                             ▼                           ▼                           ▼
                ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
                │  Summarizer sub-agent  │  │      QA sub-agent      │  │  Follow-up sub-agent   │
                │       call notes       │  │     policy score +     │  │ draft email / ticket · │
                │                        │  │     coaching flags     │  │     human approval     │
                └────────────────────────┘  └────────────────────────┘  └────────────────────────┘

  sub-agents run in parallel off the latency path; a warm handoff passes context to a human specialist mid-call
```

> **Real data: Anthropic's multi-agent research system.** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens, and token usage explained roughly 80% of the performance gap. The takeaway is to spend those tokens where task value and genuine parallelism justify them, and to stay single-agent where they do not. On the live turn you would add latency to that ledger, so the second agent belongs off it, in the background pipeline where the tokens buy quality and the caller waits for nothing. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]

The honest rule: keep the live turn single-agent, instrument it, and move every job that can wait into the background pipeline where multi-agent pays without touching latency. *Deeper:* [Agents](../../../topics/agents.md).

---

## Follow-up 6: "Make it multilingual."

The architecture does not change; the data and the evaluation do. You need per-language speech recognition and a voice per language for text-to-speech (or a speech-to-speech model that covers them), retrieval coverage of the help center in each language or cross-lingual embeddings, and an eval set with labeled calls per language so word error rate, faithfulness, and containment are measured per language rather than assumed. Turn-taking gets its own attention, because pause and interruption norms differ across languages and a single endpointing gap will feel wrong somewhere. Handoff and guardrails apply unchanged. Klarna's assistant ran in 35+ languages on one architecture. This is the recurring lesson: most "make it do X" follow-ups are answered in the data and eval layers, and the box diagram stays the same. *Deeper:* [Evaluation](../../../topics/evaluation.md).

---

## Follow-up 7: "A better real-time model drops next quarter. How do you avoid a rewrite?"

Build the harness on-policy and keep the model behind a provider-agnostic interface (the runnable code already does this, any model via one call), so swapping a pipeline model or a whole speech-to-speech model is a config change rather than a fork. Pin behavior with the eval set rather than with brittle prompt hacks, and make that eval set cover latency and speech quality rather than only correctness, so a new model that is smarter but slower cannot slip through and blow the turn budget. Pair every "do not do X" rule with an eval so you can delete the rule when a better model makes it obsolete. When the new model lands, you swap it, re-run the eval gate including the latency and MOS checks, and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Real-world reference points

- **Intercom Fin Voice:** chose the ASR, language-model, TTS pipeline over a single voice-to-voice model to keep control and observability; about 1-second responses on simple queries, spoken fillers when a query needed 3 to 4 seconds; the text predecessor resolved about 56% of conversations on average, some customers 70 to 80%. On a real deployment, inspectable seams can outweigh a theoretical latency win. [[Intercom case study](https://www.zenml.io/llmops-database/building-a-production-voice-ai-agent-for-customer-support-in-100-days)]
- **Moshi (Kyutai, 2024):** a full-duplex speech-to-speech model with about 160 ms theoretical and 200 ms practical latency, showing where near-human turn gaps from a single model are heading. [[paper](https://arxiv.org/abs/2410.00037)]
- **Klarna (2024 to 2025):** 700 agents' worth of work, 2.3M chats in month 1, resolution time from 11 minutes to under 2, 35+ languages, then a public walk-back toward keeping humans reachable. The resolution-time prize and the reason the human path stays wired in. [[press](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/)] [[walk-back](https://www.customerexperiencedive.com/news/klarna-reinvests-human-talent-customer-service-AI-chatbot/747586/)]
- **Turn detection:** human turn gaps sit around 200 to 300 ms, while silence-timer agents lag; a small semantic turn-detection model (LiveKit ships a 135M-parameter one) reads the words to decide the turn is over, closing the gap without cutting callers off. [[LiveKit](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection)]
- **tau2-bench:** pass^k collapses as k grows; reliability is the shippable bar, above average accuracy, and on the phone a failed call is a person on hold. [[paper](https://arxiv.org/abs/2506.07982)]

---

## Research to know

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) (Kyutai 2024): full-duplex speech-to-speech with sub-second latency.
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis 2020): the grounding pattern the answers rest on.
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao 2022): reason and act in a loop, the shape of the agent.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu 2023): why more retrieved context is not free, sharper when tokens are latency.
- [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): retrieval that decides when to retrieve, re-retrieve, or abstain.
- [tau2-bench](https://arxiv.org/abs/2506.07982): evaluating tool-using agents on multi-turn tasks with a reliability metric.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md)); [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../../resources/harness_engineering.md); Aishwarya's talks ([Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k) and [Generative AI in the Real World](https://www.youtube.com/watch?v=Ajiu8uyfSq0), both on O'Reilly) and her [YouTube channel](https://www.youtube.com/channel/UCf9CdAgj8AHmpMwyoe67w7w); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- OpenAI, the [Realtime API guide](https://developers.openai.com/api/docs/guides/realtime) and [Voice agents guide](https://developers.openai.com/api/docs/guides/voice-agents), for the speech-to-speech and voice-pipeline patterns.
- Deepgram docs on [endpointing](https://developers.deepgram.com/docs/endpointing) and [interim results](https://developers.deepgram.com/docs/interim-results), the streaming ASR primitives the turn-taking rides on.
- LiveKit on [turn detection, VAD, and endpointing](https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection), and the open-weight [turn-detector model](https://huggingface.co/livekit/turn-detector).
- [Arize observability docs](https://arize.com/docs/) and [LangGraph docs](https://langchain-ai.github.io/langgraph/), the tracing loop and state-machine primitives the [code](code/) uses.

---

## Related in this repo

Topics: [RAG](../../../topics/rag.md) · [Agents](../../../topics/agents.md) · [Evaluation](../../../topics/evaluation.md) · [Production and LLMOps](../../../topics/production.md) · [Safety and Security](../../../topics/safety-security.md). Guides: [Harness Engineering](../../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Other case: [Customer Support Agent](../customer-support-agent/README.md). Prepping to be asked this? See the [interview prep hub](../../README.md).
