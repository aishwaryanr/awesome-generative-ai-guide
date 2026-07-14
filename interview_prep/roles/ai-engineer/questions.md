# AI Engineer: Question Bank

Sixty-plus questions with concise, correct model answers, grouped by theme. Answers are 3 to 6 sentences: enough to anchor your own, not a script to recite. This is current for 2025-2026 (agents, MCP, context engineering, reasoning models, pass^k, cost and latency).

Pair this with the repo's broader banks: [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and [Role-Based Interview Prep](../../role_based_prep.md).

**Themes:** [LLM fundamentals](#llm-fundamentals) - [Prompting and context engineering](#prompting-and-context-engineering) - [RAG](#retrieval-and-rag) - [Agents and tool use](#agents-and-tool-use) - [Evaluation](#evaluation) - [Reasoning models](#reasoning-models) - [Cost, latency, deployment](#cost-latency-and-deployment) - [Safety and responsible AI](#safety-and-responsible-ai) - [System design](#system-design) - [Business and judgment](#business-and-judgment)

---

## LLM fundamentals

**1. What is a token, and why does tokenization matter in practice?**
A token is the unit a model reads and generates, usually a subword fragment produced by a tokenizer like byte-pair encoding, so "tokenization" is roughly 3 or 4 tokens, not one. It matters because context limits, latency, and cost are all measured in tokens, not characters or words. Tokenization also explains failure modes: models struggle with character-level tasks (counting letters, reversing strings) and with rare words or numbers because those split into odd token sequences. A rough rule of thumb is 1 token to about 4 characters or 0.75 words in English.

**2. What is the context window, and what happens as you fill it?**
The context window is the maximum number of tokens (input plus output) a model can attend to in a single call. Longer windows let you pass more instructions, history, and retrieved data, but they are not free: cost and latency grow with input length, and quality can degrade well before the hard limit. Models tend to use information at the start and end of a long context better than the middle, and very long noisy contexts can actively hurt reasoning. So the goal is the right, minimal context, not the maximum.

**3. What do temperature and top-p control, and how do you set them?**
Both shape how the next token is sampled from the model's probability distribution. Temperature scales the distribution: low values (0 to 0.3) make output more deterministic and focused, high values increase diversity and risk. Top-p (nucleus sampling) restricts sampling to the smallest set of tokens whose probabilities sum to p, cutting the long tail of unlikely tokens. For extraction, classification, or tool-calling you want low temperature for reliability; for brainstorming or creative copy you raise it. Set one deliberately and leave the other at its default rather than tuning both blindly.

**4. Why do LLMs hallucinate, and can you eliminate it?**
An LLM predicts likely text; it has no built-in notion of truth, so when the training data is thin or the prompt pushes it past what it knows, it produces fluent but false output. You cannot eliminate hallucination, only manage it: reduce it with grounding (RAG), constrained outputs, and better context; detect it with evals, citation checks, and monitoring; and contain it by showing sources, allowing correction, and routing high-stakes or low-confidence cases to a human. Treat it as a risk you bound, not a bug you fix.

**5. What is an embedding, and how is it used?**
An embedding is a dense vector that represents the meaning of a piece of text (or image, audio) so that semantically similar items sit close together in vector space. You compute embeddings for your documents once, store them in a vector index, then embed a query at request time and retrieve the nearest vectors by cosine similarity. Embeddings power semantic search, RAG retrieval, clustering, deduplication, and classification. Choosing an embedding model is a real decision: dimensionality, max input length, domain fit, and cost all matter, and you should evaluate retrieval quality on your own data rather than trusting a leaderboard.

**6. At a high level, how does a transformer work, and what is self-attention doing?**
A transformer processes a sequence of token embeddings through stacked layers of self-attention and feed-forward networks. Self-attention lets every token weigh how much to draw from every other token, so the model builds context-dependent representations (the meaning of "bank" shifts based on surrounding words). Multiple attention heads capture different relationships in parallel. For decoder-only LLMs, attention is causal (a token only attends to earlier tokens), which is what makes next-token prediction possible.

**7. What is the difference between a base model, an instruction-tuned model, and a reasoning model?**
A base (pretrained) model just predicts the next token and is not reliably steerable by instructions. An instruction-tuned (or chat) model has been post-trained with supervised fine-tuning and preference optimization to follow instructions and hold a conversation, which is what you almost always use in products. A reasoning model is further trained (often with reinforcement learning on verifiable rewards) to spend test-time compute on an internal chain of thought before answering, trading latency and cost for stronger multi-step performance. You pick based on the task: instruction-tuned for most product work, reasoning for hard multi-step problems.

**8. What is the difference between an LLM's parametric knowledge and non-parametric knowledge?**
Parametric knowledge is what the model absorbed into its weights during training; it is fast to access but frozen at the training cutoff, hard to attribute, and can be stale or wrong. Non-parametric knowledge is information you supply at inference time through the context window, typically via retrieval. RAG is the standard way to inject non-parametric knowledge so answers stay fresh, auditable, and grounded in a source you control. The design question is which facts belong in the weights (stable, general) versus in retrieval (volatile, proprietary, or requiring citations).

**9. What is a mixture-of-experts (MoE) model and why does it matter for AI Engineers?**
A mixture-of-experts model replaces some dense layers with many "expert" subnetworks and a router that activates only a few experts per token. This lets total parameter count (and capability) grow while keeping the compute per token roughly constant, so you get a stronger model at a similar inference cost. For an AI Engineer it mostly matters as an explanation for why some large models are surprisingly fast and cheap to serve, and it can affect latency variance. You rarely tune MoE internals; you consume the served model.

---

## Prompting and context engineering

**10. What is context engineering, and why can it matter more than prompt wording?**
Context engineering is deciding what actually enters the context window (instructions, retrieved documents, tool results, memory, examples) and what stays out, plus how it is ordered and formatted. It often beats clever phrasing because models degrade when context is bloated, noisy, or badly ordered, so getting the right minimal information in is the higher-leverage move. In agentic systems it is the central discipline: the context is assembled fresh each step from many sources, and quality depends on curating it. Think of the context window as a scarce, actively managed budget, not a dumping ground.

**11. Walk through the main prompting techniques and when to use each.**
Zero-shot (just the instruction) is the default and often enough for capable models. Few-shot adds a handful of input-output examples to pin down format or a tricky pattern the instruction alone does not convey. Chain-of-thought asks the model to reason step by step, which helps on multi-step problems, though modern reasoning models do this internally so you should not force it on them. Structured techniques (output schemas, role framing, decomposition into sub-prompts) improve reliability. Choose the least complex technique that clears your quality bar, and validate the choice with an eval.

**12. How do you get reliable structured output (for example JSON) from an LLM?**
Prefer native mechanisms over prompt-and-pray: use the provider's structured-output or tool-calling feature with a JSON schema so the model is constrained to valid shapes, and validate the result against that schema in code. When constrained decoding is unavailable, give an explicit schema and one example in the prompt, set a low temperature, and parse defensively. Always handle the failure path: a retry with the validation error fed back, or a fallback value. Do not extract fields with brittle regex or string slicing from free-form prose.

**13. What is "lost in the middle" and how do you design around it?**
Models retrieve information best when it sits near the start or the end of the context and worst when it is buried in the middle of a long prompt. So blindly stuffing dozens of retrieved chunks in arbitrary order wastes them. Design around it by retrieving fewer, higher-quality chunks, reranking so the most relevant content is placed at the edges, and keeping the total context tight. This is one reason "more context" is not automatically "better answers."

**14. How do you debug a prompt that works most of the time but fails intermittently?**
First reproduce and categorize the failures against a small labeled set instead of eyeballing one bad output. Isolate the cause: is it ambiguous instructions, missing context, a formatting issue, temperature-driven variance, or genuinely hard inputs. Fix the highest-leverage cause (tighten the instruction, add a schema, lower temperature, improve retrieval, add a targeted example) and re-run the eval to confirm the failure rate dropped without regressing other cases. Treat prompts like code: change one thing, measure, keep a regression set.

**15. What is few-shot prompting's cost, and when should you avoid it?**
Every example you include is input tokens on every call, so few-shot raises cost and latency and eats context budget. Avoid or trim it when a capable model already handles the task zero-shot, when the examples are long, or when you can move the pattern into a fine-tune or a schema instead. If you do use it, choose examples that cover the tricky and edge cases rather than easy ones, and check whether 2 well-chosen examples beat 8 mediocre ones.

**16. How do you manage conversation history in a long chat without blowing the context budget?**
You cannot keep appending forever, so you compress. Common tactics: keep the last few turns verbatim, summarize older turns into a running memory, and store durable facts (user preferences, decisions) in a structured memory you retrieve as needed rather than replaying the whole transcript. Prune aggressively and put the most relevant recent content near the end where the model uses it best. The goal is to preserve the information that changes the next response, not the literal history.

---

## Retrieval and RAG

**17. When would you use RAG instead of fine-tuning to give a model new knowledge?**
Use RAG when the knowledge changes often, is large or proprietary, or needs citations: retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, tone, or to bake in a stable skill, not to inject volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior. Start with prompting plus RAG because it is cheaper and faster to iterate, and reach for fine-tuning only when prompting cannot hit the bar.

**18. Walk me through a basic RAG pipeline end to end.**
Offline: ingest documents, clean and chunk them, embed each chunk, and store the vectors (plus metadata and the source text) in an index. Online: embed the user query, retrieve the top k nearest chunks (often with a hybrid of dense and keyword search), optionally rerank them, assemble a prompt that includes the query plus the retrieved context with instructions to answer only from it and cite sources, then generate. Around this you add guardrails (refuse when nothing relevant is retrieved) and evaluation on both retrieval and generation. Each stage is a tunable knob: chunking, embedding model, k, reranking, and prompt.

**19. How do you choose a chunking strategy?**
Chunking trades recall against precision: chunks too large dilute relevance and waste context, too small lose the surrounding meaning needed to answer. Start with structure-aware chunking (by section, paragraph, or semantic boundary) rather than a blind fixed character count, and add a small overlap so ideas that straddle a boundary are not lost. Match chunk size to your content and your embedding model's sweet spot, and keep metadata (source, title, section) attached for filtering and citations. Then validate: measure whether the answer-bearing chunk is actually retrieved on your eval set, and adjust.

**20. What is the difference between dense, sparse, and hybrid retrieval?**
Dense (embedding) retrieval matches on meaning, so it finds semantically related passages even without shared words, but it can miss exact terms, codes, or rare names. Sparse retrieval (BM25 and keyword methods) matches on lexical overlap, which is strong for exact terms and cheap, but blind to paraphrase. Hybrid retrieval runs both and fuses the results (for example with reciprocal rank fusion), capturing exact matches and semantic matches together, which usually beats either alone in production. The cost is added complexity, so justify it with an eval.

**21. What does a reranker do, and when is it worth it?**
A reranker is a second-stage model (often a cross-encoder) that scores each retrieved candidate against the query jointly, giving a much more accurate relevance ordering than the first-stage vector similarity. The pattern is retrieve many cheaply, then rerank to keep the best few, which raises precision and lets you pass fewer, better chunks to the model. It is worth it when first-stage retrieval returns roughly-relevant but poorly-ordered results and your generation quality is retrieval-limited. The tradeoff is added latency and cost per query, so measure the quality gain before committing.

**22. Why might a model answer worse with retrieved context than with none?**
Because irrelevant, conflicting, or low-quality retrieved text can distract or mislead the model, and long noisy context degrades reasoning. Bad retrieval (wrong chunks, duplicated content, contradictory sources) is often worse than no retrieval at all. This tells you that retrieval quality and context construction matter as much as the model, that more context is not automatically better, and that you must evaluate the retrieval stage separately so you can see when it is the problem.

**23. How do you evaluate a RAG system?**
Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually retrieved, using recall and precision at k (and hit rate). For generation, measure faithfulness (is the answer grounded in the retrieved context with no fabrication), answer relevance (does it address the question), and correctness, using a labeled question set and an LLM-as-judge with a clear rubric. Include unanswerable questions to check that the system refuses instead of hallucinating, and monitor the same signals in production, not just offline.

**24. What is the RAG triad?**
The RAG triad is three complementary checks on a RAG answer: context relevance (was the retrieved context actually relevant to the query), faithfulness or groundedness (is the answer supported by that context without contradiction or fabrication), and answer relevance (does the answer address the user's question). Together they localize failures: low context relevance points at retrieval, low faithfulness points at the generation step ignoring or overriding the context, and low answer relevance points at the prompt or the model. Tools like Ragas operationalize these, but you still validate the judge against human labels.

**25. What is agentic RAG, and when do you need it?**
Agentic RAG puts an agent in charge of retrieval: instead of a single fixed retrieve-then-generate step, the model decides whether to retrieve, reformulates queries, retrieves iteratively, chooses among multiple sources or tools, and reasons over what it gets back. You need it for complex or multi-hop questions where one retrieval pass cannot gather everything, or where the right source depends on the question. The cost is more latency, more model calls, and new failure modes (bad query reformulation, loops), so use fixed RAG when it clears the bar and reserve agentic RAG for questions that genuinely require multi-step gathering.

**26. What is GraphRAG and what problem does it solve?**
GraphRAG builds a knowledge graph of entities and relationships from your corpus and retrieves over that structure, sometimes with community summaries, instead of only over flat text chunks. It helps with global or multi-hop questions ("what themes connect these documents", "how does A relate to C through B") that plain chunk retrieval answers poorly because the relevant facts are scattered across many chunks. The tradeoff is a heavier, more expensive ingestion pipeline to build and maintain the graph. Reach for it when questions are relational or corpus-wide rather than answerable from a single passage.

**27. How does RAG change at enterprise scale?**
The model stops being the bottleneck; the data and retrieval layer becomes it. You now need robust ingestion and freshness pipelines, access control so retrieval respects per-user permissions (a user must never retrieve a document they cannot see), cost and latency management across millions of queries, evaluation across many document types, and observability on retrieval quality. Metadata, incremental indexing, and permission-aware filtering matter more than any clever prompt. Design the data layer first, because that is where enterprise RAG usually breaks.

**28. How do you handle document freshness and updates in a RAG index?**
Use incremental indexing keyed on document identity and version so you can upsert changed documents and delete removed ones rather than rebuilding the whole index. Track source timestamps and, where staleness is dangerous, expose the date in the answer or filter by recency. Decide a re-embedding policy for when the embedding model or chunking changes. Monitor for drift between the live corpus and the index, and treat freshness as an SLA, not an afterthought, since stale retrieval silently produces confidently wrong answers.

**29. A user asks something your corpus does not cover. What should the system do, and how do you make it do that?**
It should refuse gracefully ("I do not have that information") rather than hallucinate from parametric knowledge. You engineer this by instructing the model to answer only from the retrieved context and to say it cannot find the answer when the context is insufficient, by setting a retrieval-score threshold below which you treat the query as unanswerable, and by including unanswerable cases in your eval set so you can measure the refusal rate. Getting this right is a top signal in take-homes, because it separates a grounded system from a confident fabricator.

---

## Agents and tool use

**30. What makes something an agent rather than a single LLM call, and when do you actually need one?**
An agent adds tools, memory, and a loop, so it can take actions, observe results, and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow (a predetermined chain of steps) will do, because agents add cost, latency, and new failure modes. The mature default is the simplest thing that works: prompt, then workflow, then agent only when the task genuinely requires dynamic control flow.

**31. Explain the ReAct pattern.**
ReAct interleaves reasoning and acting: the model produces a thought, chooses an action (a tool call), observes the result, then loops (think, act, observe, repeat) until it decides to finish. This lets it gather information and correct course mid-task instead of committing to a plan up front. It is the backbone of most tool-using agents. The practical risks are looping without progress and reasoning that drifts from the observations, which you bound with iteration caps, good tool design, and clear stopping criteria.

**32. How does function calling / tool use work under the hood?**
You describe your tools to the model as a schema (name, description, typed parameters). The model does not call anything itself; it emits a structured request naming a tool and its arguments, your code executes that tool, and you feed the result back into the context for the next step. Reliability depends on clear tool descriptions and well-typed parameters, and on handling the cases where the model calls the wrong tool, passes bad arguments, or the tool errors. Good agents reflect on a tool error and retry with a corrected call rather than crashing.

**33. What is the Model Context Protocol (MCP) and why does it matter?**
MCP is an open standard for connecting an agent to external tools and data sources through a uniform interface. It matters because you wire a capability once as an MCP server and reuse it across any MCP-compatible agent or harness, instead of writing a bespoke integration for every tool in every app. This decouples tool-building from agent-building and is becoming the common plug for the agent ecosystem. The tradeoff to watch is security: an agent connected to many MCP servers has a wide action surface, so permissions and trust boundaries matter.

**34. How do you keep an agent from looping forever or running up cost?**
Put hard limits in: a maximum number of steps or tool calls, a wall-clock or token budget, and a cost ceiling per task. Add progress checks so the loop stops when it is not advancing (repeated identical actions, no new information). Use cheaper models for routine sub-steps and a stronger one only where needed, cache stable context and tool results, and give the model only as much reasoning budget as the step requires. Design tools so failures return actionable errors the model can recover from rather than triggering blind retries.

**35. What is agent memory, and what types are there?**
Memory is how an agent carries information beyond a single turn or context window. Short-term (working) memory is the current context: recent messages, tool results, scratchpad. Long-term memory persists across sessions and is usually stored externally (a database or vector store) and retrieved on demand, covering durable facts, user preferences, and past outcomes. The engineering job is deciding what to write to long-term memory, how to retrieve the relevant slice into context, and how to keep it from growing stale or bloated. Memory is a context-engineering problem, not just a storage one.

**36. When is a multi-agent system worth it over a single agent, and what are the risks?**
Multi-agent designs (a planner plus specialists, or agents that hand off) help when the work decomposes into distinct roles or parallelizable subtasks, or when separation of concerns improves reliability and keeps each agent's context focused. The risks are real: more cost and latency, harder debugging, error propagation and miscommunication between agents, and coordination overhead that can outweigh the benefit. Start with a single well-scoped agent and split into multiple only when a concrete limitation forces it. Communication protocols and clear contracts between agents matter as much as the prompts.

**37. What is the difference between a workflow and an agent, and why prefer workflows when you can?**
A workflow is a predetermined sequence of steps you orchestrate in code (retrieve, then summarize, then classify); the control flow is fixed and predictable. An agent decides its own next step at runtime using the model, which is more flexible but less predictable and harder to test. Prefer workflows when the steps are known in advance because they are cheaper, faster, easier to debug, and easier to guarantee. Use an agent only when you genuinely cannot enumerate the path ahead of time.

**38. How do you evaluate an agent, and why is average accuracy not enough?**
Evaluate outcomes and trajectory: did it complete the task, and did it get there sensibly (right tools, valid arguments, faithful use of tool outputs, no wasted steps). Average accuracy hides unreliability, so measure consistency with metrics like pass^k (does it succeed on all k independent attempts), because an agent that works on average but fails unpredictably is not shippable. Use trace-based evaluation to inspect each step, and an LLM-as-judge on the trajectory for tool appropriateness and grounding, validated against human labels. Production autonomy demands reliability, not just a good mean.

**39. How would you design tools for an agent so it uses them reliably?**
Treat tool design as interface design for a model: clear names, precise descriptions of when to use each tool, well-typed and minimal parameters, and return values that are informative but not bloated. Fewer, well-scoped tools beat a large sprawling set that confuses the model's selection. Make errors actionable so the model can self-correct, and consider consolidating multiple low-level calls into one higher-level tool that matches how the task is actually done. Then test tool selection and argument accuracy directly in your eval.

---

## Evaluation

**40. How do you evaluate an LLM feature that has no single correct answer (for example a summary or a draft reply)?**
You define what "good" means as checkable criteria (faithful to the source, covers the key points, right tone and length, no fabrication) and score against those rather than against one golden string. Build a small high-quality labeled set of realistic and adversarial cases, then use an LLM-as-judge with an explicit rubric to score at scale, complemented by targeted human review. Report the criteria separately so you can see which dimension is weak. Version the eval set and treat it as a defensible artifact, not a one-off script.

**41. What is LLM-as-judge, and what are its failure modes?**
LLM-as-judge uses a model to score or compare outputs against a rubric, which scales evaluation far beyond manual review. Its failure modes are well documented: position bias (favoring the first option), verbosity bias (favoring longer answers), self-preference (favoring outputs from the same model family), and inconsistency. You mitigate with a clear rubric, randomized option order, pairwise comparison where possible, and, most importantly, validating the judge against human labels before trusting it. A judge you have not calibrated against humans is a guess dressed up as a metric.

**42. What is reward hacking or eval gaming, and how do you guard against it?**
It is when a system optimizes the measured proxy rather than the real goal, for example exploiting a judge's verbosity bias or a benchmark's shortcuts to score well without being better. Guard against it with held-out and adversarial evals, multiple diverse metrics or judges instead of one, and by checking whether gains transfer to independent tasks rather than a single benchmark. Watch for suspicious jumps and inspect what actually changed. If you can game your own eval in 5 minutes, so can the optimization process.

**43. What is the difference between offline and online evaluation, and why do you need both?**
Offline evaluation runs against a fixed labeled dataset before you ship, giving a fast, repeatable signal you can gate releases on. Online evaluation measures the live system on real traffic (task success, user feedback, escalation rate, latency, cost, and sampled quality checks), catching distribution shift and failures your dataset never covered. You need both because a system can pass offline and still fail on real users, and offline sets go stale as usage evolves. Treat evaluation as release infrastructure spanning both.

**44. Why measure reliability such as pass^k for agents rather than just accuracy?**
Because an agent that succeeds on average but fails unpredictably is not shippable for autonomous work. Pass^k asks whether the agent succeeds on all k independent attempts of the same task, capturing consistency, which is what production actually requires from a long-running or high-stakes task. A 90% average that means a 10% chance of a wrong action every run is very different from one that is reliably right. Reliability metrics surface the variance that a mean hides.

**45. How do you build an eval set from scratch for a new feature?**
Start from what success means as a checkable outcome, then collect or write a small, high-quality set (dozens before hundreds) that covers realistic cases, edge cases, and adversarial ones, including cases that should be refused. Label it carefully, because a noisy eval set is worse than none. Split it so you can iterate on one part and hold out another to check you are not overfitting. Version it, document the labeling criteria, and grow it as production surfaces new failure modes.

**46. What benchmarks or metrics would you actually cite, and how much do you trust them?**
Public benchmarks (reasoning, coding, retrieval, agentic task suites) are useful for a coarse first read and for tracking the field, but they are proxies that can be contaminated or gamed and rarely match your task distribution. Trust your own eval on your own data far more than any leaderboard. For RAG, cite retrieval recall/precision at k and faithfulness/answer-relevance; for agents, task completion and pass^k; for generation, task-specific rubric scores. The honest answer in an interview is that benchmarks orient you but your bespoke eval decides.

---

## Reasoning models

**47. How do reasoning models differ from standard LLMs, and how does that change how you prompt them?**
Reasoning models are trained to spend test-time compute on an internal chain of thought before answering, often via reinforcement learning on verifiable rewards, so they perform much better on math, code, and multi-step problems. They trade latency and cost for that: a single call can take seconds and burn many hidden reasoning tokens. Prompting changes: give the goal, constraints, and context, and let the model do the step-by-step work rather than forcing your own chain-of-thought or over-specifying the method. You also expose a reasoning-effort or budget control where available.

**48. When should you use a reasoning model versus a standard model?**
Use a reasoning model for genuinely hard, multi-step tasks where a fast model fails: complex planning, tricky math or logic, hard debugging, or agentic steps that need careful deliberation. Use a standard (and cheaper, faster) model for routine work: extraction, classification, summarization, straightforward RAG answers, and most tool calls. The right architecture often routes: a cheap model handles the common path and escalates only the hard cases to a reasoning model. Decide with an eval and a cost-latency budget, not by defaulting to the biggest model.

**49. What are the practical downsides of reasoning models in production?**
They are slower and more expensive per call because of the hidden reasoning tokens, which strains latency budgets and cost ceilings, especially inside an agent loop where each step already takes 1 to 3 seconds. Their variable output length makes latency less predictable. Their internal reasoning is not always faithful to the true cause of the answer, so you cannot treat the visible chain of thought as a reliable explanation. Manage them with routing, budgets, caching, and streaming, and reserve them for steps that actually need the extra compute.

---

## Cost, latency, and deployment

**50. How would you keep an LLM feature's cost and latency under control?**
Cache the stable prefix of the context and reuse the KV cache so repeated prompts are cheaper and faster, keep context minimal and well-ordered, and use smaller or cheaper models for sub-steps, escalating only when needed. Cap tool calls and loop iterations, cache tool and retrieval results, stream tokens so perceived latency drops even when total time does not, and batch where throughput matters. Give reasoning models only as much budget as the step needs. Every one of these is a dial you set per use case against a measured budget.

**51. Explain the latency-throughput-cost triangle and the main dials you have.**
You are trading three things: how fast a single request returns (latency), how many requests you can serve per second (throughput), and dollars per request (cost). The main dials are model size (smaller is faster and cheaper but weaker), batching and continuous batching (raise throughput at some latency cost), KV-cache reuse and prompt caching (cut cost and time on repeated context), context length (shorter is cheaper and faster), and streaming (improves perceived latency). You cannot max all three, so you name the binding constraint for the use case and tune toward it.

**52. What is prompt caching / KV caching and when does it help?**
During generation the model computes and stores key-value tensors for the tokens it has processed; reusing that KV cache avoids recomputing the shared prefix on later calls. Prompt caching exposes this so a long stable system prompt, tool schema, or retrieved context that repeats across requests is billed and processed at a large discount. It helps most when many requests share a long common prefix (a fixed system prompt, few-shot block, or the same document). Order your context so the stable, cacheable part comes first and the variable part last.

**53. How do you decide which model to use for a given step (model routing)?**
Match model capability to task difficulty and constraints: use the smallest, cheapest, fastest model that clears the quality bar for that step, and escalate to a stronger or reasoning model only for the hard cases. Implement this as a router (rules or a lightweight classifier) so the common, easy path stays cheap and only the tail pays for the big model. Validate the routing with an eval so you know the cheap model actually holds quality where you use it. Revisit the choice as models get cheaper and better.

**54. How does quantization affect an AI Engineer's decisions, even if you do not train models?**
Quantization shrinks a model (for example to 4-bit or 8-bit) to cut memory and inference cost at some accuracy loss, which matters when you self-host or run at the edge. The impact is task-dependent, so if you use a quantized served model you measure the quality drop on your own eval before shipping rather than assuming it is fine. It also lowers latency and lets bigger models fit on smaller hardware. You rarely quantize yourself in this role, but you should know it as a cost-quality lever and test its effect.

**55. What does observability look like for an LLM application in production?**
You trace every request end to end: the prompt and context assembled, retrieval hits and scores, each tool call and its result, the model output, tokens, latency per stage, and cost. Tools like tracing dashboards let you inspect a full multi-step agent trajectory to see where it went wrong. You log for auditability, sample outputs for ongoing quality checks, and alert on drift in the metrics that matter (hallucination signals, refusal rate, latency, cost, error rate). Without this you are flying blind on a non-deterministic system, so build it in from the start.

---

## Safety and responsible AI

**56. What is prompt injection, and how is indirect injection different and worse?**
Prompt injection is when input text overrides your instructions and makes the model do something unintended. Direct injection comes from the user typing a malicious prompt. Indirect injection is more dangerous: the malicious instructions hide inside content the model ingests from elsewhere (a retrieved document, a web page, a tool result, an email), so the attack rides in through data the user never sees and the model treats as trusted. It is worse because your attack surface is every external source your RAG or agent touches, and the user is not the attacker.

**57. How do you defend an agentic system against prompt injection and misuse?**
There is no single fix, so you layer defenses: give tools least privilege and scope so a compromised step cannot do much, require human approval for sensitive or irreversible actions, separate trusted instructions from untrusted data in the context, and add input and output guardrails to filter obvious attacks and unsafe outputs. Sandbox tool execution, log everything for audit, and constrain what the agent can reach. Assume some injection will get through and design so the blast radius is small rather than betting on perfect detection.

**58. What are the main security concerns specific to agents with real-world tool access?**
The big ones are indirect prompt injection from retrieved or tool content, over-broad tool permissions and privilege escalation, data exfiltration (the agent leaking sensitive data through a tool or its output), and unsafe or irreversible actions taken autonomously. The wider the agent's action surface (more tools, more MCP servers, more autonomy), the larger the risk. Mitigate with least-privilege tools, human-in-the-loop for high-stakes actions, strict input and output validation, and audit logging. Treat every external input as untrusted.

**59. How do you handle PII, data privacy, and compliance in an LLM product?**
Keep data access least-privilege and permission-aware end to end, so retrieval and memory only ever surface what a given user is allowed to see. Be deliberate about what leaves your environment and what a model provider may log or retain, and use zero-retention or self-hosted options where the data demands it. Redact or minimize PII before it hits the model where you can, log for auditability, and honor deletion and consent requirements in both your index and your memory stores. Design the data boundary first; it is easier than retrofitting compliance.

**60. How do you roll out an LLM feature safely?**
Gate the release on an evaluation suite built around how the system actually breaks, start narrow with a human in the loop and a small user slice, and add monitoring and guardrails before you widen. Expand only as the metrics hold, keep a fast rollback, and use canaries or A/B splits so a regression hits few users. Watch for silent regressions after a model-provider update, since the underlying model can change under you. Treat evals and monitoring as release infrastructure, not a one-time check.

**61. A provider updates its model and your feature quietly regresses. How do you catch and handle it?**
You catch it because you have a regression eval that runs on a schedule (or before you adopt a new model version) and production monitoring on quality signals that would alarm on a drop. When it fires, you pin to the previous known-good model version if the provider allows it, reproduce the regression on your eval, adjust prompts or context if the fix is cheap, and only then move to the new version. This is why pinning model versions, keeping a regression set, and monitoring live quality are not optional for a production LLM feature.

---

## System design

**62. Design a customer support agent for an enterprise. Walk me through it.**
Start from requirements: accuracy bar, latency budget, escalation policy, cost ceiling, and compliance. Use retrieval over the company's knowledge base for grounding, tools for actions (look up an order, file a ticket) with least privilege, and a clear handoff to a human for low-confidence or high-stakes cases. Wrap it in evaluation and observability, add input and output guardrails (including against indirect injection from retrieved content), and roll out behind a fallback with a human in the loop. Name the tradeoff at every choice, and reach for an agent loop only where fixed retrieval will not do.

**63. Design a RAG system over 10 million internal documents with per-user access control.**
The data layer is the hard part, not the model. Build an ingestion pipeline that chunks, embeds, and indexes documents incrementally with rich metadata including access-control tags, and keep it fresh with upserts and deletes. At query time, filter retrieval by the requesting user's permissions before ranking so nobody ever retrieves a document they cannot see, then use hybrid retrieval plus reranking to keep precision high at that scale. Add caching for cost and latency, evaluate retrieval quality across document types, and monitor freshness and retrieval metrics. Access control and freshness are where enterprise RAG breaks, so design them first.

**64. Design the evaluation and monitoring system for an LLM feature already in production and getting hallucination complaints.**
First reproduce the complaints and build a labeled set from them, split into faithfulness, retrieval, and refusal failures, so you know which stage is at fault. Stand up offline evals (RAG triad plus task correctness with a human-validated judge) that gate any change, and online monitoring (sampled faithfulness checks, refusal rate, user feedback, citation coverage) that alarms on drift. Add tracing so you can inspect the retrieval and generation for any flagged response. Then fix the dominant failure mode (usually retrieval or a missing refusal path) and confirm the metric moves without regressing others.

**65. Design an insurance-claims agent that ingests a claim and outputs approve, deny, or needs-review.**
Requirements first: the accuracy and auditability bar is high and errors are costly, so this is a human-in-the-loop system by design. Use retrieval over policy documents and prior claims to ground each decision, structured tool calls to pull claim data, and a constrained output that returns a decision plus the cited evidence and a confidence signal. Route anything below a confidence threshold or above a value threshold to human review rather than auto-deciding, cap cost with a cheap model for triage and a stronger one only for hard claims, and add guardrails and full audit logging. Evaluate on labeled historical claims and monitor the approve/deny distribution for drift.

**66. How do you cut the p95 latency of a feature that makes one reasoning-model call per request?**
Question whether every request needs the reasoning model: route so the common, easy cases go to a fast standard model and only hard cases escalate, which alone often halves p95. Stream the response so perceived latency drops, cache the stable prompt prefix and any repeated context via KV/prompt caching, trim the context to cut input processing time, and cap the reasoning budget where the provider allows. If the work decomposes, run independent sub-steps in parallel instead of serially. Measure p95 before and after each change rather than guessing.

---

## Business and judgment

**67. How do you decide whether to build a feature with prompting, RAG, fine-tuning, or an agent?**
Match the tool to the need and start with the simplest that clears the bar. Prompting and context for behavior you can specify, RAG for fresh or proprietary knowledge that needs citations, fine-tuning for stable behavior, format, or style at scale (not for volatile facts), and an agent only when the task genuinely needs multiple dynamic steps and actions. Each step up adds cost, latency, and failure modes, so you earn the complexity with a requirement, not a hunch. Validate the choice with an eval on your own data.

**68. When is the right answer "do not use an LLM here"?**
When a deterministic rule, a lookup, a regex, or a classic ML model solves the problem more cheaply and reliably, an LLM is the wrong tool. Also when the task demands guarantees an LLM cannot give (exact arithmetic, legal correctness, perfect consistency) without heavy scaffolding, when latency or cost budgets cannot absorb it, or when the failure cost is high and you cannot put a human in the loop. Good judgment here is a strong senior signal: the best AI Engineers know where AI does not belong.

**69. How do you measure whether an AI feature is actually delivering value?**
Tie it to a user or business outcome, not a model score: task success or resolution rate, deflection or time saved, conversion, or reduction in manual work, alongside quality from your eval harness, hallucination or error rate, latency, and cost per interaction. Compare against the pre-AI baseline and watch human-override or escalation rate as a truth signal. If you cannot connect the feature to an outcome someone cares about, you cannot justify its cost, and cost per interaction is real money at scale.

**70. How do you think about dependence on a single model provider?**
It is a real business risk: pricing, availability, model behavior, and terms can change under you, and a silent model update can regress your feature. Mitigate by abstracting the provider behind an internal interface so swapping is cheap, pinning model versions and keeping a regression eval to catch changes, and periodically evaluating alternatives (including open-weight models you can self-host) on your own tasks. You do not need to be multi-provider from day one, but you should keep the option open and know your switching cost. Design so no single provider is a single point of failure for a critical feature.

**71. How do you communicate the risk and limitations of an AI feature to non-technical stakeholders?**
Be concrete about the failure modes and their likelihood, and frame quality as a distribution rather than a single impressive demo. Tie the risk to the specific decision the feature drives, so a stakeholder understands what a 5% error rate means for that use case. Set expectations with evals and a phased rollout plan instead of adjectives, and be honest about what you cannot guarantee. Credibility comes from naming the limits before they bite, not from overselling the demo.

---

Next: [resources](resources.md) and [courses](courses.md) to shore up any theme, then the [prep plan](prep-plan.md). Back to the [role README](README.md).
