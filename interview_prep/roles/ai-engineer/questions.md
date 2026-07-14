# AI Engineer: Question Bank

A large bank of questions with concise, correct model answers, grouped by theme. Every question is a click-to-open collapsible: read the question, form your own answer, then open it to check. Answers run 3 to 6 sentences, enough to anchor your own, not a script to recite. This is current for 2025-2026 (agents, MCP, context engineering, reasoning models, pass^k, contextual retrieval, cost and latency), and every answer ends with a "Learn more" pointer to go deeper.

Pair this with the repo's broader banks: [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and the [interview-prep path](../../../paths/interview-prep.md).

**Themes:** [LLM fundamentals](#llm-fundamentals) - [Prompting and context engineering](#prompting-and-context-engineering) - [RAG](#retrieval-and-rag) - [Agents and tool use](#agents-and-tool-use) - [Evaluation](#evaluation) - [Reasoning models](#reasoning-models) - [Cost, latency, deployment](#cost-latency-and-deployment) - [Safety and responsible AI](#safety-and-responsible-ai) - [System design](#system-design) - [Business and judgment](#business-and-judgment)

---

## LLM fundamentals

<details>
<summary><b>1. What is a token, and why does tokenization matter in practice?</b></summary>

A token is the unit a model reads and generates, usually a subword fragment produced by a tokenizer like byte-pair encoding, so "tokenization" is roughly 3 or 4 tokens, not 1. It matters because context limits, latency, and cost are all measured in tokens, not characters or words. Tokenization also explains failure modes: models struggle with character-level tasks (counting letters, reversing strings) and with rare words or numbers because those split into odd token sequences. A rough rule of thumb is 1 token to about 4 characters or 0.75 words in English.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>2. What is the context window, and what happens as you fill it?</b></summary>

The context window is the maximum number of tokens (input plus output) a model can attend to in a single call. Longer windows let you pass more instructions, history, and retrieved data, but they are not free: cost and latency grow with input length, and quality can degrade well before the hard limit. Models tend to use information at the start and end of a long context better than the middle, and very long noisy contexts can actively hurt reasoning. So the goal is the right, minimal context, not the maximum.

**Learn more:** [Chroma "Context Rot" research](https://research.trychroma.com/context-rot)

</details>

<details>
<summary><b>3. What do temperature and top-p control, and how do you set them?</b></summary>

Both shape how the next token is sampled from the model's probability distribution. Temperature scales the distribution: low values (0 to 0.3) make output more deterministic and focused, high values increase diversity and risk. Top-p (nucleus sampling) restricts sampling to the smallest set of tokens whose probabilities sum to p, cutting the long tail of unlikely tokens. For extraction, classification, or tool-calling you want low temperature for reliability; for brainstorming or creative copy you raise it. Set one deliberately and leave the other at its default rather than tuning both blindly.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>4. Why do LLMs hallucinate, and can you eliminate it?</b></summary>

An LLM predicts likely text; it has no built-in notion of truth, so when the training data is thin or the prompt pushes it past what it knows, it produces fluent but false output. You cannot eliminate hallucination, only manage it: reduce it with grounding (RAG), constrained outputs, and better context; detect it with evals, citation checks, and monitoring; and contain it by showing sources, allowing correction, and routing high-stakes or low-confidence cases to a human. Treat it as a risk you bound, not a bug you fix.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>5. What is an embedding, and how is it used?</b></summary>

An embedding is a dense vector that represents the meaning of a piece of text (or image, audio) so that semantically similar items sit close together in vector space. You compute embeddings for your documents once, store them in a vector index, then embed a query at request time and retrieve the nearest vectors by cosine similarity. Embeddings power semantic search, RAG retrieval, clustering, deduplication, and classification. Choosing an embedding model is a real decision: dimensionality, max input length, domain fit, and cost all matter, and you should evaluate retrieval quality on your own data rather than trusting a leaderboard.

**Learn more:** [RAG topic](../../../topics/rag.md)

</details>

<details>
<summary><b>6. At a high level, how does a transformer work, and what is self-attention doing?</b></summary>

A transformer processes a sequence of token embeddings through stacked layers of self-attention and feed-forward networks. Self-attention lets every token weigh how much to draw from every other token, so the model builds context-dependent representations (the meaning of "bank" shifts based on surrounding words). Multiple attention heads capture different relationships in parallel. For decoder-only LLMs, attention is causal (a token only attends to earlier tokens), which is what makes next-token prediction possible.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>7. What is the difference between a base model, an instruction-tuned model, and a reasoning model?</b></summary>

A base (pretrained) model just predicts the next token and is not reliably steerable by instructions. An instruction-tuned (or chat) model has been post-trained with supervised fine-tuning and preference optimization to follow instructions and hold a conversation, which is what you almost always use in products. A reasoning model is further trained (often with reinforcement learning on verifiable rewards) to spend test-time compute on an internal chain of thought before answering, trading latency and cost for stronger multi-step performance. You pick based on the task: instruction-tuned for most product work, reasoning for hard multi-step problems.

**Learn more:** [Planning and reasoning models (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md)

</details>

<details>
<summary><b>8. What is the difference between an LLM's parametric knowledge and non-parametric knowledge?</b></summary>

Parametric knowledge is what the model absorbed into its weights during training; it is fast to access but frozen at the training cutoff, hard to attribute, and can be stale or wrong. Non-parametric knowledge is information you supply at inference time through the context window, typically via retrieval. RAG is the standard way to inject non-parametric knowledge so answers stay fresh, auditable, and grounded in a source you control. The design question is which facts belong in the weights (stable, general) versus in retrieval (volatile, proprietary, or requiring citations).

**Learn more:** [Original RAG paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)

</details>

<details>
<summary><b>9. What is a mixture-of-experts (MoE) model and why does it matter for AI Engineers?</b></summary>

A mixture-of-experts model replaces some dense layers with many "expert" subnetworks and a router that activates only a few experts per token. This lets total parameter count (and capability) grow while keeping the compute per token roughly constant, so you get a stronger model at a similar inference cost. For an AI Engineer it mostly matters as an explanation for why some large models are surprisingly fast and cheap to serve, and it can affect latency variance. You rarely tune MoE internals; you consume the served model.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>10. What is the KV cache, and why is it the reason generation gets faster after the first token?</b></summary>

During generation the model computes key and value tensors for every token it has processed and caches them, so each new token attends to the stored keys and values instead of recomputing them for the whole sequence. That splits a request into a compute-heavy prefill phase (process the whole prompt at once) and a memory-bound decode phase (generate one token at a time using the cache). It matters because time-to-first-token is dominated by prefill over your input length, while each later token is cheap, and because the KV cache consumes GPU memory that grows with context length and batch size. Understanding it explains prompt caching, why long prompts are slow to start, and why serving throughput is memory-limited.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>11. Why are LLMs bad at arithmetic and character-level tasks, and what do you do about it?</b></summary>

Two reasons compound: tokenization splits numbers and words into fragments that do not align with digits or letters, and the model predicts likely text rather than executing an algorithm, so multi-digit math and letter-counting have no reliable internal procedure. The fix is not a better prompt but the right tool: give the model a calculator, code execution, or a database and have it call that instead of computing in its head. For structured math or data work, generate code and run it, then feed the result back. This is a core lesson: when a task needs guarantees, offload it to a deterministic tool rather than trusting the model's token prediction.

**Learn more:** [Tools in AI (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md)

</details>

<details>
<summary><b>12. How do you get a confidence or uncertainty signal out of an LLM?</b></summary>

There is no calibrated probability of correctness built in, so you approximate it. Token log-probabilities (where the API exposes them) give a rough per-token confidence you can aggregate, though fluent-but-wrong text often scores high. Self-consistency (sample several times and check agreement) surfaces instability, and an explicit "rate your confidence and say if you are unsure" prompt gives a coarse signal you must calibrate against real outcomes. Verbalized confidence and logprobs both need validation on your data before you gate anything on them, and for RAG the retrieval score is often a more honest uncertainty signal than anything the generator reports.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>13. What is a training cutoff, and how does it shape product decisions?</b></summary>

The training cutoff is the date beyond which the model saw no data, so anything more recent (events, prices, API changes, your own documents) is simply unknown to its weights and will be guessed or hallucinated if asked. This is why you never rely on parametric knowledge for anything time-sensitive or proprietary and instead inject it through retrieval or tools. It also affects code assistants that suggest deprecated APIs and any feature that reasons about "current" facts. Design so that volatile knowledge always enters at inference time, and surface the source and date so users can judge freshness.

**Learn more:** [What is RAG and agentic RAG (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part4_what_is_rag_and_agentic.md)

</details>

<details>
<summary><b>14. Are two identical calls to an LLM guaranteed to return the same output? Why does this matter?</b></summary>

No. At temperature 0 the sampling is greedy and largely deterministic, but you still get run-to-run variation from floating-point non-associativity across batched GPU kernels, from mixture-of-experts routing that depends on what else is in the batch, and from silent provider-side model updates. This matters because it breaks naive caching and testing, and because a feature that "worked yesterday" can drift with no code change on your side. You manage it by pinning model versions where the provider allows, keeping a regression eval, and designing so small output variation does not break downstream parsing (validate structured output, do not string-match).

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>15. How does a multimodal (vision-language) model process an image, and what should you watch for?</b></summary>

A vision-language model runs the image through an encoder that turns it into a sequence of visual tokens projected into the same embedding space as text, so the model attends over image and text tokens together. Practically, images consume tokens (often hundreds per image, scaling with resolution), so they cost real context budget and money, and very high-resolution documents may be tiled. Watch for weak spots: precise text reading in dense documents (use OCR when accuracy matters), spatial or counting tasks, and small details lost at the model's working resolution. Evaluate on your actual images rather than assuming demo-quality performance transfers.

**Learn more:** [Multimodal topic](../../../topics/multimodal.md)

</details>

<details>
<summary><b>16. What is distillation, and why might you use a distilled model?</b></summary>

Distillation trains a smaller "student" model to imitate a larger "teacher," transferring much of the capability into a model that is cheaper and faster to serve. As an AI Engineer you mostly consume distilled models the provider ships (the small, fast tier of a model family), and you can also distill your own: use a strong model to generate labeled outputs for your specific task, then fine-tune a small model on them. The tradeoff is that a distilled model narrows to what it was distilled on and loses general breadth, so you validate on your task distribution. It is a lever for cutting cost and latency once you have a task well-defined enough to specialize.

**Learn more:** [Fine-tuning 101 guide](../../../resources/fine_tuning_101.md)

</details>

---

## Prompting and context engineering

<details>
<summary><b>17. What is context engineering, and why can it matter more than prompt wording?</b></summary>

Context engineering is deciding what actually enters the context window (instructions, retrieved documents, tool results, memory, examples) and what stays out, plus how it is ordered and formatted. It often beats clever phrasing because models degrade when context is bloated, noisy, or badly ordered, so getting the right minimal information in is the higher-leverage move. In agentic systems it is the central discipline: the context is assembled fresh each step from many sources, and quality depends on curating it. Think of the context window as a scarce, actively managed budget, not a dumping ground.

**Learn more:** [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>18. Walk through the main prompting techniques and when to use each.</b></summary>

Zero-shot (just the instruction) is the default and often enough for capable models. Few-shot adds a handful of input-output examples to pin down format or a tricky pattern the instruction alone does not convey. Chain-of-thought asks the model to reason step by step, which helps on multi-step problems, though modern reasoning models do this internally so you should not force it on them. Structured techniques (output schemas, role framing, decomposition into sub-prompts) improve reliability. Choose the least complex technique that clears your quality bar, and validate the choice with an eval.

**Learn more:** [Prompting Guide](https://www.promptingguide.ai/)

</details>

<details>
<summary><b>19. How do you get reliable structured output (for example JSON) from an LLM?</b></summary>

Prefer native mechanisms over prompt-and-pray: use the provider's structured-output or tool-calling feature with a JSON schema so the model is constrained to valid shapes, and validate the result against that schema in code. When constrained decoding is unavailable, give an explicit schema and one example in the prompt, set a low temperature, and parse defensively. Always handle the failure path: a retry with the validation error fed back, or a fallback value. Do not extract fields with brittle regex or string slicing from free-form prose.

**Learn more:** [OpenAI: Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs)

</details>

<details>
<summary><b>20. What is "lost in the middle" and how do you design around it?</b></summary>

Models retrieve information best when it sits near the start or the end of the context and worst when it is buried in the middle of a long prompt. So blindly stuffing dozens of retrieved chunks in arbitrary order wastes them. Design around it by retrieving fewer, higher-quality chunks, reranking so the most relevant content is placed at the edges, and keeping the total context tight. This is one reason "more context" is not automatically "better answers."

**Learn more:** [Lost in the Middle (Liu et al.)](https://arxiv.org/abs/2307.03172)

</details>

<details>
<summary><b>21. How do you debug a prompt that works most of the time but fails intermittently?</b></summary>

First reproduce and categorize the failures against a small labeled set instead of eyeballing one bad output. Isolate the cause: is it ambiguous instructions, missing context, a formatting issue, temperature-driven variance, or genuinely hard inputs. Fix the highest-leverage cause (tighten the instruction, add a schema, lower temperature, improve retrieval, add a targeted example) and re-run the eval to confirm the failure rate dropped without regressing other cases. Treat prompts like code: change one thing, measure, keep a regression set.

**Learn more:** [Prompting topic](../../../topics/prompting.md)

</details>

<details>
<summary><b>22. What is few-shot prompting's cost, and when should you avoid it?</b></summary>

Every example you include is input tokens on every call, so few-shot raises cost and latency and eats context budget. Avoid or trim it when a capable model already handles the task zero-shot, when the examples are long, or when you can move the pattern into a fine-tune or a schema instead. If you do use it, choose examples that cover the tricky and edge cases rather than easy ones, and check whether 2 well-chosen examples beat 8 mediocre ones. With prompt caching, a stable few-shot block can be cheaper on repeated calls, which changes the tradeoff.

**Learn more:** [Prompting topic](../../../topics/prompting.md)

</details>

<details>
<summary><b>23. How do you manage conversation history in a long chat without blowing the context budget?</b></summary>

You cannot keep appending forever, so you compress. Common tactics: keep the last few turns verbatim, summarize older turns into a running memory, and store durable facts (user preferences, decisions) in a structured memory you retrieve as needed rather than replaying the whole transcript. Prune aggressively and put the most relevant recent content near the end where the model uses it best. The goal is to preserve the information that changes the next response, not the literal history.

**Learn more:** [Memory in agents (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md)

</details>

<details>
<summary><b>24. What is context rot, and how does it change how you build long-running agents?</b></summary>

Context rot is the finding that model accuracy degrades as the input grows, even when every relevant fact is still present and the window is not full, so a longer context is not a free upgrade. Chroma's research showed the drop across every major model, which is why a 200k-token window does not mean you should fill it. For long-running agents this pushes you toward active context management: compact or summarize old turns, evict stale tool output, isolate subtasks into separate contexts, and retrieve just the slice a step needs. The design goal shifts from "fit everything" to "keep the working context small, relevant, and fresh."

**Learn more:** [Chroma: Context Rot](https://research.trychroma.com/context-rot)

</details>

<details>
<summary><b>25. What is context compaction, and when do you reach for it?</b></summary>

Compaction is periodically replacing a long, accreted context (many turns of messages and verbose tool results) with a shorter summary that preserves the decisions, facts, and open goals the agent still needs. You reach for it in long-horizon agent loops that would otherwise exhaust the window or suffer context rot, and in multi-turn chats where old detail no longer changes the next action. The risk is dropping a detail that mattered, so you compact deliberately: keep recent turns and key artifacts verbatim, summarize the rest, and store anything durable in external memory you can re-retrieve. Done well it cuts cost and latency and improves reliability by removing distracting noise.

**Learn more:** [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>26. What is the difference between system, developer, and user roles, and why does it matter for injection safety?</b></summary>

Chat models take messages tagged by role, and the model is trained to weight them differently: the system (or developer) message carries durable instructions and policy, while user messages carry the request. Putting your rules in the system role gives them more standing than text buried in user content, which matters for both steerability and security. The critical safety point is that retrieved documents, tool outputs, and web content are data, not instructions, so they should never be placed where the model treats them as authoritative commands. Keeping trusted instructions and untrusted data in clearly separated roles is a first-line defense against prompt injection.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>27. Why structure a prompt with delimiters or tags, and how does it help reliability?</b></summary>

Clear delimiters (XML-style tags, fenced sections, explicit headers) tell the model where the instructions end and the data begins, which reduces ambiguity, makes it harder for injected text to blend into your instructions, and makes outputs easier to parse. Structure also helps you and the model track a long prompt: label the task, the context, the examples, and the output format as distinct blocks. It is not magic formatting, but consistent structure measurably improves adherence on complex prompts and is easy to test. Pick one convention and apply it everywhere so your prompts stay maintainable.

**Learn more:** [Prompting topic](../../../topics/prompting.md)

</details>

<details>
<summary><b>28. How do you treat prompts as versioned, testable artifacts rather than strings in code?</b></summary>

A production prompt is a dependency that changes behavior, so you manage it like code: keep it in version control, give each version an identifier, and tie every change to an eval run so you can see the quality delta before shipping. Separate the prompt template from the code that fills it, log which prompt version produced which output for debugging, and keep a regression set that any edit must pass. This lets you roll back a bad prompt, attribute a production regression to a specific change, and avoid the trap where someone "just tweaks the wording" and silently breaks a downstream case. Prompt engineering without versioning and evals is guesswork.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>29. What is prompt sensitivity, and how do you make a prompt robust to it?</b></summary>

Prompt sensitivity is when small, meaning-preserving changes (reordering examples, renaming a field, adding a stray newline) shift outputs more than they should, which makes a prompt that looks fine fragile in production. You reduce it by preferring constrained output over free-form parsing, giving explicit and unambiguous instructions, testing across paraphrases and input variations in your eval set, and avoiding over-reliance on a single lucky example ordering. Reasoning models and larger models tend to be less sensitive, but you still verify rather than assume. The senior move is to measure robustness deliberately, not to hand-tune a prompt until one demo passes.

**Learn more:** [Prompting topic](../../../topics/prompting.md)

</details>

<details>
<summary><b>30. When should you decompose a task into multiple prompts instead of one big prompt?</b></summary>

Decompose when a single prompt is trying to do several distinct jobs at once (extract, then reason, then format) and quality suffers because the model juggles them, or when different sub-steps want different models, temperatures, or validation. Breaking it into a small pipeline (or an agent) gives each step a focused context, lets you test and cache each stage, and localizes failures so you can see which step broke. The cost is more calls, more latency, and orchestration code, so you do not decompose gratuitously. Start with one prompt, and split only when an eval shows a specific step is the bottleneck.

**Learn more:** [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>31. What is meta-prompting or using a model to improve your prompts?</b></summary>

Meta-prompting is using an LLM to help write, critique, or refine prompts and rubrics: you describe the task and let a strong model draft the instruction, generate edge-case examples, or suggest where a prompt is ambiguous. It speeds up iteration and surfaces cases you would miss, and it pairs well with eval-driven development where the model proposes candidate prompts you then score. The caveat is that a model-suggested prompt is a hypothesis, not an answer, so you still validate it against a labeled set rather than trusting that it reads well. Treat it as a fast idea generator whose output you measure, not a replacement for evaluation.

**Learn more:** [Prompting topic](../../../topics/prompting.md)

</details>

---

## Retrieval and RAG

<details>
<summary><b>32. When would you use RAG instead of fine-tuning to give a model new knowledge?</b></summary>

Use RAG when the knowledge changes often, is large or proprietary, or needs citations: retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, tone, or to bake in a stable skill, not to inject volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior. Start with prompting plus RAG because it is cheaper and faster to iterate, and reach for fine-tuning only when prompting cannot hit the bar.

**Learn more:** [Agentic RAG 101 guide](../../../resources/agentic_rag_101.md)

</details>

<details>
<summary><b>33. Walk me through a basic RAG pipeline end to end.</b></summary>

Offline: ingest documents, clean and chunk them, embed each chunk, and store the vectors (plus metadata and the source text) in an index. Online: embed the user query, retrieve the top k nearest chunks (often with a hybrid of dense and keyword search), optionally rerank them, assemble a prompt that includes the query plus the retrieved context with instructions to answer only from it and cite sources, then generate. Around this you add guardrails (refuse when nothing relevant is retrieved) and evaluation on both retrieval and generation. Each stage is a tunable knob: chunking, embedding model, k, reranking, and prompt.

**Learn more:** [RAG roadmap](../../../resources/RAG_roadmap.md)

</details>

<details>
<summary><b>34. How do you choose a chunking strategy?</b></summary>

Chunking trades recall against precision: chunks too large dilute relevance and waste context, too small lose the surrounding meaning needed to answer. Start with structure-aware chunking (by section, paragraph, or semantic boundary) rather than a blind fixed character count, and add a small overlap so ideas that straddle a boundary are not lost. Match chunk size to your content and your embedding model's sweet spot, and keep metadata (source, title, section) attached for filtering and citations. Then validate: measure whether the answer-bearing chunk is actually retrieved on your eval set, and adjust.

**Learn more:** [Chroma: Evaluating chunking strategies](https://research.trychroma.com/evaluating-chunking)

</details>

<details>
<summary><b>35. What is the difference between dense, sparse, and hybrid retrieval?</b></summary>

Dense (embedding) retrieval matches on meaning, so it finds semantically related passages even without shared words, but it can miss exact terms, codes, or rare names. Sparse retrieval (BM25 and keyword methods) matches on lexical overlap, which is strong for exact terms and cheap, but blind to paraphrase. Hybrid retrieval runs both and fuses the results (for example with reciprocal rank fusion), capturing exact matches and semantic matches together, which usually beats either alone in production. The cost is added complexity, so justify it with an eval.

**Learn more:** [Weaviate: Hybrid search explained](https://weaviate.io/blog/hybrid-search-explained)

</details>

<details>
<summary><b>36. What does a reranker do, and when is it worth it?</b></summary>

A reranker is a second-stage model (often a cross-encoder) that scores each retrieved candidate against the query jointly, giving a much more accurate relevance ordering than the first-stage vector similarity. The pattern is retrieve many cheaply, then rerank to keep the best few, which raises precision and lets you pass fewer, better chunks to the model. It is worth it when first-stage retrieval returns roughly-relevant but poorly-ordered results and your generation quality is retrieval-limited. The tradeoff is added latency and cost per query, so measure the quality gain before committing.

**Learn more:** [Agentic search and retrieval research table](../../../research_updates/agentic_search_retrieval_table.md)

</details>

<details>
<summary><b>37. Why might a model answer worse with retrieved context than with none?</b></summary>

Because irrelevant, conflicting, or low-quality retrieved text can distract or mislead the model, and long noisy context degrades reasoning. Bad retrieval (wrong chunks, duplicated content, contradictory sources) is often worse than no retrieval at all. This tells you that retrieval quality and context construction matter as much as the model, that more context is not automatically better, and that you must evaluate the retrieval stage separately so you can see when it is the problem.

**Learn more:** [RAG topic](../../../topics/rag.md)

</details>

<details>
<summary><b>38. How do you evaluate a RAG system?</b></summary>

Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually retrieved, using recall and precision at k (and hit rate). For generation, measure faithfulness (is the answer grounded in the retrieved context with no fabrication), answer relevance (does it address the question), and correctness, using a labeled question set and an LLM-as-judge with a clear rubric. Include unanswerable questions to check that the system refuses instead of hallucinating, and monitor the same signals in production, not just offline.

**Learn more:** [Ragas documentation](https://docs.ragas.io/en/stable/)

</details>

<details>
<summary><b>39. What is the RAG triad?</b></summary>

The RAG triad is three complementary checks on a RAG answer: context relevance (was the retrieved context actually relevant to the query), faithfulness or groundedness (is the answer supported by that context without contradiction or fabrication), and answer relevance (does the answer address the user's question). Together they localize failures: low context relevance points at retrieval, low faithfulness points at the generation step ignoring or overriding the context, and low answer relevance points at the prompt or the model. Tools like Ragas operationalize these, but you still validate the judge against human labels.

**Learn more:** [AI evaluation 2025 research table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

<details>
<summary><b>40. What is agentic RAG, and when do you need it?</b></summary>

Agentic RAG puts an agent in charge of retrieval: instead of a single fixed retrieve-then-generate step, the model decides whether to retrieve, reformulates queries, retrieves iteratively, chooses among multiple sources or tools, and reasons over what it gets back. You need it for complex or multi-hop questions where one retrieval pass cannot gather everything, or where the right source depends on the question. The cost is more latency, more model calls, and new failure modes (bad query reformulation, loops), so use fixed RAG when it clears the bar and reserve agentic RAG for questions that genuinely require multi-step gathering.

**Learn more:** [What is RAG and agentic RAG (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part4_what_is_rag_and_agentic.md)

</details>

<details>
<summary><b>41. What is GraphRAG and what problem does it solve?</b></summary>

GraphRAG builds a knowledge graph of entities and relationships from your corpus and retrieves over that structure, sometimes with community summaries, instead of only over flat text chunks. It helps with global or multi-hop questions ("what themes connect these documents", "how does A relate to C through B") that plain chunk retrieval answers poorly because the relevant facts are scattered across many chunks. The tradeoff is a heavier, more expensive ingestion pipeline to build and maintain the graph. Reach for it when questions are relational or corpus-wide rather than answerable from a single passage.

**Learn more:** [GraphRAG (Microsoft Research paper)](https://arxiv.org/abs/2404.16130)

</details>

<details>
<summary><b>42. How does RAG change at enterprise scale?</b></summary>

The model stops being the bottleneck; the data and retrieval layer becomes it. You now need robust ingestion and freshness pipelines, access control so retrieval respects per-user permissions (a user must never retrieve a document they cannot see), cost and latency management across millions of queries, evaluation across many document types, and observability on retrieval quality. Metadata, incremental indexing, and permission-aware filtering matter more than any clever prompt. Design the data layer first, because that is where enterprise RAG usually breaks.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>43. How do you handle document freshness and updates in a RAG index?</b></summary>

Use incremental indexing keyed on document identity and version so you can upsert changed documents and delete removed ones rather than rebuilding the whole index. Track source timestamps and, where staleness is dangerous, expose the date in the answer or filter by recency. Decide a re-embedding policy for when the embedding model or chunking changes. Monitor for drift between the live corpus and the index, and treat freshness as an SLA, not an afterthought, since stale retrieval silently produces confidently wrong answers.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md)

</details>

<details>
<summary><b>44. A user asks something your corpus does not cover. What should the system do, and how do you make it do that?</b></summary>

It should refuse gracefully ("I do not have that information") rather than hallucinate from parametric knowledge. You engineer this by instructing the model to answer only from the retrieved context and to say it cannot find the answer when the context is insufficient, by setting a retrieval-score threshold below which you treat the query as unanswerable, and by including unanswerable cases in your eval set so you can measure the refusal rate. Getting this right is a top signal in take-homes, because it separates a grounded system from a confident fabricator.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>45. What is contextual retrieval, and what problem does it fix?</b></summary>

Standard chunking strips a chunk of its surrounding context, so "the revenue grew 3%" loses which company and which quarter it referred to, which hurts retrieval. Contextual retrieval, popularized by Anthropic, prepends a short model-generated description situating each chunk within its document before embedding and indexing it, so the chunk carries the context needed to be found. Reported results cut top-20 retrieval failures substantially, especially combined with BM25 and reranking. The tradeoff is a one-time ingestion cost to generate the context per chunk (mitigated by prompt caching), which is usually worth it for corpora where chunks are ambiguous out of context.

**Learn more:** [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

</details>

<details>
<summary><b>46. What is late chunking, and how does it differ from contextual retrieval?</b></summary>

Late chunking uses a long-context embedding model to embed the entire document first, then pools the token embeddings into chunk vectors, so each chunk embedding already reflects the whole document's context without adding any text. It differs from contextual retrieval, which prepends generated context to each chunk and re-embeds: late chunking is cheaper (no extra LLM calls) and purely an embedding-time technique, while contextual retrieval adds explicit descriptive text. Both target the same problem of context-poor chunks, and the reported benefit of late chunking grows with document length. Choose based on your embedding model's context limit and your ingestion budget, and validate on your own retrieval eval.

**Learn more:** [Jina AI: Late chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)

</details>

<details>
<summary><b>47. What is query rewriting or HyDE, and when does it help retrieval?</b></summary>

Real user queries are often short, underspecified, or phrased differently from the source documents, which hurts dense retrieval. Query rewriting reformulates or expands the query (fixing typos, adding synonyms, splitting a multi-part question) before retrieval, and HyDE (hypothetical document embeddings) has the model draft a hypothetical answer and embeds that, because a full pseudo-answer often sits closer to the real passages than the terse question does. These help most on conversational or vague queries and on corpora with a vocabulary mismatch. The cost is an extra model call and the risk of drifting off-topic, so measure whether it actually raises retrieval recall on your data.

**Learn more:** [HyDE (Gao et al.)](https://arxiv.org/abs/2212.10496)

</details>

<details>
<summary><b>48. How does metadata filtering improve retrieval, and how do you use it?</b></summary>

Metadata filtering restricts the candidate set before or during vector search using structured fields (date, author, document type, department, access tags), so the model only ever sees passages that are eligible. It sharpens precision (drop stale or out-of-scope documents), enables recency and permission constraints, and cuts wasted context. In practice you attach rich metadata at ingestion, then combine a metadata filter with the vector query, which most vector stores support natively. It is one of the cheapest, highest-leverage upgrades to a naive RAG system and is essential for access control at enterprise scale.

**Learn more:** [RAG roadmap](../../../resources/RAG_roadmap.md)

</details>

<details>
<summary><b>49. How do you choose a vector index (for example HNSW vs IVF vs flat) and does it matter?</b></summary>

A flat (exact) index compares the query against every vector, giving perfect recall but scaling poorly, so it is fine for small corpora or evaluation baselines. Approximate indexes trade a little recall for large speedups: HNSW (a navigable small-world graph) gives excellent query latency and recall at higher memory cost, while IVF (inverted-file clustering) is more memory-efficient and tunable but needs training and parameter care. The choice matters at scale for latency, memory, and cost, and each has knobs (ef, nlist, nprobe) that trade recall against speed. Start with the managed default your vector store recommends, then tune only if retrieval recall or latency becomes the bottleneck on your eval.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>50. With million-token context windows, is RAG obsolete? When do you still need it?</b></summary>

No. Even where a whole corpus fits in context, you still hit context rot (accuracy falls as input grows), high per-call cost and latency for stuffing everything every time, and no permission filtering or citation trail. RAG remains the way to select the relevant slice, respect access control, keep answers fresh via incremental indexing, and cite sources, all at a fraction of the tokens. Long context and RAG are complementary: long context lets you pass more retrieved evidence per call and simplifies chunking, but retrieval is still how you decide what deserves the budget. The honest answer is that long context changes the tradeoffs, it does not remove the need to choose what the model reads.

**Learn more:** [Chroma: Context Rot](https://research.trychroma.com/context-rot)

</details>

<details>
<summary><b>51. How do you do RAG over tables, PDFs, and images rather than clean text?</b></summary>

Real documents are messy, so ingestion quality dominates. Parse structure first: use layout-aware extraction for PDFs, preserve tables as structured rows rather than mangled text, and either OCR or use a vision model for scanned pages and figures. For multimodal RAG you can embed images and text into a shared space, or generate text descriptions of images and tables and retrieve over those, then pass the original when needed. Garbage extraction produces garbage retrieval regardless of the model, so invest in the parsing pipeline and evaluate retrieval on the hard document types specifically. Mature productized multimodal RAG is still emerging, so keep the pipeline simple and measured.

**Learn more:** [Multimodal topic](../../../topics/multimodal.md)

</details>

<details>
<summary><b>52. What is query routing across multiple indices or data sources?</b></summary>

When you have several corpora (product docs, code, tickets, a SQL database), a single index is rarely right, so a router first classifies the query and sends it to the appropriate source or set of sources, sometimes fanning out and merging results. Routing can be rules, a lightweight classifier, or an LLM deciding which tool or index to hit, which is the boundary where RAG shades into agentic retrieval. It raises precision and lets each source use the retrieval method that suits it (vector for prose, structured query for tables). The cost is added complexity and a new failure mode (misrouting), so log routing decisions and evaluate them.

**Learn more:** [Agentic RAG 101 guide](../../../resources/agentic_rag_101.md)

</details>

<details>
<summary><b>53. How do you build a golden evaluation set for a RAG system from real usage?</b></summary>

Start from real queries in your logs, not invented ones, because production questions are messier than anything you would write. Sample across the distribution (common, rare, ambiguous, and out-of-scope), then for each label the answer-bearing source passages and a reference answer or acceptance criteria, including cases that should be refused. Keep it small and high quality first (dozens before hundreds), version it, and grow it as production surfaces new failure modes. A noisy or unrepresentative eval set is worse than none, so the labeling discipline matters more than the size.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>54. How do you tune the retrieval score threshold that decides answerable vs unanswerable?</b></summary>

Treat it as a classification threshold you set with data, not a guess. Collect answerable and unanswerable queries, look at the distribution of top retrieval scores for each, and pick the cutoff that best separates them for your tolerance of false refusals versus hallucinations. Because raw vector similarities are not calibrated and vary by embedding model and corpus, you tune per system and re-check when either changes, and a reranker score is often a cleaner signal than first-stage similarity. Monitor the refusal rate in production so the threshold does not silently drift into refusing good queries or answering bad ones.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>55. How do you attach and verify citations so users can trust a RAG answer?</b></summary>

Carry a stable identifier (document, section, chunk) through retrieval into the prompt, instruct the model to cite the specific sources it used for each claim, and then verify rather than trust: check that cited chunks were actually retrieved and, for higher assurance, that the claim is supported by the cited text (an entailment or faithfulness check). Surface the citations in the UI so users can click through, which both builds trust and turns wrong answers into catchable ones. Unverified citations are a common failure where the model cites a real-looking source it did not use, so the verification step is what makes citations meaningful. This is the same faithfulness signal you measure in evaluation, applied at serving time.

**Learn more:** [RAG topic](../../../topics/rag.md)

</details>

---

## Agents and tool use

<details>
<summary><b>56. What makes something an agent rather than a single LLM call, and when do you actually need one?</b></summary>

An agent adds tools, memory, and a loop, so it can take actions, observe results, and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow (a predetermined chain of steps) will do, because agents add cost, latency, and new failure modes. The mature default is the simplest thing that works: prompt, then workflow, then agent only when the task genuinely requires dynamic control flow.

**Learn more:** [What are AI agents (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part1_what_are_ai_agents_anyway.md)

</details>

<details>
<summary><b>57. Explain the ReAct pattern.</b></summary>

ReAct interleaves reasoning and acting: the model produces a thought, chooses an action (a tool call), observes the result, then loops (think, act, observe, repeat) until it decides to finish. This lets it gather information and correct course mid-task instead of committing to a plan up front. It is the backbone of most tool-using agents. The practical risks are looping without progress and reasoning that drifts from the observations, which you bound with iteration caps, good tool design, and clear stopping criteria.

**Learn more:** [ReAct paper (Yao et al.)](https://arxiv.org/abs/2210.03629)

</details>

<details>
<summary><b>58. How does function calling / tool use work under the hood?</b></summary>

You describe your tools to the model as a schema (name, description, typed parameters). The model does not call anything itself; it emits a structured request naming a tool and its arguments, your code executes that tool, and you feed the result back into the context for the next step. Reliability depends on clear tool descriptions and well-typed parameters, and on handling the cases where the model calls the wrong tool, passes bad arguments, or the tool errors. Good agents reflect on a tool error and retry with a corrected call rather than crashing.

**Learn more:** [Anthropic: Tool use overview](https://docs.claude.com/en/docs/build-with-claude/tool-use/overview)

</details>

<details>
<summary><b>59. What is the Model Context Protocol (MCP) and why does it matter?</b></summary>

MCP is an open standard for connecting an agent to external tools and data sources through a uniform interface. It matters because you wire a capability once as an MCP server and reuse it across any MCP-compatible agent or harness, instead of writing a bespoke integration for every tool in every app. This decouples tool-building from agent-building and is becoming the common plug for the agent ecosystem. The tradeoff to watch is security: an agent connected to many MCP servers has a wide action surface, so permissions and trust boundaries matter.

**Learn more:** [What is MCP and why care (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part5_what_is_mcp_and_why_care.md)

</details>

<details>
<summary><b>60. How do you keep an agent from looping forever or running up cost?</b></summary>

Put hard limits in: a maximum number of steps or tool calls, a wall-clock or token budget, and a cost ceiling per task. Add progress checks so the loop stops when it is not advancing (repeated identical actions, no new information). Use cheaper models for routine sub-steps and a stronger one only where needed, cache stable context and tool results, and give the model only as much reasoning budget as the step requires. Design tools so failures return actionable errors the model can recover from rather than triggering blind retries.

**Learn more:** [Agents topic](../../../topics/agents.md)

</details>

<details>
<summary><b>61. What is agent memory, and what types are there?</b></summary>

Memory is how an agent carries information beyond a single turn or context window. Short-term (working) memory is the current context: recent messages, tool results, scratchpad. Long-term memory persists across sessions and is usually stored externally (a database or vector store) and retrieved on demand, covering durable facts, user preferences, and past outcomes. The engineering job is deciding what to write to long-term memory, how to retrieve the relevant slice into context, and how to keep it from growing stale or bloated. Memory is a context-engineering problem, not just a storage one.

**Learn more:** [Memory in agents (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md)

</details>

<details>
<summary><b>62. When is a multi-agent system worth it over a single agent, and what are the risks?</b></summary>

Multi-agent designs (a planner plus specialists, or agents that hand off) help when the work decomposes into distinct roles or parallelizable subtasks, or when separation of concerns improves reliability and keeps each agent's context focused. The risks are real: more cost and latency, harder debugging, error propagation and miscommunication between agents, and coordination overhead that can outweigh the benefit. Start with a single well-scoped agent and split into multiple only when a concrete limitation forces it. Communication protocols and clear contracts between agents matter as much as the prompts.

**Learn more:** [Multi-agent systems (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part8_multi_agent_systems.md)

</details>

<details>
<summary><b>63. What is the difference between a workflow and an agent, and why prefer workflows when you can?</b></summary>

A workflow is a predetermined sequence of steps you orchestrate in code (retrieve, then summarize, then classify); the control flow is fixed and predictable. An agent decides its own next step at runtime using the model, which is more flexible but less predictable and harder to test. Prefer workflows when the steps are known in advance because they are cheaper, faster, easier to debug, and easier to guarantee. Use an agent only when you genuinely cannot enumerate the path ahead of time.

**Learn more:** [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>64. How do you evaluate an agent, and why is average accuracy not enough?</b></summary>

Evaluate outcomes and trajectory: did it complete the task, and did it get there sensibly (right tools, valid arguments, faithful use of tool outputs, no wasted steps). Average accuracy hides unreliability, so measure consistency with metrics like pass^k (does it succeed on all k independent attempts), because an agent that works on average but fails unpredictably is not shippable. Use trace-based evaluation to inspect each step, and an LLM-as-judge on the trajectory for tool appropriateness and grounding, validated against human labels. Production autonomy demands reliability, not just a good mean.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>65. How would you design tools for an agent so it uses them reliably?</b></summary>

Treat tool design as interface design for a model: clear names, precise descriptions of when to use each tool, well-typed and minimal parameters, and return values that are informative but not bloated. Fewer, well-scoped tools beat a large sprawling set that confuses the model's selection. Make errors actionable so the model can self-correct, and consider consolidating multiple low-level calls into one higher-level tool that matches how the task is actually done. Then test tool selection and argument accuracy directly in your eval.

**Learn more:** [Tools in AI (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md)

</details>

<details>
<summary><b>66. What is the difference between planning-first and reactive (ReAct-style) agents?</b></summary>

A plan-and-execute agent generates a full plan up front, then executes the steps (optionally replanning), which gives structure, parallelism, and a visible plan you can inspect, and suits tasks whose shape is knowable in advance. A reactive ReAct-style agent decides one step at a time based on the latest observation, which adapts better to surprises but can wander or loop. Many real systems blend them: a rough plan to stay on track plus step-level reactivity to handle what the plan did not anticipate. Choose based on how predictable the task is and how much you need an auditable plan versus flexibility.

**Learn more:** [Planning in agents (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md)

</details>

<details>
<summary><b>67. What is reflection or self-critique in an agent, and when does it actually help?</b></summary>

Reflection has the agent critique its own output or trajectory and revise, for example checking whether an answer meets the requirements or whether a plan step failed, then trying again. It helps on tasks with a verifiable signal to reflect against (test results, a validator, a rubric, a tool error) where the second pass can genuinely catch mistakes. It helps less, and can waste tokens or talk the model out of a correct answer, when there is no real check and the model is just second-guessing itself. Use it where you can ground the critique in evidence, cap the number of revisions, and measure whether it improves your eval rather than assuming more reflection is better.

**Learn more:** [Agents topic](../../../topics/agents.md)

</details>

<details>
<summary><b>68. How do you manage context in a long-horizon agent that runs for many steps?</b></summary>

Long-horizon agents accumulate messages and verbose tool outputs that cause context rot and cost blowup, so you actively curate the working context rather than letting it grow. Techniques include compacting old turns into summaries, evicting or truncating stale tool results, writing durable facts to external memory and retrieving only what a step needs, and isolating subtasks into sub-agents with their own fresh context that return a short result. Anthropic's multi-agent research system, for instance, gives subagents isolated windows that return condensed summaries. The throughline is that the context is a managed budget across the whole run, not a transcript you append to forever.

**Learn more:** [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>69. What is the orchestrator-worker (sub-agent) pattern, and how do sub-agents communicate?</b></summary>

An orchestrator agent decomposes a task and delegates subtasks to worker sub-agents, each with its own isolated context and tools, then synthesizes their results. Isolation is the point: each worker keeps a small, focused context instead of one bloated shared window, which improves reliability and enables parallelism. The critical design decision is communication: sub-agents should pass state through well-defined interfaces (a short structured result or artifact), not by sharing raw traces, because dumping full transcripts between agents reintroduces the context bloat you split to avoid. Clear contracts between orchestrator and workers matter as much as the prompts.

**Learn more:** [Anthropic: Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

</details>

<details>
<summary><b>70. What is A2A (agent-to-agent) communication, and how does it relate to MCP?</b></summary>

MCP standardizes how one agent connects to tools and data (the vertical link between an agent and its capabilities), while agent-to-agent protocols standardize how independent agents discover, delegate to, and coordinate with each other (the horizontal link between peers, often across organizations). They are complementary layers: an agent might use MCP to reach its tools and an A2A protocol to hand a subtask to another team's agent. This matters as ecosystems of specialized agents emerge and need a common way to interoperate rather than bespoke glue. As with MCP, the open question is trust and security once agents can invoke each other across boundaries.

**Learn more:** [A2A Protocol](https://a2a-protocol.org/)

</details>

<details>
<summary><b>71. Why do agents fail on long tasks even when each step is likely to succeed?</b></summary>

Because errors compound multiplicatively: if each step is 95% reliable, a 20-step task succeeds only about 0.95^20, roughly 36% of the time, so high per-step accuracy still yields low task reliability. Failures also cascade, a wrong early observation or a bad tool result poisons every downstream step, and long contexts add rot on top. This is why average per-step accuracy is misleading and why you measure end-to-end task success and pass^k. The mitigations are fewer and more reliable steps, verification and recovery at each step, tight context management, and human checkpoints on the risky ones.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>72. How do you decide how much autonomy to give an agent (the agency-control tradeoff)?</b></summary>

More autonomy means the model makes more decisions itself (flexible, less code, but less predictable and harder to guarantee), while more control means you fix the path in code (reliable and testable, but rigid). You scale autonomy to the cost of a mistake and the difficulty of specifying the path: low-stakes, well-understood tasks can run more autonomously, while high-stakes or irreversible actions demand tight control and human approval. The senior instinct is to grant the least autonomy that accomplishes the task, and to earn more only as evaluation and monitoring prove the agent reliable in that scope. This tradeoff, not the model choice, is usually the real design decision in an agentic system.

**Learn more:** [Harness engineering guide](../../../resources/harness_engineering.md)

</details>

<details>
<summary><b>73. How does an agentic coding harness (like a coding agent) work, and what makes it reliable?</b></summary>

A coding harness wraps a model in a loop with tools to read and write files, run commands, execute tests, and search the codebase, so it can make a change, observe the result (compile errors, failing tests), and iterate. Reliability comes less from the model and more from the harness: good tool design, feeding real feedback (test output, type errors) back into the loop, keeping the working context focused on the relevant files, and bounding the loop. The verifiable signal (do the tests pass) is what makes coding a strong fit for agents, because the agent can ground its own reflection in results rather than guessing. This is a concrete instance of the general lesson that harness quality and grounded feedback drive agent reliability.

**Learn more:** [Anthropic: Claude Code best practices](https://www.anthropic.com/engineering/claude-code-best-practices)

</details>

<details>
<summary><b>74. How do you handle handoff from an agent to a human?</b></summary>

Design the handoff as a first-class path, not an error case: define the triggers (low confidence, high value or risk, an action the agent is not permitted to take, repeated failure), and package enough context for the human to act fast (the goal, what was tried, the current state, the specific decision needed). Preserve the ability to resume so the human's input feeds back into the loop rather than restarting. Log handoffs and their outcomes, because the handoff rate is a truth signal about where the agent is weak and where to invest. A clean handoff turns an agent's limitation into a safe, useful behavior instead of a silent failure.

**Learn more:** [Agents topic](../../../topics/agents.md)

</details>

---

## Evaluation

<details>
<summary><b>75. How do you evaluate an LLM feature that has no single correct answer (for example a summary or a draft reply)?</b></summary>

You define what "good" means as checkable criteria (faithful to the source, covers the key points, right tone and length, no fabrication) and score against those rather than against one golden string. Build a small high-quality labeled set of realistic and adversarial cases, then use an LLM-as-judge with an explicit rubric to score at scale, complemented by targeted human review. Report the criteria separately so you can see which dimension is weak. Version the eval set and treat it as a defensible artifact, not a one-off script.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>76. What is LLM-as-judge, and what are its failure modes?</b></summary>

LLM-as-judge uses a model to score or compare outputs against a rubric, which scales evaluation far beyond manual review. Its failure modes are well documented: position bias (favoring the first option), verbosity bias (favoring longer answers), self-preference (favoring outputs from the same model family), and inconsistency. You mitigate with a clear rubric, randomized option order, pairwise comparison where possible, and, most importantly, validating the judge against human labels before trusting it. A judge you have not calibrated against humans is a guess dressed up as a metric.

**Learn more:** [Judging LLM-as-a-judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)

</details>

<details>
<summary><b>77. What is reward hacking or eval gaming, and how do you guard against it?</b></summary>

It is when a system optimizes the measured proxy rather than the real goal, for example exploiting a judge's verbosity bias or a benchmark's shortcuts to score well without being better. Guard against it with held-out and adversarial evals, multiple diverse metrics or judges instead of one, and by checking whether gains transfer to independent tasks rather than a single benchmark. Watch for suspicious jumps and inspect what actually changed. If you can game your own eval in 5 minutes, so can the optimization process.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>78. What is the difference between offline and online evaluation, and why do you need both?</b></summary>

Offline evaluation runs against a fixed labeled dataset before you ship, giving a fast, repeatable signal you can gate releases on. Online evaluation measures the live system on real traffic (task success, user feedback, escalation rate, latency, cost, and sampled quality checks), catching distribution shift and failures your dataset never covered. You need both because a system can pass offline and still fail on real users, and offline sets go stale as usage evolves. Treat evaluation as release infrastructure spanning both.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>79. Why measure reliability such as pass^k for agents rather than just accuracy?</b></summary>

Because an agent that succeeds on average but fails unpredictably is not shippable for autonomous work. Pass^k asks whether the agent succeeds on all k independent attempts of the same task, capturing consistency, which is what production actually requires from a long-running or high-stakes task. A 90% average that means a 10% chance of a wrong action every run is very different from one that is reliably right. Reliability metrics surface the variance that a mean hides.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>80. How do you build an eval set from scratch for a new feature?</b></summary>

Start from what success means as a checkable outcome, then collect or write a small, high-quality set (dozens before hundreds) that covers realistic cases, edge cases, and adversarial ones, including cases that should be refused. Label it carefully, because a noisy eval set is worse than none. Split it so you can iterate on one part and hold out another to check you are not overfitting. Version it, document the labeling criteria, and grow it as production surfaces new failure modes.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>81. What benchmarks or metrics would you actually cite, and how much do you trust them?</b></summary>

Public benchmarks (reasoning, coding, retrieval, agentic task suites) are useful for a coarse first read and for tracking the field, but they are proxies that can be contaminated or gamed and rarely match your task distribution. Trust your own eval on your own data far more than any leaderboard. For RAG, cite retrieval recall/precision at k and faithfulness/answer-relevance; for agents, task completion and pass^k; for generation, task-specific rubric scores. The honest answer in an interview is that benchmarks orient you but your bespoke eval decides.

**Learn more:** [AI evaluation 2025 research table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

<details>
<summary><b>82. What is error analysis, and why do practitioners call it the highest-leverage eval activity?</b></summary>

Error analysis is reading a sample of real failures, labeling each with what actually went wrong, and clustering those labels into a taxonomy so you know which failure modes dominate. It is high-leverage because it tells you where to spend effort: chasing a metric without knowing why it is low leads to random fixes, while a failure taxonomy points straight at the biggest problem (bad retrieval vs missing refusal vs formatting vs a hard input class). It also generates targeted eval cases and reveals whether your metrics even capture the failures users care about. Do it early and repeatedly; it is the difference between measuring and understanding.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>83. What is eval-driven development, and how do you wire evals into CI?</b></summary>

Eval-driven development is the LLM-era successor to test-driven development: you define the eval set and acceptance bar first, and every change to a prompt, model, retriever, or tool is judged against it before it merges. Concretely, a pull request triggers an automated run of your judges over the golden dataset, and any regression below the baseline blocks the merge, the same way a failing unit test does. This turns "it looked better in a demo" into a measured decision and catches regressions before users do. It requires a stable, versioned eval set and a calibrated judge, which is exactly why those are worth building.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>84. How do you calibrate an LLM judge, and what agreement level should you target?</b></summary>

Calibration means checking the judge against a human-labeled reference set and measuring agreement (for example percent agreement or a correlation with human scores) before you trust the judge to gate anything. A common target is roughly 85 to 90% agreement with careful human labels; below that the judge's scores may correlate with nothing you care about. You improve agreement by tightening the rubric, adding few-shot examples of correct judgments, decomposing a vague score into specific checkable criteria, and controlling for known biases. Recalibrate when you change the judge model or the task, because an uncalibrated judge produces confident numbers that mislead.

**Learn more:** [Judging LLM-as-a-judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)

</details>

<details>
<summary><b>85. Pointwise vs pairwise judging: when do you use each?</b></summary>

Pointwise judging scores a single output against a rubric (say 1 to 5, or pass/fail per criterion), which is simple, gives an absolute signal, and suits CI gates and per-dimension diagnostics. Pairwise judging compares two outputs and picks the better one, which is more reliable for fuzzy quality where absolute scores are noisy and is the natural fit for comparing two models or prompts. Pairwise sidesteps the difficulty of calibrating an absolute scale but does not tell you if both options are bad, and it needs order randomization to avoid position bias. Use pointwise for gating and dashboards, pairwise for head-to-head model or prompt selection.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>86. How do you evaluate a multi-turn conversational agent, not just single responses?</b></summary>

Single-turn scoring misses what matters in a conversation: whether the agent reached the user's goal across turns, stayed consistent, recovered from misunderstandings, and did not lose earlier context. So you evaluate at the trajectory level, defining the task and a contract (the required outcome plus behavioral constraints checkable on the transcript) rather than one golden reply. You generate trajectories by simulating a user against the live agent over several trials per task and score the whole transcript, often with pass^k for reliability. This is closer to how the agent actually behaves and catches multi-turn failure modes that per-message evals never see.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>87. What is drift, and how does online evaluation catch it that latency monitoring cannot?</b></summary>

Drift is a slow degradation in quality that leaves infrastructure metrics untouched: a provider updates the model, your input distribution shifts, or the corpus goes stale, and faithfulness or task success falls while latency, error rate, and uptime all look fine. Latency and error monitoring cannot see it because nothing crashes or slows; the answers just get worse. You catch it by scoring a sample of live traffic continuously (commonly 1 to 5%) with your judges and alerting when a quality metric drops, for example a multi-point faithfulness fall over a week. Quality is a first-class production signal, so you monitor it, not just the system health.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>88. How do you run an A/B test for an LLM feature, and what do you measure?</b></summary>

Split live traffic between the variants (a new prompt, model, or retriever versus the current one) and compare outcomes that matter to users and the business, not just an offline score: task success or resolution rate, user acceptance or edit rate, escalation rate, latency, and cost per interaction. Guard for the LLM-specific traps: non-determinism adds noise so you need enough sample size, quality is a distribution so watch the tail not just the mean, and a change can improve one segment while hurting another. Pair the online test with your offline eval, which gates the change before it ever reaches the A/B split. The offline eval says "safe to try," the A/B test says "actually better for users."

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>89. Why do public benchmarks saturate or mislead, and what is contamination?</b></summary>

Contamination is when benchmark questions (or close variants) leak into a model's training data, so high scores reflect memorization rather than capability, which is common as benchmarks age and circulate on the web. Saturation is when frontier models cluster near the ceiling, so the benchmark no longer separates them and small differences are noise. Both mean a leaderboard number can be a poor predictor of performance on your task, especially for anything the benchmark did not test. This is why you build a private, task-specific eval that cannot leak and that measures what your product actually needs, and treat public scores as a coarse orientation only.

**Learn more:** [AI evaluation 2025 research table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

---

## Reasoning models

<details>
<summary><b>90. How do reasoning models differ from standard LLMs, and how does that change how you prompt them?</b></summary>

Reasoning models are trained to spend test-time compute on an internal chain of thought before answering, often via reinforcement learning on verifiable rewards, so they perform much better on math, code, and multi-step problems. They trade latency and cost for that: a single call can take seconds and burn many hidden reasoning tokens. Prompting changes: give the goal, constraints, and context, and let the model do the step-by-step work rather than forcing your own chain-of-thought or over-specifying the method. You also expose a reasoning-effort or budget control where available.

**Learn more:** [Planning and reasoning models (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md)

</details>

<details>
<summary><b>91. When should you use a reasoning model versus a standard model?</b></summary>

Use a reasoning model for genuinely hard, multi-step tasks where a fast model fails: complex planning, tricky math or logic, hard debugging, or agentic steps that need careful deliberation. Use a standard (and cheaper, faster) model for routine work: extraction, classification, summarization, straightforward RAG answers, and most tool calls. The right architecture often routes: a cheap model handles the common path and escalates only the hard cases to a reasoning model. Decide with an eval and a cost-latency budget, not by defaulting to the biggest model.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>92. What are the practical downsides of reasoning models in production?</b></summary>

They are slower and more expensive per call because of the hidden reasoning tokens, which strains latency budgets and cost ceilings, especially inside an agent loop where each step already takes 1 to 3 seconds. Their variable output length makes latency less predictable. Their internal reasoning is not always faithful to the true cause of the answer, so you cannot treat the visible chain of thought as a reliable explanation. Manage them with routing, budgets, caching, and streaming, and reserve them for steps that actually need the extra compute.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>93. What is a thinking budget, and how do you tune it?</b></summary>

A thinking budget is a control (for example a token allowance for the reasoning phase) that decides how much test-time compute the model spends before answering, turning "thinking" into a dial rather than a fixed mode. You raise it for hard math, complex analysis, and tricky debugging where more deliberation genuinely helps, and lower it (or use a standard model) for routine work to save latency and cost. Tune it empirically against your eval: find the point where more budget stops improving accuracy for a task class, because beyond that you are just paying for tokens. The budget is one of your main cost-latency levers when you commit to a reasoning model.

**Learn more:** [Anthropic: Extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)

</details>

<details>
<summary><b>94. What is overthinking or inverse scaling in test-time compute?</b></summary>

Overthinking is when more reasoning makes a model worse, not better: on some problems, especially easier ones, longer chains lead it to abandon a correct answer, talk itself into a wrong one, or degrade confidence calibration. Research on inverse scaling in test-time compute documents cases where accuracy falls as the reasoning budget grows. The practical lesson is that "more thinking" is not free capability, so a reasoning model is not universally better and a fixed maximal budget can hurt. You match the budget (and the choice of reasoning vs standard model) to task difficulty and verify with an eval rather than maxing out deliberation by default.

**Learn more:** [Inverse Scaling in Test-Time Compute](https://arxiv.org/abs/2507.14417)

</details>

<details>
<summary><b>95. What is interleaved thinking across tool calls?</b></summary>

Interleaved thinking lets a reasoning model deliberate not just once before answering but between tool calls: it reasons, calls a tool, sees the result, reasons about that result, then decides the next action. This matters for agents because the hard part is often reacting to what a tool returned (an error, an unexpected value), and interleaved reasoning lets the model incorporate observations into its next step instead of committing to a plan blindly. It combines the strengths of reasoning models and the ReAct loop, at the cost of more reasoning tokens per step. Use it for agentic tasks where mid-trajectory judgment matters, and budget it because the token cost adds up across steps.

**Learn more:** [Anthropic: Extended thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)

</details>

<details>
<summary><b>96. Can you trust a reasoning model's visible chain of thought as an explanation?</b></summary>

Not fully. The visible reasoning trace reads like an explanation, but research shows a model's stated chain of thought is not always faithful to the actual computation that produced the answer, so it can rationalize rather than reveal. Practically this means you should not use the chain of thought as an audit trail for high-stakes decisions or assume that fixing the visible reasoning fixes the behavior. Treat it as a useful debugging aid and a driver of better answers, but verify outcomes with evals and ground high-stakes decisions in checkable evidence, not the narrative. Faithfulness of reasoning is an open research area, not a solved guarantee.

**Learn more:** [Anthropic: Tracing the thoughts of a language model](https://www.anthropic.com/research/tracing-thoughts-language-model)

</details>

<details>
<summary><b>97. What is self-consistency, and how does it relate to reasoning models?</b></summary>

Self-consistency samples several independent reasoning paths for the same problem and takes the majority answer, which improves accuracy on tasks with a checkable final answer because errors in individual paths cancel out. It is a test-time compute technique you can apply on top of any model, and it predates built-in reasoning models, which internalize some of this benefit through training. The tradeoff is cost: you pay for several full generations per query, so you reserve it for high-value problems where the accuracy gain justifies the multiplied token spend. With a reasoning model you often get diminishing returns from stacking self-consistency on top, so measure before adding it.

**Learn more:** [Chain-of-thought prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)

</details>

---

## Cost, latency, and deployment

<details>
<summary><b>98. How would you keep an LLM feature's cost and latency under control?</b></summary>

Cache the stable prefix of the context and reuse the KV cache so repeated prompts are cheaper and faster, keep context minimal and well-ordered, and use smaller or cheaper models for sub-steps, escalating only when needed. Cap tool calls and loop iterations, cache tool and retrieval results, stream tokens so perceived latency drops even when total time does not, and batch where throughput matters. Give reasoning models only as much budget as the step needs. Every one of these is a dial you set per use case against a measured budget.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>99. Explain the latency-throughput-cost triangle and the main dials you have.</b></summary>

You are trading three things: how fast a single request returns (latency), how many requests you can serve per second (throughput), and dollars per request (cost). The main dials are model size (smaller is faster and cheaper but weaker), batching and continuous batching (raise throughput at some latency cost), KV-cache reuse and prompt caching (cut cost and time on repeated context), context length (shorter is cheaper and faster), and streaming (improves perceived latency). You cannot max all three, so you name the binding constraint for the use case and tune toward it.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>100. What is prompt caching / KV caching and when does it help?</b></summary>

During generation the model computes and stores key-value tensors for the tokens it has processed; reusing that KV cache avoids recomputing the shared prefix on later calls. Prompt caching exposes this so a long stable system prompt, tool schema, or retrieved context that repeats across requests is billed and processed at a large discount. It helps most when many requests share a long common prefix (a fixed system prompt, few-shot block, or the same document). Order your context so the stable, cacheable part comes first and the variable part last.

**Learn more:** [Anthropic: Prompt caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)

</details>

<details>
<summary><b>101. How do you decide which model to use for a given step (model routing)?</b></summary>

Match model capability to task difficulty and constraints: use the smallest, cheapest, fastest model that clears the quality bar for that step, and escalate to a stronger or reasoning model only for the hard cases. Implement this as a router (rules or a lightweight classifier) so the common, easy path stays cheap and only the tail pays for the big model. Validate the routing with an eval so you know the cheap model actually holds quality where you use it. Revisit the choice as models get cheaper and better.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>102. How does quantization affect an AI Engineer's decisions, even if you do not train models?</b></summary>

Quantization shrinks a model (for example to 4-bit or 8-bit) to cut memory and inference cost at some accuracy loss, which matters when you self-host or run at the edge. The impact is task-dependent, so if you use a quantized served model you measure the quality drop on your own eval before shipping rather than assuming it is fine. It also lowers latency and lets bigger models fit on smaller hardware. You rarely quantize yourself in this role, but you should know it as a cost-quality lever and test its effect.

**Learn more:** [Fine-tuning 101 guide](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>103. What does observability look like for an LLM application in production?</b></summary>

You trace every request end to end: the prompt and context assembled, retrieval hits and scores, each tool call and its result, the model output, tokens, latency per stage, and cost. Tools like tracing dashboards let you inspect a full multi-step agent trajectory to see where it went wrong. You log for auditability, sample outputs for ongoing quality checks, and alert on drift in the metrics that matter (hallucination signals, refusal rate, latency, cost, error rate). Without this you are flying blind on a non-deterministic system, so build it in from the start.

**Learn more:** [LangSmith documentation](https://docs.smith.langchain.com/)

</details>

<details>
<summary><b>104. What is the difference between time-to-first-token and inter-token latency, and why care?</b></summary>

Time-to-first-token (TTFT) is how long before the first output token appears, dominated by the prefill phase over your input, so it grows with prompt length. Inter-token latency (or tokens per second during decode) is how fast tokens stream after that, dominated by the memory-bound decode phase and roughly independent of input size. They matter because they call for different fixes: a slow TTFT is helped by shortening or caching the prompt, while slow token throughput is helped by a smaller or faster model, and streaming hides TTFT for chat but not for a batch job that needs the whole output. Optimize the one your user experience is actually bound by rather than "latency" as a single number.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>105. What is semantic caching, and how is it different from prompt caching?</b></summary>

Prompt caching reuses the model's computation for an exact shared prefix of tokens, cutting cost and TTFT on repeated context but still calling the model. Semantic caching instead checks whether a new query is semantically similar to a previously answered one (via embeddings) and, on a hit, returns the stored answer without any model call, saving the whole request. It shines in high-repetition workloads like FAQs, but it is riskier: a near-miss can serve a subtly wrong answer, so you set a similarity threshold, sometimes verify, and scope it to queries where reuse is safe. Teams often layer exact, semantic, and prefix caches and instrument the hit rate of each.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>106. What is speculative decoding, and why might it matter to you?</b></summary>

Speculative decoding uses a small fast "draft" model to propose several tokens ahead, which the large target model then verifies in a single pass, accepting the correct ones, so you get the big model's quality at lower latency because it generates fewer sequential steps. It is a serving-layer optimization, so you mostly benefit from it transparently through your provider or inference stack rather than implementing it, but knowing it explains why some fast endpoints match a bigger model's output. It helps latency and throughput without changing output quality, since the target model still validates every token. Understanding it lets you reason about why hosted latency varies and what a self-hosting stack can tune.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>107. What is continuous (in-flight) batching, and how does it change throughput?</b></summary>

Naive batching waits to group requests and runs them in lockstep, so a long generation holds up short ones and the GPU sits idle. Continuous batching schedules at the token level, adding new requests and retiring finished ones every step, which keeps the GPU busy and can multiply throughput for a shared endpoint. The tradeoff is a little added latency for an individual request under load, and it interacts with KV-cache memory since concurrent sequences all hold cache. You mostly get it from the serving framework rather than building it, but it explains why hosted throughput and latency depend heavily on load and why self-hosting economics hinge on batching well.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>108. API prices have dropped sharply. How should that change your architecture decisions?</b></summary>

Falling prices mean yesterday's cost-driven compromises may no longer be necessary, so you revisit them: a task you split across cheap models or heavily prompt-engineered to fit a small model might now run better on a stronger model within budget, and elaborate cost-saving scaffolding can become net-negative complexity. It also means you should avoid over-optimizing prematurely and keep the architecture flexible (a provider abstraction, a router) so you can adopt a cheaper or better model when it lands. At the same time, cheaper tokens invite scale that makes total spend grow, so you still monitor cost per interaction. The senior instinct is to treat model choice as a periodically re-evaluated decision, not a permanent one.

**Learn more:** [Business and judgment: Model provider dependence](#business-and-judgment)

</details>

<details>
<summary><b>109. How do you attribute and monitor LLM cost per feature or per user?</b></summary>

You instrument every call with metadata (feature, user or tenant, model, cached vs uncached tokens) and log input and output token counts so cost is a queryable dimension, not a single monthly invoice. That lets you see which feature or which heavy user drives spend, catch a regression where a prompt change doubled tokens, and set per-tenant budgets or rate limits. Track cost per successful interaction rather than per call, because that ties spend to value and reveals when an expensive path is not paying off. Cost is a product metric at scale, so you monitor and alert on it like latency or errors.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>110. When do you self-host an open-weight model instead of using a hosted API?</b></summary>

Self-hosting makes sense when data residency or zero-retention requirements rule out sending data to a provider, when scale makes per-token API pricing more expensive than running your own hardware, when you need a fine-tuned or specialized model the APIs do not offer, or when you need latency and availability guarantees a shared endpoint cannot promise. The costs are real: GPU capacity, serving expertise (batching, KV-cache management, autoscaling), and ongoing maintenance, so a hosted API usually wins until one of those drivers dominates. A common middle path is a hosted open-weight endpoint that avoids the ops burden while giving model choice. Decide with a concrete comparison of total cost and requirements, not ideology.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>111. How do you set up a safe fallback when a model call fails, times out, or is rate-limited?</b></summary>

Assume every model call can fail and design the unhappy path first: timeouts with retries and exponential backoff for transient errors, a circuit breaker so you stop hammering a failing provider, and a defined degraded behavior (a smaller or alternate model, a cached answer, or an honest "try again shortly"). Never let a provider outage take down the whole feature or hang the user, and never silently return a broken structured output, validate and fall back. Log failures and their fallbacks so you can see provider reliability and tune limits. A provider abstraction makes swapping to a backup model or vendor cheap when you need it.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

---

## Safety and responsible AI

<details>
<summary><b>112. What is prompt injection, and how is indirect injection different and worse?</b></summary>

Prompt injection is when input text overrides your instructions and makes the model do something unintended. Direct injection comes from the user typing a malicious prompt. Indirect injection is more dangerous: the malicious instructions hide inside content the model ingests from elsewhere (a retrieved document, a web page, a tool result, an email), so the attack rides in through data the user never sees and the model treats as trusted. It is worse because your attack surface is every external source your RAG or agent touches, and the user is not the attacker.

**Learn more:** [Simon Willison: Prompt injection series](https://simonwillison.net/series/prompt-injection/)

</details>

<details>
<summary><b>113. How do you defend an agentic system against prompt injection and misuse?</b></summary>

There is no single fix, so you layer defenses: give tools least privilege and scope so a compromised step cannot do much, require human approval for sensitive or irreversible actions, separate trusted instructions from untrusted data in the context, and add input and output guardrails to filter obvious attacks and unsafe outputs. Sandbox tool execution, log everything for audit, and constrain what the agent can reach. Assume some injection will get through and design so the blast radius is small rather than betting on perfect detection.

**Learn more:** [Securing agentic AI systems guide](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>114. What are the main security concerns specific to agents with real-world tool access?</b></summary>

The big ones are indirect prompt injection from retrieved or tool content, over-broad tool permissions and privilege escalation, data exfiltration (the agent leaking sensitive data through a tool or its output), and unsafe or irreversible actions taken autonomously. The wider the agent's action surface (more tools, more MCP servers, more autonomy), the larger the risk. Mitigate with least-privilege tools, human-in-the-loop for high-stakes actions, strict input and output validation, and audit logging. Treat every external input as untrusted.

**Learn more:** [Securing agentic AI systems guide](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>115. How do you handle PII, data privacy, and compliance in an LLM product?</b></summary>

Keep data access least-privilege and permission-aware end to end, so retrieval and memory only ever surface what a given user is allowed to see. Be deliberate about what leaves your environment and what a model provider may log or retain, and use zero-retention or self-hosted options where the data demands it. Redact or minimize PII before it hits the model where you can, log for auditability, and honor deletion and consent requirements in both your index and your memory stores. Design the data boundary first; it is easier than retrofitting compliance.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>116. How do you roll out an LLM feature safely?</b></summary>

Gate the release on an evaluation suite built around how the system actually breaks, start narrow with a human in the loop and a small user slice, and add monitoring and guardrails before you widen. Expand only as the metrics hold, keep a fast rollback, and use canaries or A/B splits so a regression hits few users. Watch for silent regressions after a model-provider update, since the underlying model can change under you. Treat evals and monitoring as release infrastructure, not a one-time check.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>117. A provider updates its model and your feature quietly regresses. How do you catch and handle it?</b></summary>

You catch it because you have a regression eval that runs on a schedule (or before you adopt a new model version) and production monitoring on quality signals that would alarm on a drop. When it fires, you pin to the previous known-good model version if the provider allows it, reproduce the regression on your eval, adjust prompts or context if the fix is cheap, and only then move to the new version. This is why pinning model versions, keeping a regression set, and monitoring live quality are not optional for a production LLM feature.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>118. What is tool poisoning in MCP, and how do you defend against it?</b></summary>

Tool poisoning is an attack where a malicious MCP server embeds hidden instructions in a tool's metadata (its name, description, or parameter docs), which the agent reads and treats as trusted, steering it into unsafe actions or leaking data, often without the user seeing the tainted description. It is a specific, dangerous form of indirect prompt injection because tool descriptions are loaded into context by design and are easy to overlook. Defenses include only connecting vetted and pinned MCP servers, reviewing tool descriptions rather than trusting them blindly, isolating and least-privileging each server, requiring approval for sensitive actions, and monitoring for unexpected tool behavior. Every connected server widens the trust boundary, so treat third-party MCP servers as untrusted code.

**Learn more:** [Securing agentic AI systems guide](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>119. What is the OWASP Top 10 for LLM applications, and why reference it?</b></summary>

It is a community-maintained list of the most critical security risks specific to LLM applications (prompt injection, insecure output handling, sensitive information disclosure, excessive agency, supply-chain and data-poisoning risks, and more), giving you a shared checklist and vocabulary for threat modeling. Referencing it in an interview or a design review signals that you think about security systematically rather than ad hoc, and it maps cleanly onto agent risks like over-broad tool permissions (excessive agency) and injection. Use it as a coverage checklist when designing an agentic or RAG system, then apply the layered mitigations for the risks that actually apply. It is a starting framework, not a guarantee, so you still test.

**Learn more:** [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

</details>

<details>
<summary><b>120. What are input and output guardrails, and what belongs in each?</b></summary>

Guardrails are the validation layer around the model: input guardrails inspect the request before it reaches the model (detect prompt injection and jailbreak patterns, block disallowed topics, strip or flag PII, enforce length and format), while output guardrails validate the response before it reaches the user or a tool (schema and grounding checks, PII and toxicity filters, policy and safety classifiers, refusing to execute an unsafe action). Some checks are cheap rules and regexes, others are small classifier models or an LLM judge, and you layer them by risk. The point is that the model is probabilistic, so you enforce hard constraints in deterministic code around it rather than hoping the prompt holds. Log what guardrails catch so you can tune false positives and see emerging attacks.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>121. What is the difference between a jailbreak and prompt injection?</b></summary>

A jailbreak tries to get the model to violate its own safety training and produce content it should refuse (the target is the model's guardrails), while prompt injection tries to override the developer's instructions and hijack the application's behavior (the target is your prompt and your agent). They overlap in technique but differ in intent and defense: jailbreak resistance is largely the provider's training plus your output filtering, whereas injection defense is your architecture, separating instructions from untrusted data, least-privilege tools, and human approval. An agent can be injected without any policy-violating content, simply doing the attacker's bidding with legitimate-looking actions. Knowing the distinction keeps you from assuming a safe model means a safe application.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>122. What is excessive agency, and how do you prevent an agent from doing too much?</b></summary>

Excessive agency is when an agent has more capability, permission, or autonomy than the task requires, so a mistake or an injection can cause outsized harm (deleting data, sending money, emailing customers). You prevent it by scoping tools to the minimum needed, using read-only or narrowly-scoped credentials, requiring human approval for irreversible or high-value actions, and putting hard limits and blast-radius controls around what the agent can touch. Design so the worst case of a compromised or confused agent is contained, rather than trusting it to always choose correctly. This is the OWASP-named risk that most directly maps to the agency-control tradeoff in agent design.

**Learn more:** [Securing agentic AI systems guide](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>123. What is red-teaming for an LLM feature, and how do you fold it into development?</b></summary>

Red-teaming is deliberately attacking your own system to find failures before users or adversaries do: crafting jailbreaks and injection payloads, probing for PII leakage, pushing the agent toward unsafe actions, and hunting for the inputs that break your guardrails. You fold it into development by turning discovered failures into permanent regression cases in your eval set, so a fixed vulnerability stays fixed, and by red-teaming again after any model, prompt, or tool change. It complements benchmark-style evals because it targets the adversarial tail rather than the average case. For higher-stakes systems this becomes an ongoing program, not a one-time audit, since new attacks and model updates keep the surface moving.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

---

## System design

<details>
<summary><b>124. Design a customer support agent for an enterprise. Walk me through it.</b></summary>

Start from requirements: accuracy bar, latency budget, escalation policy, cost ceiling, and compliance. Use retrieval over the company's knowledge base for grounding, tools for actions (look up an order, file a ticket) with least privilege, and a clear handoff to a human for low-confidence or high-stakes cases. Wrap it in evaluation and observability, add input and output guardrails (including against indirect injection from retrieved content), and roll out behind a fallback with a human in the loop. Name the tradeoff at every choice, and reach for an agent loop only where fixed retrieval will not do.

**Learn more:** [Real-world agentic systems (Agentic AI Crash Course)](../../../free_courses/agentic_ai_crash_course/part9_real_world_agentic_systems.md)

</details>

<details>
<summary><b>125. Design a RAG system over 10 million internal documents with per-user access control.</b></summary>

The data layer is the hard part, not the model. Build an ingestion pipeline that chunks, embeds, and indexes documents incrementally with rich metadata including access-control tags, and keep it fresh with upserts and deletes. At query time, filter retrieval by the requesting user's permissions before ranking so nobody ever retrieves a document they cannot see, then use hybrid retrieval plus reranking to keep precision high at that scale. Add caching for cost and latency, evaluate retrieval quality across document types, and monitor freshness and retrieval metrics. Access control and freshness are where enterprise RAG breaks, so design them first.

**Learn more:** [RAG roadmap](../../../resources/RAG_roadmap.md)

</details>

<details>
<summary><b>126. Design the evaluation and monitoring system for an LLM feature already in production and getting hallucination complaints.</b></summary>

First reproduce the complaints and build a labeled set from them, split into faithfulness, retrieval, and refusal failures, so you know which stage is at fault. Stand up offline evals (RAG triad plus task correctness with a human-validated judge) that gate any change, and online monitoring (sampled faithfulness checks, refusal rate, user feedback, citation coverage) that alarms on drift. Add tracing so you can inspect the retrieval and generation for any flagged response. Then fix the dominant failure mode (usually retrieval or a missing refusal path) and confirm the metric moves without regressing others.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>127. Design an insurance-claims agent that ingests a claim and outputs approve, deny, or needs-review.</b></summary>

Requirements first: the accuracy and auditability bar is high and errors are costly, so this is a human-in-the-loop system by design. Use retrieval over policy documents and prior claims to ground each decision, structured tool calls to pull claim data, and a constrained output that returns a decision plus the cited evidence and a confidence signal. Route anything below a confidence threshold or above a value threshold to human review rather than auto-deciding, cap cost with a cheap model for triage and a stronger one only for hard claims, and add guardrails and full audit logging. Evaluate on labeled historical claims and monitor the approve/deny distribution for drift.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>128. How do you cut the p95 latency of a feature that makes one reasoning-model call per request?</b></summary>

Question whether every request needs the reasoning model: route so the common, easy cases go to a fast standard model and only hard cases escalate, which alone often halves p95. Stream the response so perceived latency drops, cache the stable prompt prefix and any repeated context via KV/prompt caching, trim the context to cut input processing time, and cap the reasoning budget where the provider allows. If the work decomposes, run independent sub-steps in parallel instead of serially. Measure p95 before and after each change rather than guessing.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>129. Design a code assistant that answers questions over a large private codebase with a 2-second latency budget.</b></summary>

Ingest the repo with code-aware chunking (by function, class, or file with symbol metadata) and index both dense embeddings and a symbol or keyword index, because exact identifier matches matter as much as semantic similarity, so hybrid retrieval is the default. At query time, retrieve and rerank the most relevant code plus its immediate context, keep the prompt tight to hit the latency budget, stream the answer, and cache the stable system prompt and common repo context. Handle freshness with incremental re-indexing on commits, and refuse or flag when the answer is not supported by retrieved code. The tight budget pushes you toward a fast model, aggressive caching, and precise retrieval over a big context stuff.

**Learn more:** [Harness engineering guide](../../../resources/harness_engineering.md)

</details>

<details>
<summary><b>130. Design a multi-tenant LLM application. What is different from a single-tenant one?</b></summary>

Isolation is the theme across every layer: each tenant's data and vector index must be partitioned so retrieval can never cross tenants, secrets and model configuration are per tenant, and every request carries a tenant identity that scopes access control, logging, and cost attribution. You add per-tenant rate limits and budgets so one heavy or abusive tenant cannot starve or bankrupt the others, and you monitor cost and quality per tenant since their data and usage differ. Prompt caching and shared infrastructure need care so cached content never leaks across tenants. Evaluation also becomes per-tenant-shaped, because a model that works for one customer's documents may fail on another's.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>131. Design a document-processing pipeline that extracts structured data from messy PDFs at scale.</b></summary>

Separate the stages so each is testable and swappable: robust ingestion and layout-aware parsing (OCR or a vision model for scans, table structure preserved), then extraction into a strict schema using structured output with validation, then a verification pass for low-confidence or failed extractions. Route by document type and difficulty, use a cheap model for the common clean cases and escalate hard ones, and batch for throughput since this is usually offline. Build an eval set of labeled documents covering the messy edge cases, measure field-level accuracy, and put a human-review queue on low-confidence extractions. The bottleneck is almost always parsing quality and the long tail of weird documents, not the model, so instrument where it fails.

**Learn more:** [Multimodal topic](../../../topics/multimodal.md)

</details>

<details>
<summary><b>132. You need to migrate a production feature from an expensive model to a cheaper one. How do you do it safely?</b></summary>

Treat it as a measured experiment, not a swap. First build or reuse an eval set that reflects real traffic and run both models through it to quantify the quality gap per case type, so you know where the cheaper model holds and where it breaks. Where it falls short, try to close the gap with prompt or context changes or by routing only the hard cases back to the expensive model, then shadow or A/B test on live traffic while watching quality, cost, and user signals. Roll out gradually behind a fast rollback, and keep monitoring because the cheaper model may fail differently on the tail. The savings only count if quality holds, so the eval and the staged rollout are the whole job.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>133. Design a system to detect and reduce hallucinations in a generative feature.</b></summary>

Attack it at three points: reduce, detect, and contain. Reduce by grounding the model in retrieved sources and instructing it to answer only from them, using constrained outputs, and giving it a clean minimal context. Detect with a faithfulness or entailment check that verifies each claim against the cited source (an LLM judge or an NLI model), plus citation-coverage checks and confidence signals, run both offline in evals and online on sampled traffic. Contain by showing sources so users can verify, refusing when support is weak, and routing high-stakes low-confidence cases to a human. There is no single switch, so you layer grounding, verification, and human oversight and measure the residual rate.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>134. Design a semantic caching layer for an LLM API to cut cost. What are the failure modes?</b></summary>

Embed each incoming query, look up the nearest cached query, and if the similarity exceeds a tuned threshold return the stored answer instead of calling the model, otherwise call the model and store the new result. The main failure mode is a false hit: two queries look similar but need different answers (different dates, entities, or intent), so you serve a stale or wrong response, which is why you tune the threshold conservatively, scope caching to safe query classes, and consider a cheap verification step. You also handle invalidation (answers go stale when the underlying data changes) and personalization (a shared cache must not leak one user's answer to another). Instrument hit rate and false-hit rate so the cache is saving money without quietly degrading quality.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

---

## Business and judgment

<details>
<summary><b>135. How do you decide whether to build a feature with prompting, RAG, fine-tuning, or an agent?</b></summary>

Match the tool to the need and start with the simplest that clears the bar. Prompting and context for behavior you can specify, RAG for fresh or proprietary knowledge that needs citations, fine-tuning for stable behavior, format, or style at scale (not for volatile facts), and an agent only when the task genuinely needs multiple dynamic steps and actions. Each step up adds cost, latency, and failure modes, so you earn the complexity with a requirement, not a hunch. Validate the choice with an eval on your own data.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>136. When is the right answer "do not use an LLM here"?</b></summary>

When a deterministic rule, a lookup, a regex, or a classic ML model solves the problem more cheaply and reliably, an LLM is the wrong tool. Also when the task demands guarantees an LLM cannot give (exact arithmetic, legal correctness, perfect consistency) without heavy scaffolding, when latency or cost budgets cannot absorb it, or when the failure cost is high and you cannot put a human in the loop. Good judgment here is a strong senior signal: the best AI Engineers know where AI does not belong.

**Learn more:** [Journeys: Build](../../../journeys/build.md)

</details>

<details>
<summary><b>137. How do you measure whether an AI feature is actually delivering value?</b></summary>

Tie it to a user or business outcome, not a model score: task success or resolution rate, deflection or time saved, conversion, or reduction in manual work, alongside quality from your eval harness, hallucination or error rate, latency, and cost per interaction. Compare against the pre-AI baseline and watch human-override or escalation rate as a truth signal. If you cannot connect the feature to an outcome someone cares about, you cannot justify its cost, and cost per interaction is real money at scale.

**Learn more:** [Journeys: Build](../../../journeys/build.md)

</details>

<details>
<summary><b>138. How do you think about dependence on a single model provider?</b></summary>

It is a real business risk: pricing, availability, model behavior, and terms can change under you, and a silent model update can regress your feature. Mitigate by abstracting the provider behind an internal interface so swapping is cheap, pinning model versions and keeping a regression eval to catch changes, and periodically evaluating alternatives (including open-weight models you can self-host) on your own tasks. You do not need to be multi-provider from day one, but you should keep the option open and know your switching cost. Design so no single provider is a single point of failure for a critical feature.

**Learn more:** [Production topic](../../../topics/production.md)

</details>

<details>
<summary><b>139. How do you communicate the risk and limitations of an AI feature to non-technical stakeholders?</b></summary>

Be concrete about the failure modes and their likelihood, and frame quality as a distribution rather than a single impressive demo. Tie the risk to the specific decision the feature drives, so a stakeholder understands what a 5% error rate means for that use case. Set expectations with evals and a phased rollout plan instead of adjectives, and be honest about what you cannot guarantee. Credibility comes from naming the limits before they bite, not from overselling the demo.

**Learn more:** [Journeys: Use](../../../journeys/use.md)

</details>

<details>
<summary><b>140. How do you decide build vs buy for an AI capability?</b></summary>

Weigh it on proprietary data, differentiation, time to market, total cost, and risk rather than enthusiasm. Buy (a platform or API) when the capability is not your core differentiator, you need it fast, and a vendor's TCO beats building; build when the capability is a durable competitive edge, your proprietary data and workflows create value no vendor captures, or compliance forces control. In practice most enterprise AI is hybrid: buy the platform, then customize with your data, workflow logic, and human review, and let different components follow different choices. State the decision in terms of ROI and switching cost, and revisit it as the market and prices move fast.

**Learn more:** [Journeys: Build](../../../journeys/build.md)

</details>

<details>
<summary><b>141. Why do so many LLM prototypes fail to reach production, and how do you de-risk that gap?</b></summary>

A demo shows the happy path on a few hand-picked inputs; production faces the full messy distribution, adversarial inputs, scale, latency and cost budgets, access control, and the need to prove quality holds over time. The gap is usually evaluation, reliability, and the unglamorous edges (the unanswerable case, the tool error, the injection, the regression after a model update), not the core capability. You de-risk it by building the eval set and the failure handling early, testing on realistic data instead of demo inputs, and scoping the first release narrow with monitoring and a human in the loop. The teams that ship treat "make it reliable and prove it" as the actual work, not a finishing step.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>142. How do you decide where to put a human in the loop, and how does that change over time?</b></summary>

Put a human where the cost of an error is high and the model's reliability on that case is not yet proven: approving irreversible or high-value actions, handling low-confidence outputs, and labeling the edge cases that feed your eval and improvement loop. It is a deliberate design choice tied to risk, not a blanket fallback, so lower-risk paths get sampled audits rather than per-item review. Over time you earn more autonomy: as monitoring and evals show the model is reliable in a scope, you move humans from reviewing every case to spot-checking and exception-handling. The human-in-the-loop rate is also a signal, a high override rate tells you where the system is still weak.

**Learn more:** [Safety and Security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>143. Models improve and get cheaper fast. How do you build so that helps you instead of stranding your work?</b></summary>

Design for model change: abstract the provider and model behind an interface, keep prompts and evals versioned so you can re-test a new model in an afternoon, and avoid hard-coding scaffolding that only exists to compensate for today's model weakness. Maintain a task-specific eval so that when a better or cheaper model ships you can measure the upgrade objectively rather than guessing, and treat model choice as a periodic decision. Avoid over-investing in elaborate workarounds that a next-generation model will make obsolete. The goal is a system where a model improvement is a config change and a re-run of your eval, not a rewrite.

**Learn more:** [Journeys: Understand](../../../journeys/understand.md)

</details>

<details>
<summary><b>144. How do you prioritize what to improve when an AI feature is underperforming?</b></summary>

Let error analysis, not intuition, set the priority: sample real failures, build a taxonomy, and size each failure mode by frequency and cost so you fix the one that moves the outcome most. Distinguish a retrieval problem from a generation problem from a prompt or a hard-input problem, because they have different fixes and chasing the wrong one wastes effort. Weigh each candidate fix by expected impact against effort and risk, ship the highest-leverage one, and re-measure to confirm it moved the metric without regressing others. This disciplined loop (measure, diagnose, fix the biggest thing, re-measure) is what separates steady improvement from thrashing.

**Learn more:** [AI Evals for Everyone (free course)](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>145. What technical debt is unique to LLM systems, and how do you manage it?</b></summary>

Beyond normal code debt you accumulate prompt debt (a pile of undocumented, untested prompt tweaks nobody dares touch), eval debt (a stale or missing eval set so you cannot safely change anything), context and memory bloat, brittle scaffolding built around an old model's limitations, and hidden coupling to a specific model version. You manage it by versioning prompts and evals, treating the eval set as living infrastructure you maintain, periodically pruning workarounds that newer models made unnecessary, and keeping the provider abstraction clean so upgrades stay cheap. The most dangerous form is a system with no eval, because then every change is a gamble and the debt is invisible until it breaks in production. Pay it down by making quality measurable first.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

---

Next: [resources](resources.md) and [courses](courses.md) to shore up any theme, then the [prep plan](prep-plan.md). Back to the [role README](README.md).
