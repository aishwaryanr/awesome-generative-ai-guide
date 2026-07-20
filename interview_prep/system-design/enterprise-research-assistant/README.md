# Designing an Enterprise Research Assistant

## The interview question

> "Design an enterprise research assistant that answers employee questions across many internal sources (docs, wikis, tickets, code, chat) with citations, and respects each user's permissions. Walk me through it."

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real-world benchmarks and deployment data, and it ships with [runnable, provider-agnostic LangGraph code](code/) you can execute.

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that the core component is non-deterministic, so evaluation moves from a final gate to the center of the design.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/enterprise-research-assistant (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs, especially permission-scoped retrieval and the multi-agent research fan-out. When we get to the code, run `code/run.py` and explain what each node in the graph does."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine

This case is built on the same 5-layer spine as the rest of the collection: **1 the model**, **2 the wrapping layer** (the architecture: knowledge, tools, and memory), **3 evals and guardrails**, **4 production and ops**, and **5 optimization** (where multi-agent lives). System design is the work of composing them into one coherent system. For how we use this spine and why it is a starting point rather than a rulebook, see [the spine](../README.md#the-spine-how-we-think-about-ai-system-design).

The rest of this case study walks these layers for this problem, going deepest where it is hardest.

---

## The answer

### Step 1: scope before you architect

The first move is clarifying the problem rather than reaching for a vector database, because in AI the largest failures are designed in before a single prompt is written. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** An employee needs an answer that is spread across systems they already have access to: a policy in the wiki, a decision in a ticket, an owner in the code, a thread in chat. Finding it by hand means searching 4 tools, reading 10 documents, and pinging 2 colleagues, and the answer still arrives slow and unsourced. The people who know are interrupted all day to answer questions the corpus already contains.
- **Outcome, written before the system.** Return a correct, cited answer that stitches together only the sources this specific user is allowed to see, or hand off cleanly when the corpus cannot support one. Measured by answer correctness, citation faithfulness (every claim traces to a real source), a permission-leak rate held near zero, and time saved per question, with a hard ceiling on wrong answers and on any source shown to someone who should not see it.
- **The AI intervention, narrowed until it hurts.** Retrieve across the connected sources under the asking user's permissions, run the follow-up searches a hard question needs, answer with citations, and escalate when nothing readable supports an answer, staying well short of taking actions on the user's behalf.
- **System and safety.** Permissions enforced at retrieval time so a source the user cannot read never enters the context, an evaluation set that gates every release, input and output guardrails, [prompt-injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) defense on every retrieved document, a human gate on any high-impact action, full tracing, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): how many sources and how large is the corpus; how are permissions modeled in each source system and how fresh must they be; how often do questions span several sources; is a citation mandatory on every answer; what is the freshness requirement per source; may the assistant ever take an action, or only read and answer; what are the compliance constraints. This also avoids the traps that sink these projects: leading with "build a RAG agent over all our data" (solutioning in the problem statement), packing search plus writing plus workflow automation into one system (over-scoping), and designing without a measurable owner for correctness and for permissions.

> **Real data: [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).** A lead-plus-subagents setup beat a single agent by about 90.2% on Anthropic's internal research eval, while using about 15x the tokens of an ordinary chat, and token usage alone explained roughly 80% of the performance variance. The lesson for scoping: research that fans out across many independent sources is exactly the shape of problem where spending more compute buys real capability, and it is also expensive, so you scope to the questions worth that spend and keep the cheap path for the rest. Treat the percentages as a point-in-time reading on one eval and track the same measures on your own traffic over time.

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data, because every method in the wrapping layer below is chosen for a specific shape of data. Change an assumption and the method changes with it. For this case study, assume:

- **Corpus: large and heterogeneous, across several systems.** Hundreds of thousands of items spread across a wiki, a document store, a ticket tracker, a code host, and a chat tool, in wildly different formats: prose articles, PDFs and slide exports, threaded tickets, source files, and short chat messages. *This is why* the knowledge layer starts with connectors and format-aware parsing, and why a single naive index is not enough.
- **Permissions: strict and per-user, and they change constantly.** Every item carries its own access rules in its source system, and who can see what shifts every day as people join teams, projects close, and documents are reshared. *This is why* permissions are enforced at retrieval time against the asking user's identity, and why the permission data has to stay synced with the source systems rather than be baked in once at ingestion.
- **Questions: often multi-hop.** A meaningful share of questions cannot be answered by one document. The answer lives in the wiki for the policy, the code host for the owner, and a ticket for the exception, and it has to be stitched together. *This is why* single-shot retrieval is not enough here and the agent runs its own follow-up searches.
- **Answers: cited, always.** An answer without sources is unusable in an enterprise, because the reader has to be able to verify it and follow it back. *This is why* citation faithfulness is a first-class output and a first-class metric, engineered from the start.
- **Freshness: varies by source.** A code file or a live ticket changes by the minute, a policy wiki changes monthly, an archived doc effectively never changes. *This is why* the ingestion pipeline syncs different sources on different cadences rather than re-indexing everything on one schedule.
- **Volume and latency: interactive.** Employees ask throughout the day and expect an answer in a few seconds for simple questions, tolerating longer for the hard multi-source ones. *This is why* there is a latency budget the reranker and the multi-hop loop must fit inside, and why routing the easy majority to a cheap fast path pays off.

Keep these in view as you read the wrapping layer: each choice below points back to one of them. If your corpus is one small wiki with open permissions, most of this collapses to single-shot retrieval, and if your questions never span sources, the multi-hop loop is dead weight. Match the method to the data.

### Step 2: walk the layers for this system

**Layer 1, the model.** The answer here is a routing strategy: a cheap fast model handles simple lookups and drafts the individual search steps, a stronger model plans the hard multi-source questions and synthesizes the final cited answer, and a reasoning model is held for the rare question that needs careful stitching across conflicting sources. The model is non-deterministic, so the same question can produce different phrasings and different search paths. You handle that with structured output, grounding every claim in retrieved text, and evaluation. Keep the model behind a provider-agnostic interface so you can swap it when a better one ships (the runnable [code](code/) does this, and Follow-up 7 covers it).

**Layer 2, the wrapping layer (the architecture).** This is where most of the design lives, so go deep here. It is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. For a research assistant the knowledge layer is the product, and it carries two properties that make this system distinct: it spans many sources, and it is scoped to what each user is allowed to read.

**Knowledge (retrieval).** The agent must ground every answer in your actual internal sources, under the asking user's permissions, or it invents policy or leaks a document. That is a retrieval pipeline, and every stage is a decision whose right answer depends on your data. Each choice below points back to the [assumptions](#the-assumptions-we-make-about-the-data-and-the-use-case) above. Here is the pipeline, then each stage: why it matters and what data tells you how to set it.

```
┌─────────────────────────────────────────────────┐
│ Knowledge pipeline: permission-scoped retrieval │
└─────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │   wiki / docs / tickets / code / chat    │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │           1  Connect + ingest            │
  │ per-source connectors, incremental sync, │
  │         capture ACLs + freshness         │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │         2  Parse (format-aware)          │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │                 3  Chunk                 │
  │       structure-aware, contextual        │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │                 4  Embed                 │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │             5  Store / index             │
  │     vectors + metadata: acl, system,     │
  │            updated_at, tenant            │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │      6  Permission-scoped retrieve       │
  │ user + question in; filter by identity,  │
  │         THEN hybrid dense + BM25         │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │        7  Rerank (cross-encoder)         │
  └──────────────────────────────────────────┘
                        ▼
  ┌──────────────────────────────────────────┐
  │            8  Relevance floor            │
  └──────────────────────────────────────────┘
                        ▼                      yes
  ┌──────────────────────────────────────────┐    ┌──────────────────┐
  │         9  Multi-hop: need more?         │◀──▶│  agent issues a  │
  └─────────────────────┬────────────────────┘    │ follow-up search │
           ┌────────────┴────────┐                └──────────────────┘
        no ▼                     ▼ below floor
  ┌─────────────────┐  ┌──────────────────┐
  │ 10  Answer with │  │     Escalate     │
  │    citations    │  │ nothing readable │
  └─────────────────┘  └──────────────────┘
```

**1. Connect and ingest.** The corpus lives in other systems, so the first job is connectors: a piece of code per source (wiki, document store, ticket tracker, code host, chat) that reads its content through its API and pulls it into your pipeline. Every item has to bring more than text across, and two things in particular ride with it. First, its **access-control metadata**, who is allowed to read it in the source system, because that is what every later permission decision rides on. Second, its **freshness signal** (`updated_at`, version), because you sync a live ticket tracker far more often than an archived doc set. Sync incrementally (only what changed since the last run) rather than re-crawling everything, or the pipeline cannot keep up with a large corpus. *Given our assumption* of many systems on different change cadences, you run each connector on its own schedule and treat permission changes as first-class events to re-sync, the same way you treat content changes. *What data decides it:* the APIs and permission models of your actual source systems, and how fast each one changes.

**2. Parse.** The formats are wildly different, and naive text extraction flattens the structure the retriever needs. A wiki page is HTML with headings and tables, a policy is a PDF with columns, a ticket is a threaded conversation with a status, a code file is source with symbols, a chat message is a short line in a thread. Parse each format with something that respects its structure, and for PDFs or complex documents use a layout-aware parser ([Docling](https://github.com/docling-project/docling), [LlamaParse](https://github.com/run-llama/llama_cloud_services), [Reducto](https://reducto.ai/)) rather than a raw text dump. Attach metadata to every parsed unit (source system, item id, author, `updated_at`, and the access-control list), because freshness, filtering, permissions, and citations all ride on it. *Given our heterogeneous assumption,* the parser is really several parsers behind one interface. *What data decides it:* audit your messiest sources (nested tables, long code files, deep ticket threads) and measure how often parsing mangles them.

**3. Chunk.** The chunk is the unit of retrieval, so its size is a real lever, and different sources want different rules. Too big and the answer is buried among irrelevant text that also burns context budget ([Lost in the Middle](https://arxiv.org/abs/2307.03172)); too small and a sentence is severed from the context that gives it meaning. Chunk by structure: a wiki section per chunk, a code function or class per chunk, a ticket comment or resolved-thread per chunk, so each chunk is a coherent idea. The current high-leverage move is [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval): prepend a short model-generated blurb situating each chunk in its document before you embed it, which matters even more across many sources, because a chunk pulled out of a ticket needs to carry which ticket and which system it came from. *What data decides it:* your question and answer length distribution per source, and a chunk-size sweep scored on a retrieval-quality measure such as [recall@k](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29) against a labeled set, tuned per source rather than globally.

> **Real number: [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval).** Anthropic reported that adding a short generated context to each chunk before embedding cut retrieval failures by about 35%, and by about 49% when combined with a reranking step. The takeaway is that how you prepare a chunk moves retrieval quality as much as which embedder you pick. Treat the figures as one team's result on their data and re-measure on yours.

**4. Embed.** An [embedding](https://arxiv.org/abs/2004.04906) is a list of numbers that captures a piece of text's meaning, so two texts that mean similar things land close together in that number space. The embedding model you pick defines what "similar" means, so a weak or off-domain embedder caps everything downstream no matter how good your reranker is. Choose from the [MTEB retrieval leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (a public ranking of embedding models on retrieval tasks), favoring one trained near your domains, and remember your corpus mixes prose and code, which pull toward different strengths; [Matryoshka embeddings](https://arxiv.org/abs/2205.13147) are a newer style that lets you shorten the number list later to trade a little accuracy for cheaper, faster search at this corpus size. *How you would choose:* run 2 or 3 candidate embedders on your own labeled question-to-document pairs across each source type, and compare a retrieval-quality measure against embedding latency, cost per million tokens, and index memory at your scale.

**5. Store and index.** The chunks and their embeddings live in a vector store, and the *index* is the structure inside it that makes similarity search fast enough to run on every question. The index choice trades speed, memory, and accuracy. [HNSW](https://arxiv.org/abs/1603.09320) builds a navigable graph through the vectors and gives fast, high-recall search at the cost of more memory; [IVF-PQ](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) compresses the vectors into buckets, using less memory but needing more tuning to stay accurate. Managed stores (Pinecone, Weaviate, Qdrant, Turbopuffer) run this for you; [pgvector](https://github.com/pgvector/pgvector) adds vector search to Postgres. The property that matters most here is that every vector is stored **alongside its metadata**, especially its access-control list and its tenant, because the next stage filters on exactly that. *Our assumption of a large multi-source corpus with strict permissions* means the index has to support fast metadata filtering and per-tenant isolation, which narrows the store choice. *How you would choose:* corpus size, queries per second, accuracy target, and the strength of the metadata-filtering and isolation you need.

**6. Permission-scoped retrieve.** This is the load-bearing stage of the whole system, so go slow here. Retrieval pulls the handful of chunks most likely to hold the answer, and in an enterprise it must do that **only over the chunks this specific user is allowed to read**. The invariant is simple to state and unforgiving in practice: a user must never see, in an answer or a citation, a source they could not open themselves. If that can ever break, the assistant is a data-leak engine, however good its answers are.

Get the enforcement point right first. You apply the user's identity **at retrieval time, before any content reaches the model**, so a forbidden chunk is never even a candidate. That is safer than letting the model retrieve broadly and trimming afterward, because anything that reaches the context can leak into the answer or be surfaced through a clever prompt. This is the design enterprise search vendors converged on: filter every query through a permission-aware retrieval layer before content reaches the language model, so results are scoped to what the user is actually allowed to see ([Glean on permissions-aware AI](https://www.glean.com/perspectives/security-permissions-aware-ai)). In the runnable [code](code/), `retrieve()` filters the corpus by the caller's roles before it scores anything, which is the whole idea in miniature.

Two ways to model who-can-read, and you will usually carry both:
- **[Access-control lists (ACLs)](https://en.wikipedia.org/wiki/Access-control_list)** attach to each document the exact set of principals (users or groups) allowed to read it. This is the fine-grained truth for a single reshared doc.
- **[Role-based access control (RBAC)](https://en.wikipedia.org/wiki/Role-based_access_control)** grants access by role (engineering, HR, a specific project), which scales better than listing individuals and maps to how most source systems actually grant access.

The hard part is keeping the permission data true rather than the check itself. Permissions change constantly, so the copy you filter on has to stay synced with the source systems, or a user keeps seeing a document after they lost access to it. You keep it fresh by re-syncing permissions as their own events, and for the highest-stakes sources you re-verify against the source system at query time (late-binding the check) so a revoked grant takes effect immediately. Permissions also live at more than one grain: a whole document, a section, sometimes a single field, so the access-control metadata rides all the way down to the chunk. And a multi-tenant deployment adds a hard outer boundary, tenant isolation, so one customer's data can never surface in another's search regardless of any inner rule. *This entire stage exists because we assumed strict, constantly-changing, per-user permissions.* Drop that assumption and retrieval simplifies enormously; keep it and this is the wall you engineer most carefully. *What data decides it:* the permission models of your source systems, how fast grants change, and your tolerance for staleness. *Deeper:* the repo's [RAG topic page](../../../topics/rag.md) and [Safety and Security](../../../topics/safety-security.md).

Now, within the permitted set, matching a question to a chunk. There are two ways to match, and they fail in opposite places:
- **[Dense retrieval](https://arxiv.org/abs/2004.04906)** compares *meaning*. You turn the question and every chunk into embeddings and pull the chunks whose numbers sit closest to the question's. This lets "how do I take leave" find the time-off policy even though they share no words. Its weak spot is exact strings: a ticket id like `JIRA-4021`, a config key, or a function name carries no meaning to embed, so dense retrieval often slides right past it.
- **[Sparse retrieval](https://www.pinecone.io/learn/hybrid-search-intro/)** compares *words*. [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is the long-standing method: a keyword-scoring formula that ranks a chunk by how many of the query's words it contains, weighting rarer words more heavily. It catches the exact ids, symbols, and error codes that fill internal questions, and needs no model to run. Its weak spot is the mirror image: it cannot tell that "take leave" and "paid time off" mean the same thing.

Because internal questions carry both meaning and exact identifiers, you want both methods, which is **hybrid retrieval**: run dense and sparse together and merge their two ranked lists into one. The common way to merge is [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion), a simple rule that rewards a chunk for ranking highly in *either* list, so results both methods agree on rise to the top. To get started, run a BM25 index and a dense index side by side over the permitted set and fuse them with Reciprocal Rank Fusion, which most vector databases support directly.

When a question is too short or vague to retrieve well, three techniques rewrite it before searching, and you add them one at a time as needed:
- **Query rewriting**: have the model turn a fragment into a fuller question ("payments oncall" becomes "what is the on-call escalation policy for the payments service") that retrieves better. [[explainer](https://arxiv.org/abs/2305.14283)]
- **Multi-query**: have the model produce several phrasings of the same question, retrieve for each, and combine the results, so one awkward wording does not sink the search. [[explainer](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)]
- **HyDE** (Hypothetical Document Embeddings): have the model draft a rough, made-up answer, then retrieve the real chunks closest to that draft, on the idea that an answer resembles the target document more than the bare question does. You can use HyDE when questions and answers tend to be worded very differently. [[explainer](https://arxiv.org/abs/2212.10496)]

*How you would evaluate it:* bucket a sample of real questions by type and by source, and see where dense-only retrieval misses. If exact-id lookups into tickets and code are a meaningful share of misses, the sparse arm is not optional. Watch a retrieval-quality measure (recall@k is a common one) as you tune, where the exact metric matters less than tracking the same one over time.

**7. Rerank.** First-stage retrieval is tuned to cast a wide net: get the right chunk somewhere in the top ~50, even if it is not ranked first. But the model only reads the few chunks you actually hand it, so the best ones need to be at the very top. A **reranker** does exactly that: it takes those ~50 candidates and reorders them so the most relevant land in the top few you pass to the model.

The reason a reranker beats the first-stage order comes down to *how* it reads. Dense retrieval embeds the question and each chunk **separately** and then compares the two number-lists, which is fast, because every chunk's embedding is computed ahead of time, and coarse. A [**cross-encoder**](https://www.sbert.net/examples/applications/cross-encoder/README.html) reranker instead feeds the question and one chunk into a model **together**, so it can weigh how well that specific chunk answers that specific question and catch nuances the separate embeddings miss. It is slower, because it runs once per candidate at query time with nothing pre-computed, which is why you run it on ~50 candidates rather than the whole corpus. Off-the-shelf cross-encoders (Cohere Rerank, Voyage, the open bge-reranker) drop in with a single call. Reranking earns extra keep here because a multi-source pool mixes a wiki paragraph, a ticket comment, and a code snippet that all look plausible, and the cross-encoder is what sorts the truly on-point one to the top. *How you would evaluate it:* compare retrieval quality with and without the reranker on your own labeled set, and weigh any lift against the latency it adds.

**8. Relevance floor and abstention.** This is the safety valve that converts not finding an answer into an escalation instead of a confident hallucination. Set a threshold on the reranker score, or add a lightweight grader that judges whether the retrieved context is sufficient (the idea behind [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): retrieve, grade, then answer, re-retrieve, or abstain). *Because we assumed a wrong answer and a leaked source are both costly,* set the floor conservatively so the agent escalates rather than guesses, and remember the floor also protects permissions: when the only relevant sources were filtered out because the user cannot read them, the permitted set is empty, it falls below the floor, and the agent hands off cleanly rather than hinting at what it saw. The runnable code shows this exact behavior: a regular employee asking about compensation bands retrieves nothing readable and gets an escalation, while an HR user gets the answer. *What data decides it:* sweep the threshold and plot escalation precision against answer coverage on a labeled should-answer versus should-escalate set, then pick the point that keeps wrong answers under the ceiling you set in scoping.

**9. Multi-hop and agentic retrieval.** Because we assumed many questions span several sources, a single retrieval pass is not enough, and this is where the assistant stops being a search box and becomes an agent. In **multi-hop retrieval** the answer requires stitching facts from several documents, often where the second search depends on what the first one returned ([BrowseComp](https://arxiv.org/abs/2504.12516) is a current benchmark for exactly this kind of persistent, multi-hop search). In **agentic retrieval** the agent runs that loop itself: it retrieves, reads what came back, decides whether it has enough, and issues its own follow-up search if it does not, until it can answer or it gives up and escalates. This is the [ReAct](https://arxiv.org/abs/2210.03629) pattern (reason, act, observe, repeat) applied to search, and it is the fast-moving frontier of retrieval; the [agentic RAG survey](https://arxiv.org/abs/2501.09136) collects the current methods. The runnable code shows the smallest version: it plans a multi-part question into separate searches, runs them in a loop bounded by a hard step cap, and stitches the results, so "what is the payments escalation policy, and which team owns the payments-core repository" becomes two searches into two different systems, merged into one cited answer.

You bound this loop hard, because an agent that can search forever will, and each hop costs tokens and latency. When questions hop across a dense web of related entities (an org chart, a service dependency graph), [GraphRAG](https://arxiv.org/abs/2404.16130) builds a knowledge graph over the corpus so the agent can traverse relationships instead of guessing at them with repeated searches. *We assumed multi-hop is common but not graph-shaped,* so the bounded agentic loop is the right level here, and GraphRAG is the reach when your questions are genuinely about relationships. *What data decides it:* the share of questions that need more than one source, and whether those sources connect through named relationships worth modeling as a graph.

**10. Answer with citations.** Because we assumed citations are mandatory, the answer is prose where every claim traces to a source the user can open. This is **citation faithfulness**, and it is a distinct property from being correct: an answer can be right and still cite the wrong document, or cite a document that does not actually support the claim. You get faithful citations by grounding generation in the retrieved chunks and attaching each sentence to the chunk it came from, and you verify it rather than trust it. The research framing is attributed question answering, benchmarked by work like [ALCE](https://arxiv.org/abs/2305.14627), which scores whether each cited passage genuinely supports its claim. A strong production pattern separates the jobs: one pass researches and drafts, and a dedicated citation pass attaches every claim to its supporting source, which is exactly what Anthropic's research system does with a separate citation agent ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). In the runnable code the guardrail rejects any answer whose claims are not backed by a retrieved, permitted source, so an ungrounded draft escalates instead of shipping. *What data decides it:* how strict your verification must be, which follows from how much the reader relies on the citations being airtight.

**What is shifting in 2026.** Two changes sit alongside this pipeline and complement it, and neither retires the foundation above. Understanding the pipeline is what lets you place them well.

- **Agentic retrieval is becoming the default shape of hard search.** Stage 9 already runs the smallest version, and the frontier pushes it further: the agent issues its own iterative searches, reads what returns, and decides what to fetch next, instead of leaning on one static retrieve step ([agentic RAG explainer](https://weaviate.io/blog/what-is-agentic-rag)). It helps most when a question genuinely spans sources and the second search depends on the first, which is common in this corpus. The classic single-pass pipeline stays the right tool for the everyday lookup, because it is cheaper, faster, and precise, so you route the easy majority to it and reserve the agentic loop for the questions that earn the extra tokens and latency.
- **Long-context models let you fit more into the prompt.** Context windows now reach into the millions of tokens, so for a small, self-contained set of documents you can pass whole documents to the model and lean less on aggressive chunking ([long context](https://ai.google.dev/gemini-api/docs/long-context)). That trims pipeline complexity on the small, stable corners of your corpus. At this scale retrieval stays primary: hundreds of thousands of items across many systems never fit one window, cost and latency climb with every token you add, permissions still have to filter what the model may see, and a long window buries the relevant passage in the middle ([Lost in the Middle](https://arxiv.org/abs/2307.03172)). So you use a bigger window to relax chunking where the corpus is small, and keep the retrieval pipeline as the primary path for scale, cost, precision, and freshness.

**Tools (actions).** For a read-only research assistant the primary tool is search itself, wrapped so the agent can call it with a query and a scope. If you let the assistant do more than answer (open a ticket, post a summary, send a message), each such tool is a typed, allowlisted contract, and the high-impact ones are gated.

```
  search(query, source_scope) -> chunks     READ   permission-scoped to the asking user
  open_ticket(summary)                       WRITE  logged, idempotent
  post_to_channel(text, channel)             WRITE  high-impact -> human approval (Follow-up 2)
```

The calls that matter: the **tool description is a prompt** (the model picks a tool from its name and doc, so vague descriptions cause wrong calls); **least privilege** (search is cheap to trust, anything that writes or communicates is gated); **[idempotency](https://en.wikipedia.org/wiki/Idempotence)** (a retried write must not act twice); **error handling** (a failed call is caught and retried or escalated, never hallucinated over); and the **loop is bounded** (a hard step cap stops the agent from searching forever on a question it cannot resolve). The runnable code gates a request to post to an all-hands channel to a human before anything is sent, which is the pattern for every high-impact action.

**Memory.** Memory is what lets a research session build on itself instead of restarting each turn, and it comes in layers.

```
  SHORT-TERM (this session)  : the running conversation, so "and who owns that service?" resolves against the last answer
  WORKING   (this question)  : the chunks and intermediate search results the agent is reasoning over right now
  LONG-TERM (this user/org)  : prior questions and durable context, RETRIEVED on demand, never stuffed wholesale
```

The calls that matter: **what to keep and what to drop** (the context window is finite, so old turns and spent search results are summarized or cleared rather than carried verbatim); **retrieve long-term memory, do not dump it**; and **treat every retrieved document and message as untrusted**, because anything the corpus feeds into the context can carry an injected instruction, the memory-and-content arm of the lethal trifecta in Follow-up 2. This assistant sits squarely in that trifecta by design, which is why the guardrail treats retrieved text as data to cite, never as instructions to follow.

Together, permission-scoped multi-source knowledge, a small set of gated tools, and layered memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for an agent this is the center of the design, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, blocking an ungrounded answer, catching a permission leak, or firing the human handoff the moment something is off. There is no single accuracy number, and no generic metric you can copy off a shelf, because good means something different for every product. So start where we teach you to start: **from failure modes.** Ask what could go wrong that would be unacceptable for this business, then translate each into an observable, measurable behavior. For this assistant the unacceptable failures are sharp: a wrong answer stated confidently, a citation that does not support its claim, and above all a source shown to someone who should not see it. The metrics below are a menu you draw from once you know your failure modes, and the target is the minimum set that gives the most signal.

**Measure against today's baseline rather than an abstract target.** The right bar is the status quo: employees searching across tools by hand today. Where a question was too costly to chase, so people gave up and worked without the answer, an assistant that surfaces it reliably clears a low bar by existing. Where employees already find the answer through manual search, the assistant has to match or beat that on speed and correctness, measured retrospectively against how those searches actually went. That keeps the eval tied to the decision the business faces rather than chasing a round accuracy number.

Evaluate at three levels: **each component** (did retrieval find the right doc within the user's permissions, did the model ground its answer, did each citation support its claim), **the whole task** end to end (was the multi-source question actually answered correctly and completely), and **live traffic** (is it still good, and still leak-free, in production).

> **Real finding: [tau2-bench](https://arxiv.org/abs/2506.07982).** On a tool-using agent, a single-attempt pass rate looks respectable, and pass^k (succeed on all k independent tries) collapses as k grows. An assistant that answers correctly on a single try yet leaks or fabricates 1 run in 4 is not shippable. Reliability is the bar, above average accuracy, and for a permission-scoped system the leak rate is the reliability number that has to hold near zero. Track pass^k on your own scenarios and watch how it moves, rather than fixing on a single published figure.

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: did retrieval respect the user's ACL, does every citation point to a real retrieved chunk, is the output structurally valid, did it abstain when it should. Fast, reliable, cheap. Use wherever good is objectively checkable, and compare against a reference dataset here. Permission enforcement belongs here, because it is checkable and must never be left to judgment.
- **LLM judges.** One model scoring another against an explicit rubric, for subjective qualities (answer correctness, completeness across sources, whether a cited passage truly supports its claim) that code cannot capture. Scalable, and a new source of non-determinism, so it must be calibrated before you trust it (below).
- **Human evaluation.** The gold standard you calibrate the other two against. Too slow and costly to run on all traffic, so you sample: calibration, edge cases, high-stakes and permission-sensitive interactions.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of labeled examples across single-source, multi-hop, permission-blocked, adversarial-injection, and unanswerable cases, including the same question asked by users with different permissions so the leak checks have something to catch. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Retrieval** | misses the right doc, returns junk, returns a forbidden doc | recall@k, precision@k, [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank), [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain), context relevance | code-based against labeled query-to-doc pairs |
| **Permissions** | surfaces a source the user cannot read | permission-leak rate, ACL-filter correctness, cross-tenant leak rate | code-based against per-user labeled access sets |
| **Answer generation** | hallucinates, incomplete across sources, wrong or unsupported citation | [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) / groundedness, correctness vs golden, completeness, citation support rate | LLM judge with rubric + golden answers |
| **Multi-hop** | stops early, misses a needed source, loops without converging | sub-question coverage, hop success rate, search efficiency | LLM judge + code-based on the trace |
| **Safety** | executes an injected instruction, leaks data, takes an unsafe action | injection-resistance rate, data-leak rate, unsafe-action rate | adversarial red-team suite |
| **End to end** | question not actually answered | task success / answer correctness, pass@1, **pass^k**, hops-to-answer | scenario suite + judge |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: these are the metrics teams typically reach for, and your job is to choose wisely and instrument only the few that earn their place. A dashboard with 30 metrics and no owner tells you nothing. For this product, two metrics carry the most signal: **citation faithfulness**, because it measures the hallucination that erodes trust fastest, and **permission-leak rate**, because a single leak is the failure that ends the project. Pick the two or three per component that map to your real failure modes and drop the rest.

Report **pass^k** alongside the average, because a research assistant that succeeds 1 try in 1 yet fails 1 in 4 is not shippable (the tau2-bench finding above).

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem, or is it merely interesting?
- **Reliability:** human evaluation and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check is cheap, a fast judge call is medium, detailed human review is expensive.

High impact and low cost are the must-haves, and for this system the permission-leak check is the clearest example: it is a cheap deterministic check on an existential failure, so it runs on every request. High impact and high cost are strategic investments you run on a sample (a calibrated faithfulness judge). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the business, run inline so the system can act the moment they trip: a citation that does not resolve to a permitted source, an answer that is not grounded, a retrieved document carrying an injected instruction, an attempt to surface something outside the user's permissions. The action is immediate (escalate to a human, block the response). Guardrails must be fast and reliable before sophisticated.
- **Improvement flywheel (offline, batch).** Everything else: answer-quality trends, faithfulness on a sample, coverage gaps, drift. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live traffic:

| Metric | Job | Why it matters |
|---|---|---|
| permission-leak rate | guardrail | the existential failure; must stay near zero, checked on every request |
| citation faithfulness on a live sample | guardrail + flywheel | catches unsupported-citation drift before users report it |
| grounding / abstention trigger | guardrail | fires the human handoff in real time when nothing readable supports an answer |
| injection-detection rate | guardrail | flags retrieved content that tries to steer the agent |
| answer correctness on a sample | flywheel | quality as the user actually experiences it |
| escalation rate | flywheel | too high means it is not helping; too low means risky over-answering |
| multi-hop success rate | flywheel | whether the follow-up-search loop actually converges |
| p50 / p95 latency | flywheel | user experience, budgeted per stage and per hop |
| cost and tokens per answered question | flywheel | unit economics, the number finance asks about, and where multi-agent shows up |
| retrieval hit rate on live queries | flywheel | surfaces corpus and connector coverage gaps |

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label a few hundred examples, run the judge on the same set, measure agreement (percent agreement or Cohen's kappa), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the underlying model, and run the judge on a sampled percentage of live traffic to control cost.

Then run the discovery loop, because users will always find failures your metrics were never built for. Sample live traffic on **signals** (thumbs-down, retries, rephrasing, an explicit "that is wrong," abandonment, and reports that an answer cited something the reader could not open). When a signal keeps firing but your metrics look clean, that gap is the tell: a human reads those traces, names the quality dimension you were not measuring (a class of multi-hop question that stops one source short, say), and it becomes a new metric added back into the reference dataset. Evaluation is never finished. You build for the failures you can anticipate, and you monitor to discover the ones you cannot.

![Eval and discovery loop: pre-deployment validation, then production monitoring, with the reference set growing each cycle](../assets/eval_discovery_loop.png)

Instrument the LangGraph app with OpenInference so every node, search, and model call becomes a span in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. Reading traces is how you find the exact hop where the agent's judgment diverged from yours. Tooling worth knowing: Ragas and DeepEval for component metrics, promptfoo for CI gates, Arize Phoenix for tracing and online evals. *Deeper:* [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../../topics/evaluation.md).

**Layer 4, production and ops.** The loop that keeps it alive at scale: connectors and an ingestion pipeline that keep the index and the permissions fresh, vector search with metadata filtering and per-tenant isolation, reliability, latency budgets per stage and per hop, and observability so every step is traceable. Detail is in Follow-ups 1 and 3.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: model routing (a cheap model for simple lookups, a strong one for hard multi-source questions), prompt and semantic caching, advanced retrieval, and **multi-agent research fan-out**. This is the second load-bearing layer for this system, because research that spreads across many independent sources is the textbook case where a lead agent plus parallel subagents earns its cost. Follow-up 5 shows exactly when it does and how you extend the single-agent design to get there.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (permission-scoped single-shot retrieve-then-answer, abstain when nothing clears the floor), prove it with evals, and only hand the model more freedom (the multi-hop search loop, then a fan-out of subagents) once the measurement says the constrained version is solid and its limits are what block you. You earn agency through evaluation before you grant it. Every increase in autonomy is paid for with an eval that shows it helped.

Composed, the layers give one architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Observability: every node, search, and model call is a span (Arize) │
└─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────┐
  │         user + identity         │
  └─────────────────────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │         Input guardrail         │
  │         injection scan          │
  └─────────────────────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │              Plan               │
  │            decompose            │
  └─────────────────────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │   Permission-scoped retrieve    │
  │    hybrid + rerank + floor,     │
  │      filtered by identity       │
  └─────────────────────────────────┘
                   ▼                  no, need more
  ┌─────────────────────────────────┐    ┌──────────────────┐
  │    Agent: enough to answer?     │◀──▶│ multi-hop loop,  │
  └─────────────────────────────────┘    │ bounded step cap │
                   ▼                     └──────────────────┘
  ┌─────────────────────────────────┐
  │         Compose answer          │
  │        attach citations         │
  └─────────────────────────────────┘
                   ▼
  ┌─────────────────────────────────┐
  │        Output guardrail         │
  │ grounded, cited, in-permission? │
  └────────────────┬────────────────┘
            ┌──────┴─────────────────┐
        yes ▼                        ▼ no / injection / leak / high-impact
  ┌──────────────────┐  ┌────────────────────────┐
  │ Answer + sources │  │ Escalate or human gate │
  └──────────────────┘  │        -> human        │
                        └────────────────────────┘
```

Read it as the spine composed: the model (layer 1), wrapped in permission-scoped multi-source retrieval, a bounded multi-hop loop, gated tools, and memory (layer 2), gated by evals (layer 3) that run inline as guardrails and offline as a flywheel, run under production observability and fresh-index ingestion (layer 4), with optimization (layer 5) held to what this system actually needs, which is routing, caching, and a multi-agent fan-out when the question is big enough. Composing exactly these pieces into one coherent system is the system design. The permission filter and the output guardrail are what make abstention the safe default: when the agent cannot ground an answer inside the user's permissions, or a source tries to inject instructions, or a request would post something to a shared audience, it hands off to a human rather than guessing or acting.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this architecture (minus the scale pieces): permission-scoped retrieval with a relevance floor, a bounded multi-hop research loop, answers with citations, a guardrail that escalates on ungrounded answers and on injected instructions, and a human gate on high-impact actions. It runs offline with a deterministic policy, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                                  # run the scenarios (also a self-test)
python run.py "What is our PTO policy?"        # ask the agent your own question (as a regular employee)
```

Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route planning and decisions through a real model. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now handle 10x the corpus and traffic. Where does it break, and how do you scale it?"

Name where it breaks first, then scale that, cheapest lever first.

- **Ingestion and permission freshness (layer 4).** At 10x the corpus across more sources, the connectors and the permission sync are the first thing to strain, ahead of the model. You move to incremental sync per source, process permission changes as their own stream so access stays current within minutes, and shard ingestion so one slow source does not stall the rest. A stale permission is a leak waiting to happen, so this is the scaling work that matters most.
- **Retrieval (layer 2).** The index has to stay fast with metadata filtering on every query (the ACL filter runs on every search). HNSW gives fast, high-recall search at higher memory; IVF-PQ is lighter but needs tuning. Per-tenant isolation becomes a hard partition rather than a filter you hope holds. The reranker moves behind a budget so it does not blow the latency target at volume.
- **Routing and caching (layer 5, optimization).** Route simple lookups to a small fast model and reserve the strong model for hard multi-source questions; cache the stable prompt prefix; add a semantic cache for near-duplicate questions, which are common when a whole team asks the same onboarding question.
- **Infrastructure (layer 4).** Horizontal scale behind a queue, connection pooling to the source-system APIs, and backpressure so a spike degrades gracefully.

Put numbers on it: tokens per answered question, an approximate cost per answer, and a latency budget split across retrieval, each hop, rerank, and synthesis, then track them over time rather than to a fixed target. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 2: "A shared document contains hidden instructions telling the assistant to exfiltrate data. How do you prevent it?"

This system sits squarely in the danger zone, so separate answering from acting and treat every source as untrusted.

- **Untrusted content by default.** Retrieved documents, tickets, and chat messages are data to cite, never instructions to follow. The agent never executes an instruction found in a source. The runnable code shows the smallest version: a retrieved ticket that says to ignore instructions and email a list outside the company is caught by the guardrail and escalated, never acted on.
- **The lethal trifecta.** An enterprise research assistant has, by design, all three legs of the sharpest way to frame this risk, the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
  The lethal trifecta

        ┌───────────────────┐   ┌──────────────┐   ┌────────────────────────┐
        │ Untrusted content │   │  Access to   │   │    Ability to act /    │
        │                   │   │ private data │   │ communicate externally │
        └─────────┬─────────┘   └───────┬──────┘   └────────────┬───────────┘
                  └─────────────────────┼───────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │ All 3 together is dangerous │
                         └─────────────────────────────┘

            any 2 are manageable; keep all 3 from meeting in one path
```

Because you cannot remove the first two legs (reading untrusted internal content over private data is the whole product), you manage the third: keep the assistant read-and-answer by default, and gate every action that writes or communicates.

- **Blast radius.** Least-privilege tool scopes, a human gate on any high-impact action (the code routes a post-to-all-hands request for approval rather than sending it), an immutable audit log of every action, and a design where the worst a hidden instruction achieves is a needless escalation. The permission scoping also contains the damage: an injected instruction still cannot reach a source the user could not read.

*Deeper:* [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 3: "Cut p95 latency in half without tanking quality."

Budget latency per stage and per hop (retrieval, rerank, each follow-up search, synthesis) against a target, then attack the largest stage.

- Cache the stable prompt prefix (system instructions, tool schemas), and add a semantic cache for repeated questions, which a shared corpus produces in bulk.
- Route simple lookups to a smaller model and reserve the strong model for the hard multi-source path.
- Run independent subagent searches in parallel rather than in sequence, so a breadth-first question finishes in the time of its slowest branch instead of the sum of all of them.
- Stream the answer so time-to-first-token stays low even when a multi-hop question takes longer overall; trim retrieved chunks so the model reads fewer tokens.

> **Real number: [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).** Caching the stable prefix of a prompt (system instructions, tool schemas, long shared context) can cut cost by up to about 90% and latency by up to about 85% on the cached portion, per provider documentation. On an assistant whose system prompt and tool schemas repeat on every call, this is a high-leverage optimization. Measure the actual saving on your own prompt shape rather than assuming the ceiling.

Quality holds because the eval set from Layer 3 gates each change: if a faster route drops citation faithfulness or lifts the leak rate, it does not ship.

---

## Follow-up 4: "Compliance needs a record of exactly what every answer was based on, and who could see it. How does the design change?"

That single constraint promotes 2 normally-optional components into load-bearing walls: **audit logging** and **retrieval provenance**. Add an immutable, queryable log that records, for every answer, which chunks it was built from, which source systems they came from, and the permission decision that let this user see them (who asked, what their access was, what was filtered out). Nothing else in the architecture changes, because the permission-scoped retrieval already computed exactly this information, so compliance is mostly a matter of persisting it. This is the general pattern: a domain constraint is what forces you to engineer a layer you would otherwise accept as a framework default. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

For the real version of this system, a lead agent over parallel research subagents plus a dedicated citation pass is the design to recommend: enterprise research fans out across many independent sources, which is the textbook fit for multi-agent (layer 5, optimization). You still start a single-source lookup as one agent with a search tool, because a narrow question does not need the fan-out, and you let the traces show when a question has grown broad enough to justify it. Here is exactly when it pays and how the architecture extends.

**When multi-agent research fan-out earns its place:**
- **The work decomposes into independent sub-tasks that can run in parallel.** A broad question ("summarize everything we know about the payments migration") splits into searches over docs, tickets, code, and chat that have no dependency on each other, so subagents run them at once and the quality and latency gains outweigh the extra tokens.
- **The total information exceeds one context window.** A generalist agent that pulls every source into one window runs out of room and loses the middle; a subagent per source keeps each context tight and focused, then the lead synthesizes.
- **Distinct sources need distinct retrieval and reasoning.** Searching code well is a different skill from reading a ticket thread; a specialist per source can carry its own retrieval scope and prompt.
- **The question is valuable enough to pay for it.** Multi-agent trades tokens for capability, which fits high-value research and does not fit a quick lookup.

**How you would extend this architecture.** Keep the single-agent design intact and put a **lead agent** in front of **specialist subagents**, one per source or per sub-question. The lead decomposes the question, allocates effort, and spins up subagents in parallel; each subagent runs its own permission-scoped retrieval loop under the same asking user's identity, so the permission boundary holds across every agent; the lead synthesizes their findings; and a dedicated **citation pass** attaches every claim in the final answer to its supporting source. The same input and output guardrails, evals, escalation, and observability now wrap the whole system across agents. The permission scoping is the non-negotiable that must be enforced in every subagent, never just at the edge.

```
┌─────────────────────────────────────────┐
│ Observability: spans across every agent │
└─────────────────────────────────────────┘

                                   ┌──────────────────┐
                                   │ user + identity  │
                                   └──────────────────┘
                                             ▼
                                   ┌──────────────────┐
                                   │ Input guardrail  │
                                   │  injection scan  │
                                   └──────────────────┘
                                             ▼
                             ┌───────────────────────────────┐
                             │          Lead agent           │
                             │ decompose, allocate, delegate │
                             └───────────────┬───────────────┘
                 ┌───────────────────────────┴───────────────────────────┐
                 ▼             ▼             ▼             ▼             ▼
           ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
           │   Docs   │  │   Wiki   │  │ Tickets  │  │   Code   │  │   Chat   │
           │ subagent │  │ subagent │  │ subagent │  │ subagent │  │ subagent │
           └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
                 └─────────────┴─────────────┬─────────────┴─────────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │     Lead agent      │
                                  │ synthesize findings │
                                  └─────────────────────┘
                                             ▼
                            ┌────────────────────────────────┐
                            │         Citation pass          │
                            │ attach every claim to a source │
                            └────────────────────────────────┘
                                             ▼
                            ┌─────────────────────────────────┐
                            │        Output guardrail         │
                            │ grounded, cited, in-permission? │
                            └────────────────┬────────────────┘
                                ┌────────────┴────────────┐
                            yes ▼                         ▼ no / leak / injection
                      ┌──────────────────┐      ┌───────────────────┐
                      │ Answer + sources │      │ Escalate -> human │
                      └──────────────────┘      └───────────────────┘

  each subagent runs permission-scoped retrieval under the SAME asking user's identity
```

> **Real data: [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens of an ordinary chat, and token usage explained roughly 80% of the performance variance. Anthropic runs a separate citation agent as the final pass, the same split shown above. The takeaway is to spend those tokens where task value and genuine parallelism justify them, and to stay single-agent where they do not. Read the percentages as a point-in-time result on one eval and re-measure on your own workload.

The honest rule: start with the single agent, instrument it, and let the traces tell you when a class of question has genuinely outgrown one agent. Multi-agent is where this design goes once the research is broad enough to need it. *Deeper:* [Agents](../../../topics/agents.md).

---

## Follow-up 6: "Add a new source system, or make it multilingual."

The architecture does not change; the data and the evaluation do. A new source is a new connector that brings across the same two things (content and its access-control metadata), parsing tuned to that format, and labeled eval examples so quality on the new source is measured rather than assumed. Multilingual is the same shape: per-language coverage of the corpus or cross-lingual embeddings, and an eval set with labeled questions per language so quality is measured per language. Permissions, guardrails, and escalation apply unchanged. This is the recurring lesson: most "make it do X" follow-ups are answered in the data and eval layers, and the box diagram stays the same. *Deeper:* [RAG](../../../topics/rag.md).

---

## Follow-up 7: "A better model drops next quarter. How do you avoid a rewrite?"

Build the harness on-policy and keep it a layer rather than a fork. Keep the model behind the provider-agnostic interface (the runnable code already does this, any model via one call), pin behavior with the eval set rather than with brittle prompt hacks, and pair every "do not do X" rule with an eval so you can delete the rule when a better model makes it obsolete. When the new model lands, you swap it, re-run the eval gate (including the permission-leak and citation-faithfulness checks), and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved. *Deeper:* [Harness Engineering](../../../resources/harness_engineering.md).

---

## Real-world reference points

- **Anthropic multi-agent research:** +90.2% on their research eval, about 15x the tokens of an ordinary chat, and token usage explains roughly 80% of the performance variance, with a separate citation agent as the final pass. Multi-agent fan-out is worth it when the research is broad and parallel and the answer is worth the spend. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]
- **Contextual retrieval:** prepending a short generated context to each chunk before embedding cut retrieval failures by about 35%, and about 49% combined with reranking. How you prepare a chunk moves quality as much as which embedder you pick. [[Anthropic](https://www.anthropic.com/news/contextual-retrieval)]
- **Permission-aware enterprise search:** the vendors that made internal search work at scale converged on enforcing permissions in the retrieval layer, before any content reaches the model, so results are scoped to what each user may actually see. [[Glean](https://www.glean.com/perspectives/security-permissions-aware-ai)]
- **tau2-bench:** pass^k collapses as k grows; reliability is the shippable bar, above average accuracy, and for this system the permission-leak rate is the reliability number that must hold near zero. [[paper](https://arxiv.org/abs/2506.07982)]
- **Lost in the Middle:** long context buries relevant passages; retrieval needs a relevance floor and a reranker rather than stuffing every source into the window. [[paper](https://arxiv.org/abs/2307.03172)]

---

## Research to know

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) (Lewis 2020): the grounding pattern the whole system rests on.
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) (Karpukhin 2020): embeddings for retrieval, the dense arm of hybrid search.
- [ReAct](https://arxiv.org/abs/2210.03629) (Yao 2022): reason and act in a loop, the shape of the multi-hop search agent.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu 2023): why more retrieved context is not free.
- [Self-RAG](https://arxiv.org/abs/2310.11511) and [Corrective RAG](https://arxiv.org/abs/2401.15884): retrieval that decides when to retrieve, re-retrieve, or abstain.
- [Enabling LLMs to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627) (Gao 2023): benchmarking whether each citation actually supports its claim.
- [GraphRAG](https://arxiv.org/abs/2404.16130) (Edge 2024): a knowledge graph over the corpus for questions about relationships.
- [Agentic RAG: a survey](https://arxiv.org/abs/2501.09136) (Singh 2025): the current landscape of agent-driven retrieval.
- [tau2-bench](https://arxiv.org/abs/2506.07982) (Yao 2024): evaluating tool-using agents with a reliability metric.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md)); [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../../resources/harness_engineering.md); Aishwarya's YouTube ([AI Engineering: A Realistic Roadmap for Beginners](https://www.youtube.com/watch?v=pAXbl1EBHJ8), [Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k), the [full channel](https://www.youtube.com/@aishwaryanr4606)); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- Anthropic, [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and [Multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system).
- Glean, [Enhancing AI security with permissions-aware frameworks](https://www.glean.com/perspectives/security-permissions-aware-ai), on enforcing access control in the retrieval layer.
- OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- Martin Fowler / Thoughtworks, [Emerging Patterns in Building GenAI Products](https://martinfowler.com/articles/gen-ai-patterns/).
- [Arize observability docs](https://arize.com/docs/) and [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/), the tracing loop and state-machine primitives the [code](code/) uses.

---

## Related in this repo

Topics: [RAG](../../../topics/rag.md) · [Agents](../../../topics/agents.md) · [Evaluation](../../../topics/evaluation.md) · [Production and LLMOps](../../../topics/production.md) · [Safety and Security](../../../topics/safety-security.md). Guides: [Harness Engineering](../../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Prepping to be asked this? See the [interview prep hub](../../README.md).
