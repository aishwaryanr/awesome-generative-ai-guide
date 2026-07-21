# Designing an Analytics Copilot (Text to SQL)

## The interview question

> "Design an analytics copilot that answers business questions over a data warehouse in natural language (text to SQL). A user types a question like 'what was revenue by region last quarter,' the system writes SQL, runs it against the warehouse, and returns the answer. Walk me through it."

A generative AI system design interview, worked end to end: **the question, a full answer, then the follow-ups**. It is built on one spine (the 5 layers below), grounded in Problem-First design, backed by real-world benchmarks and deployment data, and it ships with [runnable, provider-agnostic LangGraph code](code/) you can execute.

**AI system design is ML system design.** Start from requirements and metrics, then design the architecture to meet them. Pick the cheapest thing that clears the bar. Design for how it fails and how you measure it. Then scale. What is new is that the core component is non-deterministic, so evaluation moves from a final gate to the center of the design. In text to SQL the stakes are sharp: the system turns a plain-English question into a query, runs it against a warehouse, and reports a number a human will act on, so a wrong number returned with confidence is the failure that matters most.

> **Read this with your coding agent.** This case study is public, so you can point Claude Code, Codex, Cursor, or any agent harness at it and have it walk you through interactively, then run the code. Paste a prompt like:
>
> *"Read https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/text-to-sql-analytics (the README.md and the code/ folder). Walk me through the 5-layer spine, then quiz me one layer at a time on the design decisions and the tradeoffs, especially schema linking, safe read-only execution, and how you evaluate whether the generated query is correct. When we get to the code, run `code/run.py` and explain what each node in the graph does."*
>
> You can also ask it to reimplement the design in a different SDK or framework you prefer.

---

## The spine

This case is built on the same 5-layer spine as the rest of the collection: **1 the model**, **2 the wrapping layer** (the architecture: knowledge, tools, and memory), **3 evals and guardrails**, **4 production and ops**, and **5 optimization** (where multi-agent lives). System design is the work of composing them into one coherent system. For how we use this spine and why it is a starting point rather than a rulebook, see [the spine](../README.md#the-spine-how-we-think-about-ai-system-design).

The rest of this case study walks these layers for this problem, going deepest where it is hardest.

---

## The answer

### Step 1: scope before you architect

The first move is to clarify the problem, because in AI the largest failures are designed in before a single prompt is written. Pin down 4 things ([Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design)).

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design). The 4 questions below are the short version.

- **User and pain.** Business users (operations, finance, growth) have questions the data can answer, and today they wait in a queue for an analyst to write the SQL. A recurring ad-hoc question takes an analyst minutes to write and a business user hours or days to get answered. A [data warehouse](https://en.wikipedia.org/wiki/Data_warehouse) (the central store of cleaned, queryable business data) already holds the answer, so the pain is access, above data.
- **Outcome, written before the system.** Let a business user get a correct, trustworthy number to a well-formed question without writing SQL, measured by answer correctness on a labeled question set, the share of questions answered without a human, and trust (do users act on the number). The hard ceiling is the rate of confident wrong answers, which must stay very low.
- **The AI intervention, narrowed until it hurts.** Generate a read-only SQL query over a known schema, run it safely, and return the number with the query shown for inspection. Abstain and hand off when the question is ambiguous, out of scope, or the query cannot be validated. This stays well short of an autonomous analyst that models data, defines new metrics, or writes to the warehouse.
- **System and safety.** An evaluation set that gates every release on query correctness, a read-only execution sandbox, a guardrail that blocks any query that is not a single read, [prompt-injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) defense on the question and on any values pulled from the warehouse, abstention as the default when confidence is low, full tracing, and a rollback plan.

The clarifying questions you need to ask (their answers set every later tradeoff): how large and how stable is the schema; which dialect does the warehouse speak; what is the tolerance for a wrong number versus an abstention; how are core metrics like "revenue" and "active user" defined, and by whom; what latency is acceptable; who may see which rows; and how fresh must the data be. This also avoids the traps that sink these projects: leading with "build a text-to-SQL model" (solutioning in the problem statement), pointing the copilot at every table in the warehouse at once (over-scoping), and shipping without a labeled correctness set that a named owner watches.

> **Real outlier: the harness closes the gap on enterprise schemas.** On [Spider 2.0](https://spider2-sql.github.io/), the benchmark of text to SQL over real enterprise warehouses (schemas of 1,000-plus columns across BigQuery, Snowflake, and DuckDB), purpose-built agent systems now clear a high bar: the top systems reach roughly 74% [execution accuracy](https://arxiv.org/abs/2411.07763) on the multi-dialect Spider 2.0-Lite split and above 90% on the Snowflake split as of 2026. A frontier model behind a generic agent scaffold lands far lower on the same tasks, near 42% on Lite and 26% on Snowflake. That spread of 30-plus points between a raw model and an engineered system is the whole reason correctness lives at the center of this design: writing straightforward SQL is largely handled, and the copilot earns trust through the schema linking, semantic grounding, and verification you build around the model. You scope for correctness first and you measure it relentlessly. [[Spider 2.0 leaderboard, 2026](https://spider2-sql.github.io/)] [[Spider 2.0 paper](https://arxiv.org/abs/2411.07763)]

### The assumptions we make about the data and the use case

Before architecting, state what you are assuming about the data, because every method in the wrapping layer below is chosen for a specific shape of data. Change an assumption and the method changes with it. For this case study, assume:

- **A known relational schema.** A curated warehouse of tens to low hundreds of tables in a mostly star-shaped layout ([star schema](https://en.wikipedia.org/wiki/Star_schema): fact tables of events surrounded by dimension tables that describe them), with column names, types, and foreign keys you can read. *This is why* the design leans on schema linking (selecting the right tables and columns) rather than on freeform document retrieval, and why the whole schema does not fit comfortably in one prompt once it grows.
- **Read-only access.** The copilot may query the warehouse and never modify it. *This is why* the execution path is a genuine read-only sandbox with least-privilege credentials, and why the guardrail allows a single SELECT and refuses everything else.
- **Correctness is paramount.** A wrong number acted on by a business user is worse than an honest abstention, because a wrong number erodes trust in the whole system and can drive a bad decision. *This is why* the design pairs execution accuracy against a gold query with a validation step and an abstention path, and sets the confidence floor conservatively.
- **Questions range from simple aggregations to multi-join analytics.** Some questions are one filter and a count; others join several fact and dimension tables, window over time, and depend on a precise business definition. *This is why* the generation step needs strong schema linking and self-correction, and why the eval set spans easy, medium, and hard questions rather than one difficulty.
- **Business terms carry definitions the schema does not.** "Revenue," "active customer," and "last quarter" mean specific things to this business, and those definitions live in people's heads or a metrics layer rather than in the column names. *This is why* a semantic layer of business definitions is part of the knowledge layer, and why ambiguity handling is a first-class concern.
- **Latency and volume are interactive.** A user waits a few seconds for an answer and asks follow-up questions in the same session. *This is why* there is a latency budget the schema retrieval and self-correction must fit inside, and why caching the schema prompt and validated queries pays off.

Keep these in view as you read the wrapping layer: each choice below points back to one of them. If your schema is thousands of tables, your access includes writes, or your questions are freeform exploration rather than well-formed metrics, revisit the assumption and pick a different method.

### Step 2: walk the layers for this system

**Layer 1, the model.** The answer here is a capable code-and-reasoning model for generation, with routing underneath: a cheaper model drafts SQL for the easy, high-frequency questions, and a stronger reasoning model handles multi-join analytics and the self-correction pass. The model is non-deterministic, so the same question can produce different SQL on different calls. You handle that with a fixed schema and few-shot exemplars in context, a validation step that executes the query before trusting it, and evaluation that pins behavior. Text-to-SQL quality tracks model capability closely, which is why the model stays behind a provider-agnostic interface so you can adopt a stronger one without a rewrite (Follow-up 7).

**Layer 2, the wrapping layer (the architecture).** This is where most of the design lives, so go deep here. It is the architecture wrapped around the model, and it is 3 things: knowledge, tools, and memory. In an analytics copilot the knowledge is your schema and business definitions, the tools are SQL generation and safe execution, and getting these two right is most of the battle.

#### Knowledge: schema linking and the semantic layer

The model cannot write correct SQL against tables it cannot see, and it writes worse SQL when it sees too many. **Schema linking** is the step that maps the natural-language question to the specific tables and columns it needs, so the generator works from a small, relevant slice of the schema instead of the whole thing. It is the retrieval problem of text to SQL, and it is the highest-leverage stage in the wrapping layer.

```
┌──────────────────────────────────────────────────────────────────┐
│ Knowledge: schema linking, then generate, validate, safe execute │
└──────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────┐
  │         Warehouse catalog          │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │             1  Profile             │
  │   tables, columns, types, keys,    │
  │    descriptions, value samples     │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │              2  Index              │
  │         per table / column         │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │              3  Link               │
  │ retrieve relevant tables + columns │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │          4  Generate SQL           │
  │   from the linked schema subset    │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │    5  Dialect + static validate    │
  │              sqlglot               │
  └────────────────────────────────────┘
                     ▼
  ┌────────────────────────────────────┐
  │     6  Safe read-only execute      │
  │              sandbox               │
  └──────────────────┬─────────────────┘
                 ┌───┴─────────────────────────┐
        on error ▼                             ▼ rows returned
  ┌────────────────────────────┐  ┌────────────────────────┐
  │      7  Self-correct,      │  │         Answer         │
  │ back to generate (bounded) │  │ the number + the query │
  └────────────────────────────┘  └────────────────────────┘
```

**1. Profile the catalog.** Before you can link a question to a table, you need a rich description of each table and column: the name, the type, the foreign keys, a one-line business description, and a few sampled values (so the model knows `status` holds `delivered` and `cancelled` rather than `1` and `0`). *Given our assumption* of a known, curated schema, you can generate these descriptions once and refresh them when the schema changes. *What data decides it:* audit how often your column names are cryptic (a `d_flg` column that means "is deleted"); the more opaque the naming, the more the descriptions and value samples carry the linking.

**2. Index the schema.** For a schema that fits in a prompt (our low-hundreds-of-tables assumption at the small end), you can pass it whole. Once it grows, you index each table and column description so you can retrieve the relevant ones per question. The index is the same machinery as document retrieval: an [embedding](https://arxiv.org/abs/2004.04906) (a list of numbers capturing meaning, so similar text lands nearby) over the descriptions, chosen from the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). *This is why* Follow-up 1 (a huge schema) is a retrieval problem more than a generation problem.

**3. Link the question to a schema subset.** This is the load-bearing stage. Two ideas do most of the work, and they fail in opposite places, so you combine them:

- **Match on words.** [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (a keyword-scoring formula that ranks by how many of the query's words a text contains, weighting rarer words more) catches exact column and table names, literal values, and codes that a user quotes verbatim. Its weak spot is vocabulary: a user asks about "sales" while the column is named `net_amount`.
- **Match on meaning.** [Dense retrieval](https://arxiv.org/abs/2004.04906) compares the embedding of the question against the embedding of each table and column description, so "sales" finds `net_amount` through its description. Its weak spot is exact identifiers a description does not mention.

Running both and merging the ranked lists with [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) (a rule that rewards an item for ranking highly in either list) is **hybrid schema retrieval**, and it is the same pattern the [RAG topic page](../../../topics/rag.md) covers for documents. On top of it, a [reranker](https://www.sbert.net/examples/applications/cross-encoder/README.html) can reorder the candidate tables so the most relevant land in the small set you pass to the generator; a [cross-encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) reads the question and one table description together, which is slower per candidate and catches nuances the separate embeddings miss. The current research on doing this well is worth reading: [RESDSQL](https://arxiv.org/abs/2302.05965) decouples schema linking from the SQL skeleton so each is ranked on its own; a [context-aware bidirectional approach](https://arxiv.org/abs/2510.14296) retrieves tables-then-columns and columns-then-tables and fuses them; and [AutoLink](https://arxiv.org/abs/2511.17190) treats linking as an agent that explores the schema step by step for very large databases. *Because we assumed questions span simple to multi-join*, get linking right or the generator never sees the join key it needs. *What data decides it:* your schema size and how many near-duplicate columns exist. Measure schema-linking recall (did the tables and columns the gold query uses appear in the retrieved subset) on a labeled set, because a miss here caps everything downstream.

**The semantic layer (what the words mean).** Schema linking finds the right columns; the semantic layer says what a business term computes to. "Revenue" might be `SUM(quantity * unit_price)` before refunds for one team and net of refunds for another, and "last quarter" depends on the fiscal calendar. A [semantic layer or metrics layer](https://docs.getdbt.com/docs/build/about-metricflow) encodes these definitions once so the copilot resolves a term to an agreed formula instead of guessing. *Because we assumed business terms carry definitions the schema does not,* this is part of the knowledge layer, and it is often the difference between a query that runs and a query that is right. MotherDuck makes the sharp version of this point: your data model is the real interface the copilot needs, and grounding generation in a governed semantic model lifts correctness more than a bigger prompt does. [[semantic layer for text to SQL](https://motherduck.com/blog/bird-bench-and-data-models/)]

#### Tools: generate SQL, then execute it safely

Tools are the copilot's hands, and here there are two: write a query, and run it. Each is a typed, allowlisted contract, and the execution tool is where most of the safety engineering lives.

**Generate SQL.** The model writes a query from the linked schema, the business definitions, and a few worked examples (few-shot exemplars of question-to-SQL pairs for this warehouse, which raise accuracy more reliably than prompt wording). Decomposing a hard question into steps helps: [DIN-SQL](https://arxiv.org/abs/2304.11015) breaks generation into schema linking, classification, generation, and a self-correction pass, and reports a clear lift over single-shot prompting on Spider. Keep the exemplars close to your real questions, because the model imitates their style and their joins.

**Dialect handling.** SQL is not one language; a query that runs on Postgres can break on BigQuery or Snowflake over date functions, quoting, and `LIMIT` versus `TOP`. Generate for your warehouse's dialect, and validate with a dialect-aware parser like [sqlglot](https://github.com/tobymao/sqlglot) (it parses SQL into a syntax tree and can transpile between dialects), which lets you catch a malformed or wrong-dialect query before it ever touches the warehouse. *Because we assumed a known dialect,* you fix the target once; supporting several dialects is a later extension (Follow-up 6) handled in the data and eval layers.

**Safe read-only execution.** This is the single most important control in the whole system, because the copilot runs generated code against your production data. Defense in depth, from outside in:

```
  static guardrail   ->   single SELECT only; reject INSERT/UPDATE/DELETE/DROP/ALTER, multi-statements, comments-as-injection
  least privilege    ->   a database role that CAN ONLY read (and ideally a read replica), never write        [https://en.wikipedia.org/wiki/Principle_of_least_privilege]
  read-only session  ->   the connection itself is read-only (SET TRANSACTION READ ONLY / query_only)          [https://www.postgresql.org/docs/current/sql-set-transaction.html]
  resource limits    ->   a statement timeout and a row cap, so a runaway query cannot melt the warehouse
  result handling    ->   never interpolate user text into SQL as strings; parameterize values                [https://owasp.org/www-community/attacks/SQL_Injection]
```

The static guardrail is a cheap allowlist: parse the query, confirm it is one [SELECT](https://en.wikipedia.org/wiki/Select_%28SQL%29) (optionally a leading `WITH`), and refuse anything else. It is exactly what the runnable code does. But a guardrail in the application is not enough on its own, because a bug could let a query through, so you back it with a database role that physically cannot write and a read-only session on the connection. Allowlisting the safe shape beats trying to blocklist every dangerous keyword, because you cannot enumerate every way to phrase a mutation. *Because we assumed read-only access,* these controls are load-bearing walls rather than nice-to-haves, and the worst a jailbreak can achieve is a wasteful read, held down by the timeout and row cap.

**Validate and self-correct.** A query that parses can still fail on execution (a wrong column name, a bad join, a type mismatch) or run and return the wrong number. Two mechanisms catch these:

- **Execution-guided self-correction.** Run the query in the sandbox; if it errors, feed the exact database error back to the model and let it rewrite, bounded by a hard retry cap so a broken query cannot loop forever. This execution feedback is far stronger than the model re-reading its own SQL: [MAC-SQL](https://arxiv.org/abs/2312.11242) uses a refiner agent that executes candidate SQL, observes the error or empty result, and rewrites, while DIN-SQL's intrinsic self-correction (no execution) contributes only a small lift. The runnable code implements exactly this loop: a first query references a column that does not exist, execution returns `no such column`, and the repair rewrites it with the correct join.
- **Self-consistency.** For a high-stakes question, sample several candidate queries, execute each, and keep the answer their result sets agree on. [Self-consistency](https://arxiv.org/abs/2203.11171) (sample multiple reasoning paths and take the majority answer) transfers cleanly to SQL, because agreement on the returned rows is a strong correctness signal. It costs more tokens, so reserve it for the questions that warrant it.

**Handle ambiguous or underspecified questions.** Many real questions are ambiguous: "top customers" by revenue or by order count, "last quarter" on which calendar, "active" by what rule. You have two moves, and the right one depends on the stakes: ask a clarifying question, or make the most likely assumption and state it plainly in the answer so the user can correct it. Systems like [AmbiSQL](https://arxiv.org/abs/2508.15276) detect ambiguity and resolve it with a quick multiple-choice question, and an [expected-information-gain approach](https://arxiv.org/abs/2507.06467) picks the single clarifying question that most reduces uncertainty. *Because we assumed correctness is paramount,* the copilot surfaces its interpretation with every answer, so a wrong assumption is visible and cheap to fix instead of silently baked into a number.

#### Memory

Memory is what makes this a session instead of a series of unrelated queries, and it comes in layers.

```
  SHORT-TERM (this session)  : the running conversation, so "and for last quarter?" resolves against the previous question
  WORKING   (this task)      : the linked schema, the candidate SQL, and the execution result the copilot is reasoning over
  LONG-TERM (this warehouse) : validated question-to-SQL pairs and business definitions, RETRIEVED on demand as few-shot exemplars
```

The calls that matter: **carry the conversation** so follow-up questions inherit the prior filters; **grow a library of validated queries** and retrieve the closest few as exemplars, which is how the copilot gets better at your specific warehouse over time; and **treat retrieved values as untrusted** (a string pulled from a table could carry an injected instruction, the memory arm of the lethal trifecta in Follow-up 2). Together, the schema and definitions, the two SQL tools, and this memory are the architecture wrapped around the model.

**Layer 3, evals and guardrails.** You cannot ship what you cannot measure, and for an analytics copilot this is the second load-bearing layer, worked throughout the build. Evals tell you whether the system is good; guardrails are the subset of those checks that run in real time and act, blocking an unsafe query and firing an abstention the moment confidence is low. There is no single accuracy number you can copy off a shelf, because "good" means something different for every warehouse. So start where we teach you to start: **from failure modes.** For text to SQL the unacceptable ones are concrete: a confident wrong number, a query that runs but answers a different question than the user asked, a wrong metric definition, a destructive or out-of-scope query, and an ambiguity resolved the wrong way silently. Translate each into an observable behavior, then measure it. The metrics below are a menu you draw from once you know your failure modes.

**Measure against today's baseline rather than an abstract target.** The right bar is the status quo: an analyst writing the query by hand, or the question going unanswered because the queue is too long. Where a question never got asked because it was too costly to route through an analyst, a copilot that answers it reliably clears a low bar by existing. Where an analyst already writes that query well, the copilot has to match or beat it, measured retrospectively against the analyst-written queries in your history. That keeps the eval tied to the decision the business faces rather than chasing a round execution-accuracy number.

Evaluate at three levels: **each component** (did schema linking retrieve the right tables, is the SQL valid, does it execute), **the whole task** end to end (does the returned number answer the question), and **live traffic** (is it still correct in production).

> **Real finding: execution accuracy is necessary and not sufficient.** [Execution accuracy](https://arxiv.org/abs/2411.07763) compares the rows your query returns against the rows a gold reference query returns. It is the standard metric, and it has two blind spots. A query can return the right rows for the wrong reason (a coincidental match), and a query can be marked wrong when it is a valid different phrasing. The benchmarks themselves carry this noise: a [2026 audit of text-to-SQL leaderboards](https://arxiv.org/abs/2601.08778) found annotation error rates of about 53% on BIRD Mini-Dev and 63% on Spider 2.0-Snow, and correcting a slice reshuffled leaderboard rankings by as many as 9 places. The lesson for your system: pair execution accuracy with component checks and a sampled human review, and treat no single automated score as ground truth. [[annotation errors in text-to-SQL benchmarks, 2026](https://arxiv.org/abs/2601.08778)]

**Three ways to measure a behavior.** Every metric is implemented one of three ways. Reach for the cheapest and most reliable one the behavior allows.
- **Code-based metrics.** Deterministic checks: does the SQL parse, is it a single read, does it execute without error, does its result set match the gold query's rows (execution accuracy), is the returned value in a plausible range. Fast, reliable, cheap. Use wherever correctness is objectively checkable, and compare against a reference dataset of question-to-gold-SQL pairs here.
- **LLM judges.** One model scoring another against an explicit rubric, for the qualities code cannot capture: does the answer actually address the question the user asked, are two different-looking result sets semantically equivalent, was the ambiguity resolved reasonably. Scalable, and a new source of non-determinism, so calibrate before you trust it (below).
- **Human evaluation.** The gold standard you calibrate the other two against, and the source of your gold queries in the first place. Too slow to run on all traffic, so you sample: calibration, hard multi-join schemas, and disagreements between the code check and the judge.

Most real systems use all three together.

**Offline: per-component evaluation metrics (the pre-deployment gate).** Build a reference dataset of labeled question-to-gold-SQL pairs across easy aggregations, medium filters and groupings, hard multi-join analytics, ambiguous questions, and out-of-scope and adversarial questions. A validated gold query per question is the asset the whole eval rests on. Then score each component with metrics chosen from its failure modes, drawing from this menu rather than instrumenting all of it.

| Component | Failure mode | Candidate metrics | Method |
|---|---|---|---|
| **Schema linking** | misses the table or column the answer needs | table recall, column recall, schema-linking precision | code-based against the columns the gold query uses |
| **SQL generation** | invalid SQL, wrong join, wrong filter, wrong metric | SQL validity (parses), execution accuracy vs gold, exact-set match, [valid-efficiency](https://arxiv.org/abs/2305.03111) | code-based against gold rows + a dialect parser |
| **Answer** | runs but answers a different question, wrong number stated | answer-addresses-question, numeric correctness, semantic result equivalence | LLM judge with rubric + human sample |
| **Safety** | executes a write, runs a destructive or injected query, leaks restricted rows | read-only-block rate, injection-resistance rate, restricted-row-leak rate | adversarial suite + code checks |
| **Abstention** | answers when it should ask or abstain, or abstains on an answerable question | abstention precision, abstention recall, false-answer rate | labeled should-answer vs should-abstain set |
| **End to end** | task not actually resolved, unreliable across retries | task success, pass@1, **pass^k**, questions-to-resolution | scenario suite + judge |

That table is deliberately broad so you know the landscape. Treat it as a reference to draw from: these are the metrics teams typically reach for, and your job is to choose wisely and instrument only the few that earn their place. A dashboard with 30 metrics and no owner tells you nothing. For this product, execution accuracy against gold queries and the false-answer rate are the two highest-signal metrics, because they measure the failure that erodes trust fastest, and you report **pass^k** (success on all k independent tries) alongside the average, because a copilot that gets a number right 3 times in 4 is not one a finance team will trust.

**Choosing which metrics to actually run: Impact, Reliability, Cost.** Running every metric is expensive, so score each candidate on three dimensions and keep the high-signal ones.
- **Impact:** does it reveal an actionable problem, or is it merely interesting?
- **Reliability:** human evaluation and validated code checks are high; a calibrated LLM judge is medium; an uncalibrated automated score is low.
- **Cost:** a code check (parse, execute, compare rows) is cheap; a judge call is medium; detailed human review of a hard query is expensive.

High impact and low cost are the must-haves (SQL validity, the read-only block, execution accuracy on a fixed gold set). High impact and high cost are strategic investments you run on a sample (a calibrated judge on answer-addresses-question, human review of hard schemas). Low impact and high cost you drop.

**Online: guardrails vs the improvement flywheel.** Live evaluation does two different jobs, and conflating them is a common mistake.
- **Guardrails (online, real-time).** The few checks whose failure would immediately hurt the business, run inline so the system can act the moment they trip: a query that is not a single read, a query that fails validation after the retry cap, a low-confidence or ambiguous question. The action is immediate (block the query, abstain, ask a clarifying question). Guardrails must be fast and reliable before sophisticated.
- **Improvement flywheel (offline, batch).** Everything else: correctness trends on a sampled slice, which question types fail most, drift as the schema evolves. This is how the system gets better over weeks, while the guardrails handle the disasters in the moment.

The metrics you watch on live traffic:

| Metric | Job | Why it matters |
|---|---|---|
| read-only-block rate | guardrail | every non-read query must be blocked before execution |
| SQL validity / execution-error rate | guardrail + flywheel | a spike means generation quality dropped or the schema changed under you |
| self-correction rate | flywheel | how often the first query fails; a rising trend flags a regression |
| abstention rate | guardrail + flywheel | too high means it is not helping; too low risks confident wrong answers |
| answered-question rate | flywheel | share of questions resolved without a human, the core outcome |
| thumbs-up rate on answers | flywheel | trust as the user actually feels it, the number that decides adoption |
| p50 / p95 latency | flywheel | user experience, budgeted across linking, generation, and execution |
| query cost, rows scanned | flywheel | unit economics and warehouse load, the number finance and data eng ask about |
| restricted-row exposure | guardrail | row-level access holds under real questions (Follow-up 4) |

**Trust the judge, then close the discovery loop.** Calibrate an LLM judge before it gates anything: hand-label a few hundred query-answer pairs, run the judge on the same set, measure agreement (percent agreement or Cohen's kappa), and refine the rubric until it aligns with the humans. Re-calibrate whenever you change the model, and run the judge on a sampled percentage of live traffic to control cost.

Then run the discovery loop, because users will always ask questions your metrics were never built for. Sample live traffic on **signals** (thumbs-down, a user re-asking the same question differently, an analyst overriding the number, abandonment). When a signal keeps firing but your metrics look clean, that gap is the tell: a human reads those traces, names the quality dimension you were not measuring (a whole class of questions the semantic layer defines wrong, say), and it becomes a new metric and new gold queries added back into the reference set. Evaluation is never finished. You build for the failures you can anticipate, and you monitor to discover the ones you cannot.

![Eval and discovery loop: pre-deployment validation, then production monitoring, with the reference set growing each cycle](../assets/eval_discovery_loop.png)

Instrument the LangGraph app with OpenInference so every node, model call, and query execution becomes a span in **Arize** (Phoenix / AX), which is where the online evals and drift alerts run. Reading traces is how you find the exact step where the copilot's SQL diverged from what the question needed. Tooling worth knowing: Ragas and DeepEval for component metrics, promptfoo for CI gates, Arize Phoenix for tracing and online evals. *Deeper:* [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md), our [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first), and the [Evaluation topic](../../../topics/evaluation.md).

**Layer 4, production and ops.** The loop that keeps it alive at scale: schema indexing that refreshes as the warehouse changes, row-level access control on every query, a read replica so analytics load never touches production writes, latency budgets, and observability so every query is traceable. Detail is in Follow-ups 1, 3, and 4.

**Layer 5, optimization.** Where a working system gets better and takes on harder problems: model routing (a cheap model for the easy majority, a strong model for multi-join analytics), caching the schema prompt and validated queries, advanced schema retrieval for very large catalogs, and multi-agent. Routing, caching, and better schema linking pay off first, and for complex analytical questions a decomposition orchestrator over a schema-linking sub-agent and an independent verification critic is the design to reach for, which Follow-up 5 lays out.

Composing these layers into one coherent system is exactly what Step 3 does.

### Step 3: the architecture

You do not have to build this in one shot, and it helps to say how you would get there: **start at high control and move toward high agency, with evaluation at every step.** Begin with the most constrained thing that could work (link the schema, generate one query, validate and execute it read-only, abstain when it fails), prove it with evals, and only hand the model more freedom (self-correction retries, self-consistency sampling, letting it explore a large schema) once the measurement says the constrained version is solid and its limits are what block you. You earn agency through evaluation before you grant it. Every increase in autonomy is paid for with an eval that shows it helped.

Composed, the layers give one architecture:

```
┌────────────────────────────────────────────────────────────────────┐
│ Observability: every node, model call, and query is a span (Arize) │
└────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────┐
  │           User            │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │      Input guardrail      │
  │     injection, scope      │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │        Schema link        │
  │      hybrid retrieve      │
  │     tables + columns      │
  │     + semantic layer      │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │       Generate SQL        │
  │   linked schema + defs    │
  │   + few-shot exemplars    │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │     Static guardrail      │
  │ single read-only SELECT?  │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │       Safe execute        │
  │ read-only role + replica, │
  │     timeout + row cap     │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │         Validate          │
  │       rows present?       │
  │   answers the question?   │
  └───────────────────────────┘
                ▼
  ┌───────────────────────────┐
  │     Output guardrail      │
  └─────────────┬─────────────┘
              ┌─┴───────────────────┐
          yes ▼                     ▼ no / abstain
  ┌──────────────────────┐  ┌───────────────┐
  │        Answer        │  │    Abstain    │
  │ the number + the SQL │  │ human analyst │
  └──────────────────────┘  └───────────────┘
```

Read it as the spine composed: the model (layer 1), wrapped in schema-and-definition knowledge, SQL-generation and safe-execution tools, and a memory of validated queries (layer 2), gated by evals (layer 3) that run inline as guardrails and offline as a flywheel, run under production observability (layer 4), with optimization (layer 5) held to what this system actually needs, which is routing, caching, and stronger schema retrieval. Composing exactly these pieces into one coherent system is the system design. The static guardrail and the read-only sandbox are what make execution safe, and the validation step is what makes abstention the default: when the copilot cannot produce a query that runs and answers the question, it hands off to an analyst with the failed query attached rather than guessing a number.

### The runnable example

[`code/`](code/) is a small, provider-agnostic LangGraph implementation of this architecture (minus the scale pieces): schema linking over a tiny in-memory warehouse, read-only SQL generation, a static guardrail that blocks anything that is not a single SELECT, execution in a read-only sandbox (`PRAGMA query_only`), execution-guided self-correction when a query errors, and abstention when the question is out of scope or returns no rows. It runs offline with deterministic logic, so it needs no API key to try.

```bash
cd code && pip install -r requirements.txt
python run.py                                 # run the scenarios (also a self-test)
python run.py "Which region has the most revenue?"    # ask the copilot your own question
```

Set a model and any provider key (OpenAI, Anthropic, Gemini, and more) to route schema linking, SQL generation, and answering through a real model; the guardrail and the read-only sandbox apply to model-written SQL exactly as they do offline. LangGraph is one choice of framework; if you prefer another, ask your coding agent to reimplement the same design in the SDK you use (the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python). See [code/README.md](code/README.md).

---

## Follow-up 1: "Now the warehouse has thousands of tables. Where does it break, and how do you scale it?"

Name where it breaks first: schema linking, ahead of generation. A schema that fit in one prompt no longer does, so the copilot cannot see every candidate table, and accuracy falls because the generator is working blind or drowning in distractor columns.

- **Schema retrieval (layer 2).** Turn linking into a real retrieval pipeline: index every table and column description, retrieve with hybrid dense plus BM25, fuse with Reciprocal Rank Fusion, and rerank the top tables with a cross-encoder before generation. This is the RAG pattern applied to the catalog, and it is why [AutoLink](https://arxiv.org/abs/2511.17190) frames large-schema linking as an agent that explores the catalog progressively rather than reading it all at once.
- **Routing and caching (layer 5, optimization).** Route easy, frequent questions to a small fast model; reserve the strong model for multi-join analytics. Cache the stable schema-and-instruction prefix, and cache validated queries so a repeated question skips generation entirely.
- **Governance (layer 4).** A [semantic layer](https://docs.getdbt.com/docs/build/about-metricflow) becomes load-bearing at this scale: it narrows thousands of raw tables to a curated set of governed metrics and entities, which both shrinks the retrieval space and raises correctness.

At scale, the catalog and the retrieval over it is the bottleneck, ahead of the model. *Deeper:* [Retrieval and RAG](../../../topics/rag.md).

---

## Follow-up 2: "A user tries to make it run a destructive or injected query. How do you prevent unsafe execution?"

Separate generating a query from running it, and lock down the running.

- **Read-only by construction.** The static guardrail allows a single SELECT and refuses writes, multi-statements, and destructive keywords. Behind it, the database credential physically cannot write, and the session is read-only, so an application bug cannot become a mutation. Allowlists over blocklists.
- **Untrusted input.** The user's question and any value read from the warehouse are untrusted, and the copilot never executes instructions found in them. The sharp way to see the risk is the **[lethal trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)**:

```
  The lethal trifecta

        ┌─────────────────┐   ┌────────────────┐   ┌────────────────────────┐
        │    Untrusted    │   │   Access to    │   │ Ability to run queries │
        │ question / data │   │ warehouse rows │   │    + return results    │
        └────────┬────────┘   └────────┬───────┘   └────────────┬───────────┘
                 └─────────────────────┼────────────────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │ All 3 together is dangerous │
                        └─────────────────────────────┘

           any 2 are manageable; keep all 3 from meeting in one path
```

- **Blast radius.** A statement timeout and a row cap bound the damage of any single query, [least-privilege](https://en.wikipedia.org/wiki/Principle_of_least_privilege) roles keep the copilot off tables it should never see, values are parameterized rather than string-concatenated ([SQL injection](https://owasp.org/www-community/attacks/SQL_Injection)), and an immutable log records every executed query. The worst a jailbreak achieves is a wasteful read that the timeout kills.

*Deeper:* [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and Security](../../../topics/safety-security.md).

---

## Follow-up 3: "Cut p95 latency in half without hurting correctness."

Budget latency per stage (schema linking, generation, execution) against a target such as p95 under a few seconds, then attack the largest stage.

- Cache the stable schema-and-instruction prefix, and cache validated queries so repeated questions return instantly.
- Route easy questions to a smaller model and reserve the strong model for the hard, multi-join path.
- Cap self-correction retries and reserve self-consistency sampling for high-stakes questions, because both trade latency for reliability.
- Push work into the warehouse: a tight query that scans fewer rows is faster and cheaper than a broad one, so favor generation that filters early.

> **Real number: [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).** Caching the stable prefix of a prompt (system instructions, schema, few-shot exemplars) can cut cost by up to about 90% and latency by up to about 85% on the cached portion, per provider documentation. A text-to-SQL prompt carries a large, identical schema block on every call, so this is the single highest-leverage optimization here.

Correctness holds because the eval set from Layer 3 gates each change: if a faster route drops execution accuracy on the gold set, it does not ship. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 4: "Finance can see all rows; a regional manager may see only their region. How does the design change?"

That single constraint promotes 2 normally-optional components into load-bearing walls: **row-level access control** and an **audit trail**. The copilot must run every query under the asking user's permissions, so it either passes through the warehouse's row-level security or injects a mandatory tenant or region filter it cannot be talked out of, and it logs who asked what and which rows were returned. The generation and execution boxes do not change shape; the execution role and the query context become per-user. This is the general pattern: a governance constraint is what forces you to engineer a layer you would otherwise accept as a framework default. *Deeper:* [Production and LLMOps](../../../topics/production.md).

---

## Follow-up 5: "When would you use multiple agents, and how would you extend this design?"

For a complex analytical question, the design to recommend splits the work across agents: a decomposition orchestrator that breaks the question into sub-queries, a schema-linking sub-agent that finds the right tables and columns for each, and an independent verification critic that executes the SQL read-only and checks the result before it ships. A simple lookup against one curated schema stays a single agent with the two tools, because one coherent thread needs no coordination, and you let the traces show when a question has grown complex enough to decompose. Here is when multi-agent (layer 5, optimization) pays off and how the architecture extends.

**When multi-agent earns its place:**
- **The work decomposes into distinct roles.** A schema-exploration agent that navigates a very large catalog, a SQL-writer agent, and a verifier agent that executes and checks the result each want their own context and prompt. This mirrors the refiner pattern in [MAC-SQL](https://arxiv.org/abs/2312.11242) and the planner-generator-evaluator split that keeps a harsh verifier out of the writer's context.
- **The question spans several warehouses or domains.** A question that joins the sales warehouse and the product warehouse wants a specialist per source that federates results, rather than one agent holding every schema at once.
- **The context window is the bottleneck.** As the schema and exemplars grow, one generalist runs out of room, while specialists each stay within budget.
- **The task is valuable enough to pay for it.** Multi-agent trades tokens for capability, which fits high-value analytics.

**How you would extend this architecture.** Keep the single-agent design intact and put a decomposition orchestrator in front of specialists. The orchestrator splits a complex question into sub-queries and allocates them; a schema-linking sub-agent maps each sub-query to the right tables and columns; and an independent verification critic executes the generated SQL read-only and checks the rows against the question before anything returns. The critic sees only the query and the result, never the writer's reasoning, so it stays a genuine second opinion, the self-correction split behind [MAC-SQL](https://arxiv.org/abs/2312.11242). The same input and output guardrails, evals, safe-execution sandbox, and observability wrap the whole system across agents.

```
┌─────────────────────────────────────────┐
│ Observability: spans across every agent │
└─────────────────────────────────────────┘

                                           ┌──────────────────┐
                                           │       User       │
                                           └──────────────────┘
                                                     ▼
                                           ┌──────────────────┐
                                           │ Input guardrail  │
                                           │ injection, scope │
                                           └──────────────────┘
                                                     ▼
                                      ┌────────────────────────────┐
                                      │ Decomposition orchestrator │
                                      │  split into sub-queries,   │
                                      │     allocate, compose      │
                                      └──────────────┬─────────────┘
                       ┌─────────────────────────────┴─────────────────────────────┐
                       ▼                             ▼                             ▼
         ┌──────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
         │      Schema-linking      │  │        SQL-writer        │  │   Verification critic    │
         │        sub-agent         │  │        sub-agent         │  │   independent: execute   │
         │      map sub-query       │  │  generate SQL for each   │  │  read-only, check rows   │
         │     to tables / cols     │  │   linked table subset    │  │     vs the question,     │
         │                          │  │                          │  │ sees only query + result │
         └─────────────┬────────────┘  └─────────────┬────────────┘  └─────────────┬────────────┘
                       └─────────────────────────────┬─────────────────────────────┘
                                                     ▼
                                           ┌──────────────────┐
                                           │ Output guardrail │
                                           │ correct & safe?  │
                                           └─────────┬────────┘
                                        ┌────────────┴───────┐
                                    yes ▼                    ▼ no / ambiguous
                              ┌──────────────────┐      ┌─────────┐
                              │ Answer + the SQL │      │ Abstain │
                              └──────────────────┘      │ analyst │
                                                        └─────────┘

  the critic sees only the query and result, a genuine second opinion; specialists keep tight contexts
```

> **Real data: Anthropic's multi-agent research system.** A lead-plus-subagents setup beat a single agent by about 90.2% on their research eval, while using about 15x the tokens, and token usage explained roughly 80% of the performance gap. The takeaway is to spend those tokens where task value and genuine parallelism justify them, and to stay single-agent where they do not. [[Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)]

The honest rule: start with the single agent, instrument it, and let the traces tell you when a huge schema or a cross-warehouse question has outgrown one agent. Multi-agent is where this design goes once the problem is big enough to need it. *Deeper:* [AI Agents](../../../topics/agents.md).

---

## Follow-up 6: "Point it at a different warehouse that speaks a different SQL dialect."

The architecture does not change; the data and the evaluation do. Generation targets the new dialect, validation uses a [dialect-aware parser](https://github.com/tobymao/sqlglot) set to it, the few-shot exemplars are rewritten in that dialect, and the eval set gains labeled gold queries in the new dialect so correctness is measured per dialect rather than assumed. Schema linking, guardrails, and the safe-execution pattern carry over unchanged. This is the recurring lesson: most follow-ups of this shape are answered in the data and eval layers, and the box diagram stays the same. *Deeper:* [Evaluation and Observability](../../../topics/evaluation.md).

---

## Follow-up 7: "A better code model drops next quarter. How do you avoid a rewrite?"

Text-to-SQL quality tracks model capability closely, so you want to adopt a stronger model the day it lands. Keep the model behind the provider-agnostic interface (the runnable code already does this, any model via one call), pin behavior with the gold-query eval set rather than with brittle prompt hacks, and pair every "do not do X" rule with an eval so you can delete the rule when a better model makes it obsolete. When the new model lands, you swap it, re-run the eval gate on your gold queries, and keep the parts that still earn their place. A harness that ages out did its job: it fit the old model, and the frontier moved. *Deeper:* [AI Agents](../../../topics/agents.md).

---

## Real-world reference points

- **Spider 2.0 (enterprise workflows):** on realistic enterprise text-to-SQL over warehouses with 1,000-plus-column schemas across BigQuery, Snowflake, and DuckDB, purpose-built agent systems reach roughly 74% on Spider 2.0-Lite and above 90% on the Snowflake split as of 2026, while a frontier model behind a generic scaffold lands near 42% on Lite and 26% on Snowflake. The engineering around the model closes that gap, which is why correctness is the center of the design. [[Spider 2.0 leaderboard, 2026](https://spider2-sql.github.io/)] [[Spider 2.0 paper](https://arxiv.org/abs/2411.07763)] [[repo](https://github.com/xlang-ai/Spider2)]
- **Enterprise schemas are still the hard part:** the classic single-database Spider is largely solved (top systems above 90% execution accuracy), and the difficulty moved to large real schemas, multi-dialect workflows, and the business definitions the columns do not carry. [[Spider 2.0](https://arxiv.org/abs/2411.07763)]
- **Benchmark noise:** a 2026 audit found annotation error rates near 53% on BIRD Mini-Dev and 63% on Spider 2.0-Snow, and correcting a slice moved leaderboard rankings by up to 9 places. Execution accuracy is necessary and not sufficient; pair it with human calibration and verify the returned number. [[annotation errors, 2026](https://arxiv.org/abs/2601.08778)]
- **Enterprise adoption:** every major BI platform has shipped a natural-language interface (for example [Copilot in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/fundamentals/copilot-fabric-overview)), which is why the design bar is a governed, trustworthy answer rather than a demo.
- **Prompt caching:** up to about 90% cost and 85% latency reduction on the cached portion; the highest-leverage optimization given a large, identical schema block on every call.

---

## Research to know

- [Spider](https://arxiv.org/abs/1809.08887) (Yu 2018): the cross-domain text-to-SQL benchmark that set execution accuracy as the measure.
- [BIRD](https://arxiv.org/abs/2305.03111) (Li 2023): text to SQL over large, messy real databases, and the benchmark that put a human expert baseline on the task.
- [Spider 2.0](https://arxiv.org/abs/2411.07763) (2025): enterprise-scale workflows where a raw model still falls short and engineered agents now lead; see the [live leaderboard](https://spider2-sql.github.io/).
- [Annotation errors break text-to-SQL leaderboards](https://arxiv.org/abs/2601.08778) (2026): why a benchmark score, on BIRD and Spider 2.0 alike, needs human calibration before you trust it.
- [RESDSQL](https://arxiv.org/abs/2302.05965) (2023): decoupling schema linking from SQL skeleton parsing.
- [DIN-SQL](https://arxiv.org/abs/2304.11015) (2023): decomposed in-context generation with self-correction.
- [MAC-SQL](https://arxiv.org/abs/2312.11242) (2023): a multi-agent framework whose refiner executes and rewrites on error.
- [Self-consistency](https://arxiv.org/abs/2203.11171) (Wang 2022): sample several candidates and take the answer they agree on.
- [Context-aware bidirectional schema linking](https://arxiv.org/abs/2510.14296) and [AutoLink](https://arxiv.org/abs/2511.17190) (2025): schema linking as retrieval and as exploration at scale.
- [AmbiSQL](https://arxiv.org/abs/2508.15276) and [expected-information-gain disambiguation](https://arxiv.org/abs/2507.06467) (2025): detecting and resolving ambiguous questions.
- [A survey of LLM-based text to SQL](https://arxiv.org/abs/2412.05208) (2024): the landscape of benchmarks, methods, and open problems.

## Further reading

The papers above are the ideas; these are the practice of assembling them. Start inside this repo.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the repo topic pages per layer ([RAG](../../../topics/rag.md), [Agents](../../../topics/agents.md), [Evaluation](../../../topics/evaluation.md), [Production and LLMOps](../../../topics/production.md), [Safety and Security](../../../topics/safety-security.md)); [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Harness Engineering](../../../resources/harness_engineering.md); Aishwarya's YouTube ([AI Engineering: A Realistic Roadmap for Beginners](https://www.youtube.com/watch?v=pAXbl1EBHJ8), [Stop Building AI Like Traditional Software](https://www.youtube.com/watch?v=GAF_ychy32k), the [full channel](https://www.youtube.com/@aishwaryanr4606)); and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first).
- [MotherDuck, your data model is the semantic layer](https://motherduck.com/blog/bird-bench-and-data-models/), and [dbt MetricFlow](https://docs.getdbt.com/docs/build/about-metricflow) on governed metric definitions.
- [Snowflake, Arctic-Text2SQL-R1](https://www.snowflake.com/en/blog/engineering/arctic-text2sql-r1-sql-generation-benchmark/), a reinforcement-learned open text-to-SQL model and a strong open baseline.
- [sqlglot](https://github.com/tobymao/sqlglot) for dialect-aware parsing and transpilation, and the [OWASP SQL injection](https://owasp.org/www-community/attacks/SQL_Injection) guidance for safe execution.
- Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) and [Multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system).

---

## Related in this repo

Topics: [RAG](../../../topics/rag.md) · [Agents](../../../topics/agents.md) · [Evaluation](../../../topics/evaluation.md) · [Production and LLMOps](../../../topics/production.md) · [Safety and Security](../../../topics/safety-security.md). Guides: [Harness Engineering](../../../resources/harness_engineering.md) · [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md). Prepping to be asked this? See the [interview prep hub](../../README.md).
