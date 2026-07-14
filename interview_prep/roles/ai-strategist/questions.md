# Question Bank

88 questions with concise model answers, grouped by theme. Answers are anchors, not scripts: know the mechanism and the trade-off so you can defend a follow-up. Every question is a click-to-open collapsible, and every answer ends with a place to go deeper. Pair this with the repo's [60 GenAI interview questions](../../60_gen_ai_questions.md) and the [role-based prep index](../../README.md).

Themes: [A. AI fundamentals and fluency](#a-ai-fundamentals-and-fluency) - [B. RAG, fine-tuning, and knowledge](#b-rag-fine-tuning-and-knowledge) - [C. Agents, reasoning, MCP, context](#c-agents-reasoning-mcp-and-context) - [D. Evaluation and quality](#d-evaluation-and-quality) - [E. Cost, latency, and production](#e-cost-latency-and-production) - [F. Business, ROI, prioritization, build-vs-buy](#f-business-roi-prioritization-and-build-vs-buy) - [G. Risk, responsible AI, governance](#g-risk-responsible-ai-and-governance) - [H. Change management and org](#h-change-management-and-org)

---

## A. AI fundamentals and fluency

<details>
<summary><b>1. Explain how a large language model works to a non-technical executive.</b></summary>

A large language model is a system trained on a very large amount of text to predict the next piece of a sentence, and by doing that at scale it learns patterns of language, facts, and reasoning steps. You give it instructions and context in plain language, and it produces a response. The two things an executive must internalize: it is probabilistic, so it can be confidently wrong (hallucinate), and it only knows what was in its training data plus what you put in front of it right now. Everything in an AI strategy flows from managing those two facts.

**Learn more:** [Foundations topic](../../../topics/foundations.md) and the [Understand AI journey](../../../journeys/understand.md)

</details>

<details>
<summary><b>2. What is the difference between generative AI and the traditional automation companies already have?</b></summary>

Traditional automation follows fixed rules a human wrote, so it is predictable and brittle: it does exactly what it was told and breaks on anything new. Generative AI produces new outputs (text, code, images) from patterns and handles ambiguity and language, which is why it can touch knowledge work that rules could never automate. The trade-off is that it is non-deterministic and needs evaluation and guardrails rather than a spec. Strategically, use rules where the process is well-defined and stakes are high, and use generative AI where the work is language-heavy, varied, and tolerant of review.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>3. What can LLMs reliably do today, and what should organizations still not trust them with?</b></summary>

They are reliable for drafting, summarizing, classifying, extracting structured data, translating, answering questions over provided documents, and writing and explaining code, all with a human reviewing high-stakes output. They are not reliable as a source of truth from memory, for exact arithmetic or fresh facts without tools, or for fully autonomous high-consequence decisions with no oversight. The pattern that works: use the model for the language-heavy first draft or the triage, and keep a human or a deterministic check on the final commit. A strategist earns trust by being precise about that line.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>4. How do you tell a good AI use case from a bad one?</b></summary>

A good use case has a clear business metric it moves, tolerates occasional error or has a cheap review step, has the data available, and would be too expensive or slow to solve with rules. A bad one needs perfect accuracy with no human check, depends on data you do not have or cannot use, or automates something that was never a real bottleneck. Favor high-volume, language-heavy, review-friendly workflows for the first wins. The fastest disqualifier is the absence of a metric: if no one can say what number should move, it is a demo, not a use case.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>5. What is a token, a context window, and why do they matter to strategy?</b></summary>

A token is a chunk of text (roughly three-quarters of a word) that the model reads and generates, and you pay per token in and out. The context window is how many tokens the model can consider at once, which bounds how much document, history, and instruction you can supply in a single call. They matter because cost, latency, and how much knowledge you can stuff into a prompt all scale with tokens. A strategist uses this to sanity-check vendor claims and to understand why long-document workflows cost more and run slower.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>6. What changed between the 2023 wave of generative AI and where we are in 2025-2026?</b></summary>

Context windows grew large, costs per token fell sharply, and reasoning models arrived that spend extra compute to think before answering, which made multi-step math, code, and analysis genuinely usable. Agents matured from demos toward production, tool and data connection got standardized through protocols like MCP, and evaluation and governance became mainstream concerns rather than afterthoughts. Capability is no longer the main blocker for most enterprise use cases; adoption, data, and change management are. A strategist who is still pitching 2023-era assumptions will misjudge both what is now feasible and where the real risk sits.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>7. What is prompt engineering, and how much does it still matter?</b></summary>

Prompt engineering is writing the instruction and examples that steer the model toward the output you want, including role, task, constraints, format, and few-shot examples. It still matters and is often the cheapest lever, but the field has moved toward context engineering: deciding what information (retrieved documents, tool results, memory) actually enters the window, not just how you phrase the ask. For strategy, the takeaway is that large quality gains are often available before any model change or fine-tune, which lowers cost and time to value. Treat prompt and context work as the first thing to exhaust, not the last.

**Learn more:** [Prompting topic](../../../topics/prompting.md) and Anthropic's [context engineering guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>8. A vendor claims 99% accuracy. What do you ask?</b></summary>

Accuracy on what task, measured against what ground truth, on whose data, and in a sandbox or in production. Ask for a named production deployment with a real use case and a measured outcome, because a demo number is not a production number. Ask what happens on the other 1%, whether errors are caught, and what the cost and latency are at your volume. If they cannot name a real customer, a real metric, and a real failure mode, treat the number as marketing.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>9. What is an agent harness, and why does the system around the model often matter more than the model itself?</b></summary>

The harness is everything assembled around the raw model to make it useful: the prompts and instructions, the tools it can call, retrieval and memory, the control loop, guardrails, and the evaluation and monitoring that keep it honest. Two teams using the same frontier model will get very different reliability depending on how well the harness is built, which is why capability gaps between top models matter less each year while system quality matters more. For a strategist this reframes vendor and build decisions: you are buying or building a system, and the durable advantage lives in data, integration, and the harness, not in whichever model is briefly ahead. Ask where the engineering effort actually sits before you attribute results to the model.

**Learn more:** [Harness engineering guide](../../../resources/harness_engineering.md) and Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>10. Open-weight versus proprietary frontier models: how should that choice shape strategy?</b></summary>

Proprietary frontier models (accessed by API) usually lead on raw capability and require no infrastructure, but you accept per-token pricing, data-handling terms, and less control over versioning. Open-weight models can be self-hosted for data residency, customized deeply, and run at predictable cost, at the price of carrying the serving, security, and MLOps burden yourself. The 2025-2026 reality is a portfolio: frontier API models for the hardest reasoning, and open-weight or smaller models for high-volume, privacy-sensitive, or cost-sensitive workloads. Decide per use case against data-residency needs, volume economics, and in-house depth rather than picking one camp for everything.

**Learn more:** [Foundations topic](../../../topics/foundations.md)

</details>

<details>
<summary><b>11. What is a small language model, and when does it beat a frontier model?</b></summary>

A small language model is a compact model (roughly 1 to 15 billion parameters) that is far cheaper and faster to run and can be fine-tuned or self-hosted, often serving 10 to 30 times cheaper than a large frontier model for the same volume. It wins when the task is narrow and well-defined (classification, extraction, routing, a bounded domain assistant), when latency or cost at scale is the binding constraint, or when data must stay on-premises. It loses on open-ended reasoning, broad world knowledge, and hard multi-step problems, where a frontier model still earns its cost. The mature pattern is a mix: a small model handles the predictable majority of traffic and escalates the genuinely hard minority to a larger model.

**Learn more:** [Fine-tuning topic](../../../topics/fine-tuning.md) and a [survey on small-and-large model collaboration](https://arxiv.org/abs/2505.07460)

</details>

<details>
<summary><b>12. What is multimodal AI, and where does it change the enterprise use-case map?</b></summary>

Multimodal models take and produce more than text: images, documents with layout, audio, and increasingly video, so a single model can read a scanned invoice, describe a photo, or transcribe and summarize a call. This opens use cases that were previously separate pipelines, such as document understanding over messy PDFs, visual quality inspection, and voice interfaces, often with less bespoke engineering than before. Strategically it expands the opportunity map into operations, insurance, healthcare, and field work where the source material was never clean text. Evaluate these use cases with the same discipline, because multimodal outputs hallucinate and need grounding and review just as text does.

**Learn more:** [Multimodal topic](../../../topics/multimodal.md)

</details>

<details>
<summary><b>13. Why are public benchmarks and leaderboards misleading for enterprise model selection?</b></summary>

Public benchmarks measure generic tasks that rarely match your workflow, data, or quality bar, and popular ones leak into training data over time, which inflates scores without improving real performance. A model that tops a leaderboard can still lose on your documents, your latency budget, or your compliance constraints. The reliable path is to build a small evaluation set from your own representative cases and score candidate models on that, weighing accuracy against cost, latency, context window, and data-residency needs. Treat leaderboards as a rough shortlist filter, then decide on your own numbers.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md) and Evidently's [LLM evaluation guide](https://www.evidentlyai.com/llm-guide/llm-evaluation)

</details>

<details>
<summary><b>14. What do post-training methods like instruction tuning and RLHF actually do, and why does it matter to strategy?</b></summary>

Pretraining teaches raw language and world patterns; post-training then shapes the model into something usable, through instruction tuning (teaching it to follow directions) and reinforcement learning from human or AI feedback (teaching it to prefer helpful, safe, well-formatted answers). This is why two models with similar raw ability can feel very different in tone, refusal behavior, and reliability. For strategy it explains why model updates can change behavior in ways that break your prompts and evals, and why enterprise fine-tuning usually adjusts this post-training layer rather than retraining from scratch. Track model versions and re-run evals on every update, because behavior is not frozen.

**Learn more:** [Fine-tuning topic](../../../topics/fine-tuning.md)

</details>

---

## B. RAG, fine-tuning, and knowledge

<details>
<summary><b>15. When would you use RAG instead of fine-tuning to give a model new knowledge?</b></summary>

Use retrieval-augmented generation when the knowledge changes often, is large or proprietary, or needs citations, because retrieval keeps answers current and auditable without retraining. Use fine-tuning to change behavior, format, tone, or to bake in a stable skill, not to inject volatile facts. In practice you often do both: RAG for the knowledge, light fine-tuning for the behavior. For most enterprise document-Q&A problems, RAG is the right first answer.

**Learn more:** [RAG topic](../../../topics/rag.md) and [Fine-tuning 101](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>16. Walk me through how a RAG system actually works.</b></summary>

Documents are split into chunks and converted to embeddings (numeric vectors capturing meaning) stored in a vector index. At query time the user's question is embedded, the most similar chunks are retrieved, and those chunks are inserted into the prompt so the model answers grounded in them, ideally with citations. Quality depends heavily on the unglamorous parts: chunking, retrieval quality, and reranking, not just the model. A hybrid of keyword and vector search plus a reranker usually beats naive vector search alone.

**Learn more:** [RAG topic](../../../topics/rag.md), [Agentic RAG 101](../../../resources/agentic_rag_101.md), and the [original RAG paper](https://arxiv.org/abs/2005.11401)

</details>

<details>
<summary><b>17. Why do RAG systems fail in production, and how do you de-risk them?</b></summary>

The common failures are retrieval misses (the answer-bearing chunk is never fetched), stale or messy source data, poor chunking that splits the answer, and the model ignoring or over-trusting the retrieved context. You de-risk by evaluating retrieval and generation separately, cleaning and maintaining the source corpus, adding reranking, and measuring faithfulness so answers stay grounded. Most RAG quality problems are data and retrieval problems, not model problems. A strategist who knows this will fund data readiness instead of a bigger model.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md) and the [RAG roadmap](../../../resources/RAG_roadmap.md)

</details>

<details>
<summary><b>18. When is fine-tuning genuinely the right call, and what does it cost?</b></summary>

Fine-tuning is right when you need a consistent behavior, style, or format, when you have enough high-quality labeled examples, or when a smaller fine-tuned model can match a larger one more cheaply at high volume. The costs are data preparation, training and evaluation cycles, and the ongoing burden of re-tuning as needs change, plus the risk of the model forgetting general ability. It is the wrong tool for fresh or frequently changing facts, where RAG wins. Parameter-efficient methods like LoRA make it far cheaper than full fine-tuning and are the sensible default.

**Learn more:** [Fine-tuning 101](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>19. What is the practical decision order for adding capability: prompt, RAG, fine-tune, or agent?</b></summary>

Start with prompting and context engineering because it is the cheapest and fastest, and it often clears the bar. Add RAG when the model needs knowledge it does not have and that knowledge changes or needs citations. Fine-tune when you need a stable behavior or a cheaper high-volume model and have the data. Reach for an agent only when the task genuinely needs multiple steps, tools, and adaptation, because each step up adds cost, latency, and new failure modes. Climbing the ladder without exhausting the lower rungs is a common way budgets get wasted.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>20. What is agentic RAG, and how does it differ from classic RAG?</b></summary>

Classic RAG runs one retrieval step and then answers, which is fast but fails when a question needs several lookups, reformulation, or reasoning about what to fetch next. Agentic RAG puts a model in a loop that can decide when to retrieve, rewrite the query, search multiple sources, and check whether it has enough before answering. It handles harder, multi-hop questions and messy sources better, at the cost of more latency, more tokens, and more failure surface. Use it when single-shot retrieval demonstrably misses, and keep evals on both the retrieval steps and the final answer.

**Learn more:** [Agentic RAG 101](../../../resources/agentic_rag_101.md) and the [Agentic search and retrieval table](../../../research_updates/agentic_search_retrieval_table.md)

</details>

<details>
<summary><b>21. Now that context windows are large and cheaper, when should you just stuff documents into context instead of building RAG?</b></summary>

Long context is the simpler choice when the relevant material is small enough to fit, fairly stable, and needed in full, because you skip the retrieval pipeline and its failure modes. RAG still wins when the corpus is large, changes often, needs citations, or when you want to control cost, since paying for a huge context on every call is expensive and models can lose focus in very long inputs. In practice many systems combine them: retrieve to narrow the field, then place the best material in a generous context window. Decide by corpus size, freshness, citation needs, and per-call economics rather than treating long context as a blanket replacement.

**Learn more:** [RAG topic](../../../topics/rag.md) and Anthropic on [context management](https://www.anthropic.com/news/context-management)

</details>

<details>
<summary><b>22. What is GraphRAG or knowledge-graph retrieval, and when is it worth the complexity?</b></summary>

GraphRAG builds a structured graph of entities and relationships from your documents and retrieves over that graph, rather than only fetching loosely similar text chunks. It helps with questions that require connecting facts across many documents or summarizing a whole corpus, where plain vector search returns fragments that miss the relationships. The cost is a heavier ingestion pipeline, graph construction and maintenance, and more moving parts to evaluate. Reach for it when multi-hop, cross-document reasoning is the actual need and standard RAG has measurably failed, not as a default.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md) and the [GraphRAG paper](https://arxiv.org/abs/2404.16130)

</details>

<details>
<summary><b>23. What data foundation must exist before RAG or fine-tuning can work?</b></summary>

You need source content that is accurate, current, deduplicated, and permissioned, plus a way to keep it fresh as it changes, because retrieval and tuning both amplify whatever is in the data. You also need access controls so the system only surfaces what a given user is allowed to see, and metadata to filter and route retrieval. Surveys in 2026 consistently put data quality and fragmentation as the top blocker to moving from pilot to production, and the fastest scalers finished data readiness before choosing a model. A strategist funds this plumbing first, because no model or prompt compensates for a broken corpus.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>24. What is model distillation, and when is it the right way to cut cost?</b></summary>

Distillation trains a smaller student model to imitate a larger teacher model, so you keep most of the quality on a target task at a fraction of the size and serving cost. It shines when you have a proven expensive model in production and high, steady volume, and you want to lock in cheaper, faster inference for a well-scoped task. The trade-offs are the engineering to build the distilled model, narrower generality than the teacher, and the need to re-distill as requirements shift. Treat it as a cost-optimization move after a use case is validated, not a way to start.

**Learn more:** [Fine-tuning 101](../../../resources/fine_tuning_101.md) and Google Cloud on [model distillation](https://cloud.google.com/discover/what-is-model-distillation)

</details>

---

## C. Agents, reasoning, MCP, and context

<details>
<summary><b>25. What makes something an agent rather than a single LLM call, and when do you actually need one?</b></summary>

An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or fresh data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow will do, because agents add cost, latency, and new failure modes like getting stuck in loops or taking wrong actions. The mature stance in 2025-2026 is that most value still comes from well-engineered workflows with narrow agentic steps, not fully autonomous agents.

**Learn more:** [Agents topic](../../../topics/agents.md) and [Agents 101](../../../resources/agents_101_guide.md)

</details>

<details>
<summary><b>26. What are the main failure modes of agents, and how do you contain them?</b></summary>

Agents fail by hallucinating a tool call, looping without progress, taking a wrong or irreversible action, compounding small errors over many steps, and running up cost and latency. You contain them by capping steps and tool calls, keeping a human in the loop for consequential actions, scoping tools tightly with permissions, adding checks between steps, and evaluating the whole trajectory, not just the final answer. The strategy lesson is that autonomy and control trade off directly: more autonomy means more value potential and more risk. Start narrow and expand autonomy only as evals and trust grow.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>27. What is the Model Context Protocol (MCP) and why should an enterprise care?</b></summary>

MCP is an open standard for connecting AI models and agents to tools and data sources through a common interface. It matters because you wire a capability or a data source once and reuse it across many agents and applications, instead of building a custom integration for every tool and every vendor. For an enterprise this reduces integration cost and lock-in and makes an internal AI platform far more composable. By 2026 it is backed across major providers, which is part of why standing up agent projects is cheaper than it was in 2024.

**Learn more:** [MCP explainer in the Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/part5_what_is_mcp_and_why_care.md) and the [MCP specification site](https://modelcontextprotocol.io/)

</details>

<details>
<summary><b>28. What is context engineering, and why can it matter more than model choice?</b></summary>

Context engineering is deciding what actually enters the model's context window: instructions, retrieved documents, tool results, and memory, and just as importantly what stays out. It matters because models degrade when the context is bloated, noisy, or poorly ordered, so getting the right, minimal, well-structured information in usually beats both clever phrasing and a bigger model. It is also a cost and latency lever, since every token in the window is paid for and slows the response. For strategy, it means a lot of quality is available through information design before you spend on models or fine-tuning.

**Learn more:** [Prompting topic](../../../topics/prompting.md) and Anthropic's [context engineering guide](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>29. How do reasoning models differ from standard LLMs, and what do they change for strategy?</b></summary>

Reasoning models are trained to spend extra compute on an internal chain of thought before answering, often using reinforcement learning on verifiable rewards. They trade higher latency and cost for much stronger performance on math, code, planning, and multi-step analysis. Strategically they expand the set of problems now worth attempting (complex analysis, agentic planning) but they are slower and pricier, so you route to them only for the hard steps and use cheaper models for the rest. Knowing when a problem needs reasoning versus a fast cheap model is a core cost-control decision.

**Learn more:** [Planning and reasoning models in the Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md) and the [chain-of-thought paper](https://arxiv.org/abs/2201.11903)

</details>

<details>
<summary><b>30. A team wants to build an autonomous multi-agent system. How do you respond?</b></summary>

First ask what business problem it solves and whether a single agent or a plain workflow would do, because multi-agent systems multiply cost, latency, and coordination failure. Multi-agent designs earn their keep for genuinely parallel or specialized sub-tasks, but they are harder to evaluate, debug, and govern, and many teams reach for them prematurely. Push for the thinnest version that delivers value: often one well-scoped agent with good tools and a human check. Fund the ambitious version only after evals prove the simple version works and the added complexity is justified.

**Learn more:** [Multi-agent systems in the Agentic AI Crash Course](../../../free_courses/agentic_ai_crash_course/part8_multi_agent_systems.md)

</details>

<details>
<summary><b>31. How would you keep an agent's cost and latency under control?</b></summary>

Cache the stable prefix of the context, keep the context minimal and well-ordered, and route sub-steps to cheaper or smaller models, reserving reasoning models for the hard parts. Cap tool calls and loop iterations, cache tool results, and give the model only as much reasoning budget as the step needs. Batch or stream where the user experience allows, and monitor cost and latency per request as first-class metrics. These are the levers that decide whether an agent is economically viable at scale.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>32. What is retrieval versus tool use versus memory in an agent, in plain terms?</b></summary>

Retrieval pulls relevant documents into the prompt so the model can ground its answer in current, specific knowledge. Tool use lets the model call external functions (search, a database, a calculator, an API) to fetch data or take actions it cannot do from memory. Memory lets the agent carry state across steps or sessions so it does not repeat work or lose context. A capable agent combines all three, and most real-world reliability comes from getting these plumbing pieces right rather than from the model itself.

**Learn more:** [Tools](../../../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md) and [memory](../../../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md) in the Agentic AI Crash Course

</details>

<details>
<summary><b>33. When do you actually need multiple agents instead of one, and what patterns are common?</b></summary>

You need multiple agents when a task splits into distinct specialties or genuinely parallel work, for example a triage agent that hands off to domain workers, or an orchestrator that fans out sub-tasks and combines results. The common patterns are handoff (one agent transfers control and context to a specialist) and orchestrator-worker (a lead agent delegates and integrates), each of which adds coordination cost and new failure modes like context loss on handoff. Most problems do not need this, and a single agent with good tools is easier to evaluate, debug, and govern. Add agents only when a single agent measurably cannot hold the task, and design the handoffs and shared state deliberately.

**Learn more:** Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) and [Agents 101](../../../resources/agents_101_guide.md)

</details>

<details>
<summary><b>34. How do you scope an agent's tools and permissions, and what is the governance-containment gap?</b></summary>

Give an agent the fewest tools and the narrowest permissions it needs, scope those permissions to the acting user, and require approval or a deterministic check before any high-consequence or irreversible action. The governance-containment gap is the widely reported 2026 problem where executives believe existing policies cover agent actions while, in practice, many agents run with broad access and little logging or oversight. Close it with an inventory of what each agent can touch, least-privilege access, full audit logging, and human approval gates on sensitive actions. Treat every tool an agent can call as attack surface and as a liability, not just a feature.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md) and [Safety and security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>35. Who does an agent act as, and why is agent identity and authorization a strategic issue?</b></summary>

An agent should act with a specific, auditable identity and only the access rights of the user or service it represents, so it cannot see or do more than that principal is entitled to. This matters because an agent that inherits broad service credentials becomes a way to bypass every existing access control, and because you need traceability to answer who did what when something goes wrong. Standards like MCP are pushing toward performing actions in the context of the current user, which improves both least privilege and audit trails. A strategist should insist that identity, scoping, and logging are designed in before an agent touches production systems.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md) and why [MCP is on executive agendas](https://www.cio.com/article/4136548/why-model-context-protocol-is-suddenly-on-every-executive-agenda.html)

</details>

<details>
<summary><b>36. Explain the agency-control trade-off and how you sequence autonomy safely.</b></summary>

Every increment of autonomy you give an agent raises both its potential value and its risk, because the same freedom that lets it handle more also lets it fail in more consequential ways. The safe path is to start with tight control (assistive suggestions, a human approving each action) and expand autonomy one step at a time, only as evals and monitoring prove the current level is reliable. Match the level of autonomy to the cost of a mistake: low-stakes, reversible tasks can run more freely, while high-stakes or irreversible ones keep a human in the loop. Sequencing autonomy this way is how you capture value without betting the business on an unproven system.

**Learn more:** [Harness engineering guide](../../../resources/harness_engineering.md) and [Agents topic](../../../topics/agents.md)

</details>

<details>
<summary><b>37. MCP is now widely adopted; what new risks does the protocol itself introduce?</b></summary>

Standardizing tool and data access makes agents far easier to build and also concentrates risk: a malicious or compromised MCP server can expose tools that poison the agent, and broadly scoped connectors become a single point of over-access. Reported concerns include tool descriptions carrying hidden instructions, servers you did not vet handling sensitive data, and weak authentication between agent and server. Manage it by using an MCP gateway or registry, vetting and pinning trusted servers, enforcing least-privilege scopes and per-user identity, and logging every call. The convenience of one common interface is worth it only with these controls, otherwise you have standardized your attack surface too.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md) and Anthropic's [MCP announcement](https://www.anthropic.com/news/model-context-protocol)

</details>

<details>
<summary><b>38. Human-in-the-loop, human-on-the-loop, or shadow mode: how do you choose the oversight design?</b></summary>

Human-in-the-loop means a person approves each consequential action before it happens, which fits high-stakes or irreversible steps. Human-on-the-loop means the system acts autonomously while a person monitors and can intervene, which fits higher-volume, lower-stakes work once it is trusted. Shadow mode runs the system alongside the current human process without acting, so you can compare its decisions to reality and build a baseline before you ever cut it in. Choose by the cost of an error and the maturity of your evals, and move from shadow to on-the-loop to fuller autonomy as evidence accumulates.

**Learn more:** [Agents topic](../../../topics/agents.md) and [AI agent observability](https://www.n-ix.com/ai-agent-observability/)

</details>

---

## D. Evaluation and quality

<details>
<summary><b>39. How would you evaluate whether an AI feature is good enough to ship?</b></summary>

Define what good means as measurable criteria tied to the business goal, build a representative labeled test set from real cases, and measure task success, error rate, and the cost of the errors that slip through. Use a mix of automated checks, human review on a sample, and an LLM-as-judge with a clear rubric for subjective quality, then set a threshold before you look at results so you are not moving the goalposts. Ship when the system beats the current baseline and the residual errors are affordable and caught.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>40. Why do you say evals are non-negotiable, and what happens without them?</b></summary>

Because generative systems are non-deterministic, evals are the only way to know if a change helped or hurt, whether the system is production-ready, and whether it is degrading over time. Without them, teams ship on vibes, cannot compare vendors or model versions, and discover regressions only when users complain. Evals also convert a fuzzy quality debate into a number executives can fund and track. A strategy that funds a build with no eval plan is funding a system no one can prove is working.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>41. How do you evaluate a RAG system specifically?</b></summary>

Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually fetched (recall and precision at k). For generation, measure faithfulness (is the answer grounded in the retrieved context with no fabrication), answer relevance, and correctness, using a labeled question set and an LLM-as-judge with a clear rubric. Then monitor the same signals in production, because source data drifts and yesterday's good retrieval decays.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md) and the [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

<details>
<summary><b>42. What is LLM-as-judge, and what are its limits?</b></summary>

LLM-as-judge uses a strong model to score outputs against a rubric, which scales evaluation of subjective qualities like helpfulness or faithfulness far more cheaply than human raters. Its limits: it can be biased toward verbose or confident answers, inconsistent without a tight rubric, and it can share blind spots with the model being judged. You control for this by writing clear rubrics, calibrating the judge against human labels on a sample, and keeping humans in the loop for high-stakes calls. It is a force multiplier for evaluation, not a replacement for human judgment on what matters.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>43. How do you monitor an AI system after launch?</b></summary>

Track quality signals continuously (task success, faithfulness, user feedback, escalation and override rates), plus operational metrics (latency, cost per request, error and timeout rates) and safety signals (harmful or off-policy outputs). Watch for drift as inputs, data, and user behavior change, and set alerts on the metrics tied to the business case. Close the loop by feeding failures back into the eval set and the roadmap. Re-score each funded use case against its original business case on a cadence (for example at month 6, 12, and 24) so value is proven, not assumed.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>44. How do you evaluate an agent, as opposed to a single answer?</b></summary>

Judge the whole trajectory, not just the final output: did the agent choose the right tools with valid arguments in a sensible order, did it recover from errors, and did it finish without wandering. Key measures are task completion, tool-call correctness, step efficiency (redundant or looping steps waste money and signal shaky reasoning), and adherence to the retrieved context and policy. You score trajectories either against a gold reference path or with an LLM judge reading the full trace, and you run the same checks in development, in continuous integration, and in production. A wrong tool call or a hallucinated argument is a failure even when the final answer happens to look right.

**Learn more:** [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md) and [AI agent observability](https://www.n-ix.com/ai-agent-observability/)

</details>

<details>
<summary><b>45. What is observability or tracing for AI systems, and why is it table stakes in 2026?</b></summary>

Observability means capturing the full trace of every request: the prompt, retrieved context, each tool call and result, the model's steps, cost, latency, and the final output, so you can see exactly what happened when something breaks. It is table stakes because agentic systems make many hidden decisions, and without traces you cannot debug a failure, attribute cost, or prove compliance. The stronger practice adds scoring on top of the traces (step-level and trace-level metrics), turning a trace viewer into a continuous quality signal. A strategist should require tracing and monitoring in the plan, because a system you cannot observe is a system you cannot govern or improve.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md) and [AI agent observability](https://www.n-ix.com/ai-agent-observability/)

</details>

<details>
<summary><b>46. How do you build an evaluation set from scratch when you have no labeled data?</b></summary>

Start by collecting real or realistic inputs (from logs, subject-matter experts, or a small pilot) and defining, with the people who own the outcome, what a good and a bad answer look like. Label a first batch by hand to anchor the rubric, cover the important edge cases and failure types, not just the easy majority, and keep the set small but representative so it is affordable to run often. You can use a model to help draft candidate cases and expected answers, but a human must confirm the labels that matter. Grow and refresh the set as production surfaces new failures, so the eval keeps reflecting reality.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) and [Evaluation topic](../../../topics/evaluation.md)

</details>

<details>
<summary><b>47. How do offline evals, online monitoring, and A/B tests fit together?</b></summary>

Offline evals run a fixed test set before you ship, so you can compare prompts, models, and versions cheaply and catch regressions in continuous integration. Online monitoring watches live traffic for quality, cost, latency, and safety, catching the drift and edge cases a static set never anticipated. A/B or shadow tests compare a change against the current system on real users or real traffic to measure actual business impact before full rollout. You need all three: offline to gate changes, A/B to prove impact, and monitoring to keep the system honest after launch.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md) and Evidently's [LLM evaluation guide](https://www.evidentlyai.com/llm-guide/llm-evaluation)

</details>

<details>
<summary><b>48. What is regression testing for AI, and how do you stop silent degradation?</b></summary>

Regression testing means re-running your evaluation set on every change (a new prompt, a model version, a retrieval tweak) so you catch quality drops before users do, the same discipline software teams apply to code. AI needs it more, not less, because model providers update models under you and a prompt that worked can quietly break. Wire evals into your release process so a change cannot ship if it falls below threshold, and monitor production metrics for slow drift between releases. Silent degradation is the default failure mode of ungoverned AI systems, and a repeatable eval gate is the fix.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

---

## E. Cost, latency, and production

<details>
<summary><b>49. How do you think about the total cost of an AI system, beyond model API fees?</b></summary>

Model calls are often the smallest line. Total cost includes data preparation and pipelines, retrieval and vector infrastructure, integration and engineering, evaluation and monitoring, human review for high-stakes output, security and compliance work, and ongoing maintenance as models and needs change. The invisible work of enablement, documentation, support, and adoption exists whether you build or buy; the question is who carries it. A credible business case prices all of this, and a common failure is modeling only the token cost.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>50. What drives latency in an LLM application, and why does it matter commercially?</b></summary>

Latency is driven by model size and whether it is a reasoning model, the number of tokens generated, context length, the number of sequential tool or retrieval calls, and any human-in-the-loop step. It matters because slow responses kill adoption in interactive workflows and can break real-time use cases entirely. You manage it by routing to smaller or faster models where quality allows, streaming output, parallelizing independent calls, caching, and trimming context. For strategy, latency is a product constraint that can make an otherwise valuable use case unviable.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>51. How do you decide which model to use for a given use case?</b></summary>

Match the model to the task, not to the leaderboard: use a fast, cheap model for simple classification or drafting, a mid-tier model for most knowledge work, and a reasoning model only for genuinely hard multi-step problems. Weigh accuracy on your own eval set against cost, latency, context window, data-residency and privacy needs, and whether you can self-host. Many production systems route different steps to different models to balance quality and cost. Avoid standardizing on one frontier model for everything, because it overpays for the easy 80% of calls.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>52. What does it take to move an AI pilot into production, and why do so many die in between?</b></summary>

Getting to production requires reliable data pipelines, evals and monitoring, security and compliance sign-off, integration into existing systems and workflows, and the change management to get people to actually use it. Roughly 80 to 95 percent of enterprise AI pilots stall, usually because they were run as technology experiments rather than operating-model changes: no baseline, no adoption plan, governance never built, and sponsorship that vanished after the demo. The fix is to design the pilot with the production path, owner, and success metric defined up front. A strategist's core value is refusing to fund pilots that have no route out of the sandbox.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md) and enterprise AI adoption in [2026 data](https://writer.com/blog/enterprise-ai-adoption-2026/)

</details>

<details>
<summary><b>53. What is model routing or a model cascade, and why is it central to cost control?</b></summary>

Routing sends each request to the cheapest model that can handle it and escalates only the hard cases to a larger model, often described as routing the predictable majority of traffic to a small model and reserving a frontier model for the genuinely complex minority. This can cut inference cost several-fold while keeping quality where it matters, because most enterprise traffic is easy and does not need a top-tier reasoning model. You implement it with a classifier or confidence check that decides when to escalate, and you evaluate the whole routed system, not just each model. Done well, routing is one of the highest-leverage cost decisions in a production AI system.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md) and a [survey on small-and-large model collaboration](https://arxiv.org/abs/2505.07460)

</details>

<details>
<summary><b>54. What is prompt caching, and how does it change agent economics?</b></summary>

Prompt caching stores the model's processing of a stable prefix (system instructions, tool definitions, a fixed knowledge block) so repeated calls reuse it instead of paying to re-process the same tokens every time. For agents that send a large, mostly unchanging context on every loop iteration, this can cut cost and latency substantially, sometimes turning an uneconomical design into a viable one. To benefit, you structure the context so the stable part comes first and the variable part comes last, keeping the cached prefix consistent. It is a concrete reason context design and cost are linked, and a lever worth asking about in any agent business case.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md) and Anthropic on [context management](https://www.anthropic.com/news/context-management)

</details>

<details>
<summary><b>55. How do you budget and forecast AI spend, and cap runaway agent cost?</b></summary>

Forecast from expected volume times cost per interaction, and stress-test it, because agents can make many model and tool calls per task, so a modest per-task cost times high volume can surprise you. Put hard controls in place: per-request and per-user token and step caps, budget alerts, rate limits, and routing so cheap models absorb the easy load. Monitor cost per successful outcome, not just raw spend, so you can see whether value is keeping pace. This is a real failure mode in 2026, where unclear ROI and escalating cost put a large share of agentic projects at risk of cancellation, so cost control belongs in the design, not the post-mortem.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

<details>
<summary><b>56. How should a strategist think about AI vendor pricing models: per-seat, consumption, or outcome-based?</b></summary>

Per-seat pricing is predictable and easy to budget but can overcharge for light users and undercharge for heavy automation. Consumption pricing (per token, per call, per agent action) aligns cost to usage but makes spend hard to forecast and can spike as adoption grows. Outcome-based pricing (paying per resolved ticket or per completed task) ties cost to value but requires trust in the measurement and clear definitions of success. Match the model to your usage pattern, negotiate caps and volume tiers, and watch for pricing that scales faster than the value you capture as agents do more work per user.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md) and a [build-vs-buy cost and ROI guide](https://www.contus.com/blog/build-vs-buy-ai/)

</details>

<details>
<summary><b>57. When does on-device, edge, or sovereign inference matter, and what does it cost you?</b></summary>

Running models on-device, at the edge, or in a controlled sovereign environment matters when data cannot leave a jurisdiction or a facility, when latency must be very low, or when connectivity is unreliable, for example in healthcare, defense, manufacturing, and regulated finance. Smaller and distilled models increasingly make this practical, since they run cheaply on modest hardware while keeping most of the quality for a bounded task. The costs are heavier engineering, model management across many endpoints, and usually a capability ceiling below the largest cloud models. Use it where data residency, latency, or offline operation are genuine requirements, not as a default.

**Learn more:** [Production and LLMOps topic](../../../topics/production.md)

</details>

---

## F. Business, ROI, prioritization, and build-vs-buy

<details>
<summary><b>58. How do you prioritize AI use cases across a company?</b></summary>

Score every candidate use case on a consistent rubric: business impact, technical feasibility, data readiness, time to production, and risk and compliance. That yields a ranked backlog, and you pick a small set (3 to 5) tied to real P&L impact rather than spreading thin, because leading enterprises fund fewer initiatives and generate more return. Sequence for early proof: a quick win to build credibility and data, then a platform investment, then a transformational bet. Make the scoring explicit so the choices are defensible to a skeptical executive.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>59. Walk me through building an ROI model for an AI initiative.</b></summary>

Start by establishing the baseline: the current cost, time, error rate, or revenue of the process today, because without a baseline you cannot claim a lift. Estimate the expected improvement (for example 30 to 50 percent time reduction, higher conversion, fewer errors), convert it to dollars, then subtract the full cost to build and run (data, engineering, infrastructure, review, change management). Compute payback period and be explicit about assumptions and their range, since false precision destroys credibility. Then commit to re-measuring against the same baseline on a cadence rather than declaring victory at launch.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>60. How do you approach build vs buy vs partner for an AI capability?</b></summary>

Buy for commodity capability where speed matters and there is no differentiation, because vendors have already solved it and you avoid carrying the maintenance. Build when the capability is a genuine differentiator, you have proprietary data and engineering depth, and control or data residency demands it. Partner to share risk on the hard middle where you need custom work but lack the full team. The decision turns on latency and compliance needs, cost model, and in-house depth, and buy-plus-partner approaches tend to reach production more reliably than fully internal builds. Beware advisors who recommend the same stack to everyone, which signals a product sale rather than a strategy.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md) and a [2026 build-vs-buy framework](https://www.contus.com/blog/build-vs-buy-ai/)

</details>

<details>
<summary><b>61. A CEO says a competitor launched an AI agent and wants one in 90 days. How do you respond?</b></summary>

Acknowledge the pressure and redirect it to the business goal: what outcome does the competitor's move threaten, and what would actually protect or grow the business. Warn against building the flashy thing that fails in pilot, and propose a 90-day plan that delivers a real, narrow win with a measured baseline instead of a demo. Frame it as moving faster in a way that will still be standing in six months, and show the roadmap from quick win to the more ambitious capability. This turns a reactive vanity request into a sequenced, defensible plan.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>62. How do you build an AI roadmap for an organization just starting out?</b></summary>

First assess readiness: data availability and quality, existing tech and talent, governance maturity, and executive alignment. Then map and score use cases, pick a small first wave weighted toward quick, low-risk wins that build capability and credibility, and invest in the shared foundations (data, platform, governance, skills) that later use cases will reuse. Sequence in waves: prove value, build the platform, then pursue transformational bets. Attach owners, metrics, and a change-management budget to each phase, because a roadmap with no adoption plan is a wish list.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md) and the [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>63. How do you measure whether an AI strategy is actually working at the portfolio level?</b></summary>

Track leading indicators (use cases in production, adoption and active usage, time from idea to production) and lagging indicators (aggregate cost saved or revenue gained against baselines, and return versus spend). Watch the conversion rate from pilot to production, because a low rate signals a prioritization or adoption problem, not a technology one. Maintain a quarterly re-measurement cadence where each funded use case is re-scored against its own business case. The honest headline metric is realized value per dollar invested, not the number of pilots launched.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>64. A client wants to fund 20 AI projects at once. What do you tell them?</b></summary>

That funding 20 at once is a reliable way to get 20 pilots and zero at scale, because attention, data, engineering, and change-management capacity are finite. The evidence is consistent: organizations that fund fewer, well-chosen initiatives get materially higher return. Recommend concentrating on 3 to 5 use cases with the clearest P&L link and readiness, resourcing them properly through to production, and using the wins to build the platform and appetite for the next wave. Discipline in saying no is a large part of the strategist's value.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>65. How is generative AI changing the consulting and services business itself?</b></summary>

It compresses the work that used to justify large leverage pyramids: research, first-draft analysis, and deck production are now partly automated, so firms are investing heavily in internal AI tools and forward-deployed delivery models. This raises the bar for what a human advisor adds: judgment, client trust, change leadership, and the ability to challenge and integrate AI output rather than produce the first draft. For a candidate, it means firms want strategists who use AI fluently in the work and can be graded on how well they prompt and pressure-test it. Positioning yourself as someone who orchestrates AI plus judgment, rather than someone AI could replace, is the winning stance.

**Learn more:** [Forward-Deployed Engineer role prep](../forward-deployed-engineer/README.md)

</details>

<details>
<summary><b>66. Build vs buy is rarely binary in 2026; how do you use hybrid approaches and avoid vendor lock-in?</b></summary>

Most enterprises now combine a bought platform for speed and reliability with custom work where it creates real advantage, sometimes called buy-then-extend or boost, so you get to production fast and still differentiate. The lock-in risk is real: proprietary data formats, non-portable prompts and fine-tunes, and pricing that scales with your success. Mitigate it by keeping your data and evals under your control, preferring standards like MCP for integration, abstracting the model layer so you can swap providers, and negotiating exit and portability terms up front. The goal is to capture vendor speed without handing them your leverage.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md) and a [2026 build-vs-buy framework](https://www.contus.com/blog/build-vs-buy-ai/)

</details>

<details>
<summary><b>67. Analysts warn a large share of agentic AI projects will be canceled by 2027. How do you keep yours off that list?</b></summary>

Those cancellations trace to unclear success criteria, escalating cost, and thin ROI rather than model limits, so you protect a project by fixing the business case before the build. Define the metric and baseline, scope the first version narrow enough to prove value quickly, cap cost with routing and budgets, and put evals and monitoring in place so you can show whether it is working. Kill or reshape it fast if the numbers do not move, rather than letting it drift as an expensive experiment. Discipline on criteria, cost, and evidence is exactly what separates the projects that survive from the ones that get cut.

**Learn more:** [Enterprise AI adoption in 2026](https://writer.com/blog/enterprise-ai-adoption-2026/) and the [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>68. What is agent-washing, and how do you cut through vendor hype in AI procurement?</b></summary>

Agent-washing is relabeling ordinary automation or a thin wrapper around a model as an autonomous agent to ride the hype, and it is common enough that a real capability check is essential. Cut through it by asking for a named production customer with a measured outcome, a live demo on your own data, the failure modes and how errors are caught, the cost and latency at your volume, and how they evaluate and monitor quality. Probe what the system actually decides and does versus what a human still has to do. If the answers are vague or every question routes back to a slide, treat it as marketing and keep looking.

**Learn more:** [Evaluation topic](../../../topics/evaluation.md) and a [build-vs-buy decision guide](https://www.contus.com/blog/build-vs-buy-ai/)

</details>

<details>
<summary><b>69. How do you decide between investing in a shared AI platform and buying point solutions?</b></summary>

Point solutions win early: they deliver a specific outcome fast, prove value, and need little internal capability, which is why the first wave of a roadmap often buys them. A shared platform (common data access, retrieval, model gateway, evals, governance, and reusable tools) pays off once you have several use cases that would otherwise each rebuild the same plumbing. The signal to invest in the platform is repetition: when the third project is re-solving integration, security, and monitoring, centralizing it lowers the cost of every future use case. Sequence it as prove value with point solutions, then build the platform the wins justify, rather than starting with a heavy platform and no proven demand.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>70. How and when do you decide to kill an AI initiative?</b></summary>

Set kill criteria before you start: the metric it must move, the baseline, the threshold, and the date by which it must show progress, so the decision is evidence-based rather than political. Kill or fundamentally reshape it when the value is not materializing, the cost is climbing without payback, the data or accuracy will not support the ambition, or the risk has grown beyond the benefit. Killing early frees budget, talent, and credibility for the use cases that will work, and a strategist who prunes well is more valuable than one who defends every bet. Frame the stop with the reason and, where possible, the condition under which it could restart later.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>71. When the benefits are soft (better quality, faster cycle time) rather than headcount savings, how do you justify the investment?</b></summary>

Soft benefits are real but need to be made measurable: tie faster cycle time to revenue pulled forward or more throughput per person, tie higher quality to fewer errors, lower rework, better retention, or higher conversion, and quantify each with a defensible estimate and a range. Where a dollar figure is genuinely uncertain, state the assumption openly and show the sensitivity rather than inventing precision. Pair the value case with the strategic reason (capability, speed, or defensive necessity) so leadership can weigh it even where the number is soft. Then commit to measuring the leading indicators after launch so the soft benefit becomes evidence, not a claim.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

---

## G. Risk, responsible AI, and governance

<details>
<summary><b>72. What are the main risks in an enterprise AI deployment, and how do you govern them?</b></summary>

The main categories are data risk (leakage, using data you have no right to, privacy), model risk (hallucination, bias, unreliable output), security risk (prompt injection, data exfiltration through tools, insecure integrations), and regulatory and reputational risk. You govern them with a tiered approach: classify each use case by risk, then apply proportionate controls such as human-in-the-loop, evals, monitoring, access scoping, red-teaming, and audit logging. Most organizations anchor this to a recognized framework like the NIST AI Risk Management Framework (govern, map, measure, manage), with the EU AI Act setting legal minimums for higher-risk systems and ISO/IEC 42001 providing a certifiable management system. Governance should be proportionate, so low-risk internal tools are not strangled while high-risk decisions get real oversight.

**Learn more:** [Safety and security topic](../../../topics/safety-security.md) and the [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

</details>

<details>
<summary><b>73. What is prompt injection, and why should a strategist care?</b></summary>

Prompt injection is when untrusted content (a web page, a document, an email the agent reads) contains hidden instructions that hijack the model into doing something it should not, like leaking data or calling a tool maliciously. It matters because any agent that reads external content and can take actions has a real attack surface, and the risk grows with autonomy and tool access. A strategist cares because it turns a capability discussion into a security and liability discussion: it shapes which tools an agent may touch, whether a human approves consequential actions, and how the system is red-teamed. Ignoring it is how an impressive agent becomes a breach.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>74. How do you handle the EU AI Act and the current regulatory landscape in a strategy?</b></summary>

Treat regulation as a design input, not an afterthought: the EU AI Act uses a risk-based regime where higher-risk uses (for example those affecting people's rights, employment, or credit) carry real obligations, with general-purpose model rules and transparency duties phasing in and broad enforcement building through 2026. Classify each use case by regulatory risk early, because that changes what controls, documentation, and human oversight are required and sometimes whether to proceed at all. Combine the legal minimum from applicable regulation with a management framework (NIST AI RMF, ISO/IEC 42001) so compliance is provable. The strategist's job is to bake this into prioritization so you do not build something you cannot legally deploy.

**Learn more:** [EU AI Act overview](https://artificialintelligenceact.eu/) and [ISO/IEC 42001](https://www.iso.org/standard/42001)

</details>

<details>
<summary><b>75. When should you recommend not using AI, or slowing down?</b></summary>

When the use case demands accuracy the technology cannot yet deliver and errors are consequential and hard to catch, when you lack the data or the right to use it, when the regulatory or reputational risk outweighs the benefit, or when a simpler non-AI solution solves the problem better. Recommending against a build is a strength: it protects budget and trust and signals judgment rather than hype. Frame the no with a reason and, where possible, an alternative or a condition under which it becomes viable later. Interviewers specifically probe for this because a strategist who never says no is a liability.

**Learn more:** [Safety and security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>76. What is shadow AI, and how do you govern employees using unsanctioned tools?</b></summary>

Shadow AI is staff using consumer AI tools or unapproved agents outside any policy, which risks leaking confidential data into third-party systems, producing ungoverned outputs, and creating compliance exposure no one is tracking. Banning it outright usually fails, because people reach for these tools when the sanctioned path is too slow or missing. The better approach is to provide safe, approved alternatives that are genuinely useful, publish clear guidance on what data may go where, and add monitoring and access controls to see and steer usage. Treat widespread shadow AI as a signal of unmet demand and a governance gap to close, not only a violation to punish.

**Learn more:** [Safety and security topic](../../../topics/safety-security.md) and [IBM on AI governance](https://www.ibm.com/think/topics/ai-governance)

</details>

<details>
<summary><b>77. How do you handle data governance and intellectual property risk in AI deployments?</b></summary>

Establish, per use case, that you have the right to use the data for AI (customer contracts, licensing, consent), that personal data is minimized and protected, and that the vendor's terms do not train on your inputs unless you agree. On the output side, clarify ownership and the risk of generated content resembling copyrighted material, and keep provenance and audit trails so you can answer where an answer came from. Enforce access controls so retrieval and agents only expose what a given user may see, and log usage for accountability. These questions decide whether a use case is deployable, so they belong in prioritization, not in a late legal review.

**Learn more:** [Safety and security topic](../../../topics/safety-security.md) and the [EU AI Act overview](https://artificialintelligenceact.eu/)

</details>

<details>
<summary><b>78. What agent security risks go beyond prompt injection, and how do you contain them?</b></summary>

Excessive agency (an agent with more permissions or tools than the task needs) turns a small mistake into a large one, tool poisoning hides malicious instructions in a tool or its description, and memory poisoning plants false information the agent later trusts and acts on. Add data exfiltration through connected tools, cascading failures in multi-agent setups, and impersonation when identity is weak. Contain them with least-privilege access, per-user identity, vetted and gated tools and MCP servers, validation of tool outputs, approval gates on consequential actions, and full logging so you can detect and trace abuse. The through-line is that autonomy plus tool access is the attack surface, so security scales with how much the agent can do.

**Learn more:** [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>79. How do you stand up an AI governance function: who owns it, and what is the operating model?</b></summary>

Effective governance is cross-functional: a named owner or council spanning legal, security, data, risk, and the business lines, with clear decision rights over which use cases proceed and under what controls. Anchor it to a recognized framework (NIST AI RMF for the process of govern, map, measure, manage, ISO/IEC 42001 for a certifiable management system) so requirements are consistent and auditable, and tier controls by risk so low-stakes tools move fast while high-stakes ones get review. Give it teeth with an inventory of AI systems, standard intake and risk classification, required evals and monitoring, and incident response. The aim is proportionate governance that speeds safe use rather than a committee that blocks everything.

**Learn more:** [NIST AI RMF playbook](https://www.nist.gov/itl/ai-risk-management-framework/nist-ai-rmf-playbook) and [IBM on AI governance](https://www.ibm.com/think/topics/ai-governance)

</details>

<details>
<summary><b>80. How do you evaluate bias and fairness in an enterprise AI deployment?</b></summary>

Start by asking whether the use case can affect people unfairly (hiring, lending, pricing, access to services), because that determines how much fairness work is required and what regulation applies. Test with representative data across the groups that matter, measure outcome differences, and probe for proxies that encode bias indirectly, using both quantitative metrics and human review of real cases. Bias can enter through training data, retrieval sources, prompts, and thresholds, so treat it as a system property to monitor over time, not a one-time check. Keep humans in the loop on consequential decisions and document your testing, since regulators and courts increasingly expect evidence, not assurances.

**Learn more:** [Safety and security topic](../../../topics/safety-security.md)

</details>

<details>
<summary><b>81. How do you assess third-party and vendor AI risk during procurement?</b></summary>

Diligence the vendor as you would any critical supplier plus the AI-specific questions: what data they train on and retain, where data is processed and stored, their security posture and certifications, how they evaluate and monitor quality, their incident history, and their model-update and deprecation policy. Ask for evidence (a named reference, an eval on your data, documented controls) rather than assurances, and map their claims to your regulatory obligations, since you often remain accountable for outcomes even when a vendor builds the system. Negotiate data-handling terms, portability, and exit rights so a vendor problem does not become your outage or breach. Fold this into a standard intake so every bought AI capability clears the same bar.

**Learn more:** [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) and a [build-vs-buy decision guide](https://www.contus.com/blog/build-vs-buy-ai/)

</details>

---

## H. Change management and org

<details>
<summary><b>82. Why do most AI pilots fail to scale, and what does a strategist do about it?</b></summary>

They fail because they are run as technology experiments rather than changes to how the organization works: missing data infrastructure, no change management, governance never built, metrics untethered from business outcomes, and executive sponsorship that evaporates after the demo. Only a minority of organizations report investing meaningfully in the change management, training, and incentives that adoption requires. A strategist fixes this by designing each initiative with a named owner, a business metric, a production path, and a change budget from the start, and by protecting executive sponsorship through the messy middle. The lesson is blunt: adoption, not the model, is where value is won or lost.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>83. How do you drive adoption of an AI tool in a resistant organization?</b></summary>

Start with the workflow and the people, not the technology: understand what the affected employees fear (job loss, more work, being blamed for the model's errors) and design for it. Secure a visible executive sponsor, pick an early group that will benefit and champion it, and provide training, clear guidance on when to trust and when to override the tool, and incentives aligned with using it. Budget real money for enablement (a common rule of thumb is 20 to 30 percent of the AI investment) and measure adoption and override rates, not just accuracy. Communicate honestly about what the tool does and does not do, because overselling it once destroys trust for every future rollout.

**Learn more:** [Enterprise AI adoption in 2026](https://writer.com/blog/enterprise-ai-adoption-2026/)

</details>

<details>
<summary><b>84. How do you plan for workforce impact and reskilling as AI changes the work?</b></summary>

Be honest and specific: map which tasks AI will absorb, which roles shift rather than vanish, and what new skills the work now demands, then invest in moving people toward the higher-judgment parts of their jobs. A useful framing is the shift from producing first drafts to reviewing and verifying AI output, which raises the value of domain judgment, critical review, and orchestration. Pair reskilling with clear communication so people understand the plan, because fear of job loss is the fastest way to kill adoption. Treat the human transition as part of the initiative's cost and timeline, not a side effect to manage after launch.

**Learn more:** [Enterprise AI adoption in 2026](https://writer.com/blog/enterprise-ai-adoption-2026/)

</details>

<details>
<summary><b>85. How do you structure an AI operating model: center of excellence, embedded, or hub-and-spoke?</b></summary>

A central center of excellence concentrates scarce talent, sets standards, and builds shared platforms, but can become a bottleneck far from the business. Fully embedded teams sit close to the work and move fast, but duplicate effort and drift on standards. Most enterprises land on hub-and-spoke: a central hub owns platform, governance, and enablement, while embedded pods in each business line own use cases and adoption. Choose based on AI maturity and organizational size, and expect to shift from more central early (to build capability and guardrails) toward more federated as fluency spreads.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>86. How do you measure adoption honestly, and should you mandate use?</b></summary>

Measure active, repeated use tied to the workflow the tool is meant to change, not logins or one-time trials, alongside override and escalation rates that reveal whether people actually trust the output. Low adoption is usually a signal that the tool does not fit the workflow, is not trusted, or was never enabled properly, so treat it as a diagnostic rather than a discipline problem. A blunt mandate can force surface usage while breeding workarounds and resentment, so pair any expectation to use with genuine value, training, and manager buy-in. The healthiest signal is voluntary, sustained use because the tool makes the work better.

**Learn more:** [Enterprise AI adoption in 2026](https://writer.com/blog/enterprise-ai-adoption-2026/)

</details>

<details>
<summary><b>87. How do you build AI literacy across executives and the broader workforce?</b></summary>

Executives need enough fluency to fund and govern well: what models can and cannot do, why evals and data matter, how cost and risk behave, and how to read a business case, taught through their own decisions rather than abstract lectures. The workforce needs practical, role-specific enablement: how to use the sanctioned tools on real tasks, when to trust and when to override, and what data may go where. Make it hands-on and continuous, because the field moves fast and one-time training goes stale. Rising literacy is also what lets governance be proportionate, since informed teams need fewer hard gates.

**Learn more:** [Understand AI journey](../../../journeys/understand.md) and the [courses index](../../../courses.md)

</details>

<details>
<summary><b>88. What is the forward-deployed model of delivery, and when does it work for internal AI?</b></summary>

Forward-deployed delivery embeds a technical person alongside the users, learning the actual workflow, shipping a thin working system, and iterating in place rather than handing over a specification. It works well for AI because value lives in the messy specifics of a real process, and a demo rarely survives contact with production data, permissions, and users without close, on-the-ground iteration. Applied internally, it means placing capable builders inside business units to find the highest-value workflow and drive it to production, feeding reusable patterns back to the platform team. Use it for high-value, poorly-understood workflows where being close to the work beats a formal requirements process.

**Learn more:** [Forward-Deployed Engineer role prep](../forward-deployed-engineer/README.md) and a [2026 FDE interview guide](https://www.tryexponent.com/blog/forward-deployed-engineer-interview-the-definitive-2026-guide-fde)

</details>

---

Next: **[resources.md](resources.md)** and **[courses.md](courses.md)**.
