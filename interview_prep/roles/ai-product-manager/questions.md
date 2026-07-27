# AI Product Manager: Question Bank

94 questions with concise model answers, grouped by theme. Every question is a click-to-open collapsible: the answer is 3 to 6 sentences, enough to anchor your own, not a script to recite, and each ends with a "Learn more" pointer for going deeper. Add a specific example from your own work to every one before an interview. Pair with the [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and the [Role-Based Prep](../../README.md).

Themes:
1. [Capability judgment and product sense](#1-capability-judgment-and-product-sense) (Q1-13)
2. [Model and architecture literacy](#2-model-and-architecture-literacy) (Q14-34)
3. [Evaluation and metrics](#3-evaluation-and-metrics) (Q35-52)
4. [Cost, latency, and unit economics](#4-cost-latency-and-unit-economics) (Q53-62)
5. [Responsible AI, safety, and agents](#5-responsible-ai-safety-and-agents) (Q63-76)
6. [Business and strategy](#6-business-and-strategy) (Q77-87)
7. [Execution and behavioral](#7-execution-and-behavioral) (Q88-94)

---

## 1. Capability judgment and product sense

<details>
<summary><b>Q1. How do you decide whether a problem needs AI at all, versus a rules engine or plain software?</b></summary>

Start from the user outcome and the cost of being wrong, not from the technology. Use deterministic software or a rules engine when the logic is stable, auditable, and the cost of an error is high; reach for a model only when the input is messy, open-ended, or high-variety in a way rules cannot cover. A useful test: if you can write the rules down and they do not explode in number, write the rules. Boring, deterministic solutions often win, are cheaper, and never hallucinate, so the burden of proof is on adding the model.

**Learn more:** [Understand AI journey](../../../journeys/understand.md)

</details>

<details>
<summary><b>Q2. When would you reach for a rules-based system, a classic ML model, and an LLM?</b></summary>

Rules fit stable, explainable logic with clear inputs (tax brackets, eligibility checks). Classic ML (gradient boosting, logistic regression) fits structured, tabular prediction at scale with labeled history: fraud scoring, churn, ranking. LLMs fit unstructured language and generation: summarization, extraction from free text, conversation, code. The choice also weighs latency, cost, interpretability, and how much labeled data you have; you often stack them, for example rules to gate and an LLM to handle the long tail.

**Learn more:** [Foundations](../../../topics/foundations.md)

</details>

<details>
<summary><b>Q3. Design an AI feature for our product. Where would the model add value, and where would it just add risk?</b></summary>

First name the user and the job to be done, then split the workflow into steps and ask which steps are language-heavy, high-variety, or currently manual and slow. Put the model where it removes real toil (drafting, summarizing, triaging) and keep deterministic control where correctness is legally or financially load-bearing (final calculations, permissions, sending money). For each model-touched step, decide the fallback when it is wrong and how you will measure quality. The answer that scores well says out loud where you would refuse to put a model.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>Q4. A stakeholder wants to add AI to a feature that works fine without it. How do you handle that conversation?</b></summary>

Get to the underlying goal: is this a real user problem, a differentiation worry, or leadership pressure to look AI-forward. Acknowledge the goal, then compare the current solution against an AI version on user value, cost, latency, and new failure modes, ideally with a cheap experiment rather than an opinion. Often the right move is a narrow pilot on one high-variety slice where AI clearly helps, while leaving the working path alone. Protecting a good deterministic experience from an unnecessary model is a product win, not obstruction.

**Learn more:** [rounds.md, AI product sense](rounds.md)

</details>

<details>
<summary><b>Q5. Pick an AI product you admire. What is the one design decision that made it work?</b></summary>

Choose a product you have actually used and name a single decision that shaped its success, tied to a principle you can reuse. For example, a coding assistant that shows a diff you accept or reject keeps the human in control and makes wrongness cheap to reject, which is why people trust it despite imperfect output. The point is to show you can reverse-engineer why a design works: control, low cost of error, fast feedback, or graceful degradation. Avoid vague praise; name the mechanism.

**Learn more:** [Harness Engineering guide](../../../resources/harness_engineering.md)

</details>

<details>
<summary><b>Q6. How is product sense for AI different from classic product sense?</b></summary>

Classic product sense starts at "what should we build." AI product sense starts one step earlier, at "should a model be anywhere near this," because a model brings non-determinism, cost, latency, and new failure modes that a deterministic feature does not. You also design for wrongness from the first sketch (fallbacks, confidence, undo, citations) and you carry an evaluation and cost story through the whole design rather than treating quality as an engineering detail. Everything downstream (users, use cases, metrics) still applies.

**Learn more:** [README: what the job actually is](README.md)

</details>

<details>
<summary><b>Q7. How would you design an AI feature so users trust it?</b></summary>

Trust comes from control, transparency, and graceful failure, not from the model being perfect. Keep the user in the loop for consequential actions (preview, confirm, undo), show sources or reasoning where it helps, and set honest expectations about what the feature can and cannot do. When the model is unsure, say so and route to a human or a safe default rather than bluffing. Then measure trust: acceptance rate, edit rate, escalation rate, and repeat use, and treat a confident wrong answer as a serious incident.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q8. How do you decide how much autonomy to give a feature: suggest, draft, or act on the user's behalf?</b></summary>

Match autonomy to reversibility and the cost of a mistake, not to how impressive the demo looks. Low-stakes, reversible work (drafting a reply, summarizing, suggesting an edit) can run with light oversight because the user reviews before anything ships. Irreversible or high-consequence actions (sending money, emailing a customer, deleting data) need explicit confirmation, tight scopes, and an easy undo, and you start further down the autonomy ladder and earn your way up as evals and trust improve. A clean answer names the autonomy spectrum from suggest to draft to act and ties each rung to a wrongness cost and a control mechanism.

**Learn more:** [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>Q9. Design a voice or multimodal AI feature. What changes compared to a text feature?</b></summary>

Multimodal inputs (voice, images, screen, documents) widen what users can express but add new failure modes: transcription errors, ambiguous references, and latency from processing audio or images. Voice in particular removes the safety net of a visible draft, so confirmation, readback of consequential actions, and easy correction matter more, and turn-taking latency (time to first response) drives whether it feels natural. You also design for context the model cannot see or mishears, and you decide what to keep text-based because it is safer to review. The evaluation set has to include noisy audio, accents, poor images, and mixed-modality inputs, not just clean text.

**Learn more:** [Multimodal](../../../topics/multimodal.md)

</details>

<details>
<summary><b>Q10. How do you set the "good enough" quality bar for launching an AI feature?</b></summary>

Derive the bar from the cost of being wrong and the quality of the alternative the user has today, not from a round number. Define the failure that matters (a wrong number in a finance tool, an unsafe instruction), pick a metric that captures it, and set a threshold on the offline eval set plus a guardrail you will not cross in production. Compare against the real baseline: if the feature beats the current manual process or the status quo on the outcome that matters, "good enough" can be well below perfect. State the bar as a launch gate before you build, so quality is a decision rather than a debate at the finish line.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>Q11. You are designing an AI feature with no usage data and no labeled eval set yet. How do you start?</b></summary>

Bootstrap the eval set by hand: write 20 to 50 realistic and adversarial inputs yourself, based on the user problem and the questions you expect, and label the expected behavior or a rubric. Seed it with edge cases you already fear (ambiguous asks, out-of-scope requests, hostile inputs) so the cold-start set stresses the design rather than flattering it. Ship to a small internal or beta group behind a flag, capture real transcripts, and use error analysis on those to grow and re-weight the set toward the failures that actually occur. The set is a living product artifact that starts small and compounds, not a one-time deliverable.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>Q12. In a live prototype round, you are handed an AI builder tool and asked to build a demo. How do you approach it?</b></summary>

Scope hard first: pick the 10 percent of the feature that proves the core value and fake the rest, and say out loud what you are faking and why. Prompt the tool clearly, then sanity-check its output instead of trusting it, because the round grades your judgment and your ability to catch the tool being wrong, not your syntax. Narrate the product tradeoffs as you go (what you would measure, where it would break, what the fallback is) so the signal is product thinking, not coding. Freezing because you are not an engineer is the classic miss; the bar is taste and direction.

**Learn more:** [rounds.md, live prototype round](rounds.md)

</details>

<details>
<summary><b>Q13. How do you tell genuine AI product value from a good demo that will not hold up in production?</b></summary>

Demos run on cherry-picked happy paths; production runs on the messy long tail, so probe the gap. Ask what the failure rate is on real, adversarial, and out-of-distribution inputs, what the cost and latency are at scale, and whether there is an eval harness catching regressions, because a feature that shines in a scripted demo often collapses under real variety. Look for whether wrongness is designed for (fallbacks, confidence, undo) rather than assumed away. The mature read is to trust measured behavior on representative traffic over a polished single run.

**Learn more:** [Production](../../../topics/production.md)

</details>

## 2. Model and architecture literacy

<details>
<summary><b>Q14. When would you use RAG instead of fine-tuning to give a model new knowledge?</b></summary>

Use RAG when the knowledge changes often, is large or proprietary, or needs citations: retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, or style, or to bake in a stable skill, not for volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior. RAG is also usually faster and cheaper to iterate on, since you update an index instead of running a training job.

**Learn more:** [RAG](../../../topics/rag.md) and [Fine-tuning](../../../topics/fine-tuning.md)

</details>

<details>
<summary><b>Q15. Explain RAG to a non-technical executive in under a minute.</b></summary>

The model is smart but does not know our private or up-to-date information, and it will confidently make things up to fill the gap. RAG fixes that by first searching our own trusted documents for the relevant passages, then handing those passages to the model and asking it to answer using only them, with citations. So instead of guessing from memory, it answers open-book from our sources, which cuts hallucinations and lets us update knowledge by updating documents, not by retraining. The tradeoff is that answer quality now depends on whether the search step actually finds the right passage.

**Learn more:** [Agentic RAG 101](../../../resources/agentic_rag_101.md)

</details>

<details>
<summary><b>Q16. A RAG system is giving wrong answers. How do you diagnose retrieval versus generation?</b></summary>

Split the pipeline and inspect each half. Pull the retrieved chunks for the failing questions: if the answer-bearing passage is not in the retrieved set, it is a retrieval problem (fix chunking, embeddings, the query, or re-ranking). If the right passage was retrieved but the model still answered wrong or ignored it, it is a generation or grounding problem (fix the prompt, force citation, or use a stronger model). Measuring retrieval (recall@k) and generation (faithfulness) separately turns a vague "it is bad" into a fixable diagnosis.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md)

</details>

<details>
<summary><b>Q17. What makes something an agent rather than a single LLM call, and when do you actually need one?</b></summary>

An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow (a chain of prompts you designed) will do, because agents add cost, latency, and new failure modes and are harder to evaluate and control. The mature default is the simplest thing that works: prompt, then workflow, then agent only if the task genuinely demands autonomy.

**Learn more:** [Agents](../../../topics/agents.md) and [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>Q18. What is context engineering, and why can it matter more than prompt wording?</b></summary>

Context engineering is deciding what actually enters the model's context window: instructions, retrieved documents, tool results, memory, and what stays out. It matters because models degrade when the context is bloated, noisy, or poorly ordered, so getting the right minimal information in the right place usually beats clever phrasing. For a PM it reframes many quality problems as context problems (wrong or missing retrieval, stale memory, tool output the model cannot parse) rather than prompt-tweaking. Managing the context budget is now a core product lever for quality, cost, and latency.

**Learn more:** [Effective context engineering (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

</details>

<details>
<summary><b>Q19. What are reasoning models, and how do they change what you can build?</b></summary>

Reasoning models are trained to spend extra test-time compute on an internal chain of thought before answering, often via reinforcement learning on verifiable rewards. They are much stronger on math, code, and multi-step planning, at the cost of higher latency and price per answer. As a PM you route hard, multi-step, high-stakes tasks to a reasoning model and keep fast, simple tasks on a cheaper model, because paying reasoning cost and latency on every trivial call is waste. You also prompt them differently: give the goal and constraints and let the model do the steps rather than over-scripting them.

**Learn more:** [Planning and reasoning models](../../../free_courses/agentic_ai_crash_course/part6_planning_in_agents_reasoning_models.md)

</details>

<details>
<summary><b>Q20. What is the Model Context Protocol (MCP) and why does a PM care?</b></summary>

MCP is an open standard for connecting an agent or app to tools and data sources through a common interface. You care because it turns integrations into reusable building blocks: wire a capability once (a database, a ticketing system, a search tool) and reuse it across agents and harnesses instead of a bespoke integration per tool. That lowers the cost and time to give your product new capabilities and makes the ecosystem of connectors something you can buy into rather than rebuild. It is an execution and platform-strategy lever, not just an engineering detail.

**Learn more:** [What is MCP and why care](../../../free_courses/agentic_ai_crash_course/part5_what_is_mcp_and_why_care.md) and [MCP announcement (Anthropic)](https://www.anthropic.com/news/model-context-protocol)

</details>

<details>
<summary><b>Q21. How do you decide between a third-party foundation-model API, a fine-tune, and a custom-trained model?</b></summary>

Default to a third-party API: fastest to ship, no training cost, and you inherit frontier quality and safety work. Move to fine-tuning a base or open model when you need behavior, tone, or a narrow skill the API cannot reliably deliver through prompting, or when volume makes the per-call price painful. Train a custom model almost never, only with a real data moat, scale, and a capability gap no vendor fills, because the cost and talent bar are extreme. Weigh token cost against control, latency against time to market, and the switching risk of betting the product on one vendor's roadmap.

**Learn more:** [Fine-tuning 101](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>Q22. Explain embeddings to a non-technical executive in under a minute.</b></summary>

An embedding turns a piece of text (or an image) into a list of numbers that captures its meaning, so that things with similar meaning land near each other in that number space. That lets a computer measure similarity: "cancel my plan" and "how do I end my subscription" sit close together even though they share few words. We use this to power semantic search, retrieval for RAG, clustering, and recommendations. The practical upshot is that we can find relevant content by meaning rather than exact keywords.

**Learn more:** [Foundations](../../../topics/foundations.md)

</details>

<details>
<summary><b>Q23. What is prompt engineering versus fine-tuning versus RAG, and how do they combine?</b></summary>

Prompting shapes behavior at call time through instructions, examples, and context, with zero training cost and instant iteration. Fine-tuning changes the model's weights to bake in a stable behavior, format, or skill. RAG injects fresh external knowledge at call time by retrieving relevant documents. They compose: RAG supplies the facts, fine-tuning locks in the voice or output format, and prompting orchestrates the call, so a real product often uses all three, and you always try prompting first because it is the cheapest to change.

**Learn more:** [Prompting](../../../topics/prompting.md)

</details>

<details>
<summary><b>Q24. What is agent memory, and what types matter for a product?</b></summary>

Memory is how an agent carries information across steps and sessions so it is not starting cold every turn. It usually splits into three tiers: short-term working context (the current conversation and recent tool results in the context window), long-term semantic memory (past interactions or documents stored as embeddings and retrieved when relevant), and structured memory (facts about the user, account, or policies kept in a database). As a PM you decide what is worth remembering, what to forget for privacy and cost, and how stale memory can silently poison an answer. Bad memory is a real failure mode: the agent confidently acts on an outdated fact it stored earlier.

**Learn more:** [Memory in agents](../../../free_courses/agentic_ai_crash_course/part7_memory_in_agents.md)

</details>

<details>
<summary><b>Q25. Single agent versus multi-agent: when is the added complexity of multiple agents worth it?</b></summary>

Default to a single agent with good tools, because multi-agent systems multiply cost, latency, coordination failures, and evaluation difficulty. Reach for multiple agents when the work genuinely decomposes into specialized roles that can run in parallel or need isolation (a researcher, a writer, a checker), or when separate context windows keep each subtask focused and cheaper. The common pattern is a supervisor that plans and delegates to sub-agents, with clear handoffs. A strong answer resists multi-agent as a default and treats it as a structure you earn once a single agent provably cannot keep the task coherent.

**Learn more:** [Multi-agent systems](../../../free_courses/agentic_ai_crash_course/part8_multi_agent_systems.md)

</details>

<details>
<summary><b>Q26. What is model routing or a model cascade, and why does it matter to a PM?</b></summary>

Routing sends each request to the cheapest model that can handle it: easy calls (classification, short extraction, simple chat) go to a small fast model, and only hard calls escalate to a flagship or reasoning model. A cascade tries the cheap model first and escalates when confidence is low or a check fails. Done well this cuts cost several fold with little quality loss, which is why it is a core production lever rather than a nicety. The PM's job is to define the difficulty signal, set the escalation policy, and measure quality per tier so routing does not quietly degrade the hard cases.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q27. What are function calling, tool use, and structured outputs, and why should a PM care?</b></summary>

Function calling (tool use) is the model returning a structured request to run a specific tool or API with specific arguments, which the system executes and feeds back; it is how an LLM reaches beyond text to fetch data or take actions. Structured outputs constrain the model to return valid JSON matching a schema, so downstream code can rely on the shape instead of parsing free text. For a PM these are what turn a chatbot into a product that does things reliably: they make integrations deterministic at the boundary even though the model is probabilistic. The failure modes to watch are wrong tool choice, malformed arguments, and hallucinated parameters, all of which you evaluate separately.

**Learn more:** [What are tools in AI](../../../free_courses/agentic_ai_crash_course/part3_what_are_tools_in_ai.md)

</details>

<details>
<summary><b>Q28. Long context windows keep growing. Has long context killed RAG?</b></summary>

No: bigger context windows change the tradeoff but do not remove the case for retrieval. Stuffing a million tokens into every call is far more expensive and slower than retrieving a few hundred relevant chunks, and models still lose the needle in a very long haystack, so accuracy can drop even when it all fits. RAG also keeps knowledge fresh and auditable by updating an index rather than a prompt, and it scales to corpora far larger than any window. The pragmatic answer is that long context and RAG are complementary: use the window for the working set, use retrieval to decide what earns a place in it.

**Learn more:** [Agentic RAG 101](../../../resources/agentic_rag_101.md)

</details>

<details>
<summary><b>Q29. What retrieval levers should a PM understand: chunking, hybrid search, and reranking?</b></summary>

Chunking is how you split documents before embedding: too large and a chunk buries the answer, too small and it loses context, so chunk size and overlap directly shape retrieval quality. Hybrid search combines semantic (embedding) search with keyword search so you catch both meaning and exact terms like product codes or names that embeddings miss. Reranking runs a second, stronger model over the top candidates to reorder them by true relevance before they reach the LLM, which often lifts quality more than swapping the embedding model. A PM does not tune these personally, but knowing they exist turns "retrieval is bad" into a concrete list of fixes to prioritize.

**Learn more:** [RAG roadmap](../../../resources/RAG_roadmap.md)

</details>

<details>
<summary><b>Q30. What is agentic RAG, and how does it differ from classic RAG?</b></summary>

Classic RAG runs a fixed pipeline: embed the query, retrieve top-k chunks once, and generate. Agentic RAG puts a reasoning loop around retrieval so the system can decide whether to search, reformulate the query, search multiple sources, judge whether the results are good enough, and retrieve again before answering. This handles multi-hop questions and ambiguous queries that a single retrieval pass fails, at the cost of more calls, latency, and new failure points to evaluate. As a PM you reach for it when questions genuinely need iterative or multi-source lookup, and you keep classic RAG when a single well-tuned pass already clears the bar.

**Learn more:** [What is RAG and agentic RAG](../../../free_courses/agentic_ai_crash_course/part4_what_is_rag_and_agentic.md)

</details>

<details>
<summary><b>Q31. What are the main flavors of fine-tuning, and when would you use each?</b></summary>

Supervised fine-tuning (SFT) trains on input-output pairs to teach a format, style, or narrow skill, and covers most product cases. Preference tuning (RLHF and lighter variants) shapes behavior toward what humans prefer among alternatives, useful for tone, helpfulness, and refusal behavior that is hard to specify with examples. Distillation trains a small model to imitate a larger one, trading a little quality for big cost and latency wins at scale. Parameter-efficient methods (adapters such as LoRA) make all of this cheaper by training a small set of weights instead of the whole model, which is why fine-tuning is far more accessible than it sounds.

**Learn more:** [Fine-tuning 101](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>Q32. What should a PM understand about temperature, sampling, and determinism?</b></summary>

Temperature and related sampling settings control how much randomness the model uses when picking the next token: low values make output more focused and repeatable, high values make it more varied and creative. Even at the lowest setting, LLM output is generally not perfectly deterministic in production, so you cannot assume the same input always yields the identical string. For a PM this means creative features want more variety while extraction, classification, and anything feeding downstream code want low randomness plus schema constraints. It also means your evals must tolerate acceptable variation rather than demanding exact-match on every run.

**Learn more:** [Foundations](../../../topics/foundations.md)

</details>

<details>
<summary><b>Q33. When do small or on-device models win over a frontier API?</b></summary>

Small models win when the task is narrow and well-defined (classification, routing, extraction, short structured replies) and a tuned small model already clears the quality bar, because it is cheaper, faster, and easier to run at scale. On-device or self-hosted small models also win when data cannot leave the user's device or your environment for privacy or regulatory reasons, or when you need offline operation and predictable latency. The tradeoff is lower ceiling on hard reasoning and more work to fine-tune and maintain. The disciplined move is to route by difficulty and reserve the frontier model for the calls that actually need it.

**Learn more:** [The 4 types of agentic systems](../../../free_courses/agentic_ai_crash_course/part2_the_4_types_of_agentic_systems.md)

</details>

<details>
<summary><b>Q34. What is a guardrail or classifier model, and how does it differ from the main model?</b></summary>

A guardrail model is a separate, usually smaller and cheaper model (or classifier) that screens inputs and outputs around your main model: it flags prompt injection, unsafe requests, PII, off-topic queries, or policy violations before or after the main generation. Keeping safety in a dedicated layer means you can tune and update it independently, run it fast on every call, and keep an auditable decision, rather than hoping the main model always self-polices. For a PM it is a design pattern for defense in depth: the main model does the work, the guardrail decides what is allowed through. You evaluate it on its own with precision and recall on the harms you care about, because over-blocking frustrates users and under-blocking ships harm.

**Learn more:** [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md)

</details>

## 3. Evaluation and metrics

<details>
<summary><b>Q35. Walk me through your evaluation harness for an AI feature. What is offline, and what is online?</b></summary>

Offline, I maintain a labeled eval set of representative and adversarial inputs with expected outcomes or rubrics, run it as a regression suite on every model or prompt change, and gate releases on it so quality cannot silently drop. Online, I track production signals the offline set cannot capture: task success, user acceptance and edit rates, escalation, latency, cost, and sampled human review. Offline catches regressions before launch; online catches distribution shift and real-world failure modes after. The two together, plus a diff alert when they diverge, are the harness.

**Learn more:** [Evaluation](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>Q36. How do AI product metrics differ from normal engagement and retention metrics?</b></summary>

Normal features lean on engagement and retention because more usage usually means more value. For AI features usage can rise while quality falls, so you add a quality layer underneath the outcome: correctness or faithfulness, acceptance and edit rates, hallucination rate, and escalation. The north-star is still a user outcome (task completed, time saved, ticket resolved), but you pair it with guardrail metrics that catch the model getting worse even as engagement looks fine. Input metrics alone (prompts sent, sessions) are a red flag because they measure activity, not value.

**Learn more:** [Evaluation](../../../topics/evaluation.md)

</details>

<details>
<summary><b>Q37. You ship an assistant, the north-star goes up, but the model may be getting worse. How would you know?</b></summary>

A single top-line metric can rise for reasons unrelated to quality (a UI change, a new user cohort, novelty), so I never trust it alone. I run a layered scheme: the product outcome on top, model-quality metrics beneath it (faithfulness, acceptance, edit distance, escalation, sampled human ratings), and an alarm when the two diverge. I also keep a fixed offline regression set so I can detect drift independent of traffic mix, and I sample real transcripts weekly. Silent degradation shows up as rising edits, rising escalation, or falling faithfulness even while sessions climb.

**Learn more:** [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

<details>
<summary><b>Q38. How would you measure whether a RAG system is actually working?</b></summary>

Measure retrieval and generation separately. For retrieval, use recall@k and mean reciprocal rank on a labeled set to check that the answer-bearing chunk is actually retrieved. For generation, measure faithfulness (is the answer grounded in the retrieved context with no invention), answer relevance, and correctness, using human labels and a validated LLM-as-judge. Then monitor the same signals in production plus user behavior (edits, thumbs, escalation), because a system can retrieve well and still generate badly, or vice versa.

**Learn more:** [RAG research table](../../../research_updates/rag_research_table.md)

</details>

<details>
<summary><b>Q39. Define hallucination. How would you measure the rate, and what rate would you refuse to launch above?</b></summary>

A hallucination is model output that is fluent and confident but unsupported by the source or by fact: an invented citation, a wrong number, a fabricated policy. I measure it by sampling outputs and having humans or a validated judge label each claim as supported, unsupported, or contradicted against the grounding, yielding a rate. The acceptable rate is domain-dependent: for a movie blurb a few percent is fine; for medical, legal, or financial answers the tolerable rate is near zero and I would gate hard and require citations or abstention. The honest answer names the wrongness cost of the domain rather than a universal number.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q40. What is LLM-as-judge, and what are its failure modes?</b></summary>

LLM-as-judge uses a strong model with a clear rubric to score outputs at scale, which is far cheaper than human review and good for regression testing. Its failure modes are real: position and verbosity bias, self-preference for its own style, inconsistency, and being gameable, so it can drift from human judgment. You mitigate by validating the judge against a human-labeled set, measuring their agreement, pinning the model and rubric, randomizing order, and keeping humans in the loop for high-stakes calls. Treat the judge as a calibrated instrument you audit, not as ground truth.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>Q41. What metrics matter for an agent, beyond a single-answer accuracy score?</b></summary>

Agents run multi-step, so you measure the trajectory, not just the final token. Core metrics: task success or automated resolution (did it actually accomplish the goal), containment rate and escalation rate for support agents, steps or tool calls per task, tool-call correctness, cost and latency per completed task, and safety violations. Containment measures the absence of escalation, not the presence of resolution, so pair it with a resolution or CSAT check so a vague answer the user gives up on does not count as a win. You also track where in the trajectory failures happen to know what to fix.

**Learn more:** [LLM agent evaluation guide (Confident AI)](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)

</details>

<details>
<summary><b>Q42. How do you evaluate a feature where there is no single correct answer, like summarization or open chat?</b></summary>

When there is no gold answer, shift from exact match to rubric-based grading of the qualities you care about (faithfulness, completeness, tone, safety, helpfulness), scored by humans and a validated judge on a fixed sample. Use pairwise comparisons (is A better than B) which are more reliable than absolute scores, and anchor to real user signals like acceptance, edits, and thumbs. Build a small curated set of representative and adversarial cases so the score is stable release over release. The goal is a repeatable, defensible signal, not a single accuracy number.

**Learn more:** [Evaluation](../../../topics/evaluation.md)

</details>

<details>
<summary><b>Q43. What is the difference between offline and online evaluation, and why do you need both?</b></summary>

Offline evaluation runs a fixed, labeled dataset in a controlled setting, so it is repeatable, cheap, and can gate releases before anything reaches users. Online evaluation measures the live system on real traffic through A/B tests, production metrics, and sampled human review, capturing distribution shift and real user behavior the offline set cannot foresee. Offline gives you a fast, safe regression gate; online gives you ground truth about real impact and drift. Relying on only one leaves you either blind to real-world failure or unable to catch regressions before launch.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q44. How do you build an eval set from scratch, and what is error analysis?</b></summary>

Start by collecting real or realistic inputs, then run the system and read a sample of outputs by hand, labeling each as pass or fail and writing down why it failed. That reading is error analysis: you cluster the failures into recurring modes (missing retrieval, wrong tone, ignored constraint, unsafe output) and let the biggest clusters set your priorities and your metrics. From there you curate a fixed, versioned set that covers the common cases, the important edge cases, and the failure modes you found, weighted toward what matters. The discipline is that the eval set is grown from observed failures, not invented in the abstract, so it keeps pointing at the problems that actually cost users.

**Learn more:** [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md)

</details>

<details>
<summary><b>Q45. What are trace-based or trajectory evals, and why do agents need them?</b></summary>

A trace is the full record of an agent's run: each reasoning step, tool call, tool result, and decision, not just the final answer. Trace-based evals score the path, so you can catch an agent that reached a right answer through a lucky wrong route, or a wrong answer whose failure you can localize to a specific bad tool call or planning step. This matters because the more autonomy an agent has, the less a final-answer score tells you about reliability or safety. As a PM you use traces to pinpoint where to fix (retrieval, tool choice, planning, handoff) and to build metrics per step rather than one blunt success rate.

**Learn more:** [Agentic search and retrieval table](../../../research_updates/agentic_search_retrieval_table.md)

</details>

<details>
<summary><b>Q46. What is pass^k, and why does reliability matter more than average accuracy for agents?</b></summary>

Average accuracy hides variance: an agent that succeeds 80 percent of the time on average can still fail unpredictably on the same task from run to run. pass^k measures the probability that the agent succeeds on the same task across k independent attempts, so it captures consistency, which is what a user actually experiences when they rely on a feature repeatedly. A high average with low pass^k means the agent is a coin flip in disguise, and for anything users trust to act on their behalf, reliability under repetition is the bar. As a PM you report reliability alongside average quality and gate on it for high-stakes flows.

**Learn more:** [tau-bench: benchmarking tool-agent-user interaction (arXiv)](https://arxiv.org/abs/2406.12045)

</details>

<details>
<summary><b>Q47. What is the difference between observability and evals, and do you need both?</b></summary>

Observability is instrumentation: logging every prompt, response, tool call, latency, and cost so you can see what the system did and trace an incident after the fact. Evals are judgment: a scored test of whether outputs are actually good against a rubric or labels. You need both because observability tells you what happened but not whether it was correct, and evals tell you quality but only on the slices you test. In practice observability is easier and more widely adopted, so the gap in most teams is real evals, and a strong PM pushes to close it rather than mistaking dashboards for quality measurement.

**Learn more:** [State of Agent Engineering (LangChain)](https://www.langchain.com/state-of-agent-engineering)

</details>

<details>
<summary><b>Q48. How do you evaluate whether an agent is calling tools correctly?</b></summary>

Tool-call correctness breaks into several checkable questions: did the agent choose the right tool for the step, pass valid and correct arguments, call it at the right time, and use the result properly. You build a labeled set of tasks with the expected tool calls and score selection accuracy, argument correctness, and whether unnecessary or missing calls occurred, separately from final-answer quality. This isolates a common failure class, because an agent can reason well but wire the wrong parameters or ignore a tool result. As a PM you care because tool errors are where agents quietly take wrong actions, so this eval is a safety and reliability signal, not a nicety.

**Learn more:** [LLM agent evaluation guide (Confident AI)](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)

</details>

<details>
<summary><b>Q49. A new model version drops. How do you decide whether to upgrade?</b></summary>

Never swap blind: run the candidate model against your offline regression set first, because a newer or higher-benchmark model can still regress on your specific tasks, prompts, and formats. Compare quality, cost, and latency per task, look for behavior changes (tone, refusal rate, output format) that could break downstream code or user expectations, and check the failure modes you care about most. If it clears the bar, roll it out behind a flag with an A/B or shadow test before full traffic, and keep the ability to roll back. The eval harness is exactly what makes a model upgrade a fast, safe decision instead of a gamble.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q50. Public leaderboards are saturating. How much should benchmarks drive your model choice?</b></summary>

Treat public benchmarks as a coarse filter, not a decision: many popular ones are saturated, with top models clustered above 90, so they no longer separate frontier models and they say nothing about your specific task. They are also vulnerable to contamination and to optimization for the test rather than the use case. What actually decides model choice is your own eval set on your real inputs, plus cost, latency, safety behavior, and reliability under repetition. Use leaderboards to shortlist candidates, then let your task-specific evals pick the winner.

**Learn more:** [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

<details>
<summary><b>Q51. Why is A/B testing an AI feature harder than a normal feature, and how do you do it well?</b></summary>

AI features complicate A/B testing because the output is non-deterministic, quality is multi-dimensional (helpfulness, correctness, safety, cost, latency), and novelty effects can inflate early engagement. A simple click or conversion metric can move for reasons unrelated to answer quality, so you pair the outcome metric with quality guardrails and sometimes human-rated samples per arm. You also watch cost and latency per arm, run long enough to get past novelty, and segment by input type because a model change can help common cases and hurt the long tail. The rigor is defining the success metric and the guardrails before the test, not fishing for a win after.

**Learn more:** [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/)

</details>

<details>
<summary><b>Q52. How do you keep an eval set from going stale or being overfit to?</b></summary>

An eval set decays as usage shifts and as the team implicitly tunes toward it, so treat it like a product that needs maintenance. Version it, keep a portion held out and rotated so you are not overfitting to a fixed set, and periodically add fresh cases sampled from recent production failures. Watch for the gap where offline scores climb while online quality stalls, which is the signature of overfitting to the test. The habit is continuous curation from real traffic plus rotation, so the set keeps measuring current reality rather than last quarter's.

**Learn more:** [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md)

</details>

## 4. Cost, latency, and unit economics

<details>
<summary><b>Q53. Walk me through the unit economics of an LLM feature.</b></summary>

Cost per call is driven by input tokens (the prompt, instructions, retrieved context, and history) plus output tokens, times the model's per-token price, plus retrieval and infrastructure overhead. Multiply per-call cost by expected calls per user per day and by the user base to get the real bill: a feature at 11 cents a call at a million calls a day is a budget conversation, not a rounding error. Levers to pull are a smaller or cheaper model for easy calls, prompt and context trimming, caching stable prefixes and repeated queries, capping output length, and batching. You manage cost as a product constraint that shapes design, not as an afterthought for finance.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q54. How does latency change what you build, and how do you manage it?</b></summary>

Latency shapes both the interaction and the architecture: a p95 of 4 seconds tanks adoption for an interactive feature no matter how good the answer is. You manage it by streaming tokens so perceived latency drops, routing simple requests to faster models, retrieving and reasoning in parallel where possible, caching, and reserving slow reasoning models for tasks that justify the wait. For asynchronous jobs (a nightly report) latency barely matters, so match the model and pattern to the interaction. Always design against the tail (p95 and p99), not the average, because the slow requests are the ones users remember.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q55. Accuracy or speed: improve the model by 2 points or ship a month sooner. How do you decide?</b></summary>

There is no universal answer; tie it to the cost of being wrong in this domain and the value of shipping now. In a low-stakes, reversible feature, ship sooner, learn from real users, and improve in production, because 2 offline points may not move the user outcome. In a high-stakes domain where errors are costly or hard to reverse, the 2 points may be the difference between launchable and not, so you wait. Decide with data: what do those 2 points do to the user-facing failure rate and to the metric that matters.

**Learn more:** [rounds.md, analytical and metrics](rounds.md)

</details>

<details>
<summary><b>Q56. How do you control cost and latency in an agent without wrecking quality?</b></summary>

Cache the stable prefix of the context, keep the context minimal and well-ordered, and use cheaper or smaller models for sub-steps while reserving the strong model for the hard ones. Cap tool calls and loop iterations so a stuck agent cannot spend unboundedly, cache tool results, and give the model only as much reasoning budget as the step needs. Add early-exit conditions and confidence checks so it stops when it has an answer. Then measure cost and latency per completed task so you optimize the whole trajectory, not one call.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q57. When is a bigger or more expensive model the wrong choice?</b></summary>

When a smaller model already clears the quality bar for the task, the larger one just adds cost and latency with no user-visible gain. Many production tasks (classification, extraction, short structured responses, routing) are handled well by small or mid models, so you reserve the frontier model for genuinely hard reasoning. The disciplined approach is to set the quality bar, find the cheapest model that meets it on your eval set, and route by difficulty. Defaulting to the biggest model everywhere is a common and expensive mistake.

**Learn more:** [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/)

</details>

<details>
<summary><b>Q58. What is prompt caching, and how much does it change unit economics?</b></summary>

Prompt caching stores the model's processing of a stable prefix (system instructions, tool definitions, a long document, few-shot examples) so repeated calls that share that prefix skip re-processing it, cutting both cost and latency on the cached portion substantially. It pays off whenever many calls share a large fixed context, which is common in agents, RAG over the same corpus, and long system prompts. As a PM this reshapes design: you order context so the stable part is cacheable and the variable part comes last, and you can afford richer system prompts than raw per-token math suggests. The catch is that only the shared prefix benefits, so you architect prompts to maximize what stays constant.

**Learn more:** [Prompt caching (Anthropic)](https://www.anthropic.com/news/prompt-caching)

</details>

<details>
<summary><b>Q59. How do you protect gross margin on an AI product as usage scales?</b></summary>

Unlike traditional software, inference is a real marginal cost per use, so heavy users can erode or invert margin if pricing is flat and unlimited. You protect margin by routing easy calls to cheaper models, caching, trimming context, capping output and loop length, and setting fair-use limits or usage-based tiers so the biggest consumers cover their cost. You also watch cost per active user and cost per unit of value delivered as first-class business metrics, not just aggregate spend. The strategic tailwind is that model prices fall over time, but you design for margin now rather than betting the product on future price cuts.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>Q60. Batch versus real-time inference: when does each matter?</b></summary>

Real-time inference serves an interactive request where the user is waiting, so latency and streaming dominate the design and you accept higher per-call cost for responsiveness. Batch inference processes many inputs together on a relaxed schedule (overnight enrichment, bulk classification, generating embeddings for a corpus), which is cheaper and often available at a discount, and where a few minutes or hours of latency is irrelevant. As a PM you sort each workload by whether a human is waiting: interactive goes real-time, background and bulk work goes batch to save money. Mixing them up wastes budget on speed nobody needs or starves an interactive feature that needed it.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q61. Time to first token versus total latency: which do you optimize?</b></summary>

For interactive, streamed experiences, time to first token (how fast something starts appearing) drives perceived responsiveness more than total completion time, because a fast start with streaming feels quick even if the full answer takes longer. For non-streamed or batch outputs, total latency is what matters because the user only sees the finished result. So you optimize time to first token for chat and assistants (stream early, keep the opening cheap) and total latency for tasks that deliver a complete artifact at once. Knowing which one the interaction depends on tells you whether streaming and prompt ordering or raw model speed is the right lever.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q62. Inference costs keep falling fast. How should that shape strategy and pricing?</b></summary>

Falling prices mean a feature that is too expensive today may be viable soon, so you keep a portfolio: ship what is affordable now and stage the ones that are one price cut away rather than abandoning them. On pricing, avoid locking in flat unlimited plans that assume today's cost forever; prefer models that let margin improve as cost drops, and pass some savings to users to drive adoption. Architect behind a model abstraction so you can adopt the next cheaper or better model quickly, which is where the eval harness earns its keep. The strategic risk is over-building expensive infrastructure for a cost problem the market will solve, or under-pricing against a cost that has not fallen yet.

**Learn more:** [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)

</details>

## 5. Responsible AI, safety, and agents

<details>
<summary><b>Q63. Your model performs well for 90 percent of users but poorly for one demographic. What do you do, and on what timeline?</b></summary>

Treat it as a live product defect, not a research footnote, and quantify the gap and the harm first. If the harm is serious (denying service, unsafe output), gate or roll back the affected path immediately while you fix, rather than shipping a known biased experience. Then diagnose the cause (unrepresentative training or eval data, thresholds, retrieval gaps), fix it, and add a slice of that demographic to the permanent eval set so it cannot regress silently. The timeline is driven by severity: mitigate now, root-cause and durably fix on a committed schedule, and report transparently.

**Learn more:** [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)

</details>

<details>
<summary><b>Q64. How would you design guardrails for an agentic system that can take actions on a user's behalf?</b></summary>

Constrain what the agent can do before you constrain what it says: scoped, least-privilege permissions so it can only touch what the task requires. Add a human confirmation step for anything irreversible or high-consequence (sending money, deleting data, emailing customers), plus spending and rate limits, and input and output filters for prompt injection, PII leakage, and unsafe actions. Log every action for audit and give someone a kill switch they actually own and have tested. Then red-team the agent against adversarial and injection scenarios before launch and monitor for anomalies after.

**Learn more:** [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>Q65. What is prompt injection, and why does it matter more for agents?</b></summary>

Prompt injection is when malicious instructions hidden in user input or in content the model reads (a web page, an email, a document) hijack the model into ignoring its real instructions. It matters more for agents because they read untrusted external content and can take actions, so a successful injection can make them exfiltrate data, call tools maliciously, or perform unauthorized actions, not just produce bad text. Defenses include separating trusted instructions from untrusted content, least-privilege tool permissions, output and action filtering, human confirmation for sensitive steps, and red-teaming. There is no complete fix, so you design assuming some injections will get through.

**Learn more:** [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md) and [OWASP Top 10 for LLM Apps](https://genai.owasp.org/llm-top-10/)

</details>

<details>
<summary><b>Q66. How do you build fairness, privacy, and transparency into the spec instead of auditing for them later?</b></summary>

Write them in as requirements and acceptance criteria from the PRD onward, not as a review at the end. For fairness, define the user segments and the eval slices up front and set thresholds each must meet to launch. For privacy, minimize what data enters the prompt and any vendor, specify retention and redaction, and decide what may be used for training. For transparency, spec what the product tells users about AI involvement, confidence, and sources. Making these launch gates means they shape design rather than becoming expensive retrofits.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q67. What is your fallback the first time the model is confidently wrong in front of a customer?</b></summary>

Assume it will happen and design the unhappy path before launch. Put a confidence gate in front of consequential output, and when confidence is low or a check fails, degrade gracefully: say "I am not sure, here is a human" or a safe default rather than bluffing. Log the incident with the input and output, alert if the rate crosses a threshold, and feed the case into the eval set and a retraining or prompt-fix trigger. Anyone can describe the happy path; the maturity signal is having already built for the wrong one.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q68. How do regulations like the EU AI Act affect an AI PM's launch decisions?</b></summary>

The EU AI Act classifies systems by risk, and high-risk uses (employment, credit, health, and similar) carry obligations: risk management, data governance, human oversight, transparency, logging, and documentation. As a PM you classify your feature early, because a high-risk classification adds required launch gates (documented evals, human-in-the-loop, audit-ready logging, disclosure) and changes the timeline and design. Even outside the EU, this maps well onto good practice frameworks like the NIST AI Risk Management Framework, so treating audit-ready telemetry and human oversight as defaults is both compliant and good product. The practical move is to know your risk tier before you commit a date.

**Learn more:** [EU AI Act explorer](https://artificialintelligenceact.eu/)

</details>

<details>
<summary><b>Q69. How do you reduce hallucinations in a customer-facing product?</b></summary>

There is no way to eliminate hallucination, so you stack mitigations and design for the residual. Ground answers in retrieval (RAG) so the model answers from trusted sources, require citations, and adopt an abstain-when-unsupported policy so an unsupported claim yields "I do not know" plus a human handoff rather than a confident guess. Use verification steps (re-query sources, self-check, or a second model) for high-stakes claims, and measure the hallucination rate continuously with a gate you will not launch above. Match the strictness to the wrongness cost of the domain.

**Learn more:** [RAG](../../../topics/rag.md)

</details>

<details>
<summary><b>Q70. Prompt injection and jailbreaks are often confused. What is the difference?</b></summary>

A jailbreak is a user trying to trick the model into breaking its own safety rules for that user, for example coaxing it to produce disallowed content. Prompt injection is a third party smuggling instructions into content the model consumes (a web page, an email, a document) so the model acts against the user or the system without the user's intent. The distinction matters because the threat model differs: jailbreaks are about output policy, injection is about untrusted data hijacking behavior and tools, which is far more dangerous for agents that take actions. You defend jailbreaks mainly with safety training and output filters, and injection with data-versus-instruction separation, least privilege, and action confirmation.

**Learn more:** [OWASP Top 10 for LLM Apps](https://genai.owasp.org/llm-top-10/)

</details>

<details>
<summary><b>Q71. How do you handle training on customer data and PII in an AI product?</b></summary>

Start from data minimization: only send the model what the task needs, redact or tokenize PII where you can, and be explicit about what leaves your environment to any vendor. Separate the questions of what data is used for inference versus what is used for training, get clear consent and contractual terms for each, and default to not training on customer data unless you have explicit permission. Specify retention, deletion, and access controls, and check whether the deployment (region, on-prem, or a no-training vendor tier) matches the sensitivity and any regulation like GDPR or HIPAA. As a PM you write these as requirements, because a privacy mistake with customer data is a trust and legal event, not a bug.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q72. Define levels of agent autonomy and when each requires human approval.</b></summary>

Think of a ladder: read-only (the agent retrieves and informs), suggest (it proposes an action the user takes), draft (it prepares an action the user reviews and confirms), and act (it executes within scoped limits). Human approval is required as you climb toward irreversible or high-consequence actions, so sending money, external communication, and data deletion sit behind explicit confirmation regardless of how good the model is. You place a feature on the ladder by reversibility and cost of error, start lower than you think you need, and earn autonomy as evals and production reliability prove out. The point is that autonomy is a dial you set deliberately per action, not a single on-off switch for the whole agent.

**Learn more:** [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)

</details>

<details>
<summary><b>Q73. How do you red-team an AI product before launch?</b></summary>

Red-teaming is deliberately trying to make the system misbehave before real users do: you gather a diverse set of adversarial inputs (jailbreaks, prompt injection, edge cases, offensive or manipulative prompts, out-of-scope asks) and run them against the product, logging every failure. You cover the harms that matter for your domain (unsafe instructions, PII leakage, biased output, unauthorized actions for agents) and mix human creativity with automated attack generation for scale. The findings feed guardrails, prompt fixes, and permanent additions to the eval set so the same attack cannot regress. As a PM you make a red-team pass a launch gate, not an optional extra, for anything customer-facing or action-taking.

**Learn more:** [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md)

</details>

<details>
<summary><b>Q74. Copyright, IP, and generated content: what does an AI PM need to worry about?</b></summary>

Generated output raises several distinct risks: the model may reproduce copyrighted or trademarked material, it may leak proprietary data that entered the prompt, and the ownership and licensing status of AI-generated content itself can be unsettled. For a product you decide what sources the model may draw on, whether you filter or attribute outputs, and what you promise users about originality and rights. You also watch inbound risk (users pasting confidential or infringing content) and set terms and guardrails accordingly. The PM stance is to treat this as a policy and design question with legal, not a technical afterthought, because the exposure is commercial and reputational.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q75. Over-refusal is a real product problem. How do you balance safety and helpfulness?</b></summary>

Safety tuning that is too aggressive makes a model refuse legitimate requests, which frustrates users and drives them to a competitor, so refusal is a quality metric you measure, not just a safety win. You define, ideally in a written model or product spec, what the product should refuse, what it should do carefully, and what it should freely help with, then evaluate both harmful-request blocking and legitimate-request completion so you can see the tradeoff explicitly. When over-refusal shows up, you narrow the guardrails toward the actual harms rather than broad topic bans, and you test on benign edge cases that look risky but are fine. The mature answer treats helpfulness and safety as a calibrated balance with metrics on both sides, not a one-way ratchet toward blocking.

**Learn more:** [Safety and Security](../../../topics/safety-security.md)

</details>

<details>
<summary><b>Q76. For a generative media product, how do you handle provenance and misuse (deepfakes, synthetic content)?</b></summary>

Generative image, audio, and video products carry misuse risk (impersonation, non-consensual content, misinformation) that you design against from the start: input and output filters, restrictions on generating real people or protected content, and rate and identity controls. On provenance, you can attach signals that content is AI-generated (visible labels and machine-readable metadata or watermarks) so downstream systems and people can tell, which is increasingly expected and in some places required. You also plan a reporting and takedown path and monitor for abuse patterns. As a PM you weigh creative value against harm and build the safeguards as launch gates, because the failure mode here is real-world harm to third parties, not just a bad user experience.

**Learn more:** [Multimodal](../../../topics/multimodal.md)

</details>

## 6. Business and strategy

<details>
<summary><b>Q77. Build versus buy for a core AI capability: how do you decide?</b></summary>

Weigh token cost against control, latency against time to market, the data moat you actually own against the one you wish you had, and the switching risk of betting the product on one vendor's roadmap. Buying an API wins on speed, frontier quality, and lower upfront cost and is right for most features. Building (fine-tuning or self-hosting) wins when you have proprietary data that creates a real advantage, when volume makes per-call price painful, or when you need control the vendor cannot give (latency, privacy, on-prem). Decide per capability, not once for the company, and keep an exit path so one vendor cannot hold the product hostage.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>Q78. How would you think about pricing an AI feature given variable inference costs?</b></summary>

Unlike traditional software, marginal cost is real and scales with usage, so flat unlimited pricing can lose money on power users. Options include usage-based pricing, tiered plans with caps, or bundling AI into a higher tier, and increasingly outcome-based pricing where you charge for a resolved ticket or completed task rather than per call. Model the cost per unit of value delivered and make sure price sits comfortably above it at the usage levels you expect, with room for cost to fall as models get cheaper. Also watch that pricing does not discourage the very usage that creates value.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q79. What is a durable moat for an AI product when everyone can call the same models?</b></summary>

The model is rarely the moat because competitors can call the same API, so durability comes from what surrounds it: proprietary data and feedback loops, deep workflow integration and distribution, switching costs, and a superior evaluation and quality process that competitors cannot easily copy. A tight loop where usage generates data that improves the product compounds over time. Brand, trust, and being embedded in a user's real workflow also hold. As a PM you invest in the data flywheel, the integration surface, and the eval discipline rather than assuming the model choice is the differentiator.

**Learn more:** [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/)

</details>

<details>
<summary><b>Q80. How do you build an AI product roadmap when the underlying models change every few months?</b></summary>

Anchor the roadmap to durable user problems and outcomes, then treat model capability as a fast-moving input you re-check often. Architect so you can swap models behind an abstraction, so a better or cheaper model is an upgrade rather than a rebuild. Sequence bets by what is reliable now versus what is one capability jump away, and keep near-term commitments concrete while holding a portfolio of options for capabilities that are close but not ready. Build the eval harness first, because it is what lets you adopt a new model safely and quickly when it lands.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

<details>
<summary><b>Q81. How would you measure the ROI of an AI feature to justify continued investment?</b></summary>

Tie it to a business outcome, not to activity: time saved, tickets deflected or resolved, conversion lift, revenue influenced, or cost reduced, measured ideally with a holdout or A/B comparison against the pre-AI baseline. Net out the true cost, including inference, retrieval, human review, and maintenance, so you report value minus cost rather than gross usage. Attribute carefully, because usage alone (prompts sent) is not value. A credible ROI story pairs a specific outcome metric with a controlled comparison and the fully loaded cost.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>Q82. The model provider could ship your feature as a native capability. How do you survive platform risk?</b></summary>

Assume the frontier labs will absorb thin wrappers, so you build where they will not or cannot: deep integration into a specific workflow, proprietary data and context they do not have, domain expertise and trust, and the last-mile reliability, compliance, and support an enterprise needs. You compete on being embedded in how a particular user or industry actually works, not on being a nicer chat window over the same model. Owning the data flywheel and the eval discipline for your niche is what a general model release does not replicate. The strategic question in every roadmap review is whether a new base-model feature makes you redundant or makes you better, and you steer toward the latter.

**Learn more:** [Emerging Architectures for LLM Applications (a16z)](https://a16z.com/emerging-architectures-for-llm-applications/)

</details>

<details>
<summary><b>Q83. Open-weight models versus proprietary APIs: how do you think about the strategy?</b></summary>

Proprietary APIs give you frontier quality, safety work, and zero infrastructure at the cost of per-call price, data-handling constraints, and vendor dependence. Open-weight models give you control, the ability to self-host for privacy or on-prem needs, no per-token fee, and freedom from one vendor's roadmap, at the cost of running and maintaining infrastructure and often a quality gap on the hardest tasks. You choose by requirements: pick open weights when data cannot leave your environment, when volume makes hosting cheaper than API fees, or when you need to fine-tune deeply; pick the API for fastest access to the best quality. Many teams run a hybrid, and keeping a model abstraction lets you move between them as economics and quality shift.

**Learn more:** [Fine-tuning 101](../../../resources/fine_tuning_101.md)

</details>

<details>
<summary><b>Q84. How do you build and defend a data flywheel for an AI product?</b></summary>

A data flywheel is a loop where usage generates data (interactions, corrections, thumbs, accepted or rejected outputs) that you feed back to improve the product, which drives more usage. You design it deliberately: instrument the signals worth capturing, build the pipeline to turn them into better retrieval, prompts, fine-tunes, or eval cases, and close the loop so improvement is continuous rather than a one-off. It defends against competitors because the loop compounds on data they do not have and cannot buy. As a PM you protect it by getting consent and rights to use the data, keeping the feedback signal clean, and making sure the improvement actually reaches users rather than sitting in a warehouse.

**Learn more:** [GenAI roadmap](../../../resources/genai_roadmap.md)

</details>

<details>
<summary><b>Q85. How do you measure product-market fit for an AI product?</b></summary>

Standard PMF signals still apply (retention, organic growth, users who would be very disappointed to lose it), but you read them through an AI lens because novelty can fake early traction. You separate curiosity usage from durable value by watching whether users return for the real job, whether quality is high enough that they trust the output without heavy editing, and whether the outcome metric (time saved, task completed) holds up beyond the honeymoon. Rising edit rates, falling repeat use, or engagement that decays after the novelty wears off signal shallow fit even when sign-ups look great. Real PMF for an AI product is users relying on it for the outcome, not trying it because it is new.

**Learn more:** [README: what the job actually is](README.md)

</details>

<details>
<summary><b>Q86. Users do not trust or adopt the AI feature. How do you handle adoption and change management?</b></summary>

Low adoption of a good feature is usually a trust and workflow problem, not a model problem, so diagnose why: is it hard to find, does one bad early experience sour users, does it not fit how they actually work, or do they fear being wrong or replaced. You build trust with transparency (show sources and confidence), keep the human in control so trying it is low-risk, and prove value on a narrow high-confidence use case before expanding. Onboarding, in-product education, and honest framing of what it can and cannot do matter as much as the model quality. You measure adoption alongside quality, because a feature nobody trusts enough to use delivers zero value no matter how good the offline scores are.

**Learn more:** [Use AI journey](../../../journeys/use.md)

</details>

<details>
<summary><b>Q87. When the base model is a commodity every competitor can call, how do you position the product?</b></summary>

You compete on everything the shared model does not give anyone: a specific workflow solved end to end, proprietary data and context, trust and reliability in a domain, integration and distribution, and a quality process competitors cannot copy. Positioning shifts from "we use the best AI" to "we solve this job better than anyone," because the model is table stakes, not a differentiator. You lean on outcomes and depth in a niche rather than raw model capability in your messaging. The strongest position is being the system of record and workflow for a particular user, where switching away means losing accumulated context and integration, not just changing a chat window.

**Learn more:** [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md)

</details>

## 7. Execution and behavioral

<details>
<summary><b>Q88. How do you write a PRD for a feature whose output is probabilistic and occasionally wrong?</b></summary>

Beyond the normal problem, users, and goals, a probabilistic PRD specifies the quality bar and how it is measured (the eval set and thresholds that gate launch), the expected failure modes, and exactly how the experience degrades around each one (confidence gates, fallbacks, undo, citations, escalation). It states the cost and latency budgets, the responsible-AI requirements as acceptance criteria, and the rollback plan. It treats prompts and the eval set as first-class spec artifacts, because for an AI feature the prompt is part of the product definition. The theme is designing the unhappy path deliberately rather than assuming the model is always right.

**Learn more:** [rounds.md, cross-functional and execution](rounds.md)

</details>

<details>
<summary><b>Q89. Tell me about an AI feature you killed or an AI concept you had to defend to a skeptical stakeholder.</b></summary>

Use a real story in STAR form with numbers. For a kill: name the feature, the metric that told you it was not working (low acceptance, high hallucination or escalation, cost that outran value), the experiment or data behind the call, and what you did with the team and the learning afterward. For defending a concept: describe the stakeholder who wanted more than the model could reliably do, how you translated the limitation into plain business terms, and how you held a responsible line while still finding a path to value. The signal interviewers want is judgment and ownership: you have a graveyard, you decide with data, and you take responsibility for the call.

**Learn more:** [rounds.md, behavioral and leadership](rounds.md)

</details>

<details>
<summary><b>Q90. Write the launch-readiness checklist for a customer-facing LLM feature.</b></summary>

A good checklist gates on the AI-specific risks, not just the normal ones: the offline eval set passes its thresholds, a red-team pass covers jailbreaks and injection, the fallback and abstention paths are built and tested, and cost and latency budgets are met at expected scale. It confirms observability and logging are in place, a rollback and kill switch exist and have been exercised, the hallucination and safety guardrails have measured rates you accept, and any compliance or disclosure requirements for the risk tier are satisfied. It names the online metrics and alarms that will watch for degradation after launch, and it identifies the owner for incidents. The difference from a normal launch checklist is that quality, safety, and the unhappy path are explicit gates rather than assumptions.

**Learn more:** [Production](../../../topics/production.md)

</details>

<details>
<summary><b>Q91. You have 1 data scientist and 3 AI features waiting. How do you prioritize?</b></summary>

Prioritize by expected value against the wrongness cost and the real cost to build, not by which model work is most exciting. Estimate for each feature the user or business upside, the probability the model is reliable enough to ship, and the effort the scarce data-science time actually requires, then sequence the highest value-per-unit-of-scarce-resource first. Prefer the feature where a cheap prompt-and-eval approach can validate value before you spend heavy fine-tuning time, so you de-risk with the least scientist effort. You also protect the data scientist's bandwidth from flashy work that does not move the outcome, and you make the tradeoff visible to stakeholders rather than quietly starving two features.

**Learn more:** [rounds.md, cross-functional and execution](rounds.md)

</details>

<details>
<summary><b>Q92. The eng lead says your eval bar is unrealistic and will slip the date. How do you handle it?</b></summary>

Treat it as a real tradeoff to reason through together, not a line to defend on authority. Get concrete: which specific thresholds are at issue, what the failure looks like if you ship below them, and what the cost of that failure is in this domain, because a bar that is safety-critical is non-negotiable while a stylistic one might flex. Look for options that protect users and the date: a staged rollout behind a flag, launching to a low-risk segment first, or shipping with a tighter fallback while quality improves in production. You hold the line where the wrongness cost demands it and give ground where it does not, and you make the decision and its risk explicit so it is a shared call, not a standoff.

**Learn more:** [rounds.md, cross-functional and execution](rounds.md)

</details>

<details>
<summary><b>Q93. Tell me about a model you owned that degraded in production. How did you catch it, and what did you do?</b></summary>

Use a real STAR story that shows a working detection system, not luck. Describe the layered monitoring that caught it (a guardrail metric like rising edit or escalation rate, or a faithfulness drop on sampled transcripts, diverging from a steady north-star), and how a fixed offline regression set let you confirm real drift rather than a traffic-mix artifact. Then the response: mitigate fast (roll back the model or prompt, tighten the fallback), root-cause the degradation (a model update, distribution shift, a data or retrieval change), fix it durably, and add the failure to the permanent eval set so it cannot recur silently. The signal is ownership and a system that surfaces silent failure, plus taking responsibility instead of blaming the model.

**Learn more:** [rounds.md, behavioral and leadership](rounds.md)

</details>

<details>
<summary><b>Q94. How do you manage leadership's expectations when AI hype outpaces what the model can reliably do?</b></summary>

Translate capability into plain business terms and anchor on what is measured, not what a demo suggested: show the reliability rate on real inputs, the cost and latency, and the failure modes, so the conversation is grounded in evidence. Reframe the ask toward the durable user problem and a path that ships value now while riskier capability matures, rather than promising magic on a deadline. Offer a cheap experiment to settle a disagreement with data instead of opinion, and be willing to say what the model cannot yet do reliably, because a credible no protects the roadmap and your credibility. The maturity signal is holding a responsible line under pressure while still finding a real path to value.

**Learn more:** [Microsoft AI PM interview guide (Exponent)](https://www.tryexponent.com/guides/microsoft-ai-product-manager-interview)

</details>

---

Keep going: verify your weak areas against [resources.md](resources.md), the [topic pages](../../../topics/foundations.md), and the [60 GenAI Interview Questions](../../60_gen_ai_questions.md). Then run the [prep plan](prep-plan.md).
