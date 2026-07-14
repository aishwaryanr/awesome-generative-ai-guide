# AI Product Manager: Question Bank

45 questions with concise model answers, grouped by theme. Answers are 3 to 6 sentences: enough to anchor your own, not a script to recite. Add a specific example from your own work to every one before an interview. Pair with the [60 GenAI Interview Questions](../../60_gen_ai_questions.md) and the [Role-Based Prep](../../role_based_prep.md).

Themes:
1. [Capability judgment and product sense](#1-capability-judgment-and-product-sense) (Q1-7)
2. [Model and architecture literacy](#2-model-and-architecture-literacy) (Q8-17)
3. [Evaluation and metrics](#3-evaluation-and-metrics) (Q18-26)
4. [Cost, latency, and unit economics](#4-cost-latency-and-unit-economics) (Q27-31)
5. [Responsible AI, safety, and agents](#5-responsible-ai-safety-and-agents) (Q32-38)
6. [Business and strategy](#6-business-and-strategy) (Q39-43)
7. [Execution and behavioral](#7-execution-and-behavioral) (Q44-45)

---

## 1. Capability judgment and product sense

**Q1. How do you decide whether a problem needs AI at all, versus a rules engine or plain software?**
Start from the user outcome and the cost of being wrong, not from the technology. Use deterministic software or a rules engine when the logic is stable, auditable, and the cost of an error is high; reach for a model only when the input is messy, open-ended, or high-variety in a way rules cannot cover. A useful test: if you can write the rules down and they do not explode in number, write the rules. Boring, deterministic solutions often win, are cheaper, and never hallucinate, so the burden of proof is on adding the model.

**Q2. When would you reach for a rules-based system, a classic ML model, and an LLM?**
Rules fit stable, explainable logic with clear inputs (tax brackets, eligibility checks). Classic ML (gradient boosting, logistic regression) fits structured, tabular prediction at scale with labeled history: fraud scoring, churn, ranking. LLMs fit unstructured language and generation: summarization, extraction from free text, conversation, code. The choice also weighs latency, cost, interpretability, and how much labeled data you have; you often stack them, for example rules to gate and an LLM to handle the long tail.

**Q3. Design an AI feature for our product. Where would the model add value, and where would it just add risk?**
First name the user and the job to be done, then split the workflow into steps and ask which steps are language-heavy, high-variety, or currently manual and slow. Put the model where it removes real toil (drafting, summarizing, triaging) and keep deterministic control where correctness is legally or financially load-bearing (final calculations, permissions, sending money). For each model-touched step, decide the fallback when it is wrong and how you will measure quality. The answer that scores well says out loud where you would refuse to put a model.

**Q4. A stakeholder wants to add AI to a feature that works fine without it. How do you handle that conversation?**
Get to the underlying goal: is this a real user problem, a differentiation worry, or leadership pressure to look AI-forward. Acknowledge the goal, then compare the current solution against an AI version on user value, cost, latency, and new failure modes, ideally with a cheap experiment rather than an opinion. Often the right move is a narrow pilot on one high-variety slice where AI clearly helps, while leaving the working path alone. Protecting a good deterministic experience from an unnecessary model is a product win, not obstruction.

**Q5. Pick an AI product you admire. What is the one design decision that made it work?**
Choose a product you have actually used and name a single decision that shaped its success, tied to a principle you can reuse. For example, a coding assistant that shows a diff you accept or reject keeps the human in control and makes wrongness cheap to reject, which is why people trust it despite imperfect output. The point is to show you can reverse-engineer why a design works: control, low cost of error, fast feedback, or graceful degradation. Avoid vague praise; name the mechanism.

**Q6. How is product sense for AI different from classic product sense?**
Classic product sense starts at "what should we build." AI product sense starts one step earlier, at "should a model be anywhere near this," because a model brings non-determinism, cost, latency, and new failure modes that a deterministic feature does not. You also design for wrongness from the first sketch (fallbacks, confidence, undo, citations) and you carry an evaluation and cost story through the whole design rather than treating quality as an engineering detail. Everything downstream (users, use cases, metrics) still applies.

**Q7. How would you design an AI feature so users trust it?**
Trust comes from control, transparency, and graceful failure, not from the model being perfect. Keep the user in the loop for consequential actions (preview, confirm, undo), show sources or reasoning where it helps, and set honest expectations about what the feature can and cannot do. When the model is unsure, say so and route to a human or a safe default rather than bluffing. Then measure trust: acceptance rate, edit rate, escalation rate, and repeat use, and treat a confident wrong answer as a serious incident.

## 2. Model and architecture literacy

**Q8. When would you use RAG instead of fine-tuning to give a model new knowledge?**
Use RAG when the knowledge changes often, is large or proprietary, or needs citations: retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, or style, or to bake in a stable skill, not for volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior. RAG is also usually faster and cheaper to iterate on, since you update an index instead of running a training job. See [RAG](../../../topics/rag.md) and [Fine-tuning](../../../topics/fine-tuning.md).

**Q9. Explain RAG to a non-technical executive in under a minute.**
The model is smart but does not know our private or up-to-date information, and it will confidently make things up to fill the gap. RAG fixes that by first searching our own trusted documents for the relevant passages, then handing those passages to the model and asking it to answer using only them, with citations. So instead of guessing from memory, it answers open-book from our sources, which cuts hallucinations and lets us update knowledge by updating documents, not by retraining. The tradeoff is that answer quality now depends on whether the search step actually finds the right passage.

**Q10. A RAG system is giving wrong answers. How do you diagnose retrieval versus generation?**
Split the pipeline and inspect each half. Pull the retrieved chunks for the failing questions: if the answer-bearing passage is not in the retrieved set, it is a retrieval problem (fix chunking, embeddings, the query, or re-ranking). If the right passage was retrieved but the model still answered wrong or ignored it, it is a generation or grounding problem (fix the prompt, force citation, or use a stronger model). Measuring retrieval (recall@k) and generation (faithfulness) separately turns a vague "it is bad" into a fixable diagnosis.

**Q11. What makes something an agent rather than a single LLM call, and when do you actually need one?**
An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow (a chain of prompts you designed) will do, because agents add cost, latency, and new failure modes and are harder to evaluate and control. The mature default is the simplest thing that works: prompt, then workflow, then agent only if the task genuinely demands autonomy. See [Agents](../../../topics/agents.md).

**Q12. What is context engineering, and why can it matter more than prompt wording?**
Context engineering is deciding what actually enters the model's context window: instructions, retrieved documents, tool results, memory, and what stays out. It matters because models degrade when the context is bloated, noisy, or poorly ordered, so getting the right minimal information in the right place usually beats clever phrasing. For a PM it reframes many quality problems as context problems (wrong or missing retrieval, stale memory, tool output the model cannot parse) rather than prompt-tweaking. Managing the context budget is now a core product lever for quality, cost, and latency.

**Q13. What are reasoning models, and how do they change what you can build?**
Reasoning models are trained to spend extra test-time compute on an internal chain of thought before answering, often via reinforcement learning on verifiable rewards. They are much stronger on math, code, and multi-step planning, at the cost of higher latency and price per answer. As a PM you route hard, multi-step, high-stakes tasks to a reasoning model and keep fast, simple tasks on a cheaper model, because paying reasoning cost and latency on every trivial call is waste. You also prompt them differently: give the goal and constraints and let the model do the steps rather than over-scripting them.

**Q14. What is the Model Context Protocol (MCP) and why does a PM care?**
MCP is an open standard for connecting an agent or app to tools and data sources through a common interface. You care because it turns integrations into reusable building blocks: wire a capability once (a database, a ticketing system, a search tool) and reuse it across agents and harnesses instead of a bespoke integration per tool. That lowers the cost and time to give your product new capabilities and makes the ecosystem of connectors something you can buy into rather than rebuild. It is an execution and platform-strategy lever, not just an engineering detail.

**Q15. How do you decide between a third-party foundation-model API, a fine-tune, and a custom-trained model?**
Default to a third-party API: fastest to ship, no training cost, and you inherit frontier quality and safety work. Move to fine-tuning a base or open model when you need behavior, tone, or a narrow skill the API cannot reliably deliver through prompting, or when volume makes the per-call price painful. Train a custom model almost never, only with a real data moat, scale, and a capability gap no vendor fills, because the cost and talent bar are extreme. Weigh token cost against control, latency against time to market, and the switching risk of betting the product on one vendor's roadmap.

**Q16. Explain embeddings to a non-technical executive in under a minute.**
An embedding turns a piece of text (or an image) into a list of numbers that captures its meaning, so that things with similar meaning land near each other in that number space. That lets a computer measure similarity: "cancel my plan" and "how do I end my subscription" sit close together even though they share few words. We use this to power semantic search, retrieval for RAG, clustering, and recommendations. The practical upshot is that we can find relevant content by meaning rather than exact keywords.

**Q17. What is prompt engineering versus fine-tuning versus RAG, and how do they combine?**
Prompting shapes behavior at call time through instructions, examples, and context, with zero training cost and instant iteration. Fine-tuning changes the model's weights to bake in a stable behavior, format, or skill. RAG injects fresh external knowledge at call time by retrieving relevant documents. They compose: RAG supplies the facts, fine-tuning locks in the voice or output format, and prompting orchestrates the call, so a real product often uses all three, and you always try prompting first because it is the cheapest to change.

## 3. Evaluation and metrics

**Q18. Walk me through your evaluation harness for an AI feature. What is offline, and what is online?**
Offline, I maintain a labeled eval set of representative and adversarial inputs with expected outcomes or rubrics, run it as a regression suite on every model or prompt change, and gate releases on it so quality cannot silently drop. Online, I track production signals the offline set cannot capture: task success, user acceptance and edit rates, escalation, latency, cost, and sampled human review. Offline catches regressions before launch; online catches distribution shift and real-world failure modes after. The two together, plus a diff alert when they diverge, are the harness. See [Evaluation](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md).

**Q19. How do AI product metrics differ from normal engagement and retention metrics?**
Normal features lean on engagement and retention because more usage usually means more value. For AI features usage can rise while quality falls, so you add a quality layer underneath the outcome: correctness or faithfulness, acceptance and edit rates, hallucination rate, and escalation. The north-star is still a user outcome (task completed, time saved, ticket resolved), but you pair it with guardrail metrics that catch the model getting worse even as engagement looks fine. Input metrics alone (prompts sent, sessions) are a red flag because they measure activity, not value.

**Q20. You ship an assistant, the north-star goes up, but the model may be getting worse. How would you know?**
A single top-line metric can rise for reasons unrelated to quality (a UI change, a new user cohort, novelty), so I never trust it alone. I run a layered scheme: the product outcome on top, model-quality metrics beneath it (faithfulness, acceptance, edit distance, escalation, sampled human ratings), and an alarm when the two diverge. I also keep a fixed offline regression set so I can detect drift independent of traffic mix, and I sample real transcripts weekly. Silent degradation shows up as rising edits, rising escalation, or falling faithfulness even while sessions climb.

**Q21. How would you measure whether a RAG system is actually working?**
Measure retrieval and generation separately. For retrieval, use recall@k and mean reciprocal rank on a labeled set to check that the answer-bearing chunk is actually retrieved. For generation, measure faithfulness (is the answer grounded in the retrieved context with no invention), answer relevance, and correctness, using human labels and a validated LLM-as-judge. Then monitor the same signals in production plus user behavior (edits, thumbs, escalation), because a system can retrieve well and still generate badly, or vice versa.

**Q22. Define hallucination. How would you measure the rate, and what rate would you refuse to launch above?**
A hallucination is model output that is fluent and confident but unsupported by the source or by fact: an invented citation, a wrong number, a fabricated policy. I measure it by sampling outputs and having humans or a validated judge label each claim as supported, unsupported, or contradicted against the grounding, yielding a rate. The acceptable rate is domain-dependent: for a movie blurb a few percent is fine; for medical, legal, or financial answers the tolerable rate is near zero and I would gate hard and require citations or abstention. The honest answer names the wrongness cost of the domain rather than a universal number.

**Q23. What is LLM-as-judge, and what are its failure modes?**
LLM-as-judge uses a strong model with a clear rubric to score outputs at scale, which is far cheaper than human review and good for regression testing. Its failure modes are real: position and verbosity bias, self-preference for its own style, inconsistency, and being gameable, so it can drift from human judgment. You mitigate by validating the judge against a human-labeled set, measuring their agreement, pinning the model and rubric, randomizing order, and keeping humans in the loop for high-stakes calls. Treat the judge as a calibrated instrument you audit, not as ground truth.

**Q24. What metrics matter for an agent, beyond a single-answer accuracy score?**
Agents run multi-step, so you measure the trajectory, not just the final token. Core metrics: task success or automated resolution (did it actually accomplish the goal), containment rate and escalation rate for support agents, steps or tool calls per task, tool-call correctness, cost and latency per completed task, and safety violations. Containment measures the absence of escalation, not the presence of resolution, so pair it with a resolution or CSAT check so a vague answer the user gives up on does not count as a win. You also track where in the trajectory failures happen to know what to fix.

**Q25. How do you evaluate a feature where there is no single correct answer, like summarization or open chat?**
When there is no gold answer, shift from exact match to rubric-based grading of the qualities you care about (faithfulness, completeness, tone, safety, helpfulness), scored by humans and a validated judge on a fixed sample. Use pairwise comparisons (is A better than B) which are more reliable than absolute scores, and anchor to real user signals like acceptance, edits, and thumbs. Build a small curated set of representative and adversarial cases so the score is stable release over release. The goal is a repeatable, defensible signal, not a single accuracy number.

**Q26. What is the difference between offline and online evaluation, and why do you need both?**
Offline evaluation runs a fixed, labeled dataset in a controlled setting, so it is repeatable, cheap, and can gate releases before anything reaches users. Online evaluation measures the live system on real traffic through A/B tests, production metrics, and sampled human review, capturing distribution shift and real user behavior the offline set cannot foresee. Offline gives you a fast, safe regression gate; online gives you ground truth about real impact and drift. Relying on only one leaves you either blind to real-world failure or unable to catch regressions before launch.

## 4. Cost, latency, and unit economics

**Q27. Walk me through the unit economics of an LLM feature.**
Cost per call is driven by input tokens (the prompt, instructions, retrieved context, and history) plus output tokens, times the model's per-token price, plus retrieval and infrastructure overhead. Multiply per-call cost by expected calls per user per day and by the user base to get the real bill: a feature at 11 cents a call at a million calls a day is a budget conversation, not a rounding error. Levers to pull are a smaller or cheaper model for easy calls, prompt and context trimming, caching stable prefixes and repeated queries, capping output length, and batching. You manage cost as a product constraint that shapes design, not as an afterthought for finance.

**Q28. How does latency change what you build, and how do you manage it?**
Latency shapes both the interaction and the architecture: a p95 of 4 seconds tanks adoption for an interactive feature no matter how good the answer is. You manage it by streaming tokens so perceived latency drops, routing simple requests to faster models, retrieving and reasoning in parallel where possible, caching, and reserving slow reasoning models for tasks that justify the wait. For asynchronous jobs (a nightly report) latency barely matters, so match the model and pattern to the interaction. Always design against the tail (p95 and p99), not the average, because the slow requests are the ones users remember.

**Q29. Accuracy or speed: improve the model by 2 points or ship a month sooner. How do you decide?**
There is no universal answer; tie it to the cost of being wrong in this domain and the value of shipping now. In a low-stakes, reversible feature, ship sooner, learn from real users, and improve in production, because 2 offline points may not move the user outcome. In a high-stakes domain where errors are costly or hard to reverse, the 2 points may be the difference between launchable and not, so you wait. Decide with data: what do those 2 points do to the user-facing failure rate and to the metric that matters.

**Q30. How do you control cost and latency in an agent without wrecking quality?**
Cache the stable prefix of the context, keep the context minimal and well-ordered, and use cheaper or smaller models for sub-steps while reserving the strong model for the hard ones. Cap tool calls and loop iterations so a stuck agent cannot spend unboundedly, cache tool results, and give the model only as much reasoning budget as the step needs. Add early-exit conditions and confidence checks so it stops when it has an answer. Then measure cost and latency per completed task so you optimize the whole trajectory, not one call. See [Production](../../../topics/production.md).

**Q31. When is a bigger or more expensive model the wrong choice?**
When a smaller model already clears the quality bar for the task, the larger one just adds cost and latency with no user-visible gain. Many production tasks (classification, extraction, short structured responses, routing) are handled well by small or mid models, so you reserve the frontier model for genuinely hard reasoning. The disciplined approach is to set the quality bar, find the cheapest model that meets it on your eval set, and route by difficulty. Defaulting to the biggest model everywhere is a common and expensive mistake.

## 5. Responsible AI, safety, and agents

**Q32. Your model performs well for 90 percent of users but poorly for one demographic. What do you do, and on what timeline?**
Treat it as a live product defect, not a research footnote, and quantify the gap and the harm first. If the harm is serious (denying service, unsafe output), gate or roll back the affected path immediately while you fix, rather than shipping a known biased experience. Then diagnose the cause (unrepresentative training or eval data, thresholds, retrieval gaps), fix it, and add a slice of that demographic to the permanent eval set so it cannot regress silently. The timeline is driven by severity: mitigate now, root-cause and durably fix on a committed schedule, and report transparently.

**Q33. How would you design guardrails for an agentic system that can take actions on a user's behalf?**
Constrain what the agent can do before you constrain what it says: scoped, least-privilege permissions so it can only touch what the task requires. Add a human confirmation step for anything irreversible or high-consequence (sending money, deleting data, emailing customers), plus spending and rate limits, and input and output filters for prompt injection, PII leakage, and unsafe actions. Log every action for audit and give someone a kill switch they actually own and have tested. Then red-team the agent against adversarial and injection scenarios before launch and monitor for anomalies after.

**Q34. What is prompt injection, and why does it matter more for agents?**
Prompt injection is when malicious instructions hidden in user input or in content the model reads (a web page, an email, a document) hijack the model into ignoring its real instructions. It matters more for agents because they read untrusted external content and can take actions, so a successful injection can make them exfiltrate data, call tools maliciously, or perform unauthorized actions, not just produce bad text. Defenses include separating trusted instructions from untrusted content, least-privilege tool permissions, output and action filtering, human confirmation for sensitive steps, and red-teaming. There is no complete fix, so you design assuming some injections will get through. See [Securing Agentic AI Systems](../../../resources/securing_agentic_ai_systems.md).

**Q35. How do you build fairness, privacy, and transparency into the spec instead of auditing for them later?**
Write them in as requirements and acceptance criteria from the PRD onward, not as a review at the end. For fairness, define the user segments and the eval slices up front and set thresholds each must meet to launch. For privacy, minimize what data enters the prompt and any vendor, specify retention and redaction, and decide what may be used for training. For transparency, spec what the product tells users about AI involvement, confidence, and sources. Making these launch gates means they shape design rather than becoming expensive retrofits.

**Q36. What is your fallback the first time the model is confidently wrong in front of a customer?**
Assume it will happen and design the unhappy path before launch. Put a confidence gate in front of consequential output, and when confidence is low or a check fails, degrade gracefully: say "I am not sure, here is a human" or a safe default rather than bluffing. Log the incident with the input and output, alert if the rate crosses a threshold, and feed the case into the eval set and a retraining or prompt-fix trigger. Anyone can describe the happy path; the maturity signal is having already built for the wrong one.

**Q37. How do regulations like the EU AI Act affect an AI PM's launch decisions?**
The EU AI Act classifies systems by risk, and high-risk uses (employment, credit, health, and similar) carry obligations: risk management, data governance, human oversight, transparency, logging, and documentation. As a PM you classify your feature early, because a high-risk classification adds required launch gates (documented evals, human-in-the-loop, audit-ready logging, disclosure) and changes the timeline and design. Even outside the EU, this maps well onto good practice frameworks like the NIST AI Risk Management Framework, so treating audit-ready telemetry and human oversight as defaults is both compliant and good product. The practical move is to know your risk tier before you commit a date.

**Q38. How do you reduce hallucinations in a customer-facing product?**
There is no way to eliminate hallucination, so you stack mitigations and design for the residual. Ground answers in retrieval (RAG) so the model answers from trusted sources, require citations, and adopt an abstain-when-unsupported policy so an unsupported claim yields "I do not know" plus a human handoff rather than a confident guess. Use verification steps (re-query sources, self-check, or a second model) for high-stakes claims, and measure the hallucination rate continuously with a gate you will not launch above. Match the strictness to the wrongness cost of the domain.

## 6. Business and strategy

**Q39. Build versus buy for a core AI capability: how do you decide?**
Weigh token cost against control, latency against time to market, the data moat you actually own against the one you wish you had, and the switching risk of betting the product on one vendor's roadmap. Buying an API wins on speed, frontier quality, and lower upfront cost and is right for most features. Building (fine-tuning or self-hosting) wins when you have proprietary data that creates a real advantage, when volume makes per-call price painful, or when you need control the vendor cannot give (latency, privacy, on-prem). Decide per capability, not once for the company, and keep an exit path so one vendor cannot hold the product hostage.

**Q40. How would you think about pricing an AI feature given variable inference costs?**
Unlike traditional software, marginal cost is real and scales with usage, so flat unlimited pricing can lose money on power users. Options include usage-based pricing, tiered plans with caps, or bundling AI into a higher tier, and increasingly outcome-based pricing where you charge for a resolved ticket or completed task rather than per call. Model the cost per unit of value delivered and make sure price sits comfortably above it at the usage levels you expect, with room for cost to fall as models get cheaper. Also watch that pricing does not discourage the very usage that creates value.

**Q41. What is a durable moat for an AI product when everyone can call the same models?**
The model is rarely the moat because competitors can call the same API, so durability comes from what surrounds it: proprietary data and feedback loops, deep workflow integration and distribution, switching costs, and a superior evaluation and quality process that competitors cannot easily copy. A tight loop where usage generates data that improves the product compounds over time. Brand, trust, and being embedded in a user's real workflow also hold. As a PM you invest in the data flywheel, the integration surface, and the eval discipline rather than assuming the model choice is the differentiator.

**Q42. How do you build an AI product roadmap when the underlying models change every few months?**
Anchor the roadmap to durable user problems and outcomes, then treat model capability as a fast-moving input you re-check often. Architect so you can swap models behind an abstraction, so a better or cheaper model is an upgrade rather than a rebuild. Sequence bets by what is reliable now versus what is one capability jump away, and keep near-term commitments concrete while holding a portfolio of options for capabilities that are close but not ready. Build the eval harness first, because it is what lets you adopt a new model safely and quickly when it lands.

**Q43. How would you measure the ROI of an AI feature to justify continued investment?**
Tie it to a business outcome, not to activity: time saved, tickets deflected or resolved, conversion lift, revenue influenced, or cost reduced, measured ideally with a holdout or A/B comparison against the pre-AI baseline. Net out the true cost, including inference, retrieval, human review, and maintenance, so you report value minus cost rather than gross usage. Attribute carefully, because usage alone (prompts sent) is not value. A credible ROI story pairs a specific outcome metric with a controlled comparison and the fully loaded cost.

## 7. Execution and behavioral

**Q44. How do you write a PRD for a feature whose output is probabilistic and occasionally wrong?**
Beyond the normal problem, users, and goals, a probabilistic PRD specifies the quality bar and how it is measured (the eval set and thresholds that gate launch), the expected failure modes, and exactly how the experience degrades around each one (confidence gates, fallbacks, undo, citations, escalation). It states the cost and latency budgets, the responsible-AI requirements as acceptance criteria, and the rollback plan. It treats prompts and the eval set as first-class spec artifacts, because for an AI feature the prompt is part of the product definition. The theme is designing the unhappy path deliberately rather than assuming the model is always right.

**Q45. Tell me about an AI feature you killed or an AI concept you had to defend to a skeptical stakeholder.**
Use a real story in STAR form with numbers. For a kill: name the feature, the metric that told you it was not working (low acceptance, high hallucination or escalation, cost that outran value), the experiment or data behind the call, and what you did with the team and the learning afterward. For defending a concept: describe the stakeholder who wanted more than the model could reliably do, how you translated the limitation into plain business terms, and how you held a responsible line while still finding a path to value. The signal interviewers want is judgment and ownership: you have a graveyard, you decide with data, and you take responsibility for the call.

---

Keep going: verify your weak areas against [resources.md](resources.md), the [topic pages](../../../topics/foundations.md), and the [60 GenAI Interview Questions](../../60_gen_ai_questions.md). Then run the [prep plan](prep-plan.md).
