# Question Bank

40 questions with concise model answers, grouped by theme. Answers are anchors, not scripts: know the mechanism and the trade-off so you can defend a follow-up. Pair this with the repo's [60 GenAI interview questions](../../60_gen_ai_questions.md) and [role-based prep](../../role_based_prep.md).

Themes: [A. AI fundamentals and fluency](#a-ai-fundamentals-and-fluency) - [B. RAG, fine-tuning, and knowledge](#b-rag-fine-tuning-and-knowledge) - [C. Agents, reasoning, MCP, context](#c-agents-reasoning-mcp-and-context) - [D. Evaluation and quality](#d-evaluation-and-quality) - [E. Cost, latency, and production](#e-cost-latency-and-production) - [F. Business, ROI, prioritization, build-vs-buy](#f-business-roi-prioritization-and-build-vs-buy) - [G. Risk, responsible AI, governance](#g-risk-responsible-ai-and-governance) - [H. Change management and org](#h-change-management-and-org)

---

## A. AI fundamentals and fluency

**1. Explain how a large language model works to a non-technical executive.**
A large language model is a system trained on a very large amount of text to predict the next piece of a sentence, and by doing that at scale it learns patterns of language, facts, and reasoning steps. You give it instructions and context in plain language, and it produces a response. The two things an executive must internalize: it is probabilistic, so it can be confidently wrong (hallucinate), and it only knows what was in its training data plus what you put in front of it right now. Everything in an AI strategy flows from managing those two facts.

**2. What is the difference between generative AI and the traditional automation companies already have?**
Traditional automation follows fixed rules a human wrote, so it is predictable and brittle: it does exactly what it was told and breaks on anything new. Generative AI produces new outputs (text, code, images) from patterns and handles ambiguity and language, which is why it can touch knowledge work that rules could never automate. The trade-off is that it is non-deterministic and needs evaluation and guardrails rather than a spec. Strategically, use rules where the process is well-defined and stakes are high, and use generative AI where the work is language-heavy, varied, and tolerant of review.

**3. What can LLMs reliably do today, and what should organizations still not trust them with?**
They are reliable for drafting, summarizing, classifying, extracting structured data, translating, answering questions over provided documents, and writing and explaining code, all with a human reviewing high-stakes output. They are not reliable as a source of truth from memory, for exact arithmetic or fresh facts without tools, or for fully autonomous high-consequence decisions with no oversight. The pattern that works: use the model for the language-heavy first draft or the triage, and keep a human or a deterministic check on the final commit. A strategist earns trust by being precise about that line.

**4. How do you tell a good AI use case from a bad one?**
A good use case has a clear business metric it moves, tolerates occasional error or has a cheap review step, has the data available, and would be too expensive or slow to solve with rules. A bad one needs perfect accuracy with no human check, depends on data you do not have or cannot use, or automates something that was never a real bottleneck. Favor high-volume, language-heavy, review-friendly workflows for the first wins. The fastest disqualifier is the absence of a metric: if no one can say what number should move, it is a demo, not a use case.

**5. What is a token, a context window, and why do they matter to strategy?**
A token is a chunk of text (roughly three-quarters of a word) that the model reads and generates, and you pay per token in and out. The context window is how many tokens the model can consider at once, which bounds how much document, history, and instruction you can supply in a single call. They matter because cost, latency, and how much knowledge you can stuff into a prompt all scale with tokens. A strategist uses this to sanity-check vendor claims and to understand why long-document workflows cost more and run slower.

**6. What changed between the 2023 wave of generative AI and where we are in 2025-2026?**
Context windows grew large, costs per token fell sharply, and reasoning models arrived that spend extra compute to think before answering, which made multi-step math, code, and analysis genuinely usable. Agents matured from demos toward production, tool and data connection got standardized through protocols like MCP, and evaluation and governance became mainstream concerns rather than afterthoughts. Capability is no longer the main blocker for most enterprise use cases; adoption, data, and change management are. A strategist who is still pitching 2023-era assumptions will misjudge both what is now feasible and where the real risk sits.

**7. What is prompt engineering, and how much does it still matter?**
Prompt engineering is writing the instruction and examples that steer the model toward the output you want, including role, task, constraints, format, and few-shot examples. It still matters and is often the cheapest lever, but the field has moved toward context engineering: deciding what information (retrieved documents, tool results, memory) actually enters the window, not just how you phrase the ask. For strategy, the takeaway is that large quality gains are often available before any model change or fine-tune, which lowers cost and time to value. Treat prompt and context work as the first thing to exhaust, not the last.

**8. A vendor claims 99% accuracy. What do you ask?**
Accuracy on what task, measured against what ground truth, on whose data, and in a sandbox or in production. Ask for a named production deployment with a real use case and a measured outcome, because a demo number is not a production number. Ask what happens on the other 1%, whether errors are caught, and what the cost and latency are at your volume. If they cannot name a real customer, a real metric, and a real failure mode, treat the number as marketing.

---

## B. RAG, fine-tuning, and knowledge

**9. When would you use RAG instead of fine-tuning to give a model new knowledge?**
Use retrieval-augmented generation when the knowledge changes often, is large or proprietary, or needs citations, because retrieval keeps answers current and auditable without retraining. Use fine-tuning to change behavior, format, tone, or to bake in a stable skill, not to inject volatile facts. In practice you often do both: RAG for the knowledge, light fine-tuning for the behavior. For most enterprise document-Q&A problems, RAG is the right first answer.

**10. Walk me through how a RAG system actually works.**
Documents are split into chunks and converted to embeddings (numeric vectors capturing meaning) stored in a vector index. At query time the user's question is embedded, the most similar chunks are retrieved, and those chunks are inserted into the prompt so the model answers grounded in them, ideally with citations. Quality depends heavily on the unglamorous parts: chunking, retrieval quality, and reranking, not just the model. See the repo's [RAG topic](../../../topics/rag.md) and [Agentic RAG 101](../../../resources/agentic_rag_101.md) for depth.

**11. Why do RAG systems fail in production, and how do you de-risk them?**
The common failures are retrieval misses (the answer-bearing chunk is never fetched), stale or messy source data, poor chunking that splits the answer, and the model ignoring or over-trusting the retrieved context. You de-risk by evaluating retrieval and generation separately, cleaning and maintaining the source corpus, adding reranking, and measuring faithfulness so answers stay grounded. Most RAG quality problems are data and retrieval problems, not model problems. A strategist who knows this will fund data readiness instead of a bigger model.

**12. When is fine-tuning genuinely the right call, and what does it cost?**
Fine-tuning is right when you need a consistent behavior, style, or format, when you have enough high-quality labeled examples, or when a smaller fine-tuned model can match a larger one more cheaply at high volume. The costs are data preparation, training and evaluation cycles, and the ongoing burden of re-tuning as needs change, plus the risk of the model forgetting general ability. It is the wrong tool for fresh or frequently changing facts, where RAG wins. Parameter-efficient methods like LoRA make it far cheaper than full fine-tuning and are the sensible default. See [Fine-tuning 101](../../../resources/fine_tuning_101.md).

**13. What is the practical decision order for adding capability: prompt, RAG, fine-tune, or agent?**
Start with prompting and context engineering because it is the cheapest and fastest, and it often clears the bar. Add RAG when the model needs knowledge it does not have and that knowledge changes or needs citations. Fine-tune when you need a stable behavior or a cheaper high-volume model and have the data. Reach for an agent only when the task genuinely needs multiple steps, tools, and adaptation, because each step up adds cost, latency, and new failure modes. Climbing the ladder without exhausting the lower rungs is a common way budgets get wasted.

---

## C. Agents, reasoning, MCP, and context

**14. What makes something an agent rather than a single LLM call, and when do you actually need one?**
An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or fresh data, or adaptation to intermediate results. Avoid it when a single call or a fixed workflow will do, because agents add cost, latency, and new failure modes like getting stuck in loops or taking wrong actions. The mature stance in 2025-2026 is that most value still comes from well-engineered workflows with narrow agentic steps, not fully autonomous agents. See [Agents topic](../../../topics/agents.md) and [Agents 101](../../../resources/agents_101_guide.md).

**15. What are the main failure modes of agents, and how do you contain them?**
Agents fail by hallucinating a tool call, looping without progress, taking a wrong or irreversible action, compounding small errors over many steps, and running up cost and latency. You contain them by capping steps and tool calls, keeping a human in the loop for consequential actions, scoping tools tightly with permissions, adding checks between steps, and evaluating the whole trajectory, not just the final answer. The strategy lesson is that autonomy and control trade off directly: more autonomy means more value potential and more risk. Start narrow and expand autonomy only as evals and trust grow.

**16. What is the Model Context Protocol (MCP) and why should an enterprise care?**
MCP is an open standard for connecting AI models and agents to tools and data sources through a common interface. It matters because you wire a capability or a data source once and reuse it across many agents and applications, instead of building a custom integration for every tool and every vendor. For an enterprise this reduces integration cost and lock-in and makes an internal AI platform far more composable. Strategically it is part of why 2026 agent projects are cheaper to stand up than 2024 ones.

**17. What is context engineering, and why can it matter more than model choice?**
Context engineering is deciding what actually enters the model's context window: instructions, retrieved documents, tool results, and memory, and just as importantly what stays out. It matters because models degrade when the context is bloated, noisy, or poorly ordered, so getting the right, minimal, well-structured information in usually beats both clever phrasing and a bigger model. It is also a cost and latency lever, since every token in the window is paid for and slows the response. For strategy, it means a lot of quality is available through information design before you spend on models or fine-tuning.

**18. How do reasoning models differ from standard LLMs, and what do they change for strategy?**
Reasoning models are trained to spend extra compute on an internal chain of thought before answering, often using reinforcement learning on verifiable rewards. They trade higher latency and cost for much stronger performance on math, code, planning, and multi-step analysis. Strategically they expand the set of problems now worth attempting (complex analysis, agentic planning) but they are slower and pricier, so you route to them only for the hard steps and use cheaper models for the rest. Knowing when a problem needs reasoning versus a fast cheap model is a core cost-control decision.

**19. A team wants to build an autonomous multi-agent system. How do you respond?**
First ask what business problem it solves and whether a single agent or a plain workflow would do, because multi-agent systems multiply cost, latency, and coordination failure. Multi-agent designs earn their keep for genuinely parallel or specialized sub-tasks, but they are harder to evaluate, debug, and govern, and many teams reach for them prematurely. Push for the thinnest version that delivers value: often one well-scoped agent with good tools and a human check. Fund the ambitious version only after evals prove the simple version works and the added complexity is justified.

**20. How would you keep an agent's cost and latency under control?**
Cache the stable prefix of the context, keep the context minimal and well-ordered, and route sub-steps to cheaper or smaller models, reserving reasoning models for the hard parts. Cap tool calls and loop iterations, cache tool results, and give the model only as much reasoning budget as the step needs. Batch or stream where the user experience allows, and monitor cost and latency per request as first-class metrics. These are the levers that decide whether an agent is economically viable at scale.

**21. What is retrieval versus tool use versus memory in an agent, in plain terms?**
Retrieval pulls relevant documents into the prompt so the model can ground its answer in current, specific knowledge. Tool use lets the model call external functions (search, a database, a calculator, an API) to fetch data or take actions it cannot do from memory. Memory lets the agent carry state across steps or sessions so it does not repeat work or lose context. A capable agent combines all three, and most real-world reliability comes from getting these plumbing pieces right rather than from the model itself.

---

## D. Evaluation and quality

**22. How would you evaluate whether an AI feature is good enough to ship?**
Define what good means as measurable criteria tied to the business goal, build a representative labeled test set from real cases, and measure task success, error rate, and the cost of the errors that slip through. Use a mix of automated checks, human review on a sample, and an LLM-as-judge with a clear rubric for subjective quality, then set a threshold before you look at results so you are not moving the goalposts. Ship when the system beats the current baseline and the residual errors are affordable and caught. See [Evaluation topic](../../../topics/evaluation.md) and the repo's [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md).

**23. Why do you say evals are non-negotiable, and what happens without them?**
Because generative systems are non-deterministic, evals are the only way to know if a change helped or hurt, whether the system is production-ready, and whether it is degrading over time. Without them, teams ship on vibes, cannot compare vendors or model versions, and discover regressions only when users complain. Evals also convert a fuzzy quality debate into a number executives can fund and track. A strategy that funds a build with no eval plan is funding a system no one can prove is working.

**24. How do you evaluate a RAG system specifically?**
Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually fetched (recall and precision at k). For generation, measure faithfulness (is the answer grounded in the retrieved context with no fabrication), answer relevance, and correctness, using a labeled question set and an LLM-as-judge with a clear rubric. Then monitor the same signals in production, because source data drifts and yesterday's good retrieval decays. See the repo's [RAG research table](../../../research_updates/rag_research_table.md) and [AI evaluation 2025 table](../../../research_updates/ai_evaluation_2025_table.md).

**25. What is LLM-as-judge, and what are its limits?**
LLM-as-judge uses a strong model to score outputs against a rubric, which scales evaluation of subjective qualities like helpfulness or faithfulness far more cheaply than human raters. Its limits: it can be biased toward verbose or confident answers, inconsistent without a tight rubric, and it can share blind spots with the model being judged. You control for this by writing clear rubrics, calibrating the judge against human labels on a sample, and keeping humans in the loop for high-stakes calls. It is a force multiplier for evaluation, not a replacement for human judgment on what matters.

**26. How do you monitor an AI system after launch?**
Track quality signals continuously (task success, faithfulness, user feedback, escalation and override rates), plus operational metrics (latency, cost per request, error and timeout rates) and safety signals (harmful or off-policy outputs). Watch for drift as inputs, data, and user behavior change, and set alerts on the metrics tied to the business case. Close the loop by feeding failures back into the eval set and the roadmap. Re-score each funded use case against its original business case on a cadence (for example at month 6, 12, and 24) so value is proven, not assumed.

---

## E. Cost, latency, and production

**27. How do you think about the total cost of an AI system, beyond model API fees?**
Model calls are often the smallest line. Total cost includes data preparation and pipelines, retrieval and vector infrastructure, integration and engineering, evaluation and monitoring, human review for high-stakes output, security and compliance work, and ongoing maintenance as models and needs change. The invisible work of enablement, documentation, support, and adoption exists whether you build or buy; the question is who carries it. A credible business case prices all of this, and a common failure is modeling only the token cost.

**28. What drives latency in an LLM application, and why does it matter commercially?**
Latency is driven by model size and whether it is a reasoning model, the number of tokens generated, context length, the number of sequential tool or retrieval calls, and any human-in-the-loop step. It matters because slow responses kill adoption in interactive workflows and can break real-time use cases entirely. You manage it by routing to smaller or faster models where quality allows, streaming output, parallelizing independent calls, caching, and trimming context. For strategy, latency is a product constraint that can make an otherwise valuable use case unviable.

**29. How do you decide which model to use for a given use case?**
Match the model to the task, not to the leaderboard: use a fast, cheap model for simple classification or drafting, a mid-tier model for most knowledge work, and a reasoning model only for genuinely hard multi-step problems. Weigh accuracy on your own eval set against cost, latency, context window, data-residency and privacy needs, and whether you can self-host. Many production systems route different steps to different models to balance quality and cost. Avoid standardizing on one frontier model for everything, because it overpays for the easy 80% of calls.

**30. What does it take to move an AI pilot into production, and why do so many die in between?**
Getting to production requires reliable data pipelines, evals and monitoring, security and compliance sign-off, integration into existing systems and workflows, and the change management to get people to actually use it. Roughly 80 to 95 percent of enterprise AI pilots stall, usually because they were run as technology experiments rather than operating-model changes: no baseline, no adoption plan, governance never built, and sponsorship that vanished after the demo. The fix is to design the pilot with the production path, owner, and success metric defined up front. A strategist's core value is refusing to fund pilots that have no route out of the sandbox.

---

## F. Business, ROI, prioritization, and build-vs-buy

**31. How do you prioritize AI use cases across a company?**
Score every candidate use case on a consistent rubric: business impact, technical feasibility, data readiness, time to production, and risk and compliance. That yields a ranked backlog, and you pick a small set (3 to 5) tied to real P&L impact rather than spreading thin, because leading enterprises fund fewer initiatives and generate more return. Sequence for early proof: a quick win to build credibility and data, then a platform investment, then a transformational bet. Make the scoring explicit so the choices are defensible to a skeptical executive.

**32. Walk me through building an ROI model for an AI initiative.**
Start by establishing the baseline: the current cost, time, error rate, or revenue of the process today, because without a baseline you cannot claim a lift. Estimate the expected improvement (for example 30 to 50 percent time reduction, higher conversion, fewer errors), convert it to dollars, then subtract the full cost to build and run (data, engineering, infrastructure, review, change management). Compute payback period and be explicit about assumptions and their range, since false precision destroys credibility. Then commit to re-measuring against the same baseline on a cadence rather than declaring victory at launch.

**33. How do you approach build vs buy vs partner for an AI capability?**
Buy for commodity capability where speed matters and there is no differentiation, because vendors have already solved it and you avoid carrying the maintenance. Build when the capability is a genuine differentiator, you have proprietary data and engineering depth, and control or data residency demands it. Partner to share risk on the hard middle where you need custom work but lack the full team. The decision turns on latency and compliance needs, cost model, and in-house depth; buy-plus-partner approaches tend to reach production more reliably than fully internal builds. Beware advisors who recommend the same stack to everyone, which signals a product sale rather than a strategy.

**34. A CEO says a competitor launched an AI agent and wants one in 90 days. How do you respond?**
Acknowledge the pressure and redirect it to the business goal: what outcome does the competitor's move threaten, and what would actually protect or grow the business. Warn against building the flashy thing that fails in pilot; propose a 90-day plan that delivers a real, narrow win with a measured baseline instead of a demo. Frame it as moving faster in a way that will still be standing in six months, and show the roadmap from quick win to the more ambitious capability. This turns a reactive vanity request into a sequenced, defensible plan.

**35. How do you build an AI roadmap for an organization just starting out?**
First assess readiness: data availability and quality, existing tech and talent, governance maturity, and executive alignment. Then map and score use cases, pick a small first wave weighted toward quick, low-risk wins that build capability and credibility, and invest in the shared foundations (data, platform, governance, skills) that later use cases will reuse. Sequence in waves: prove value, build the platform, then pursue transformational bets. Attach owners, metrics, and a change-management budget to each phase, because a roadmap with no adoption plan is a wish list. Ground this in the repo's [GenAI roadmap](../../../resources/genai_roadmap.md) and [State of AI 2025 report](../../../research_updates/state_of_ai_2025_report/README.md).

**36. How do you measure whether an AI strategy is actually working at the portfolio level?**
Track leading indicators (use cases in production, adoption and active usage, time from idea to production) and lagging indicators (aggregate cost saved or revenue gained against baselines, and return versus spend). Watch the conversion rate from pilot to production, because a low rate signals a prioritization or adoption problem, not a technology one. Maintain a quarterly re-measurement cadence where each funded use case is re-scored against its own business case. The honest headline metric is realized value per dollar invested, not the number of pilots launched.

**37. A client wants to fund 20 AI projects at once. What do you tell them?**
That funding 20 at once is a reliable way to get 20 pilots and zero at scale, because attention, data, engineering, and change-management capacity are finite. The evidence is consistent: organizations that fund fewer, well-chosen initiatives get materially higher return. Recommend concentrating on 3 to 5 use cases with the clearest P&L link and readiness, resourcing them properly through to production, and using the wins to build the platform and appetite for the next wave. Discipline in saying no is a large part of the strategist's value.

**38. How is generative AI changing the consulting and services business itself?**
It compresses the work that used to justify large leverage pyramids: research, first-draft analysis, and deck production are now partly automated, so firms are investing billions in internal AI tools and forward-deployed delivery models. This raises the bar for what a human advisor adds: judgment, client trust, change leadership, and the ability to challenge and integrate AI output rather than produce the first draft. For a candidate, it means firms want strategists who use AI fluently in the work and can be graded on how well they prompt and pressure-test it. Positioning yourself as someone who orchestrates AI plus judgment, rather than someone AI could replace, is the winning stance.

---

## G. Risk, responsible AI, and governance

**39. What are the main risks in an enterprise AI deployment, and how do you govern them?**
The main categories are data risk (leakage, using data you have no right to, privacy), model risk (hallucination, bias, unreliable output), security risk (prompt injection, data exfiltration through tools, insecure integrations), and regulatory and reputational risk. You govern them with a tiered approach: classify each use case by risk, then apply proportionate controls such as human-in-the-loop, evals, monitoring, access scoping, red-teaming, and audit logging. Most organizations anchor this to a recognized framework like the NIST AI Risk Management Framework (govern, map, measure, manage), with the EU AI Act setting legal minimums for higher-risk systems and ISO/IEC 42001 providing a certifiable management system. Governance should be proportionate, so low-risk internal tools are not strangled while high-risk decisions get real oversight. See [Safety and security](../../../topics/safety-security.md) and [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md).

**40. What is prompt injection, and why should a strategist care?**
Prompt injection is when untrusted content (a web page, a document, an email the agent reads) contains hidden instructions that hijack the model into doing something it should not, like leaking data or calling a tool maliciously. It matters because any agent that reads external content and can take actions has a real attack surface, and the risk grows with autonomy and tool access. A strategist cares because it turns a capability discussion into a security and liability discussion: it shapes which tools an agent may touch, whether a human approves consequential actions, and how the system is red-teamed. Ignoring it is how an impressive agent becomes a breach. See [Securing agentic AI systems](../../../resources/securing_agentic_ai_systems.md).

**41. How do you handle the EU AI Act and the current regulatory landscape in a strategy?**
Treat regulation as a design input, not an afterthought: the EU AI Act uses a risk-based regime where higher-risk uses (for example those affecting people's rights, employment, or credit) carry real obligations, with general-purpose model rules already phasing in and enforcement building through 2026 and 2027. Classify each use case by regulatory risk early, because that changes what controls, documentation, and human oversight are required and sometimes whether to proceed at all. Combine the legal minimum from applicable regulation with a management framework (NIST AI RMF, ISO/IEC 42001) so compliance is provable. The strategist's job is to bake this into prioritization so you do not build something you cannot legally deploy.

**42. When should you recommend not using AI, or slowing down?**
When the use case demands accuracy the technology cannot yet deliver and errors are consequential and hard to catch, when you lack the data or the right to use it, when the regulatory or reputational risk outweighs the benefit, or when a simpler non-AI solution solves the problem better. Recommending against a build is a strength: it protects budget and trust and signals judgment rather than hype. Frame the no with a reason and, where possible, an alternative or a condition under which it becomes viable later. Interviewers specifically probe for this because a strategist who never says no is a liability.

---

## H. Change management and org

**43. Why do most AI pilots fail to scale, and what does a strategist do about it?**
They fail because they are run as technology experiments rather than changes to how the organization works: missing data infrastructure, no change management, governance never built, metrics untethered from business outcomes, and executive sponsorship that evaporates after the demo. Only a minority of organizations report investing meaningfully in the change management, training, and incentives that adoption requires. A strategist fixes this by designing each initiative with a named owner, a business metric, a production path, and a change budget from the start, and by protecting executive sponsorship through the messy middle. The lesson is blunt: adoption, not the model, is where value is won or lost.

**44. How do you drive adoption of an AI tool in a resistant organization?**
Start with the workflow and the people, not the technology: understand what the affected employees fear (job loss, more work, being blamed for the model's errors) and design for it. Secure a visible executive sponsor, pick an early group that will benefit and champion it, and provide training, clear guidance on when to trust and when to override the tool, and incentives aligned with using it. Budget real money for enablement (a common rule of thumb is 20 to 30 percent of the AI investment) and measure adoption and override rates, not just accuracy. Communicate honestly about what the tool does and does not do, because overselling it once destroys trust for every future rollout.

---

Next: **[resources.md](resources.md)** and **[courses.md](courses.md)**.
