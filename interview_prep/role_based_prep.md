# Role-Based Interview Questions

Real generative AI interview questions with answers, grouped by the role you're interviewing for. Pair this with the broader [60 GenAI Interview Questions](60_gen_ai_questions.md).

Answers are concise on purpose: enough to anchor your own, not a script to memorize.

---

## 🏗️ AI / LLM Engineer

**1. When would you use RAG instead of fine-tuning to give a model new knowledge?**
Use RAG when the knowledge changes often, is large or proprietary, or needs citations: retrieval keeps answers fresh and auditable without retraining. Use fine-tuning to change behavior, format, or style, or to bake in a stable skill, not for volatile facts. In practice you often do both: RAG for knowledge, light fine-tuning for behavior.

**2. What is context engineering, and why can it matter more than prompt wording?**
Context engineering is deciding what actually enters the model's context window, instructions, retrieved documents, tool results, memory, and what stays out. It matters because models degrade when the context is bloated or noisy, so getting the right, minimal, well-ordered information in usually beats clever phrasing.

**3. How do you evaluate a RAG system?**
Evaluate retrieval and generation separately. For retrieval, measure whether the answer-bearing chunk is actually retrieved (recall and precision at k). For generation, measure faithfulness (is the answer grounded in the retrieved context, with no hallucination), answer relevance, and correctness, using a labeled question set and an LLM-as-judge with a clear rubric, then monitor the same signals in production.

**4. What makes something an agent rather than a single LLM call, and when do you actually need one?**
An agent adds tools, memory, and a loop, so it can take actions and iterate toward a goal instead of answering once. Reach for one when the task needs multiple steps, external actions or data, or adaptation. Avoid it when a single call or a fixed workflow will do, since agents add cost, latency, and new failure modes.

**5. How would you keep an agent's cost and latency under control?**
Cache the stable prefix of the context, keep the context minimal and well-ordered, use cheaper or smaller models for sub-steps, cap tool calls and loop iterations, cache tool results, and give the model only as much reasoning budget as the step needs.

**6. What is the Model Context Protocol (MCP) and why does it matter?**
MCP is an open standard for connecting an agent to tools and data sources. It matters because you wire a capability once and reuse it across agents and harnesses, instead of building a custom integration for every tool and every app.

*Study:* [RAG](../topics/rag.md), [AI Agents](../topics/agents.md), [Prompting and Context](../topics/prompting.md), [Evaluation](../topics/evaluation.md).

---

## 🔧 ML / Fine-tuning Engineer

**1. LoRA versus full fine-tuning: what's the tradeoff?**
Full fine-tuning updates all weights: most expressive, but expensive in compute and memory and easy to overfit or forget. LoRA and other parameter-efficient methods train small adapter matrices while freezing the base model: far cheaper, portable, and usually close to full fine-tuning quality for most adaptation tasks. Default to LoRA unless you have a strong reason and the budget for full tuning.

**2. Walk through SFT, DPO, RLHF, and GRPO. What is each for?**
Supervised fine-tuning (SFT) teaches format and behavior from labeled examples. RLHF optimizes against a learned reward model from human preferences. DPO reaches a similar preference-alignment goal directly from preference pairs, without training a separate reward model, so it is simpler and more stable. GRPO is a reinforcement learning method that works well when you have a verifiable reward (math, code, tool-use), and is central to modern reasoning-model training.

**3. How do you prevent catastrophic forgetting during fine-tuning?**
Use low learning rates and few epochs, prefer parameter-efficient methods that leave the base frozen, mix in some general data alongside your task data, and evaluate on held-out general benchmarks (not just your task) to catch regressions early.

**4. When is fine-tuning the wrong tool?**
When the need is fresh or frequently changing knowledge (use RAG), when you have too little high-quality data, or when prompting plus context already meets the bar. Fine-tuning is for changing behavior and style, not for injecting volatile facts.

**5. How do you evaluate a fine-tuned model?**
Hold out a task-specific eval set and compare against the base model and a strong prompted baseline. Check that you did not regress on general capability, and, for preference-tuned models, evaluate on the behaviors you optimized for plus safety, since preference tuning can introduce new failure modes.

**6. What are the tradeoffs of quantization?**
Quantization shrinks the model (for example to 4-bit) to cut memory and cost, at some accuracy loss. QLoRA lets you fine-tune a quantized base cheaply. For inference, measure the quality drop on your own eval before shipping a heavily quantized model, since the impact is task-dependent.

*Study:* [Fine-tuning and Post-training](../topics/fine-tuning.md), [Foundations](../topics/foundations.md), [Production and LLMOps](../topics/production.md).

---

## 🔬 Applied Scientist / Research Engineer

**1. How do reasoning models differ from standard LLMs?**
Reasoning models are trained to spend test-time compute on an internal chain of thought before answering, often with reinforcement learning on verifiable rewards. They trade latency and cost for much stronger performance on math, code, and multi-step problems, and they change how you prompt: give the goal and constraints, and let the model do the step-by-step work.

**2. How would you design an evaluation for a brand-new capability?**
Start from what success means as a checkable outcome, build a small high-quality labeled set that covers realistic and adversarial cases, choose metrics that isolate the capability (not proxies), and validate your judge against human labels before trusting it at scale. Treat the eval as an artifact you version and defend.

**3. What is reward hacking or eval gaming, and how do you guard against it?**
It is when a model optimizes the measured proxy rather than the real goal, for example exploiting a judge's biases or a benchmark's shortcuts. Guard against it with held-out and adversarial evals, multiple diverse judges or metrics, and by checking whether gains transfer to independent tasks rather than one benchmark.

**4. Why can a model do worse with retrieved context than with none, and what does that tell you?**
Because irrelevant or conflicting retrieved text can distract or mislead the model, and long noisy context degrades reasoning. It tells you retrieval quality and context construction matter as much as the model, and that more context is not automatically better.

**5. Why measure reliability (for example pass^k) for agents, not just average accuracy?**
Because an agent that succeeds on average but fails unpredictably is not shippable. Metrics like pass^k (does it succeed on all k independent attempts) capture consistency, which is what production actually requires from a long, autonomous task.

*Study:* [Understand journey](../journeys/understand.md), the [research tables](../research_updates/), and the [State of AI report](../research_updates/state_of_ai_2025_report/README.md).

---

## 📋 AI Product Manager

**1. How do you decide whether an AI feature should be built with prompting, RAG, fine-tuning, or an agent?**
Match the tool to the need: prompting and context for behavior you can specify, RAG for fresh or proprietary knowledge, fine-tuning for stable behavior or style at scale, and an agent only when the task genuinely needs multiple steps and actions. Start with the simplest option that clears the quality bar, since each step up adds cost and risk.

**2. What metrics would you track for an AI product?**
Beyond engagement, track task success or resolution rate, quality from an eval harness, hallucination or error rate, latency, cost per interaction, and human-escalation or override rate. Tie them to a user outcome, not just model scores.

**3. How do you handle hallucination in a user-facing product?**
Reduce it (grounding via RAG, constrained outputs, better context), detect it (evals and monitoring, confidence and citation checks), and contain it (show sources, allow easy correction, and route uncertain or high-stakes cases to a human). You manage the risk; you do not eliminate it.

**4. How do you communicate model limitations and risk to stakeholders?**
Be concrete about failure modes and their likelihood, frame quality as a distribution rather than a single demo, and tie risk to the specific decision the feature drives. Set expectations with evals and a rollout plan, not adjectives.

**5. How do you reason about the cost, latency, and quality tradeoff?**
Treat it as a dial per use case: a high-stakes step may justify a slower, more expensive reasoning model, while a routine step should use a small fast one. Decide with evals on your own data, and revisit as models get cheaper.

*Study:* [Use journey](../journeys/use.md), [AI Agents](../topics/agents.md), [Evaluation](../topics/evaluation.md), [Safety and Security](../topics/safety-security.md).

---

## 🏛️ AI Solutions Architect / Enterprise AI

**1. Design a customer support agent for an enterprise. Walk me through it.**
Start from requirements: accuracy bar, latency, escalation policy, and cost. Use retrieval over the company's knowledge base for grounding, tools for actions (look up an order, file a ticket), and a clear handoff to humans for low-confidence or high-stakes cases. Wrap it in evaluation and observability, add guardrails on inputs and outputs, and roll out behind a fallback. Name the tradeoffs at each choice.

**2. What are the main security concerns for an agentic system?**
Prompt injection (especially indirect injection from retrieved or tool content), over-broad tool permissions and privilege escalation, data exfiltration, and unsafe actions taken with real-world autonomy. Mitigate with least-privilege tools, input and output guardrails, human approval for sensitive actions, and audit logging.

**3. How do you roll out an AI system safely?**
Gate on an evaluation suite built around how the system breaks, start with a narrow scope and a human in the loop, add monitoring and guardrails, and expand only as the metrics hold. Keep a fast rollback and treat evals as release infrastructure, not a one-time check.

**4. What changes when RAG has to run at enterprise scale?**
Ingestion and freshness pipelines, access control so retrieval respects per-user permissions, cost and latency of retrieval, evaluation across many document types, and observability on retrieval quality. The model is rarely the bottleneck; the data and retrieval layer usually is.

**5. How do you handle data privacy and compliance?**
Keep sensitive data access least-privilege and permission-aware end to end, be deliberate about what leaves your environment and what a vendor may retain, log for auditability, and design retrieval and memory so a user only ever sees what they are allowed to.

*Study:* [Build 301](../journeys/build.md#build-301-production-and-frontier), [Production and LLMOps](../topics/production.md), [Safety and Security](../topics/safety-security.md), and [Securing Agentic AI Systems](../resources/securing_agentic_ai_systems.md).

---

> More question sets and system-design walkthroughs are planned. Contributions of real, high-quality questions and answers are welcome.
