# Analytics Copilot (Text to SQL): the PM interview

## The interview question

> "Business teams wait on the data team for every number. You are the PM. Should we give them an AI analytics copilot, what does success look like, and how do you keep people from trusting wrong numbers?"

The same scenario as the [engineering version](README.md), worked for an **AI Product Manager** loop. The interviewer here is testing product judgment: whether this should exist, what good looks like, how you de-risk it, and how you make the calls when the model is probabilistic and sometimes returns a wrong number with confidence. The technical design (schema linking, safe read-only execution, evals, the harness) lives in the [engineering writeup](README.md). This page is the reasoning of the PM who owns the outcome.

> **Read this with your coding agent.** This case is public. Point Claude Code, Codex, or any agent at `https://github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/interview_prep/system-design/text-to-sql-analytics` and have it quiz you on the product decisions, or pressure-test your metrics and rollout plan.

**AI product management is product management with a probabilistic core.** You still start from the user and the business problem, you still pick the smallest thing that moves a real metric, and you still design for failure. What changes is that the core component is non-deterministic, so evaluation, trust, and the human fallback move from edge cases to the center of the product. In an analytics copilot the stakes are sharp, because the output is a number a person acts on, so a confident wrong number presented as fact is the failure that can end the project.

## The PM spine: 6 questions for any AI product

This is the product companion to the [5-layer engineering spine](../README.md#the-spine-how-we-think-about-ai-system-design). Where the engineer reasons about the model, the wrapping layer, evals, production, and optimization, the PM reasons about whether the thing should exist and how to ship it without breaking trust. Bring these 6 questions to any AI product.

- **1 Problem and user.** Whose pain, what outcome, and is it worth doing at all.
- **2 Fit.** Is AI the right tool, is it good enough yet, and do you build or buy.
- **3 Success metrics.** What you will move, and the number that must not break.
- **4 Experience.** The probabilistic UX: verification, transparency, trust, and the human handoff.
- **5 Evaluation and risk.** How you know it is ready, the guardrails, and the one failure that ends the project.
- **6 Rollout.** Shadow, pilot, staged go and no-go gates, and when to pull back.

Product management is composing these into a decision you can defend to a business user, an engineer, and a CFO in the same meeting.

## The answer

### 1 Problem and user

Start from the problem, and reach for the technology second. This is [Problem-First](https://maven.com/aishwarya-kiriti/genai-system-design), and it is the part a weak PM skips. It is also plain product discipline: classic product discovery de-risks value and viability before a team commits to building, which is the [four big product risks](https://www.svpg.com/four-big-risks/) frame applied to an AI feature.

> **What is Problem-First?** A framework we developed and teach at [LevelUp Labs](https://levelup-labs.ai): work backwards from the business problem and the user's pain, scope hard, and only then reach for AI, choosing the smallest intervention that moves a measurable outcome. It is the spine of our [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design).

Frame it in numbers a business owns. Two users have pain, and they are on opposite sides of the same queue. A business user in operations, finance, or growth has a question the warehouse can already answer, and today they file a ticket and wait: a recurring ad-hoc question takes an analyst on the order of minutes to write and a business user hours or days to get back (treat these as illustrative until you measure your own baseline). On the other side, a data analyst spends a large share of the week on repetitive pulls instead of the modeling only they can do. The pain is access to a number that already exists, above analysis.

Write the outcome before the system: let a business user get a correct, trustworthy number to a well-formed question without writing SQL, so time-to-answer drops from days to seconds and analysts get their week back for deep work. Narrow the intervention until it hurts: read-only questions over a known schema, the query shown for inspection, and an abstention whenever the question is ambiguous or cannot be validated. A copilot that models new data, defines new metrics, or writes back to the warehouse in version 1 is the over-scoping trap.

The clarifying questions you need to ask a stakeholder: what is the cost of a wrong number versus an honest abstention; which questions are genuinely self-serve versus the ones that still need an analyst; who is allowed to see which rows; how are core metrics like "revenue" and "active customer" defined, and by whom; what is the current time-to-answer baseline; and how fresh must the data be.

### 2 Fit: is AI the right tool, and is it good enough yet

Capability judgment is the AI PM's signature skill: deciding whether a model belongs anywhere near the problem, and where it earns autonomy. Today's models write reliable SQL for well-scoped, grounded questions over a curated schema, and they earn a user's trust when the copilot shows the query and abstains on anything it cannot validate. They are far less reliable on very large schemas, on questions that hinge on a business definition the columns do not carry, and on unbounded analysis, which is why the design keeps the copilot read-only and hands ambiguous questions back to an analyst. Match the ambition to what the model does reliably, gate the consequential work behind human oversight, and let evaluation tell you where that line sits for your warehouse rather than guessing. This is the through-line of both major agent guides: hand the model the work it handles well, and put a human on the path where a mistake is expensive ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents), [OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)). To pick the first slice, favor a high-volume set of questions with clear, agreed metric definitions and a low cost of being wrong, which is how enterprise teams that scale AI tend to sequence their use cases ([OpenAI, Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf)).

> **Real outlier: correctness on enterprise schemas is where the work is.** On [Spider 2.0](https://spider2-sql.github.io/), the benchmark of text to SQL over real enterprise warehouses, purpose-built agent systems now reach roughly 74% on the multi-dialect Lite split and above 90% on the Snowflake split as of 2026, while a frontier model behind a generic agent scaffold lands near 42% and 26% on the same tasks. Writing straightforward SQL is largely handled; the value comes from the engineering that grounds a question in your schema and business definitions and verifies the number before a person sees it. Correctness stays at the center of this product, because the copilot is useful only to the degree a person can trust the number, and even the benchmarks that grade this carry high annotation-error rates, so you measure it relentlessly and verify what you return. [[Spider 2.0 leaderboard, 2026](https://spider2-sql.github.io/)] [[benchmark annotation errors, 2026](https://arxiv.org/abs/2601.08778)]

Build or buy is a real fork. Off-the-shelf analytics copilots (for example [Copilot in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/fundamentals/copilot-fabric-overview), Snowflake Cortex Analyst, Databricks Genie) get you live faster and carry their own guardrails. Building in-house gives you control over behavior, over how the copilot resolves your business definitions, and over cost, and it compounds when trustworthy self-serve analytics is core to how the company runs. Decide on a few axes: how governed and company-specific your metric definitions are, how sensitive your data and row-level access rules are, token cost against control, time to market against the trust you build by owning the answer, vendor lock-in, and whether you have the team to own an evaluation and monitoring loop for the life of the product. One thing stays yours either way: the semantic layer that says what "revenue" and "last quarter" mean is your governance asset, and grounding any copilot in it lifts correctness more than a bigger model does ([MotherDuck](https://motherduck.com/blog/bird-bench-and-data-models/), [dbt MetricFlow](https://docs.getdbt.com/docs/build/about-metricflow)). A common answer is to buy to learn fast, then build the parts that become differentiating, which mirrors how enterprises that scale AI describe the path ([OpenAI, AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf)).

### 3 Success metrics

Pick the minimum set that tells you the truth, and name the number that must not break.

- **Time-to-answer.** How long a business user waits for a number, from days in the queue to seconds. The core value the copilot exists to deliver.
- **Self-serve resolution rate.** The share of questions answered without an analyst touching them. The core adoption outcome.
- **False-answer rate.** The hard ceiling. A confident wrong number acted on by a business user is the number that must stay near zero, and it can sink the project on its own.
- **Trust and repeat use.** Do users act on the number and come back, measured by thumbs-up on answers and returning users. Deflection means nothing if people quietly stop believing the output.
- **Query cost and warehouse load.** The unit economics finance and data engineering will ask about.
- **Abstention rate.** Too high means the copilot is not helping; too low means it is answering questions it should have handed off, which risks confident wrong numbers.

Separate leading signals (schema-linking recall, execution accuracy on a gold set) from lagging outcomes (repeat use, analyst time recovered). Pair the business outcome (time-to-answer, self-serve resolution) with a quality metric beneath it (false-answer rate, execution accuracy) and treat a gap between the two as the alarm, which is the reliability-plus-value framing for agent metrics ([Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents)). Report reliability across repeated tries (pass^k, success on all k independent attempts) alongside the average, because a copilot that gets a number right 3 times in 4 is one a finance team learns to distrust. The anti-metric to watch is raw query volume: a copilot that answers every question confidently, including the ones it should have refused, looks busy on a dashboard while it quietly erodes trust in the numbers.

Frame correctness as necessary and not sufficient. A query can return the right rows and still answer a different question than the user meant, because "top customers" can mean revenue or order count and "last quarter" depends on the fiscal calendar. The number has to be correct and it has to answer the real question, so the experience in section 4 is built to surface the interpretation, and the eval in section 5 checks that the answer addresses the question rather than only that the SQL runs.

### 4 The experience: designing for a model that is sometimes wrong

The product is the experience you build around a probabilistic core. Design it so a wrong number is rare, visible, and cheap to catch before anyone acts on it. This has an established design playbook: help the user form an accurate picture of what the copilot can do, make its uncertainty and its assumptions legible, and give a fast path to correct it or reach an analyst ([Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/), [Microsoft Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/)).

- **Show the work.** Every answer shows the generated SQL and a plain-language reading of it, so a business user and an analyst can both verify the number instead of trusting it blind. Transparency is what turns a black-box guess into a checkable answer.
- **State the interpretation.** The copilot says the assumptions it made ("revenue is net of refunds, last quarter is fiscal Q2") alongside the number, so a wrong assumption is visible and cheap to fix rather than silently baked into a decision ([mental models, PAIR](https://pair.withgoogle.com/chapter/mental-models/)).
- **Gate on confidence and hand off gracefully.** When the question is ambiguous or the query cannot be validated, the copilot asks one clarifying question or hands the analyst the failed query with full context, rather than guessing a number ([designing for AI failure, PAIR](https://pair.withgoogle.com/chapter/errors-failing/)). The promise that a person can always reach an analyst is a product decision made up front rather than a fallback bolted on later.
- **Scope who may self-serve.** A business user sees only the rows they are entitled to, so a regional manager gets their region and finance gets the whole picture, and the copilot sets that expectation plainly instead of appearing to answer a question it is not allowed to.
- **Make correction cheap.** Let the user rephrase, correct the interpretation, or ask an analyst in one step, and route that feedback back into evaluation ([feedback and control, PAIR](https://pair.withgoogle.com/chapter/feedback-controls/)).
- **Latency and tone.** A few seconds is fine for an interactive question, and the copilot should present a number with its caveats rather than false certainty.

### 5 Evaluation and risk

You cannot ship what you cannot measure, and for an AI product the evaluation plan is the launch plan. Build a labeled set of real questions with a validated gold query for each, spanning easy aggregations, medium filters and groupings, hard multi-join analytics, ambiguous questions, and out-of-scope and adversarial ones, and treat it as the release gate: a change ships only if execution accuracy against the gold queries and the false-answer ceiling hold. Report pass^k alongside the average so a copilot that is right on average but flaky under repetition never clears the bar. This mirrors what we teach in the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course, and it is what separates a demo from a product.

Name the one failure that ends the project, and design against it first. Here it is a confident wrong number presented as fact that a business user acts on, for example a revenue figure that lands in a board deck or drives a budget call and turns out to be built on the wrong join or the wrong definition of "revenue." Everything in the experience layer exists to cap that failure: showing the query and the interpretation makes a wrong number catchable, abstention makes an unanswerable question a handoff, and self-consistency on high-stakes questions (sampling several queries and keeping the answer their results agree on) raises confidence where it matters. The guardrails are product-safety features rather than engineering details: a read-only guardrail that allows a single SELECT and blocks anything destructive or out-of-scope, row-level access on every query, and abstention as the default all cap the worst case at a needless handoff rather than a wrong number or a query that should never have run.

### 6 Rollout

Ship it the way you de-risk any high-stakes launch, in stages with gates.

- **Shadow first.** Run the copilot alongside analysts on real questions without showing users its answers, and compare against both the eval bar and the analyst's number.
- **Pilot a slice.** Turn it on for one team with well-defined metrics (finance ops is a good first slice), read every transcript, and watch the metrics from section 3.
- **Stage the rollout** behind go and no-go gates on time-to-answer, self-serve resolution, and the false-answer ceiling, with a rollback plan you can trigger in minutes.
- **Monitor and keep the analyst path.** Sample live traffic, watch for schema drift as the warehouse changes under you, and guarantee a user can always reach an analyst.

Pull back the moment the false-answer rate crosses its ceiling or a wrong number drives a bad decision, and diagnose before you re-expand. The audit trail (every answer carries the query that produced it) is what makes a rollback surgical rather than a full stop: you can find the exact questions that failed, add them to the eval set, and re-open the gate once they pass.

## Follow-ups an interviewer asks

**"How do you measure success, and how do you avoid vanity metrics?"** Lead with time-to-answer and self-serve resolution paired with trust and repeat use, gate on the false-answer ceiling, and refuse to celebrate raw query volume, which rewards a copilot for answering questions it should have refused. Report reliability with pass^k, because a copilot that is right most of the time can still be wrong often enough that a finance team stops trusting the numbers.

**"The copilot gives finance a wrong number that lands in a board deck. Walk me through your response."** Contain the impact first: correct the number and notify everyone who saw it. Then trace the failure to the step that broke, which the audit trail makes possible because every answer carries its query: was it schema linking picking the wrong table, generation writing a bad join, or the semantic layer defining "revenue" the wrong way. Add the case to the eval set with a gold query so it cannot regress, and decide whether it was a one-off or a pattern that should pause the rollout. Showing the query and keeping an analyst path are what make this recoverable.

**"Engineering says 6 months. Scope an MVP."** Cut to one team, a dozen well-defined metrics, and read-only questions with the query shown and generous abstention, no ambiguous or exploratory questions. Prove time-to-answer and the false-answer ceiling there before you widen the schema or take on harder questions. The follow-ups add capability the way versions would, without betting the launch on the hardest slice.

**"Who should be allowed to self-serve, and how do you keep a regional manager from seeing every region's numbers?"** Row-level access is a load-bearing wall here rather than a framework default. Run every query under the asking user's permissions, so the copilot passes through the warehouse's row-level security or injects a mandatory filter it cannot be talked out of, and log who asked what and which rows came back. Start the pilot with a trusted group and widen access as the guardrail proves out on real questions.

**"Build or buy?"** Decide on how governed and company-specific your metric definitions are, how sensitive your data and row-level rules are, and whether you can own an evaluation loop for the product's life. Buy an analytics copilot to learn fast, and keep the semantic layer and the eval loop as yours, because that governance is what makes an answer trustworthy and it is the part that becomes your advantage.

**"Leadership frames this as replacing the data team. How do you set the goal?"** Frame it as capacity and speed: self-serve the repetitive questions so analysts spend their week on modeling and the hard analysis only they can do, with the false-answer ceiling and a guaranteed analyst path as guardrails. A copilot that displaces analysts while quietly shipping wrong numbers is the cautionary version of this launch.

## Further reading

Start inside this repo, then branch to the primary sources behind the product decisions above.

- **LevelUp Labs (Aishwarya and Kiriti's work):** the [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) folder (rounds, question bank, prep plan), the [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) course for the evaluation and metrics round, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) and [Advanced AI Evals course](https://maven.com/aishwarya-kiriti/evals-problem-first) for the Problem-First method and eval depth.
- **Designing a probabilistic UX:** the [Google PAIR People + AI Guidebook](https://pair.withgoogle.com/guidebook/) (mental models, feedback and control, and [designing for AI failure](https://pair.withgoogle.com/chapter/errors-failing/)) and Microsoft's [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/haxtoolkit/ai-guidelines/).
- **Capability judgment and agent design:** Anthropic, [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents), and OpenAI, [A Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
- **Enterprise adoption and build-vs-buy:** OpenAI, [Identifying and Scaling AI Use Cases](https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf) and [AI in the Enterprise](https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf), and [Copilot in Microsoft Fabric](https://learn.microsoft.com/en-us/fabric/fundamentals/copilot-fabric-overview) as the natural-language-over-the-warehouse baseline every BI platform now ships.
- **Correctness and the semantic layer:** [Spider 2.0](https://arxiv.org/abs/2411.07763) and its [live leaderboard](https://spider2-sql.github.io/) for how hard enterprise correctness is and how far engineered agents have pushed it, a [2026 audit of benchmark annotation errors](https://arxiv.org/abs/2601.08778) for why you verify the number, and [MotherDuck](https://motherduck.com/blog/bird-bench-and-data-models/) and [dbt MetricFlow](https://docs.getdbt.com/docs/build/about-metricflow) on why a governed metric layer lifts correctness.
- **Metrics and product discovery:** Google Cloud, [Measuring the value of AI agents](https://cloud.google.com/blog/products/ai-machine-learning/measuring-the-value-of-ai-agents), and Silicon Valley Product Group on [the four big product risks](https://www.svpg.com/four-big-risks/).

## Related in this repo

- [Engineering version of this case](README.md) for the technical design (schema linking, safe read-only execution, evals, the harness).
- [AI Product Manager interview prep](../../roles/ai-product-manager/README.md) for the full role loop.
- [Evaluation topic](../../../topics/evaluation.md) and [AI Evals for Everyone](../../../free_courses/ai_evals_for_everyone/README.md) for the metrics and eval plan, and the [AI System Design course](https://maven.com/aishwarya-kiriti/genai-system-design) for the Problem-First method.
