# Part 2: The 4 Types of Agentic Systems (and When to Use What)

Hi there,

In the previous part, we looked at what makes AI agentic — it’s not just about understanding or generating content, it’s about performing actions and handling tasks end-to-end.

But as teams rush to “add agents” to their stack, here’s the catch:  
Not all agents are built the same, and not all problems need highly autonomous systems.

In this lesson, we’ll walk through four types of agentic systems (as discussed yesterday), using a simple but powerful lens:

- How much autonomy does the agent have?  
- How much control does the human or system retain?

This balance impacts how the system behaves, how you evaluate it, and what infrastructure you need to build.

---
<img width="1216" height="413" alt="image" src="https://github.com/user-attachments/assets/58097651-e6d0-4835-9c13-042f647cf437" />


## The Tool-Augmented LLM

At the core of most modern agents is an **LLM (Large Language Model)** acting as the brain of the system.  
Throughout this course, we use the term LLM to refer broadly to generative AI models — not just text-only models.

On its own, it can generate content, but to turn it into an agent, you augment it with:  
- **Tools** → APIs, functions, databases it can call  
- **Planning** → The ability to break a goal into multiple steps  
- **Memory** → So it can track past actions and outcomes  
- **State and Control Logic** → To know what’s done, what failed, and what to do next  

When connected to these components, the LLM becomes more than a chatbot.  
It becomes a goal-driven system that can reason, take action, and adapt.

But depending on how much you trust it to act without supervision, you end up with different types of agents.  
Let’s walk through them, starting from the least autonomous.

---

## 1. Rule-Based Systems/Agents  
**Low Autonomy, Low Control**

These systems don’t use LLMs at all. They’re built with traditional *if-this-then-that* logic. Every decision path is manually scripted. There’s no reasoning or learning. Rule-based agents have existed long before the LLM era.

> Wait, aren’t we talking about AI agents?  
> Yes — but not every problem needs an AI model. Start with the problem, not the AI. If you can solve it without AI, don’t overcomplicate it.

**What problems do they solve?**  
Well-structured, repetitive tasks with fixed inputs and outputs.

**Examples:**  
- Automatically approve reimbursements under a fixed amount  
- Rename files in a folder based on filename patterns  
- Copy data from Excel sheets into form fields  

**Pros:** Fast, auditable, predictable  
**Cons:** Brittle to change, can’t handle ambiguity  
**Best used when:** You know all the conditions ahead of time and there’s no need for flexibility.

---

## 2. Workflow Agents  
**Low Autonomy, High Control**

This is often the first step for enterprises introducing LLMs into their workflows.  
Here, the LLM enhances an existing workflow but doesn’t execute actions independently. A human stays in control.

**What problems do they solve?**  
Repetitive tasks that benefit from natural language understanding, summarization, or generation, but still need human decision-making.

**Examples:**  
- Suggesting first-draft responses in a support tool like Zendesk  
- Generating summaries of meeting transcripts  
- Translating natural language queries into structured search inputs for BI dashboards  

**How the LLM is used:**  
It reads input (text, tickets, documents), understands context, and generates useful content, but doesn’t act on it.  
A human still decides what to do.

**Pros:** Easy to deploy, low risk, quick value  
**Cons:** Can’t execute or plan, limited end-to-end value  
**Best used when:** You want to augment your team’s productivity without giving up oversight.

---

## 3. Semi-Autonomous Agents  
**Moderate to High Autonomy, Moderate Control**

These are true agentic systems. They not only understand tasks but can plan multi-step actions, invoke tools, and complete goals with minimal supervision. However, they often operate with some constraints or monitoring built in.

**What problems do they solve?**  
Multi-step workflows that are well-understood but too tedious or time-consuming for humans.

**Examples:**  
- A lead follow-up agent that drafts, personalizes, and sends emails based on CRM data, while logging results  
- A document automation agent that extracts details from contracts and updates internal systems  
- A research agent that pulls data from multiple sources, compares findings, and sends a structured report  

**How the LLM is used:**  
The LLM plans the steps, calls APIs to fetch or push data, keeps track of progress, and adapts if something goes wrong.  
It often includes fallback paths or checkpoints for human review.

**Pros:** Automates complex workflows, saves time, higher ROI  
**Cons:** Needs infrastructure (planning, memory, tool calling), harder to test  
**Best used when:** You want to automate well-bounded business workflows while retaining some control.

---

## 4. Autonomous Agents  
**High Autonomy, Low Control**

These agents are fully goal-driven. You give them a broad objective, and they figure out what to do, how to do it, when to retry, and when to escalate. They act independently, often across systems and over time.

**What problems do they solve?**  
High-effort, async, or long-running tasks that span multiple systems or steps and don’t need constant human input.

**Examples:**  
- A competitive research agent that pulls data over days, summarizes updates, and generates weekly insight briefs  
- An ops automation agent that detects issues in pipelines, diagnoses root causes, and files tickets with suggested fixes  
- A testing agent that autonomously runs product flows, logs results, and suggests new edge-case scenarios  

**How the LLM is used:**  
The LLM is the planner, decision-maker, tool-user, memory tracker, and communicator. It manages retries, evaluates whether goals are met, and decides when to stop or adapt.

**Pros:** Extremely scalable, can handle complex tasks  
**Cons:** High risk if not monitored, hard to evaluate or trace, infra-heavy  
**Best used when:** The task is high-leverage, async, and doesn’t require human feedback at every step.

---
<img width="683" height="316" alt="image" src="https://github.com/user-attachments/assets/cbea3f8f-b2c5-4dbc-8d0f-5b102430d675" />


## How to Decide What to Build

Not by picking your favorite architecture.  
You start with the **problem**.

Ask yourself:  
- Is it repetitive and structured?  
- Does it involve language understanding or generation?  
- Is it a multi-step task that needs decision-making?  
- Do you trust an AI system to execute the entire task, or do you want a human in the loop?  

Here’s the key:  
- These approaches aren’t mutually exclusive.  
- A single system can mix them — some parts might require high control, others can benefit from high autonomy.  
- Each problem type can be tackled by either a single agent or a group of collaborating agents.

We’ll dive deeper into **single-agent vs. multi-agent design** later in the course.  
For now, remember:  
> Don’t start with “How do I build a multi-agent system?”  
> Start with “What’s the problem I’m solving, and what kind of autonomy does it require?”  

Let the problem shape the agentic design, not the other way around.

---

In the next part, we’ll dive deeper into the **role of tools** in agentic systems. They’re the reason AI has become far more usable — and we’ll break down exactly how and why in our deep dive.



