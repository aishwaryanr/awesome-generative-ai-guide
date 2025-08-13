# Part 4: Retrieval-Augmented Generation (RAG) and the Rise of Agentic RAG

In the previous part, we looked at how tools help AI agents interact with real-world systems — send emails, file tickets, trigger APIs.

But what if the model doesn’t need to act?  
What if it just needs access to the right information?

That’s the case in many enterprise settings:  
- Internal docs spread across teams  
- Policy PDFs no one remembers writing  
- Customer insights buried in CRM notes  
- Dashboards and emails with useful context  

Tools won’t help here. The model needs to think with your data.  
That’s where **RAG** comes in.

---

## What Is RAG?

RAG stands for **Retrieval-Augmented Generation**.  
It’s a system design where the model retrieves relevant information from your own data — just before generating a response.

Instead of relying only on what the model was trained on, RAG gives it access to **live, contextual information** from your enterprise systems. This makes answers more accurate, grounded, and auditable.

You might be wondering:  
> “Why not just give all the data to the model directly?”

The problem is:  
- Models can only process a limited amount of text at a time.  
- Even within that limit, they struggle when too much irrelevant or noisy information is included.  
- This makes responses less focused and more error-prone.

---

## The RAG Process (at a Glance)

<img width="1024" height="356" alt="image" src="https://github.com/user-attachments/assets/2e7c2384-a564-45c1-9a08-57cfb02ee435" />


Here’s what it looks like in practice:

1. **Data** – Your internal content (PDFs, emails, notes, wikis)  
2. **Chunking** – Broken into smaller parts for better indexing  
3. **Prompt + Context** – At query time, the system retrieves relevant pieces (retrieval phase)  
4. **LLM** – The model uses that context to generate a response  
5. **Output** – The result is based on your data, not just what the model “knows”  

_Image Source: https://hyperight.com/7-practical-applications-of-rag-models-and-their-impact-on-society/_

---

## Why RAG Is Everywhere in Enterprise AI

You’ll often hear this number:  
> From what I’ve seen across clients and systems, **70% of enterprise GenAI use-cases use RAG**.

Why RAG is invaluable to enterprises:  
- Enterprise knowledge changes frequently  
- Fine-tuning models is expensive and slow  
- Retrieval is faster, safer, and easier to control  
- It brings structure and traceability into LLM systems  
- It works on both unstructured (docs) and semi-structured (dashboards, notes) data  

So instead of asking:  
> “How do I teach the model everything we know?”  
Most teams ask:  
> “How do I let the model fetch what we already have?”

---

## RAG = LLM + Additional Retrieved Data

RAG became the dominant pattern in 2024 for a reason:  
It bridged the gap between general-purpose LLMs and private, task-specific enterprise knowledge.

At its core, RAG is simple:  
- You take an LLM  
- You feed it additional, retrieved information right before generation  

This makes the model more accurate, more context-aware, and less reliant on memorized facts.  
It’s especially useful for tasks like **Q&A, summarization, and policy lookups** — particularly in data-rich environments like **legal, finance, and support**.

No wonder 2024 was dubbed **“the year of RAG.”**

---

## But Now We’re Moving Into the Agentic Era

RAG isn’t going away, but it’s evolving.

Today’s systems don’t just retrieve once and generate an answer.  
In **agentic workflows**, retrieval becomes part of a broader, dynamic reasoning loop.

Agents plan, retrieve, reflect, and retrieve again — not just once, but as many times as needed throughout a task.

That’s where **Agentic RAG** comes in.

---

## What Is Agentic RAG?

<img width="1456" height="971" alt="image" src="https://github.com/user-attachments/assets/8a7347b0-2ead-4d56-9f33-ef57667d0f00" />


Traditional RAG:  
- One query  
- One retrieval  
- One response  

It works well for standalone questions like:  
> “What’s our policy on PTO rollover?”

But most real-world enterprise workflows aren’t one-shot.

---

**Example:**  
Let’s say you’re building a deal assistant for your sales team.  
In a single task, the agent may need to:  
- Pull the customer’s CRM history  
- Retrieve current pricing for their segment  
- Look up regional legal terms  
- Reference past contract clauses  
- Generate a custom proposal  
- Double-check facts  
- Log the interaction  

---

In **agentic systems**, retrieval isn’t just a setup step.  
It’s how the agent:  
- Gathers missing context  
- Checks its assumptions  
- Adapts mid-task  

That means RAG becomes:  
- A tool for in-task learning  
- A method for reducing hallucinations  
- A mechanism for handling dynamic workflows  
- A bridge between reasoning and grounded enterprise knowledge  

Agentic RAG turns retrieval into a **first-class decision-making loop** by using retrieval as part of the model’s thinking process.

---

## RAG as a Tool

If you think about it, RAG is also a kind of **tool**.  
But instead of triggering an action, it helps the agent pull the right information from a large volume of data.

In practice, agents often combine:  
- **RAG**  
- **Tools**  
- **Planning**  

…to complete complex tasks **reliably and contextually**.

---

## A Note on Scope

RAG is a deep and rapidly evolving space — honestly, it could be its own course.  
If you're curious to explore further:  
- I’ve curated a **GitHub repo** of key RAG papers that covers the landscape well  
- I have a **101 guide on Agentic RAG** too

That said, not every RAG optimization is necessary for every use-case.  
In our 6-week course, we focus on helping you understand **when and where** each technique makes sense, rather than applying them blindly.

---


In the next part, we’ll dive into one of the most talked-about concepts lately: **Model Context Protocol (MCP)**.  

To get the most out of it, I’d recommend revisiting **Part 3 on tools**, since MCP builds directly on that concept!

PS: We also teach a widely loved course on how to actually build AI systems in this fast-changing environment, using a problem-first approach. It’s designed for PMs, leaders, engineers, decision-makers etc. who are working within real-world constraints. Alumni come from Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte, and more. Our next cohort starts soon. Our next cohort starts soon. Early bird pricing is live: use the code "GITHUB" to get $300 off (Valid only for August 2025) to [register here](https://maven.com/aishwarya-kiriti/genai-system-design)!!
