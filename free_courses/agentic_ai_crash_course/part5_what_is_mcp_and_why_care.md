# Part 5: What Is MCP and Why Should You Care?

---

## First, a Quick Recap

- **Part 3:** We learned that tools let models **do things**.  
- **Part 4:** We saw that RAG helps models **find relevant info** before answering.

These are **external supports** — they help the model act smarter, but the coordination still sits outside the model.  

But what if you could pass **all the context a model needs** — tools, retrieved data, memory, instructions — in one clean, structured format?

That’s what **Model Context Protocol (MCP)** is trying to solve.

---

## So What Is MCP?
<img width="737" height="452" alt="image" src="https://github.com/user-attachments/assets/2e17fcb6-ab35-4a4c-88b4-c74db165e4b8" />


At its core, **Model Context Protocol** is a standardized way to give an LLM everything it needs to reason and respond.

Think of it like packaging up:

- The task you want the model to do  
- The tools/APIs it can use  
- The documents or memory it might need  
- The prior messages in the conversation  

…and then handing all of that over in one go.

It’s **not** a tool, library, or product.  
It’s a **protocol** — a structure for communication between the model and the outside world.

If you’re from the tech world, equivalents would be: **HTTP**, **TCP/IP**, or **SMTP**.  
If you’re not, just remember: tech folks love standardization — it makes things easier to reuse and plug together.

---

## Why Does This Matter?

Let’s say you’re building an agent.  
Right now, you’re probably juggling:

- Sending a prompt  
- Passing retrieved documents  
- Registering tools  
- Managing state  
- Keeping track of what happened before  

MCP says:  
> “Let’s standardize how we give all of that to the model, so we don’t reinvent the wheel for every use case.”

And for **enterprises**, this matters a lot.  
As agents get more complex, coordinating **tools**, **RAG**, **memory**, and **outputs** becomes messy.

MCP makes that orchestration **composable**, **modular**, and easier to plug into other systems.

If you’ve ever worked with APIs, think of MCP like a **well-defined request schema**.  
Instead of tossing everything into one long string and hoping the model figures it out, the model still sees text — but it’s **structured**, with **clear context, options, and grounding**.

---

## Why Did MCP Catch On So Fast?

Given that MCP is just a protocol, you might be wondering:  
> What makes it better, and why did everyone jump on board?

Here’s what helped:

1. **AI-Native** — MCP was built for AI agents. It makes space for everything agents use today: tools, prompts, memory, documents, and more.
2. **Strong docs and examples** — Anthropic (creators of MCP) released not just the spec but also clients, SDKs, testing tools, and real-world demos.
3. **Network effect** — Released quietly in Nov 2024, most people slept on it… until 2025, when it exploded. Tools, startups, and even OpenAI began supporting it.

---

## Common Misunderstandings

- **MCP isn’t a new API or product** — It’s just a pattern, a clean way to frame what you send to the model.  
- **It doesn’t make models smarter** — It just gives them better, more structured context.  
- **It’s not just for agents** — Even simple assistants benefit from better context management.

---

## So… Should You Care?

If you’re building toy prompts or quick demos — probably not (yet).  

But if you’re working on:

- Enterprise-grade agents  
- Multi-tool workflows  
- LLMs that need to access **memory + RAG + planning**  
- Systems where **context management** is a bottleneck  

…then **yes**, you should care. MCP is about getting better at passing evolving, structured context into models.

But keep in mind: MCP is just a protocol.  
Like all standards, it only works if it’s widely adopted.  
If something better comes along before MCP becomes “the HTTP of agents,” the ecosystem might shift again.

---

## Further Reading & Resources

- We did a **full deep-dive article** on MCP, including clients, servers, and real-world use cases (written by Kiriti Badam).  
- We also ran a **free live session** yesterday — you can catch the recording here.

---

In the next part, we’ll learn about the **planning** component of agentic systems and why it matters.

PS: We also teach a widely loved course on how to actually build AI systems in this fast-changing environment, using a problem-first approach. It’s designed for PMs, leaders, engineers, decision-makers etc. who are working within real-world constraints. Alumni come from Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte, and more. Our next cohort starts soon. Our next cohort starts soon. Early bird pricing is live: use the code "GITHUB" to get $300 off (Valid only for August 2025) to [register here](https://maven.com/aishwarya-kiriti/genai-system-design)!!

