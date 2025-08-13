# Part 9: Real-world Agentic Systems (Under the hood)



---

So far, we’ve covered all the ingredients that make up an agent:  
**tools**, **planning**, **RAG**, **memory**, **structure**, and **coordination** in multi-agent setups.

But you might be thinking:  
> “Where does all this actually show up in the real world?”

Let’s walk through a few public-facing systems that exhibit **agentic behavior** — as far as we can tell.

⚠️ **Note:**  
These aren’t open source. We don’t know their exact internals.  
What follows is an informed simplification based on how they behave externally — just enough to understand how the agentic stack might show up in practice.

---

## **NotebookLM (Google): Agentic Search on Your Own Data**

Google’s NotebookLM acts like a personal research assistant. You upload your files, and it helps you work with them — summarizing, answering questions, even generating audio versions or study guides.

**Core focus:** Q&A over your content — essentially a scaled-up, personal RAG system.

**How it likely works:**
1. **User uploads files** (PDFs, notes, slides, etc.)  
2. **Preprocessing** — Stores them for retrieval later.  
3. **User asks a question** — e.g., _“What were the key insights from my Q2 strategy deck?”_  
4. **Planning** — Interprets task type (summary, Q&A, comparison?), identifies relevant docs/sections.  
5. **RAG** — Retrieves the most relevant document chunks.  
6. **LLM Generation** — Responds clearly, grounded in your content.  
7. **Memory** —  
   - Short-term: Tracks the conversation.  
   - Long-term: Likely minimal or none.  
8. **Tools** — Possibly file viewers, summarization modules.

**What makes it agentic:** Interprets goals, searches across your data, and composes responses — not just static outputs.

---

## **Perplexity: Agentic Search on the Open Web**

Perplexity gives you a direct, answer-like response with sources — instead of a page of links.

**How it likely works:**
1. **User asks a question** — e.g., _“What’s the latest research on Alzheimer’s treatments?”_  
2. **Planning** — Interprets intent (“latest,” “credible”), decides search approach.  
3. **Tool Use** — Issues queries via web APIs.  
4. **RAG** — Retrieves relevant page snippets.  
5. **LLM Response** — Synthesizes an answer with citations.  
6. **Memory** —  
   - Short-term: Session context.  
   - Long-term: May store preferences (e.g., “always use WSJ for news”).

**What makes it agentic:** Fetches info, decides what to use, and constructs an answer in a multi-step loop.

---

## **DeepResearch (OpenAI): Deep Agentic Workflows**

DeepResearch tackles **open-ended, complex research tasks** — e.g., market analysis, competitive landscapes, technical deep dives.

**How it likely works:**
1. **User asks a broad task** — e.g., _“Analyze the generative AI landscape for education startups.”_  
2. **Planning** — Breaks into subtasks (funding, trends, companies, risks), forms an execution plan.  
3. **Tools** — Likely includes:  
   - Web search  
   - Document readers (PDFs)  
   - Data tools (spreadsheets, graphs)  
   - Report generation modules  
4. **Agentic RAG** — Not one-shot retrieval — fetches, reflects, re-fetches as task evolves.  
5. **Memory** —  
   - Episodic: Tracks which parts are done.  
   - Semantic: Stores key facts/names.  
6. **Multi-step Reasoning** — Loops: plan → retrieve → read → rethink → generate → refine → repeat.

**What makes it agentic:** Heavy planning, iterative tool use, self-directed progress.

---

## **Connecting to Day 2: Levels of Autonomy**
<img width="694" height="370" alt="image" src="https://github.com/user-attachments/assets/48812496-309d-42cd-9c86-8ef3cb345ec2" />
<img width="969" height="231" alt="image" src="https://github.com/user-attachments/assets/12489209-a75b-4a31-853e-33dda02e1aaa" />



**NotebookLM** — Between Level 2 and Level 3.  
- High-control workflow agent.  
- Strong retrieval, limited autonomous decision-making.  

**Perplexity** — Level 3 (maybe touching Level 4).  
- Plans queries, organizes sources, crafts answers.  

**DeepResearch** — Strong Level 4.  
- Takes high-level goals, breaks down tasks, works iteratively with minimal guidance.

---

## Try It Yourself

They all have free versions — experiment and watch for:  
- How much **control** you have  
- How much the **system decides** on its own  

It’s a great way to sharpen your instinct for agent design.

---

## Up Next

In the next part, we’ll wrap up the series:  
- Summarize what we’ve learned  
- Share best practices  
- Take a quick look at where **agentic AI** is headed

PS: We also teach a widely loved course on how to actually build AI systems in this fast-changing environment, using a problem-first approach. It’s designed for PMs, leaders, engineers, decision-makers etc. who are working within real-world constraints. Alumni come from Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte, and more. Our next cohort starts soon. Our next cohort starts soon. Early bird pricing is live: use the code "GITHUB" to get $300 off (Valid only for August 2025) to [register here](https://maven.com/aishwarya-kiriti/genai-system-design)!!

