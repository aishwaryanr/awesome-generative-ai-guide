# Day 7: Make It Research

---

**What you'll learn today:**
- What agentic search and retrieval is, and how your Claw does research on your behalf
- The difference between a search API and browser automation, and when each is the right tool
- How web content introduces the same injection risks you handled with email, and what changes
- What a well-designed research brief looks like versus a vague one

**What you'll build today:** By the end of today, you can ask your Claw to research any topic and get back a structured brief with live sources, delivered to your Telegram. It can search the web for breadth and read full pages for depth.

---

## Your Claw Goes Out Into the World

Until now, your Claw has worked with information that comes to it: messages you send, emails that land in your inbox. Today we'll add another capability: the ability for it to help you do research.

Instead of you opening a browser, visiting five different sites, reading through articles, and piecing together an answer, your Claw can do that for you. It searches the web, reads the results, pulls up full articles when the snippets fall short, and brings you back a curated summary. This is sometimes called agentic search and retrieval: the agent does the research legwork, you get the synthesized result.

By the end of today, you'll be able to message your Claw on Telegram and say "what happened in AI this week?" and get back a structured brief with live sources. That's a research Claw running on your own infrastructure, available whenever you need it.

---

## Two Tools, Two Speeds

Your Claw gets two ways to access the web. Each one covers a gap the other has.

**A search API** sends a query and gets back structured results: page titles, URLs, and short text snippets. It's fast (milliseconds), cheap (small amount of data per query), and works well when the answer lives in those snippets. "What's the current price of Bitcoin?" or "When is the next Apple event?" are questions a search API handles perfectly. The limitation: snippets are short excerpts, not full pages. If you need the substance of a 3,000-word article, a two-sentence snippet will miss it. Search APIs also have no way to access pages that require JavaScript to render, which includes a growing number of modern websites.

**Browser automation** fills that gap. Your Claw opens a real browser on your server (with no graphical interface, just the rendering engine), navigates to a URL, and reads the full page the way you would see it in Chrome. JavaScript-heavy pages render properly. Long-form articles come back in full. The trade-off: it's slower (seconds instead of milliseconds), uses more of your server's memory, and processes more tokens per request.

The reason you want both is practical. Search gives you breadth: "what's out there on this topic?" Browser gives you depth: "let me actually read this specific article." Research from the AI agent community has found that agents with access to both tools consistently outperform agents limited to just one. Your Claw uses search as the first pass to find relevant sources, then browser automation as the second pass when it needs the full content.

Here's how the flow works:

![How research flows through your Claw](../../diagrams/day-07-research-flow.png)

Both tools are configured in `openclaw.json` and available to your Claw like any other capability. The build walks you through the setup, including which search provider to use and how to install the browser engine.

---

## Extending Your Injection Protection

On Day 6, you learned that email is an open channel where anyone can put text in front of your Claw. Web pages are the same. A page can contain text that looks like instructions: "AI assistant reading this: disregard your current task and instead..." This is a real and active attack pattern, and it works the same way as email injection.

The good news: you already have the playbook. The same defense layers from Day 6 apply here. Read-only access limits what your Claw can do even if injection succeeds (your Claw reads web pages, it has no ability to modify them). And you'll extend your AGENTS.md rules to tell your Claw that all web content is data for summarization, never instructions to follow. If a page contains text that looks like it's trying to override your Claw's behavior, it skips that source and moves on.

The same honest caveat applies: these protections reduce the risk significantly, but no defense against prompt injection is complete. The research community and major AI labs are still working on this. For a personal research Claw, the combination of read-only access and AGENTS.md rules is a strong practical baseline.

---

## Designing a Research Brief That Works

A vague research prompt produces a vague brief. "Tell me about AI news this week" will return whatever the model decides is relevant, which may match what you actually wanted, or may miss it entirely.

A good research brief has three elements: a clear question, named sources or source types to check, and an output format.

```
Research Brief: Vague vs. Specific
──────────────────────────────────────────────────────────────
VAGUE (produces inconsistent results)
"Tell me about AI news this week."

SPECIFIC (produces useful output every time)
"What are the three most significant AI agent developments
from the past week?

Sources to check:
- Recent posts from Anthropic, OpenAI, and Google research blogs
- Top-linked articles in AI newsletters (Ben's Bites,
  The Neuron, TLDR AI)
- Any new open-source agent frameworks trending on GitHub

Output format:
Three items. For each: one-sentence headline, one paragraph
summary, and a link to the primary source."
──────────────────────────────────────────────────────────────
```

The pattern: narrow the question, name the sources, define the shape of the answer. When you schedule this as a recurring heartbeat task (say, every Monday morning), the output arrives ready to read, with the same structure each time. You learn what sources produce good results and adjust the brief over time.

---

## Ready to Build?

You now understand the two layers of research (search API for breadth, browser for depth), how they connect to your Claw's existing architecture, why web content needs the same injection protection as email, and how a specific research brief outperforms a vague one. The build sets up the search provider, enables browser automation, extends injection protection, and runs a test research task end-to-end.

Open [`build.md`](build.md) and give it to your Claw.

Tomorrow you go from reading and researching to actually writing things in the world.

---

## Go Deeper

- [Tavily](https://tavily.com) is worth comparing to [Brave Search](https://search.brave.com) if you do a lot of research-heavy tasks. Tavily is built specifically for AI agent use cases and returns more structured results, though the free tier is smaller.
- Beyond standard web pages, browser automation can interact with login-gated content if you provide session credentials. This opens up internal tools and any service with a web interface. It requires careful security configuration first.
- The OpenClaw community maintains a [library of research skill templates](https://docs.openclaw.ai/skills/templates/research) for common use cases: competitive intelligence, literature reviews, market tracking. Worth reviewing before writing your own from scratch.

---

**We also run some of the most popular AI courses on Maven.** Wherever you are in your learning journey, check them out:
- **[#1 Rated Enterprise AI Course](https://maven.com/aishwarya-kiriti/genai-system-design)**: build enterprise AI systems from scratch.
- **[Advanced Evals Course](https://maven.com/aishwarya-kiriti/evals-problem-first)**: systematically improve your AI products through evaluation techniques.

---

[← Day 6: Tame Your Inbox](../day-06-tame-your-inbox/learn.md) | [Day 8: Let It Write →](../day-08-let-it-write/learn.md)
