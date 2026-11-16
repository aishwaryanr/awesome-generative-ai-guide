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

## Try It: Discover a Tool and Check Its Evidence

In Part 3, we saw how a program can call a tool. Here, you'll ask an MCP server which tools it offers, inspect the search tool's inputs, and find the passage that supports a claim.

Use Python 3.10 or later and the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v2.3.0). In a virtual environment, install the versions used in this exercise.

```sh
python -m pip install mcp==2.3.0 httpx2==2.16.2
```

Save the code below as `mcp_search.py` and run `python mcp_search.py`. It sends the example query to the [public Parallel MCP server](https://docs.parallel.ai/integrations/mcp/quickstart). The exercise requires internet access and runs without an account, an API key, or a model.

```python
import asyncio
import json
from urllib.parse import urlsplit

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

ENDPOINT = "https://mcp.parallel.ai/mcp"
CLAIM = "Our solar system has eight planets."


async def main():
    async with httpx2.AsyncClient(
        headers={"User-Agent": "awesome-generative-ai-guide/mcp-lab"},
        timeout=30.0,
        trust_env=False,
    ) as http:
        transport = streamable_http_client(ENDPOINT, http_client=http)
        async with Client(transport, read_timeout_seconds=30) as client:
            tools = (await client.list_tools()).tools
            print("Available tools:", ", ".join(tool.name for tool in tools))
            search = next((tool for tool in tools if tool.name == "web_search"), None)
            if search is None:
                raise SystemExit("The MCP server did not advertise web_search.")
            print("Search input schema:", json.dumps(search.input_schema, indent=2))

            result = await client.call_tool(
                search.name,
                {
                    "objective": "Find a NASA passage stating how many planets are in our solar system.",
                    "search_queries": ["site:science.nasa.gov solar system eight planets"],
                },
            )

    if result.is_error:
        raise SystemExit(f"Search failed: {result.content}")

    for source in (result.structured_content or {}).get("results", []):
        url = urlsplit(source["url"])
        if url.scheme != "https" or url.hostname != "science.nasa.gov":
            continue
        for excerpt in source.get("excerpts", []):
            for line in excerpt.splitlines():
                if "our solar system has eight planets:" in line.casefold():
                    print(f"\n{CLAIM} [1]\n[1] {source['url']}")
                    print("\nPassage returned by the tool:\n" + line)
                    return

    raise SystemExit("No returned NASA passage supported the example claim. Try a more specific query.")


if __name__ == "__main__":
    asyncio.run(main())
```

The script prints the tool names and the search tool's input schema before making the call. It prints the example claim only when it finds a matching NASA excerpt, then shows the source URL and the exact text returned by the tool. Search results can change. Open the URL and check the excerpt in context. Then change the objective and search query to ask a different public question. To inspect the new results, print their URLs and excerpts in place of the NASA matching loop. Pick a passage and write a sentence it supports. If the service rate-limits the request, wait a bit and run the script again.

---

## Further Reading & Resources

- We did a **[full deep-dive article](https://thenuancedperspective.substack.com/p/mcp-overhyped-misunderstood-and-actually)** on MCP, including clients, servers, and real-world use cases (written by Kiriti Badam, OpenAI).  
- We also ran a **free live session** — you can catch the [recording](https://maven.com/p/82345a) here.

---

In the next part, we’ll learn about the **planning** component of agentic systems and why it matters.

PS: We also teach a widely loved course on how to actually build AI systems in this fast-changing environment, using a problem-first approach. It’s designed for PMs, leaders, engineers, decision-makers etc. who are working within real-world constraints. Alumni come from Google, Meta, Apple, Netflix, AWS, Spotify, Snapchat, Deloitte, and more. Our next cohort starts soon. Our next cohort starts soon. Early bird pricing is live: use the code "GITHUB" to get $300 off (Valid only for August 2025) to [register here](https://maven.com/aishwarya-kiriti/genai-system-design)!!


