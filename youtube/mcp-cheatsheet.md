# MCP in 2026: the cheat sheet

**Video: Coming up soon.** · [Video notes and resources](mcp-in-2026.md)

![MCP in 2026 visual cheat sheet](images/mcp-in-2026.png)

The companion cheat sheet for my MCP explainer: what it is, how it works, APIs vs. MCP vs. skills vs. plugins, and the four changes to know in 2026.

The protocol changes below refer to the **2026-07-28 specification**. Older servers and clients may still use an earlier revision; extensions need support on both sides.

---

## Why MCP

**Model Context Protocol (MCP)** is an open standard for connecting AI applications to external tools and data.

An API gives software access to a service. An agent also needs to know which operations are available, what each one does, and what arguments it expects. Developers used to build their own versions of that middle layer for each integration.

MCP gives them a common way to discover capabilities, call tools, and receive results. The underlying service can still use its existing API. The AI application decides how to use the capabilities MCP exposes. [MCP architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture).

---

## How MCP works

| Piece | What it does | Email example |
|---|---|---|
| **Host** | The AI application. Manages the model, context, permissions and tool use | The app where you ask to send event reminders |
| **Client** | The component inside the host that speaks MCP | Discovers the email tools and sends MCP requests |
| **Server** | A program that exposes tools and data using MCP | Receives the request and calls Gmail's API |

A server can be built by the service provider, a third party, or your own team.

### Follow one request

You ask: “Email everyone registered for tomorrow's event.”

1. The client discovers the server's tools, descriptions and argument schemas through `tools/list`.
2. The host makes the relevant tool information available to the model.
3. The model chooses a tool and generates its arguments.
4. The host applies its permission rules, then the client sends `tools/call`.
5. The server calls the underlying API. The result returns through the client to the host.

MCP standardizes the exchange. Tool descriptions, model decisions and application permissions still determine whether the action is appropriate.

---

## Local and remote

| Setup | Where the MCP server runs | How the client talks to it |
|---|---|---|
| **Local** | On your machine, usually launched by the host | Standard input/output (`stdio`) |
| **Remote** | On infrastructure run by a provider or another operator | Streamable HTTP |

“Local” describes where the MCP server runs. A local Gmail integration still calls Gmail over the internet. It also changes who operates that middle layer and manages its credentials.

---

## APIs vs. MCP vs. Skills vs. Plugins

| Layer | Its job | In the event-reminder workflow |
|---|---|---|
| **API** | Gives software access to the service's operations | Gmail's API sends the email |
| **MCP** | Standardizes how an AI application discovers and uses capabilities | Exposes an email-sending tool and its required arguments |
| **Skill** | Provides reusable instructions, resources and optional scripts for a workflow | Check registrations, use the right template, avoid duplicate reminders |
| **Plugin** | Packages capabilities for installation in a particular app | Could bundle the email integration and reminder skill |

A skill can use MCP tools, but it can also use other tools or scripts. [Agent Skills overview](https://agentskills.io/home).

A plugin's contents depend on the application; it does not have to include an MCP server.

---

## The four changes in 2026

### 1. Stateless MCP

Earlier revisions used an initialization handshake and could create protocol sessions. Keeping session state across multiple server instances meant coordinating shared state or routing requests back to the same instance.

The July revision removes the handshake and protocol session IDs. Each request carries its protocol version and client capabilities. `server/discover` advertises the server's versions and capabilities. This reduces the need for infrastructure to maintain protocol sessions.

Application state can still exist: conversation history, jobs and business records do not disappear. [July specification changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog).

**If the server needs more input:** Multi Round-Trip Requests (MRTR) let it return an `input_required` result. The client gathers the requested information and retries the original request with `inputResponses` and the server's `requestState`.

For example, an email tool could request confirmation before sending a batch. The client collects that confirmation and carries it into the retry. [MRTR specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr).

### 2. Tasks for long-running work

A large codebase analysis or batch-processing job can take minutes or hours. Keeping one request open for the entire operation risks connection timeouts.

With the **Tasks extension**, the server can return a durable task handle. The client polls `tasks/get` for progress and receives the final result when the task completes. If more input is needed, the client supplies it through `tasks/update`.

Tasks were experimental in the earlier core protocol. The redesigned version lives in the official `io.modelcontextprotocol/tasks` extension and requires explicit client and server support. [Tasks overview](https://modelcontextprotocol.io/extensions/tasks/overview).

### 3. Stronger authorization

The July authorization changes tighten how clients identify authorization servers and handle credentials:

- Validate a returned issuer identifier against the expected authorization server.
- Keep credentials tied to the issuer that created them.
- Prefer Client ID Metadata Documents over the deprecated Dynamic Client Registration mechanism.

These rules strengthen identity and access handling. They do not eliminate tool poisoning, prompt injection or inappropriate tool use. [Authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization).

**Enterprise-Managed Authorization** lets organizations provision access through their identity provider. Employees can use approved servers through their company login, with access based on organizational policy. [Enterprise-Managed Authorization](https://blog.modelcontextprotocol.io/posts/enterprise-managed-auth/).

### 4. Capabilities for production use

| Change | What it enables |
|---|---|
| **Extensions** | Optional capabilities can evolve outside the core protocol |
| **MCP Apps** | Tools can return interactive dashboards, forms and other interfaces inside a supporting host |
| **Caching** | Freshness and cache-scope metadata let clients reuse responses instead of repeatedly fetching unchanged information |
| **Request headers** | Method and tool-name headers help infrastructure route and handle MCP traffic |
| **Deprecation policy** | A minimum 12-month deprecation window gives builders time to plan migrations |

[MCP Apps](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) render interactive interfaces in a sandboxed frame. Check that your host supports the extension before depending on that UI.

Caching, request headers and the deprecation policy are covered in the [July specification changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog). Deprecated features remain available during their deprecation window; deprecation and removal are separate states.
