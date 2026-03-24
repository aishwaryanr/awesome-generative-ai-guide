# Day 9: Enable Teamwork

Follow these instructions exactly. Your goal is to connect the main agent and the writer agent safely and report exactly what changed.

Before making changes, read these files if they exist:

- `~/.openclaw/openclaw.json`
- `~/.openclaw/workspace/AGENTS.md`
- `~/.openclaw/workspace-writer/SOUL.md`
- `~/.openclaw/workspace-writer/AGENTS.md`

Use them to confirm the writer exists and to preserve any existing config.

---

## 1. Verify Prerequisites

Confirm all of these are true before continuing:

- the `writer` named agent exists
- `~/.openclaw/workspace-writer/SOUL.md` exists
- `~/.openclaw/workspace-writer/USER.md` exists
- `~/.openclaw/workspace-writer/AGENTS.md` exists
- `~/.openclaw/workspace-writer/MEMORY.md` exists

If something is missing, stop and tell the user what to fix first.

---

## 2. Enable Main to Writer Delegation

Update the current config so `main` and `writer` can use agent-to-agent messaging for this Day 9 setup.

Rules:

- preserve existing unrelated config
- avoid duplicate blocks
- scope the allow list to `main` and `writer` for this lesson unless there is already a narrower safe rule in place
- reload or restart only if needed

---

## 3. Add a Long-Form Delegation Rule to the Main Workspace

Update `~/.openclaw/workspace/AGENTS.md`.

If a suitable section already exists, extend it. Otherwise add a short section with this content:

```md
## Named Agent Delegation

For long-form essays, newsletters, opinionated explainers, and substantial rewrites, delegate the drafting step to `writer`.

Stay the coordinator. Gather the brief, pass the topic, angle, and audience clearly, then return the writer's draft to the user without rewriting the voice unless the user asks for that.

Any sending, posting, publishing, or other external action stays with the main agent and still requires confirmation.
```

Keep the addition brief and do not rewrite unrelated parts of the file.

---

## 4. Confirm Completion

After the changes are done:

- summarize exactly what changed
- confirm whether a reload or restart happened
- give the user these two exact test prompts for next:
  1. `Write a short Substack post, about 500 to 700 words, on why most productivity advice is backwards. The audience is skeptical knowledge workers.`
  2. `I need a Substack draft about how personal AI assistants are becoming the new operating system for knowledge work. Delegate this to the writer agent and bring me back the draft.`
- stop
