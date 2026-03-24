# Day 2 Build: Give It a Soul

This is the user-facing guide for Day 2. Today you turn a generic OpenClaw install into *your* Claw by creating four identity files:

- `SOUL.md`
- `USER.md`
- `AGENTS.md`
- `MEMORY.md`

---

## What You Need Before Starting

- Day 1 complete: OpenClaw installed, reachable, and named
- Access to your Claw through the web chat

---

## How To Run Day 2

Work through the files in this order:

1. [`claw-instructions-create-soul.md`](./claw-instructions-create-soul.md)
2. [`claw-instructions-create-user.md`](./claw-instructions-create-user.md)
3. [`claw-instructions-create-agents.md`](./claw-instructions-create-agents.md)
4. [`claw-instructions-create-memory.md`](./claw-instructions-create-memory.md)
5. [`claw-instructions-finalize-identity.md`](./claw-instructions-finalize-identity.md)

[`claw-instructions-create-soul.md`](./claw-instructions-create-soul.md) verifies the workspace and creates `SOUL.md`. [`claw-instructions-finalize-identity.md`](./claw-instructions-finalize-identity.md) locks permissions, restarts the gateway, and verifies that the new identity loaded correctly.

---

## Step 1: Create SOUL.md

Copy and paste the following message into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-02-give-it-a-soul/claw-instructions-create-soul.md` and follow every step. Ask the questions in order, create `SOUL.md`, and stop when you're done.

That [instruction file](./claw-instructions-create-soul.md) tells the Claw to:

- verify `~/.openclaw/workspace/` and `~/.openclaw/workspace/memory/` exist
- ask you the Day 2 identity questions in order
- turn your answers into a finished `SOUL.md`

This is the most important file. It defines:

- who the Claw is
- how it behaves when uncertain
- what it must never do
- how it should sound

Take your time here. Specific prohibitions and concrete language habits are more useful than abstract aspirations.

---

## Step 2: Create USER.md

After `SOUL.md` is finished, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-02-give-it-a-soul/claw-instructions-create-user.md` and follow every step. Ask the questions in order, create `USER.md`, and stop when you're done.

[`claw-instructions-create-user.md`](./claw-instructions-create-user.md) creates the briefing document about *you*: your role, location, working style, and what is currently on your plate. Keep sensitive or private details out of `USER.md`. That kind of context belongs in `MEMORY.md`, which is private-session-only.

The most important part of `USER.md` is the **Focus** section. Make it current. If it gets stale, the Claw's help gets stale too.

---

## Step 3: Create AGENTS.md

After `USER.md` is finished, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-02-give-it-a-soul/claw-instructions-create-agents.md` and follow every step. Create `AGENTS.md`, confirm the names are consistent, and stop when you're done.

[`claw-instructions-create-agents.md`](./claw-instructions-create-agents.md) creates the operating manual the Claw follows every session:

- startup checklist
- memory rules
- security rules
- confirmation protocol
- response defaults

Most of it is pre-written. The main thing to verify is that the Claw name and user name match the names already used in `SOUL.md` and `USER.md`.

---

## Step 4: Create MEMORY.md

After `AGENTS.md` is finished, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-02-give-it-a-soul/claw-instructions-create-memory.md` and follow every step. Ask the questions in order, create `MEMORY.md`, and stop when you're done.

[`claw-instructions-create-memory.md`](./claw-instructions-create-memory.md) is the only identity file designed to grow over time. It stores:

- current private context
- open loops
- sensitive personal details you do not want in group chats
- durable patterns about how you operate

If you do not want to answer one of the setup questions yet, skip it. The file can start sparse and get better over time.

---

## Step 5: Lock It Down and Verify

After `MEMORY.md` is finished, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-02-give-it-a-soul/claw-instructions-finalize-identity.md` and follow every step. Set permissions, restart the gateway, run the verification, and report PASS or FAIL for each item.

That [instruction file](./claw-instructions-finalize-identity.md) tells it to:

- set file permissions
- restart the gateway
- run the two verification questions
- report whether the identity files actually loaded

That is the formal verification. Once it passes, use the setup right away:

## Two Quick Wins

**Quick Win 1**

```text
Give me the short version of how you plan to work with me. Use what you know about my role, current focus, and preferred style.
```

The response should sound like *your* Claw, not a generic assistant. It should reflect your real focus and the way you asked it to communicate.

**Quick Win 2**

```text
Based on what you know about me so far, what are the 2-3 most useful ways you can help me this week? Also tell me what kinds of actions you will always check with me before taking.
```

This should feel specific to your actual priorities. It should also show that the confirmation rules from `SOUL.md` and `AGENTS.md` are in place, without sounding like it is just reciting a config file back to you.

---

## What Should Be True After Day 2

- [ ] `~/.openclaw/workspace/SOUL.md` exists
- [ ] `~/.openclaw/workspace/USER.md` exists
- [ ] `~/.openclaw/workspace/AGENTS.md` exists
- [ ] `~/.openclaw/workspace/MEMORY.md` exists
- [ ] `~/.openclaw/workspace/memory/` exists
- [ ] Identity files have permissions `600`
- [ ] `memory/` has permissions `700`
- [ ] The gateway restarted cleanly
- [ ] The Claw can explain how it plans to work with you using real context from `USER.md`
- [ ] The Claw can show its boundaries and confirmation rules in a way that reflects `SOUL.md` and `AGENTS.md`

---

## Troubleshooting

**The Claw asks all the questions at once**
Ask it to follow the instruction file exactly and ask the questions in order. The goal is a guided setup, not a form dump.

**The Claw writes generic identity files**
Your answers were probably too abstract. Rewrite with concrete defaults, prohibitions, and current priorities.

**The Claw keeps showing tool output between questions**
Tell it: "Do not write interim notes or update memory files during this interview. Ask the remaining questions in plain chat and write the file once at the end." Day 2 works better when the interview feels like a conversation, not a running audit log.

**The verification answers are generic**
Check that the files were written to `~/.openclaw/workspace/` and that the gateway was restarted after creation.

**The names do not match across files**
Have the Claw update the files so the same user name and Claw name appear consistently in `SOUL.md`, `USER.md`, and `AGENTS.md`.

**You want to revise the tone later**
Start with `SOUL.md`. Most personality drift comes from vague or conflicting instructions there.

---

[← Day 2 Learn](./learn.md) | [Day 3: Connect a Channel →](../day-03-connect-a-channel/build.md)
