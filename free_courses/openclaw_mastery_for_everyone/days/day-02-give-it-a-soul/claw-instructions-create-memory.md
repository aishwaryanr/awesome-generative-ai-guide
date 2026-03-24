# Day 2: Create MEMORY.md

Follow these instructions exactly. Your goal is to ask the user the MEMORY.md setup questions in order, then create `~/.openclaw/workspace/MEMORY.md`.

Before asking anything, read these files if they exist:

- `~/.openclaw/workspace/SOUL.md`
- `~/.openclaw/workspace/USER.md`
- `~/.openclaw/workspace/AGENTS.md`

This file is private-session-only. It is the correct place for sensitive personal context that should not appear in group-safe files.

---

## 1. Ask the Questions in Order

Ask one question at a time. Wait for the user's answer before continuing. If the user skips a question, accept that and move on.

During this interview:

- ask the questions in plain chat
- do not run tools or write files between questions
- do not append notes to `memory/YYYY-MM-DD.md` while collecting answers
- hold the answers in the conversation, then write `MEMORY.md` once after the questions are complete

1. What is the most important thing you are working on right now that did not already come up in USER.md? This could be a personal goal, a side project, or something you are thinking about that is not strictly work.
2. What commitments do you have open right now, to yourself or to someone else, that you do not want to forget?
3. Is there anything about your personal situation that the Claw should know but that you would not want visible in a group chat? Only share what you are comfortable with.
4. What is something about how you operate that took you a while to figure out about yourself?

---

## 2. Write MEMORY.md

Use the answers to create `~/.openclaw/workspace/MEMORY.md`. Replace every placeholder. Do not leave bracketed placeholders in the file. If the user skips a question, write a short note like "Nothing yet. Will be updated over time."

Use today's actual date in `YYYY-MM-DD` format.

```md
# Memory

*Last updated: [TODAY'S DATE]*

## Current Context
[FROM QUESTION 1: what the user is focused on beyond their work priorities]

## Open Loops
Items committed to but not yet closed:
- [FROM QUESTION 2]
- [ADD MORE AS RELEVANT]

## Personal Context
[FROM QUESTION 3: sensitive context the user shared. If they skipped this, write "Nothing yet."]

## Patterns
Things that are true about how this person works:
- [FROM QUESTION 4]
- [ADD MORE OVER TIME]
```

Target length: under 100 lines. This is a curated reference, not a full transcript.

---

## 3. Confirm Completion

After writing the file:

- report where it was written
- summarize the open loops captured
- note any sections intentionally left sparse so they can be filled in later

Do not proceed to permissions or restart steps in the same run unless the user explicitly asks you to.
