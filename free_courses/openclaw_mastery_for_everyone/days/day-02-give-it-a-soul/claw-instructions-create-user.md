# Day 2: Create USER.md

Follow these instructions exactly. Your goal is to ask the user the USER.md setup questions in order, then create `~/.openclaw/workspace/USER.md`.

Before asking anything, read `~/.openclaw/workspace/SOUL.md` if it exists so the Claw name and user name stay consistent across files.
If `SOUL.md` already contains the user's name, reuse it. Only ask for corrections or missing details rather than collecting the same information from scratch again.

---

## 1. Ask the Questions in Order

Ask one question at a time. Wait for the user's answer before continuing.

During this interview:

- ask the questions in plain chat
- do not run tools or write files between questions
- do not append notes to `memory/YYYY-MM-DD.md` while collecting answers
- hold the answers in the conversation, then write `USER.md` once after the questions are complete

### Who

1. Confirm the user's name from `SOUL.md` if it is already there, then ask only for preferred pronouns and any correction to the name if needed. If the name is missing, ask for full name and preferred pronouns.
2. What city and timezone are you in?

### Contact

3. What is your primary email address? Do you have any rules about response time?

### Focus

4. What is your role?
5. What are the specific things on your plate right now? List the 2-3 items you are actively working on this week.

### Style

6. What should the default output format look like? Short sentences, detailed paragraphs, bullet lists, or something else?
7. Is there a formatting preference you feel strongly about? For example: no bullets in chat, match my message style, or keep things to one screen.

### Patterns

8. What are your working hours? Are there times you prefer not to be messaged?
9. Is there something important about how you work that a new assistant should know on day one?

---

## 2. Write USER.md

Use the answers to create `~/.openclaw/workspace/USER.md` with this structure. Replace every placeholder. Do not leave any brackets in the file.

```md
# User Profile

## Who
Name: [FULL NAME]
Pronouns: [PRONOUNS]
Location: [CITY], [TIMEZONE e.g., "UTC+5:30 / IST"]

## Contact
Email: [PRIMARY EMAIL]
[RESPONSE TIME RULES IF PROVIDED]

## Focus
Role: [JOB TITLE / DESCRIPTION]
Organization: [COMPANY OR "Independent"]

What I am working on right now:
- [SPECIFIC ITEM 1]
- [SPECIFIC ITEM 2]
- [SPECIFIC ITEM 3 if applicable]

## Style
[SPECIFIC FORMAT PREFERENCES FROM QUESTIONS 6-7. Write as direct instructions, not descriptions.]

## Patterns
Working hours: [FROM QUESTION 8]
[INSIGHT FROM QUESTION 9]
```

Target length: about 250 words.

Important rule: keep sensitive personal information out of `USER.md`. If the user volunteers details that belong in a private context, briefly note that those should go into `MEMORY.md` instead and continue.

---

## 3. Confirm Completion

After writing the file:

- report where it was written
- summarize the current role and active focus items you captured
- flag anything that may need a later update because it will go stale

Do not proceed to AGENTS.md in the same run unless the user explicitly asks you to.
