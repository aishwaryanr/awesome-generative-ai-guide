# Day 4: Create a Daily Reflection Cron Job

Create a recurring daily reflection using cron.

Goal:

- schedule the reflection at the user's chosen time
- bind it to the current session
- deliver it explicitly to Telegram
- leave the user with a clear summary of what was created

Before you start:

- confirm Telegram is already configured
- confirm `~/.openclaw/workspace/` and `~/.openclaw/workspace/memory/` exist
- reuse `USER.md`, `MEMORY.md`, and current-session context
- ask only for missing decisions
- do not write interim memory notes during setup

If a prerequisite is missing, stop and report it.

---

## 1. Gather the Decisions

- confirm the user's preferred reflection time
- reuse the known timezone unless it is missing, unclear, or outdated
- propose 2 or 3 short reflection-question options based on what you already know about the user
- ask the user to pick one option or tweak it

Keep the reflection short enough to answer from a phone.

---

## 2. Explain the Write Action

Before editing anything, say clearly that you are about to:

- create a recurring cron job which will deliver the message to Telegram explicitly
- offer to run it once if the user wants an immediate check, or mention that they can also use the `Cron Jobs` tab

Wait for explicit confirmation before creating the job.

---

## 3. Create the Cron Job

Create the job using the cron tool or `openclaw cron add`. Do not edit cron storage files directly.

Make sure it:

- runs daily at the user's chosen time, in their timezone
- stays bound to the current session
- delivers explicitly to Telegram using the known `to` target
- sends the chosen reflection prompt and saves the next reply to `memory/YYYY-MM-DD.md` under `## Reflection`

Keep the job prompt concise and avoid repeated follow-up nudges.

After creating the job, report:

- job name
- job ID
- cron schedule
- timezone
- Telegram delivery target

---

## 4. Final Report

Report PASS or FAIL for:

- recurring reflection cron job created
- schedule set to the chosen daily time
- timezone set correctly
- session bound to current
- Telegram delivery configured explicitly
- job details were reported clearly to the user

Stop when the report is complete.
