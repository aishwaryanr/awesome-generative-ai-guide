# Day 6 Build: Tame Your Inbox

This is the user-facing guide for Day 6. Today you do the inbox flow in stages. First you create a Gmail App Password. Then you inspect a ClawHub skill. Then your Claw installs it, creates one small triage skill on top of it, adds the email safety rules, and wires the result into a morning Telegram cron job.

The whole day assumes a personal Gmail inbox. Google says App Passwords may be unavailable on work or school accounts, accounts using Advanced Protection, and accounts using 2-Step Verification only with security keys. For this lesson, keep it simple and use a personal Gmail account.

---

## What You Need Before Starting

- Day 1 complete: OpenClaw installed and secured
- Day 2 complete: identity files created and loading correctly
- Day 3 complete: Telegram connected and working
- Day 4 complete: a proactive workflow already exists
- Day 5 complete: you have already inspected and installed a ClawHub skill once
- Access to your Claw through the web chat
- Access to a personal Gmail account
- Ability to open your Google Account settings in a browser

---

## How To Run Day 6

Work through the files in this order:

1. create a Gmail App Password
2. inspect `imap-smtp-email` in chat
3. [`claw-instructions-install-imap-smtp-email.md`](./claw-instructions-install-imap-smtp-email.md)
4. [`claw-instructions-create-email-triage.md`](./claw-instructions-create-email-triage.md)
5. [`claw-instructions-finalize-inbox.md`](./claw-instructions-finalize-inbox.md)

This order matters. You inspect before install, keep the send side out of scope, then build one small skill on top of the shared Gmail connection.

For this day, stay on the same cron path you used on Day 4. It is the better fit on Hostinger for an exact-time morning delivery.

---

## Step 1: Create a Gmail App Password

Open [App Passwords](https://myaccount.google.com/apppasswords).

If Google sends you somewhere else first, turn on [2-Step Verification](https://myaccount.google.com/security) and come back to the App Passwords page. Google's current help page for this flow is [Sign in with app passwords](https://support.google.com/accounts/answer/185833).

Create a new App Password:

- App name: `openclaw-imap`
- Copy the 16-digit password Google generates

Google shows each App Password once. Keep it somewhere safe long enough to finish this setup.

---

## Step 2: Inspect `imap-smtp-email`

Copy and paste this into the OpenClaw web chat:

> Inspect `imap-smtp-email` from ClawHub and explain, in plain English, what it does, what Gmail credentials it needs, where it stores its config, what could be risky, and how we can keep Day 6 on the inbox-reading side only. Do not install anything yet.

You are checking two things here: whether the skill matches the name, and whether its behavior fits the boundary for today. Day 6 uses the Gmail reading side. Day 8 returns to the same skill for sending.

---

## Step 3: Install `imap-smtp-email`

After you are happy with the inspection, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-06-tame-your-inbox/claw-instructions-install-imap-smtp-email.md` and follow every step. Install `imap-smtp-email` for this workspace, configure Gmail inbox reading for Day 6, tell me where the config lives, and stop when the install report is complete.

That [instruction file](./claw-instructions-install-imap-smtp-email.md) tells the Claw to:

- install `imap-smtp-email` into this workspace
- ask you for your Gmail address and App Password if needed
- configure the Gmail IMAP side in `~/.config/imap-smtp-email/.env`
- leave the SMTP side for Day 8
- tell you where the skill and config live

After this step, type `/new` in OpenClaw before you continue.

---

## Step 4: Create `email-triage`

Copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-06-tame-your-inbox/claw-instructions-create-email-triage.md` and follow every step. Create `email-triage`, add the Day 6 email safety rules, create the morning Gmail cron job, tell me how to trigger it, and stop when the report is complete.

That [instruction file](./claw-instructions-create-email-triage.md) tells the Claw to:

- create the `email-triage` workspace skill
- keep the summary at sender, subject, category, and counts unless you request one specific email
- add `Email Security Protocols` to AGENTS.md
- create a recurring morning Gmail cron job

This is the layer that makes the inbox feel like yours. The shared skill gives your Claw Gmail access. `email-triage` gives it your rules.

After this step, type `/new` in OpenClaw before you continue.

---

## Step 5: Finalize and Verify

Copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-06-tame-your-inbox/claw-instructions-finalize-inbox.md` and follow every step. Verify the Day 6 Gmail inbox setup and report PASS or FAIL.

That [instruction file](./claw-instructions-finalize-inbox.md) tells it to:

- confirm the Gmail IMAP setup exists
- confirm `imap-smtp-email` is available for Day 6 inbox reading
- confirm `email-triage`, AGENTS.md, and the morning cron job are in place
- report the verification as PASS or FAIL

---

## What Should Be True After Day 6

- [ ] A personal Gmail App Password was created
- [ ] `imap-smtp-email` was inspected before install
- [ ] `imap-smtp-email` was installed from ClawHub for this workspace
- [ ] Gmail IMAP settings were stored in `~/.config/imap-smtp-email/.env`
- [ ] `~/.config/imap-smtp-email/.env` permissions are owner-only
- [ ] SMTP settings are still left for Day 8
- [ ] `email-triage` exists as a workspace skill
- [ ] AGENTS.md includes email security protocols
- [ ] A recurring cron job exists for the morning Gmail summary
- [ ] The cron job schedule matches your chosen morning time
- [ ] The cron job timezone matches your timezone
- [ ] Your Claw can return a structured Gmail triage summary
- [ ] Your Claw flags prompt-injection text instead of following it

---

## Troubleshooting

**You can't find App Passwords in your Google account**
Check the official Google help page: [Sign in with app passwords](https://support.google.com/accounts/answer/185833). The common blockers are missing 2-Step Verification, a work or school Google account, Advanced Protection, or 2-Step Verification set up only with security keys. For this day, switch to a personal Gmail account if needed.

**Gmail says the password is wrong**
Use the 16-digit App Password, not your regular Gmail password. If you already closed the Google dialog, generate a new App Password. Google only shows each one once.

**The skill can read Gmail, but the send side also looks configured**
Day 6 keeps SMTP out of scope. Ask your Claw to open `~/.config/imap-smtp-email/.env` and confirm that the `SMTP_` values are still absent. Day 8 is where those values get added.

**The summary is showing too much email body text**
Ask your Claw to tighten the `email-triage` skill so summaries stay at sender, subject, category, and counts unless you request one specific email.

**The morning summary does not arrive**
Ask your Claw to inspect the cron job's schedule, timezone, session target, and Telegram delivery target together. Most misses come from one of those four being wrong.

**You start getting duplicate morning summaries**
Ask your Claw to list the active cron jobs and look for an older morning-summary job that should be disabled or removed.

**The new skills do not seem active yet**
Type `/new` in OpenClaw before testing. Day 6 adds new skills, and a fresh session makes the triggers available cleanly.

---

## Validate It

Type `/new` in OpenClaw first.

Then ask your Claw:

```text
Tell me the morning Gmail cron job you just created: the schedule, timezone, session target, and where it delivers.
```

The answer should clearly name the daily time, your timezone, the session binding, and the Telegram destination.

Then ask your Claw:

```text
Scan my Gmail inbox and give me a triage summary for the last 48 hours.
```

The answer should:

- use the four categories
- show sender and subject for Urgent and Important
- keep full body text out
- stay on summarization instead of execution

Then run the injection test your Claw gives you in the finalize step.

---

## Quick Win

From Telegram, send:

```text
Check my Gmail and tell me only what needs attention today.
```

This is the Day 6 payoff. Your Claw reads the inbox noise, compresses it, and gives you the part that actually deserves your time.

---

[← Day 6 Learn](./learn.md) | [Day 7: Make It Research →](../day-07-make-it-research/build.md)
