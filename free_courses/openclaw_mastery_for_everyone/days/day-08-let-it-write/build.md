# Day 8 Build: Let It Write

This is the user-facing guide for Day 8. Today you return to the Gmail skill from Day 6, turn on the send side, add outbound email rules, and create one small follow-up workflow on top of it.

The operational steps are split between this file and a few small instruction files. This file is for you. The instruction files are for your Claw.

---

## What You Need Before Starting

- Day 1 complete: OpenClaw installed and secured
- Day 2 complete: identity files created and loading correctly
- Day 3 complete: Telegram connected and working
- Day 4 complete: a proactive workflow already exists
- Day 5 complete: skills are working
- Day 6 complete: Gmail inbox reading is working through `imap-smtp-email`
- Day 7 complete: web search is working
- Access to your Claw through the OpenClaw web chat
- Access to the same personal Gmail account you used on Day 6

---

## How To Run Day 8

Work through the steps in this order:

1. inspect the send side of `imap-smtp-email` in chat
2. [`claw-instructions-configure-outbound-email.md`](./claw-instructions-configure-outbound-email.md)
3. [`claw-instructions-create-follow-up-email.md`](./claw-instructions-create-follow-up-email.md)
4. [`claw-instructions-finalize-outbound-email.md`](./claw-instructions-finalize-outbound-email.md)

This order keeps the setup legible. You inspect the existing shared skill first, then your Claw turns on SMTP, then you add one reusable workflow on top, then you verify both the approval path and the cancel path.

---

## Step 1: Inspect the Send Side of `imap-smtp-email`

Copy and paste this into the OpenClaw web chat:

> Inspect `imap-smtp-email` from ClawHub again, this time for outbound email. Explain in plain English what SMTP settings it needs, how the approval step should work, what could be risky, and how we can keep Day 8 on compose-only email. Do not change anything yet.

You are checking two things here: whether the skill is ready for the send workflow, and whether the Day 8 boundary is still clear. Today is compose-only with explicit approval before every send. Reply and forward stay for later.

---

## Step 2: Configure Outbound Email

After you are happy with the inspection, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-08-let-it-write/claw-instructions-configure-outbound-email.md` and follow every step. Reuse my Day 6 Gmail setup, add the SMTP side for Day 8, add the outbound email rules, tell me exactly what changed, and stop when the setup report is complete.

That [instruction file](./claw-instructions-configure-outbound-email.md) tells the Claw to:

- confirm `imap-smtp-email` is installed and ready
- reuse your Day 6 Gmail address and App Password if it still needs them
- add the Gmail SMTP settings to `~/.config/imap-smtp-email/.env`
- keep the config file owner-only
- add `Outbound Email Protocols` to `AGENTS.md`
- keep Day 8 on compose-only email with explicit approval before every send

The Day 6 App Password is the same credential you use here. Gmail App Passwords work for both IMAP and SMTP.

---

## Step 3: Create `follow-up-email`

After outbound email is configured, copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-08-let-it-write/claw-instructions-create-follow-up-email.md` and follow every step. Create `follow-up-email` for this workspace, tell me how to trigger it, and stop when you're done.

That [instruction file](./claw-instructions-create-follow-up-email.md) tells the Claw to create one custom workspace skill that:

- triggers on requests like `send a follow-up to ... about ...`
- finds or asks for the recipient address
- writes a short follow-up email in your Claw's voice
- formats the body as clean plain text with real paragraph breaks
- shows the full draft for approval before sending
- stays inside the Day 8 compose-only boundary

After this step, type `/new` in OpenClaw before you test the new skill.

---

## Step 4: Finalize and Verify

Copy and paste this into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-08-let-it-write/claw-instructions-finalize-outbound-email.md` and follow every step. Verify the Day 8 outbound email setup, give me the exact test messages to use, and report PASS or FAIL.

That [instruction file](./claw-instructions-finalize-outbound-email.md) tells it to:

- confirm the Gmail SMTP setup exists
- confirm `imap-smtp-email` is installed and ready
- confirm `Outbound Email Protocols` and `follow-up-email` are in place
- give you one exact approve test and one exact cancel test
- remind you to test from a fresh session after `/new`

---

## Validate It

Type `/new` in OpenClaw first.

Then ask your Claw:

```text
Send me an email with subject "OpenClaw Day 8 Test" and body "This is a test of outbound email from my Claw. Day 8 is working."
```

Your Claw should show you the full draft, including `To`, `Subject`, and `Body`, and wait for approval before sending. Approve it, then check your inbox and Sent folder.

Then run the cancel test your Claw gives you in the finalize step. The draft should stop at the approval gate and never be sent.

---

## Quick Win

From Telegram, send:

```text
Send a follow-up to myself about the OpenClaw Day 8 setup. Keep it short and friendly.
```

This is the Day 8 shift: your Claw is no longer just reading and summarizing. It can draft a real message for someone else, pause for review, and send it only after you approve it.

---

## What Should Be True After Day 8

- [ ] `imap-smtp-email` was inspected for the send workflow before any changes
- [ ] Gmail SMTP settings were added to `~/.config/imap-smtp-email/.env`
- [ ] the config file permissions are still owner-only
- [ ] `imap-smtp-email` skill installed and `ready: true`
- [ ] Outbound email rules added to AGENTS.md
- [ ] Test email sent to yourself and received
- [ ] Approval gate cancellation verified (no email sent on cancel)
- [ ] `follow-up-email` workspace skill created
- [ ] You started a fresh OpenClaw session with `/new` before testing the new skill
- [ ] `follow-up-email` was tested successfully

---

## Troubleshooting

**The Claw starts replying or forwarding instead of composing a new email**
Tell it that Day 8 is compose-only. Reply and forward are out of scope for this lesson.

**SMTP authentication fails**
Use the same Gmail App Password from Day 6. If Google rejects it, generate a new App Password and have your Claw update both the IMAP and SMTP values in `~/.config/imap-smtp-email/.env`.

**The Claw says the Gmail send settings are missing**
Ask it to inspect `~/.config/imap-smtp-email/.env` and confirm that `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, and `SMTP_FROM` are present.

**The draft sends without showing you the full email first**
Ask the Claw to inspect `AGENTS.md` and confirm that `Outbound Email Protocols` requires showing the full draft and waiting for approval before every send.

**The email body shows literal `\\n`, quote marks, or other odd characters**
Ask the Claw to update `follow-up-email` so the body is rendered as normal plain text before approval and before sending. The visible draft should read like a normal email, not a serialized string.

**The test email does not arrive**
Check spam first. If it is not there, ask the Claw to verify that `SMTP_FROM` matches your Gmail address and that `imap-smtp-email` still reports `ready: true`.

**The new skill does not seem active**
Type `/new` in OpenClaw, then test `follow-up-email` again.

---

[← Day 8 Learn](./learn.md) | [Day 9: Give It a Team →](../day-09-give-it-a-team/build.md)
