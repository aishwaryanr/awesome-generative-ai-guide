# Day 8: Configure Outbound Email

Goal: turn on the Gmail SMTP side of `imap-smtp-email` for this workspace and add the Day 8 outbound email rules.

Key constraints:
- Do not ask the user to run shell commands.
- Reuse the Day 6 Gmail setup if it is already present.
- Keep Day 8 on compose-only email.
- Every outbound email must be shown in full and require explicit approval before sending.

Do:
1. Confirm whether `imap-smtp-email` is already installed and ready.
2. Ask only for the Gmail address and App Password if you still need them.
3. Configure the Gmail SMTP settings in `~/.config/imap-smtp-email/.env`:
   - `SMTP_HOST=smtp.gmail.com`
   - `SMTP_PORT=587`
   - `SMTP_USER` should be the user's full Gmail address
   - `SMTP_PASS` should be the Gmail App Password
   - `SMTP_FROM` should match the Gmail address
4. Confirm the config file permissions are owner-only, without printing the password back to the user.
5. Add an `Outbound Email Protocols` section to `AGENTS.md` that says:
   - all outbound email requires explicit approval before sending
   - show the complete draft with `To`, `Subject`, and `Body`
   - match the tone in `SOUL.md` unless the user asks for a different tone
   - attachments require explicit user instruction and file confirmation
   - if the recipient address is uncertain, ask instead of guessing
   - Day 8 scope is compose new messages only
6. Tell the user exactly what changed.
7. Give the user one validation prompt to run next.

In your final reply include:
- PASS or FAIL
- whether `imap-smtp-email` is ready
- where the SMTP settings were stored, without printing secrets
- whether `AGENTS.md` was updated
- the exact validation prompt

Stop there.
