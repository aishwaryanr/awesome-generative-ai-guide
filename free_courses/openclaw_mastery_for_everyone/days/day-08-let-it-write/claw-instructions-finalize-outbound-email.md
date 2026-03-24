# Day 8: Finalize and Verify Outbound Email

Verify that the Day 8 outbound email setup is present and usable.

Do not reinstall or rewrite anything in this run unless the user explicitly asks.

Report PASS or FAIL for:

- Gmail SMTP settings present in `~/.config/imap-smtp-email/.env`
- config file permissions are owner-only
- `imap-smtp-email` installed
- `imap-smtp-email` ready
- `Outbound Email Protocols` present in `AGENTS.md`
- `follow-up-email` created
- reminder to test one approved send and one cancelled send
- reminder that the user should test from a fresh OpenClaw session after typing `/new`

Stop when the verification report is complete.
