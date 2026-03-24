# Day 6: Finalize and Verify Inbox Setup

Verify that the Day 6 Gmail inbox setup is present and usable.

Day 6 is IMAP-only. Do not expect SMTP to be configured yet.
Do not require `imap-smtp-email` to report `ready` if that status depends on SMTP being present.
Do not look for a standalone `imap-smtp-email` shell executable on PATH.

Do not reinstall or rewrite anything in this run unless the user explicitly asks.

Report PASS or FAIL for:

- Gmail IMAP settings present in `~/.config/imap-smtp-email/.env`
  Check only the IMAP side: `IMAP_HOST`, `IMAP_PORT`, `IMAP_USER`, `IMAP_PASS`
- config file permissions are owner-only
- `imap-smtp-email` skill is installed or otherwise available to OpenClaw for Day 6 inbox reading
- `email-triage` created
- `Email Security Protocols` present in `AGENTS.md`
- recurring morning Gmail cron job created
- cron job schedule set to the chosen morning time
- cron job timezone set correctly

Stop when the verification report is complete.
