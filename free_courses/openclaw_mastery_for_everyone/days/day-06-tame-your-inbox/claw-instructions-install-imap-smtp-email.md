# Day 6: Install `imap-smtp-email`

Install `imap-smtp-email` into this workspace.

If the user has not already approved the install, ask for confirmation first.
If it is already installed and ready, report that instead of duplicating it.

Ask the user for:

- their personal Gmail address
- the Gmail App Password they generated for Day 6

Configure the skill for Day 6 with these constraints:

- use `imap.gmail.com` on port `993`
- use the user's full Gmail address as the IMAP user
- use Gmail IMAP settings for inbox reading
- store the config in `~/.config/imap-smtp-email/.env`
- do not echo the App Password back to the user
- leave the SMTP settings unset for now because Day 8 handles sending
- confirm the config file permissions are owner-only if the skill setup does not already enforce that

After install and configuration, tell the user:

- where the skill lives
- where the config file lives
- whether the config file permissions look correct
- that Day 6 configured Gmail inbox reading only
- that they should type `/new` in OpenClaw before trying newly added skills
- one exact test message they can use now to confirm Gmail reading works

Stop when the install report is complete.
