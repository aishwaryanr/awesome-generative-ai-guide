# Day 3: Connect Telegram

Follow these instructions exactly. Your goal is to connect Telegram as the user's first messaging channel, keep it private by default, and verify that it works from the user's phone.

Before doing anything:

- confirm `~/.openclaw/openclaw.json` exists
- read the current `channels.telegram` configuration if it exists
- if Telegram is already configured, report what is there and ask whether to replace it or verify it
- ask for one thing at a time
- never print the bot token back in full; if you need to confirm it, mask all but the last 4 characters

---

## 1. Collect The Required Values

Ask the user for these items in order:

1. whether they already created a Telegram bot with BotFather
2. the Telegram bot token
3. their numeric Telegram user ID

If they do not know their Telegram user ID, tell them to get it first, then continue once they have it. Guide them with appropriate instructions where required.

---

## 2. Explain The Write Action

Before editing anything, state clearly:

- you are about to update `~/.openclaw/openclaw.json`
- you will configure Telegram under `channels.telegram`
- you will keep `dmPolicy` as `"pairing"`
- you will keep `groupPolicy` as `"disabled"`

Then wait for explicit confirmation before writing the file.

---

## 3. Update The Configuration

Write or update Telegram under `channels.telegram` in `~/.openclaw/openclaw.json`.

The final Telegram configuration should include:

- `botToken`
- `allowFrom` with the user's numeric Telegram ID
- `dmPolicy: "pairing"`
- `groupPolicy: "disabled"`

Preserve all unrelated config.

If Telegram appears under `plugins.entries.telegram`, move it to `channels.telegram`. Telegram is a built-in channel.

After writing the file, read it back and report only:

- that `channels.telegram` exists
- the masked token
- `allowFrom`
- `dmPolicy`
- `groupPolicy`

---

## 4. Restart The Gateway

Run:

```bash
openclaw gateway restart
```

If the restart fails because of a config mistake, fix the file and try once more. Report the result.

---

## 5. Pair And Verify

Guide the user through these steps:

1. find the bot in Telegram
2. send `/start`
3. complete the pairing flow
4. send a real message from their phone

If it does not work, diagnose the most likely cause:

- bot token copied incorrectly
- Telegram user ID wrong or stored as the wrong type
- gateway restart did not apply cleanly
- Telegram configured as a plugin instead of a built-in channel

Stay with the user until the Telegram channel responds successfully, or report exactly what is still blocked.

---

## 6. Final Report

Report PASS or FAIL for each item:

- `~/.openclaw/openclaw.json` exists
- Telegram configured under `channels.telegram`
- bot token stored under `channels.telegram.botToken`
- user's Telegram ID added to `allowFrom`
- `dmPolicy` is `"pairing"`
- `groupPolicy` is `"disabled"`
- gateway restart succeeded
- pairing succeeded
- Telegram responded to a real message from the user's phone

Once that report is complete, stop.
