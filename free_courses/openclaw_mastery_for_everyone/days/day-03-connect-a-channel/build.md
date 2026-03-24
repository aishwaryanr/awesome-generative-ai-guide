# Day 3 Build: Connect a Channel

This is the user-facing guide for Day 3. Today you connect Telegram so you can talk to your Claw from your phone.

The operational steps live in [`claw-instructions-connect-telegram.md`](./claw-instructions-connect-telegram.md). This file is for you. The instruction file is for your Claw.

---

## What You Need Before Starting

- Day 1 complete: OpenClaw installed and working
- Day 2 complete: identity files created and loading correctly
- Telegram installed on your phone or desktop
- Access to your Claw through the web chat

All configuration for this step lives in `~/.openclaw/openclaw.json`.

---

## Step 1: Start the Setup in Web Chat

Copy and paste the following message into the web chat:

> Read `https://raw.githubusercontent.com/aishwaryanr/awesome-generative-ai-guide/main/free_courses/openclaw_mastery_for_everyone/days/day-03-connect-a-channel/claw-instructions-connect-telegram.md` and follow every step. Ask me only for what you need, configure Telegram, verify it works, and stop when you're done.

[`claw-instructions-connect-telegram.md`](./claw-instructions-connect-telegram.md) tells the Claw to:

- collect your Telegram bot token and user ID one at a time
- update `~/.openclaw/openclaw.json`
- keep Telegram private by default
- restart the gateway
- walk you through pairing and a live test from your phone

At a high level, here is what you are doing:

- use BotFather in Telegram to create a bot and get its token
- give that token to your Claw when it asks
- follow the instructions in Telegram as the Claw walks you through pairing
- if anything feels unclear, ask your Claw questions in the web chat while you go

When the setup is complete, you should be able to text the bot from Telegram and get a response back.

---

## What the Claw Should Ask You For

During setup, expect the Claw to ask for:

- your Telegram bot token from BotFather
- your numeric Telegram user ID

It should then configure Telegram under `channels.telegram` in `~/.openclaw/openclaw.json` and keep the channel restrictive:

- `dmPolicy: "pairing"`
- `groupPolicy: "disabled"`

If it places Telegram under `plugins.entries.telegram`, that is wrong. Telegram is a built-in channel, not a plugin.

---

## Pairing

Once the Claw has configured Telegram and restarted the gateway, it should have you:

1. Find your bot in Telegram
2. Send `/start`
3. Approve the pairing request
4. Send a real message from your phone

Once that works, the channel is live.

---

## Validate It

This should be simpler than the quick wins. Use the web chat for one explicit verification, then look for the result in Telegram.

Ask your Claw in the web chat:

```text
Send me a short cheerful greeting with emojis on Telegram so I can confirm this channel is working end to end.
```

You should see that greeting arrive in Telegram within a few seconds. If it does, the channel is working.

---

## Two Quick Wins

Once you know Telegram is working, use it for something you would actually do from your phone.

**Quick Win 1**

```text
I'm on my phone. Keep this short: what are the 2-3 best things I can ask you to do over Telegram now that this channel is live?
```

**Quick Win 2**

```text
Let's make this practical. Give me 3 examples of the kinds of short messages I can send you during the day when I'm away from my laptop.
```

This is the moment it starts feeling real. Your Claw is now something you can text from your phone while moving through your day.

---

## What Should Be True After Day 3

- [ ] Telegram bot created through BotFather
- [ ] Bot token stored under `channels.telegram.botToken`
- [ ] Your Telegram user ID added to `allowFrom`
- [ ] `dmPolicy` set to `"pairing"`
- [ ] `groupPolicy` set to `"disabled"`
- [ ] Gateway restarted successfully
- [ ] Pairing completed from your phone
- [ ] You can message your Claw on Telegram and get a response
- [ ] The Claw answers on Telegram in a way that reflects your identity files and channel rules

---

## Troubleshooting

**The Claw says Telegram is a plugin**
It is not. Tell it to configure Telegram under `channels.telegram` in `~/.openclaw/openclaw.json`.

**The bot does not respond after setup**
Have the Claw check the gateway logs and verify the bot token was copied correctly, with no missing characters or extra spaces.

**Pairing does not work**
Make sure your numeric Telegram user ID is in `allowFrom` and that it is stored as a number, not a quoted string.

**The Claw responds on Telegram but sounds generic**
Check that the workspace path is still correct and that the identity files from Day 2 are loading.

**The token looks valid but Telegram still fails**
Ask the Claw to verify that the full BotFather token was copied on one line, including the numeric prefix before the colon.

---

[← Day 3 Learn](./learn.md) | [Day 4: Make It Proactive →](../day-04-make-it-proactive/build.md)
