# Day 2: Finalize Identity Setup

Follow these instructions exactly. Your goal is to lock down the identity files, restart the gateway, and verify that the new identity loaded correctly.

Before doing anything, confirm these files exist:

- `~/.openclaw/workspace/SOUL.md`
- `~/.openclaw/workspace/USER.md`
- `~/.openclaw/workspace/AGENTS.md`
- `~/.openclaw/workspace/MEMORY.md`
- `~/.openclaw/workspace/memory/`

If any are missing, stop and report what is missing.

---

## 1. Set File Permissions

Set these permissions:

- `SOUL.md`: `600`
- `USER.md`: `600`
- `AGENTS.md`: `600`
- `MEMORY.md`: `600`
- `memory/` directory: `700`

After changing permissions, read them back and report the result.

---

## 2. Restart the Gateway

Run:

```bash
openclaw gateway restart
```

Report whether the restart succeeded.

---

## 3. Verify the Identity Loaded

Run both verification prompts through the active OpenClaw instance:

### Test 1

```text
What do you know about me?
```

Expected result: the response includes real context from `USER.md`, such as the user's name, role, and current focus.

### Test 2

```text
What are your rules?
```

Expected result: the response reflects the Claw's name, prohibited behaviors, tone, and confirmation protocol from `SOUL.md` and `AGENTS.md`.

If the responses are generic, report that the workspace files may not have loaded and say to check the configured workspace path in `openclaw.json`.

---

## 4. Final Report

Report the status of each item as PASS or FAIL:

- `SOUL.md` exists
- `USER.md` exists
- `AGENTS.md` exists
- `MEMORY.md` exists
- `memory/` exists
- identity files are `600`
- `memory/` is `700`
- gateway restart succeeded
- "What do you know about me?" used real user context
- "What are your rules?" used real identity rules
