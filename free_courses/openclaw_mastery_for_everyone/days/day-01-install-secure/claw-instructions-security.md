# Day 1: Security Verification

This file contains the security checks for a fresh OpenClaw deployment on Hostinger. The Hostinger one-click template runs OpenClaw inside a Docker container. Everything below accounts for that environment.

Run each section in order. Report the result of each check as PASS, FAIL, or EXPECTED. For anything that fails, fix it and re-run that check to confirm.

If prompted for a risk posture, choose **VPS Hardened**: deny-by-default, minimal surface, tightest settings.

---

## 1. OS Info

Report the operating system, kernel version, and whether the process is running as a non-root user.

**Expected:** Debian 12 or 13, or Ubuntu 22.04/24.04. The process should be running as a non-root user (typically `node`). If security patches are pending, apply them.

A supported, patched OS is the foundation. Running as non-root limits what damage a compromised process can do.

---

## 2. Open Ports

List every port currently listening, along with the address it is bound to.

```
ss -tlnp
```

**Expected:**
- Port `18789` on `127.0.0.1` (the OpenClaw gateway)
- A few additional OpenClaw internal ports on `127.0.0.1`
- One port on `0.0.0.0`, typically in the 60000+ range

That `0.0.0.0` port is the Control UI. Hostinger's proxy forwards traffic to it, which is how the web chat works. This is expected and safe. Hostinger manages access at the host level.

The gateway itself should never be on `0.0.0.0`. If it is, anyone on the internet can reach it directly. Fix it by setting the gateway address to `127.0.0.1` in `~/.openclaw/openclaw.json` and restarting.

---

## 3. Firewall

Check whether a firewall is active inside the container.

**Expected:** No firewall tools found (no UFW, iptables, or nft). This is normal for Docker. Hostinger manages the firewall on the host. Mark this as EXPECTED.

On a bare VPS, you would want a firewall active. Inside a managed Docker container, the host handles that layer.

---

## 4. OpenClaw Security Audit

Run the built-in security audit:

```
openclaw security audit --deep
```

**Expected:** Zero critical failures. A few warnings are normal on a fresh deploy:
- Reverse proxy headers not trusted (trusted proxies not set)
- Permissive tool policy on extension plugins
- Unpinned plugin npm versions

These are low risk for personal use and informational at this stage. What matters is no critical failures.

---

## 5. Gateway Configuration

Read `~/.openclaw/openclaw.json` and verify these values:

| Setting | Expected value |
|---------|---------------|
| Gateway mode | `local` |
| Gateway address | `127.0.0.1` |
| Authentication mode | `token` |
| dmPolicy | Either set to a restrictive value or not set. If not set, the default is owner-only, which is safe. |
| groupPolicy | Either set to `disabled` or not set. If not set, the default is no group chats, which is safe. |

Token authentication means every request needs a valid token. The DM and group policy defaults keep unknown senders and group chats locked out until explicitly allowed.

If any value is wrong, fix it in `openclaw.json`.

---

## 6. File Permissions

Check the permissions on the credentials directory:

```
ls -ld ~/.openclaw/credentials
```

**Expected:** `drwx------` (700), owner only. If it shows `644` or `755`, fix it:

```
chmod 700 ~/.openclaw/credentials
```

API keys live in this directory. No other user on the system should be able to read them.

---

## 7. Channels

List which messaging channels are currently enabled and whether any have stored credentials.

**Expected:** Several channels (Telegram, WhatsApp, Discord, Slack, etc.) may show as enabled but not configured. This is fine. No channel should have credentials stored yet. Channels are set up on a later day.

If any channel already has credentials, report where they came from.

---

## 8. Web Search

Check whether web search is enabled. If it is, disable it.

**Expected:** Disabled after this step. The Hostinger template may enable web search by default. In this course, every capability gets added deliberately. Web search has not been introduced yet.

---

## 9. Disable the Heartbeat

Set the heartbeat interval to `0m` in `openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "heartbeat": {
        "every": "0m"
      }
    }
  }
}
```

Confirm the change by reading the value back.

The heartbeat runs scheduled tasks on a loop. With no identity or channel connected, those tasks produce nothing useful. It gets enabled on a later day once the Claw has an identity and a channel.

---

## 10. Restart and Final Verify

Restart the gateway and run both checks one more time:

```
openclaw gateway restart
openclaw doctor
openclaw security audit
```

**Expected:** `openclaw doctor` shows all checks passing. `openclaw security audit` shows no critical failures. If anything fails after the restart, report what failed and fix it.

---

## Summary

After completing all sections, the following should be true:

- [ ] OS is supported and patched, process running as non-root
- [ ] Gateway on `127.0.0.1` with token auth
- [ ] DM and group policies are restrictive
- [ ] Credentials directory is `700`
- [ ] No channels have stored credentials
- [ ] Web search is disabled
- [ ] Heartbeat is set to `0m`
- [ ] `openclaw doctor` passes
- [ ] `openclaw security audit` shows no critical failures

Report the final status of each item.
