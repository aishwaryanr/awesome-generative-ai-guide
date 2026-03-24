# Day 8: Create `follow-up-email`

Goal: create a `follow-up-email` skill for this workspace.

Key constraints:
- Keep the skill in the compose-only path for Day 8.
- Always require approval before sending.
- Do not guess recipient email addresses.
- The email body must be normal human-readable plain text, not a serialized string.

Do:
1. Create `follow-up-email` as a workspace skill.
2. Make it trigger on requests like `send a follow-up to ... about ...`.
3. Write a detailed `SKILL.md` with:
   - frontmatter with name, description, and version
   - a short "What it does" section
   - a workflow that:
     - identifies the recipient from context or asks the user if the address is missing
     - drafts a short follow-up email under 150 words
     - uses a subject line shaped like `Follow-up: [topic]`
     - includes one clear next step or ask
     - renders the email body as real plain text with actual paragraph breaks
     - never outputs escaped newline sequences like `\n`, quoted JSON-style body strings, Markdown code fences, or stray prefix characters in the email body
     - presents the full draft for approval before sending
   - guardrails for uncertain recipients, attachments, formatting mistakes, and staying inside compose-only email
4. In the skill instructions, add a final self-check before approval:
   - verify the visible email body reads like a normal email
   - if the body contains escape sequences such as `\n`, leading quote marks, surrounding quotes, code fences, or other serialization artifacts, rewrite it into clean plain text before showing it to the user
5. Tell the user to type `/new` in OpenClaw before testing the new skill.

In your final reply include:
- PASS or FAIL
- where the skill was created
- the exact trigger phrase to test it
- one example prompt to run next

Stop there.
