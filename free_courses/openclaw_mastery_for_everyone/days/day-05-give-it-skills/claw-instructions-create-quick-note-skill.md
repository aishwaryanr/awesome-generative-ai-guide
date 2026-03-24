# Day 5: Create `quick-note`

Create a workspace skill called `quick-note`.

If it already exists, update it carefully instead of duplicating it.
Before writing the file, tell the user what you are about to create and wait for confirmation.

The skill should:

- trigger on `note:`
- capture the text after `note:`
- classify it as an `idea`, `task`, `follow-up`, or `reminder`
- rewrite it into one short clean entry without changing the meaning
- append it to `memory/YYYY-MM-DD.md` with a timestamp and label
- if it implies future action, add a short item to the open-loops section in `MEMORY.md`
- reply with a short confirmation that says how it was categorized

After writing, tell the user:

- the final file path along with the contents of the SKILL.md file
- the exact trigger message to test it
- that they should type `/new` in OpenClaw before trying to use the new skill

Stop when the report is complete.
