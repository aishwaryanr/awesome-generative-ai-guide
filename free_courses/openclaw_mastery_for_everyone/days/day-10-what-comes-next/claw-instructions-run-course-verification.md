# Day 10: Run Course Verification

Goal: review the user's setup across Days 1 through 10 and return one optional completion code for the assessment form.

Key constraints:
- Do not ask the user to run shell commands.
- Inspect the setup yourself.
- Use the current visible setup as the source of truth.
- Do not present the code as cryptographic or authoritative. It is a simple completion summary.

Do:
1. Review the current setup day by day.
2. Score one point for each day that looks substantially complete.
3. Use these day-level checks:
   - Day 1: security hardening and a working Claw
   - Day 2: `SOUL.md`, `USER.md`, `AGENTS.md`, and `MEMORY.md` exist with real content
   - Day 3: at least one real channel is connected
   - Day 4: at least one real recurring cron job exists
   - Day 5: at least one installed skill is present and ready
   - Day 6: Gmail inbox reading is configured
   - Day 7: live web search is configured
   - Day 8: outbound email setup exists with approval rules
   - Day 9: a named `writer` agent exists and delegation is configured
   - Day 10: this final review was completed
4. Report PASS or FAIL for each day with one short reason.
5. Count the total score out of 10.
6. Generate one optional completion code in this exact format:
   `LUL-OC-[SCORE]OF10-[USERINITIALS]-[YYYYMMDD]`
7. Get `USERINITIALS` from the user's name if possible. If the user's name is missing, use `USER`.
8. End by reminding the user:
   - the code is optional
   - it belongs in the last question of the Google form
   - the form still works without it

In your final reply include:
- the total score
- the PASS or FAIL list for Days 1 through 10
- the optional completion code on its own line
- one short sentence on which day to revisit first if something failed

Stop there.
