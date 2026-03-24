# AGENTS.md

## Repo

- This repo is a Markdown course, not an application codebase.
- Keep the 10-day Hostinger + OpenClaw arc sequential and consistent across days.

## Files

- `learn.md`: concepts and mental model.
- `build.md`: learner-facing steps.
- `claw-instructions-*.md`: only the parts the learner hands to OpenClaw.

## Rules

- Prefer surgical edits. Keep the existing `learn.md` structure unless there is a real reason to change it.
- Keep `learn.md`, `build.md`, and the "What Should Be True After Day X" checks aligned.
- Make references to repo Markdown files real relative links.
- Assume learners interact through OpenClaw chat. Do not tell them to run shell commands unless shell access is explicitly part of the lesson.
- If `learn.md` mentions paths, CLI commands, or registry tooling, frame them as under-the-hood context unless the Claw is the actor.
- Size `build.md` as learner steps. Split inspect, decide, install, create, and verify when those are distinct phases.
- Default against one giant prompt that completes the whole day in one run. Use it only when the task is truly atomic.
- Use the fewest `claw-instructions-*.md` files needed. Inline short inspection, explanation, or one-turn check prompts in `build.md`.
- Keep `claw-instructions-*.md` short and high-signal: goal, key constraints, required output, stop point.
- Keep the instruction file brief, but make the artifact it asks OpenClaw to create as detailed as the lesson requires. A `SKILL.md`, `SOUL.md`, or similar target file can be much more explicit than the instruction that produces it.
- Keep learner-facing copy out of authoring context. State the constraint directly instead of narrating the hosting setup, container model, or course-production reasoning.
- Reuse existing workspace and session context before asking questions. Ask only for missing decisions.
- Use one mechanism per day unless the lesson is explicitly about comparison. For exact-time schedules, prefer cron over heartbeat.
- For channel or scheduled delivery, set explicit routing such as `to` when a stable recipient is known.
- For tool-integration days, stage the build as: understand the tool, configure the tool, then build one reusable skill on top of it.
- Mention Hostinger UI screens only when they help, and keep them secondary unless the UI is the lesson.
- End `build.md` with a short validation section, then quick wins. Validation proves setup; quick wins show payoff.
- Prefer one concrete example over a menu of options when the day is teaching a mechanism.
- Link official OpenClaw docs inline when `learn.md` explains product behavior.
- When a new skill is added, explicitly tell the learner to type `/new` before testing it.

## Writing

- Follow [`BRAND_VOICE.md`](./BRAND_VOICE.md).
- Be direct, practical, and concrete.
- Explain why a step exists.
- Keep security matter-of-fact.
