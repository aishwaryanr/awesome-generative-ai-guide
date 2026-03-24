# OpenClaw Mastery for Everyone: Brand Voice Guide

This guide is specifically for course content: README files, day chapters, and the Big Book.

---

## Who Is Speaking

Aishwarya Naresh Reganti and Kiriti Badam. Practitioners who have actually implemented AI systems at scale, not commentators. The course is written in first-person plural ("we") or second-person ("you"), never a detached third-person narrator. It feels like a knowledgeable friend walking you through something, not a textbook.

---

## The Core Tone

**Direct. Practical. A little playful in the right places.**

The course title is "for Everyone" — that's a promise, not a tagline. The writing makes complexity accessible without dumbing it down. Readers should feel respected, not lectured.

---

## Chapter Content Voice

### The Right Register

The chapters are written like someone who has done this themselves and is sharing what they learned, not presenting a curriculum. Not a professor. Not a thought leader. A practitioner who ran into the same walls the reader is about to run into.

**Sounds right:**
- "Most tutorials tell you to skip the security setup and come back to it. We disagree. Here's why."
- "SOUL.md sounds philosophical. It isn't. It's the most practical file in your entire setup."
- "You don't need to understand what vectorization is. You just need to know what it makes possible."

**Sounds wrong:**
- "In this chapter, we will explore the foundational concepts of..."
- "It is important to note that security must be configured before..."
- "As we embark on this journey into OpenClaw..."

### Specificity Over Generality

Every claim needs grounding. Don't say "this is powerful" — show what specifically it does.

❌ "SOUL.md is a powerful way to personalize your assistant."
✅ "SOUL.md is the reason your assistant eventually sounds like it knows you, instead of like a help desk bot."

❌ "Memory helps OpenClaw remember things across sessions."
✅ "Without a memory system, every conversation with your bot starts from scratch. With one, it notices when you keep mentioning a project you haven't shipped yet."

### Security-First Without Fear-Mongering

Security is Day 1, not an afterthought. But the tone isn't "beware" or "danger." It's practical: here's why we do this, here's what happens if you don't.

❌ "Warning: failing to configure security could expose your system to serious threats."
✅ "We do the security setup on Day 1 because once the bot has access to your email and calendar, it's much harder to lock things down retroactively. Five minutes now saves a headache later."

### Explain the Why, Not Just the What

Every step in `prompt.md` gets a one-line explanation in the README of why it exists. Readers follow instructions better when they understand the reasoning.

❌ "Run: `chmod 700 ~/.openclaw/credentials`"
✅ "Lock the credentials file so only your user account can read it. (`chmod 700 ~/.openclaw/credentials`). OpenClaw stores your API keys here — you don't want other processes on the machine touching it."

### Short Paragraphs

Three to four sentences max per paragraph. If a concept needs more, break it into two paragraphs or use a visual example. Walls of text lose people.

### Concrete Before Abstract

Introduce a concept with what it does before explaining what it is.

❌ "HEARTBEAT.md is a configuration file that defines scheduled background tasks for the OpenClaw daemon."
✅ "HEARTBEAT.md is how you make the bot proactive. Instead of waiting for you to ask, it wakes up on a schedule and checks things on its own. You define what it checks and when."

---

## README / Navigation Voice

The main README and day-level READMEs use a slightly warmer, more welcoming register. This is the front door of the course — it should feel alive.

### Emojis: Sparingly, For Navigation Only

Emojis go in section headers on READMEs (📚 Course Days, 🎓 Certification, 🔗 Resources). They do not go in chapter body content. They are navigational aids, not decoration.

### Course Day Titles

Titles should be punchy and specific. They describe what the learner builds or gains, not what the topic is.

❌ "Day 4: Scheduling and Automation"
✅ "Day 4: Make It Proactive"

❌ "Day 6: Email Integration"
✅ "Day 6: Tame Your Inbox"

Each title gets one descriptive line that tells the reader exactly what they'll have by end of day. Specific outcome, not vague promise.

❌ "Learn to connect email and manage your inbox with AI."
✅ "Set up email triage, draft-only replies, and injection protection. By end of day, your inbox is managed."

---

## What to Never Write

### AI Slop (applies to all content)
- "The uncomfortable truth is..."
- "Here's what I learned..." (use "Here's what we found" or just state the thing)
- "Let that sink in."
- "Here's the thing..."
- "Delve into", "Embark on", "Unlock the potential of"
- "Game-changing", "Revolutionary", "Cutting-edge"
- "In today's fast-paced world..."
- "It's important to note that..."

### Preachy Instruction Voice
- "You should always..."
- "It's crucial that you..."
- "Never forget to..."
- "Make sure to always..."

Replace with: "We do X because Y" or "X means Z" — state the practice and the reason, don't issue commands.

### Contrast/Negation as a Writing Crutch

Never define something by what it isn't. Never open with "is not", "doesn't", "not just", or "not X but Y". Say what the thing IS directly.

**The pattern to avoid:**
- "OpenClaw is not a chat interface." → Just say what it is.
- "What OpenClaw actually is under the hood, not just what it does." → Remove "not just what it does" entirely.
- "Why security is a Day 1 problem, not a Day 10 cleanup task." → "Why security setup has to happen before any integrations are connected."
- "Not because setup is interesting, but because skipping it..." → "Skipping it is why most AI assistant projects get abandoned."
- "The gateway doesn't start when you open an app and stop when you close it. It's always on." → "The gateway runs continuously."
- "A chat tool waits for you. OpenClaw doesn't wait." → Just describe what OpenClaw does. Cut the comparison.
- "Your laptop doesn't qualify." → "The gateway needs a machine that stays on 24/7."
- "This is not cosmetic." → "The name matters more than it seems." Or just cut the opener.
- "Not the entry plan." → The positive version (which plan to choose) already makes this implicit. Cut it.

**The rule:** If you catch yourself writing "not" or "doesn't" at the start of a sentence or bullet, or using the pattern "X, not Y" anywhere in a sentence, stop and find the positive version. Positive descriptions are shorter, more confident, and easier to read. The reader does not need to hold "not X" in their head to understand what you mean by "Y".

Specific pattern to catch: "X, not Y" mid-sentence is just as banned as "not X" at the start.
❌ "The separation is architectural, not arbitrary." → ✅ "The separation has three concrete reasons behind it."
❌ "It is intentional, not accidental." → ✅ "It is intentional." (The positive version already implies this.)

### Manufactured Problems and False Contrast

Never position OpenClaw by making everything that came before it sound broken or inadequate. ChatGPT works. Cursor works. Other tools work. OpenClaw solves a different problem, not a better version of the same one.

The swing to the opposite extreme is just as dishonest as the contrarian framing. "The tools were capable, but the friction was unsustainable" is manufactured drama. It implies everyone was struggling before OpenClaw arrived. They weren't.

❌ "People kept building AI workflows and abandoning them. The friction of using them wasn't sustainable."
✅ "A different use case emerged: people wanted an AI already running in the background, not one they had to go visit."

❌ "ChatGPT leaves you doing all the work to make it useful."
✅ "ChatGPT works well for on-demand tasks. OpenClaw is for tasks that need to run on their own."

**The rule:** State what OpenClaw does. If comparison is necessary, state what the other tool does too — accurately, without inflating the problem it supposedly fails to solve.

### Contrarian Positioning ("Most People Skip X")

Never use meta-commentary to position the content as special. This includes:
- "Most tutorials skip this."
- "Most developers do X, but we're going to do Y."
- "What most people miss..."
- "Unlike other tools..."
- "The thing everyone gets wrong..."
- "Most people think X. They're wrong."

This pattern is hollow. It signals that the writer couldn't lead with the actual point, so they invented a foil instead. Just say the thing directly.

❌ "Most OpenClaw tutorials move straight from installation to integrations. They skip the part that matters: the default settings ship for a demo environment."
✅ "OpenClaw ships with demo defaults. On a VPS with a public IP and live credentials, those defaults leave a door open."

❌ "Most tutorials tell you to skip the security setup and come back to it. We disagree."
✅ "Security setup happens on Day 1 because once the bot has access to your email and calendar, it's much harder to lock things down retroactively."

**The rule:** If you catch yourself writing "most [people/tutorials/developers]", stop. The reader doesn't care what most tutorials do. They care about what to do. Cut the framing and lead with the substance.

### Overpromising
- "By the end of this course, you'll have an AI assistant that runs your entire life."
- "This will save you hours every day."

Replace with specific, honest claims: "By end of Day 3, you'll have a morning brief landing in your Telegram every day." What you can deliver, stated plainly.

### Em-Dashes
Never. Not once. Restructure the sentence or use a comma, colon, or period instead.

### Exclamation Marks in Prose
Not in body content. They are fine in course README section headers (Happy Learning!) following the existing course format, but nowhere in chapter text.

### Name-Dropping
Do not reference specific people by name as examples (e.g., "Peter Yang named his bot Zoe", "Jordan Lyall's setup"). The Go Deeper section may link to external guides by author name for attribution, but chapter body content should never use real people as illustrative examples. Let the concept stand on its own.

---

## The "Go Deeper" Section

Each day has an optional Go Deeper section for technical learners. The tone here becomes more peer-to-peer — assumes comfort with terminals, config files, APIs. Still no jargon without explanation, but you can skip the hand-holding.

The framing: "If you want to go further than today's setup, here's what's worth reading." Not "advanced users only" — anyone can read it.

---

## "Created By" Attribution

Consistent across all files:

> *Created by [Aishwarya Reganti](https://www.linkedin.com/in/areganti/) & [Kiriti Badam](https://www.linkedin.com/in/sai-kiriti-badam/)*

Full name "Aishwarya Reganti" in the link text (not "Aish"). Kiriti's full name. Both LinkedIn links present.

---

## The Certification Tone

The certification should feel earned and a little fun, not corporate.

❌ "Congratulations. You have successfully completed the OpenClaw Mastery for Everyone certification program."
✅ "You finished all 10 days. Your lobster is fully grown. Take the assessment and claim your certificate."

---

## Selling LevelUp (In-Course)

The course is free. The Maven courses are paid. The upsell appears in two places only: the main README (Additional Resources section, with discount code) and the Big Book of OpenClaw (last section). It never appears inside a day chapter.

When it appears, it follows the existing pattern from the AI Evals course: matter-of-fact, no apology, no hard sell. "If you want to go deeper with us, here's where to do it."

---

## Aish's Feedback: What Works and What Doesn't

### Aish Hates
- **Generic content**: "I could get this from ChatGPT." Every claim needs a non-obvious insight, specific example, or real research behind it.
- **Paraphrasing the outline back**: If Aish gives an outline, the job is to research and flesh it out, not rephrase it at the same level of detail.
- **Contrarian positioning**: "Most tutorials skip X." "Most people do Y but we do Z." Cut the foil and lead with the substance.
- **Manufactured problems**: Making everything before OpenClaw sound broken or inadequate just to position OpenClaw as the solution.
- **Negation patterns**: "doesn't know your name", "is not", "not just", "X, not Y" anywhere in a sentence. Define things by what they ARE.
- **Doomsday scenarios**: Dramatizing a point by painting worst-case outcomes. State the risk plainly, then state the solution.
- **Incident-by-incident news reporting**: Don't list CVE names and researcher names like a security bulletin. Tell what categories of things go wrong and what the course does about it.
- **Security with no solution**: Listing problems without explaining the course's response is useless and anxiety-inducing.
- **External quotes and citations in the body**: "As Turing College said..." Readers don't care. State the insight directly.
- **Technical jargon for non-technical audiences**: Flask, ngrok, WebSocket internals — if a non-developer can't follow it, cut it or rephrase it.
- **Introducing too many terms at once**: Name things only when needed. Don't front-load lingo.
- **Using jargon before explaining it**: Every term that appears in a chapter must have been explained either earlier in the same chapter or in a previous day. "Heartbeat", "skill", "gateway", "compaction" — if the reader hasn't seen it defined yet, define it before using it. When a concept belongs to a later day, describe the behavior in plain language and add "Day X covers this in detail."
- **Founder/company news irrelevant to the learner**: What the creator is doing now doesn't help the reader build their agent.
- **Controversy and PR drama**: Keep focus on what the learner is building.
- **learn.md referencing build.md steps**: learn.md must stand alone. No "the last step in build.md is X."
- **"Go Deeper" before "Ready to Build"**: Order is: content → Ready to Build → Go Deeper.
- **Bland "Ready to Build" handoffs**: Don't just say "open build.md." Transition from what was learned and briefly preview what build.md actually does.
- **Dumping too many numbers and statistics**: A few specific, meaningful numbers. Not a stats dump that loses the narrative.
- **Sections that fragment rather than flow**: Fewer, better sections. Don't give every minor point its own heading.
- **Build steps in learn.md**: "Name your bot" and similar build actions don't belong in learn.md. Keep them in build.md.
- **Technical units readers can't visualize**: KB, characters, token counts mean nothing to most readers. Use word counts and page equivalents instead ("~500 words, about one page").
- **Jargon in section headings**: "The Boot Sequence Enforcer" when "The Operating Manual" says the same thing. Headings should be immediately clear to someone who has never seen the term before.
- **Presenting one tool's design as the only way**: If OpenClaw makes a specific architectural choice, acknowledge that other tools (Claude Code, Codex) do it differently. The reader should understand it is a design decision with tradeoffs.
- **Abstract reassurance instead of actionable advice**: "These are living documents" tells the reader nothing useful. "Tell the agent to update SOUL.md when it makes a mistake" is something they can actually do. Advice should include the specific action.
- **Stating technical facts without a verifiable source**: If a claim about how OpenClaw works can't be traced back to documentation or a specific source, do not state it with confidence. Say what you know and where it came from.
- **Inconsistency between diagrams, text, and tables**: If the diagram says two files reload every turn but the text says four, the reader loses trust. Every element in the same chapter must tell the same story.

### Aish Likes
- **Story that works as a standalone read**: Someone who never opens build.md should finish learn.md and feel they understand the day's concept.
- **Real, specific examples**: The $4,200 car deal, the inbox cleared overnight. Concrete beats abstract every time.
- **Aggregate patterns over individual incidents**: What kinds of things go wrong, not who got hurt on which date.
- **Explaining terms before using them**: ClawHub, skills, gateway — define on first use, don't assume knowledge.
- **Architecture before parts**: Explain how a system works as a whole before explaining each component individually.
- **The "one more thing" framing**: Introducing an unexpected element positively ("one more thing before the four files") rather than as a problem or caveat.
- **Diagrams and visual structure**: ASCII tables and flow diagrams to show how things interact. A visual beats three paragraphs of explanation.
- **The three-era framing**: Historical arc gives concepts a home and makes the "why now" clear.
- **Security sections that end with the solution**: State the problem, then explain exactly how the course addresses it.
- **Short paragraphs**: Three to four sentences. Break it up.
- **Balanced takes**: Tools before OpenClaw worked fine. OpenClaw solves a different problem, not a better version of the same one.
- **Research that surfaces non-obvious insights**: Something the reader wouldn't get from a 5-minute Google search.
- **"Ready to Build" as a transition**: Recap what was learned, preview what build.md does, then hand off.
- **Community links in Go Deeper with URL attribution**: Reference external resources with the author's name in the URL, not called out by name in the body text.
- **Positive framing throughout**: "Here is why this is good for you" rather than "here is what goes wrong without it."
- **Word counts and page equivalents**: "~500 words, about one page" is immediately useful. The reader can picture it. Always prefer these over KB or character counts.
- **Acknowledging design alternatives honestly**: "Claude Code puts everything in one file. Codex uses AGENTS.md. OpenClaw splits by role. There is no single right answer." Balanced, respectful of other tools, helps the reader understand the choice.
- **Comforting the reader about scope**: When introducing a subset of a larger system, say so. "OpenClaw has other config files too. These four are what you need to get started." Reduces overwhelm.
- **Actionable iteration advice**: Tell the reader exactly what to do when something goes wrong. "Say 'update SOUL.md to include this rule.'" Give them the words, not just the concept.
- **Honest expectations about the learning curve**: "Expect it to feel rough at first. Corrections will be frequent, then taper off over a week or two." Prepares the reader without scaring them.

### The Deepest Pattern

Every writing shortcut — contrarian positioning, negation framing, citing external names for authority, doomsday scenarios — appears when there is not enough substance to lead with. The shortcuts are a symptom of insufficient research or insufficient thought about what the reader actually needs to know.

The fix is always the same: research until there is something real to say, then say it directly. Strong substance needs no foil, no contrast, no manufactured stakes. It stands on its own.

---

## Reference Courses (for format)

- AI Evals for Everyone: `github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/free_courses/ai_evals_for_everyone`
- Agentic AI Crash Course: `github.com/aishwaryanr/awesome-generative-ai-guide/tree/main/free_courses/agentic_ai_crash_course`
