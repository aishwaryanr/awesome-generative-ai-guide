# Frequently Asked Questions

---

### Do I need a Mac mini or any external hardware?

No. The entire course runs on a cloud VPS (Virtual Private Server). You rent one from Hostinger using their one-click OpenClaw template, and Day 1 walks you through it. All you need is a laptop or desktop with a browser and a Telegram account on your phone.

Some experienced users run OpenClaw on a Mac mini at home. That works well, and you can always port to hardware once you have learned how OpenClaw works. But most people get overwhelmed with the hardware setup requirements and end up never configuring their Claw properly. This course is designed to save you from that overwhelm: you learn the concepts and build a working setup first, on infrastructure that is ready in minutes.

🎁 Complete the course and score 10/10 on the certification assessment? You could walk away with a free Mac mini. You will learn more about it as you go through the course.

---

### How does the certification work?

The assessment has a few questions to check your understanding of OpenClaw concepts: identity files, security, the three eras of AI tools, skills, multi-agent systems, and more. There is also an optional question where your Claw verifies your hands-on setup and generates a unique code that you paste into the form.

You can earn the certificate by completing the questions. The optional Claw verification is there for those who want to prove their setup is fully operational.

Take the assessment here: [OpenClaw Mastery Assessment](https://docs.google.com/forms/d/e/1FAIpQLSeoR5wfheIkD0hCaf3eYmJ6s8aNMbylfJ00hi6djlkpIuF1FA/viewform)

---

### Why does the course start with security?

Because the features that make OpenClaw useful are the same features that create risk. An agent that runs continuously, reads your email, and sends messages on your behalf has a larger attack surface than a tool you open and close. The community learned this the hard way through uncontrolled agent actions, malicious skills on ClawHub, and prompt injection through email.

Day 1 locks down the gateway before anything else gets connected. Every day after that adds capability incrementally, with guardrails in place before each new integration goes live. This is deliberate. You understand what each capability does and where the risks are before you turn it on.

---

### Is my data private?

Yes. OpenClaw runs on a VPS that you control. Your conversations, emails, files, and API keys stay on your server. Nothing is routed through a third party beyond the AI provider you choose for model inference (Anthropic, OpenAI, or Google). The course does not use any shared infrastructure.

---

### Can I use any AI provider?

OpenClaw supports Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek, and local models. You pick the provider and model you want during setup and can switch later. The course recommends starting with a mid-tier model (Claude Sonnet, GPT-5.4, or Gemini Flash) for daily use, and explains when to upgrade or downgrade for specific tasks.

---

### How much does this cost to run?

Three costs to consider:

1. **VPS hosting**: About $25/month on Hostinger.
2. **AI API usage**: Depends on your provider and how much you use your Claw. Mid-tier models are significantly cheaper than top-tier. We cover which providers work well and how to reduce your costs in the [API key guide](getting-your-api-key.md).
3. **The course itself**: Free. Always will be.

---

### What if I am not technical?

Zero prior experience with servers, Docker, or AI agents is required. The course is AI-first: you read the concepts in `learn.md`, then hand the `build.md` file to an AI coding agent (Claude Code, Cursor, or Codex on Days 1 and 2, then your own Claw from Day 3 onward). The agent handles the technical setup. You focus on understanding what each piece does and why.

---

### Can I skip days or do them out of order?

Not recommended. Each day builds on the previous one. Day 3 requires the security setup from Day 1 and the identity files from Day 2. Day 6 requires the channel from Day 3. The progressive layering is the point: you verify each capability works before adding the next.

If you already have a running OpenClaw instance, you can skim the learn files for earlier days and focus on the builds for the days that cover what you have not set up yet.

---

### What if something breaks during a build?

Every `build.md` file has a troubleshooting section at the bottom covering the most common issues for that day. If you hit something not covered there, you have a few options:

- **Live sessions**: Join one of our scheduled sessions and ask directly.
- **OpenClaw community**: Browse questions and answers on the [OpenClaw community](https://docs.openclaw.ai).
- **Open an issue**: If you found a bug or a gap in the troubleshooting guide, [open an issue on this repository](https://github.com/aishwaryanr/awesome-generative-ai-guide/issues). We will add it to the troubleshooting section so it helps everyone who comes after you. Help us make this the best free course out there.

---

### I finished the course. Now what?

Use what you built. Talk to your Claw every day for a week. Notice where the tone is off, where the rules are too strict or too loose, where the memory is missing context. Then update SOUL.md, USER.md, or AGENTS.md to fix those specific gaps. After that, explore new integrations one at a time: Google Calendar, Obsidian, Slack, or more specialist agents. Day 10's learn file has a full list of next steps.

For ideas on what to build next, check out our [Best OpenClaw Resources by Category](best-openclaw-resources.md). It has use cases, community content, and the best guides we found for getting more out of OpenClaw.

---

[← Back to Course Overview](README.md)
