# Getting Your API Key

OpenClaw works with multiple AI providers. You only need one to get started. Pick whichever provider you prefer, follow the steps below, and you'll have a key in a few minutes.

---

## What's an API Key?

An API key is a unique code that lets OpenClaw communicate with your AI provider (OpenAI, Google, Anthropic) on your behalf. It's separate from your regular account login. Having a ChatGPT, Gemini, or Claude subscription does not automatically give you an API key. You need to generate one specifically.

Think of it this way: your subscription lets *you* use the chatbot. An API key lets *your Claw* use the AI model. They're billed separately and managed in different places.

The one exception is OpenAI, which lets you connect OpenClaw directly to your ChatGPT subscription via OAuth (more on that below).

---

## A Note on Costs

OpenClaw is an always-on agent. Unlike a chatbot you open and close, it runs continuously, processes messages, executes scheduled tasks, and calls your AI provider throughout the day. That means token costs can add up quickly, especially with more capable models.

For this course, we recommend setting aside **$20 to $30** for API usage if you're paying per token. That's a comfortable upper bound assuming you're experimenting and playing around as you learn. If you already have a **ChatGPT Plus or Pro subscription**, OpenAI lets you connect OpenClaw via OAuth so your usage draws from your subscription instead of per-token billing. That's the most predictable way to manage costs.

Many people in the OpenClaw community run local models on Mac minis or other dedicated hardware to avoid API costs entirely. That's a valid path, but it comes with a larger learning curve. If you've never set up OpenClaw before, we recommend starting with an API key or subscription so you can focus on learning how OpenClaw works. You can always add local models and dedicated hardware later once you're comfortable.

---

## OpenAI (GPT)

OpenAI is the only provider that lets you use a subscription instead of paying per token. If you already have a ChatGPT Plus or Pro plan, this is the most cost-effective way to run OpenClaw.

### Option A: Use your ChatGPT subscription (recommended)

If you have ChatGPT Plus ($20/month) or Pro ($200/month), you can connect OpenClaw via OAuth. Your Claw's usage draws from your subscription limits instead of per-token billing. This is covered in the Day 1 setup on Hostinger.

### Option B: Use an API key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Click **"Sign up"** and register with email, Google, Microsoft, or Apple
3. Verify your email address
4. **Verify your phone number** via SMS (required)
5. In the left sidebar, click **"API keys"**
6. Click **"Create new secret key"** and give it a name
7. Copy the key immediately. It's only shown once
8. Go to **Settings > Billing** to add a payment method and load at least $20 in credits


---

## Google (Gemini)

Google has the most generous free tier for getting started. No credit card, no phone verification, and you get ongoing free access to capable models. The trade-off: it's pay-per-token only, with no subscription option for OpenClaw.

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Accept the Terms of Service for Generative AI
4. Click **"Get API Key"** in the left sidebar
5. Select your project (or let Google create a default one for you)
6. Click **"Create Key"**
7. Copy the key and store it somewhere safe


---

## Anthropic (Claude)

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Click **"Sign Up"** and register with Google, Microsoft, Apple, or email
3. Verify your email if you signed up with email
4. In the left sidebar, go to **Settings > API Keys**
5. Click **"Create Key"** and give it a name
6. Copy the key immediately. It starts with `sk-ant-` and is only shown once
7. Go to **Settings > Billing** to add a payment method and load credits


---

[← Back to Course Overview](README.md)
