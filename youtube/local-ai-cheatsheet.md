# Local AI: the cheat sheet

The companion cheat sheet for [You need to learn Local AI in 2026](local-ai.md).

To estimate the model size and memory your own use case needs, use the
**[What do you want Local AI to do tool](https://localai.levelup-labs.ai/)**.

---

## What counts as local

Every AI product has 2 layers: the **model** (the brain, which decides) and the **harness** (the
software around it, which acts: tools, file access, memory, code execution).

For something to count as local AI, the model itself runs on your hardware. A desktop app that calls
a cloud model is still using cloud AI.

| Setup | Model runs | Harness runs | Local? |
|---|---|---|---|
| Claude Code or Codex with a provider's hosted model | Provider's cloud | Your machine | No |
| Claude Code or Codex pointed at a model on your machine | Your machine | Your machine | Yes |
| A local chat app such as Unsloth with an open model | Your machine | Your machine | Yes |

You need downloadable model weights and a license that permits your use. Local inference has no
provider token bill, and hardware and electricity still cost money.

---

## Why now

- **Cost and control.** Agents run for hours and use far more tokens than chat. Provider limits and
  access can change.
- **Better open models.** More capable models are available to download and run yourself.

---

## 3 questions before you choose a model

### 1. What should it do?

| Use case | Typical model size |
|---|---|
| Writing emails, summarizing documents | Small models, which run on a laptop |
| Coding copilots, Claude Code level work | 70 to 80 billion parameters or more for complex use cases |

"B" means billion parameters. Parameter count is a rough indicator of size, and size alone is not
quality. Test on your own work.

### 2. Will it fit in memory?

A back-of-the-envelope rule for a 4-bit model: **about 0.5 GB per billion parameters**.

| Model | Memory to hold the weights |
|---|---|
| 8B | about 4 to 5 GB |
| 30B | about 15 to 20 GB |

This is weights only. Add memory for context, the runtime, and the operating system. On a Mac, the
number that matters is unified memory. On Windows or Linux with a discrete GPU, it is the GPU's own
VRAM.

### 3. Will it be fast enough?

Fitting in memory does not guarantee speed. Generation speed (tokens per second) depends on your CPU
or GPU and on the inference software. Test with real prompts, longer context, and repeated agent
turns.

---

## What you can run, by memory

Approximate model sizes at 4-bit quantization, after allowing for the operating system and some
context. Treat the boundaries as estimates, not measurements.

### Apple silicon (unified memory)

| RAM | Model size that fits |
|---|---|
| 8 GB | up to 4B |
| 16 GB | 7B to 8B |
| 24 GB | around 14B |
| 32 GB | 20B to 32B |
| 48 GB | 30B to 40B |
| 96 GB | 70B to 120B |

### Windows and Linux laptops (GPU VRAM)

System RAM is not the limit here. The graphics card's own memory is. Anything larger spills into
system memory and slows down sharply.

| GPU VRAM | Model size that fits |
|---|---|
| 8 GB | up to 8B |
| 12 GB | around 14B |
| 16 GB | 14B to 20B |
| 24 GB | around 32B, with limited context |

---

## How companies think about it

| Why own the stack | What comes with it |
|---|---|
| Control over data | Upfront hardware costs |
| Control over access | Maintenance and uptime |
| High-volume workloads | Security responsibility |

Local inference can keep prompts on your machine. Tools and integrations may still send data
outside.

## The future is hybrid

- **Local or self-hosted models** for private, repetitive, or high-volume work.
- **Frontier cloud models** for harder tasks that need stronger capabilities.
- Route by task, and check results either way. For example, run Claude Code or Codex on one large
  hosted model and point the sub-agents at smaller local models.
