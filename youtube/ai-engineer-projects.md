# AI Engineer Projects That Get You Hired

[Watch on YouTube](https://www.youtube.com/@aish_reganti) · 2026-10-08
<!-- TODO: replace with the video URL -->

## In this video

- **What gets an AI engineer hired**: projects that show the decisions you made, the architecture trade-offs, and how close your build is to what companies ship, instead of a list of keywords like RAG, agents and MCP
- **6 projects in increasing order of difficulty**: a document intelligence pipeline, enterprise search, a conversation intelligence system, a sovereign AI engine, a multi-step autonomous agent, and a project you build backwards from the companies you want to join
- **The tool stack for each project**: the framework, models, databases, and deployment options on screen for every build
- **The datasets to start with**: CUAD, EnterpriseRAG-Bench, AMI, BillSum, PubMedQA, and tau2-bench
- **The evals for each project**: precision and recall per field, factuality and context recall, word error rate and LLM judges, cost and throughput against a frontier API, and pass^k for agents
- **What to put on your resume**: the decisions from each build that interviewers ask about

## The 6 projects

Each project lists the dataset to start from, the tool stack shown in the video, and the decisions to put on your resume.

### Project 1: Document intelligence pipeline

*Difficulty 1 of 5.* Takes messy documents in on one end and returns clean structured records as JSON on the other. Document processing is one of the most common workflows inside any company.

- **Dataset**: [CUAD](https://www.atticusprojectai.org/cuad), a collection of commercial contracts with labels, so you can measure from the start
- **Tool stack**: [LlamaIndex](https://www.llamaindex.ai) or the OpenAI / Anthropic SDKs for the document workflow, [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/) or [Docling](https://docling-project.github.io/docling/) for parsing, [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) for extraction, [Pydantic](https://pydantic.dev) for schema validation, [DuckDB](https://duckdb.org) as the analytics store, [FastAPI](https://fastapi.tiangolo.com), [Docker](https://www.docker.com) with [Cloud Run](https://cloud.google.com/run), and [Arize](https://arize.com)
- **Evals**: precision and recall scored per field, broken down by field type
- **What it proves to a hiring manager**: which documents you chose, the performance-against-cost trade-off between OCR-style and multimodal parsers, which categories the model handled well and which it did not, and how you handled schema validation

### Project 2: Enterprise search

*Difficulty 2 of 5.* A question answering system with a citation back to the document each answer came from, built for company scale instead of as a generic RAG chatbot. Ingestion, chunking, embeddings, a vector database, keyword, semantic and hybrid retrieval, reranking, and enforced citations.

- **Dataset**: [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), a simulated company with synthetic Slack, email, Jira, Confluence and Drive data, and around 500 questions written against it
- **Tool stack**: [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) or the OpenAI / Anthropic SDKs, [OpenAI text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large) for embeddings, [Qdrant](https://qdrant.tech) for dense and sparse hybrid search, [Cohere Rerank 4.0 Pro](https://docs.cohere.com/docs/rerank) for reranking, FastAPI, Docker with Cloud Run, and Arize
- **Evals**: factuality for generation, context recall and context precision for retrieval, and P95 and P99 latency
- **What it proves to a hiring manager**: your chunking decisions and why you made them, the embedding model you picked, the vector database parameters you looked at, how you enforced citations, and how evals guided each choice

### Project 3: Conversation intelligence system

*Difficulty 3 of 5.* Takes a conversation and generates a structured summary with the decisions and action items. Covers transcription, diarization (who spoke when), a structured record, and optional real-time processing and speech synthesis.

- **Dataset**: [AMI](https://groups.inf.ed.ac.uk/ami/corpus/), around 100 hours of real recorded meetings with human transcripts, speaker labels and written summaries
- **Tool stack**: [Deepgram Nova-3](https://deepgram.com) for transcription, [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) for batch diarization, [ElevenLabs Eleven v3 Conversational](https://elevenlabs.io) for speech synthesis, [LiveKit Agents](https://docs.livekit.io/agents/) as the voice framework, FastAPI, LiveKit Cloud with Docker containers, and Arize
- **Evals**: word error rate, diarization error rate, LLM judges with a calibrated rubric, and for real time, time to first output, time to first word, words per second, and time to first audio
- **What it proves to a hiring manager**: your model choice, how you handled overlapping speakers, how you evaluated a summary that has no single right answer, and the batch against real-time decision

### Project 4: Sovereign AI engine

*Difficulty 4 of 5.* A summarization or question answering system that runs on an open-weight model on your own hardware. You baseline a small open model against a frontier API, then decide between fine-tuning, quantization, and inference optimization.

- **Dataset**: [BillSum](https://arxiv.org/abs/1910.00523), US congressional bills paired with reference summaries, with [PubMedQA](https://pubmedqa.github.io) as the harder version
- **Tool stack**: [Unsloth](https://github.com/unslothai/unsloth) for fine-tuning, [Hugging Face PEFT](https://github.com/huggingface/peft) for LoRA adapters, [vLLM](https://docs.vllm.ai) for running open-weight models, FastAPI, Docker on an on-prem GPU or private GPU VM, and Arize
- **Evals**: quality, P95 latency, cost per 1,000 calls, and throughput, compared against the frontier API
- **What it proves to a hiring manager**: why you fine-tuned at all, which method you picked, what quantization cost you, and where the volume crossover sits

### Project 5: Multi-step autonomous agent

*Difficulty 5 of 5.* A ReAct-style agent that reasons, calls a tool, reads the result, and decides again over multiple steps, with tracing, guardrails, and a human sign-off before irreversible actions.

- **Benchmark**: [tau2-bench](https://github.com/sierra-research/tau2-bench), customer support tasks across retail, airline and telecom, each with its own tools and policy rules
- **Tool stack**: [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) for state, tools and human approval, the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for tool integration, [Temporal](https://temporal.io) as an optional durable workflow layer, FastAPI, Docker with Cloud Run, and Arize
- **Evals**: task success rate, tool accuracy, escalation rate, and pass^k
- **What it proves to a hiring manager**: where the human sits, what evals you built, what guardrails you put in, and how you debugged the traces

### Project 6: Working backwards

You pick 2 or 3 companies you are targeting and build something specific for them, using their public job listings and published engineering work to work out what they are likely building. It takes more effort than the other 5, and it is how you stand out among candidates who all built the same thing.

- **Data**: a synthetic dataset, or a public one close to the company's domain
- **Tool stack**: the [Greenhouse Job Board API](https://docs.greenhouse.io/job-board.html) and [Lever Postings API](https://github.com/lever/postings-api) for public job listings, LangGraph or the OpenAI / Anthropic SDKs, company engineering blogs as research sources, FastAPI, Docker with Cloud Run, and Arize

#### The prompt from the video

Paste this into an agent that can browse, such as Claude Code or Codex, and swap in the companies and roles you're targeting:

```
I'm applying to these companies: Palantir, Ramp, Airbnb.

The roles I'm targeting are Applied AI Engineer, AI Engineer, Applied Scientist, and Forward
Deployed AI Engineer.

Go and look up their current openings for those roles, and look up what their engineering teams
have published publicly, so engineering blogs, talks, open-source repos.

Then do 4 things, per company.

1. From their business and their job listings, work out which AI use cases they're most likely
   building right now. Be specific about the problem, not the technology. Say which part of the
   listing or the business led you to each one.

2. From what their engineers have published, identify what they've said is hard, slow, or
   unsolved. Quote the line where they say it. If a company has published little, say so rather
   than filling the gap.

3. Propose 2 projects per company that I could build in under 2 weeks, and that mirror those
   problems closely enough that somebody on that team would recognise them. For each one:
   - the problem it mirrors, and where in their material you got that from
   - what I would measure to show it works
   - the one sentence I would put in an email to the hiring manager

4. For each project, share public datasets that are close to the same problem, so I can build on
   real data instead of theirs. For each dataset tell me what it contains, whether it comes with
   labels, and what I'd have to generate synthetically if it doesn't. Then tell me what to focus
   on while building it, so which signals matter, what usually goes wrong with this kind of
   problem, and what would tell me the system is actually working.

Finally, tell me if any project would work for more than one of these companies, so I can build
once and send it to several.

Rules: nothing that takes longer than 2 weeks. If their public material doesn't support a claim,
say so instead of inventing it.
```

## Resources

- [Artificial Analysis](https://artificialanalysis.ai), for comparing models on performance across different tasks
- [Arize](https://arize.com), the observability tool in every project's stack
- [LlamaIndex](https://www.llamaindex.ai), [LlamaParse](https://developers.llamaindex.ai/llamaparse/parse/), and [Docling](https://docling-project.github.io/docling/), for document workflows and parsing
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) and [Pydantic](https://pydantic.dev), for structured extraction and schema validation
- [DuckDB](https://duckdb.org), [FastAPI](https://fastapi.tiangolo.com), [Docker](https://www.docker.com), and [Cloud Run](https://cloud.google.com/run)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) and the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), for agent state, tools and approvals
- [Temporal](https://temporal.io), for durable background workflows
- [OpenAI text-embedding-3-large](https://developers.openai.com/api/docs/models/text-embedding-3-large), [Qdrant](https://qdrant.tech), and [Cohere Rerank](https://docs.cohere.com/docs/rerank), for the enterprise search stack
- [Deepgram](https://deepgram.com), [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), [ElevenLabs](https://elevenlabs.io), and [LiveKit Agents](https://docs.livekit.io/agents/), for the voice stack
- [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face PEFT](https://github.com/huggingface/peft), and [vLLM](https://docs.vllm.ai), for fine-tuning and self-hosted inference
- [Qwen3](https://github.com/QwenLM/Qwen3) and [Gemma](https://deepmind.google/models/gemma/), small open-weight models for the baseline in Project 4
- [Greenhouse Job Board API](https://docs.greenhouse.io/job-board.html) and [Lever Postings API](https://github.com/lever/postings-api), for public job listings
- [LevelUp Labs](https://levelup-labs.ai/)
- [The Nuanced Perspective (newsletter)](https://thenuancedperspective.substack.com)
- [LevelUp Labs education](https://levelup-labs.ai/education)
- [Awesome Generative AI Guide](https://github.com/aishwaryanr/awesome-generative-ai-guide)
- [My courses on Maven](https://maven.com/aishwarya-kiriti)

## Sources

- Gartner, *"Gartner Data & Analytics Summit 2026 London: Day 2 Highlights"*, 12 May 2026, for the statistic that 70% to 90% of enterprise data is unstructured. [gartner.com](https://www.gartner.com/en/newsroom/press-releases/2026-05-12-gartner-data-and-analytics-summit-london-2026-day-2-highlights)
- Hendrycks et al., *"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"*, 2021, the contract dataset in Project 1. [arxiv.org](https://arxiv.org/abs/2103.06268)
- Sun et al., *"EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge"*, 2026, a synthetic enterprise corpus with 500 questions, used in Project 2. [arxiv.org](https://arxiv.org/abs/2605.05253)
- The AMI Meeting Corpus, 100 hours of meeting recordings including scenario and naturally occurring meetings, used in Project 3. [groups.inf.ed.ac.uk](https://groups.inf.ed.ac.uk/ami/corpus/)
- Kornilova and Eidelman, *"BillSum: A Corpus for Automatic Summarization of US Legislation"*, 2019, used in Project 4. [arxiv.org](https://arxiv.org/abs/1910.00523)
- Jin et al., *"PubMedQA: A Dataset for Biomedical Research Question Answering"*, 2019, the harder option in Project 4. [arxiv.org](https://arxiv.org/abs/1909.06146)
- Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs"*, 2023, for QLoRA training adapters through a frozen 4-bit base. [arxiv.org](https://arxiv.org/abs/2305.14314)
- Barres et al., *"tau2-Bench: Evaluating Conversational Agents in a Dual-Control Environment"*, 2025, the benchmark in Project 5. [arxiv.org](https://arxiv.org/abs/2506.07982)

## Transcript

_Transcribed from the recording. Will be replaced with the YouTube captions once they are generated._

If you're applying for AI engineering roles today, listing a bunch of keywords like RAG, agents and MCP, and building out starter projects, isn't what gets you hired. To stand out in today's job market, your projects have to show the decisions you made, the architecture trade-offs, and whether what you built is similar to what companies are shipping today. I've interviewed over 300 AI engineers during my time as an AI tech lead at AWS and for my own startup. These are the projects I wish I had seen on resumes. And they aren't random. A lot of research went into picking them.

I went through 2,000 AI engineer job listings to find the skills companies are hiring for. Then, to see how those skills get applied in real use cases, I distilled about 1,200 AI engineering blogs from top companies, along with the AI engineering work we do at my own company. I packed all of that into projects in increasing order of difficulty, so you end up with a portfolio that gets your resume picked, and by the time you're done you've built the skills interviewers are looking for. And while I go through each project, I'll put the tool stack for it on screen, so you can screenshot it as we go.

The datasets and the tool stack options are all in the GitHub repo too, so you can begin right after watching this. And a few of these projects ask you to compare models. One place you can do that, on performance across different tasks, is Artificial Analysis. This video was a lot of work, so I hope you like it.

So let's start with the first one, which is a document intelligence pipeline. Now, the reason I'd start here is that document intelligence is one of the most common workflows any company lives with. According to Gartner, somewhere between 70 and 90% of all company data is unstructured. So it's sitting in contracts, claims, invoices, and every other kind of document, and all of it has to become structured before anything can actually use it. Companies are building this everywhere right now. What you're building here takes messy documents in on one end, and gives you clean structured records on the other, as JSON.

You can do this with any set of documents you already have, but there's also a dataset I recommend called CUAD that you can get started with. CUAD is which is a collection of commercial contracts that comes with labels, so you can measure how well you're doing from the start. Here are the steps you'll follow while building this. The first thing is extraction, which is getting the right fields out. You'll try a few parsers and see which one holds up. There are OCR-style models, and there are multimodal models that take the document directly.

They behave differently and they cost differently, so you're making a call about performance against cost, and you should be able to explain the one you made. And if you go the model route, you'll also learn how to prompt these things properly. Because documents don't write anything formally. The same field shows up phrased 5 different ways across 5 documents, and your prompt has to survive that. A lot of documents also carry their own vocabulary, so you'll need good context engineering to make sure those terms are understood during parsing.

Then schema validation, which is checking that the right fields came back in the right shape, and that the model didn't hallucinate one. Now, every decision I've just described, which parser, which model, how much context to send, where validation goes, you make through evals. And this is going to come up in every project in this video, because working out what to measure is a skill in its own right. So for each one I'll tell you what kind of evals you'd be building. For this project, you take a small part of your dataset and get the labels for it, which are the correct outputs of parsing.

And the ones you'd look at here are precision and recall, scored per field. Precision is, of the fields that were parsed out, how many were right. Recall is, of the fields that were in the document, how many it managed to parse. Then break both down by field type. Understanding which fields or which categories are harder to parse is the kind of insight that shows an interviewer you actually understood the problem. Now, the decisions you made here are what you want on your resume, and what you'll get asked about in interviews.

The documents you chose, the model trade-off you made, which categories the model did well on and which it didn't, and how you handled schema validation. Screenshot these, because you'll have answers to all of them once you've actually gone through it.

Now that you can pull structured data out of a pile of documents, let's look at another problem every company has. Companies are sitting on enormous amounts of internal data. HR policies, support tickets, engineering docs, project wikis, months of Slack threads and email. So searching across all of that is one of the key things AI can actually do better, instead of somebody opening 5 different apps to find one answer. So the second project is an enterprise search system. What you're building is a question answering system with a citation back to the document the answer came from.

And if you're hearing this, you're probably thinking this is yet another RAG application. But we're going to look at it very differently. You're building it for company scale, not a generic RAG chatbot. The dataset that you can use here is EnterpriseRAG-Bench, which simulates a company with synthetic internal data, so Slack, email, Jira, Confluence, Drive, with around 500 questions written against it. So, what do you actually build? The first piece is an ingestion system. That's taking all of those documents and getting them into a format you can actually retrieve from later. And it has 3 parts to it, and you're making a decision at each one.

The first is chunking, which is how you split a document up before you store it. Chunking can range from very simple options to more complex ones, and which one works depends on your data, so you'll try a few and work out what makes the most sense. The second is embedding, which is turning each chunk into a vector, a list of numbers that captures what the text means, so that you can search by meaning. And the third is storing those vectors in a database. Then comes retrieval, and this is where you decide what kind of search you're running. You'd usually start with something as simple as keyword search, which matches exact terms.

Then, depending on your data, you'd play around with semantic search, which matches meaning. Hybrid search runs both together and combines the results, which is what most real systems end up doing. And for more complex use cases there's reranking, where you take the top retrieved chunks and change their order using a reranking model. Then you make sure the system enforces citations, because that's what makes it reliable. Now, one additional thing you can add to this project is deployment, so somebody else can actually use what you built. Which means you also have to start thinking about latency.

And all of these choices are things you figure out with evals. For a RAG-based system there are typically 2 kinds: evals for generation, and evals for retrieval. On the generation side, one you'd commonly use is factuality, which checks whether the answer is actually supported by the context that came back, or whether the model filled it in. On the retrieval side, context recall, which is whether the chunks you pulled back contained what was needed, and context precision, which is how much of what you pulled was relevant and whether the useful chunks ranked near the top.

And then latency at P95 or P99, which just means the time that 95 or 99 out of every 100 requests come back within. These are what you track instead of the average, because an average hides the latency your slowest users actually experienced. So it tells you what your slower users are actually sitting through. For all of these options, evals are your guiding line.

Now, for your resume and your interviews, it's the decisions you made along the way, so your chunking decisions and why you made them, which embedding model you picked, what vector database parameters you looked at, how you enforced citations, and how you used evals to guide all of those decisions.

So far we've handled documents and text. Now let's move to another modality that's getting popular inside companies, which is processing voice. Because every company runs sales calls, support calls, customer interviews, standups, handovers between shifts, and a lot of decisions get made in those conversations. So understanding what happened on those calls and being able to act on it matters. And Gartner expects conversational AI to be handling more than half of all enterprise contact centre volume by 2027. So the third project is a conversation intelligence system.

Imagine what you're building takes a conversation and generates a structured summary out of it, so the decisions and the action items. And remember, this is not as simple as generating a transcript. Voice behaves very differently from anything you've handled so far. At any scale you're working against time and budget, real conversations have several people talking over each other, and you have to work out what a good summary even looks like. The dataset that you can use here is AMI, which is around 100 hours of real recorded meetings.

It includes both scenario-based and natural meetings. Use recordings with human transcripts, speaker labels and written summaries, so you have ground truth for the words, for who said them, and for the summary. Now, the build. The first piece is transcription. There are several speech-to-text models and they behave differently, so you'll compare a few. Accuracy moves with audio quality, accents, how many people are on the call, and whether the conversation uses vocabulary the model has never seen. The second piece is turning that transcript into an attributed, structured record.

Most transcription models give you the words but not who said them, so you need diarization, which is working out who spoke when. Usually that runs as a separate step you merge back on, though some models do it as the audio arrives. Either way it fails separately from transcription, so you can end up with a near-perfect transcript that has the wrong person attached to half of it. Then you pull the record out of that. Speech doesn't behave like the documents you worked with in the first project. People interrupt, they trail off, they reverse themselves, so deciding what counts as a decision is the work here.

Now, one way to make this more advanced is to make it run live. Once you're processing audio in chunks as it arrives, you're committing to the start of a sentence before you've heard the end of it. And if you want to go a step further, generate the summary back out as speech. That closes the loop, because you've built both directions, and those 2 halves together are what an actual voice AI system is. And once again, the choices come down to evals, and the ones you'd reach for here are different again. Typically you'd start with word error rate on the transcript, and diarization error rate on the attribution.

Then the summary, which is different from the first 2 projects, because there's no single correct answer to compare against. So you'll learn about LLM judges here. You write a rubric that says what a good record looks like, then you calibrate the judge, which means checking it scores the way a person would. And once you go real time, you need a second group of them, which are operational rather than quality. Common ones are time to first output, which is how long before anything comes back at all. Time to first word. Words per second, which is whether you're keeping pace with how fast people actually talk.

And time to first audio, if you added the spoken summary. Those are what decide whether the thing feels natural to sit in front of. On the resume and in interviews, it's your model choice, how you handled overlapping speakers, how you evaluated something with no right answer, and the batch against real-time call.

Now, every project so far, you've built with closed-source models over an API. Somebody else hosts the model, you send your data to them, and you pay per request. So for the fourth project, you're going to build with open-source models and host them yourself. And this is picking up a lot of momentum in CXO circles right now. It's called sovereign AI, which essentially means running models inside the company's own infrastructure. Some categories of company should be doing this already, because data residency requirements leave them no choice, so banks, hospitals, defence.

And a lot of others are thinking about moving, because frontier models keep getting more expensive and they want control over their own data and their own costs. It's become a board-level topic in 2026, so it's a good one to pick up. So that's the fourth project, and it's a sovereign AI engine. So you're building a summarization or question answering system that runs on an open-weight model, on your own hardware. A dataset you could start with is BillSum, which is US congressional bills paired with reference summaries written for them.

And if you want a harder version, PubMedQA is biomedical research questions, and it ships a large training split meant for fine-tuning with a smaller expert-labelled split for testing. Both are public datasets you can use to practise before working with private documents in those domains. And they're domains that genuinely need sovereign AI, which is why I picked them. They're also small enough that you can work with them on your laptop. Now, the build. The first step is picking an open-weight model and getting a baseline. So Qwen 3, Gemma 4, any of the small open models in that range. They run on a laptop, or on free Colab.

Then you run the same task through a frontier API with a good prompt, and now you have 2 reference points: the performance your open model starts at, and the upper bound you're trying to reach. And the performance of your system is what drives everything you do next, because you're using it to decide between 3 optimizations. The first is fine-tuning. LoRA and QLoRA are the methods to look at, and they're parameter-efficient, meaning they train a small number of extra weights instead of the whole model, which is what keeps this on hardware you already have.

And it's worth trying both, since QLoRA trains adapters through a frozen 4-bit base and reduces the memory needed for fine-tuning. Then you measure again. Did the fine-tune beat the baseline, and did it beat what you'd have got by spending the same effort on the prompt? Because those are different questions and they often have different answers. The second is quantization, which stores the weights in fewer bits so they take up less memory. Whether it runs faster depends on the hardware and the inference engine. And what you're measuring here is how quantization impacts performance, because the impact isn't evenly spread.

Some parts of the task hold up and others fall away early. And the third is inference optimization. Because a local model serving one request at a time is easy, but serving 50 people at once is a different problem, and it's where you'll run into continuous batching, which groups incoming requests together on the fly, and KV caching, which stores the intermediate state so the model doesn't recompute everything on every token. There's real theory underneath both, and it's a good place to go deeper if that's an area you want to understand.

You can deploy this one too, the same way you did before, behind an API and containerized so it runs the same on any machine. So by the end, what you have is a comparison. Your model against the frontier API, on 4 things. And sometimes the API still wins. Document which cases those are. And again, for the resume and the interviews, it's why you fine-tuned at all, which method you picked, what quantization cost you, and where the volume crossover sits.

Now, autonomous agents are the space moving fastest right now. These are systems that make decisions and act on higher-order goals, rather than workflow-style systems that answer a question and hand the work back. And companies are going to deploy them carefully, and the reason is that an agent can take actions that can't be rolled back. So that's the part you'll be paying the most attention to while you build this. It's the frontier of what these systems can do, so it's a good one to learn, and it gives you a different perspective on what can go wrong. So the fifth one is a multi-step autonomous agent.

So you're building an autonomous agent that works over multiple steps to reach an objective, where it decides what to do next over and over, and every decision changes the ones after it. A dataset you could start with is tau2-bench, which is a customer support dataset, and customer support is inherently a multi-step, autonomous agent-style use case. So you're building a support agent with a set of tools it can call, and it goes back and forth with a simulated customer over several turns until it reaches a resolution.

And it covers a few domains, so retail, airline and telecom, and each one has its own tools and its own policy rules the agent has to work inside. Now, the build. You'd build a ReAct-style agent, which loops. So it reasons about what to do next, takes an action by calling a tool, looks at what came back, and decides again. You give it access to the tools, MCP is the standard way to expose those now, and you give it the goal. And because it loops, logging every one of those steps becomes really important. In the earlier projects latency told you the system was slow.

Here you want the trace of the whole conversation, every reasoning step and every tool call, so you can see which step is slow and which one is the bottleneck. And when a run fails somewhere in the middle, the trace is the only thing that tells you where. Guardrails matter more here than in any of the earlier projects, because this is the one that can take an action you can't take back. So you decide what the agent is allowed to touch, and where a human has to sign off before it does something irreversible. And the evals here look different from anything you've done so far.

The ones that usually come up for agents are task success rate, tool accuracy, and escalation rate, which is how often it gave up and handed back to a human. And then pass^k, which is the common one for agents working toward a goal. Because a single successful run tells you very little, since these systems aren't deterministic and the same task can succeed once and fail the next 3 times. So pass^k asks whether it succeeds every time across k attempts, and running the same task repeatedly is the difference between a demo and an evaluation. You can deploy this one as well, except an agent isn't a request that returns in a second.

It runs for minutes, so you're handling queued jobs and runs that need to be resumed. And an advanced thing you can do here is build it as a multi-agent pipeline, where one agent holds the plan and another executes against it. And the thing to get out of that is understanding why a multi-agent system was needed at all, what you decided and what you traded off, rather than building one for the sake of it. And for your resume and your interviews, it comes down to where the human sits, what evals you built, what guardrails you put in, and how you debugged the traces.

Now, those 5 projects are built to be useful to any employer. The last one works the other way around. You pick 2 or 3 companies you're actually targeting, and you build something very specific for them. It's harder to do than the others. It also gives you a much higher chance of your resume getting picked up. Because companies publish what they're currently working on. And given a company's background and the kind of use cases they deal with, it's not hard to extrapolate what they're likely building. And you can use AI agents to do a lot of that work for you. Their job listings are public as well.

Companies publish their own boards through applicant tracking systems, and those expose an endpoint anyone can read. So this is a prompt you can use for exactly that, and I'll walk you through it. Now, the one thing you'll have an issue with here is getting the data, because you don't have theirs and you never will. So you use a synthetic dataset, or something public that's close to that domain, and that works fine. So this one takes more effort than the other 5. But if you have the time to do it, definitely do it, because it's how you stand out in a crowd of candidates who all built the same thing.

So hopefully that's given you some good ideas to get started with. Do go and check out the GitHub repository, because that's where all the datasets I've spoken about are, along with the tool stack options for each project. And if you like these kinds of well-researched ideas from somebody with real experience, do subscribe. It really helps spread the word in a world full of noise.
