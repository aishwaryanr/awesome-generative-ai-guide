# Is RAG Dead?

[Watch on YouTube](https://www.youtube.com/watch?v=FJQTT2B-imk) · 2026-08-18

![What we cover in this video](images/is-rag-dead.png)

<!-- agenda slide -->

## In this video

- **What RAG actually is**: chunking, embeddings, retrieval, and why the retrieval half is where the real work happens
- **Why people say RAG is dead**: camp 1 context windows got huge, camp 2 one-shot RAG breaks, camp 3 vector databases are dead
- **Agentic retrieval**: search, inspect, rewrite, search again, synthesise, instead of one lookup
- **Beyond vector search**: keyword, SQL, knowledge graphs, direct file reading, and hybrid retrieval
- **The knowledge runtime**: citations, access control, conflict detection, audit trail, grounding
- **When to use what**: one-shot RAG, agentic, hybrid, or a full knowledge runtime

## Resources

- [LevelUp Labs](https://levelup-labs.ai/)
- [The Nuanced Perspective (newsletter)](https://thenuancedperspective.substack.com)
- [LevelUp Labs education](https://levelup-labs.ai/education)
- [Awesome Generative AI Guide](https://github.com/aishwaryanr/awesome-generative-ai-guide)
- [My courses on Maven](https://maven.com/aishwarya-kiriti)

## Sources

- Latent Space, *"RAG is Dead, Context Engineering is King"*, with Jeff Huber of Chroma, 19 Aug 2025. The phrase is the episode's headline, not a quote from him; in the episode he says he dislikes the term RAG. [latent.space](https://www.latent.space/p/chroma)
- Nicolas Bustamante, *"The RAG Obituary: Killed by agents, buried by context windows"*, 2025. [nicolasbustamante.com](https://www.nicolasbustamante.com/p/the-rag-obituary-killed-by-agents)

## Transcript

The internet can't seem to agree on RAG. Half say it's dead, while the other half say it's one of the most important things that you can learn in AI right now. So this video settles it once and for all. I will break down what RAG actually is, the three most popular reasons people keep calling it dead, and the design patterns that the best AI teams are using to quietly evolve RAG into a full knowledge runtime layer.

So here's everything we'll cover. You can take a screenshot and keep it, or the HD version is also available on my GitHub repository that's linked in the description.

I have been building these systems for several years now, first as a tech lead at AWS, and now in my own company. So everything that we discuss in this video is what I've seen firsthand on the field. So let's get started.

RAG stands for retrieval augmented generation, and it's pretty much this, right? Before an AI answers a question, it looks up a knowledge base of information. And that is because on its own, the model only knows what it was trained on. It doesn't know your company's documents, and its knowledge stops at its training date. And RAG is almost like the open book version of AI. Fetch the relevant pages, then answer.

So first things first, what actually is RAG? At its simplest, it's the step that runs in the gap between your question and the AI's answer. It goes and finds the right information even before the model responds.

So picture a real question. Someone asks, "How much vacation time do I actually get in this company?" And that answer lives in your company handbook, which the model has probably never read. So on its own, all it can do is guess. And RAG's job is to find that page before the model says even a single word.

Seems pretty simple, right? But the process itself is not that trivial. You can't just hand the model everything you've got. There's a limit on how much information it can take in at once, and that's usually called the context length or the context window of the model. And your handbook, your policies, years of documentation cannot fit all at once into the model, and that's where we go to the next step. You'll have to retrieve and pull all of the relevant information that can be given to your AI model so that it can come up with the right answer.

So how do we do that? The solution to that is that you break it up into small passages, usually called chunks, and then you pull back a single passage or chunk instead of the entire manual, depending on what the question is. So you can pretty much store all of these chunks of information, and depending on the kind of question that is being asked, you can retrieve the relevant chunks, maybe based on the keywords that are being used in the question, etc., right? So that is the whole process of chunking and retrieval.

Then you hit another problem, which is let's say an employee asked about vacation time, but the page might say paid time off, right? Like a synonym of the word you're looking for. So if there are no shared words, matching on words alone might not be enough.

And that is where RAG systems usually use this notion of vector-based searching or meaning-based searching. Each of these chunks is run through a model called an embedding or a vector model and converted into a list of numbers that capture the meaning or the semantics of these chunks. And generally, these numbers as a whole are called embeddings or vectors. Passages that mean similar things end up with similar numbers, and they're pretty much sitting close to each other. And you store all of them in a specific kind of a database called a vector database. Vacation time finds annual leave instantly, or any synonym instantly, because these numbers are derived based on meaning and not the exact keywords.

Everything we discussed so far is the first half of RAG, or retrieval augmented generation, which is retrieval, basically. The second part is generation, which is kind of easier compared to retrieval. The model takes that chunk and writes your answer from it, right? Whatever was retrieved in the retrieval phase.

So the whole point of RAG is that your AI stops guessing and starts working from the actual sources. The policy that changed last week, the document your team updated yesterday, the data that actually lives in your systems, and what usually AI models are not trained on.

In its original form, this is pretty simple and static. You index your documents once, and every question runs through a single lookup and gets one answer. This is usually called naive RAG or one-shot RAG. For knowledge that is stable and contained, a product manual or a benefits policy or a support FAQ that's not changing very frequently, this is genuinely enough, and it still works as of today.

And for a good amount of time, maybe starting 2023 and up till 2024, one-shot RAG on vector databases was pretty much everywhere. It was one of the first things anyone building with AI would learn. And then something happened in late 2024, and the tone pretty much flipped. And the same rooms that couldn't stop talking about RAG started calling it dead.

So the argument RAG is dead never gets settled because it's not really one argument. It's three, coming from three different groups or camps of people. And each camp is usually talking about something very different. So let's understand them one by one.

When generative AI models got super popular back in late 2022, they could only read a little text at once. For instance, a few thousand words, maybe roughly a few pages. Now, that gap is called the context length or context window, like we discussed before. So if you wanted to ask AI to maybe answer questions about your company's documents, you had to be very selective about what you fed it, and that was the problem that RAG was trying to solve, right?

But over the years, now we're in 2026, something significantly changed. The latest models can now take millions of words all at once, and that gave rise to the first camp, which is the context window argument. If the context window of an AI model is so large, why do we have to build a retrieval pipeline? Just dump your entire knowledge base, meaning all of your documents, along with the question as context to the model, and let the model sort it out.

Now, there are two problems with this argument, right? And both of them start to make sense not in demos, but actually on large-scale data.

The first is that a big context window doesn't necessarily mean that the model can perform accurately on the entire window. There's tons of research in the past that says that models tend to get lost when there is a lot of information that is stored in large context windows. This entire problem is called the lost in the middle problem, and there is evidence that models, although they promise large context windows, cannot reason through the entire length of them just because they get confused.

The second is the most practical one, which is cost. And this is the one that kind of shows up in practice, right? You literally pay for every word that you put into a model on every single question. And you're pretty much paying a large amount of cost when retrieval could have done the job for you. So even if context windows or models got 10 or even 20 times bigger tomorrow, you'd still want to send in relevant and small information so that you can save on cost and latency and all these kind of operational things.

So when someone says RAG is dead because context windows or models are getting bigger than ever, you now know why even though context windows get much bigger in the future, RAG might not be dead in the sense that the need for retrieval might not be completely eliminated, given that we need to be really thinking about operational costs and how we build these systems for large-scale databases.

Now, that leaves us with two more camps. So one of these camps says that the basic version of RAG, which is one-shot RAG, is pretty much over at this point. And the other says that the technology that it runs on is dead. And both of them are worth thinking about seriously if you're someone who's building RAG systems. So now let's get a little deeper into what camp 2 is saying.

Camp 2 is about the original version of RAG, which is one question, one lookup, and one answer. Real questions can get way more complicated than that can handle. And for a while, it was pretty much enough for most enterprises. But as questions got more complex, as models got smarter, one lookup was rarely enough.

Users would generally ask something with multiple parts. For instance, compare these options, summarize what changed in the quarter, brief me on something that I don't understand yet from multiple sources, and the system would come back with a half-baked answer because it would have only one opportunity to get the right documents.

Engineers and researchers started building manual workarounds around this, and it almost became a game of whack-a-mole. Across 2023 and 2024, there was a flood of research papers trying to fix this exact problem.

And that's when the question started flipping, right? Instead of hand-engineering all of these fixes, what if the AI system itself was smart enough to do this? For instance, when it notices on its first search that the answer that's come up is probably incomplete, or the documents retrieved were incomplete, can it fire another search in order to make sure that that incomplete information can be re-retrieved, right?

And that's exactly what happened with this whole idea of agentic retrieval. And that's one of the most common ways RAG has transformed in the recent past. And the core idea is very similar to how humans do research, right? You search, you skim, you realize you asked the wrong thing, you regenerate questions, you look for information, and do multiple turns of it before coming back with a final answer instead of trying to retrieve all documentation at once.

Now, as this entire idea of agentic retrieval became way more common, model companies started post-training their models, or deliberately teaching their models, to work in this way. So when the first search is weak, the model rewrites its own search and tries again with nobody scripting it. The reranking, the multiple searches, going in loops, all of that is handled by the system itself, rather than hand engineering a bunch of tricks that do not scale.

So remember that all methods have their pros and cons. For instance, agentic retrieval costs more per answer because the model is thinking and looping instead of doing one quick fetch. So it's super useful only when the problem actually calls for it.

And once the agent or the model starts deciding how to search, it's no longer stuck with one way to do it. That's exactly what camp three is about. Let's talk about that.

Now, the third camp pretty much goes after the technology under RAG. They're half right, and once you see the other half, you'll kind of understand what the debate is about.

Now, the argument goes somewhat like this, right? They think vector databases are the foundation that RAG was built on, and vector databases are kind of dead in many domains, so RAG must be dead too. That's mainly because of the changes that coding assistants or harnesses like Claude Code or Codex have brought up in the recent past.

So a lot of coding assistants that work across large repositories, they don't embed code bases into vector databases. They pretty much open folders, they scan file names, and they read the relevant files directly, almost the way a developer would do. All of this information is not embedded like we explained in the vector database case. And turns out that for tasks like coding, this actually works pretty well and much better than embedding information, because code is very structural in nature over large-scale documents which are very subjective and semantic in nature.

Now, people glued the word RAG to vector database so tightly that the moment vector databases looked like the wrong tool for some domains and jobs, it sounded like all of retrieval was dying, or all of RAG was dying. But vector search was only just one method all the time. Retrieval can happen in multiple different ways, and depending on the domain and depending on the kind of documents that are available, that exact method has to be decided by the engineer who's building it. Sometimes it could be stored as a knowledge graph, sometimes it could just happen through keyword search, sometimes in coding assistants it can just happen through grep, which is very similar to keyword search, and so on. And in my experience, many real systems end up blending several of these at once, and that is usually called hybrid retrieval, which means use different methods to improve your retrieval.

So that begs the question, are vector databases actually dead? It really depends entirely on your domain. For use cases that involve generating code, it has been empirically proven that it's barely needed, because code is full of exact things: function names, file paths, specific terms, and the agent finds those faster by searching keywords rather than embedding them into a vector database.

But think of a use case like customer support, where you wired up a knowledge base, and the agent has to answer based on the knowledge base's information. And let's say a customer comes up and says that my card keeps getting declined. And the actual document that answers this question is probably titled something like payment authorization failures. It doesn't have one single shared word with the question, yet the meaning or the semantics are very identical. So keyword search cannot really work well in this case, because it's all about the meaning and not the exact words.

So maybe the lesson for you is to match the method with your data and domain, and treat anyone selling a single answer to this question with suspicion.

Now that we've seen all of these three camps and dug deep on what they actually mean, let's understand what the future of RAG is.

Now, getting RAG to give you a good answer is one problem, but getting it to give you an answer you can actually trust is the harder problem. And it's what the next generation of these systems is being built around.

Now, picture giving two AI tools the exact same question, each using RAG to find answers from your company's documentation. And the first one loops through a few searches and gives you a clean and confident answer. But you have no idea which documents it pulled from, whether it was even authorized to read them, or whether two of these sources contradict each other. It's pretty much opaque as to why that answer was received.

And the second one gives you the same answer, but with every claim tied to the exact source and reasoning as to why it came. It has notes on why two documents disagree or why one was chosen, and a flag of which part of the answer that it affects, and also a signal of how certain it is so that you can know whether to act or dig deeper. And all of this is super auditable and has a bunch of access rules.

Now, the first answer is what you take at face value. The second is something that you can hand to an auditor. But what's being added in some of the fastest growing companies that are building RAG systems today is a lot of guardrails. Every claim tied to the source, a record of what that looks like, a flag when two sources contradict, and an honest signal of how sure it is, or a confidence score. Now, you can call this a knowledge runtime, which is the evolution of where RAG is at today.

And the biggest issue with all of these three camps that think RAG is dead is that they're making the same underlying mistake, which is treating retrieval as a one-time decision rather than something that evolves over time.

So that leaves us with the question we started with. Is RAG actually dead? And hope by now you have the answer to it. Basic one-shot RAG is definitely fading, and retrieval has still become one of the most important components of building modern AI systems.

And the only hypothetical case where RAG would be dead is that a model can read your entire knowledge base with GBs and GBs of data one single time, remember every detail of it perfectly, and never has to look up anything again. In that world, you wouldn't need retrieval. The model would simply know everything.

But again, here's why that world will pretty much never arrive. Your knowledge never stops changing, new documents keep coming up, new decisions keep getting changed, new policies, and all of this, right? So there's no way the model can read everything at once and remember it forever, because knowledge itself is changing. So even if a model had perfect memory, it would have to keep rereading all of it just to stay current. And the moment something changes, it has to notice that and update information, which is retrieval, right?

So what is the biggest lesson from today's video? Retrieval, or RAG, isn't really dying, but it's quietly evolving into a much more comprehensive knowledge or a context layer that some of the best teams are actually building with today.

So the next time someone tells you RAG is dead, you'll know exactly what they mean. You'll know what questions to ask them, and you'll understand what the hype is all about. All the very best.
