# Build your own company harness

**Watch:** [2,000+ Hours of Learning Harness Engineering in 18 Minutes](https://www.youtube.com/watch?v=mv9kfF_kJd0)

**Published:** October 29, 2026.

**Published YouTube description:** [Read the full description](descriptions/build-your-own-harness.txt)
**Code:** [NS Work notebook and application](code/ns-work-harness/README.md) · [Download the source ZIP](code/ns-work-harness.zip)

Build NS Work, a harness around an AI model for Northstar's sales, engineering, and finance teams. The example starts with Deep Agents and adds Context, Actions, Boundaries, and Persistence through one cumulative notebook.

## Start with the code

1. Download the ZIP or clone this repository.
2. Open `youtube/code/ns-work-harness/` and follow its [setup instructions](code/ns-work-harness/README.md#set-up).
3. Open [00_NS_Work.ipynb](code/ns-work-harness/00_NS_Work.ipynb) in VS Code or Jupyter and select your Python 3.12 environment.
4. Run each chapter's configuration cells, then use its interface link to submit a request.

The source includes the recorded notebook, the shared harness configuration, all 4 company tools, company instructions, shared skills, a local web interface, and an offline verification script. The ZIP contains the same source as the code folder.

## The 4 responsibilities

| Responsibility | What you configure | Example |
|---|---|---|
| Context | Company guidance and reusable skills | Prepare proposals using Northstar's conventions |
| Actions | Company tools, planning, and a specialist | Look up Acme and ask a reviewer to check a commitment |
| Boundaries | Role permissions, approval, protected files, and call limits | Finance can read revenue; Sales needs approval before recording a proposal send |
| Persistence | Working files, checkpoints, and context management | Save a proposal and return to its task later |

This is a local teaching application with synthetic company records. Proposal sends are local outbox records. The [code README](code/ns-work-harness/README.md#execution-scope) explains deployment boundaries, storage, and model usage.

## Library references

- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization)
- [Skills](https://docs.langchain.com/oss/python/deepagents/skills)
- [Human approval](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
- [Filesystem backends](https://docs.langchain.com/oss/python/deepagents/backends)

The example pins the library versions used in the recording. Follow its `requirements.txt` to reproduce the configuration.

## Transcript

_English captions from the published video. Automatic caption wording is retained; timestamps link to the video._

**[00:00](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=0s)**

There's this new AI skill companies are suddenly hiring for everywhere. And there are also roles where literally this is the whole job. It's called harness engineering, which is basically the skill of building a harness or the operating system around an AI model that lets it carry out tasks reliably in a production AI application. Now, in this video, we're going to understand what harness engineering is and the major components that go into building a production harness. And instead of just giving theory, which you'll probably

**[00:30](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=30s)**

forget, we learn it by building our own company harness step by step. But don't worry, it won't be super hard. Just follow along. Now, before we get started, let's first understand why we even need a harness. [music] Now, as models have gotten more powerful, [clears throat] we have started expecting them to do a lot more than just answer questions in a chat window. We want them to complete entire tasks. for instance, prepare a customer proposal or look up right information in our company systems, send it to the customer once it's approved or engineers are probably looking to make code changes across the whole

**[01:00](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=60s)**

repository, [music] run tests, fix them, and all of these end-to-end tasks. And for that to happen, the model needs to work inside an ecosystem with our data, our tools, and the way that our teams actually do work. And once you start building applications like this, you realize that the AI model is literally just one small part of the system. The model can just decide what to do next, but something will have to give it the right information, carry out these actions, check what it's allowed to do, and keep track of the work as it keeps

**[01:31](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=91s)**

continuing. That entire software layer is called the harness. And you can pretty much think of it as the operating system around the model that lets it get things done. And even products that you use every day like Chad GPT, clot code, codecs, etc. have software around the model that connects it to tools, files, and the rest of the application. And that's pretty much the harness layer. And how well you build this harness layer itself can make a huge difference in the performance of your agentic AI systems. There's literally research

**[02:01](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=121s)**

showing that the exact same model performs up to six times better by simply building a better harness system around the agent. And this is also why a lot of companies are building their own harnesses around models. For instance, Stripe recently built minions for its engineers which is a coding harness designed specifically around the stripe ecosystem and companies like ramp Spotify and many other companies have been doing something similar. So let's understand the major components of harness engineering by actually building one ourselves. Let's say our company is

**[02:31](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=151s)**

Northstar an analytics company and we want to build our own internal harness for our employees and for fun we'll call it NS work. similar to you know cloud co-work and we want to build this harness for three kinds of employees within the company for instance sales engineering and finance and for this harness to actually be useful you need to be thinking about four major components and all through this video we'll go through all of these components and how we can build them now the first component is context it needs to understand northstar how the company works and the processes employees

**[03:02](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=182s)**

already follow now second is actions it needs [music] to actually interact with company systems like looking up customer information, pulling an issue from our database, looking up our code base and all of those things. And the third one is boundaries. [music] Different employees should have different access and some actions should require approval even before they happen. So all of that should be enforced in the harness. [music] And the fourth component is persistence. If someone starts a longer task, the harness needs to keep track of that work, say what matters, and then come back to them and continue later. So

**[03:32](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=212s)**

let's build each of this one by one. Now to build this we're going to use a library called deep agents by lang graph. Now the good part about it is that it gives us the basic agent loop along with the ability to add additional components that we need for our harness so that we don't have to start from scratch. [music] So I'm going to set up a notebook and then going to initialize deep agents. I'm going to just create a simple harness here with the create deep agent function and then I'll probably use like a GPT 5.4 um to initialize this harness. I've also built a very simple

**[04:04](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=244s)**

interface um just because it's fun and you can actually see the chat happening. I won't be going too deep into how we build that. It should be pretty straightforward to get something running. I'll leave a prompt that you can give to your agent to build an interface like this. We'll be talking much more about the back end or the harness we build instead of the front end interface. Right? So this is my very first version of NS work. It's a simple chat window and I only connected a model to it at this point. Right? I can ask something like do you know about our company Northstar? All right, it says I

**[04:36](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=276s)**

don't have built-in knowledge of your company obviously because I've not given it any information. So let's go further and build the first component which is context for our harness. Context means understanding Northstar well enough that employees don't have to keep explaining about the company every time they use it. Right? We can start off by providing some company information that should always be available like what Northstar does and the rules that the employee should follow. And I can save all of that in a single file. In my demo, I'm going to do it through this file called agents.mmptd. You can name it anything, but I'll walk you through what agents.md

**[05:07](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=307s)**

looks like. So this is just a simple file that I've written to talk about the company and to write a bunch of working conventions, right? For instance, raised concise answers, cite the record IDs and stuff like that. Essentially, this almost like a system prompt that runs every time an employee is chatting with my harness and they don't have to give this information again and again. After the harness knows about my company, the next thing we want to do is make the harness understand about certain processes that we already follow within the company. For instance, our sales

**[05:37](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=337s)**

team has a specific way of writing proposals. Engineering teams might have, you know, a specific way of investigating an issue. And I want to make sure that this process is repeatable and everybody doesn't have to keep writing newer prompts. Which is why I'm going to set up the next piece of context, which is skills. So, if you've never heard of it, a skill is basically a set of instructions for how to perform a particular task. So, when an employee asks to write a proposal, the agent can pretty much load that skill and follow the process instead of the employee themselves explaining all [music] of it. Right? I've set up a few example skills

**[06:09](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=369s)**

for my harness. Some of them are company level skills. For instance, writing a proposal. So, this is our skill where there is an established process of, you know, these are the different sections that it should have, the scope, the assumptions and all of that. There are also some team level skills that I just set up as simple examples. For instance, how does our team triage issues? There's a skill to do that, right? So once we set up the shared skills, we can go to the next version of our harness which is a harness with context as well. I'm

**[06:39](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=399s)**

taking the same create deep agent code, but this time I'm also giving it access to my agents.mmd file with this parameter called memory. You can give it access to other files as well, but remember that this is the file that will load up every time the employee is talking. So, it's almost like a system prompt. You want to use it carefully. We're also giving it access to my skills directory so that it has access to my company and team skills. Now, once we've loaded that up, let me get to the second version of our harness which is loaded with context. Now, you'll see that I

**[07:09](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=429s)**

also set up a small example and I'm saying use our proposal skill to draft a proposal for Acme. I'm writing a bit about who Acme is and let's see if our harness is able to do that. Now you can also see that the harness is actually reading my proposal skill and it's going through the process. So that is my proposal and it's used the proposal skill to do that. It has the sections that I want and all of this which is pretty cool. Now all of this is fun. It's gotten much better than my first version. But if you see every time I ask

**[07:39](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=459s)**

my harness to draft a proposal I need to speak about who ACME is, what is their background and all of this right which is already sitting in our company's systems. So it would be pretty cool if my harness could understand about ACME without me having to explain who they are and if my sales team has to use it regularly they don't have to copy paste all of this information which takes us to building the next component for our harness which is the actions component. Now the goal of this component is to make sure that my harness can interact with our company systems [music] and the way to implement that is to give it

**[08:10](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=490s)**

access to the tools or the APIs that are already part of our company. Here are a few example tools that I've set up. You can see that there's a customer tool which essentially has a query to get details about a customer if I know their name. There's also descriptions that I've written here so that my harness can understand what certain tools are for. There are other tools that I have as well. For instance, there's an issue lookup tool. There's also a proposal tool which is essentially to send a proposal to a customer or store it in an inbox and stuff like that, right?

**[08:40](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=520s)**

There's also a revenue tool that I put together. These are small examples. And then after we have all of these tools written down, we want to make sure that the harness has access to it. Now, once we connect to these tools, a request that the harness gets can have several steps. And in order to do that, I'll use a feature within deep agents. It's called middleware. Middleware is essentially code that we can attach to the agent's loop to add behavior at specific points like before a model call or around a tool call or after a model call etc. Right? For example, suppose

**[09:10](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=550s)**

the agent keeps calling tools and goes back to the model without finishing. Middleware can check how many model calls it's made and stop when it reaches a certain limit. So that's one of the model call limit middleware that I'm adding here. I also added another to-do list middleware here. And what this essentially does is it asks my harness to create a to-do list for itself whenever it's given a long task so that I can audit what are these steps that are happening which might be helpful for logging and for further improvement of my harness. [music] There are a lot of

**[09:40](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=580s)**

middleware options that deep agents provides. I'm just adding two of these so that you understand the functionality but then you can look it up. I'll leave resources. [music] Another optimization that I want to do in this component is this whole idea of sub agents. A sub agent is basically like a mini specialist agent with its own instructions, model and tools. And the main agent can give it a specific task. It can delegate something to it and the sub agent can finish it in parallel and it can work in a separate thread using just the task and the information that the main agent has given it. It's usually used for

**[10:10](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=610s)**

parallelization and also for context cleanliness and all of this. Now you can spin up any number of sub agents. I'm going to spin up one called proposal reviewer and what it essentially does is it reviews a proposal against the customer facts and it flags [music] unsupported commitments. So it has its own system prompt. I can also set up a specific model for it. Sometimes sub agents use smaller models as compared to the main agent so that you can save costs [music] and it can also have its own tools. You don't have to give it access to all of the tools and middleware as well. So I'm going to set up this proposal reviewer sub agent

**[10:41](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=641s)**

[music] and then create the next version of my harness which also is equipped with actions. So you can see here that I'm again using the create deep agent function but this time [music] I'm giving it access to my latest tools to sub aents as well as my middleware. Let's see how my UI looks now. [music] Okay, now I'm going to just say draft a proposal for ACME and review if the proposal looks good. And now you can see that it's [music] using the proposal skill. It's also looking up ACME from my

**[11:12](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=672s)**

database which means it's doing the tool call and it's also creating a plan for itself. And you can also see it's using a sub aent which is my proposal reviewer in order to review the proposal because that's how we set up the sub aent. So all the features we implemented is pretty much working here. [music] All right. So this is the proposal we have and it's used all of these action pieces that we added. Now, we've given a harness quite a few capabilities, right? And remember that with great power comes great responsibility. So, we need to be very careful about what it's allowed to

**[11:43](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=703s)**

do, which brings us to the third component, boundaries. Now, boundaries are essentially rules or guardrails that you want to enforce around the harness so that it knows what to do and where it needs to stop and get approval. Now, you want to take care of two things here. One is permissions, the other is approvals. Let's start with permissions. Now remember that depending on the role of my employee, they might need access to different tools and you should also guard access to some tools. For instance, a salesperson might need access to customer information and proposal tools. An engineer might need

**[12:14](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=734s)**

issue data. Finance might need revenue information, but an engineer should not be able to look at revenue data just because that's not the tool that must be exposed [music] to them. So one way to enforce which role has access to what can be defined in the permission section. Now here I'm going to set up a dictionary called company access where I'm saying that sales has access to these tools, engineering has access to this and finance has access to this and I'm going to set that up. The other kind of permissions that I'm going to take care of is file system permissions which

**[12:45](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=765s)**

are essentially what kind of files are users allowed to change, what kind of files are they not allowed to change. I want to set a simple file permission here. Although in real harnesses it can get much more complex which is something like the company and skills folder should not be written into right because that's something that I want the harness developer to change not the user to change. Now the second thing that I was talking about which is approvals basically this allows the harness to ask for permission for certain kinds of

**[13:15](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=795s)**

actions. And the argument that allows for it is this interrupt on where I can specifically say for the send proposal function you should ask a human in order to give you approval and the allowed decisions are approve or reject right so you can interrupt the agent at certain checkpoints and make sure that it gets an approval from a human. So that's my next version of my harness which has boundaries as well. So you can see that now I'm giving it a more scoped set of allowed tools based on roles. I'm also

**[13:45](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=825s)**

giving it file permissions as well as interrupt on mode to make sure that whenever there are certain actions being taken, it [music] asks the human. All right, that brings us to the next version. I'm going to give the same example which is drafting the Acme proposal and then I'm going to say send the generated proposal to acme 1 acme.com. Right? So I just want to check if my boundaries are implemented well. So you can see that it says I can't directly send an email and it's also identifying that the email I provided is

**[14:16](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=856s)**

not the same one in my database. So it's an unverified email. So you can kind of see what it's allowed to do and it's not allowed to do. And we'll also test another use case here which is let's go to the profile and [music] say this is an engineering profile, right? And then let's say what was the revenue of our company this year. [music] So if an engineer were to use this harness, we want to make sure that they don't have access to certain tools that probably the finance department has access to. So it pretty much says I don't have verified access to this, which means that you cannot be looking at this. So

**[14:47](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=887s)**

it doesn't give me anything versus if the same question were asked by finance. What's the revenue of our company this year? Pretty seamless, right? So all of this is super fun. We build different versions of our harness. Now, one thing that we didn't account for is the fact that our employees can talk to the agent, get a lot of work done. But what happens if they have a longunning task that they've half done today and want to come back tomorrow and continue working on that task? We haven't implemented some way of storing a particular state or picking up from an existing conversation, which brings us to our

**[15:18](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=918s)**

last component, persistence. Now the whole goal here is to make sure that we have a snapshot of the work done so that users can come back and continue to work on tasks. And in order to implement persistence I'm going to be doing two things. The first thing is to make sure we have a writable folder. So whatever the harness generates it has to be able to save it somewhere. Here I'm giving it permission to a particular workspace within my local computer so that it can get access to it. But when systems get more complex, you might have to build sandboxes so that the agent doesn't have

**[15:49](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=949s)**

access to local systems or it doesn't have access to locations where you wouldn't want it to look for. But we're going to keep it simple for this demo. And the second thing we're going to do is implement this idea of checkpoints. I'm going to use SQLite to do that. You can do it in different methods, but the idea over here is to save a checkpoint of a conversation so that it can be loaded up back. Right? I can quickly show you an older checkpoint that was saved here. So this is an example of a checkpoint conversation that can be stored so that it can be brought back

**[16:19](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=979s)**

up, right? It has all of the conversation that happened, additional metadata and all of that. And deep agents makes it [music] a lot more easier to save these checkpoints. You don't have to worry too much about the minute details. I'm also going to do a optimization here which is this whole idea of context management. Remember that when you're having long conversations with these harnesses, it's possible that you hit the context limit of the model, which means that the context has to be compacted or summarized so that the conversation can

**[16:49](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=1009s)**

continue. And that can be implemented on deep agents using this idea of summarization middleware. That's what I've imported here. And here what I'm doing is trying to make sure that summarization is triggered at 12,000 tokens. And I keep the last six messages as is. Now, you can decide what your summarization scheme is based on the kind of use cases, but this is a quick way to make sure that you're managing context as well and not letting it overflow. Now, after we' added these three pieces, I'm going to update my

**[17:19](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=1039s)**

harness to include all of the things that we've built so far along with persistence. Let's see the final version of my harness. All right, so you can see that it's giving me options to restart from a work that I already started off from. It's able to load up past conversations. I can also start a new task and it pretty much has memory of what was discussed before. Right? So that's our final version of NS work which is the harness that we built on our own. Now there are a lot more bells

**[17:49](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=1069s)**

and whistles that you can add to this. This is just a starting point but you kind of see how all of them [clears throat] belong to these four buckets that we spoke about. It can get much more complicated. For instance, you can give every employee personalized memory. You can decide on much more comprehensive compaction mechanisms. [music] You can set up scheduled tasks just like cloudco work does. And you can do all of that specific for how your company works and how the systems within your company have been set up. But if you kind of zoom out, a harness consists of all of these components. And you'll have to build each of these components.

**[18:20](https://www.youtube.com/watch?v=mv9kfF_kJd0&t=1100s)**

And this is the skill of harness engineering which companies are looking for. Hope you had fun building this harness with me. I'll leave the code and demo on my GitHub repository and you can build on top of it. I truly think this is a really good skill that you can build if you want to become an AI engineer. Hopefully, you'll build on top of it. I'll see you in the next video. Until then, all the very
