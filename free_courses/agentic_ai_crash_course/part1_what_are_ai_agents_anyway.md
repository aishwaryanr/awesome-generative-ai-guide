Day 1: What Are Agents Anyway?
Hi {{ subscriber.first_name }},
These days, everyone seems to be racing to “build agents”, but pause for a second. What even is an AI agent? And why is the whole world suddenly obsessed?
To be honest, there’s no widely accepted definition.
But here’s a simple and useful one for our purposes:
Generative AI is great at understanding and generating content.
​Agentic AI goes a step further—it understands, generates content, and performs actions.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/3bf03a13-eb32-487f-9ddb-2c97919f1e80" />

Let’s rewind a bit. In 2022, ChatGPT blew up because, for the first time, AI felt conversational.
You didn’t need to write code or train models—you could just talk to it.
Let’s compare:
Traditional programming → Needed code to operateTraditional ML → Needed feature engineeringDeep learning → Needed task-specific trainingChatGPT → Could reason across tasks and respond without training
This is known as zero-shot learning (no examples needed) or even in-context learning (understands tasks just from instructions).
But by 2024, people wanted more. Talking was cool—but what if the AI could actually do things?
For example:
Instead of just giving you a list of leads, could it email them?Instead of summarizing a doc, could it file it in the right folder and create a task in your workflow?Instead of suggesting a product to a user, could it automatically customize the landing page?
That’s where agents came in.
So… how do agents take action?
The magic lies in the tools.
Most agents are paired with APIs, function calls, or plugins that let them interact with external systems. The LLM doesn’t just respond with text—it outputs structured commands like:
“Call the send_email() function with the following inputs…”
“Fetch records from the CRM using this query…”
“Schedule a meeting for Tuesday at 2PM…”
This works because of a mechanism called tool use (or function calling). The agent is told what tools are available, and it figures out when and how to use them—either directly or through some planning mechanism.
In more advanced agents, this is enhanced with:
Memory → To remember past steps or contextPlanning modules → To decide what to do next, especially for multi-step tasksState management → So the agent can track progress and avoid loops or failures
Think of the LLM as the brain, and tools as the hands. Without tools, an agent just talks. With tools, it acts.

<img width="1216" height="413" alt="image" src="https://github.com/user-attachments/assets/9cd7fa10-21a0-42a3-95fb-3bf081e10af1" />



Two ways to define agents:
Technical view → Agents = LLM + Tools + Planning + Memory (and all the above components discussed above)Business view → Agents = Systems that complete tasks end-to-end
But don’t get confused: Today's Agents are not AI innovations.
They are engineering wrappers around AI models.The underlying intelligence still comes from the AI Models. The agent just helps act on that intelligence.



So how do you actually build agentic AI applications?
Here’s where most people go wrong:
They start with “Let’s build an agent!” instead of “What real-world problem are we solving?”
Flip the narrative.
Start with the real-world/enterprise pain points:
A support team drowning in repetitive queriesAn analyst switching between dashboards to find insightsA sales team manually logging and tracking customer activity
This course is focused on building agents that work in the real world—not just demos. Sure, you can always spin up quick personal agents or prototypes without much structure, but when you're building for the enterprise, design choices matter.



A useful mental model: Autonomy vs. Control
Once you've identified the problem, the next decision is:
​How autonomous should your agent be?
Think of it as a tradeoff:
How much autonomy are you giving the agent vs.
How much control do you want to retain on the human side?
This isn't a one-size-fits-all decision, it's contextual.
Different problems demand different levels of agent involvement.






Tomorrow (same time), we’ll go deeper into this autonomy-control tradeoff and walk through how to design agents based on the level of autonomy your use case actually needs.
-​Aish​, ​Kiriti​





