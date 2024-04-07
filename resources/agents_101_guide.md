# LLM Agents 101

![llm_guide.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/llm_guide.png)

## Introduction to LLM Agents

LLM agents, short for Large Language Model agents, are gaining quite some popularity because they blend advanced language processing with other crucial components like planning and memory. They smart systems that can handle complex tasks by combining a large language model with other tools.

Imagine you're trying to create a virtual assistant that helps people plan their vacations. You want it to be able to handle simple questions like "What's the weather like in Paris next week?" or "How much does it cost to fly to Tokyo in July?"

A basic virtual assistant might be able to answer those questions using pre-programmed responses or by searching the internet. But what if someone asks a more complicated question, like "I want to plan a trip to Europe next summer. Can you suggest an itinerary that includes visiting historic landmarks, trying local cuisine, and staying within a budget of $3000?"

That's a tough question because it involves planning, budgeting, and finding information about different destinations. An LLM agent could help with this by using its knowledge and tools to come up with a personalized itinerary. It could search for flights, hotels, and tourist attractions, while also keeping track of the budget and the traveler's preferences.

To build this kind of virtual assistant, you'd need an LLM as the main "brain" to understand and respond to questions. But you'd also need other modules for planning, budgeting, and accessing travel information. Together, they would form an LLM agent capable of handling complex tasks and providing personalized assistance to users.

![Screenshot 2024-04-07 at 2.39.12 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_2.39.12_PM.png)

Image Source: [https://arxiv.org/pdf/2309.07864.pdf](https://arxiv.org/pdf/2309.07864.pdf)

The above image represents a potential theoretical structure of an LLM-based agent proposed by the paper “**[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)**”

It comprises three integral components: the brain, perception, and action. 

- Functioning as the central controller, the **brain** module engages in fundamental tasks such as storing information, processing thoughts, and making decisions.
- Meanwhile, the **perception** module is responsible for interpreting and analyzing various forms of sensory input from the external environment.
- Subsequently, the **action** module executes tasks using appropriate tools and influences the surrounding context.

The above framework represents one approach to breaking down the design of an LLM agent into distinct, self-contained components. However, please note that this framework is just one of many possible configurations.

In essence, an LLM agent goes beyond basic question-answering capabilities of an LLM. It processes feedback, maintains memory, strategizes for future actions, and collaborates with various tools to make informed decisions. This functionality resembles rudimentary human-like behavior, marking LLM agents as stepping stones towards the notion of Artificial General Intelligence (AGI). Here, LLMs can autonomously undertake tasks without human intervention, representing a significant advancement in AI capabilities.

## LLM Agent Framework

In the preceding section, we discussed one framework for comprehending LLM agents, which involved breaking down the agent into three key components: the brain, perception, and action. In this section, we will explore a more widely used framework for structuring agent components. 

This framework comprises the following essential elements:

![Screenshot 2024-04-07 at 2.53.23 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_2.53.23_PM.png)

Image Source: [https://developer.nvidia.com/blog/introduction-to-llm-agents/](https://developer.nvidia.com/blog/introduction-to-llm-agents/)

1. **Agent Core:** The agent core functions as the central decision-making component within an AI agent. It oversees the core logic and behavioral patterns of the agent. Within this core, various aspects are managed, including defining the agent's overarching goals, providing instructions for tool utilization, specifying guidelines for employing different planning modules, incorporating pertinent memory items from past interactions, and potentially shaping the agent's persona.
2. **Memory Module:** Memory modules are essential components of AI agents, serving as repositories for storing internal logs and user interactions. These modules consist of two main types: 
    1. Short-term memory: captures the agent's ongoing thought processes as it attempts to respond to a single user query. 
    2. Long-term memory: maintains a historical record of conversations spanning extended periods, such as weeks or months. 
    
    Memory retrieval involves employing techniques based on semantic similarity, complemented by factors such as importance, recency, and application-specific metrics.
    
3. **Tools:** Tools represent predefined executable workflows utilized by agents to execute tasks effectively. They encompass various capabilities, such as RAG pipelines for context-aware responses, code interpreters for tackling complex programming challenges, and APIs for conducting internet searches or accessing simple services like weather forecasts or messaging.
4. **Planning Module:** Complex problem-solving often requires well structured approaches. LLM-powered agents tackle this complexity by employing a blend of techniques within their planning modules. These techniques may involve task decomposition, breaking down complex tasks into smaller, manageable parts, and reflection or critique, engaging in thoughtful analysis to arrive at optimal solutions.

## Multi-agent systems (MAS)

While LLM-based agents demonstrate impressive text understanding and generation capabilities, they typically operate in isolation, lacking the ability to collaborate with other agents and learn from social interactions. This limitation hinders their potential for enhanced performance through multi-turn feedback and collaboration in complex scenarios.

 LLM-based multi-agent systems (MAS) prioritize diverse agent profiles, interactions among agents, and collective decision-making. Collaboration among multiple autonomous agents in LLM-MA systems enables tackling dynamic and complex tasks through unique strategies, behaviors, and communication between agents.

An LLM-based multi-agent system offers several advantages, primarily based on the principle of the division of labor. Specialized agents equipped with domain knowledge can efficiently handle specific tasks, leading to enhanced task efficiency and collective decision improvement. Decomposing complex tasks into multiple subtasks can streamline processes, ultimately improving system efficiency and output quality.

**Types of Multi-Agent Interactions**

Multi-agent interactions in LLM-based systems can be broadly categorized into cooperative and adversarial interactions.

![Screenshot 2024-04-07 at 3.03.48 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_3.03.48_PM.png)

Image source: [https://arxiv.org/pdf/2309.07864.pdf](https://arxiv.org/pdf/2309.07864.pdf)

### **Cooperative Interaction:**

In cooperative multi-agent systems, agents assess each other's needs and capabilities, actively seeking collaborative actions and information sharing. This approach enhances task efficiency, improves collective decision-making, and resolves complex real-world problems through synergistic complementarity. Existing cooperative multi-agent applications can be classified into disordered cooperation and ordered cooperation

1. **Disordered cooperation**: multiple agents within a system express their perspectives and opinions freely without adhering to a specific sequence or collaborative workflow. However, without a structured workflow, coordinating responses and consolidating feedback can be challenging, potentially leading to inefficiencies.
2. **Ordered cooperation**: agents adhere to specific rules or sequences when expressing opinions or engaging in discussions. Each agent follows a predefined order, ensuring a structured and organized interaction.

Therefore, disordered cooperation allows for open expression and flexibility but may lack organization and pose challenges in decision-making. On the other hand, ordered cooperation offers improved efficiency and clarity but may be rigid and dependent on predefined sequences. Each approach has its own set of benefits and challenges, and the choice between them depends on the specific requirements and goals of the multi-agent system.

### **Adversarial Interaction**

While cooperative methods have been extensively explored, researchers increasingly recognize the benefits of introducing concepts from game theory into multi-agent systems. Adversarial interactions foster dynamic adjustments in agent strategies, leading to robust and efficient behaviors. Successful applications of adversarial interaction in LLM-based multi-agent systems include debate and argumentation, enhancing the quality of responses and decision-making.

Despite the promising advancements in multi-agent systems, several challenges persist, including limitations in processing prolonged debates, increased computational overhead in multi-agent environments, and the risk of convergence to incorrect consensus. Further development of multi-agent systems requires addressing these challenges and may involve integrating human guides to compensate for agent limitations and promote advancements.

MAS is a dynamic field of study with significant potential for enhancing collaboration, decision-making, and problem-solving in complex environments. Continued research and development in this area promise to pave way for  new opportunities for intelligent agent interaction and cooperation leading to progress in AGI.

## Real World LLM Agents: BabyAGI

BabyAGI is a popular task-driven autonomous agent designed to perform diverse tasks across various domains. It utilizes technologies such as OpenAI's GPT-4 language model, Pinecone vector search platform, and the LangChain framework. Here's a breakdown of its key components as discussed [by the author](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/) .

1. GPT-4 (Agent Core):
    - OpenAI's GPT-4 serves as the core of the system, enabling it to complete tasks, generate new tasks based on completed results, and prioritize tasks in real-time. It leverages the powerful text-based language model capabilities of GPT-4.
2. Pinecone(Memory Module):
    - Pinecone is utilized for efficient storage and retrieval of task-related data, including task descriptions, constraints, and results. It provides robust search and storage capabilities for high-dimensional vector data, enhancing the system's efficiency.
3. LangChain Framework (Tooling Module):
    - The LangChain framework enhances the system's capabilities, particularly in task completion and decision-making processes. It allows the AI agent to be data-aware and interact with its environment, contributing to a more powerful and differentiated system.
4. Task Management (Planning Module):
    - The system maintains a task list using a deque data structure, enabling it to manage and prioritize tasks autonomously. It dynamically generates new tasks based on completed results and adjusts task priorities accordingly.
    
    ![babyAGI](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/babyAGI.png)
    
    Image Source: [https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/](https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/)
    

 BabyAGI operates through the following steps:

1. Completing Tasks: The system processes tasks from the task list using GPT-4 and LangChain capabilities to generate results, which are then stored in Pinecone.
2. Generating New Tasks: Based on completed task results, BabyAGI employs GPT-4 to generate new tasks, ensuring non-overlapping tasks with existing ones.
3. Prioritizing Tasks: Task prioritization is conducted based on new task generation and priorities, with assistance from GPT-4 to facilitate the prioritization process.

You can find the code to test and play around with BabyAGI [here](https://github.com/yoheinakajima/babyagi)

Other popular LLM based agents are listed [here](https://www.promptingguide.ai/research/llm-agents#notable-llm-based-agents)

## **Evaluating LLM Agents**

Despite their remarkable performance in various domains, quantifying and objectively evaluating LLM-based agents remain challenging.  Several benchmarks have been designed to evaluate LLM agents. Some examples include

1. [AgentBench](https://github.com/THUDM/AgentBench)
2. [IGLU](https://arxiv.org/abs/2304.10750)
3. [ClemBench](https://arxiv.org/abs/2305.13455)
4. [ToolBench](https://arxiv.org/abs/2305.16504)
5. [GentBench](https://arxiv.org/pdf/2308.04030.pdf)

![Screenshot 2024-04-07 at 3.28.33 PM.png](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/resources/img/Screenshot_2024-04-07_at_3.28.33_PM.png)

Image: Tasks and Datasets supported by the GentBench framework

Apart from task specific metrics, some dimensions in which agents can be evaluated include

- **Utility**: Focuses on task completion effectiveness and efficiency, with success rate and task outcomes being primary metrics.
- **Sociability**: Includes language communication proficiency, cooperation, negotiation abilities, and role-playing capability.
- **Values**: Ensures adherence to moral and ethical guidelines, honesty, harmlessness, and contextual appropriateness.
- **Ability to Evolve Continually**: Considers continual learning, autotelic learning ability, and adaptability to new environments.
- **Adversarial Robustness**: LLMs are susceptible to adversarial attacks, impacting their robustness. Traditional techniques like adversarial training are employed, along with human-in-the-loop supervision.
- **Trustworthiness**: Calibration problems and biases in training data affect trustworthiness. Efforts are made to guide models to exhibit thought processes or explanations to enhance credibility.

## Build Your Own  Agent (Resources)

Now that you have gained an understanding of LLM agents and their functioning, here are some top resources to help you construct your own LLM agent.

1. [How to Create your own LLM Agent from Scratch: A Step-by-Step Guide](https://gathnex.medium.com/how-to-create-your-own-llm-agent-from-scratch-a-step-by-step-guide-14b763e5b3b8)
2. [Building Your First LLM Agent Application](https://developer.nvidia.com/blog/building-your-first-llm-agent-application/)
3. [Building Agents on LangChain](https://python.langchain.com/docs/use_cases/tool_use/agents/)
4. [Building a LangChain Custom Medical Agent with Memory](https://www.youtube.com/watch?v=6UFtRwWnHws)
5. [LangChain Agents - Joining Tools and Chains with Decisions](https://www.youtube.com/watch?v=ziu87EXZVUE)

## References:

1. [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf)
2. [Large Language Model based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/pdf/2402.01680.pdf)
3. [LLM Agents by Prompt Engineering Guide](https://www.promptingguide.ai/research/llm-agents#notable-llm-based-agents)
4. [Introduction to LLM Agents, Nvidia Blog](https://developer.nvidia.com/blog/introduction-to-llm-agents/)