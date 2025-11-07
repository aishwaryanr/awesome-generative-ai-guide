# CHAPITRE 21 : AI AGENTS

## Introduction

Un **AI Agent** est un système autonome qui peut:
1. **Percevoir** son environnement (inputs)
2. **Raisonner** sur les actions à prendre
3. **Agir** en utilisant des outils
4. **Observer** les résultats
5. **Adapter** son comportement

Contrairement à un LLM simple (prompt → réponse), un agent peut:
- Effectuer plusieurs étapes de raisonnement
- Utiliser des outils externes (APIs, calculatrices, search)
- Maintenir une mémoire
- Corriger ses erreurs
- Planifier des tâches complexes

## 21.1 Architecture des Agents

### 21.1.1 Composants Fondamentaux

```
┌─────────────────────────────────────────────────────────────┐
│                        AI AGENT                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐│
│  │  1. PERCEPTION (Input Processing)                      ││
│  │     - User query                                        ││
│  │     - Environment state                                 ││
│  │     - Tool outputs                                      ││
│  └────────────────┬───────────────────────────────────────┘│
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │  2. MEMORY                                             ││
│  │     - Short-term (conversation)                        ││
│  │     - Long-term (knowledge base)                       ││
│  │     - Episodic (past actions)                          ││
│  └────────────────┬───────────────────────────────────────┘│
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │  3. PLANNING & REASONING (LLM Core)                    ││
│  │     - Decompose task                                   ││
│  │     - Select actions                                   ││
│  │     - Generate plans                                   ││
│  └────────────────┬───────────────────────────────────────┘│
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │  4. TOOL USE (Action Execution)                        ││
│  │     - Web search                                       ││
│  │     - Calculator                                       ││
│  │     - Code execution                                   ││
│  │     - API calls                                        ││
│  └────────────────┬───────────────────────────────────────┘│
│                   ▼                                          │
│  ┌────────────────────────────────────────────────────────┐│
│  │  5. OBSERVATION (Feedback Loop)                        ││
│  │     - Parse tool outputs                               ││
│  │     - Update memory                                    ││
│  │     - Decide next action                               ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 21.1.2 Agent Patterns

**1. ReAct (Reasoning + Acting)**

Pattern le plus populaire, alterne raisonnement et action.

```
Thought → Action → Observation → Thought → Action → ...
```

**Exemple concret:**
```
Question: "What was the temperature in Paris on the day the Eiffel Tower opened?"

Thought: I need to find when the Eiffel Tower opened first.
Action: search("When did the Eiffel Tower open")
Observation: The Eiffel Tower opened on March 31, 1889.

Thought: Now I need to find the temperature in Paris on March 31, 1889.
Action: weather_historical("Paris", "1889-03-31")
Observation: Historical weather data shows 12°C (54°F).

Thought: I have the answer now.
Final Answer: The temperature in Paris on March 31, 1889 was 12°C (54°F).
```

**Implémentation:**
```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

class ReActAgent:
    """
    Implémentation d'un agent ReAct
    """
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

        # Create agent
        self.agent_chain = self._create_agent_chain()

        # Create executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent_chain,
            tools=tools,
            verbose=True,
            max_iterations=10,
        )

    def _create_agent_chain(self):
        """Crée la chaîne de raisonnement ReAct"""

        # Prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        class CustomPromptTemplate(StringPromptTemplate):
            template: str
            tools: list

            def format(self, **kwargs) -> str:
                # Get intermediate steps
                intermediate_steps = kwargs.pop("intermediate_steps", [])
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += f"\nAction: {action.tool}\n"
                    thoughts += f"Action Input: {action.tool_input}\n"
                    thoughts += f"Observation: {observation}\n"
                    thoughts += f"Thought: "

                kwargs["agent_scratchpad"] = thoughts
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
                kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

                return self.template.format(**kwargs)

        prompt = CustomPromptTemplate(
            template=template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Parse output
        from langchain.agents import AgentOutputParser
        from langchain.schema import AgentAction, AgentFinish
        import re

        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str):
                # Check si final answer
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )

                # Parse action
                regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)

                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")

                action = match.group(1).strip()
                action_input = match.group(2)

                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        output_parser = CustomOutputParser()

        # Create agent
        from langchain.agents import LLMSingleActionAgent

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
        )

        return agent

    def run(self, query):
        """Execute agent on query"""
        result = self.agent_executor.run(query)
        return result

# Définir tools
from langchain.tools import Tool

def search_tool(query):
    # Simuler recherche web
    return f"Search results for: {query}"

def calculator_tool(expression):
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching information on the web"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for mathematical calculations"
    )
]

# Create agent
llm = OpenAI(temperature=0)
agent = ReActAgent(llm, tools)

# Run
result = agent.run("What is the square root of 144 plus 10?")
print(result)
```

**2. Plan-and-Execute**

L'agent crée d'abord un plan complet, puis l'exécute étape par étape.

```
Question → Plan (list of steps) → Execute step 1 → Execute step 2 → ...
```

```python
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

class PlanAndExecuteAgent:
    """
    Agent qui planifie d'abord, puis exécute
    """
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

        # Create planner
        self.planner = load_chat_planner(llm)

        # Create executor
        self.executor = load_agent_executor(llm, tools, verbose=True)

        # Combine into plan-and-execute
        self.agent = PlanAndExecute(
            planner=self.planner,
            executor=self.executor,
            verbose=True
        )

    def run(self, query):
        """Execute query with planning"""
        result = self.agent.run(query)
        return result

# Usage
agent = PlanAndExecuteAgent(llm, tools)
result = agent.run("Research the GDP of France and compare it to Germany")

# Output example:
# Plan:
# 1. Search for GDP of France
# 2. Search for GDP of Germany
# 3. Compare the two values
# 4. Provide comparison summary
#
# Executing step 1...
# Executing step 2...
# ...
```

**3. Reflexion (Self-Correction)**

L'agent évalue ses propres outputs et se corrige.

```python
class ReflexionAgent:
    """
    Agent qui critique et améliore ses réponses
    """
    def __init__(self, llm):
        self.llm = llm
        self.max_iterations = 3

    def run(self, query):
        """Run with self-correction"""
        current_answer = None

        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")

            # Generate answer
            if current_answer is None:
                prompt = f"Answer this question: {query}"
            else:
                prompt = f"""Previous answer: {current_answer}

Critique: {critique}

Improve your answer to: {query}"""

            current_answer = self.llm(prompt)
            print(f"Answer: {current_answer}")

            # Self-critique
            critique_prompt = f"""Critique this answer to the question "{query}":

Answer: {current_answer}

Provide constructive critique on accuracy, completeness, and clarity:"""

            critique = self.llm(critique_prompt)
            print(f"Critique: {critique}")

            # Check if good enough
            if "excellent" in critique.lower() or "perfect" in critique.lower():
                print("Answer is satisfactory!")
                break

        return current_answer

# Usage
agent = ReflexionAgent(llm)
final_answer = agent.run("Explain quantum computing")
```

## 21.2 Tool Use (Function Calling)

Les tools permettent à l'agent d'interagir avec le monde extérieur.

### 21.2.1 Définir des Tools

```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    """Input for calculator tool"""
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    """
    Calculator tool pour opérations mathématiques
    """
    name = "calculator"
    description = "Useful for mathematical calculations. Input should be a valid Python math expression."
    args_schema = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute calculation"""
        try:
            # Safe eval avec mathématiques de base
            import math
            allowed_names = {
                k: v for k, v in math.__dict__.items()
                if not k.startswith("__")
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version"""
        return self._run(expression)

# Test
calc = CalculatorTool()
result = calc.run("sqrt(144) + 10")
print(result)  # Output: 22.0
```

### 21.2.2 Web Search Tool

```python
import requests
from bs4 import BeautifulSoup

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")

class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo
    """
    name = "web_search"
    description = "Search the web for current information"
    args_schema = WebSearchInput

    def _run(self, query: str) -> str:
        """Perform web search"""
        try:
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Extract answer
            if data.get("Abstract"):
                return data["Abstract"]
            elif data.get("RelatedTopics"):
                # Get first related topic
                first_topic = data["RelatedTopics"][0]
                if "Text" in first_topic:
                    return first_topic["Text"]

            return "No results found"

        except Exception as e:
            return f"Search error: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
```

### 21.2.3 Code Execution Tool

```python
import subprocess
import tempfile
import os

class CodeExecutionInput(BaseModel):
    code: str = Field(description="Python code to execute")

class CodeExecutionTool(BaseTool):
    """
    Execute Python code in isolated environment
    """
    name = "python_executor"
    description = "Execute Python code and return output. Use for computations and data processing."
    args_schema = CodeExecutionInput

    def _run(self, code: str) -> str:
        """Execute code safely"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5,  # 5 second timeout
            )

            # Clean up
            os.unlink(temp_file)

            # Return output
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, code: str) -> str:
        return self._run(code)

# Usage
executor = CodeExecutionTool()
result = executor.run("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {arr.mean()}")
print(f"Std: {arr.std()}")
""")
print(result)
# Output:
# Mean: 3.0
# Std: 1.4142135623730951
```

### 21.2.4 API Call Tool

```python
class APICallInput(BaseModel):
    endpoint: str = Field(description="API endpoint URL")
    method: str = Field(default="GET", description="HTTP method")
    data: Optional[dict] = Field(default=None, description="Request data")

class APICallTool(BaseTool):
    """
    Generic API call tool
    """
    name = "api_call"
    description = "Make HTTP API calls to external services"
    args_schema = APICallInput

    def _run(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> str:
        """Make API call"""
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint, params=data, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(endpoint, json=data, timeout=10)
            else:
                return f"Unsupported method: {method}"

            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            return f"API error: {str(e)}"

    async def _arun(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> str:
        return self._run(endpoint, method, data)
```

### 21.2.5 Tool Collection

```python
class ToolKit:
    """
    Collection de tools prêts à l'emploi
    """
    def __init__(self):
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "code_executor": CodeExecutionTool(),
            "api_call": APICallTool(),
        }

    def get_tool(self, name):
        """Get tool by name"""
        return self.tools.get(name)

    def get_all_tools(self):
        """Get list of all tools"""
        return list(self.tools.values())

    def add_custom_tool(self, tool):
        """Add custom tool"""
        self.tools[tool.name] = tool

# Usage
toolkit = ToolKit()
all_tools = toolkit.get_all_tools()

# Create agent avec tous les tools
agent = ReActAgent(llm, all_tools)
```

## 21.3 Memory Systems

Les agents ont besoin de mémoire pour maintenir le contexte et apprendre.

### 21.3.1 Short-Term Memory (Conversation Buffer)

```python
from langchain.memory import ConversationBufferMemory

class ShortTermMemory:
    """
    Mémoire de conversation courte
    """
    def __init__(self, max_token_limit=2000):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max_token_limit,
        )

    def add_message(self, role, content):
        """Add message to memory"""
        if role == "user":
            self.memory.chat_memory.add_user_message(content)
        else:
            self.memory.chat_memory.add_ai_message(content)

    def get_history(self):
        """Get conversation history"""
        return self.memory.chat_memory.messages

    def clear(self):
        """Clear memory"""
        self.memory.clear()

# Usage
memory = ShortTermMemory()
memory.add_message("user", "Hello!")
memory.add_message("assistant", "Hi! How can I help?")
memory.add_message("user", "What's the weather?")

history = memory.get_history()
for msg in history:
    print(f"{msg.type}: {msg.content}")
```

### 21.3.2 Long-Term Memory (Vector Store)

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class LongTermMemory:
    """
    Mémoire long-terme avec vector database
    """
    def __init__(self, persist_directory="./agent_memory"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

    def store(self, content, metadata=None):
        """Store information"""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata] if metadata else None,
        )

    def retrieve(self, query, k=3):
        """Retrieve relevant memories"""
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def search_with_score(self, query, k=3, threshold=0.7):
        """Retrieve with relevance score"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        return filtered

# Usage
ltm = LongTermMemory()

# Store facts
ltm.store("The user's name is Alice", metadata={"type": "user_info"})
ltm.store("Alice prefers Python over JavaScript", metadata={"type": "preference"})
ltm.store("Last project was a chatbot", metadata={"type": "history"})

# Retrieve relevant
query = "What programming language does the user like?"
memories = ltm.retrieve(query, k=2)
for mem in memories:
    print(mem.page_content)
# Output:
# Alice prefers Python over JavaScript
# The user's name is Alice
```

### 21.3.3 Episodic Memory (Action History)

```python
from datetime import datetime
from typing import List, Dict

class EpisodicMemory:
    """
    Mémoire des actions passées de l'agent
    """
    def __init__(self):
        self.episodes = []

    def record_episode(
        self,
        action: str,
        input_data: str,
        output: str,
        success: bool,
        timestamp: datetime = None
    ):
        """Record an episode"""
        episode = {
            "timestamp": timestamp or datetime.now(),
            "action": action,
            "input": input_data,
            "output": output,
            "success": success,
        }
        self.episodes.append(episode)

    def get_recent_episodes(self, n=5):
        """Get n most recent episodes"""
        return self.episodes[-n:]

    def get_successful_episodes(self):
        """Get all successful episodes"""
        return [ep for ep in self.episodes if ep["success"]]

    def get_episodes_by_action(self, action_name):
        """Get episodes for specific action"""
        return [ep for ep in self.episodes if ep["action"] == action_name]

    def summarize(self):
        """Generate summary of episodes"""
        total = len(self.episodes)
        successful = len(self.get_successful_episodes())
        success_rate = (successful / total * 100) if total > 0 else 0

        return {
            "total_episodes": total,
            "successful": successful,
            "success_rate": f"{success_rate:.1f}%",
            "actions": list(set(ep["action"] for ep in self.episodes)),
        }

# Usage
episodic = EpisodicMemory()

# Record actions
episodic.record_episode(
    action="web_search",
    input_data="quantum computing",
    output="Quantum computing uses quantum mechanics...",
    success=True
)

episodic.record_episode(
    action="calculator",
    input_data="2 + 2",
    output="4",
    success=True
)

# Get summary
summary = episodic.summarize()
print(summary)
```

### 21.3.4 Unified Memory System

```python
class AgentMemory:
    """
    Système de mémoire complet pour agent
    """
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()

    def remember_conversation(self, role, content):
        """Store conversation message"""
        self.short_term.add_message(role, content)

    def remember_fact(self, fact, metadata=None):
        """Store long-term fact"""
        self.long_term.store(fact, metadata)

    def remember_action(self, action, input_data, output, success):
        """Record action in episodic memory"""
        self.episodic.record_episode(action, input_data, output, success)

    def recall(self, query, memory_types=["short", "long"]):
        """
        Recall information across memory systems
        """
        results = {}

        if "short" in memory_types:
            results["conversation"] = self.short_term.get_history()

        if "long" in memory_types:
            results["facts"] = self.long_term.retrieve(query)

        if "episodic" in memory_types:
            # Find relevant episodes (simple keyword match)
            relevant_episodes = [
                ep for ep in self.episodic.episodes
                if query.lower() in ep["input"].lower() or query.lower() in ep["output"].lower()
            ]
            results["past_actions"] = relevant_episodes

        return results

# Usage dans un agent
class AgentWithMemory:
    """Agent avec système de mémoire complet"""
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = AgentMemory()

    def run(self, query):
        """Execute avec mémoire"""
        # Recall relevant context
        context = self.memory.recall(query)

        # Build enhanced prompt with context
        prompt = self._build_contextual_prompt(query, context)

        # Execute action
        result = self.execute_action(prompt)

        # Remember this interaction
        self.memory.remember_conversation("user", query)
        self.memory.remember_conversation("assistant", result)

        return result

    def _build_contextual_prompt(self, query, context):
        """Build prompt with memory context"""
        prompt_parts = [f"Query: {query}\n\n"]

        if context.get("conversation"):
            prompt_parts.append("Recent conversation:\n")
            for msg in context["conversation"][-5:]:
                prompt_parts.append(f"{msg.type}: {msg.content}\n")

        if context.get("facts"):
            prompt_parts.append("\nRelevant facts:\n")
            for fact in context["facts"]:
                prompt_parts.append(f"- {fact.page_content}\n")

        return "".join(prompt_parts)
```

---

*[Le chapitre continue avec Planning & Reasoning, Multi-Agent Systems, et cas pratiques complets...]*

*[Contenu total du Chapitre 21: ~80-90 pages]*
