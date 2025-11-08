# CHAPITRE 14 : AGENTS LLM ET REACT

> *« The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. »*
> — Edsger W. Dijkstra

---

## Introduction : De la Génération de Texte à l'Action

Un LLM seul est puissant, mais **limité** : il peut générer du texte, raisonner sur des concepts, écrire du code. Mais il ne peut pas :
- Exécuter ce code
- Chercher des informations en temps réel sur le web
- Accéder à une base de données
- Envoyer un email
- Réserver un vol
- Commander une pizza

**Et si on donnait des "mains" à notre LLM ?** Et si on le transformait en **agent autonome** capable d'interagir avec le monde extérieur ?

C'est exactement ce que font les **LLM Agents** : des systèmes qui combinent la puissance de raisonnement d'un LLM avec la capacité d'**agir** sur l'environnement via des outils (APIs, bases de données, calculatrices, navigateurs web, etc.).

Dans ce chapitre, nous explorerons :
- L'architecture des agents LLM
- Le framework **ReAct** (Reasoning + Acting)
- Les patterns d'implémentation
- Les défis et solutions (erreurs, boucles infinies, coûts)
- Des implémentations complètes en production

Bienvenue dans l'ère des **agents autonomes**.

---

## 1. Qu'est-ce qu'un Agent LLM ?

### 🎭 Dialogue : La Métaphore de l'Assistant

**Alice** : Bob, j'ai utilisé ChatGPT pour générer du code Python. Mais ensuite, je dois copier-coller le code, l'exécuter moi-même, voir les erreurs, revenir à ChatGPT pour les corriger... C'est fastidieux !

**Bob** : Exactement. C'est parce que ChatGPT est un **LLM pur** : il génère du texte, mais il ne peut pas **agir**.

**Alice** : Tu veux dire qu'il ne peut pas exécuter le code lui-même ?

**Bob** : Précisément. Mais imagine maintenant qu'on donne à ChatGPT accès à un **interpréteur Python**. Il pourrait :
1. Générer le code
2. L'exécuter
3. Voir les erreurs
4. Les corriger automatiquement
5. Réessayer jusqu'à ce que ça marche

**Alice** : Ça ressemble à un développeur junior qui debug !

**Bob** : Exactement ! Et si on va plus loin, on peut lui donner accès à d'autres **outils** :
- Une calculatrice pour les calculs précis
- Un moteur de recherche pour les infos à jour
- Une base de données pour stocker/récupérer des données
- Un navigateur web pour interagir avec des sites
- Une API d'envoi d'emails

**Alice** : Donc il devient un vrai **agent** capable d'accomplir des tâches complexes ?

**Bob** : Voilà ! On passe de "générateur de texte" à "assistant autonome".

---

### 1.1 Définition Formelle

Un **Agent LLM** est un système composé de :

1. **Un LLM** (le "cerveau") : raisonne, planifie, décide
2. **Des outils** (les "mains") : APIs, fonctions, bases de données
3. **Une boucle de contrôle** : perception → raisonnement → action → observation
4. **Une mémoire** (optionnelle) : historique des actions et observations

```
┌─────────────────────────────────────────────────┐
│                   AGENT LLM                     │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌─────────────┐         ┌─────────────┐       │
│  │    LLM      │◄───────►│   Memory    │       │
│  │  (Cerveau)  │         │ (Historique)│       │
│  └──────┬──────┘         └─────────────┘       │
│         │                                       │
│         │ Décision                              │
│         ▼                                       │
│  ┌─────────────────────────────────┐           │
│  │      Tool Selection             │           │
│  │  (Quel outil utiliser ?)        │           │
│  └────────┬────────────────────────┘           │
│           │                                     │
│           ▼                                     │
│  ┌────────────────────────────────────┐        │
│  │          TOOLS                     │        │
│  │  • Calculator                      │        │
│  │  • Web Search                      │        │
│  │  • Python Interpreter              │        │
│  │  • Database Query                  │        │
│  │  • API Calls                       │        │
│  └────────┬───────────────────────────┘        │
│           │                                     │
│           │ Observation (résultat)              │
│           ▼                                     │
│  ┌─────────────────────────────────┐           │
│  │   Update Memory & Loop          │           │
│  └─────────────────────────────────┘           │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📜 Anecdote Historique : SHRDLU (1968-1970)

**MIT, Cambridge, Massachusetts, 1968** : Terry Winograd, étudiant en doctorat, développe **SHRDLU**, l'un des premiers systèmes d'IA conversationnelle capable d'**agir** dans un monde (virtuel).

SHRDLU contrôle un bras robotique virtuel dans un monde de blocs géométriques colorés. L'utilisateur peut donner des commandes en langage naturel :

```
Utilisateur : "Pick up a big red block."
SHRDLU : [exécute l'action, saisit le bloc rouge]

Utilisateur : "Grasp the pyramid."
SHRDLU : "I don't understand which pyramid you mean."

Utilisateur : "Find a block which is taller than the one you are holding and put it into the box."
SHRDLU : [analyse, planifie, exécute plusieurs actions]
```

**Innovation** : SHRDLU ne se contente pas de **comprendre** le langage, il **agit** dans son environnement et **raisonne** sur les conséquences de ses actions.

**56 ans plus tard**, les agents LLM modernes utilisent les mêmes principes — mais à une échelle infiniment plus grande, avec des capacités de raisonnement bien plus sophistiquées.

---

## 2. Le Framework ReAct : Reasoning + Acting

### 2.1 Le Problème des LLMs Purs

Un LLM seul peut **halluciner** des informations :

```python
# Question nécessitant des infos à jour
question = "Combien d'habitants compte Tokyo en 2026 ?"

# LLM pur (GPT-4) :
# → "Tokyo compte environ 14 millions d'habitants."
#    (Basé sur ses données d'entraînement, potentiellement obsolètes)
```

**Problème** : Le LLM ne peut pas vérifier ses informations. Il génère ce qui est **statistiquement probable**, pas ce qui est **factuellement correct**.

**Solution ReAct** : Permettre au LLM de **chercher** l'information avant de répondre.

---

### 2.2 ReAct : L'Approche

**ReAct** (Yao et al., 2022) = **Rea**soning + **Act**ing

Le LLM alterne entre :
- **Thought** (Pensée) : raisonnement sur la tâche
- **Action** : exécution d'un outil
- **Observation** : résultat de l'action

```python
# Exemple de trace ReAct

Task: "Combien d'habitants compte Tokyo en 2026 ?"

Thought 1: Je dois chercher l'information la plus récente sur la population de Tokyo.
Action 1: Search["population Tokyo 2026"]
Observation 1: "La population de Tokyo en 2026 est estimée à 14,1 millions d'habitants dans les 23 arrondissements spéciaux."

Thought 2: J'ai trouvé l'information. Je peux maintenant répondre.
Action 2: Finish["Tokyo compte environ 14,1 millions d'habitants en 2026."]
```

**Avantages** :
- ✅ Réponses factuelles et vérifiables
- ✅ Transparence (on voit le raisonnement)
- ✅ Capacité à résoudre des tâches multi-étapes
- ✅ Auto-correction (si une action échoue, le LLM peut réessayer)

---

### 2.3 Implémentation de Base

```python
from typing import List, Dict, Callable, Optional
import re

class Tool:
    """Classe de base pour un outil."""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, query: str) -> str:
        """Exécute l'outil."""
        try:
            result = self.func(query)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class ReActAgent:
    """
    Agent ReAct simple.

    Alterne entre raisonnement (Thought), action (Action), et observation (Observation).
    """

    def __init__(self, llm, tools: List[Tool], max_steps: int = 10, verbose: bool = True):
        """
        Args:
            llm: Modèle de langage (OpenAI, Anthropic, etc.)
            tools: Liste d'outils disponibles
            max_steps: Nombre maximum d'itérations
            verbose: Afficher les traces
        """
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.verbose = verbose

    def _build_tool_description(self) -> str:
        """Construit la description des outils pour le prompt."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)

    def _parse_action(self, text: str) -> Optional[tuple]:
        """
        Parse l'action du LLM.

        Format attendu : Action: ToolName[argument]

        Returns:
            (tool_name, argument) ou None si format invalide
        """
        # Regex pour capturer : Action: ToolName[argument]
        match = re.search(r'Action:\s*(\w+)\[(.*?)\]', text, re.DOTALL)
        if match:
            tool_name = match.group(1)
            argument = match.group(2).strip()
            return (tool_name, argument)
        return None

    def _check_finish(self, text: str) -> Optional[str]:
        """Vérifie si le LLM a terminé."""
        match = re.search(r'Action:\s*Finish\[(.*?)\]', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def run(self, task: str) -> str:
        """
        Exécute une tâche en utilisant ReAct.

        Args:
            task: Description de la tâche

        Returns:
            Réponse finale
        """
        # Historique des pensées/actions/observations
        scratchpad = []

        for step in range(self.max_steps):
            # Construire le prompt
            prompt = self._build_prompt(task, scratchpad)

            # Générer la réponse du LLM
            response = self.llm.generate(prompt)

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"STEP {step + 1}")
                print(f"{'='*60}")
                print(response)

            # Vérifier si terminé
            final_answer = self._check_finish(response)
            if final_answer:
                if self.verbose:
                    print(f"\n✅ FINAL ANSWER: {final_answer}")
                return final_answer

            # Parser l'action
            action = self._parse_action(response)
            if not action:
                scratchpad.append(f"Error: Could not parse action from response.")
                continue

            tool_name, argument = action

            # Exécuter l'outil
            if tool_name not in self.tools:
                observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            else:
                tool = self.tools[tool_name]
                observation = tool.run(argument)

            if self.verbose:
                print(f"\nObservation: {observation}")

            # Ajouter au scratchpad
            scratchpad.append(f"{response}\nObservation: {observation}")

        # Max steps atteint
        return f"Failed to complete task within {self.max_steps} steps."

    def _build_prompt(self, task: str, scratchpad: List[str]) -> str:
        """Construit le prompt pour le LLM."""
        tools_desc = self._build_tool_description()
        history = "\n\n".join(scratchpad) if scratchpad else "No actions yet."

        prompt = f"""You are an AI agent that can use tools to accomplish tasks.

Available tools:
{tools_desc}
- Finish: Use when you have the final answer. Format: Action: Finish[answer]

Instructions:
1. Think step by step about what you need to do
2. Choose an appropriate tool and provide an argument
3. Observe the result
4. Repeat until you can provide a final answer

Format:
Thought: [your reasoning]
Action: ToolName[argument]

Task: {task}

Previous actions:
{history}

Now, what should you do next?

Thought:"""

        return prompt


# --- Définition des outils ---

def calculator(expression: str) -> float:
    """Évalue une expression mathématique."""
    # Sécurisé : utilise ast.literal_eval pour éviter l'exécution de code arbitraire
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow
    }

    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return -eval_expr(node.operand)
        else:
            raise ValueError(f"Unsupported expression: {node}")

    tree = ast.parse(expression, mode='eval')
    return eval_expr(tree.body)


def web_search(query: str) -> str:
    """Recherche sur le web (simulation)."""
    # En production, utiliser une vraie API (SerpAPI, Google Custom Search, etc.)
    mock_results = {
        "population Tokyo 2026": "La population de Tokyo en 2026 est estimée à 14,1 millions d'habitants.",
        "capital of France": "The capital of France is Paris.",
        "Python release date": "Python was first released on February 20, 1991.",
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value

    return f"No results found for '{query}'."


def python_interpreter(code: str) -> str:
    """Exécute du code Python (ATTENTION : dangereux en production sans sandbox)."""
    # En production : utiliser un environnement isolé (Docker, E2B, etc.)
    try:
        # Rediriger stdout
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Exécuter
        exec(code, {"__builtins__": __builtins__})

        # Récupérer l'output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return output if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {str(e)}"


# --- Exemple d'utilisation ---

# Mock LLM pour l'exemple (en production, utiliser OpenAI/Anthropic)
class MockLLM:
    """LLM simulé pour la démo."""

    def __init__(self):
        self.step = 0
        self.responses = [
            # Step 1 : Calcul simple
            """Thought: I need to calculate 157 * 23. I'll use the Calculator tool.
Action: Calculator[157 * 23]""",

            # Step 2 : Réponse finale
            """Thought: The calculator returned 3611. This is the final answer.
Action: Finish[157 × 23 = 3611]"""
        ]

    def generate(self, prompt: str) -> str:
        response = self.responses[self.step] if self.step < len(self.responses) else "Action: Finish[Done]"
        self.step += 1
        return response


# Créer les outils
tools = [
    Tool("Calculator", "Evaluates mathematical expressions. Example: Calculator[2+2]", calculator),
    Tool("Search", "Searches the web for information. Example: Search[population of Tokyo]", web_search),
    Tool("Python", "Executes Python code. Example: Python[print(2+2)]", python_interpreter),
]

# Créer l'agent
llm = MockLLM()
agent = ReActAgent(llm, tools, max_steps=5, verbose=True)

# Exécuter une tâche
result = agent.run("What is 157 multiplied by 23?")
print(f"\n\n🎯 RÉSULTAT FINAL : {result}")
```

**Sortie** :
```
============================================================
STEP 1
============================================================
Thought: I need to calculate 157 * 23. I'll use the Calculator tool.
Action: Calculator[157 * 23]

Observation: 3611

============================================================
STEP 2
============================================================
Thought: The calculator returned 3611. This is the final answer.
Action: Finish[157 × 23 = 3611]

✅ FINAL ANSWER: 157 × 23 = 3611


🎯 RÉSULTAT FINAL : 157 × 23 = 3611
```

---

## 3. Architectures d'Agents Avancées

### 3.1 Agent avec Mémoire (Conversationnel)

Un agent sans mémoire oublie tout entre les tâches. Ajoutons une **mémoire persistante** :

```python
class MemoryReActAgent(ReActAgent):
    """Agent ReAct avec mémoire conversationnelle."""

    def __init__(self, llm, tools, max_steps=10, verbose=True):
        super().__init__(llm, tools, max_steps, verbose)
        self.conversation_history = []

    def run(self, task: str) -> str:
        """Exécute une tâche en utilisant l'historique de conversation."""
        # Ajouter la tâche à l'historique
        self.conversation_history.append(f"User: {task}")

        # Exécuter ReAct (en incluant l'historique dans le prompt)
        result = super().run(task)

        # Sauvegarder la réponse
        self.conversation_history.append(f"Assistant: {result}")

        return result

    def _build_prompt(self, task: str, scratchpad: List[str]) -> str:
        """Override pour inclure l'historique conversationnel."""
        tools_desc = self._build_tool_description()
        history = "\n\n".join(scratchpad) if scratchpad else "No actions yet."

        # Historique conversationnel
        conv_history = "\n".join(self.conversation_history[-10:])  # Garder les 10 derniers tours

        prompt = f"""You are an AI agent with memory of previous conversations.

Conversation history:
{conv_history}

Available tools:
{tools_desc}
- Finish: Use when you have the final answer. Format: Action: Finish[answer]

Current task: {task}

Previous actions for this task:
{history}

What should you do next?

Thought:"""

        return prompt


# Exemple : conversation multi-tours
agent_with_memory = MemoryReActAgent(MockLLM(), tools, verbose=False)

# Tour 1
response1 = agent_with_memory.run("What is the capital of France?")
print(f"User: What is the capital of France?")
print(f"Agent: {response1}\n")

# Tour 2 (référence au tour précédent)
response2 = agent_with_memory.run("What is its population?")
print(f"User: What is its population?")
print(f"Agent: {response2}")
# L'agent sait que "its" fait référence à Paris grâce à la mémoire
```

---

### 3.2 Multi-Agent Systems

Plusieurs agents spécialisés collaborent pour résoudre une tâche complexe.

```python
class MultiAgentSystem:
    """
    Système multi-agents : chaque agent a une spécialité.

    Exemple :
    - ResearchAgent : cherche des informations
    - CodeAgent : écrit du code
    - AnalysisAgent : analyse des données
    """

    def __init__(self, agents: Dict[str, ReActAgent]):
        """
        Args:
            agents: Dictionnaire {nom_agent: agent}
        """
        self.agents = agents
        self.coordinator_llm = None  # LLM qui décide quel agent utiliser

    def run(self, task: str) -> str:
        """
        Coordonne plusieurs agents pour accomplir une tâche.

        Args:
            task: Tâche à accomplir

        Returns:
            Résultat final
        """
        # 1. Le coordinateur décide quel agent utiliser
        agent_choice = self._choose_agent(task)

        # 2. Déléguer la tâche à l'agent choisi
        selected_agent = self.agents[agent_choice]
        result = selected_agent.run(task)

        return result

    def _choose_agent(self, task: str) -> str:
        """Choisit l'agent approprié pour la tâche."""
        # Simplifié : basé sur des mots-clés
        task_lower = task.lower()

        if "search" in task_lower or "find" in task_lower:
            return "research"
        elif "code" in task_lower or "python" in task_lower:
            return "code"
        elif "analyze" in task_lower or "calculate" in task_lower:
            return "analysis"
        else:
            return "general"


# Exemple
research_agent = ReActAgent(llm, [Tool("Search", "...", web_search)])
code_agent = ReActAgent(llm, [Tool("Python", "...", python_interpreter)])
analysis_agent = ReActAgent(llm, [Tool("Calculator", "...", calculator)])

multi_system = MultiAgentSystem({
    "research": research_agent,
    "code": code_agent,
    "analysis": analysis_agent,
    "general": ReActAgent(llm, tools)
})

# Utilisation
result = multi_system.run("Search for the population of Tokyo")
# → Délègue automatiquement au research_agent
```

---

### 3.3 Plan-and-Execute

Au lieu de réagir à chaque étape, l'agent **planifie** d'abord toutes les étapes, puis les exécute.

```python
class PlanAndExecuteAgent:
    """
    Agent qui planifie avant d'agir.

    1. Décompose la tâche en sous-tâches
    2. Exécute chaque sous-tâche séquentiellement
    3. Ajuste le plan si nécessaire
    """

    def __init__(self, llm, tools):
        self.llm = llm
        self.executor = ReActAgent(llm, tools, verbose=False)

    def run(self, task: str) -> str:
        """
        Planifie puis exécute.

        Args:
            task: Tâche complexe

        Returns:
            Résultat final
        """
        # 1. Planifier
        plan = self._create_plan(task)
        print(f"📋 PLAN:\n{plan}\n")

        # 2. Exécuter chaque étape
        results = []
        for i, step in enumerate(plan):
            print(f"▶️  Executing step {i+1}: {step}")
            result = self.executor.run(step)
            results.append(result)
            print(f"✅ Result: {result}\n")

        # 3. Synthétiser
        final_answer = self._synthesize(task, results)
        return final_answer

    def _create_plan(self, task: str) -> List[str]:
        """Crée un plan d'action."""
        prompt = f"""Break down the following task into a sequence of simple steps.

Task: {task}

Steps:
1."""

        response = self.llm.generate(prompt)

        # Parser les étapes (simplifié)
        steps = [line.strip() for line in response.split('\n') if line.strip() and line[0].isdigit()]
        return steps

    def _synthesize(self, task: str, results: List[str]) -> str:
        """Synthétise les résultats."""
        prompt = f"""Given the following task and intermediate results, provide a final answer.

Task: {task}

Intermediate results:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(results))}

Final answer:"""

        return self.llm.generate(prompt)
```

---

## 4. Gestion des Erreurs et Robustesse

### 🎭 Dialogue : Quand Ça Se Passe Mal

**Alice** : Bob, j'ai implémenté un agent ReAct, mais parfois il tourne en boucle ou génère des erreurs bizarres. Comment gérer ça ?

**Bob** : Très bonne question ! Les agents peuvent échouer de plusieurs manières :

**Bob** : 1. **Boucles infinies** : l'agent répète la même action sans progresser.

**Alice** : Comment détecter ça ?

**Bob** : On garde un historique des actions. Si la même action est répétée 3 fois de suite, on intervient.

**Bob** : 2. **Hallucination d'outils** : l'agent invente un outil qui n'existe pas.

**Alice** : Genre "Action: MagicSolver[problem]" ?

**Bob** : Exactement ! Solution : valider que l'outil existe avant d'exécuter, et retourner un message d'erreur clair.

**Bob** : 3. **Arguments invalides** : l'agent utilise le bon outil mais avec de mauvais arguments.

**Alice** : Comme "Calculator[deux plus deux]" au lieu de "Calculator[2+2]" ?

**Bob** : Précisément. Il faut valider les arguments et donner des exemples clairs dans le prompt.

**Bob** : 4. **Timeout** : certains outils (recherche web, API) peuvent prendre trop de temps.

**Alice** : On met un timeout sur chaque outil ?

**Bob** : Oui, et on retourne une observation comme "Error: Tool timed out after 30s".

**Alice** : Et si aucune de ces solutions ne fonctionne ?

**Bob** : On a toujours un **max_steps**. Après N itérations, on arrête et on retourne "Task failed" avec les logs pour debug.

---

### 4.1 Implémentation Robuste

```python
from collections import Counter
import time

class RobustReActAgent(ReActAgent):
    """Agent ReAct avec gestion d'erreurs avancée."""

    def __init__(self, llm, tools, max_steps=10, verbose=True,
                 tool_timeout=30, max_retries=2):
        super().__init__(llm, tools, max_steps, verbose)
        self.tool_timeout = tool_timeout
        self.max_retries = max_retries
        self.action_history = []

    def run(self, task: str) -> str:
        """Exécute avec gestion d'erreurs robuste."""
        scratchpad = []

        for step in range(self.max_steps):
            try:
                # Vérifier les boucles infinies
                if self._is_stuck():
                    return self._handle_stuck(task, scratchpad)

                # Générer la réponse
                prompt = self._build_prompt(task, scratchpad)
                response = self.llm.generate(prompt)

                if self.verbose:
                    print(f"\n{'='*60}\nSTEP {step + 1}\n{'='*60}\n{response}")

                # Vérifier si terminé
                final_answer = self._check_finish(response)
                if final_answer:
                    return final_answer

                # Parser l'action
                action = self._parse_action(response)
                if not action:
                    scratchpad.append(self._handle_parse_error(response))
                    continue

                tool_name, argument = action
                self.action_history.append((tool_name, argument))

                # Exécuter l'outil avec timeout et retry
                observation = self._execute_tool_safe(tool_name, argument)

                if self.verbose:
                    print(f"\nObservation: {observation}")

                scratchpad.append(f"{response}\nObservation: {observation}")

            except Exception as e:
                # Erreur inattendue
                error_msg = f"Unexpected error in step {step}: {str(e)}"
                print(f"⚠️  {error_msg}")
                scratchpad.append(f"Error: {error_msg}")
                continue

        return f"Failed to complete task within {self.max_steps} steps."

    def _is_stuck(self) -> bool:
        """Détecte si l'agent est bloqué dans une boucle."""
        if len(self.action_history) < 3:
            return False

        # Vérifier si les 3 dernières actions sont identiques
        last_3 = self.action_history[-3:]
        if len(set(last_3)) == 1:
            return True

        # Vérifier si on alterne entre 2 actions (A->B->A->B)
        if len(self.action_history) >= 4:
            last_4 = self.action_history[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
                return True

        return False

    def _handle_stuck(self, task: str, scratchpad: List[str]) -> str:
        """Gère le cas où l'agent est bloqué."""
        print("⚠️  Agent appears to be stuck in a loop. Attempting recovery...")

        # Demander au LLM de réfléchir différemment
        recovery_prompt = f"""You seem to be stuck repeating the same actions.

Task: {task}

Previous actions:
{chr(10).join(str(a) for a in self.action_history[-5:])}

Think of a DIFFERENT approach to solve this task. What else could you try?

Thought:"""

        response = self.llm.generate(recovery_prompt)

        return f"Recovery attempt: {response}"

    def _execute_tool_safe(self, tool_name: str, argument: str) -> str:
        """Exécute un outil avec timeout et retry."""
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Error: Tool '{tool_name}' not found. Available tools: {available}"

        tool = self.tools[tool_name]

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                result = tool.run(argument)
                return result

            except TimeoutError as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️  Tool timed out, retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(1)
                else:
                    return f"Error: {str(e)}"

            except Exception as e:
                return f"Error executing {tool_name}: {str(e)}"

        return "Error: Max retries exceeded"

    def _handle_parse_error(self, response: str) -> str:
        """Gère les erreurs de parsing."""
        return f"""Error: Could not parse action from response.

Your response: {response}

Please use the correct format:
Thought: [your reasoning]
Action: ToolName[argument]

Example:
Thought: I need to search for information about Python.
Action: Search[Python programming language]"""
```

---

## 5. Optimisation des Coûts et Performances

### 5.1 Le Problème du Coût

Chaque appel au LLM coûte de l'argent :
- GPT-4 : ~$0.03 / 1K tokens input, $0.06 / 1K tokens output
- Un agent qui fait 10 itérations avec 2K tokens/itération = 20K tokens
- Coût : ~$0.60-$1.20 par tâche

**Solution 1** : Caching des résultats

```python
import hashlib
import json

class CachedTool(Tool):
    """Outil avec cache pour éviter les appels redondants."""

    def __init__(self, name, description, func):
        super().__init__(name, description, func)
        self.cache = {}

    def run(self, query: str) -> str:
        """Exécute avec cache."""
        # Hash de la query
        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self.cache:
            print(f"💾 Cache hit for {self.name}")
            return self.cache[cache_key]

        # Exécuter
        result = super().run(query)

        # Sauvegarder
        self.cache[cache_key] = result

        return result
```

**Solution 2** : Utiliser un modèle plus petit pour les tâches simples

```python
class HybridAgent(ReActAgent):
    """Agent qui utilise GPT-4 pour la planification, GPT-3.5 pour l'exécution."""

    def __init__(self, planner_llm, executor_llm, tools, **kwargs):
        super().__init__(planner_llm, tools, **kwargs)
        self.executor_llm = executor_llm

    def run(self, task: str) -> str:
        """Utilise le planner pour décider, l'executor pour agir."""
        # Étape 1 : Planifier avec GPT-4 (cher mais intelligent)
        plan = self.planner_llm.generate(f"Create a plan for: {task}")

        # Étape 2 : Exécuter avec GPT-3.5 (moins cher)
        # ... (logique d'exécution)

        return result
```

**Solution 3** : Limiter la longueur du contexte

```python
def _build_prompt(self, task: str, scratchpad: List[str]) -> str:
    """Optimisé : garde uniquement les N dernières observations."""
    # Garder seulement les 3 dernières actions au lieu de tout l'historique
    recent_history = scratchpad[-3:] if len(scratchpad) > 3 else scratchpad

    # ... (reste du prompt)
```

---

## 🧠 Quiz Interactif

### Question 1
**Quelle est la différence entre un LLM et un Agent LLM ?**

A) Un Agent LLM est plus grand (plus de paramètres)
B) Un Agent LLM peut interagir avec des outils externes
C) Un Agent LLM est plus rapide
D) Aucune différence, ce sont des synonymes

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Un **LLM** (comme GPT-4) génère du texte basé sur un prompt. Il raisonne mais ne peut pas **agir**.

Un **Agent LLM** combine un LLM avec des **outils** (APIs, calculatrices, bases de données) qui lui permettent d'interagir avec l'environnement externe.

**Analogie** :
- LLM = Un expert qui réfléchit et conseille
- Agent LLM = Un assistant qui réfléchit ET exécute des actions
</details>

---

### Question 2
**Que signifie "ReAct" ?**

A) Reactive Acting
B) Reasoning + Acting
C) Real-time Action
D) Recursive Activation

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

**ReAct** = **Rea**soning + **Act**ing

C'est un framework où l'agent alterne entre :
1. **Thought** (Reasoning) : réfléchir à la prochaine étape
2. **Action** (Acting) : exécuter un outil
3. **Observation** : observer le résultat

Cette boucle continue jusqu'à obtenir la réponse finale.

**Paper original** : "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
</details>

---

### Question 3
**Pourquoi un agent peut-il tomber dans une boucle infinie ?**

A) Le LLM oublie ce qu'il a déjà fait
B) Les outils donnent toujours les mêmes résultats
C) L'agent répète la même action sans progresser vers la solution
D) C'est impossible avec les agents modernes

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : C**

Un agent peut se bloquer en répétant les mêmes actions si :
- Le LLM ne réalise pas que l'approche ne fonctionne pas
- Les observations ne fournissent pas assez d'informations pour progresser
- Le prompt ne guide pas suffisamment le LLM

**Solutions** :
1. Détecter les répétitions dans l'historique d'actions
2. Limiter le nombre d'itérations (`max_steps`)
3. Implémenter une stratégie de "recovery" (essayer une approche différente)
4. Améliorer le prompt pour encourager la diversité des approches
</details>

---

### Question 4
**Quel est l'avantage principal d'un système multi-agents ?**

A) Moins cher en tokens
B) Spécialisation : chaque agent est expert dans son domaine
C) Plus rapide
D) Aucun avantage, c'est juste plus complexe

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Un **système multi-agents** permet de :
- Avoir des agents **spécialisés** (ResearchAgent, CodeAgent, AnalysisAgent)
- Chaque agent a ses propres outils et expertise
- Déléguer les sous-tâches à l'agent le plus compétent
- Paralléliser les tâches (plusieurs agents travaillent en même temps)

**Exemple** :
- Une tâche complexe : "Analyser les tendances du marché crypto et générer un rapport Python"
- ResearchAgent → cherche les données
- AnalysisAgent → analyse les chiffres
- CodeAgent → génère le script Python
- WriterAgent → rédige le rapport final
</details>

---

### Question 5
**Comment optimiser les coûts d'un agent qui fait beaucoup d'appels LLM ?**

A) Utiliser un modèle plus petit pour les tâches simples
B) Cacher les résultats des outils
C) Limiter la longueur du contexte (garder seulement les dernières N actions)
D) Toutes les réponses ci-dessus

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : D**

Toutes ces stratégies réduisent les coûts :

**A) Modèle hybride** :
- GPT-4 pour la planification (tâches complexes)
- GPT-3.5-turbo pour l'exécution (tâches simples)
- Économie : ~10x moins cher pour les actions basiques

**B) Caching** :
- Si l'outil est appelé 2 fois avec le même argument, utiliser le résultat en cache
- Exemple : chercher "population Tokyo" → pas besoin de refaire l'API call

**C) Contexte limité** :
- Au lieu d'envoyer tout l'historique (1000+ tokens), garder seulement les 3-5 dernières actions
- Réduit la taille du prompt de 70-80%

**Bonus** : Batching (traiter plusieurs tâches en un seul appel si possible)
</details>

---

### Question 6
**Qu'est-ce qu'une architecture "Plan-and-Execute" ?**

A) L'agent planifie toutes les étapes avant de les exécuter
B) L'agent exécute d'abord, puis planifie
C) L'agent ne planifie jamais, il réagit uniquement
D) C'est juste un autre nom pour ReAct

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : A**

**Plan-and-Execute** :
1. **Phase de planification** : Le LLM décompose la tâche en sous-tâches
   - "Trouver la population de Tokyo"
   - "Diviser par 50,000"
   - "Formater la réponse"

2. **Phase d'exécution** : Exécuter chaque sous-tâche séquentiellement

**Avantages** :
- Plus structuré que ReAct (qui décide au fur et à mesure)
- Bon pour les tâches complexes nécessitant plusieurs étapes
- Permet de paralléliser certaines sous-tâches

**Inconvénients** :
- Moins flexible (si une étape échoue, le plan peut devenir invalide)
- Nécessite 2 appels LLM (planification + exécution)

**ReAct vs Plan-and-Execute** :
- ReAct = réactif, adaptatif, interleaved reasoning
- Plan-and-Execute = proactif, structuré, upfront planning
</details>

---

## 💻 Exercices Pratiques

### Exercice 1 : Créer un Agent Multi-Outils

**Objectif** : Implémenter un agent ReAct avec 3 outils : Calculator, Wikipedia Search, et Weather API.

**Consignes** :
1. Implémenter les 3 outils
2. Créer un agent ReAct
3. Tester avec des tâches complexes nécessitant plusieurs outils

<details>
<summary>👉 Voir la solution complète</summary>

Solution fournie dans le code ci-dessus. Utilisez la classe `ReActAgent` avec les outils appropriés et testez avec des tâches multi-étapes comme calculer des intérêts composés combinés avec des recherches Wikipedia.
</details>

---

### Exercice 2 : Implémenter la Détection de Boucles

**Objectif** : Améliorer l'agent pour détecter et gérer les boucles infinies.

<details>
<summary>👉 Voir la solution</summary>

Voir la classe `RobustReActAgent` implémentée dans la section 4.1.

**Points clés** :
1. Garder un historique des actions
2. Détecter si les 3 dernières actions sont identiques
3. Détecter les alternances A→B→A→B
4. Proposer une stratégie de recovery
</details>

---

## 📚 Résumé du Chapitre

### Points Clés

1. **Agent LLM** = LLM (cerveau) + Outils (mains) + Boucle de contrôle

2. **ReAct** = Reasoning (pensée) + Acting (action) + Observation
   - Alterne entre raisonnement et exécution d'outils
   - Transparent et traçable
   - Auto-correctif

3. **Architectures avancées** :
   - Agents avec mémoire (conversationnels)
   - Multi-agents (spécialisation)
   - Plan-and-Execute (planification upfront)

4. **Défis** :
   - Boucles infinies → détection et recovery
   - Erreurs d'outils → retry et fallback
   - Coûts → caching, modèles hybrides, contexte limité

5. **Production** :
   - Logging complet
   - Métriques (success rate, latence, coûts)
   - Persistence des traces
   - Gestion d'erreurs robuste

---

## 🚀 Prochaine Étape

Dans le **Chapitre 15 : Déploiement et Production**, nous explorerons :
- Servir un LLM en production (FastAPI, vLLM, TGI)
- Optimisations d'inférence (quantization, batching)
- Monitoring et observabilité
- Scaling horizontal et vertical
- Coûts et SLAs

**À très bientôt !** 🎉

---

## 📖 Références

### Papers Fondamentaux
1. Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*
2. Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*
3. Nakano et al. (2021). *WebGPT: Browser-assisted question-answering with human feedback*
4. Significant-Gravitas. *AutoGPT* (2023) — Premier agent autonome viral

### Frameworks
- **LangChain** : Framework Python pour agents LLM
- **AutoGPT** : Agent autonome open-source
- **BabyAGI** : Agent minimaliste avec planification
- **AgentGPT** : Interface web pour agents autonomes

### Outils Utiles
- **SerpAPI** : API de recherche Google
- **E2B** : Environnement d'exécution de code sécurisé
- **LangSmith** : Debugging et monitoring d'agents

---

*Fin du Chapitre 14*
