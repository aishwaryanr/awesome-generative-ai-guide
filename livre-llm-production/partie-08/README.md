# Partie 8 : Outils, agents et intégration avancée

## Objectifs d'apprentissage

- Implémenter tool use et function calling
- Construire un système RAG robuste et fiable
- Orchestrer des agents multi-outils
- Gérer des mémoires prolongées et personnalisation

## Prérequis

- Parties 1-7 validées
- Compréhension des APIs et JSON schemas
- Bases de données vectorielles (FAISS, Pinecone, etc.)

**Références** : Gupta et al. (4.1) - RAG, Watson et al. (6.1) - Outils et agents

---

## 8.1 Tool Use et Function Calling

### 8.1.1 Principe

**Objectif** : Permettre au LLM d'appeler des fonctions externes pour accomplir des tâches.

**Workflow** :

```
User query → LLM détecte besoin d'outil
           → Génère appel de fonction (JSON)
           → Fonction exécutée
           → Résultat retourné au LLM
           → LLM génère réponse finale
```

### 8.1.2 Définir des outils avec JSON Schema

```python
tools = [
    {
        "name": "get_weather",
        "description": "Obtenir la météo pour une ville donnée",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Nom de la ville"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Effectuer un calcul mathématique",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Expression mathématique"}
            },
            "required": ["expression"]
        }
    }
]
```

### 8.1.3 Implémentation avec OpenAI function calling

```python
import openai
import json

def execute_function(function_name, arguments):
    """Router vers la bonne fonction."""
    if function_name == "get_weather":
        return get_weather(**arguments)
    elif function_name == "calculate":
        return calculate(**arguments)
    else:
        return {"error": "Unknown function"}

# Interaction
messages = [{"role": "user", "content": "Quelle est la météo à Paris ?"}]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    functions=tools,
    function_call="auto"
)

# Si le modèle veut appeler une fonction
if response.choices[0].finish_reason == "function_call":
    function_call = response.choices[0].message.function_call
    function_name = function_call.name
    arguments = json.loads(function_call.arguments)

    # Exécuter
    result = execute_function(function_name, arguments)

    # Retourner le résultat au modèle
    messages.append(response.choices[0].message)
    messages.append({
        "role": "function",
        "name": function_name,
        "content": json.dumps(result)
    })

    # Génération finale
    final_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    print(final_response.choices[0].message.content)
```

### 8.1.4 Validation et sécurité

**Validation de schéma** :

```python
from jsonschema import validate, ValidationError

def safe_execute_function(function_name, arguments, schema):
    """Valider avant d'exécuter."""
    try:
        validate(instance=arguments, schema=schema)
    except ValidationError as e:
        return {"error": f"Invalid arguments: {e.message}"}

    # Exécuter seulement si validation OK
    return execute_function(function_name, arguments)
```

**Contrôles d'accès** :

```python
# Liste blanche de fonctions autorisées
ALLOWED_FUNCTIONS = {"get_weather", "calculate", "search_web"}

def execute_with_permission_check(function_name, arguments, user_permissions):
    if function_name not in ALLOWED_FUNCTIONS:
        return {"error": "Function not allowed"}

    if function_name not in user_permissions:
        return {"error": "Permission denied"}

    return execute_function(function_name, arguments)
```

---

## 8.2 Retrieval-Augmented Generation (RAG)

**Référence** : Gupta et al. (4.1)

### 8.2.1 Architecture RAG

```
Query → [Retriever] → Documents pertinents
                             ↓
                    [LLM Generator] → Réponse augmentée
```

**Avantages** :
- Connaissances à jour (pas figées dans les poids)
- Traçabilité (citations de sources)
- Réduction des hallucinations

### 8.2.2 Indexation vectorielle

**Workflow** :

1. Chunker les documents
2. Générer des embeddings
3. Stocker dans une base vectorielle

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Charger un modèle d'embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Chunker les documents
def chunk_document(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

documents = [...]  # Liste de documents
chunks = []
for doc in documents:
    chunks.extend(chunk_document(doc))

# 3. Générer embeddings
embeddings = embedder.encode(chunks)

# 4. Créer index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))
```

### 8.2.3 Retrieval et reranking

**Retrieval** :

```python
def retrieve(query, index, chunks, k=5):
    """Récupérer les k chunks les plus pertinents."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), k)

    results = [(chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return results
```

**Reranking** (avec cross-encoder) :

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, candidates, top_k=3):
    """Reranker avec un modèle plus précis."""
    pairs = [[query, cand] for cand in candidates]
    scores = reranker.predict(pairs)

    # Trier par score
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [cand for cand, score in ranked[:top_k]]
```

### 8.2.4 Génération augmentée

```python
def rag_generate(query, index, chunks, llm):
    """Pipeline RAG complet."""
    # 1. Retrieval
    retrieved = retrieve(query, index, chunks, k=10)
    candidates = [chunk for chunk, dist in retrieved]

    # 2. Reranking
    relevant_chunks = rerank(query, candidates, top_k=3)

    # 3. Construire le contexte
    context = "\n\n".join(relevant_chunks)

    # 4. Génération
    prompt = f"""Contexte:
{context}

Question: {query}

Réponse basée sur le contexte ci-dessus:"""

    response = llm.generate(prompt)
    return response, relevant_chunks  # Retourner aussi les sources
```

### 8.2.5 Évaluation RAG

**Métriques** (référence 4.1) :

1. **Pertinence** : Les documents récupérés sont-ils pertinents ?
2. **Exactitude** : La réponse est-elle correcte ?
3. **Fidélité** : La réponse est-elle fidèle aux sources ?
4. **Complétude** : Toutes les informations nécessaires sont-elles présentes ?

```python
def evaluate_rag(query, gold_answer, rag_response, retrieved_docs):
    """Évaluer la qualité d'une réponse RAG."""
    metrics = {}

    # Pertinence (retrieval)
    # TODO: Mesurer si retrieved_docs contiennent la réponse

    # Exactitude (comparer à gold_answer)
    # TODO: Utiliser un modèle pour scorer la similarité

    # Fidélité (la réponse est-elle supportée par les docs ?)
    # TODO: NLI model pour vérifier entailment

    return metrics
```

---

## 8.3 Agents et orchestration

**Référence** : Watson et al. (6.1)

### 8.3.1 Architecture d'agent

**Boucle perception-action-observation** :

```
Percevoir l'état → Planifier → Agir → Observer résultat → Répéter
```

### 8.3.2 Agent simple avec ReAct

**ReAct** : Reasoning + Acting

```
Pensée: Je dois chercher la météo à Paris
Action: get_weather("Paris")
Observation: 15°C, ensoleillé
Pensée: Maintenant je peux répondre
Réponse finale: Il fait 15°C et ensoleillé à Paris.
```

**Implémentation** :

```python
def react_agent(query, tools, max_iterations=5):
    """Agent ReAct simple."""
    messages = [{"role": "user", "content": query}]

    for i in range(max_iterations):
        # Générer pensée + action
        response = llm.generate(messages + [
            {"role": "system", "content": "Utilise le format:\nPensée: [ton raisonnement]\nAction: [fonction à appeler]\nObservation: [résultat]"}
        ])

        # Parser
        if "Action:" in response:
            action_line = [l for l in response.split('\n') if l.startswith('Action:')][0]
            action_call = parse_action(action_line)  # Extraire fonction et args

            # Exécuter
            observation = execute_function(action_call['function'], action_call['args'])

            # Ajouter observation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "system", "content": f"Observation: {observation}"})

        else:
            # Réponse finale
            return response

    return "Max iterations atteintes"
```

### 8.3.3 Frameworks d'orchestration

**LangChain** :

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Définir outils
tools = [
    Tool(name="Calculator", func=calculate, description="Pour les calculs"),
    Tool(name="Search", func=search_web, description="Pour chercher sur le web")
]

# Initialiser agent
llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Exécuter
result = agent.run("Quelle est la capitale de la France ? Et quelle est sa population au carré ?")
```

**AutoGPT pattern** (boucles autonomes) :

```python
class AutoAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, goal, max_steps=10):
        """Exécuter jusqu'à atteinte du goal."""
        for step in range(max_steps):
            # Récupérer contexte
            context = self.memory.get_relevant_context(goal)

            # Planifier prochaine action
            action = self.llm.plan(goal, context)

            # Exécuter
            result = self.execute_action(action)

            # Mémoriser
            self.memory.add(action, result)

            # Vérifier si goal atteint
            if self.is_goal_achieved(goal, self.memory):
                return self.memory.summarize()

        return "Goal non atteint"
```

---

## 8.4 Mémoires prolongées et personnalisation

### 8.4.1 Types de mémoire

**Mémoire à court terme** : Contexte de la conversation actuelle

**Mémoire à long terme** :
- **Épisodique** : Historique des interactions
- **Sémantique** : Faits et connaissances extraits
- **Procédurale** : Préférences et patterns d'utilisation

### 8.4.2 Implémentation avec vector store

```python
class LongTermMemory:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def add(self, text, metadata=None):
        """Ajouter à la mémoire."""
        embedding = self.embedder.encode([text])[0]
        self.vector_store.add(embedding, {"text": text, "metadata": metadata})

    def recall(self, query, k=5):
        """Récupérer des souvenirs pertinents."""
        query_embedding = self.embedder.encode([query])[0]
        results = self.vector_store.search(query_embedding, k=k)
        return [r['text'] for r in results]
```

### 8.4.3 Profils utilisateurs

```python
class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = {}
        self.conversation_history = []

    def update_preference(self, key, value):
        self.preferences[key] = value

    def get_context_for_llm(self):
        """Contexte personnalisé."""
        return f"""Profil utilisateur:
- ID: {self.user_id}
- Préférences: {self.preferences}
- Historique récent: {self.conversation_history[-5:]}
"""
```

---

## 8.5 Labs pratiques

### Lab 1 : Pipeline RAG complet

1. Indexer un corpus de documents (ex: Wikipedia articles)
2. Implémenter retrieval + reranking
3. Générer des réponses augmentées
4. Évaluer avec métriques (pertinence, fidélité)

### Lab 2 : Agent multi-outils

Construire un agent capable de :
- Chercher sur le web
- Faire des calculs
- Accéder à une base de données
- Synthétiser les résultats

### Lab 3 : Système avec mémoire

Créer un chatbot avec :
- Mémoire de conversation
- Profil utilisateur persistant
- Rappel de contexte pertinent

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Tool use et function calling avec validation
- ✅ RAG complet (indexation, retrieval, reranking, génération)
- ✅ Agents et orchestration (ReAct, LangChain)
- ✅ Mémoires prolongées et personnalisation

**Prochaine étape** : [Partie 9 - Inference et optimisation](../partie-09/README.md)
