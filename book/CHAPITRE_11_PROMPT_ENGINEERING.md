# CHAPITRE 11 : PROMPT ENGINEERING - L'ART DE PARLER AUX LLMs

> *« Le prompt engineering, c'est transformer un LLM généraliste en expert spécialisé... sans une seule ligne de code. »*

---

## Introduction : La Communication Homme-Machine Réinventée

### 🎭 Dialogue : Le Pouvoir des Mots

**Alice** : Bob, j'ai essayé ChatGPT pour générer du code Python. Parfois c'est brillant, parfois c'est nul. Pourquoi ?

**Bob** : Montre-moi tes prompts.

**Alice** : "Écris du code pour trier une liste"

**Bob** : Voilà ton problème. Compare avec :

```
Prompt amélioré:
"Tu es un expert Python. Écris une fonction `sort_list(items)` qui:
1. Prend une liste de nombres en entrée
2. La trie par ordre croissant
3. Retourne la liste triée
4. Inclut docstring et tests unitaires
5. Utilise la complexité optimale O(n log n)

Exemple d'utilisation:
>>> sort_list([3, 1, 4, 1, 5])
[1, 1, 3, 4, 5]"
```

**Alice** : Wow, ça change tout !

**Bob** : Exactement. Le **prompt engineering** transforme un modèle médiocre en assistant brillant. C'est l'interface entre ton intention et le modèle.

### 📊 Évolution du Prompting

| Ère | Méthode | Exemple | Performance |
|-----|---------|---------|-------------|
| **2018-2020** | Zero-shot simple | "Traduis en anglais: Bonjour" | Faible |
| **2020-2021** | Few-shot | 3 exemples + tâche | Moyenne |
| **2021-2022** | Chain-of-Thought | "Pensons étape par étape..." | Bonne |
| **2022-2023** | Advanced (ReAct, ToT) | Raisonnement + Actions | Excellente |
| **2023+** | Multimodal + Tools | Texte + Images + API calls | SOTA |

### 🎯 Anecdote : GPT-3 et le "Let's think step by step"

**Mai 2022, Google Research**

Kojima et al. testent GPT-3 sur des problèmes de maths. Performance : **17% accuracy**.

Puis ils ajoutent une phrase magique au prompt : **"Let's think step by step."**

**Résultat** : **78% accuracy** !

Aucun fine-tuning, aucun exemple. Juste 5 mots qui déclenchent le raisonnement du modèle.

**Impact** : Naissance du **Chain-of-Thought prompting**, technique devenue standard pour GPT-4, Claude, etc.

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Concevoir des prompts efficaces (structure, clarté, contexte)
- ✅ Appliquer few-shot learning pour des tâches spécifiques
- ✅ Utiliser Chain-of-Thought pour problèmes complexes
- ✅ Implémenter des techniques avancées (ReAct, Tree of Thoughts)
- ✅ Optimiser automatiquement vos prompts
- ✅ Gérer les hallucinations et biais
- ✅ Évaluer la qualité des prompts

**Difficulté** : 🟡🟡⚪⚪⚪ (Intermédiaire)
**Prérequis** : Utilisation basique d'un LLM (ChatGPT, Claude, etc.)
**Temps de lecture** : ~110 minutes

---

## Anatomie d'un Bon Prompt

### Les 6 Composants Essentiels

#### 1. Rôle (Persona)

**Principe** : Définir l'expertise du modèle.

```
❌ Mauvais: "Explique la relativité"

✅ Bon: "Tu es un physicien théoricien. Explique la relativité générale..."
```

**Exemples de rôles** :
- Expert technique : "Tu es un développeur senior Python avec 10 ans d'expérience"
- Pédagogue : "Tu es un professeur qui explique à un enfant de 10 ans"
- Créatif : "Tu es un romancier primé spécialisé en science-fiction"

#### 2. Tâche

**Principe** : Spécifier clairement l'action attendue.

```
❌ Vague: "Aide-moi avec ce texte"

✅ Précis: "Résume ce texte en 3 bullet points, en conservant les chiffres clés"
```

**Verbes d'action** :
- Analyse : Résume, Classifie, Compare, Évalue
- Création : Génère, Écris, Conçois, Imagine
- Transformation : Traduis, Reformule, Simplifie, Développe

#### 3. Contexte

**Principe** : Fournir les informations nécessaires.

```python
prompt = f"""
Contexte: Tu analyses des avis clients pour une boutique e-commerce.

Texte: "{customer_review}"

Tâche: Extraire le sentiment (positif/négatif/neutre) et les aspects mentionnés (prix, qualité, livraison).
"""
```

#### 4. Exemples (Few-Shot)

**Principe** : Montrer des exemples de sortie attendue.

```
Exemple 1:
Input: "Ce produit est cher mais la qualité est au rendez-vous"
Output: {"sentiment": "positif", "aspects": ["prix": "négatif", "qualité": "positif"]}

Exemple 2:
Input: "Livraison rapide, je recommande!"
Output: {"sentiment": "positif", "aspects": ["livraison": "positif"]}

Maintenant, analyse ceci:
Input: "{new_review}"
Output:
```

#### 5. Format de Sortie

**Principe** : Spécifier le format exact attendu.

```
❌ Vague: "Liste les capitales européennes"

✅ Précis: "Liste 5 capitales européennes au format JSON:
{
  "cities": [
    {"name": "Paris", "country": "France", "population": 2161000},
    ...
  ]
}"
```

#### 6. Contraintes

**Principe** : Définir les limites et exigences.

```
Contraintes:
- Maximum 200 mots
- Ton professionnel
- Éviter le jargon technique
- Inclure au moins 2 exemples concrets
- Format Markdown avec titres
```

### Template de Prompt Complet

```python
PROMPT_TEMPLATE = """
[RÔLE]
Tu es {role}.

[CONTEXTE]
{context}

[TÂCHE]
{task}

[EXEMPLES]
{examples}

[FORMAT]
Réponds au format suivant:
{output_format}

[CONTRAINTES]
- {constraint_1}
- {constraint_2}
- {constraint_3}

Maintenant, procède:
{input}
"""

# Utilisation
prompt = PROMPT_TEMPLATE.format(
    role="un analyste financier expert",
    context="Tu analyses des rapports trimestriels d'entreprises tech",
    task="Extraire les métriques clés (revenue, profit, croissance)",
    examples="...",
    output_format="JSON avec clés 'revenue', 'profit', 'growth_rate'",
    constraint_1="Chiffres en millions USD",
    constraint_2="Croissance en pourcentage",
    constraint_3="Ajouter comparaison vs trimestre précédent",
    input=company_report
)
```

---

## Zero-Shot, One-Shot, Few-Shot

### Zero-Shot : Sans Exemple

**Principe** : Le modèle infère la tâche depuis la description.

```python
def zero_shot_classification(text, labels):
    """
    Classification zero-shot.
    """
    prompt = f"""
Classifie le texte suivant dans une de ces catégories: {', '.join(labels)}

Texte: "{text}"

Catégorie:"""

    return prompt

# Exemple
text = "Ce film est absolument génial, j'ai adoré !"
labels = ["positif", "négatif", "neutre"]

prompt = zero_shot_classification(text, labels)
# Modèle devrait retourner: "positif"
```

**Quand utiliser** :
- ✅ Tâches simples et communes (sentiment, traduction)
- ✅ Modèles puissants (GPT-4, Claude)
- ❌ Tâches spécialisées ou format strict

### One-Shot : Un Exemple

```python
def one_shot_extraction(text):
    prompt = f"""
Extrais les entités (personnes, lieux, organisations) du texte.

Exemple:
Input: "Barack Obama a visité Paris en 2015 pour rencontrer le président français."
Output: {{
  "personnes": ["Barack Obama"],
  "lieux": ["Paris"],
  "organisations": [],
  "dates": ["2015"]
}}

Maintenant:
Input: "{text}"
Output:"""

    return prompt
```

### Few-Shot : Plusieurs Exemples

**Règle d'or** : 3-5 exemples optimaux.

```python
def few_shot_translation(text, source_lang, target_lang, examples):
    """
    Traduction few-shot avec exemples.
    """
    examples_str = "\n\n".join([
        f"{source_lang}: {ex['source']}\n{target_lang}: {ex['target']}"
        for ex in examples
    ])

    prompt = f"""
Traduis du {source_lang} vers le {target_lang}.

Exemples:
{examples_str}

Maintenant:
{source_lang}: {text}
{target_lang}:"""

    return prompt

# Utilisation
examples = [
    {"source": "Bonjour, comment allez-vous ?", "target": "Hello, how are you?"},
    {"source": "Je vais bien, merci.", "target": "I'm fine, thank you."},
    {"source": "Quelle heure est-il ?", "target": "What time is it?"}
]

prompt = few_shot_translation(
    "Où se trouve la gare ?",
    "Français",
    "Anglais",
    examples
)
```

### Sélection Dynamique d'Exemples

**Principe** : Choisir les exemples les plus similaires à l'input.

```python
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

class DynamicFewShot:
    """
    Sélectionne dynamiquement les meilleurs exemples.
    """
    def __init__(self, example_pool, num_examples=3):
        self.example_pool = example_pool
        self.num_examples = num_examples

        # Encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Pré-calculer embeddings des exemples
        self.example_texts = [ex['input'] for ex in example_pool]
        self.example_embeddings = self.encoder.encode(self.example_texts)

    def select_examples(self, query):
        """
        Sélectionne les N exemples les plus similaires.
        """
        # Embedding de la query
        query_embedding = self.encoder.encode([query])

        # Similarités
        similarities = cosine_similarity(query_embedding, self.example_embeddings)[0]

        # Top-N indices
        top_indices = np.argsort(similarities)[::-1][:self.num_examples]

        # Retourner exemples
        selected = [self.example_pool[i] for i in top_indices]
        return selected

# Utilisation
example_pool = [
    {"input": "Ce produit est excellent", "output": "positif"},
    {"input": "Très déçu de cet achat", "output": "négatif"},
    {"input": "Qualité correcte pour le prix", "output": "neutre"},
    # ... 100+ exemples
]

selector = DynamicFewShot(example_pool, num_examples=3)

# Pour une nouvelle query
query = "Je recommande vivement ce service"
selected_examples = selector.select_examples(query)

# Construire prompt avec exemples sélectionnés
# ...
```

---

## Chain-of-Thought (CoT) Prompting

### Principe : Décomposer le Raisonnement

**Sans CoT** :
```
Q: Roger a 5 balles de tennis. Il achète 2 boîtes de 3 balles. Combien a-t-il de balles maintenant ?
A: 11
```

**Avec CoT** :
```
Q: Roger a 5 balles de tennis. Il achète 2 boîtes de 3 balles. Combien a-t-il de balles maintenant ?
A: Réfléchissons étape par étape.
1. Roger commence avec 5 balles
2. Il achète 2 boîtes de 3 balles chacune
3. 2 boîtes × 3 balles = 6 balles
4. Total : 5 + 6 = 11 balles
La réponse est 11.
```

### Zero-Shot CoT : "Let's think step by step"

```python
def zero_shot_cot(question):
    """
    Chain-of-Thought zero-shot.
    """
    prompt = f"""
Question: {question}

Let's think step by step:"""

    return prompt

# Exemple
question = "Si un train part de Paris à 14h à 120 km/h et arrive à Lyon (450 km) à quelle heure ?"

prompt = zero_shot_cot(question)

# Réponse attendue:
# 1. Distance = 450 km
# 2. Vitesse = 120 km/h
# 3. Temps = Distance / Vitesse = 450 / 120 = 3.75 heures = 3h45min
# 4. Arrivée = 14h + 3h45min = 17h45
```

### Few-Shot CoT

```python
FEW_SHOT_COT_PROMPT = """
Q: Dans un café, il y a 23 clients. 17 partent et 9 arrivent. Combien reste-t-il de clients ?
A: Commençons par identifier ce qu'on sait :
- Au départ : 23 clients
- Partent : 17 clients
- Arrivent : 9 clients

Calculons étape par étape :
1. Après les départs : 23 - 17 = 6 clients
2. Après les arrivées : 6 + 9 = 15 clients

Réponse finale : 15 clients.

Q: Marie a 4 pommes. Elle en donne la moitié à Jean, puis achète 3 oranges. Combien de fruits a-t-elle ?
A: Décomposons le problème :
- Début : 4 pommes
- Donne la moitié à Jean : 4 / 2 = 2 pommes données, reste 2 pommes
- Achète 3 oranges

Calcul final :
- Pommes restantes : 2
- Oranges : 3
- Total fruits : 2 + 3 = 5 fruits

Réponse finale : 5 fruits.

Q: {question}
A: """
```

### Self-Consistency : Échantillonner Plusieurs Raisonnements

**Principe** : Générer plusieurs CoT, prendre la réponse majoritaire.

```python
import openai
from collections import Counter

def self_consistency_cot(question, num_samples=5, temperature=0.7):
    """
    Self-consistency avec Chain-of-Thought.
    """
    prompt = f"""
Question: {question}

Let's think step by step:"""

    answers = []

    for _ in range(num_samples):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )

        full_response = response.choices[0].message.content

        # Extraire réponse finale (simplifié)
        # En pratique : parser avec regex ou demander format structuré
        answer = extract_final_answer(full_response)
        answers.append(answer)

    # Vote majoritaire
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0][0]

    return {
        "answer": most_common,
        "confidence": answer_counts[most_common] / num_samples,
        "all_answers": dict(answer_counts)
    }

# Exemple
result = self_consistency_cot(
    "Un bus a 25 passagers. À l'arrêt 1, 8 descendent et 13 montent. À l'arrêt 2, 5 descendent. Combien reste-t-il de passagers ?"
)
print(f"Réponse: {result['answer']} (confiance: {result['confidence']:.0%})")
```

---

## Techniques Avancées

### ReAct : Reasoning + Acting

**Principe** : Alterner raisonnement et actions (appels API, recherche web, etc.).

```python
REACT_PROMPT = """
Tu résous des problèmes en alternant Pensée (Thought), Action, et Observation.

Outils disponibles:
- search(query): Recherche Google
- calculate(expression): Calculatrice
- wikipedia(topic): Recherche Wikipedia

Exemple:
Question: Quelle est la population de la capitale du Japon en 2023 ?

Thought: Je dois d'abord identifier la capitale du Japon
Action: wikipedia("Japon capitale")
Observation: La capitale du Japon est Tokyo

Thought: Maintenant je cherche la population de Tokyo en 2023
Action: search("population Tokyo 2023")
Observation: La population de Tokyo est environ 14 millions (2023)

Thought: J'ai la réponse
Final Answer: 14 millions d'habitants

---

Question: {question}

Thought:"""

def react_agent(question, max_iterations=5):
    """
    Agent ReAct simple.
    """
    prompt = REACT_PROMPT.format(question=question)
    history = []

    for i in range(max_iterations):
        # Générer pensée + action
        response = call_llm(prompt)

        # Parser
        thought, action, params = parse_react_response(response)

        history.append({"thought": thought, "action": action})

        # Exécuter action
        if action == "search":
            observation = google_search(params)
        elif action == "calculate":
            observation = eval(params)  # Attention: unsafe en production!
        elif action == "wikipedia":
            observation = wikipedia_search(params)
        elif action == "FINISH":
            return {"answer": params, "history": history}

        # Ajouter observation au prompt
        prompt += f"\nObservation: {observation}\n\nThought:"

    return {"answer": "Max iterations atteinte", "history": history}
```

### Tree of Thoughts (ToT)

**Principe** : Explorer plusieurs chemins de raisonnement en arbre.

```python
class TreeOfThoughts:
    """
    Tree of Thoughts pour exploration de solutions.
    """
    def __init__(self, problem, num_branches=3, depth=3):
        self.problem = problem
        self.num_branches = num_branches
        self.depth = depth

    def generate_thoughts(self, state, depth):
        """Génère N pensées possibles depuis un état."""
        prompt = f"""
Problème: {self.problem}

État actuel: {state}

Génère {self.num_branches} prochaines étapes de raisonnement possibles.
Format:
1. [Étape 1]
2. [Étape 2]
3. [Étape 3]
"""

        response = call_llm(prompt)
        thoughts = parse_thoughts(response)
        return thoughts

    def evaluate_thought(self, thought):
        """Évalue la promesse d'une pensée (0-10)."""
        prompt = f"""
Problème: {self.problem}
Pensée: {thought}

Sur une échelle de 0 à 10, évalue la probabilité que cette pensée mène à la solution correcte.
Score:"""

        response = call_llm(prompt)
        score = int(response.strip())
        return score

    def search(self):
        """Recherche en profondeur avec élagage."""
        best_solution = None
        best_score = -1

        def dfs(state, depth, path):
            nonlocal best_solution, best_score

            if depth == self.depth:
                # Évaluer solution finale
                score = self.evaluate_thought(state)
                if score > best_score:
                    best_score = score
                    best_solution = path
                return

            # Générer et évaluer pensées
            thoughts = self.generate_thoughts(state, depth)
            scored_thoughts = [(t, self.evaluate_thought(t)) for t in thoughts]

            # Prendre les meilleures
            sorted_thoughts = sorted(scored_thoughts, key=lambda x: x[1], reverse=True)

            # Explorer récursivement
            for thought, score in sorted_thoughts[:self.num_branches]:
                dfs(thought, depth + 1, path + [thought])

        dfs("", 0, [])
        return {"solution": best_solution, "score": best_score}

# Exemple
problem = "Résoudre: x^2 + 5x + 6 = 0"
tot = TreeOfThoughts(problem, num_branches=3, depth=3)
result = tot.search()
```

---

## Gestion des Hallucinations

### Techniques de Mitigation

#### 1. Demander des Citations

```python
CITATION_PROMPT = """
Réponds à la question suivante en citant tes sources.

Format:
Réponse: [Ta réponse]
Sources: [Citation 1], [Citation 2], ...

Si tu n'es pas sûr, dis "Je ne sais pas" plutôt que d'inventer.

Question: {question}
"""
```

#### 2. Contraindre avec Contexte

```python
RAG_PROMPT = """
Contexte fourni:
{context}

Règles:
- Réponds UNIQUEMENT en te basant sur le contexte ci-dessus
- Si l'information n'est pas dans le contexte, réponds "Information non disponible dans le contexte fourni"
- Cite les passages pertinents entre guillemets

Question: {question}
Réponse:"""
```

#### 3. Vérification Multi-Étapes

```python
def verify_response(question, answer):
    """
    Vérifie la cohérence d'une réponse.
    """
    verification_prompt = f"""
Question originale: {question}
Réponse donnée: {answer}

Tâches:
1. Vérifier si la réponse est cohérente avec la question
2. Identifier les affirmations factuelles dans la réponse
3. Évaluer la confiance pour chaque affirmation (faible/moyenne/élevée)
4. Signaler les affirmations potentiellement fausses

Format JSON:
{{
  "coherent": true/false,
  "claims": [
    {{"text": "...", "confidence": "élevée/moyenne/faible"}}
  ],
  "potential_hallucinations": [...]
}}
"""

    verification = call_llm(verification_prompt)
    return parse_verification(verification)
```

---

## Optimisation Automatique de Prompts

### Prompt Tuning : Recherche Automatique

```python
import itertools

class PromptOptimizer:
    """
    Optimise automatiquement un prompt via recherche.
    """
    def __init__(self, test_cases):
        """
        Args:
            test_cases: Liste de (input, expected_output)
        """
        self.test_cases = test_cases

    def evaluate_prompt(self, prompt_template):
        """Évalue un template de prompt."""
        correct = 0

        for input_data, expected in self.test_cases:
            prompt = prompt_template.format(input=input_data)
            output = call_llm(prompt)

            if self.is_correct(output, expected):
                correct += 1

        accuracy = correct / len(self.test_cases)
        return accuracy

    def is_correct(self, output, expected):
        """Vérifie si output correspond à expected."""
        # Implémentation dépend de la tâche
        # Peut être exact match, similarité sémantique, etc.
        return output.strip().lower() == expected.strip().lower()

    def optimize(self, prompt_variations):
        """
        Teste différentes variations de prompt.
        """
        results = []

        for variation in prompt_variations:
            accuracy = self.evaluate_prompt(variation)
            results.append((variation, accuracy))

        # Trier par accuracy
        results.sort(key=lambda x: x[1], reverse=True)

        return results

# Utilisation
test_cases = [
    ("Ce film est génial", "positif"),
    ("Quelle déception", "négatif"),
    ("Pas mal", "neutre"),
    # ... 50+ exemples
]

optimizer = PromptOptimizer(test_cases)

# Variations à tester
variations = [
    "Classifie le sentiment: {input}\nRéponse:",
    "Sentiment de ce texte: {input}\nRéponse:",
    "Analyse de sentiment:\nTexte: {input}\nSentiment:",
    "Tu es un expert en analyse de sentiment. Classifie:\n{input}\nRéponse:",
]

results = optimizer.optimize(variations)

print("Meilleur prompt:")
print(results[0][0])
print(f"Accuracy: {results[0][1]:.2%}")
```

### APE : Automatic Prompt Engineer

**Principe** : Utiliser un LLM pour générer et optimiser des prompts.

```python
def ape_generate_prompts(task_description, num_prompts=10):
    """
    Génère automatiquement des prompts candidats.
    """
    meta_prompt = f"""
Tâche: {task_description}

Génère {num_prompts} prompts différents pour accomplir cette tâche.
Chaque prompt doit être clair, précis et optimisé pour obtenir les meilleurs résultats.

Format:
1. [Prompt 1]
2. [Prompt 2]
...
"""

    response = call_llm(meta_prompt)
    prompts = parse_prompts(response)
    return prompts

# Exemple
task = "Extraire les noms de personnes mentionnées dans un texte"
candidate_prompts = ape_generate_prompts(task, num_prompts=5)

# Évaluer et sélectionner le meilleur
optimizer = PromptOptimizer(test_cases)
results = optimizer.optimize(candidate_prompts)
```

---

## Évaluation de Prompts

### Métriques

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class PromptEvaluator:
    """
    Évalue la qualité d'un prompt sur différentes métriques.
    """
    def __init__(self, test_set):
        self.test_set = test_set

    def evaluate(self, prompt_template):
        """
        Évalue un prompt.

        Returns:
            dict avec métriques
        """
        predictions = []
        ground_truth = []
        latencies = []
        costs = []

        for example in self.test_set:
            start_time = time.time()

            # Générer prédiction
            prompt = prompt_template.format(**example['input'])
            prediction = call_llm(prompt)

            # Métriques
            latency = time.time() - start_time
            cost = estimate_cost(prompt, prediction)

            predictions.append(prediction)
            ground_truth.append(example['output'])
            latencies.append(latency)
            costs.append(cost)

        # Calculer métriques
        accuracy = accuracy_score(ground_truth, predictions)
        avg_latency = np.mean(latencies)
        total_cost = sum(costs)

        return {
            'accuracy': accuracy,
            'avg_latency_ms': avg_latency * 1000,
            'total_cost_usd': total_cost,
            'cost_per_example': total_cost / len(self.test_set)
        }

# Utilisation
evaluator = PromptEvaluator(test_set)

prompt_v1 = "Classifie: {text}"
prompt_v2 = "Tu es un expert. Analyse le sentiment de: {text}"

results_v1 = evaluator.evaluate(prompt_v1)
results_v2 = evaluator.evaluate(prompt_v2)

print("Prompt V1:", results_v1)
print("Prompt V2:", results_v2)
```

---

## Bibliothèque de Prompts Réutilisables

### Classification

```python
CLASSIFICATION_PROMPT = """
Classifie le texte suivant dans une des catégories: {categories}

Texte: "{text}"

Réfléchis étape par étape:
1. Quel est le sujet principal ?
2. Quels mots-clés indiquent la catégorie ?
3. Quelle catégorie correspond le mieux ?

Catégorie:"""
```

### Extraction d'Information

```python
NER_PROMPT = """
Extrais les entités nommées du texte.

Texte: "{text}"

Retourne au format JSON:
{{
  "personnes": [...],
  "organisations": [...],
  "lieux": [...],
  "dates": [...]
}}

JSON:"""
```

### Génération de Code

```python
CODE_GENERATION_PROMPT = """
Tu es un expert programmeur {language}.

Tâche: {task}

Exigences:
- Code propre et commenté
- Gestion des erreurs
- Tests unitaires
- Complexité optimale
- Docstrings

Exemple d'utilisation:
{usage_example}

Code:
```{language}
"""
```

### Résumé

```python
SUMMARY_PROMPT = """
Résume le texte suivant en {num_sentences} phrases.

Texte:
{text}

Consignes:
- Conserver les informations clés
- Ton {tone}
- Maximum {max_words} mots

Résumé:"""
```

---

## 💡 Analogie : Le Prompt comme une Recette de Cuisine

- **Rôle** = Chef (italien, pâtissier, vegan...)
- **Tâche** = Type de plat (entrée, dessert)
- **Contexte** = Occasion (dîner formel, goûter enfants)
- **Exemples** = Photos du plat attendu
- **Format** = Présentation (assiette, portion)
- **Contraintes** = Allergies, budget, temps

Un bon prompt, comme une bonne recette, est :
- **Précis** : Quantités exactes, étapes claires
- **Contextualisé** : Adapté à la situation
- **Reproductible** : Même résultat à chaque fois
- **Optimisé** : Efficient en temps et ressources

---

## Conclusion

### 🎭 Dialogue Final : Le Prompt Engineering, Compétence Clé

**Alice** : Le prompt engineering, c'est vraiment un métier maintenant ?

**Bob** : Absolument ! En 2024, "Prompt Engineer" peut payer $200k+/an. Pourquoi ?
1. **Coût** : Un bon prompt économise des milliers en API calls
2. **Performance** : Différence entre 40% et 90% accuracy
3. **Rapidité** : Prompting vs fine-tuning = heures vs semaines

**Alice** : Quels sont les principes clés ?

**Bob** :
1. **Clarté** : Spécifique > vague
2. **Contexte** : Donner les informations nécessaires
3. **Exemples** : Few-shot > zero-shot pour tâches complexes
4. **Structure** : CoT pour raisonnement
5. **Itération** : Tester, mesurer, améliorer

**Alice** : Et le futur ?

**Bob** : **Prompts multimodaux** (texte + images + code), **auto-optimisation** (AI qui améliore ses propres prompts), **prompts universels** (fonctionnent sur GPT, Claude, LLaMA...).

Le prompt engineering évolue de l'art vers la science.

---

## Ressources

### 📚 Papers Fondamentaux

1. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** (Wei et al., 2022)
2. **"Large Language Models are Zero-Shot Reasoners"** (Kojima et al., 2022) - "Let's think step by step"
3. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022)
4. **"Tree of Thoughts: Deliberate Problem Solving with Large Language Models"** (Yao et al., 2023)

### 🛠️ Outils

```bash
# Frameworks de prompting
pip install langchain guidance

# Évaluation
pip install prompttools

# Optimisation
pip install dspy-ai
```

### 🔗 Ressources

- **Prompt Engineering Guide** : https://www.promptingguide.ai/
- **OpenAI Prompt Examples** : https://platform.openai.com/examples
- **Awesome Prompts** : https://github.com/f/awesome-chatgpt-prompts
- **Learn Prompting** : https://learnprompting.org/

---

**🎓 Bravo !** Vous maîtrisez maintenant le prompt engineering, l'interface cruciale entre humains et LLMs. Prochain chapitre : **Chapitre 12 - RAG (Retrieval-Augmented Generation)** pour combiner prompting et recherche d'information ! 🚀

