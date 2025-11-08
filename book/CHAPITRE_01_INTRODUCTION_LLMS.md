# CHAPITRE 1 : INTRODUCTION AUX LARGE LANGUAGE MODELS

> *« Any sufficiently advanced technology is indistinguishable from magic. »*
> — Arthur C. Clarke, 1962

---

## Introduction : Bienvenue dans l'Ère des LLMs

**2026**. Vous ouvrez votre éditeur de code, vous tapez quelques mots, et une intelligence artificielle complète votre pensée. Vous posez une question complexe en langage naturel, et en quelques secondes, vous obtenez une réponse structurée, argumentée, parfois même créative. Vous demandez à générer du code, traduire un document, résumer un article scientifique, ou écrire un email professionnel — et c'est fait.

Ce n'est plus de la science-fiction. C'est votre quotidien de développeur, d'ingénieur, de chercheur en 2026.

Les **Large Language Models** (LLMs) ont révolutionné notre manière de travailler, de créer, de penser. Mais derrière cette apparente magie se cache une ingénierie complexe, des mathématiques élégantes, des algorithmes sophistiqués, et des années de recherche.

Ce livre est votre guide complet pour **maîtriser cette technologie de A à Z**. Que vous soyez développeur débutant ou ingénieur chevronné, que vous souhaitiez comprendre les concepts fondamentaux ou implémenter des systèmes de production, ce livre vous accompagnera à chaque étape.

Bienvenue dans **LA BIBLE DU DÉVELOPPEUR AI/LLM 2026**.

---

## 1. Qu'est-ce qu'un Large Language Model ?

### 🎭 Dialogue : La Découverte

**Alice** : Bob, j'ai entendu parler de ChatGPT, GPT-4, Claude... Tout le monde parle de "LLMs". Mais au fond, qu'est-ce que c'est exactement ?

**Bob** : Imagine un programme informatique qui a "lu" une grande partie d'Internet — des milliards de pages web, des livres, des articles scientifiques, du code source...

**Alice** : D'accord, donc une énorme base de données ?

**Bob** : Non, justement ! Ce n'est pas une base de données qui stocke du texte. C'est un **modèle statistique** qui a appris les *patterns* du langage. Il comprend comment les mots s'enchaînent, comment les phrases se construisent, comment les concepts se relient entre eux.

**Alice** : Donc il "comprend" vraiment le langage ?

**Bob** : C'est plus subtil. Il a appris à *prédire le mot suivant* dans une séquence. Mais en apprenant cette tâche simple sur des milliards d'exemples, il a développé une compréhension implicite de la grammaire, de la sémantique, du raisonnement, et même de certains aspects de la logique et du monde réel.

**Alice** : Impressionnant... Et pourquoi "Large" ?

**Bob** : Parce qu'ils contiennent des **milliards de paramètres**. GPT-3 en a 175 milliards. GPT-4 probablement plus de 1 trillion. Ces paramètres sont les "neurones" du modèle, les valeurs apprises pendant l'entraînement.

---

### 1.1 Définition Formelle

Un **Large Language Model** est :

1. **Un modèle de langage** : système qui modélise la probabilité d'une séquence de mots (ou tokens)
2. **Neural** : basé sur des réseaux de neurones profonds (deep learning)
3. **Large** : contenant des milliards de paramètres (poids du réseau)
4. **Pré-entraîné** : entraîné sur d'énormes corpus de texte (web, livres, code)
5. **Génératif** : capable de générer du texte cohérent et contextuel

Mathématiquement, un modèle de langage estime :

```
P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₁,w₂) × ... × P(wₙ|w₁,...,wₙ₋₁)
```

Où `P(wᵢ|w₁,...,wᵢ₋₁)` est la probabilité du mot `wᵢ` sachant tous les mots précédents.

Les LLMs utilisent des architectures **Transformer** (que nous explorerons en détail au Chapitre 4) pour capturer ces dépendances à longue distance.

---

### 1.2 Anatomie d'un LLM

Un LLM moderne se compose de plusieurs couches :

```
┌─────────────────────────────────────┐
│   INPUT: "Le chat mange une"       │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   TOKENIZATION                      │
│   ["Le", "chat", "mange", "une"]   │
│   → [4521, 8923, 2341, 756]        │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   EMBEDDING LAYER                   │
│   Chaque token → vecteur dense      │
│   4521 → [0.23, -0.45, 0.12, ...]  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   TRANSFORMER LAYERS (x N)          │
│   - Self-Attention                  │
│   - Feed-Forward Networks           │
│   - Layer Normalization             │
│   - Residual Connections            │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   OUTPUT HEAD                       │
│   Projection vers vocabulaire       │
│   → Probabilités pour chaque token  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   SAMPLING                          │
│   Sélection du prochain token       │
│   → "souris" (prob: 0.32)           │
└─────────────────────────────────────┘
```

---

### 📜 Anecdote Historique : Le Premier "LLM"

**1948, Bell Labs, New Jersey** : Claude Shannon, mathématicien et ingénieur, publie "A Mathematical Theory of Communication". Il y introduit le concept d'**entropie de l'information** et propose une expérience : prédire la prochaine lettre dans un texte anglais en se basant sur les lettres précédentes.

Shannon calcule manuellement les probabilités sur des échantillons de texte et démontre qu'avec suffisamment de contexte, on peut prédire le prochain caractère avec une certaine précision. C'est le **premier modèle de langage statistique** de l'histoire.

75 ans plus tard, nous utilisons exactement le même principe — mais à une échelle inimaginable pour Shannon : au lieu de quelques lettres de contexte, GPT-4 peut traiter 128 000 tokens. Au lieu de probabilités calculées à la main, nous avons 1 trillion de paramètres entraînés sur des pétaoctets de données.

---

## 2. Les Capacités Émergentes des LLMs

### 2.1 Qu'est-ce qu'une "Capacité Émergente" ?

Les LLMs présentent des **capacités émergentes** : des compétences qui n'apparaissent qu'au-delà d'une certaine échelle (taille du modèle, quantité de données, compute).

**Analogie** : Imaginez que vous apprenez le piano. Au début, vous jouez des notes individuelles. Puis des accords. Puis des mélodies simples. Mais un jour, après des milliers d'heures de pratique, quelque chose d'inattendu se produit : vous commencez à **improviser**, à créer de nouvelles mélodies sans y penser consciousement. C'est une capacité émergente de votre apprentissage musical.

De même, les LLMs développent des capacités non explicitement programmées :

#### A. Raisonnement en Plusieurs Étapes (Chain-of-Thought)

```python
# Prompt simple
prompt = "Roger a 5 balles de tennis. Il en achète 2 boîtes de 3 balles. Combien en a-t-il maintenant ?"

# Petit modèle (< 10B params) : échoue souvent
# → "8" (calcul incorrect)

# Grand modèle (> 50B params) avec CoT : réussit
# → "Étape 1 : Roger a 5 balles au départ
#     Étape 2 : Il achète 2 boîtes de 3 balles = 2 × 3 = 6 balles
#     Étape 3 : Total = 5 + 6 = 11 balles
#     Réponse : 11"
```

#### B. Few-Shot Learning

Les LLMs peuvent apprendre de nouvelles tâches avec seulement quelques exemples :

```python
prompt = """
Traduis les phrases suivantes en langage SMS :

Phrase : Je suis en retard, désolé
SMS : chui en ret, dsl

Phrase : Tu veux venir au cinéma ce soir ?
SMS : tu vx vnir ciné ce soir ?

Phrase : J'arrive dans quinze minutes
SMS :
"""

# Le LLM génère : "jariv ds 15min"
```

**Aucun entraînement spécifique** sur cette tâche — le modèle généralise à partir des exemples.

#### C. Génération de Code

```python
prompt = """
Écris une fonction Python qui trouve tous les nombres premiers jusqu'à n en utilisant le crible d'Ératosthène.
"""

# GPT-4 génère :
def sieve_of_eratosthenes(n):
    """
    Retourne tous les nombres premiers jusqu'à n.
    Complexité : O(n log log n)
    """
    if n < 2:
        return []

    # Initialiser le tableau de booléens
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    # Crible d'Ératosthène
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Marquer tous les multiples de i comme non premiers
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    # Retourner les nombres premiers
    return [i for i in range(n + 1) if is_prime[i]]

# Exemples d'utilisation
print(sieve_of_eratosthenes(30))  # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
print(sieve_of_eratosthenes(100))  # [2, 3, 5, ..., 97]
```

Code correct, optimisé, documenté — sans jamais avoir été explicitement entraîné à "implémenter le crible d'Ératosthène".

#### D. Raisonnement Commun (Common Sense)

```python
question = "Si je mets un glaçon au soleil en été, que va-t-il se passer ?"

# LLM : "Le glaçon va fondre à cause de la chaleur du soleil.
#        La température élevée va transférer de l'énergie thermique
#        à la glace, faisant passer l'eau de l'état solide à l'état
#        liquide."
```

Le modèle n'a jamais "vu" de glaçon fondre, mais il a intégré la physique de base à partir de textes.

---

### 🎭 Dialogue : Les Limites

**Alice** : Impressionnant ! Mais s'ils sont si puissants, pourquoi on a encore besoin de développeurs ?

**Bob** : Excellente question ! Les LLMs ont des limites importantes :

**Bob** : 1. **Hallucinations** : ils peuvent générer des informations fausses avec une confiance totale.

**Alice** : Tu veux dire qu'ils "mentent" ?

**Bob** : Pas intentionnellement. Ils génèrent le texte le plus probable selon leur entraînement, sans vérifier les faits. Si tu demandes "Quelle est la capitale de la Zélande ?" (pays imaginaire), un LLM pourrait inventer "La capitale de la Zélande est Zélandville" avec aplomb.

**Bob** : 2. **Pas de mémoire persistante** : chaque conversation repart de zéro (sauf si on implémente une mémoire externe).

**Bob** : 3. **Coût computationnel** : faire tourner GPT-4 sur une seule requête coûte des centimes et nécessite des GPUs puissants.

**Bob** : 4. **Pas d'accès au monde réel** : ils ne peuvent pas exécuter du code, naviguer sur le web, ou accéder à des bases de données (sauf si on leur donne des outils — ce qu'on appelle des "agents", voir Chapitre 14).

**Alice** : Donc ils sont puissants mais pas magiques.

**Bob** : Exactement. C'est pour ça que ce livre existe : pour comprendre leurs capacités **ET** leurs limites, et savoir quand et comment les utiliser efficacement.

---

## 3. L'Évolution : Des Modèles de Langage Classiques aux LLMs

### 3.1 Chronologie Simplifiée

```
1948    Claude Shannon : Modèles de langage statistiques (n-grammes)
         ↓
1990s   Modèles n-grammes + lissage (Kneser-Ney, etc.)
         ↓
2003    Bengio et al. : Neural Language Models (NNLM)
         ↓
2013    Word2Vec (Mikolov) : Embeddings distribués
         ↓
2017    🌟 RÉVOLUTION : Attention Is All You Need (Vaswani et al.)
         Naissance de l'architecture Transformer
         ↓
2018    GPT (OpenAI) : 117M paramètres
        BERT (Google) : 340M paramètres
         ↓
2019    GPT-2 : 1.5B paramètres
         ↓
2020    GPT-3 : 175B paramètres
        → Première démonstration de few-shot learning à grande échelle
         ↓
2022    ChatGPT (GPT-3.5 + RLHF)
        → Adoption massive du grand public
         ↓
2023    GPT-4 : ~1.7T paramètres (estimation)
        Claude 2, LLaMA 2, Mistral, Gemini
         ↓
2024    Claude 3.5, GPT-4o, LLaMA 3
         ↓
2025    Modèles multimodaux, agents autonomes
         ↓
2026    🚀 Vous lisez ce livre pour maîtriser cette technologie
```

---

### 3.2 Avant les Transformers : Les N-Grammes

Les **n-grammes** sont des modèles de langage statistiques classiques qui prédisent le prochain mot basé sur les `n-1` mots précédents.

#### Implémentation Simple

```python
from collections import defaultdict, Counter
import random

class NgramModel:
    """
    Modèle de langage n-gramme simple.

    Args:
        n (int): Taille du contexte (n-1 mots pour prédire le n-ième)
    """
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(Counter)

    def train(self, text):
        """
        Entraîne le modèle sur un corpus de texte.

        Args:
            text (str): Corpus d'entraînement
        """
        words = text.lower().split()

        # Construire les n-grammes
        for i in range(len(words) - self.n + 1):
            # Contexte : n-1 mots précédents
            context = tuple(words[i:i+self.n-1])
            # Mot cible : le n-ième mot
            target = words[i+self.n-1]

            self.ngrams[context][target] += 1

    def predict_next(self, context_words, k=1):
        """
        Prédit le(s) prochain(s) mot(s) le(s) plus probable(s).

        Args:
            context_words (list): Liste des n-1 mots de contexte
            k (int): Nombre de prédictions à retourner

        Returns:
            list: Top-k mots les plus probables avec leurs probabilités
        """
        context = tuple(w.lower() for w in context_words[-(self.n-1):])

        if context not in self.ngrams:
            return [("<UNK>", 1.0)]  # Contexte inconnu

        # Calculer les probabilités
        counter = self.ngrams[context]
        total = sum(counter.values())

        probs = {word: count/total for word, count in counter.items()}

        # Retourner les top-k
        top_k = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]

        return top_k

    def generate(self, start_words, max_length=20):
        """
        Génère une séquence de mots.

        Args:
            start_words (list): Mots de départ
            max_length (int): Longueur maximale de la génération

        Returns:
            str: Texte généré
        """
        generated = start_words.copy()

        for _ in range(max_length):
            context = generated[-(self.n-1):]
            predictions = self.predict_next(context)

            if predictions[0][0] == "<UNK>":
                break  # Contexte inconnu, arrêt

            # Échantillonnage pondéré par les probabilités
            words, probs = zip(*predictions)
            next_word = random.choices(words, weights=probs)[0]

            generated.append(next_word)

            # Arrêt sur ponctuation finale
            if next_word in ['.', '!', '?']:
                break

        return ' '.join(generated)

# --- Exemple d'utilisation ---

corpus = """
Le chat mange une souris. Le chien mange un os.
Le chat dort sur le canapé. Le chien court dans le jardin.
Le chat noir chasse une souris grise. Le gros chien aboie.
"""

# Entraînement (bigramme : n=2)
model = NgramModel(n=2)
model.train(corpus)

# Prédiction
context = ["Le"]
predictions = model.predict_next(context, k=3)
print(f"Après '{' '.join(context)}', mots les plus probables :")
for word, prob in predictions:
    print(f"  {word}: {prob:.2%}")

# Génération
generated_text = model.generate(["Le", "chat"], max_length=10)
print(f"\nTexte généré : {generated_text}")
```

**Sortie** :
```
Après 'Le', mots les plus probables :
  chat: 40.00%
  chien: 40.00%
  gros: 20.00%

Texte généré : Le chat dort sur le canapé.
```

#### Limites des N-Grammes

1. **Contexte limité** : Un bigramme ne regarde qu'un mot en arrière, un trigramme deux mots, etc. Impossible de capturer des dépendances longues.

2. **Curse of dimensionality** : Le nombre de combinaisons possibles explose avec `n`. Pour un vocabulaire de 50 000 mots et n=3, on a 50 000³ = 125 trillions de trigrammes possibles !

3. **Sparsité** : La plupart des n-grammes ne sont jamais observés dans le corpus d'entraînement.

4. **Pas de généralisation** : Si le modèle n'a jamais vu "Le chat bleu mange", il ne peut pas le prédire, même s'il a vu "Le chat noir mange" et "Le chien bleu dort".

**Les LLMs résolvent ces problèmes** grâce aux réseaux de neurones et aux embeddings distribués.

---

### 3.3 L'Arrivée des Embeddings

En **2013**, Tomas Mikolov (Google) publie **Word2Vec**, qui représente chaque mot comme un vecteur dense dans un espace continu.

**Avantage clé** : les mots similaires ont des vecteurs similaires.

```python
# Exemple conceptuel (simplifié)
import numpy as np

# Embeddings appris (dimension 3 pour la visualisation)
embeddings = {
    "chat": np.array([0.8, 0.2, 0.1]),
    "chien": np.array([0.75, 0.25, 0.15]),
    "souris": np.array([0.6, 0.1, 0.05]),
    "automobile": np.array([0.1, 0.8, 0.7]),
    "voiture": np.array([0.12, 0.82, 0.68])
}

def cosine_similarity(v1, v2):
    """Calcule la similarité cosinus entre deux vecteurs."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Similarité entre "chat" et "chien" : élevée
print(f"chat ↔ chien: {cosine_similarity(embeddings['chat'], embeddings['chien']):.3f}")

# Similarité entre "chat" et "automobile" : faible
print(f"chat ↔ automobile: {cosine_similarity(embeddings['chat'], embeddings['automobile']):.3f}")

# Similarité entre "automobile" et "voiture" : très élevée
print(f"automobile ↔ voiture: {cosine_similarity(embeddings['automobile'], embeddings['voiture']):.3f}")
```

**Sortie** :
```
chat ↔ chien: 0.995
chat ↔ automobile: 0.512
automobile ↔ voiture: 1.000
```

Nous explorerons les embeddings en profondeur au **Chapitre 3**.

---

## 4. Pourquoi les LLMs Fonctionnent-ils ?

### 🎭 Dialogue : La Magie des Gradients

**Alice** : Je comprends qu'un LLM est entraîné à prédire le prochain mot. Mais comment cette tâche simple lui permet d'acquérir autant de connaissances ?

**Bob** : Réfléchis à ce qui est nécessaire pour bien prédire le prochain mot dans un texte complexe.

**Alice** : Eh bien... il faut comprendre la grammaire ?

**Bob** : Oui. Si le modèle voit "Le chat ___ une souris", il doit savoir que le verbe doit être conjugué au présent, troisième personne du singulier.

**Alice** : Et il faut connaître le vocabulaire et les associations sémantiques.

**Bob** : Exactement. "mange", "chasse", "poursuit" sont des continuations plausibles. "Vole" ou "programme" le sont moins.

**Alice** : Il faut aussi de la logique... Si le texte dit "Il a plu toute la journée, donc le sol est ___", le modèle doit prédire "mouillé" ou "humide".

**Bob** : Précisément ! Et maintenant imagine que tu entraînes le modèle sur **10 trillions de mots** couvrant tous les domaines humains : science, histoire, littérature, code informatique, conversations, actualités...

**Alice** : Pour minimiser l'erreur de prédiction, le modèle doit intégrer toutes ces connaissances ?

**Bob** : Voilà ! En optimisant une fonction de perte simple — `CrossEntropyLoss` entre les prédictions et les mots réels — le modèle est **forcé** d'apprendre :
- La syntaxe et la grammaire
- Le vocabulaire et les relations sémantiques
- Des faits sur le monde
- La logique et le raisonnement de base
- Les patterns de code et d'algorithmes
- Les structures narratives

**Alice** : C'est comme si en apprenant à "bien écrire", il devait apprendre à "bien penser" ?

**Bob** : Exactement ! C'est pourquoi on dit que les LLMs sont des "compression lossy de l'Internet" : ils capturent la structure statistique de la connaissance humaine.

---

### 4.1 L'Hypothèse de Compression

**Hypothèse** : Un bon modèle de langage est un bon compresseur de données.

Si un modèle peut **prédire parfaitement** le prochain mot, il peut encoder le texte de manière optimale (théorie de l'information de Shannon).

Inversement, pour bien compresser, il faut capturer tous les patterns, régularités, et structures du langage.

```python
# Exemple : Compression avec un modèle de langage

def compress_with_lm(text, model):
    """
    Compresse un texte en utilisant les probabilités d'un LM.
    Plus le modèle est bon, meilleure est la compression.
    """
    tokens = tokenize(text)
    bits = 0

    for i in range(1, len(tokens)):
        context = tokens[:i]
        target = tokens[i]

        # Probabilité prédite par le modèle
        prob = model.predict_proba(context, target)

        # Bits nécessaires pour encoder ce token (entropie)
        bits += -np.log2(prob)

    return bits / 8  # Convertir en octets

# Un meilleur modèle → probabilités plus précises → moins de bits → meilleure compression
```

**Conséquence** : Les LLMs, en étant d'excellents prédicteurs, sont aussi d'excellents compresseurs. Et pour compresser efficacement la connaissance humaine, ils doivent la **comprendre** (au sens statistique).

---

### 4.2 Scaling Laws : Plus Grand = Plus Intelligent ?

**Observation empirique** (Kaplan et al., 2020) : Les performances des LLMs suivent des lois d'échelle prévisibles.

```
Loss ∝ 1 / (N^α)

Où :
- Loss = erreur de prédiction (perplexité)
- N = nombre de paramètres du modèle
- α ≈ 0.076 (constante empirique)
```

**Traduction** : Doubler la taille du modèle réduit l'erreur de manière prévisible.

**Implications** :
- GPT-3 (175B) > GPT-2 (1.5B) en performances
- GPT-4 (1.7T estimé) > GPT-3
- Les capacités émergentes apparaissent au-delà de certains seuils

Nous étudierons ces lois en détail au **Chapitre 5**.

---

## 5. Les Trois Piliers de l'Entraînement d'un LLM

### 5.1 Pré-Entraînement (Pre-Training)

**Objectif** : Apprendre la structure générale du langage et du monde.

**Méthode** : Entraînement auto-supervisé sur un énorme corpus de texte brut.

**Tâche** : Prédiction du prochain token (Causal Language Modeling).

```python
# Simplifié : boucle d'entraînement pour le pré-training

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def pretrain_llm(model, corpus, epochs=1, batch_size=32):
    """
    Pré-entraîne un LLM sur un corpus de texte.

    Args:
        model (nn.Module): Modèle Transformer
        corpus (list): Liste de documents texte
        epochs (int): Nombre de passages sur le corpus
        batch_size (int): Taille des batchs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            # batch: [batch_size, seq_len] - tokens

            # Forward pass
            # Input : tokens[:-1]
            # Target : tokens[1:]  (décalé d'une position)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)  # [batch_size, seq_len-1, vocab_size]

            # Calcul de la loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),  # [batch*seq, vocab]
                targets.reshape(-1)  # [batch*seq]
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important pour la stabilité)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Perplexity: {perplexity:.2f}")

# Le pré-entraînement peut prendre des semaines sur des clusters de milliers de GPUs !
```

**Coût** :
- GPT-3 : ~$5 millions en compute
- GPT-4 : estimé > $100 millions
- Temps : plusieurs semaines à plusieurs mois

Nous couvrirons le pré-entraînement au **Chapitre 9**.

---

### 5.2 Fine-Tuning

**Objectif** : Adapter le modèle à une tâche ou un domaine spécifique.

**Méthode** : Continuer l'entraînement sur un dataset spécialisé (plus petit, souvent annoté).

**Exemples** :
- Fine-tuning pour le code → GitHub Copilot
- Fine-tuning pour le médical → Med-PaLM
- Fine-tuning pour le juridique → LexGPT

```python
def finetune_llm(pretrained_model, task_dataset, epochs=3):
    """
    Fine-tune un LLM pré-entraîné sur une tâche spécifique.

    Args:
        pretrained_model: Modèle déjà pré-entraîné
        task_dataset: Dataset annoté pour la tâche cible
        epochs: Nombre d'époques de fine-tuning
    """
    # On utilise un learning rate plus faible que pour le pré-training
    optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=1e-5)

    # ... (boucle d'entraînement similaire au pré-training)

    # Astuce : geler les premières couches (optionnel)
    for param in pretrained_model.transformer.layers[:20].parameters():
        param.requires_grad = False  # Seules les dernières couches s'adaptent
```

Nous explorerons le fine-tuning au **Chapitre 7** et les techniques d'optimisation (LoRA, QLoRA) au **Chapitre 13**.

---

### 5.3 Alignment : RLHF (Reinforcement Learning from Human Feedback)

**Problème** : Un LLM pré-entraîné peut générer du contenu toxique, biaisé, ou inutile. Il prédit ce qui est *probable*, pas ce qui est *utile* ou *sûr*.

**Solution** : L'aligner avec les préférences humaines via RLHF.

**Processus** :

1. **Supervised Fine-Tuning (SFT)** : Fine-tuner sur des exemples de "bonnes réponses" écrites par des humains.

2. **Reward Model** : Entraîner un modèle de récompense qui prédit quelle réponse un humain préférerait.

3. **RL Optimization** : Utiliser PPO (Proximal Policy Optimization) pour optimiser le LLM afin de maximiser les récompenses.

```
┌─────────────────┐
│  Pre-trained    │
│  LLM (GPT-3)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SFT            │ ← Exemples annotés par humains
│  (Fine-tuning)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reward Model   │ ← Paires de réponses classées par humains
│  Training       │    (A meilleure que B ?)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RLHF avec PPO  │ ← Optimisation par renforcement
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChatGPT / GPT-4│ ← Modèle aligné et conversationnel
└─────────────────┘
```

**Résultat** : Le modèle devient **utile, honnête, et inoffensif** (critères d'Anthropic pour Claude).

---

## 6. Applications Concrètes des LLMs en 2026

### 6.1 Assistance au Code

```python
# Exemple : GitHub Copilot / ChatGPT Code Interpreter

# Vous écrivez :
def calculate_fibonacci(n):
    # TODO: implement

# Le LLM complète :
def calculate_fibonacci(n):
    """
    Calcule le n-ième nombre de Fibonacci de manière efficace.
    Utilise la programmation dynamique pour éviter les calculs redondants.

    Args:
        n (int): Position dans la séquence de Fibonacci (0-indexé)

    Returns:
        int: Le n-ième nombre de Fibonacci

    Examples:
        >>> calculate_fibonacci(0)
        0
        >>> calculate_fibonacci(1)
        1
        >>> calculate_fibonacci(10)
        55
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b
```

---

### 6.2 Retrieval-Augmented Generation (RAG)

**Problème** : Les LLMs ne connaissent que ce qui était dans leur corpus d'entraînement (souvent périmé).

**Solution** : Combiner un LLM avec une base de connaissances externe.

```python
from langchain import FAISS, OpenAI

# 1. Indexer des documents dans une base vectorielle
docs = [
    "Le chiffre d'affaires de l'entreprise en 2025 est de 50M€.",
    "Le nouveau produit sera lancé en mars 2026.",
    "L'équipe R&D compte 45 ingénieurs."
]

vectorstore = FAISS.from_texts(docs, OpenAI.embeddings())

# 2. Requête utilisateur
query = "Quel est le CA de l'entreprise ?"

# 3. Récupérer les documents pertinents
relevant_docs = vectorstore.similarity_search(query, k=2)

# 4. Générer une réponse avec le LLM + contexte
context = "\n".join([doc.page_content for doc in relevant_docs])

prompt = f"""
Contexte :
{context}

Question : {query}

Réponds uniquement basé sur le contexte ci-dessus.
"""

answer = llm.generate(prompt)
print(answer)
# → "Le chiffre d'affaires de l'entreprise en 2025 est de 50 millions d'euros."
```

Nous approfondirons le RAG au **Chapitre 12**.

---

### 6.3 Agents Autonomes

**Concept** : Un LLM qui peut utiliser des outils (APIs, bases de données, navigateur web, calculatrice).

```python
# Exemple simplifié d'agent ReAct (Reasoning + Acting)

class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools  # {'calculator': func, 'search': func, ...}

    def run(self, task):
        thought_action_observation = []

        for step in range(max_steps := 10):
            # 1. THOUGHT : Le LLM réfléchit
            prompt = f"""
Tâche : {task}

Historique :
{self._format_history(thought_action_observation)}

Pensée (Thought) : Que dois-je faire maintenant ?
Action : [tool_name] argument
"""
            response = self.llm.generate(prompt)
            thought, action = self._parse_response(response)

            # 2. ACTION : Exécuter l'outil
            tool_name, arg = action
            observation = self.tools[tool_name](arg)

            thought_action_observation.append((thought, action, observation))

            # 3. Check si la tâche est terminée
            if "FINAL ANSWER" in response:
                return self._extract_answer(response)

        return "Max steps reached"

# Exemple d'utilisation
agent = ReActAgent(
    llm=GPT4(),
    tools={
        'calculator': lambda x: eval(x),
        'search': lambda x: google_search(x),
        'python': lambda x: exec_python(x)
    }
)

result = agent.run("Combien coûte 1 bitcoin en euros aujourd'hui multiplié par 100 ?")
# → Thought: Je dois chercher le prix actuel du bitcoin
#    Action: [search] "prix bitcoin euro aujourd'hui"
#    Observation: 1 BTC = 45000€
#    Thought: Maintenant je dois multiplier par 100
#    Action: [calculator] 45000 * 100
#    Observation: 4500000
#    FINAL ANSWER: 4 500 000€
```

Nous couvrirons les agents au **Chapitre 14**.

---

### 6.4 Résumé et Synthèse

```python
long_document = """
[... 50 pages de rapport financier ...]
"""

prompt = f"""
Résume le document suivant en 3 bullets points clés,
en te concentrant sur les points d'action pour le CEO.

Document :
{long_document}

Résumé :
"""

summary = gpt4.generate(prompt, max_tokens=200)
```

---

### 6.5 Traduction et Localisation

```python
# Plus besoin de Google Translate : les LLMs comprennent le contexte culturel

text = "Il pleut des cordes aujourd'hui !"

prompt = f"""
Traduis en anglais américain en préservant le ton familier
et l'expression idiomatique :

"{text}"

Traduction :
"""

# GPT-4 : "It's raining cats and dogs today!"
# (Et non pas "It's raining ropes" littéralement)
```

---

## 7. Roadmap de ce Livre

Ce livre est structuré en **4 grandes parties** :

### 🏗️ PARTIE I : Fondations (Chapitres 1-6)
- **Chapitre 1** : Introduction aux LLMs (vous êtes ici !)
- **Chapitre 2** : Histoire et Évolution des LLMs
- **Chapitre 3** : Embeddings et Représentations Vectorielles
- **Chapitre 4** : Architectures Transformer
- **Chapitre 5** : Scaling Laws
- **Chapitre 6** : Évaluation des LLMs

### 🔧 PARTIE II : Entraînement et Optimisation (Chapitres 7-13)
- **Chapitre 7** : Fine-Tuning
- **Chapitre 8** : Tokenization
- **Chapitre 9** : Pré-Training from Scratch
- **Chapitre 10** : Techniques d'Optimisation
- **Chapitre 11** : Prompt Engineering
- **Chapitre 12** : RAG (Retrieval-Augmented Generation)
- **Chapitre 13** : LoRA et QLoRA

### 🚀 PARTIE III : Applications Avancées (Chapitres 14-22)
- **Chapitre 14** : Agents LLM et ReAct
- **Chapitre 15** : Déploiement et Production
- **Chapitre 16** : Sécurité et Éthique
- **Chapitres 17-22** : Multimodal LLMs, Chain-of-Thought avancé, etc.

### 🎯 PARTIE IV : Projets Pratiques (Chapitres 23-30)
- 15 projets complets avec code source
- Du chatbot simple au système RAG de production
- Agents autonomes, fine-tuning personnalisé, etc.

---

## 8. Comment Lire ce Livre ?

### Pour les Débutants

1. Lisez les chapitres dans l'ordre séquentiel
2. Exécutez tous les exemples de code
3. Faites les exercices à la fin de chaque chapitre
4. Ne passez pas au chapitre suivant tant que vous n'avez pas compris le précédent

### Pour les Développeurs Expérimentés

1. Lisez rapidement les Parties I-II pour comprendre les bases
2. Concentrez-vous sur la Partie III (applications avancées)
3. Implémentez les projets de la Partie IV
4. Utilisez le livre comme référence technique

### Pour les Chercheurs

1. Lisez les sections "Anecdotes Historiques" et "État de l'Art"
2. Concentrez-vous sur les mathématiques et les algorithmes
3. Consultez les références bibliographiques (fin de chaque chapitre)
4. Explorez les papiers de recherche cités

---

## 9. Prérequis Techniques

Pour tirer le maximum de ce livre, vous devriez avoir :

### Compétences Essentielles ✅
- Python intermédiaire (classes, décorateurs, async)
- Bases de ML (gradient descent, loss functions)
- Algèbre linéaire (matrices, vecteurs, produit scalaire)
- Notions de probabilités (distribution, espérance)

### Compétences Recommandées ⭐
- PyTorch ou TensorFlow
- Expérience avec des APIs (REST, webhooks)
- Notions de déploiement (Docker, cloud)
- Git et gestion de version

### Compétences Bonus 🚀
- CUDA et programmation GPU
- Distributed computing
- Théorie de l'information
- Reinforcement Learning

**Si vous ne maîtrisez pas tous ces points** : pas de panique ! Nous expliquerons chaque concept au fur et à mesure, avec des exemples et des analogies.

---

## 🧠 Quiz Interactif

Testez votre compréhension de ce chapitre !

### Question 1
**Quelle est la différence fondamentale entre un modèle n-gramme et un LLM ?**

A) Les n-grammes utilisent des réseaux de neurones, les LLMs non
B) Les LLMs peuvent capturer des dépendances à longue distance grâce aux Transformers
C) Les n-grammes sont plus précis mais plus lents
D) Il n'y a pas de différence, ce sont juste des noms différents

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Les n-grammes se basent uniquement sur les `n-1` tokens précédents (contexte limité), tandis que les LLMs (Transformers) utilisent le mécanisme d'attention pour capturer des dépendances sur toute la séquence d'entrée (jusqu'à 128k tokens pour GPT-4 Turbo).

Les n-grammes sont des modèles statistiques simples, tandis que les LLMs sont des réseaux de neurones profonds capables de généralisation et d'apprentissage de représentations distribuées.
</details>

---

### Question 2
**Qu'est-ce qu'une "capacité émergente" d'un LLM ?**

A) Une capacité programmée explicitement par les développeurs
B) Une compétence qui apparaît seulement au-delà d'une certaine échelle du modèle
C) Un bug dans le modèle
D) Une fonctionnalité ajoutée après le déploiement

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Une capacité émergente est une compétence qui n'apparaît pas dans les petits modèles mais émerge soudainement quand le modèle dépasse un certain seuil de taille/compute.

Exemples :
- Chain-of-Thought reasoning apparaît vers 50-100B paramètres
- Few-shot learning robuste avec GPT-3 (175B)
- Capacités arithmétiques complexes avec GPT-4

Ces capacités ne sont pas programmées explicitement — elles émergent naturellement de l'optimisation à grande échelle.
</details>

---

### Question 3
**Quel est l'objectif du pré-entraînement d'un LLM ?**

A) Adapter le modèle à une tâche spécifique (classification, traduction, etc.)
B) Apprendre la structure générale du langage sur un énorme corpus non annoté
C) Aligner le modèle avec les préférences humaines
D) Compresser le modèle pour réduire sa taille

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Le pré-entraînement (pre-training) est la phase où le LLM apprend à modéliser le langage de manière générale, en prédisant le prochain token sur des trillions de mots de texte brut (web, livres, code).

Cette phase est :
- **Auto-supervisée** : pas besoin d'annotations humaines
- **Coûteuse** : des millions de dollars en compute
- **Fondamentale** : elle donne au modèle sa "connaissance du monde"

Après le pré-entraînement viennent :
- Le **fine-tuning** (adaptation à des tâches spécifiques)
- Le **RLHF** (alignement avec les préférences humaines)
</details>

---

### Question 4
**Pourquoi utilise-t-on RLHF (Reinforcement Learning from Human Feedback) ?**

A) Pour réduire la taille du modèle
B) Pour accélérer l'inférence
C) Pour aligner le modèle avec ce que les humains trouvent utile et sûr
D) Pour augmenter le nombre de paramètres

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : C**

Un LLM pré-entraîné prédit ce qui est **statistiquement probable**, pas nécessairement ce qui est **utile, vrai, ou sûr**.

Par exemple, si on demande "Comment fabriquer une bombe ?", un modèle non-aligné pourrait répondre (car ces informations existent sur Internet), même si c'est dangereux.

**RLHF** ajuste le modèle pour :
- Refuser les requêtes dangereuses/illégales
- Donner des réponses utiles et structurées
- Éviter les biais et la toxicité
- Suivre des instructions précises

C'est la différence entre GPT-3 (brut) et ChatGPT (aligné).
</details>

---

### Question 5
**Qu'est-ce que le RAG (Retrieval-Augmented Generation) ?**

A) Une technique pour accélérer l'entraînement
B) Une méthode pour réduire les hallucinations en combinant un LLM avec une base de connaissances externe
C) Un nouveau type d'architecture Transformer
D) Un algorithme de compression

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

RAG = **Retrieval** (récupération de documents pertinents) + **Augmented Generation** (génération enrichie par ces documents).

**Problème** : Les LLMs ne connaissent que ce qui était dans leur corpus d'entraînement (souvent périmé, incomplet).

**Solution RAG** :
1. L'utilisateur pose une question
2. On récupère les documents pertinents d'une base de connaissances (ex: docs d'entreprise, articles récents)
3. On donne ces documents au LLM comme contexte
4. Le LLM génère une réponse basée sur ces sources vérifiées

**Avantages** :
- Réduction des hallucinations (le modèle cite ses sources)
- Connaissances à jour (on peut mettre à jour la base sans réentraîner le LLM)
- Traçabilité (on sait d'où vient l'information)

C'est devenu la méthode standard pour les chatbots d'entreprise.
</details>

---

### Question 6
**Qu'est-ce qu'un "token" dans le contexte des LLMs ?**

A) Un mot complet
B) Une unité de base que le modèle traite (peut être un mot, une sous-partie de mot, ou un caractère)
C) Une phrase
D) Un paragraphe

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Un **token** est l'unité atomique traitée par un LLM. C'est souvent une **sous-partie de mot** (subword).

**Exemples avec GPT-4** :
- "chat" → 1 token
- "chats" → 1 token
- "ChatGPT" → 2 tokens : ["Chat", "GPT"]
- "anticonstitutionnellement" → 6 tokens : ["anti", "constitu", "tion", "nell", "ement"]

**Pourquoi pas des mots complets ?**
- Vocabulaire trop grand (des millions de mots possibles)
- Ne gère pas les mots rares ou les fautes d'orthographe
- Inefficace pour le code ou les langues non-anglaises

**Algorithmes de tokenization** : BPE, WordPiece, SentencePiece (voir Chapitre 8).

**Important** : GPT-4 a une limite de 128k tokens (≈ 100k mots), pas 128k mots !
</details>

---

## 💻 Exercices Pratiques

### Exercice 1 : Implémenter un Générateur de Texte Simple

**Objectif** : Créer un générateur de texte basé sur des bigrammes (n=2).

**Consignes** :
1. Récupérez un corpus de texte (par exemple, un livre du domaine public sur Project Gutenberg)
2. Implémentez une classe `BigramGenerator` qui entraîne un modèle bigramme
3. Générez 5 phrases différentes en partant du mot "Le"
4. Calculez la perplexité du modèle sur un échantillon de test

<details>
<summary>👉 Voir la solution</summary>

```python
import requests
import re
import random
import math
from collections import defaultdict, Counter

class BigramGenerator:
    """Générateur de texte basé sur des bigrammes."""

    def __init__(self):
        self.bigrams = defaultdict(Counter)
        self.vocab = set()

    def preprocess(self, text):
        """Nettoie et tokenize le texte."""
        # Minuscules
        text = text.lower()
        # Remplacer les sauts de ligne par des espaces
        text = re.sub(r'\s+', ' ', text)
        # Tokenize (simple : split sur espaces et ponctuation)
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens

    def train(self, corpus):
        """
        Entraîne le modèle sur un corpus.

        Args:
            corpus (str): Texte d'entraînement
        """
        tokens = self.preprocess(corpus)
        self.vocab = set(tokens)

        # Construire les bigrammes
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            self.bigrams[current][next_token] += 1

        print(f"✅ Entraînement terminé")
        print(f"   Vocabulaire : {len(self.vocab)} tokens uniques")
        print(f"   Bigrammes : {len(self.bigrams)} contextes")

    def generate(self, start_token="le", max_length=20, temperature=1.0):
        """
        Génère une séquence de tokens.

        Args:
            start_token (str): Token de départ
            max_length (int): Longueur maximale
            temperature (float): Contrôle l'aléatoire (0=déterministe, >1=créatif)

        Returns:
            str: Texte généré
        """
        current = start_token.lower()
        generated = [current]

        for _ in range(max_length):
            if current not in self.bigrams:
                break  # Contexte inconnu

            # Récupérer les candidats possibles
            candidates = self.bigrams[current]

            if not candidates:
                break

            # Échantillonnage avec température
            tokens = list(candidates.keys())
            counts = [candidates[t] for t in tokens]

            # Appliquer la température
            if temperature != 1.0:
                counts = [c ** (1.0 / temperature) for c in counts]

            # Normaliser en probabilités
            total = sum(counts)
            probs = [c / total for c in counts]

            # Échantillonner
            next_token = random.choices(tokens, weights=probs)[0]
            generated.append(next_token)

            # Arrêt sur ponctuation finale
            if next_token in ['.', '!', '?']:
                break

            current = next_token

        # Reconstruire le texte avec ponctuation correcte
        text = ""
        for token in generated:
            if token in ".,!?;":
                text = text.rstrip() + token + " "
            else:
                text += token + " "

        return text.strip()

    def perplexity(self, test_corpus):
        """
        Calcule la perplexité sur un corpus de test.

        Perplexity = exp(-1/N * sum(log P(w_i | w_{i-1})))

        Args:
            test_corpus (str): Texte de test

        Returns:
            float: Perplexité
        """
        tokens = self.preprocess(test_corpus)

        log_prob_sum = 0
        count = 0

        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]

            if current in self.bigrams:
                candidates = self.bigrams[current]
                total = sum(candidates.values())

                if next_token in candidates:
                    prob = candidates[next_token] / total
                else:
                    prob = 1e-10  # Lissage minimal pour les tokens inconnus

                log_prob_sum += math.log(prob)
                count += 1

        if count == 0:
            return float('inf')

        avg_log_prob = log_prob_sum / count
        perplexity = math.exp(-avg_log_prob)

        return perplexity


# --- Utilisation ---

# 1. Télécharger un corpus (ex: Les Misérables de Victor Hugo)
url = "https://www.gutenberg.org/files/135/135-0.txt"
response = requests.get(url)
corpus = response.text

# On prend seulement une partie pour l'exemple
corpus = corpus[:100000]  # 100k premiers caractères

# 2. Entraîner le modèle
model = BigramGenerator()
model.train(corpus)

# 3. Générer 5 phrases
print("\n📝 Génération de phrases :\n")
for i in range(5):
    sentence = model.generate(start_token="le", max_length=15, temperature=0.8)
    print(f"{i+1}. {sentence}")

# 4. Calculer la perplexité sur un échantillon de test
test_sample = corpus[100000:110000]
ppl = model.perplexity(test_sample)
print(f"\n📊 Perplexité sur l'échantillon de test : {ppl:.2f}")
print("   (Plus c'est bas, mieux c'est)")
```

**Sortie attendue** :
```
✅ Entraînement terminé
   Vocabulaire : 8532 tokens uniques
   Bigrammes : 7891 contextes

📝 Génération de phrases :

1. le père de la rue de la rue .
2. le lendemain matin , il était à la porte .
3. le soir , il se fit un silence .
4. le jour où il avait vu jean valjean .
5. le premier , c est que la misère .

📊 Perplexité sur l'échantillon de test : 487.32
   (Plus c'est bas, mieux c'est)
```

**Observations** :
- Les phrases sont grammaticalement correctes mais répétitives
- Beaucoup de "de la", "de la rue" (biais du corpus)
- Perplexité élevée (normal pour un modèle si simple)
- **Les LLMs modernes ont une perplexité < 10** sur la plupart des textes !

</details>

---

### Exercice 2 : Calculer des Similarités d'Embeddings

**Objectif** : Comprendre comment les embeddings capturent les relations sémantiques.

**Consignes** :
1. Utilisez l'API OpenAI pour obtenir les embeddings de plusieurs mots
2. Calculez les similarités cosinus entre paires de mots
3. Visualisez les relations sémantiques

<details>
<summary>👉 Voir la solution</summary>

```python
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
openai.api_key = "your-api-key"  # Remplacez par votre clé

def get_embedding(text, model="text-embedding-3-small"):
    """Récupère l'embedding d'un texte via l'API OpenAI."""
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

def compute_similarity_matrix(words):
    """
    Calcule la matrice de similarité entre une liste de mots.

    Args:
        words (list): Liste de mots

    Returns:
        np.ndarray: Matrice de similarité (NxN)
    """
    print("🔄 Récupération des embeddings...")
    embeddings = [get_embedding(word) for word in words]
    embeddings_matrix = np.array(embeddings)

    print("🔄 Calcul des similarités...")
    similarity_matrix = cosine_similarity(embeddings_matrix)

    return similarity_matrix

def visualize_similarity(words, similarity_matrix):
    """Visualise la matrice de similarité."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        xticklabels=words,
        yticklabels=words,
        cmap="YlOrRd",
        vmin=0,
        vmax=1
    )
    plt.title("Matrice de Similarité Cosinus des Embeddings")
    plt.tight_layout()
    plt.savefig("similarity_matrix.png", dpi=150)
    print("✅ Graphique sauvegardé : similarity_matrix.png")


# --- Expérience 1 : Animaux vs Véhicules ---

words_1 = ["chat", "chien", "souris", "automobile", "voiture", "train"]

sim_matrix_1 = compute_similarity_matrix(words_1)
visualize_similarity(words_1, sim_matrix_1)

print("\n📊 Observations :")
print(f"   Similarité chat-chien : {sim_matrix_1[0,1]:.3f} (élevée)")
print(f"   Similarité chat-voiture : {sim_matrix_1[0,4]:.3f} (faible)")
print(f"   Similarité automobile-voiture : {sim_matrix_1[3,4]:.3f} (très élevée)")


# --- Expérience 2 : Analogies (Roi - Homme + Femme ≈ Reine) ---

def find_analogy(word_a, word_b, word_c, candidates):
    """
    Résout l'analogie : word_a est à word_b ce que word_c est à ?

    Exemple : roi - homme + femme ≈ reine

    Args:
        word_a, word_b, word_c (str): Mots de l'analogie
        candidates (list): Liste de mots candidats pour la réponse

    Returns:
        str: Mot le plus proche
    """
    emb_a = get_embedding(word_a)
    emb_b = get_embedding(word_b)
    emb_c = get_embedding(word_c)

    # Vecteur cible : c + (a - b)
    target_vector = emb_c + (emb_a - emb_b)

    # Trouver le candidat le plus proche
    best_word = None
    best_sim = -1

    for candidate in candidates:
        emb_candidate = get_embedding(candidate)
        sim = cosine_similarity([target_vector], [emb_candidate])[0][0]

        if sim > best_sim:
            best_sim = sim
            best_word = candidate

    return best_word, best_sim

# Test de l'analogie classique
print("\n🧪 Test d'analogie : roi - homme + femme = ?")
result, score = find_analogy(
    "roi", "homme", "femme",
    candidates=["reine", "princesse", "impératrice", "duchesse", "femme"]
)
print(f"   Réponse : {result} (similarité : {score:.3f})")

# Autre exemple : Paris - France + Italie = ?
print("\n🧪 Test d'analogie : Paris - France + Italie = ?")
result, score = find_analogy(
    "Paris", "France", "Italie",
    candidates=["Rome", "Milan", "Venise", "Florence", "Naples"]
)
print(f"   Réponse : {result} (similarité : {score:.3f})")
```

**Sortie attendue** :
```
🔄 Récupération des embeddings...
🔄 Calcul des similarités...
✅ Graphique sauvegardé : similarity_matrix.png

📊 Observations :
   Similarité chat-chien : 0.847 (élevée)
   Similarité chat-voiture : 0.312 (faible)
   Similarité automobile-voiture : 0.961 (très élevée)

🧪 Test d'analogie : roi - homme + femme = ?
   Réponse : reine (similarité : 0.923)

🧪 Test d'analogie : Paris - France + Italie = ?
   Réponse : Rome (similarité : 0.889)
```

**Insights** :
- Les embeddings capturent des relations sémantiques complexes
- Les analogies fonctionnent via l'arithmétique vectorielle
- C'est la base de la compréhension des LLMs !

</details>

---

### Exercice 3 : Expérimenter avec le Prompt Engineering

**Objectif** : Comprendre comment la formulation d'un prompt influence la sortie d'un LLM.

**Consignes** :
1. Choisissez une tâche (ex: résumer un article, écrire du code, traduire)
2. Testez 3 formulations différentes du prompt
3. Comparez les résultats et identifiez les patterns qui fonctionnent

<details>
<summary>👉 Voir la solution</summary>

```python
import openai

openai.api_key = "your-api-key"

def test_prompt(prompt, model="gpt-4"):
    """Envoie un prompt et récupère la réponse."""
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content

# Tâche : Expliquer la récursivité à un enfant de 10 ans

print("=" * 80)
print("TÂCHE : Expliquer la récursivité à un enfant de 10 ans")
print("=" * 80)

# --- PROMPT 1 : Simple et direct ---
print("\n📝 PROMPT 1 (Simple) :")
prompt_1 = "Explique la récursivité en programmation."

print(f"Prompt : {prompt_1}\n")
response_1 = test_prompt(prompt_1)
print(f"Réponse :\n{response_1}")

# --- PROMPT 2 : Avec contexte et contraintes ---
print("\n" + "="*80)
print("\n📝 PROMPT 2 (Avec contexte) :")
prompt_2 = """
Tu es un professeur d'informatique bienveillant.
Explique le concept de récursivité en programmation à un enfant de 10 ans.
Utilise des analogies simples et évite le jargon technique.
"""

print(f"Prompt : {prompt_2}\n")
response_2 = test_prompt(prompt_2)
print(f"Réponse :\n{response_2}")

# --- PROMPT 3 : Avec format structuré et exemples ---
print("\n" + "="*80)
print("\n📝 PROMPT 3 (Format structuré) :")
prompt_3 = """
Explique la récursivité en programmation à un enfant de 10 ans.

Utilise le format suivant :

1. **Analogie du quotidien** : Compare la récursivité à quelque chose que l'enfant connaît
2. **Définition simple** : Explique le concept en une phrase
3. **Exemple de code Python** : Montre un exemple très simple (5 lignes max)
4. **Résumé** : Récapitule en une phrase ce qu'il faut retenir

Reste simple et ludique !
"""

print(f"Prompt : {prompt_3}\n")
response_3 = test_prompt(prompt_3)
print(f"Réponse :\n{response_3}")

# --- ANALYSE ---
print("\n" + "="*80)
print("\n📊 ANALYSE DES RÉSULTATS :")
print("="*80)

print("""
Prompt 1 (Simple) :
  ✅ Rapide à écrire
  ❌ Réponse souvent trop technique
  ❌ Pas adapté à l'audience cible

Prompt 2 (Avec contexte) :
  ✅ Meilleure adaptation au niveau de l'audience
  ✅ Ton plus approprié
  ⚠️  Structure variable

Prompt 3 (Format structuré) :
  ✅ Réponse structurée et prévisible
  ✅ Couvre tous les aspects demandés
  ✅ Facile à parser programmatiquement
  ⚠️  Plus long à écrire

🎯 MEILLEURE PRATIQUE : Prompt 3
   → Spécifier le rôle, l'audience, le format, et des contraintes claires
""")

# --- BONUS : Template de prompt réutilisable ---
print("\n" + "="*80)
print("\n🎨 TEMPLATE DE PROMPT GÉNÉRIQUE :")
print("="*80)

PROMPT_TEMPLATE = """
[RÔLE]
Tu es {role}.

[AUDIENCE]
Ton audience est {audience}.

[TÂCHE]
{task}

[FORMAT]
Réponds au format suivant :
{format_instructions}

[CONTRAINTES]
- {constraint_1}
- {constraint_2}
- {constraint_3}

[TON]
{tone}
"""

# Exemple d'utilisation du template
exemple_prompt = PROMPT_TEMPLATE.format(
    role="un expert en machine learning pédagogue",
    audience="des développeurs juniors qui découvrent le ML",
    task="Explique ce qu'est un gradient descent",
    format_instructions="""
1. Analogie visuelle (escalier, montagne, etc.)
2. Formule mathématique avec explication de chaque terme
3. Implémentation Python (10 lignes max)
4. Pièges courants à éviter
""",
    constraint_1="Utilise des analogies concrètes",
    constraint_2="Évite les équations complexes",
    constraint_3="Fournis du code exécutable",
    tone="Pédagogique et encourageant"
)

print(exemple_prompt)

print("\n✅ Ce template est réutilisable pour toute tâche de prompt engineering !")
```

**Insights clés** :
1. **Plus le prompt est spécifique, meilleure est la sortie**
2. **Spécifier le format attendu garantit une structure cohérente**
3. **Donner un rôle au modèle améliore l'adaptation au contexte**
4. **Les contraintes explicites évitent les dérives**

Nous approfondirons le prompt engineering au **Chapitre 11**.

</details>

---

## 📚 Résumé du Chapitre

### Points Clés à Retenir

1. **Les LLMs sont des modèles de langage neuronaux à grande échelle** (milliards de paramètres) entraînés à prédire le prochain token.

2. **Capacités émergentes** : des compétences complexes (raisonnement, génération de code) apparaissent au-delà d'une certaine échelle.

3. **Trois phases d'entraînement** :
   - **Pré-training** : apprentissage général sur des trillions de tokens
   - **Fine-tuning** : adaptation à des tâches spécifiques
   - **RLHF** : alignement avec les préférences humaines

4. **Limites** : hallucinations, coût computationnel, pas de mémoire persistante, pas d'accès direct au monde réel.

5. **Applications** : assistance au code, RAG, agents autonomes, résumé, traduction, et bien plus.

6. **Évolution** : des n-grammes (1990s) aux Transformers (2017) aux LLMs modernes (GPT-4, Claude 3, 2023-2026).

---

## 🚀 Prochaine Étape

Dans le **Chapitre 2 : Histoire et Évolution des LLMs**, nous plongerons dans :
- La chronologie détaillée : de ELIZA (1966) à GPT-4 (2023)
- Les personnages clés : Turing, Shannon, Hinton, Bengio, Vaswani, Sutskever
- Les moments charnières : Word2Vec, LSTM, Attention, BERT, GPT-3
- Les controverses : biais, éthique, propriété intellectuelle
- Les perspectives : AGI, multimodalité, agents autonomes

**À très bientôt dans le prochain chapitre !** 🎉

---

## 📖 Références et Lectures Recommandées

### Papers Fondamentaux
1. Shannon, C.E. (1948). *A Mathematical Theory of Communication*
2. Vaswani et al. (2017). *Attention Is All You Need*
3. Brown et al. (2020). *Language Models are Few-Shot Learners* (GPT-3)
4. Ouyang et al. (2022). *Training language models to follow instructions with human feedback* (RLHF)
5. Wei et al. (2022). *Emergent Abilities of Large Language Models*

### Livres
- Jurafsky & Martin. *Speech and Language Processing* (3rd ed.)
- Goodfellow, Bengio & Courville. *Deep Learning*
- Tunstall et al. *Natural Language Processing with Transformers*

### Ressources en Ligne
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Hugging Face Course](https://huggingface.co/course) — Gratuit et pratique
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) — Exemples de code

---

*Fin du Chapitre 1*
