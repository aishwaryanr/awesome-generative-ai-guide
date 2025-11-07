# CHAPITRE 3 : EMBEDDINGS ET REPRÉSENTATIONS VECTORIELLES

> *« Comment représenter le mot 'chat' pour qu'un ordinateur comprenne qu'il est plus proche de 'chien' que de 'voiture' ? Les embeddings sont la solution. »*

---

## Introduction : Du Symbole au Vecteur

### 🎭 Dialogue : Le Problème de la Représentation

**Alice** : Bob, un ordinateur ne comprend que des nombres. Comment lui faire comprendre le sens des mots ?

**Bob** : Excellente question ! Historiquement, on utilisait du **one-hot encoding** :

```
chat    = [1, 0, 0, 0, 0, ...]  (position 1 dans vocabulaire)
chien   = [0, 1, 0, 0, 0, ...]  (position 2)
voiture = [0, 0, 1, 0, 0, ...]  (position 3)
```

**Alice** : Mais tous les mots sont équidistants ! "chat" est aussi différent de "chien" que de "voiture".

**Bob** : Exactement le problème. Les **embeddings** résolvent ça en représentant chaque mot comme un vecteur dense dans un espace où la **distance reflète la similarité sémantique** :

```
chat    = [0.2, 0.8, -0.1, 0.5, ...]  (300D)
chien   = [0.3, 0.7, -0.2, 0.6, ...]  (proche de chat!)
voiture = [-0.5, 0.1, 0.9, -0.3, ...] (éloigné)
```

**Alice** : Et comment on obtient ces vecteurs magiques ?

**Bob** : C'est tout l'art du chapitre ! De Word2Vec (2013) à BERT (2018) et au-delà.

### 📊 Évolution des Embeddings

| Époque | Méthode | Dimensionalité | Contextuel | Modèle |
|--------|---------|----------------|------------|--------|
| **Pré-2013** | One-hot | Vocab size (50k-1M) | ❌ | - |
| **2013** | Word2Vec | 100-300 | ❌ | Skip-gram, CBOW |
| **2014** | GloVe | 50-300 | ❌ | Matrix factorization |
| **2018** | ELMo | 1024 | ✅ | Bi-LSTM |
| **2018** | BERT | 768-1024 | ✅ | Transformer |
| **2019+** | GPT, T5, etc. | 768-12288 | ✅ | Transformer |

**Révolution clé** : Passage de **statique** (un vecteur par mot) à **contextuel** (vecteur dépend du contexte).

### 🎯 Anecdote : Word2Vec Change Tout

**Été 2013, Google Research**

Tomas Mikolov et son équipe publient "Efficient Estimation of Word Representations in Vector Space".

**Innovation** : Entraîner un réseau de neurones peu profond à prédire le contexte d'un mot. Les poids appris = embeddings !

**Résultat magique** :
```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**Impact** :
- 10× plus rapide à entraîner que les méthodes précédentes
- Qualité supérieure sur toutes les tâches NLP
- Devient le standard de facto (2013-2018)

Même aujourd'hui, Word2Vec reste utilisé pour des applications où les embeddings contextuels sont trop lourds.

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Comprendre intuitivement ce qu'est un embedding
- ✅ Implémenter Word2Vec from scratch
- ✅ Utiliser GloVe et comprendre la différence avec Word2Vec
- ✅ Comprendre les embeddings contextuels (BERT, GPT)
- ✅ Visualiser des embeddings avec t-SNE et UMAP
- ✅ Calculer similarité et distance sémantique
- ✅ Applications pratiques : recherche sémantique, clustering

**Difficulté** : 🟡🟡⚪⚪⚪ (Intermédiaire)
**Prérequis** : Algèbre linéaire, réseaux de neurones basiques
**Temps de lecture** : ~100 minutes

---

## One-Hot Encoding : Le Point de Départ

### Représentation Basique

**Principe** : Chaque mot = vecteur de taille vocab_size avec un seul 1.

```python
import numpy as np

class OneHotEncoder:
    """
    Encodage one-hot basique.
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word_to_id = {word: i for i, word in enumerate(vocabulary)}
        self.id_to_word = {i: word for i, word in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)

    def encode(self, word):
        """Encode un mot en vecteur one-hot."""
        if word not in self.word_to_id:
            raise ValueError(f"Mot '{word}' inconnu")

        vector = np.zeros(self.vocab_size)
        vector[self.word_to_id[word]] = 1
        return vector

    def decode(self, vector):
        """Décode un vecteur one-hot en mot."""
        idx = np.argmax(vector)
        return self.id_to_word[idx]

# Exemple
vocab = ["chat", "chien", "voiture", "maison", "arbre"]
encoder = OneHotEncoder(vocab)

# Encoder
vec_chat = encoder.encode("chat")
vec_chien = encoder.encode("chien")

print(f"chat:   {vec_chat}")    # [1. 0. 0. 0. 0.]
print(f"chien:  {vec_chien}")   # [0. 1. 0. 0. 0.]

# Distance (toujours identique!)
dist_chat_chien = np.linalg.norm(vec_chat - vec_chien)
dist_chat_voiture = np.linalg.norm(vec_chat - encoder.encode("voiture"))

print(f"\nDistance chat-chien:   {dist_chat_chien:.2f}")      # 1.41
print(f"Distance chat-voiture: {dist_chat_voiture:.2f}")     # 1.41 (identique!)
```

### Problèmes du One-Hot

**1. Dimensionalité Explosive**
- Vocabulaire 50,000 mots → vecteurs 50,000D
- Matrice embeddings : [batch_size, seq_len, 50000] (énorme!)

**2. Pas de Similarité Sémantique**
- Tous les mots équidistants
- Impossible de capturer "chat" ≈ "chien"

**3. Sparse**
- 99.99% de zéros → inefficace computationnellement

**4. Pas de Généralisation**
- "chats" (pluriel) totalement différent de "chat"

**Conclusion** : One-hot OK pour petits vocabs, inadéquat pour NLP moderne.

---

## Word2Vec : La Révolution Dense

### Principe : Prédire le Contexte

**Hypothèse distributionnelle** (Harris, 1954) :
> *« Un mot est caractérisé par la compagnie qu'il fréquente. »*

**Word2Vec** entraîne un réseau à prédire le contexte d'un mot. Les poids appris = embeddings !

### Les Deux Architectures

#### 1. Skip-Gram

**Objectif** : Prédire le contexte à partir du mot central.

```
Phrase: "le chat mange la souris"
Mot central: "mange"
Contexte: ["le", "chat", "la", "souris"]

Task: Étant donné "mange", prédire les mots voisins
```

**Architecture** :
```
Input:    "mange" (one-hot: [0,0,1,0,0,...])
           ↓
        Embedding Layer (W_in: [vocab_size, embedding_dim])
           ↓
        Hidden: embedding de "mange" (300D)
           ↓
        Output Layer (W_out: [embedding_dim, vocab_size])
           ↓
        Softmax: probabilités pour chaque mot du vocabulaire
           ↓
        Prédictions: ["le": 0.3, "chat": 0.25, "la": 0.2, ...]
```

#### 2. CBOW (Continuous Bag of Words)

**Objectif** : Prédire le mot central à partir du contexte.

```
Contexte: ["le", "chat", "la", "souris"]
Task: Prédire "mange"
```

**Différence** : Skip-gram fonctionne mieux pour petits datasets, CBOW plus rapide.

### Implémentation Skip-Gram

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np

class SkipGramModel(nn.Module):
    """
    Modèle Skip-Gram pour Word2Vec.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Matrice d'embeddings (poids de la couche cachée)
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Matrice de sortie
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialisation
        self.in_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.out_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, center_word, context_words, negative_samples):
        """
        Forward pass avec negative sampling.

        Args:
            center_word: [batch_size] indices du mot central
            context_words: [batch_size] indices des mots de contexte
            negative_samples: [batch_size, num_neg] indices des samples négatifs
        """
        # Embedding du mot central
        center_embeds = self.in_embeddings(center_word)  # [batch, embed_dim]

        # Embeddings contexte (positifs)
        context_embeds = self.out_embeddings(context_words)  # [batch, embed_dim]

        # Embeddings négatifs
        neg_embeds = self.out_embeddings(negative_samples)  # [batch, num_neg, embed_dim]

        # Score positif : dot product
        pos_score = torch.sum(center_embeds * context_embeds, dim=1)  # [batch]
        pos_loss = -F.logsigmoid(pos_score).mean()

        # Score négatif
        neg_score = torch.bmm(neg_embeds, center_embeds.unsqueeze(2)).squeeze()  # [batch, num_neg]
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=1).mean()

        return pos_loss + neg_loss

    def get_embedding(self, word_idx):
        """Récupère l'embedding d'un mot."""
        return self.in_embeddings.weight[word_idx].detach()


# Préparation des données
def build_vocab(corpus, min_count=5):
    """Construit le vocabulaire."""
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence.lower().split())

    # Filtrer mots rares
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word_to_id = {word: i for i, word in enumerate(vocab)}

    return vocab, word_to_id


def generate_training_data(corpus, word_to_id, window_size=2):
    """
    Génère paires (mot_central, contexte).
    """
    data = []

    for sentence in corpus:
        words = sentence.lower().split()
        word_ids = [word_to_id[w] for w in words if w in word_to_id]

        for i, center_word in enumerate(word_ids):
            # Fenêtre de contexte
            start = max(0, i - window_size)
            end = min(len(word_ids), i + window_size + 1)

            for j in range(start, end):
                if j != i:
                    context_word = word_ids[j]
                    data.append((center_word, context_word))

    return data


# Entraînement
def train_word2vec(corpus, embedding_dim=100, epochs=5, lr=0.01):
    """
    Entraîne Word2Vec Skip-Gram.
    """
    # Build vocab
    vocab, word_to_id = build_vocab(corpus)
    vocab_size = len(vocab)

    # Generate training pairs
    training_data = generate_training_data(corpus, word_to_id)

    # Model
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    batch_size = 128
    num_batches = len(training_data) // batch_size

    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(training_data)

        for i in range(num_batches):
            batch = training_data[i * batch_size:(i + 1) * batch_size]

            # Préparer batch
            center_words = torch.tensor([pair[0] for pair in batch])
            context_words = torch.tensor([pair[1] for pair in batch])

            # Negative samples (échantillonner aléatoirement)
            negative_samples = torch.randint(0, vocab_size, (batch_size, 5))

            # Forward + backward
            optimizer.zero_grad()
            loss = model(center_words, context_words, negative_samples)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, vocab, word_to_id


# Exemple d'utilisation
corpus = [
    "le chat mange la souris",
    "le chien mange la viande",
    "la souris court vite",
    "le chat dort sur le canapé",
] * 100  # Répéter pour avoir assez de données

model, vocab, word_to_id = train_word2vec(corpus, embedding_dim=50, epochs=10)

# Récupérer embeddings
chat_embedding = model.get_embedding(word_to_id["chat"])
chien_embedding = model.get_embedding(word_to_id["chien"])
souris_embedding = model.get_embedding(word_to_id["souris"])

print(f"\nEmbedding 'chat':  {chat_embedding[:5]}")  # Premiers 5 dims
print(f"Embedding 'chien': {chien_embedding[:5]}")
```

### Negative Sampling

**Problème** : Softmax sur vocabulaire complet (50k) est trop coûteux.

**Solution** : Au lieu de normaliser sur tout le vocab, échantillonner k exemples négatifs (k ≈ 5-20).

**Objectif** :
- Maximiser score pour paires (mot, contexte réel)
- Minimiser score pour paires (mot, mot aléatoire)

```python
# Pseudo-code
loss = -log(σ(v_center · v_context))  # Positif
      - Σ log(σ(-v_center · v_negative))  # Négatifs
```

---

## GloVe : Matrix Factorization

### Principe

**GloVe (Global Vectors)** : Combiner statistiques globales de co-occurrence avec apprentissage local.

**Étape 1** : Construire matrice de co-occurrence X
```
X[i,j] = nombre de fois que mot i apparaît dans contexte de mot j
```

**Étape 2** : Factoriser pour apprendre embeddings
```
Objectif: w_i · w_j + b_i + b_j ≈ log(X[i,j])
```

### Utilisation de GloVe Pré-Entraîné

```python
import numpy as np

def load_glove_embeddings(glove_file):
    """
    Charge embeddings GloVe pré-entraînés.

    Download GloVe: https://nlp.stanford.edu/projects/glove/
    """
    embeddings = {}

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector

    return embeddings

# Charger GloVe (exemple: glove.6B.100d.txt)
# glove = load_glove_embeddings("glove.6B.100d.txt")

# Utilisation
# vec_king = glove["king"]
# vec_queen = glove["queen"]
# similarity = np.dot(vec_king, vec_queen) / (np.linalg.norm(vec_king) * np.linalg.norm(vec_queen))
# print(f"Similarité king-queen: {similarity:.3f}")
```

### Word2Vec vs GloVe

| Aspect | Word2Vec | GloVe |
|--------|----------|-------|
| **Méthode** | Prédictive (NN) | Count-based + factorization |
| **Objectif** | Prédire contexte | Factoriser co-occurrences |
| **Entraînement** | Stochastique (SGD) | Global (batch) |
| **Vitesse** | Rapide (online) | Plus lent |
| **Performance** | Légèrement meilleur pour certains tasks | Légèrement meilleur pour d'autres |

**Consensus** : Performances similaires, choisir selon préférence/infrastructure.

---

## Embeddings Contextuels : La Révolution

### Le Problème du Statique

**Word2Vec/GloVe** : Un vecteur par mot, indépendant du contexte.

**Exemple** :
```
"J'ai ouvert un compte en banque"
"Il y a 3 pommes dans mon compte"

"compte" a le MÊME embedding dans les deux phrases!
```

**Problème** : Polysémie ignorée.

### ELMo : Embeddings from Language Models

**ELMo (2018, Peters et al.)** : Utiliser un LSTM bidirectionnel entraîné sur tâche de language modeling.

**Principe** :
```
Phrase: "le chat mange"

→ LSTM forward:  le → chat → mange
→ LSTM backward: mange → chat → le

Embedding("chat") = concat(forward_hidden, backward_hidden)
```

**Résultat** : Embedding dépend du contexte !

```python
# Pseudo-code (avec AllenNLP)
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

sentences = [
    ["Le", "chat", "mange"],
    ["Le", "compte", "est", "ouvert"]
]

embeddings = elmo.embed_sentences(sentences)
# embeddings[0][1] = embedding contextuel de "chat"
# embeddings[1][1] = embedding contextuel de "compte"
```

### BERT : Transformer-Based Contextuels

**BERT (2018)** : Remplace LSTM par Transformer encoder.

```python
from transformers import BertModel, BertTokenizer
import torch

# Charger BERT
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_contextual_embedding(sentence, target_word):
    """
    Obtient l'embedding contextuel d'un mot dans une phrase.
    """
    # Tokeniser
    inputs = tokenizer(sentence, return_tensors="pt")
    tokens = tokenizer.tokenize(sentence)

    # Trouver position du mot cible
    target_idx = None
    for i, token in enumerate(tokens):
        if target_word.lower() in token:
            target_idx = i + 1  # +1 car [CLS] en position 0
            break

    if target_idx is None:
        raise ValueError(f"Mot '{target_word}' non trouvé")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

    # Embedding du mot cible
    embedding = hidden_states[0, target_idx, :]
    return embedding

# Exemples
sentence1 = "J'ai ouvert un compte en banque"
sentence2 = "Il y a 3 pommes dans mon compte"

emb1 = get_contextual_embedding(sentence1, "compte")
emb2 = get_contextual_embedding(sentence2, "compte")

# Similarité
similarity = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
print(f"Similarité 'compte' (contextes différents): {similarity.item():.3f}")
# Plus faible que si même contexte!
```

### Comparaison : Statique vs Contextuel

```python
# Avec Word2Vec (statique)
# "compte" a toujours le même vecteur

# Avec BERT (contextuel)
sentence1 = "compte bancaire"
sentence2 = "compte les pommes"

emb_banque = get_contextual_embedding(sentence1, "compte")
emb_pommes = get_contextual_embedding(sentence2, "compte")

# Ces deux embeddings seront DIFFÉRENTS!
```

---

## Visualisation des Embeddings

### t-SNE : Projection 2D

**t-SNE** : Réduit dimensions élevées (300D) à 2D en préservant les distances locales.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def visualize_embeddings(words, embeddings, title="Word Embeddings"):
    """
    Visualise embeddings en 2D avec t-SNE.

    Args:
        words: Liste de mots
        embeddings: Matrice [num_words, embedding_dim]
    """
    # Réduction dimensionnelle
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    # Annoter chaque point
    for i, word in enumerate(words):
        plt.annotate(
            word,
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 2),
            textcoords='offset points',
            fontsize=12
        )

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Exemple avec GloVe
# words = ["king", "queen", "man", "woman", "cat", "dog", "car", "truck"]
# embeddings = np.array([glove[w] for w in words])
# visualize_embeddings(words, embeddings)
```

### UMAP : Alternative Plus Rapide

```python
from umap import UMAP

def visualize_with_umap(words, embeddings):
    """
    Visualise avec UMAP (plus rapide que t-SNE).
    """
    umap = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

    for i, word in enumerate(words):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title("UMAP Visualization")
    plt.show()
```

---

## Similarité et Distance Sémantique

### Cosine Similarity

**Métrique standard** pour embeddings.

```python
def cosine_similarity(vec1, vec2):
    """
    Similarité cosinus entre deux vecteurs.

    Returns:
        Similarité [-1, 1] (1 = identiques, -1 = opposés)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Exemple
# sim_cat_dog = cosine_similarity(glove["cat"], glove["dog"])
# sim_cat_car = cosine_similarity(glove["cat"], glove["car"])
# print(f"cat-dog: {sim_cat_dog:.3f}")  # ~0.8 (très similaire)
# print(f"cat-car: {sim_cat_car:.3f}")  # ~0.2 (peu similaire)
```

### Most Similar Words

```python
def most_similar(word, embeddings_dict, top_n=10):
    """
    Trouve les N mots les plus similaires.
    """
    if word not in embeddings_dict:
        return []

    target_vec = embeddings_dict[word]
    similarities = {}

    for other_word, other_vec in embeddings_dict.items():
        if other_word != word:
            sim = cosine_similarity(target_vec, other_vec)
            similarities[other_word] = sim

    # Trier
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

# Exemple
# similar_to_king = most_similar("king", glove, top_n=5)
# for word, sim in similar_to_king:
#     print(f"{word}: {sim:.3f}")

# Output attendu:
# queen: 0.85
# monarch: 0.82
# prince: 0.78
# ...
```

### Analogies : King - Man + Woman = ?

```python
def solve_analogy(a, b, c, embeddings_dict, top_n=5):
    """
    Résout analogie: a est à b ce que c est à ?

    Exemple: king - man + woman = queen
    """
    if a not in embeddings_dict or b not in embeddings_dict or c not in embeddings_dict:
        return []

    # Vecteur résultat: king - man + woman
    result_vec = embeddings_dict[a] - embeddings_dict[b] + embeddings_dict[c]

    # Trouver mot le plus proche
    similarities = {}
    for word, vec in embeddings_dict.items():
        if word not in [a, b, c]:
            sim = cosine_similarity(result_vec, vec)
            similarities[word] = sim

    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

# Exemple
# result = solve_analogy("king", "man", "woman", glove)
# print(f"king - man + woman = {result[0][0]}")  # Expected: "queen"

# Plus d'exemples:
# Paris - France + Italy = Rome
# good - bad + ugly = beautiful
```

---

## Applications Pratiques

### 1. Recherche Sémantique

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    """
    Moteur de recherche sémantique basé sur embeddings.
    """
    def __init__(self, documents, model, tokenizer):
        """
        Args:
            documents: Liste de textes
            model: Modèle BERT pour embeddings
            tokenizer: Tokenizer BERT
        """
        self.documents = documents
        self.model = model
        self.tokenizer = tokenizer

        # Pré-calculer embeddings des documents
        self.doc_embeddings = self._compute_embeddings(documents)

    def _compute_embeddings(self, texts):
        """Calcule embeddings moyens pour chaque texte."""
        embeddings = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Moyenne des embeddings de tous les tokens
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

            embeddings.append(embedding.numpy())

        return np.array(embeddings)

    def search(self, query, top_k=5):
        """
        Recherche les k documents les plus pertinents.
        """
        # Embedding de la requête
        query_embedding = self._compute_embeddings([query])[0]

        # Similarités
        similarities = cosine_similarity([query_embedding], self.doc_embeddings)[0]

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            {"document": self.documents[i], "score": similarities[i]}
            for i in top_indices
        ]

        return results

# Exemple
# documents = [
#     "Le chat mange des croquettes",
#     "Les voitures électriques sont écologiques",
#     "Le chien aboie dans le jardin",
#     "Les véhicules thermiques polluent l'environnement"
# ]
#
# search_engine = SemanticSearch(documents, model, tokenizer)
# results = search_engine.search("animaux domestiques")
#
# for r in results:
#     print(f"{r['score']:.3f}: {r['document']}")
```

### 2. Clustering de Documents

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def cluster_documents(documents, model, tokenizer, n_clusters=3):
    """
    Cluster des documents basés sur leurs embeddings.
    """
    # Compute embeddings
    search_engine = SemanticSearch(documents, model, tokenizer)
    embeddings = search_engine.doc_embeddings

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # Visualiser avec t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)

    # Annoter documents
    for i, doc in enumerate(documents):
        plt.annotate(
            doc[:30] + "..." if len(doc) > 30 else doc,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8
        )

    plt.colorbar(scatter, label='Cluster')
    plt.title("Document Clustering")
    plt.show()

    # Grouper par cluster
    clustered_docs = {i: [] for i in range(n_clusters)}
    for doc, cluster_id in zip(documents, clusters):
        clustered_docs[cluster_id].append(doc)

    return clustered_docs
```

---

## 💡 Analogie : Les Embeddings comme une Carte

Imaginez que vous devez représenter des villes sur une carte :

- **One-hot** : Chaque ville a sa propre dimension. Paris = Nord, Rome = Sud... Impossible de mesurer "proximité" !

- **Word2Vec** : Carte 2D où distance géographique est préservée. Paris proche de Londres, loin de Tokyo. **Mais** : Paris a toujours la même position, même si contexte change.

- **BERT (contextuel)** : Carte qui se **déforme** selon le contexte. "Paris" dans "Paris is romantic" est proche de "love", mais dans "Paris is a capital" proche de "government". Position change !

---

## Quiz Interactif

### Question 1 : One-Hot vs Dense

**Pourquoi les embeddings denses sont meilleurs que one-hot ?**

A) Plus rapides à calculer
B) Capturent similarité sémantique
C) Prennent moins de mémoire
D) B et C

<details>
<summary>Voir la réponse</summary>

**Réponse : D) B et C**

**Embeddings denses (Word2Vec, GloVe)** :
- ✅ Capturent similarité : "cat" proche de "dog"
- ✅ Moins de mémoire : 300D au lieu de 50,000D
- ✅ Généralisent mieux

**One-hot** :
- ❌ Tous mots équidistants
- ❌ Sparse, inefficace
</details>

---

### Question 2 : Word2Vec Architectures

**Quelle est la différence entre Skip-Gram et CBOW ?**

A) Skip-Gram prédit contexte → mot, CBOW prédit mot → contexte
B) Skip-Gram prédit mot → contexte, CBOW prédit contexte → mot
C) Aucune différence
D) Skip-Gram est plus récent

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Skip-Gram prédit mot → contexte, CBOW prédit contexte → mot**

**Skip-Gram** : Input = mot central, Output = mots du contexte
**CBOW** : Input = contexte, Output = mot central

**Performance** : Skip-gram meilleur pour petits datasets, CBOW plus rapide.
</details>

---

### Question 3 : Contextuels

**Quelle affirmation est vraie sur BERT ?**

A) Un vecteur par mot (comme Word2Vec)
B) Embedding dépend du contexte
C) Plus lent que Word2Vec pour inference
D) B et C

<details>
<summary>Voir la réponse</summary>

**Réponse : D) B et C**

**BERT** :
- ✅ Embeddings **contextuels** (dépendent de la phrase)
- ✅ Plus lent (Forward pass Transformer entier)
- ❌ Pas un vecteur statique par mot

**Avantage** : Polysémie gérée ("compte" bancaire vs "compte" les pommes).
</details>

---

### Question 4 : Analogies

**L'analogie "king - man + woman ≈ queen" fonctionne car :**

A) C'est codé en dur dans l'algorithme
B) Les embeddings capturent relations sémantiques via arithmétique vectorielle
C) C'est une coïncidence
D) Ça ne fonctionne pas vraiment

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Les embeddings capturent relations sémantiques via arithmétique vectorielle**

**Explication** :
- "king" - "man" ≈ vecteur "royauté"
- "woman" + vecteur "royauté" ≈ "queen"

Les embeddings apprennent que certaines **directions** dans l'espace correspondent à des **relations** (genre, taille, etc.).
</details>

---

### Question 5 : Métriques

**Pour mesurer similarité entre embeddings, on utilise :**

A) Distance euclidienne
B) Cosine similarity
C) Manhattan distance
D) Toutes les réponses

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Cosine similarity**

**Cosine similarity** est la métrique standard car :
- Invariante à la magnitude (seule direction compte)
- Range [-1, 1] (facile à interpréter)
- Correspond à intuition sémantique

**Distance euclidienne** peut être biaisée par magnitude des vecteurs.
</details>

---

## Exercices Pratiques

### Exercice 1 : Implémenter CBOW

**Objectif** : Compléter l'implémentation CBOW (inverse de Skip-Gram).

```python
class CBOWModel(nn.Module):
    """
    Continuous Bag of Words model.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # TODO: Définir layers
        pass

    def forward(self, context_words):
        """
        Args:
            context_words: [batch_size, context_size] indices
        Returns:
            logits: [batch_size, vocab_size]
        """
        # TODO:
        # 1. Récupérer embeddings du contexte
        # 2. Moyenner les embeddings
        # 3. Passer par couche de sortie
        # 4. Retourner logits
        pass
```

<details>
<summary>Voir la solution</summary>

```python
class CBOWModel(nn.Module):
    """
    Continuous Bag of Words model.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        """
        Args:
            context_words: [batch_size, context_size]
        Returns:
            logits: [batch_size, vocab_size]
        """
        # 1. Embeddings du contexte
        embeds = self.embeddings(context_words)  # [batch, context_size, embed_dim]

        # 2. Moyenne
        mean_embed = embeds.mean(dim=1)  # [batch, embed_dim]

        # 3. Projection vers vocabulaire
        logits = self.linear(mean_embed)  # [batch, vocab_size]

        return logits

# Training
model = CBOWModel(vocab_size=10000, embedding_dim=100)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Exemple batch
context = torch.randint(0, 10000, (32, 4))  # 32 exemples, contexte de 4 mots
target = torch.randint(0, 10000, (32,))     # Mot central à prédire

# Forward
logits = model(context)
loss = criterion(logits, target)

# Backward
loss.backward()
optimizer.step()

print(f"Loss: {loss.item():.4f}")
```
</details>

---

### Exercice 2 : Évaluer Qualité des Embeddings

**Objectif** : Tester embeddings sur tâches d'analogies.

```python
def evaluate_analogies(embeddings_dict, analogy_dataset):
    """
    Évalue embeddings sur dataset d'analogies.

    Format analogy_dataset:
    [
        {"a": "king", "b": "man", "c": "woman", "d": "queen"},
        {"a": "Paris", "b": "France", "c": "Italy", "d": "Rome"},
        ...
    ]

    Returns:
        accuracy: % d'analogies correctement résolues
    """
    # TODO: Implémenter
    pass
```

<details>
<summary>Voir la solution</summary>

```python
def evaluate_analogies(embeddings_dict, analogy_dataset):
    """
    Évalue embeddings sur analogies.
    """
    correct = 0
    total = 0

    for analogy in analogy_dataset:
        a, b, c, d_true = analogy["a"], analogy["b"], analogy["c"], analogy["d"]

        # Vérifier que tous les mots sont dans le vocabulaire
        if all(word in embeddings_dict for word in [a, b, c, d_true]):
            # Résoudre: a - b + c = ?
            result_vec = embeddings_dict[a] - embeddings_dict[b] + embeddings_dict[c]

            # Trouver mot le plus proche (exclure a, b, c)
            best_word = None
            best_sim = -1

            for word, vec in embeddings_dict.items():
                if word not in [a, b, c]:
                    sim = cosine_similarity(result_vec, vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_word = word

            # Vérifier si correct
            if best_word == d_true:
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# Dataset d'analogies
analogies = [
    {"a": "king", "b": "man", "c": "woman", "d": "queen"},
    {"a": "Paris", "b": "France", "c": "Italy", "d": "Rome"},
    {"a": "good", "b": "better", "c": "bad", "d": "worse"},
    # ... plus d'exemples
]

# Évaluer
# accuracy = evaluate_analogies(glove, analogies)
# print(f"Accuracy: {accuracy:.2%}")
```
</details>

---

### Exercice 3 : Visualiser Évolution des Embeddings

**Objectif** : Tracer évolution des embeddings durant training Word2Vec.

```python
def visualize_training_evolution(model, words, word_to_id, epochs=10):
    """
    Visualise comment les embeddings évoluent durant training.

    Affiche t-SNE à epochs 1, 5, et 10.
    """
    # TODO: Implémenter
    pass
```

<details>
<summary>Voir la solution</summary>

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy

def visualize_training_evolution(corpus, words_to_track, embedding_dim=50, epochs=10):
    """
    Visualise évolution des embeddings durant training.
    """
    vocab, word_to_id = build_vocab(corpus)
    training_data = generate_training_data(corpus, word_to_id)

    model = SkipGramModel(len(vocab), embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Snapshots à différentes epochs
    snapshots = {1: None, 5: None, 10: None}

    batch_size = 128
    num_batches = len(training_data) // batch_size

    for epoch in range(1, epochs + 1):
        # Training
        np.random.shuffle(training_data)
        for i in range(num_batches):
            batch = training_data[i * batch_size:(i + 1) * batch_size]

            center_words = torch.tensor([pair[0] for pair in batch])
            context_words = torch.tensor([pair[1] for pair in batch])
            negative_samples = torch.randint(0, len(vocab), (batch_size, 5))

            optimizer.zero_grad()
            loss = model(center_words, context_words, negative_samples)
            loss.backward()
            optimizer.step()

        # Sauvegarder snapshot
        if epoch in snapshots:
            embeddings = []
            for word in words_to_track:
                if word in word_to_id:
                    idx = word_to_id[word]
                    emb = model.get_embedding(idx).numpy()
                    embeddings.append(emb)

            snapshots[epoch] = np.array(embeddings)

    # Visualiser les 3 snapshots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (epoch, embeddings) in zip(axes, snapshots.items()):
        if embeddings is not None:
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Plot
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

            for i, word in enumerate(words_to_track):
                if word in word_to_id:
                    ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

            ax.set_title(f"Epoch {epoch}")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Exemple
# words_to_track = ["cat", "dog", "car", "truck", "king", "queen"]
# visualize_training_evolution(corpus, words_to_track, epochs=10)
```
</details>

---

## Conclusion

### 🎭 Dialogue Final : Des Symboles aux Vecteurs Intelligents

**Alice** : Maintenant je comprends : les embeddings transforment des symboles arbitraires en géométrie sémantique !

**Bob** : Exactement. C'est la **pierre angulaire** de tout le NLP moderne :
- **2013-2018** : Word2Vec, GloVe → embeddings statiques
- **2018+** : BERT, GPT → embeddings contextuels
- **Futur** : Embeddings multimodaux (texte + image + audio)

**Alice** : Quelle est la clé d'un bon embedding ?

**Bob** : Trois critères :
1. **Similarité sémantique** : Mots similaires → vecteurs proches
2. **Compositionnalité** : "king - man + woman = queen"
3. **Généralisation** : Fonctionne sur mots jamais vus (via subwords)

**Alice** : Et les limites ?

**Bob** : Biais ! Les embeddings reflètent les biais du corpus :
```
doctor - man + woman ≈ nurse  (stéréotype de genre!)
```

Il faut **debiaser** ou être conscient des biais.

### 🎯 Points Clés à Retenir

| Concept | Essence |
|---------|---------|
| **Embedding** | Représentation dense d'un mot dans un espace vectoriel |
| **Word2Vec** | Prédire contexte → apprendre embeddings statiques |
| **GloVe** | Factoriser matrice co-occurrences |
| **BERT** | Embeddings **contextuels** (dépendent de la phrase) |
| **Similarité** | Cosine similarity (standard) |
| **Analogies** | Arithmétique vectorielle capture relations |

### 📊 Quand Utiliser Quoi ?

| Cas d'usage | Méthode Recommandée |
|-------------|---------------------|
| **Classification texte** | BERT fine-tuné |
| **Recherche sémantique** | Sentence-BERT, embeddings contextuels |
| **Analogies, maths mots** | Word2Vec, GloVe (plus simple) |
| **Multilingual** | XLM-RoBERTa, mBERT |
| **Ressources limitées** | Word2Vec (léger, rapide) |
| **SOTA performance** | Modèles récents (GPT, T5, etc.) |

---

## Ressources

### 📚 Papers Fondamentaux

1. **"Efficient Estimation of Word Representations in Vector Space"** (Mikolov et al., 2013) - Word2Vec
2. **"GloVe: Global Vectors for Word Representation"** (Pennington et al., 2014)
3. **"Deep contextualized word representations"** (Peters et al., 2018) - ELMo
4. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)

### 🛠️ Code et Ressources

```bash
# Gensim (Word2Vec, FastText)
pip install gensim

# HuggingFace Transformers (BERT, GPT, etc.)
pip install transformers

# Visualisation
pip install umap-learn scikit-learn
```

**Embeddings pré-entraînés** :
- GloVe : https://nlp.stanford.edu/projects/glove/
- FastText : https://fasttext.cc/
- Word2Vec : https://code.google.com/archive/p/word2vec/

---

**🎓 Bravo !** Vous maîtrisez maintenant les embeddings, fondation de tout le NLP moderne. Prochain chapitre : **Chapitre 4 - Architectures Transformers** pour voir comment ces embeddings sont utilisés dans les LLMs ! 🚀

