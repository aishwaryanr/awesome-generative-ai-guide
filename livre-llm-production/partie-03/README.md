# Partie 3 : Bases de deep learning appliquées au texte

## Objectifs d'apprentissage

- Comprendre les graphes computationnels et la backpropagation
- Maîtriser les couches fondamentales des réseaux de neurones
- Connaître les modèles séquentiels pré-Transformer (RNN, LSTM, GRU)
- Comprendre la tokenisation et son impact sur les performances
- Construire un modèle de langage simple from scratch

## Prérequis

- Partie 2 validée (fondations mathématiques)
- Python et PyTorch
- Compréhension des gradients et de l'optimisation

---

## 3.1 Graphes computationnels et autodifférenciation

### 3.1.1 Principe des graphes computationnels

Un réseau de neurones est un graphe orienté acyclique (DAG) où :
- **Nœuds** : opérations mathématiques (addition, multiplication, activation)
- **Arêtes** : flux de données (tenseurs)

**Exemple simple** :

```
x → × → + → σ → y
    ↑   ↑
    w   b
```

Représente : `y = σ(w × x + b)` où σ est une fonction d'activation.

### 3.1.2 Forward pass et backward pass

**Forward pass** : Calcul de la sortie en propageant les données.

```python
# Forward
z = w * x + b
y = torch.sigmoid(z)
```

**Backward pass** : Calcul des gradients par la règle de la chaîne.

```
∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w
```

**Implémentation PyTorch** :

```python
import torch

# Créer des tensors avec suivi de gradient
x = torch.randn(10, requires_grad=True)
w = torch.randn(10, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Forward
y = torch.sigmoid(w * x + b)
loss = y.sum()

# Backward
loss.backward()

# Accéder aux gradients
print(w.grad)  # ∂L/∂w
print(b.grad)  # ∂L/∂b
```

### 3.1.3 Backpropagation

**Algorithme** :

1. **Forward** : Calculer toutes les sorties intermédiaires et la loss finale
2. **Backward** : Pour chaque couche, de la sortie vers l'entrée :
   - Recevoir le gradient de la couche suivante
   - Calculer le gradient local
   - Propager le gradient à la couche précédente

**Exemple détaillé** :

```python
class SimpleLinear:
    def __init__(self, in_features, out_features):
        self.W = torch.randn(in_features, out_features, requires_grad=True)
        self.b = torch.randn(out_features, requires_grad=True)

    def forward(self, x):
        self.x = x  # sauvegarder pour backward
        return x @ self.W + self.b

    def backward(self, grad_output):
        # ∂L/∂W = x^T @ ∂L/∂y
        self.W.grad = self.x.T @ grad_output
        # ∂L/∂b = sum(∂L/∂y, axis=0)
        self.b.grad = grad_output.sum(dim=0)
        # ∂L/∂x = ∂L/∂y @ W^T
        return grad_output @ self.W.T
```

---

## 3.2 Couches fondamentales

### 3.2.1 Couche linéaire (fully-connected)

**Définition** :

```
y = xW^T + b
```

où x ∈ ℝ^(batch × in_features), y ∈ ℝ^(batch × out_features).

**PyTorch** :

```python
linear = torch.nn.Linear(in_features=768, out_features=3072)
x = torch.randn(32, 768)  # batch=32
y = linear(x)  # shape: [32, 3072]
```

### 3.2.2 Fonctions d'activation

**ReLU** (Rectified Linear Unit) :

```
ReLU(x) = max(0, x)
```

Avantages : simple, pas de saturation pour x > 0.

**GELU** (Gaussian Error Linear Unit) :

```
GELU(x) = x × Φ(x)  où Φ est la CDF gaussienne
```

Utilisée dans BERT, GPT-2, GPT-3 (plus douce que ReLU).

**Comparaison** :

```python
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)
relu = torch.relu(x)
gelu = torch.nn.functional.gelu(x)

plt.plot(x, relu, label='ReLU')
plt.plot(x, gelu, label='GELU')
plt.legend()
plt.show()
```

### 3.2.3 Normalisation

**BatchNorm** (pour CNN, moins pour NLP) :

Normalise sur la dimension batch.

**LayerNorm** (standard pour Transformers) :

```
LN(x) = γ × (x - μ) / σ + β
```

où μ et σ sont la moyenne et l'écart-type sur la dimension des features.

```python
layer_norm = torch.nn.LayerNorm(768)
x = torch.randn(32, 512, 768)  # [batch, seq, hidden]
x_normalized = layer_norm(x)
```

**Pourquoi LayerNorm ?**
- Stabilise l'entraînement profond
- Indépendant de la taille du batch
- Meilleure généralisation pour le NLP

### 3.2.4 Connexions résiduelles

**Principe** :

```
y = x + F(x)
```

Permet au gradient de circuler directement (évite le vanishing gradient).

**Implémentation** :

```python
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x + residual  # connexion résiduelle
```

### 3.2.5 Dropout

**Principe** : Désactiver aléatoirement des neurones pendant l'entraînement.

```python
dropout = torch.nn.Dropout(p=0.1)  # 10% des neurones désactivés

x = torch.randn(32, 768)
x_dropped = dropout(x)  # en train mode
# En eval mode : dropout.eval() → pas de dropout
```

**Effet** : Régularisation, réduit l'overfitting.

---

## 3.3 Modèles séquentiels pré-Transformer

### 3.3.1 Modèles N-gramme

**Principe** : Prédire le prochain mot basé sur les N-1 mots précédents.

```
P(w_t | w_{t-N+1}, ..., w_{t-1})
```

**Limites** :
- Contexte fixe et court (N petit pour éviter sparsité)
- Pas de partage de représentation entre contextes similaires

### 3.3.2 RNN (Recurrent Neural Networks)

**Architecture** :

```
h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y
```

**Implémentation PyTorch** :

```python
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
```

**Problèmes** :
- **Vanishing gradient** : Difficile de capturer dépendances longues
- **Traitement séquentiel** : Pas de parallélisation

### 3.3.3 LSTM (Long Short-Term Memory)

**Architecture** : Ajoute des gates pour contrôler le flux d'information.

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)  # forget gate
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)  # input gate
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)  # output gate

C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)  # candidate cell state
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t        # cell state
h_t = o_t ⊙ tanh(C_t)                   # hidden state
```

**PyTorch** :

```python
lstm = torch.nn.LSTM(input_size=768, hidden_size=512, num_layers=2)
x = torch.randn(20, 32, 768)  # [seq_len, batch, input_size]
output, (h_n, c_n) = lstm(x)
```

### 3.3.4 GRU (Gated Recurrent Unit)

Version simplifiée de LSTM (2 gates au lieu de 3).

```
z_t = σ(W_z × [h_{t-1}, x_t])  # update gate
r_t = σ(W_r × [h_{t-1}, x_t])  # reset gate
h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**Avantages GRU vs LSTM** :
- Moins de paramètres
- Entraînement plus rapide
- Performances souvent similaires

---

## 3.4 Tokenisation

### 3.4.1 Pourquoi tokeniser ?

**Problème** : Les modèles travaillent avec des IDs numériques, pas du texte brut.

**Solutions** :
1. **Caractères** : Vocabulaire petit (~100), séquences longues
2. **Mots** : Vocabulaire énorme (100k+), OOV (out-of-vocabulary)
3. **Subwords** : Compromis optimal (vocabulaire ~30k-50k)

### 3.4.2 Byte-Pair Encoding (BPE)

**Algorithme** :

1. Commencer avec un vocabulaire de caractères
2. Itérativement fusionner la paire la plus fréquente
3. Répéter jusqu'à atteindre la taille de vocabulaire souhaitée

**Exemple** :

```
Texte : "low low low lower lowest"

Itération 1 : "l o w" → "lo w" (fusion "l" + "o")
Itération 2 : "lo w" → "low" (fusion "lo" + "w")
Itération 3 : "low e r" → "low er" (fusion "e" + "r")
...
```

**Implémentation avec tokenizers** :

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialiser
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Entraîner
trainer = BpeTrainer(vocab_size=10000, special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)

# Utiliser
output = tokenizer.encode("Hello world!")
print(output.tokens)  # ['Hello', 'world', '!']
print(output.ids)     # [234, 1045, 78]
```

### 3.4.3 WordPiece et Unigram

**WordPiece** (BERT) :
- Similaire à BPE mais choisit les fusions basé sur la likelihood

**Unigram** (SentencePiece, T5) :
- Approche probabiliste, supprime itérativement des tokens

### 3.4.4 SentencePiece

Framework unifié pour BPE, Unigram, char, word.

```python
import sentencepiece as spm

# Entraîner
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe'
)

# Charger
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Tokeniser
tokens = sp.encode('Hello world!', out_type=str)
ids = sp.encode('Hello world!', out_type=int)

print(tokens)  # ['▁Hello', '▁world', '!']
print(ids)     # [284, 296, 67]
```

### 3.4.5 Impact de la tokenisation sur les performances

**Taille de vocabulaire** :

| Vocab size | Avantages                         | Inconvénients                   |
|------------|-----------------------------------|---------------------------------|
| Petit (~1k)| Modèle compact, peu de paramètres | Séquences longues, moins précis |
| Moyen (~30k)| Bon compromis                    | Standard actuel                 |
| Grand (~100k+)| Séquences courtes, précis      | Beaucoup de paramètres embeddings|

**Longueur moyenne des tokens** :

Plus le vocabulaire est grand, moins de tokens par phrase, mais plus de paramètres dans l'embedding matrix.

---

## 3.5 Labs : Construire un modèle de langage simple

### Lab 1 : Character-level language model

**Objectif** : Prédire le caractère suivant dans une séquence.

```python
import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: [seq_len, batch]
        x = self.embedding(x)  # [seq_len, batch, hidden]
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # [seq_len, batch, vocab_size]
        return out, hidden

# Entraînement simple
vocab_size = 100  # ASCII étendu
model = CharRNN(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dataset exemple (à remplacer par vrai corpus)
text = "Hello world" * 1000
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([char_to_idx[ch] for ch in text])

# Training loop
for epoch in range(10):
    for i in range(0, len(data) - 10, 10):
        x = data[i:i+10].unsqueeze(1)  # [seq_len, batch=1]
        y = data[i+1:i+11].unsqueeze(1)

        out, _ = model(x)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Lab 2 : Subword language model avec BPE

```python
# TODO: Entraîner un tokenizer BPE sur un corpus
# TODO: Construire un modèle LSTM avec embeddings BPE
# TODO: Comparer perplexité avec le modèle char-level
```

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Graphes computationnels et backpropagation
- ✅ Couches fondamentales (linéaire, activation, normalisation, dropout, résiduelle)
- ✅ Modèles séquentiels (RNN, LSTM, GRU) et leurs limites
- ✅ Tokenisation (BPE, WordPiece, Unigram, SentencePiece)
- ✅ Construction d'un LM simple

**Prochaine étape** : [Partie 4 - Architectures de LLM modernes](../partie-04/README.md)
