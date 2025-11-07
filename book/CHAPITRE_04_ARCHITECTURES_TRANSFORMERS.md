# CHAPITRE 4 : ARCHITECTURES TRANSFORMERS - SOUS LE CAPOT

> *« Attention is all you need. » — Trois mots qui ont changé l'IA pour toujours. Mais que se cache-t-il vraiment sous le capot de cette architecture révolutionnaire ?*

---

## 📖 Table des matières

1. [Introduction : La Révolution de l'Attention](#1-introduction)
2. [Anatomie d'un Transformer](#2-anatomie)
3. [Self-Attention : Le Cœur du Système](#3-self-attention)
4. [Multi-Head Attention](#4-multi-head-attention)
5. [Position Encodings](#5-position-encodings)
6. [Feed-Forward Networks](#6-feed-forward)
7. [Layer Normalization & Residual Connections](#7-layer-norm)
8. [Les Trois Familles d'Architectures](#8-trois-familles)
9. [Variantes Modernes](#9-variantes-modernes)
10. [Implémentation from Scratch](#10-implementation)
11. [Quiz Interactif](#11-quiz)
12. [Exercices Pratiques](#12-exercices)
13. [Conclusion](#13-conclusion)
14. [Ressources](#14-ressources)

---

## 1. Introduction : La Révolution de l'Attention {#1-introduction}

### 🎭 Dialogue : Le Mystère du Transformer

**Alice** : Bob, j'ai entendu dire que les Transformers ont révolutionné l'IA. Mais concrètement, qu'est-ce qui les rend si spéciaux ?

**Bob** : Imagine que tu lis une phrase : "La banque a refusé mon prêt car mon **compte** était insuffisant."

**Alice** : Ok...

**Bob** : Pour comprendre "compte", tu dois regarder "banque" et "prêt". Un RNN lirait mot par mot, séquentiellement. Un Transformer **regarde tous les mots en même temps** et calcule : "compte est lié à banque (attention forte) et à prêt (attention forte), mais pas à 'La' (attention faible)".

**Alice** : C'est comme avoir une vision globale plutôt que tunnel !

**Bob** : Exactement. Et cette capacité à "prêter attention" à n'importe quel mot, peu importe la distance, c'est le secret de leur puissance.

### 📊 Avant et Après les Transformers

| Aspect | RNN/LSTM (avant 2017) | Transformer (2017+) |
|--------|----------------------|---------------------|
| **Traitement** | Séquentiel (lent) | Parallèle (rapide) |
| **Mémoire longue** | Oublie après ~100 tokens | Attention sur 1000s de tokens |
| **Entraînement** | Difficile (vanishing gradients) | Stable |
| **Scalabilité** | Limitée | Excellente |
| **SOTA sur NLP** | Peu de tâches | Toutes les tâches |

### 🎯 Anecdote : La Naissance des Transformers

**Été 2017, Google Brain, Mountain View**

Une équipe de chercheurs menée par Ashish Vaswani travaille sur la traduction automatique. Les modèles LSTM atteignent un plateau de performance.

*Vaswani* : "Et si on supprimait complètement la récurrence ? Si on ne gardait que l'attention ?"

*Collègue* : "Impossible. Comment le modèle saurait l'ordre des mots ?"

*Vaswani* : "Avec des positional encodings. L'attention pour le contenu, les encodings pour la position."

6 mois plus tard, le paper **"Attention is All You Need"** sort. Résultats sur WMT translation :
- **LSTM SOTA** : 25.2 BLEU
- **Transformer (base)** : 27.3 BLEU
- **Transformer (big)** : **28.4 BLEU** (nouveau record)

Et surtout : **10x plus rapide à entraîner**.

Le reste appartient à l'histoire : BERT (2018), GPT-2 (2019), GPT-3 (2020), ChatGPT (2022)...

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Comprendre chaque composant du Transformer (attention, FFN, layer norm, etc.)
- ✅ Implémenter un Transformer from scratch en PyTorch
- ✅ Distinguer encoder-only, decoder-only, et encoder-decoder
- ✅ Connaître les variantes modernes (GPT, BERT, T5, LLaMA)
- ✅ Optimiser les Transformers (Flash Attention, ALiBi, etc.)

**Difficulté** : 🔴🔴🔴⚪⚪ (Avancé)
**Prérequis** : Algèbre linéaire, réseaux de neurones, PyTorch
**Temps de lecture** : ~120 minutes

---

## 2. Anatomie d'un Transformer {#2-anatomie}

### 2.1 Vue d'Ensemble

Un Transformer est composé de **blocs empilés**, chacun contenant :

```
┌─────────────────────────────────────┐
│         INPUT EMBEDDINGS            │
│    (tokens → vectors 512D)          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      POSITIONAL ENCODING            │
│   (injecter l'ordre des mots)       │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │  TRANSFORMER  │ ×N layers (ex: 12)
       │     BLOCK     │
       └───────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌────────────┐      ┌────────────┐
│ MULTI-HEAD │      │ MULTI-HEAD │
│ ATTENTION  │      │ ATTENTION  │
│ (self)     │      │ (cross)    │
└─────┬──────┘      └─────┬──────┘
      │                   │
      ▼                   ▼
┌────────────┐      ┌────────────┐
│  ADD &     │      │  ADD &     │
│  NORM      │      │  NORM      │
└─────┬──────┘      └─────┬──────┘
      │                   │
      ▼                   ▼
┌────────────┐
│ FEED-      │
│ FORWARD    │
└─────┬──────┘
      │
      ▼
┌────────────┐
│  ADD &     │
│  NORM      │
└─────┬──────┘
      │
      ▼
┌─────────────────────────────────────┐
│        LINEAR + SOFTMAX             │
│     (prédiction du token)           │
└─────────────────────────────────────┘
```

### 2.2 Hyperparamètres Typiques

| Modèle | Layers (N) | Hidden Dim (d_model) | Heads | Params | Context |
|--------|-----------|---------------------|-------|--------|---------|
| **BERT-base** | 12 | 768 | 12 | 110M | 512 |
| **BERT-large** | 24 | 1024 | 16 | 340M | 512 |
| **GPT-2** | 12 | 768 | 12 | 117M | 1024 |
| **GPT-3** | 96 | 12288 | 96 | 175B | 2048 |
| **LLaMA-7B** | 32 | 4096 | 32 | 7B | 2048 |
| **LLaMA-65B** | 80 | 8192 | 64 | 65B | 2048 |

### 💡 Analogie : Le Transformer comme une Usine

Imaginez une **usine de compréhension de texte** :

1. **Entrée** : Camions de mots arrivent
2. **Embeddings** : Chaque mot reçoit un badge numérique (vecteur 512D)
3. **Position** : On tamponne "1er", "2ème", etc. sur les badges
4. **Attention** : Salle de réunion où chaque mot discute avec tous les autres
5. **Feed-Forward** : Chaque mot passe par une machine de transformation individuelle
6. **Sortie** : Produits finis (prédictions, traductions, etc.)

Et on répète 12-96 fois (selon le modèle) !

---

## 3. Self-Attention : Le Cœur du Système {#3-self-attention}

### 3.1 Le Problème à Résoudre

**Phrase** : "The **animal** didn't cross the street because **it** was too tired."

**Question** : À quoi réfère "it" ?

Un humain comprend immédiatement : **"it" = "the animal"** (pas "the street").

Comment le modèle peut-il le déduire ? Via **l'attention** !

### 3.2 Mécanisme d'Attention : Queries, Keys, Values

**Idée** : Chaque mot génère 3 vecteurs :

1. **Query (Q)** : "Ce que je cherche"
2. **Key (K)** : "Ce que je propose"
3. **Value (V)** : "L'information que je porte"

**Processus** :
1. Calculer **scores d'attention** : Similitude entre Q de "it" et K de tous les mots
2. Normaliser avec **softmax**
3. Pondérer les **Values** par ces scores
4. Sommer pour obtenir la représentation finale

### 3.3 Formule Mathématique

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Où :
- `Q` : Matrice de queries (shape: [seq_len, d_k])
- `K` : Matrice de keys (shape: [seq_len, d_k])
- `V` : Matrice de values (shape: [seq_len, d_v])
- `d_k` : Dimension des keys (pour normalisation)
- `√d_k` : Scaling factor (évite que softmax sature)

### 3.4 Visualisation Étape par Étape

**Phrase** : "The cat sat"

**Étape 1 : Embeddings**
```
The  → [0.2, 0.5, 0.1, ...] (512D)
cat  → [0.8, 0.1, 0.3, ...]
sat  → [0.3, 0.7, 0.2, ...]
```

**Étape 2 : Projections linéaires**
```
Q_the = W_q × emb_the
K_the = W_k × emb_the
V_the = W_v × emb_the

(idem pour "cat" et "sat")
```

**Étape 3 : Scores d'attention (pour "cat")**
```
score_cat→the = Q_cat · K_the / √d_k = 0.3
score_cat→cat = Q_cat · K_cat / √d_k = 0.9  (forte!)
score_cat→sat = Q_cat · K_sat / √d_k = 0.6
```

**Étape 4 : Softmax**
```
weights = softmax([0.3, 0.9, 0.6])
        = [0.15, 0.55, 0.30]
```

**Étape 5 : Pondération des Values**
```
output_cat = 0.15×V_the + 0.55×V_cat + 0.30×V_sat
```

**Interprétation** : "cat" prête **55% d'attention à lui-même**, 30% à "sat", 15% à "the".

### 3.5 Implémentation PyTorch

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Queries [batch, seq_len, d_k]
            K: Keys    [batch, seq_len, d_k]
            V: Values  [batch, seq_len, d_v]
            mask: Masque [batch, seq_len, seq_len] (optionnel)

        Returns:
            output: [batch, seq_len, d_v]
            attention_weights: [batch, seq_len, seq_len]
        """
        d_k = Q.size(-1)

        # 1. Calcul des scores : QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # Shape: [batch, seq_len, seq_len]

        # 2. Application du masque (pour causal attention dans GPT)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax sur la dernière dimension
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. Pondération des values
        output = torch.matmul(attention_weights, V)
        # Shape: [batch, seq_len, d_v]

        return output, attention_weights


# Test
batch_size, seq_len, d_model = 2, 5, 512
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

attention = ScaledDotProductAttention()
output, weights = attention(Q, K, V)

print(f"Output shape: {output.shape}")        # [2, 5, 512]
print(f"Attention weights: {weights.shape}")  # [2, 5, 5]
print(f"Somme des poids (ligne 1): {weights[0, 0].sum()}")  # ≈ 1.0 (softmax)
```

### 3.6 Masking : Causal vs Bidirectionnel

#### A) Bidirectional Attention (BERT)

Chaque token voit **tous les tokens** (passé + futur).

```
Matrice d'attention (no mask):
     The  cat  sat
The  [1]  [1]  [1]   ← "The" voit tout
cat  [1]  [1]  [1]   ← "cat" voit tout
sat  [1]  [1]  [1]   ← "sat" voit tout
```

**Usage** : Compréhension de texte (classification, NER, etc.)

#### B) Causal Attention (GPT)

Chaque token voit **seulement le passé** (pas de triche !).

```
Matrice d'attention (causal mask):
     The  cat  sat
The  [1]  [0]  [0]   ← "The" ne voit que lui-même
cat  [1]  [1]  [0]   ← "cat" voit The + cat
sat  [1]  [1]  [1]   ← "sat" voit tout le passé
```

**Implémentation du masque** :
```python
def create_causal_mask(seq_len):
    """
    Crée un masque triangulaire inférieur.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # Inverser : True = peut voir, False = masqué

# Exemple
mask = create_causal_mask(5)
print(mask)
# tensor([[ True, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True, False, False],
#         [ True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True]])
```

**Usage** : Génération de texte (autoregressive).

---

## 4. Multi-Head Attention {#4-multi-head-attention}

### 4.1 Pourquoi Plusieurs Têtes ?

**Problème** : Une seule attention capture **un seul type de relation**.

**Exemple** :
- Tête 1 : Relations syntaxiques (sujet-verbe)
- Tête 2 : Relations sémantiques (coréférences)
- Tête 3 : Relations positionnelles (mots adjacents)

**Solution** : **Plusieurs têtes en parallèle** !

### 4.2 Architecture Multi-Head

```
Input (d_model=512)
    │
    ├──────────┬──────────┬──────────┐
    ▼          ▼          ▼          ▼
  Head 1     Head 2    Head 3    ... Head h
  (64D)      (64D)     (64D)         (64D)
    │          │         │            │
    └──────────┴─────────┴────────────┘
                  │
            Concatenate
                  │
              Linear(512)
                  │
               Output
```

**Formule** :
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

où head_i = Attention(Q×W_Q^i, K×W_K^i, V×W_V^i)
```

### 4.3 Implémentation PyTorch

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec h têtes parallèles.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model doit être divisible par num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension par tête

        # Projections linéaires pour Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Projection de sortie
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] (optionnel)

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size = Q.size(0)

        # 1. Projections linéaires
        Q = self.W_q(Q)  # [batch, seq_len, d_model]
        K = self.W_k(K)
        V = self.W_v(V)

        # 2. Reshape pour multi-head : (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Attention sur chaque tête
        if mask is not None:
            mask = mask.unsqueeze(1)  # Broadcast pour toutes les têtes

        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # attn_output: [batch, num_heads, seq_len, d_k]

        # 4. Concaténer les têtes
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch, seq_len, num_heads, d_k]

        attn_output = attn_output.view(batch_size, -1, self.d_model)
        # [batch, seq_len, d_model]

        # 5. Projection finale
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attn_weights


# Test
d_model, num_heads = 512, 8
batch_size, seq_len = 2, 10

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)

output, weights = mha(x, x, x)  # Self-attention : Q=K=V
print(f"Output shape: {output.shape}")  # [2, 10, 512]
print(f"Nombre de têtes: {num_heads}")
print(f"Dimension par tête: {d_model // num_heads}")  # 64
```

### 4.4 Visualisation des Têtes

**Phrase** : "The cat sat on the mat"

**Tête 1 (syntaxe)** :
```
     The  cat  sat  on  the  mat
The  0.1  0.2  0.1  0.1 0.4  0.1
cat  0.1  0.6  0.2  0.0 0.0  0.1  ← "cat" attend à "sat" (sujet→verbe)
sat  0.0  0.5  0.3  0.2 0.0  0.0
```

**Tête 2 (sémantique)** :
```
     The  cat  sat  on  the  mat
mat  0.0  0.4  0.0  0.5 0.0  0.1  ← "mat" attend à "cat" et "on" (relations sémantiques)
```

**Observation** : Chaque tête apprend **des patterns différents** !

### 💡 Analogie : L'Orchestre

- **Violons (Tête 1)** : Jouent la mélodie syntaxique
- **Contrebasses (Tête 2)** : Jouent les relations sémantiques profondes
- **Percussions (Tête 3)** : Marquent les positions et rythmes
- **Chef d'orchestre (Projection W_O)** : Harmonise tout

---

## 5. Position Encodings {#5-position-encodings}

### 5.1 Le Problème

**Attention is position-agnostic** : L'attention seule ne distingue pas :
- "The cat ate the mouse"
- "The mouse ate the cat"

Sans information de position, les deux phrases auraient les **mêmes représentations** !

### 5.2 Solution : Positional Encodings

**Idée** : Ajouter un vecteur de position à chaque embedding.

```
final_embedding = word_embedding + positional_encoding
```

### 5.3 Encodage Sinusoïdal (Original Transformer)

**Formule** :
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Où :
- `pos` : Position du token (0, 1, 2, ...)
- `i` : Indice de la dimension (0 à d_model/2)
- `2i, 2i+1` : Dimensions paires et impaires

**Propriétés** :
- ✅ Valeurs bornées [-1, 1]
- ✅ Déterministe (pas de paramètres à apprendre)
- ✅ Fonctionne pour séquences arbitrairement longues
- ✅ Relations linéaires : PE(pos+k) peut être exprimé comme fonction de PE(pos)

### 5.4 Implémentation

```python
class PositionalEncoding(nn.Module):
    """
    Positional Encoding sinusoïdal (Vaswani et al. 2017).
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Créer la matrice PE : [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # Appliquer sin aux indices pairs, cos aux impairs
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Ne sera pas entraîné

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Visualisation
import matplotlib.pyplot as plt

d_model = 512
pe = PositionalEncoding(d_model)

# Créer un input factice
dummy_input = torch.zeros(1, 100, d_model)
output = pe(dummy_input)

# Extraire les encodings
encodings = pe.pe[0, :100, :].numpy()

plt.figure(figsize=(15, 5))
plt.imshow(encodings.T, aspect='auto', cmap='RdBu')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.colorbar()
plt.title('Positional Encodings (sinusoïdal)')
plt.tight_layout()
# plt.savefig('positional_encodings.png')
```

### 5.5 Variantes Modernes

| Méthode | Description | Modèle |
|---------|-------------|--------|
| **Sinusoïdal** | Fixe, basé sur sin/cos | Transformer original |
| **Learned** | Paramètres appris | BERT, GPT-2 |
| **Relative** | Encodages relatifs (distance entre tokens) | T5, Transformer-XL |
| **ALiBi** | Biais d'attention basés sur distance | LLaMA, MPT |
| **RoPE** | Rotary Position Embeddings | LLaMA, GPT-NeoX |

#### RoPE (Rotary Position Embedding)

**Idée** : Faire tourner les vecteurs Q et K selon leur position.

```python
def apply_rotary_emb(x, position):
    """
    Applique RoPE (simplifié).
    """
    seq_len, d = x.shape
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    t = position.float()
    freqs = torch.outer(t, freqs)  # [seq_len, d/2]

    # Construire matrice de rotation
    cos_freqs = freqs.cos()
    sin_freqs = freqs.sin()

    # Rotation (application sur x)
    x_rot = torch.zeros_like(x)
    x_rot[:, 0::2] = x[:, 0::2] * cos_freqs - x[:, 1::2] * sin_freqs
    x_rot[:, 1::2] = x[:, 0::2] * sin_freqs + x[:, 1::2] * cos_freqs

    return x_rot
```

**Avantages** :
- Meilleure extrapolation à des séquences plus longues
- Conserve les distances relatives
- Utilisé dans LLaMA, GPT-NeoX

---

## 6. Feed-Forward Networks {#6-feed-forward}

### 6.1 Rôle du FFN

Après l'attention, chaque token passe par un **réseau feed-forward** :

```
FFN(x) = max(0, x×W_1 + b_1) × W_2 + b_2
       = ReLU(x×W_1 + b_1) × W_2 + b_2
```

**Structure** :
- Couche 1 : `d_model → d_ff` (expansion, typiquement d_ff = 4 × d_model)
- Activation : ReLU (ou GeLU)
- Couche 2 : `d_ff → d_model` (compression)

**Intuition** : Transformation non-linéaire appliquée **indépendamment** à chaque position.

### 6.2 Implémentation

```python
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Ou GeLU pour BERT/GPT

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        x = self.linear1(x)       # [batch, seq_len, d_ff]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)       # [batch, seq_len, d_model]
        x = self.dropout(x)
        return x


# Exemple
d_model, d_ff = 512, 2048
ffn = FeedForward(d_model, d_ff)

x = torch.randn(2, 10, d_model)
output = ffn(x)
print(f"Output shape: {output.shape}")  # [2, 10, 512]

# Nombre de paramètres
params = sum(p.numel() for p in ffn.parameters())
print(f"Paramètres FFN: {params:,}")  # ~2M params pour d_model=512
```

### 6.3 Variantes d'Activation

| Activation | Formule | Modèle |
|------------|---------|--------|
| **ReLU** | max(0, x) | Transformer original |
| **GeLU** | x × Φ(x) | BERT, GPT-2, GPT-3 |
| **SwiGLU** | Swish(xW) ⊙ (xV) | LLaMA, PaLM |
| **GeGLU** | GeLU(xW) ⊙ (xV) | GLaM |

**GeLU** (Gaussian Error Linear Unit) est devenu le standard :
```python
import torch.nn.functional as F

def gelu(x):
    """
    GeLU activation : approximation de x × Φ(x)
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

# Ou directement avec PyTorch
x = torch.randn(10)
y = F.gelu(x)
```

---

## 7. Layer Normalization & Residual Connections {#7-layer-norm}

### 7.1 Residual Connections (Skip Connections)

**Problème** : Réseaux profonds (96 couches pour GPT-3) souffrent de vanishing gradients.

**Solution** : Connexions résiduelles (He et al. 2016, ResNet)

```
output = x + SubLayer(x)
```

**Dans un Transformer** :
```
# Après attention
x = x + MultiHeadAttention(x)

# Après FFN
x = x + FeedForward(x)
```

**Avantage** : Les gradients peuvent "sauter" les couches → entraînement stable.

### 7.2 Layer Normalization

**Normalisation** : Stabilise l'entraînement en normalisant les activations.

**Formule** :
```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```

Où :
- `μ` : Moyenne sur la dimension des features
- `σ²` : Variance sur la dimension des features
- `γ, β` : Paramètres appris (scale & shift)
- `ε` : Petit terme pour stabilité numérique (1e-5)

**Différence avec Batch Norm** :
- **Batch Norm** : Normalise sur le batch (problématique pour NLP car séquences de longueurs variables)
- **Layer Norm** : Normalise sur les features (chaque exemple indépendamment)

### 7.3 Implémentation

```python
class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            normalized x
        """
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# Test
x = torch.randn(2, 10, 512)
ln = LayerNorm(512)
output = ln(x)

print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
# Output devrait avoir mean ≈ 0, std ≈ 1 (par dimension)
```

### 7.4 Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer)** :
```python
# Attention
x = x + MultiHeadAttention(x)
x = LayerNorm(x)

# FFN
x = x + FeedForward(x)
x = LayerNorm(x)
```

**Pre-Norm (Moderne, ex: GPT-2)** :
```python
# Attention
x = x + MultiHeadAttention(LayerNorm(x))

# FFN
x = x + FeedForward(LayerNorm(x))
```

**Avantages Pre-Norm** :
- ✅ Plus stable pour réseaux très profonds
- ✅ Peut entraîner sans learning rate warmup
- ❌ Performance légèrement inférieure parfois

**Modèles** :
- **Post-Norm** : BERT, Transformer original
- **Pre-Norm** : GPT-2, GPT-3, LLaMA

---

## 8. Les Trois Familles d'Architectures {#8-trois-familles}

### 8.1 Encoder-Only (BERT)

**Usage** : Compréhension de texte (classification, NER, QA)

**Architecture** :
```
[CLS] The cat sat [SEP]
   ↓    ↓   ↓   ↓    ↓
 ┌─────────────────────┐
 │ Bidirectional Attn  │ (tous les tokens se voient)
 └─────────────────────┘
         ↓
   Representations
```

**Caractéristiques** :
- ✅ Attention bidirectionnelle (voit futur)
- ✅ Excellent pour classification
- ❌ Ne peut pas générer de texte

**Exemples** : BERT, RoBERTa, ALBERT, DeBERTa

### 8.2 Decoder-Only (GPT)

**Usage** : Génération de texte (chat, complétion, etc.)

**Architecture** :
```
The cat sat
 ↓   ↓   ↓
┌──────────────┐
│ Causal Attn  │ (masque triangulaire)
└──────────────┘
      ↓
   Predict next token
```

**Caractéristiques** :
- ✅ Génération autoregressive
- ✅ Scaling excellent (GPT-3, GPT-4)
- ❌ Ne voit que le passé

**Exemples** : GPT, GPT-2, GPT-3, GPT-4, LLaMA, Claude

### 8.3 Encoder-Decoder (T5)

**Usage** : Seq2seq (traduction, résumé)

**Architecture** :
```
Encoder                Decoder
Source: "Hello"        Target: "Bonjour"
   ↓                      ↓
┌──────────┐          ┌──────────┐
│Bidir Attn│          │Causal+   │
│          │  ──────→ │Cross Attn│
└──────────┘          └──────────┘
```

**Caractéristiques** :
- ✅ Encoder : comprend l'input
- ✅ Decoder : génère l'output
- ✅ Cross-attention : Decoder attend sur Encoder
- ❌ Plus complexe, plus de paramètres

**Exemples** : T5, BART, mT5, Flan-T5

### 8.4 Comparaison

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder |
|--------|-------------|--------------|-----------------|
| **Attention** | Bidirectionnelle | Causale | Les deux |
| **Génération** | ❌ | ✅ | ✅ |
| **Compréhension** | ✅ | ⚠️ (via prompting) | ✅ |
| **Tâches** | Classification, NER | Chat, code, QA | Traduction, résumé |
| **Scaling** | Moyen | Excellent | Bon |
| **Exemples** | BERT | GPT, LLaMA | T5, BART |

---

## 9. Variantes Modernes {#9-variantes-modernes}

### 9.1 Optimisations d'Attention

#### A) Flash Attention

**Problème** : Attention classique est O(n²) en mémoire pour séquence de longueur n.

**Solution** : Flash Attention (Dao et al. 2022) utilise **tiling** pour réduire accès mémoire.

**Résultats** :
- 3-4× plus rapide
- Utilise jusqu'à 10× moins de mémoire
- **Exact** (pas d'approximation)

```python
# Utilisation avec PyTorch 2.0+
import torch.nn.functional as F

# Activer Flash Attention (si GPU compatible)
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
```

#### B) Sparse Attention

**Problème** : O(n²) interdit contextes ultra-longs.

**Solutions** :
- **Local Attention** : Chaque token attend seulement sur voisins proches
- **Strided Attention** : Attention sur 1 token sur k
- **Block Sparse Attention** : Patterns pré-définis

**Modèles** : Longformer, BigBird, Sparse Transformer

#### C) Linear Attention

**Idée** : Approximer attention en O(n) au lieu de O(n²).

**Méthodes** :
- **Performers** (Choromanski et al. 2021) : Kernel trick avec random features
- **Linformer** : Projections low-rank de K et V

**Compromis** : ✅ Rapide, ❌ Moins expressif

### 9.2 Alternatives Architecturales

| Modèle | Innovation | Avantage |
|--------|-----------|----------|
| **Transformer-XL** | Recurrence + relative position | Contextes ultra-longs |
| **Reformer** | LSH attention | Mémoire O(n log n) |
| **Synthesizer** | Apprendre patterns d'attention | Pas besoin de QK |
| **S4 (State Spaces)** | Remplacer attention par SSM | O(n) temps et mémoire |
| **Mamba** | SSM optimisés | Alternative aux Transformers |

### 9.3 LLaMA : Transformer Optimisé

**Innovations LLaMA (Meta, 2023)** :

1. **RoPE** : Rotary Position Embeddings
2. **SwiGLU** : Activation dans FFN
3. **RMSNorm** : Simplification de LayerNorm
4. **Pre-Norm** : Normalisation avant attention/FFN

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Utilisé dans LLaMA, T5.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        RMSNorm(x) = x / RMS(x) × γ
        où RMS(x) = √(mean(x²))
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
```

---

## 10. Implémentation from Scratch {#10-implementation}

### 10.1 Bloc Transformer Complet

```python
class TransformerBlock(nn.Module):
    """
    Un bloc Transformer complet (Multi-Head Attention + FFN + Layer Norm).
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] (optionnel)

        Returns:
            output: [batch, seq_len, d_model]
        """
        # Sub-layer 1: Multi-Head Attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Residual + Norm

        # Sub-layer 2: Feed-Forward
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # Residual + Norm

        return x


# Test
block = TransformerBlock(d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)
output = block(x)
print(f"Output shape: {output.shape}")  # [2, 10, 512]
```

### 10.2 Decoder GPT-Style Complet

```python
class GPTDecoder(nn.Module):
    """
    Decoder Transformer style GPT (causal, decoder-only).
    """
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6,
                 d_ff=2048, max_len=1024, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (embedding = lm_head)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: [batch, seq_len] (token IDs)
            mask: [batch, seq_len, seq_len] (causal mask)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        # [batch, seq_len, d_model]

        # 2. Positional encodings
        x = self.position_encoding(x)

        # 3. Créer causal mask si non fourni
        if mask is None:
            mask = create_causal_mask(seq_len).to(input_ids.device)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. Passer à travers les blocs Transformer
        for block in self.blocks:
            x = block(x, mask)

        # 5. Normalisation finale
        x = self.norm(x)

        # 6. Projection vers vocabulaire
        logits = self.lm_head(x)
        # [batch, seq_len, vocab_size]

        return logits


# Instanciation
vocab_size = 50000
model = GPTDecoder(vocab_size, d_model=512, num_heads=8, num_layers=6)

# Nombre total de paramètres
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~80M params

# Test forward pass
input_ids = torch.randint(0, vocab_size, (2, 20))  # Batch de 2, séquence de 20
logits = model(input_ids)
print(f"Logits shape: {logits.shape}")  # [2, 20, 50000]

# Génération du prochain token
next_token_logits = logits[:, -1, :]  # Dernier token
next_token = torch.argmax(next_token_logits, dim=-1)
print(f"Predicted next tokens: {next_token}")  # [token_id_1, token_id_2]
```

### 10.3 Génération Autoregressive

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
    """
    Génère du texte de manière autoregressive.

    Args:
        model: GPTDecoder
        input_ids: [batch, seq_len] - Prompt
        max_new_tokens: Nombre de tokens à générer
        temperature: Contrôle l'aléatoire (0 = déterministe, >1 = aléatoire)
        top_k: Échantillonner parmi les top-k tokens

    Returns:
        generated_ids: [batch, seq_len + max_new_tokens]
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Forward pass
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        # Prendre les logits du dernier token
        logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Softmax pour probabilités
        probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]

        # Échantillonner
        next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

        # Concaténer
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


# Exemple de génération
prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Token IDs du prompt
generated = generate(model, prompt, max_new_tokens=20, temperature=0.8, top_k=50)
print(f"Generated sequence: {generated}")
```

---

## 11. Quiz Interactif {#11-quiz}

### Question 1 : Complexité de l'Attention

**Quelle est la complexité computationnelle de l'attention standard pour une séquence de longueur n ?**

A) O(n)
B) O(n log n)
C) O(n²)
D) O(n³)

<details>
<summary>Voir la réponse</summary>

**Réponse : C) O(n²)**

Le calcul de `QK^T` produit une matrice de taille `[n, n]`, nécessitant O(n²) opérations. C'est le principal goulot d'étranglement des Transformers pour les longues séquences.

**Solutions** :
- Flash Attention (optimisation mémoire)
- Sparse Attention (attention limitée)
- Linear Attention (approximations)
</details>

---

### Question 2 : Multi-Head Attention

**Pourquoi utiliser 8 têtes d'attention au lieu d'une seule ?**

A) Pour paralléliser sur 8 GPUs
B) Pour capturer différents types de relations
C) Pour augmenter le nombre de paramètres
D) Pour accélérer l'entraînement

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Pour capturer différents types de relations**

Chaque tête apprend des patterns différents :
- Tête 1 : Relations syntaxiques (sujet-verbe)
- Tête 2 : Coréférences (it → cat)
- Tête 3 : Voisinage local
- etc.

**Nombre de paramètres** : Identique entre 1 tête de dimension 512 et 8 têtes de dimension 64 !
</details>

---

### Question 3 : Positional Encodings

**Que se passe-t-il si on omet les positional encodings ?**

A) Le modèle ne compile pas
B) "The cat ate the mouse" = "The mouse ate the cat"
C) L'entraînement est plus rapide
D) Rien, l'attention capture l'ordre

<details>
<summary>Voir la réponse</summary>

**Réponse : B) "The cat ate the mouse" = "The mouse ate the cat"**

L'attention est **invariante par permutation** : sans encodings de position, l'ordre des mots est ignoré. Les deux phrases auraient des représentations identiques !

**Solution** : Positional encodings (sinusoïdal, learned, RoPE, etc.)
</details>

---

### Question 4 : Causal Masking

**Dans GPT, pourquoi utilise-t-on un masque causal ?**

A) Pour accélérer le training
B) Pour empêcher de "tricher" en voyant le futur
C) Pour économiser de la mémoire
D) C'est une erreur historique

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Pour empêcher de "tricher" en voyant le futur**

En génération autoregressive, chaque token ne doit voir que le **passé**. Sinon, durant l'entraînement, le modèle "triche" en regardant le token qu'il doit prédire !

**Masque causal** : Triangle inférieur de 1s, reste à 0.
</details>

---

### Question 5 : Pre-Norm vs Post-Norm

**Quelle affirmation est vraie sur Pre-Norm ?**

A) Plus ancien que Post-Norm
B) Meilleur pour réseaux très profonds (>50 layers)
C) Toujours meilleure performance
D) Inventé pour BERT

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Meilleur pour réseaux très profonds (>50 layers)**

**Pre-Norm** (LayerNorm avant attention/FFN) stabilise l'entraînement pour modèles très profonds comme GPT-3 (96 layers).

**Trade-off** : Légèrement moins performant sur certaines tâches, mais entraînement plus stable et sans warmup.
</details>

---

### Question 6 : Encoder vs Decoder

**Quelle architecture pour une tâche de classification de sentiment ?**

A) Encoder-only (BERT)
B) Decoder-only (GPT)
C) Encoder-Decoder (T5)
D) Toutes équivalentes

<details>
<summary>Voir la réponse</summary>

**Réponse : A) Encoder-only (BERT)**

**Classification** = comprendre le texte (pas générer). L'encoder-only avec attention bidirectionnelle est optimal.

**GPT** peut aussi faire de la classification (via prompting), mais moins efficace.
</details>

---

## 12. Exercices Pratiques {#12-exercices}

### Exercice 1 : Visualiser les Attention Weights

**Objectif** : Créer une heatmap des poids d'attention pour comprendre ce que le modèle "regarde".

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """
    Visualise les poids d'attention sous forme de heatmap.

    Args:
        attention_weights: [seq_len, seq_len] (numpy array)
        tokens: Liste de tokens (strings)
    """
    # TODO: Créer une heatmap avec seaborn
    # Axes: tokens (source et target)
    # Couleur: intensité de l'attention
    pass

# Test
tokens = ["The", "cat", "sat", "on", "the", "mat"]
# Simuler des poids (en vrai, extraire depuis model)
weights = torch.softmax(torch.randn(6, 6), dim=-1).numpy()

visualize_attention(weights, tokens)
```

<details>
<summary>Voir la solution</summary>

```python
def visualize_attention(attention_weights, tokens):
    """
    Visualise les poids d'attention sous forme de heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.xlabel('Key (Source)')
    plt.ylabel('Query (Target)')
    plt.title('Attention Weights Heatmap')
    plt.tight_layout()
    plt.show()

# Exemple avec vraie attention
model = GPTDecoder(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
input_ids = torch.randint(0, 100, (1, 6))

# Hook pour extraire les poids
attention_weights_list = []

def hook_fn(module, input, output):
    # output[1] contient les attention weights
    attention_weights_list.append(output[1].detach())

# Enregistrer le hook sur la première tête
model.blocks[0].attention.attention.register_forward_hook(hook_fn)

# Forward
_ = model(input_ids)

# Extraire et visualiser
weights = attention_weights_list[0][0, 0].numpy()  # [seq_len, seq_len]
tokens = [f"T{i}" for i in range(6)]
visualize_attention(weights, tokens)
```
</details>

---

### Exercice 2 : Implémenter Learned Positional Embeddings

**Objectif** : Remplacer les encodings sinusoïdaux par des embeddings appris (comme dans GPT-2).

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Positional embeddings appris (paramètres entraînables).
    """
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        # TODO: Créer un Embedding de taille [max_len, d_model]
        # TODO: Ajouter dropout
        pass

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            x + positional embeddings
        """
        # TODO: Récupérer les embeddings pour positions 0..seq_len-1
        # TODO: Ajouter à x
        pass
```

<details>
<summary>Voir la solution</summary>

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Positional embeddings appris (GPT-2 style).
    """
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # [1, seq_len]

        # Embeddings positionnels
        pos_emb = self.position_embeddings(positions)
        # [1, seq_len, d_model]

        # Ajouter aux embeddings de tokens
        x = x + pos_emb
        return self.dropout(x)

# Test
x = torch.randn(2, 10, 512)
pos_emb = LearnedPositionalEmbedding(max_len=1024, d_model=512)
output = pos_emb(x)
print(f"Output shape: {output.shape}")  # [2, 10, 512]

# Nombre de paramètres
params = sum(p.numel() for p in pos_emb.parameters())
print(f"Paramètres: {params:,}")  # 1024 × 512 = 524,288
```
</details>

---

### Exercice 3 : Générer avec Different Sampling Strategies

**Objectif** : Implémenter greedy, top-k, top-p (nucleus), et temperature sampling.

```python
def sample_next_token(logits, strategy='greedy', temperature=1.0, top_k=None, top_p=None):
    """
    Échantillonne le prochain token selon différentes stratégies.

    Args:
        logits: [vocab_size] - Scores pour chaque token
        strategy: 'greedy', 'top_k', 'top_p', 'temperature'
        temperature: Contrôle l'aléatoire
        top_k: Nombre de tokens à considérer (top-k)
        top_p: Probabilité cumulative (nucleus sampling)

    Returns:
        next_token: Index du token échantillonné
    """
    # TODO: Implémenter les 4 stratégies
    pass
```

<details>
<summary>Voir la solution</summary>

```python
def sample_next_token(logits, strategy='greedy', temperature=1.0, top_k=None, top_p=None):
    """
    Échantillonne le prochain token.
    """
    if strategy == 'greedy':
        # Toujours prendre le token le plus probable
        return torch.argmax(logits).item()

    # Appliquer temperature
    logits = logits / temperature

    if strategy == 'temperature':
        # Échantillonnage selon distribution de probabilité
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    elif strategy == 'top_k':
        # Top-k sampling
        assert top_k is not None, "top_k doit être spécifié"
        v, indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, -float('Inf'))
        logits_filtered[indices] = v

        probs = F.softmax(logits_filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    elif strategy == 'top_p':
        # Nucleus (top-p) sampling
        assert top_p is not None, "top_p doit être spécifié"

        # Trier par probabilité décroissante
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Probabilité cumulative
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Retirer tokens dont cumsum > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Garder au moins 1 token
        sorted_indices_to_remove[0] = False

        # Créer masque
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits_filtered = logits.clone()
        logits_filtered[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits_filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

# Test
vocab_size = 50000
logits = torch.randn(vocab_size)

print("Greedy:", sample_next_token(logits, 'greedy'))
print("Temperature 0.7:", sample_next_token(logits, 'temperature', temperature=0.7))
print("Top-k (k=50):", sample_next_token(logits, 'top_k', top_k=50))
print("Top-p (p=0.9):", sample_next_token(logits, 'top_p', top_p=0.9))
```
</details>

---

## 13. Conclusion {#13-conclusion}

### 🎭 Dialogue Final : L'Élégance du Transformer

**Alice** : Après tout ça, je réalise que le Transformer est... étonnamment simple.

**Bob** : Exactement ! C'est son génie. Pas de magie, juste :
1. **Attention** : Regarder tous les mots simultanément
2. **Feed-Forward** : Transformer chaque mot indépendamment
3. **Residual + Norm** : Stabiliser l'entraînement
4. **Répéter N fois**

**Alice** : Et de ce pattern simple naissent GPT-4, Claude, Gemini...

**Bob** : Oui. Le Transformer est comme les échecs : **règles simples, complexité émergente**. En empilant ces blocs et en ajoutant des milliards de paramètres, on obtient des capacités qu'on ne comprend pas encore totalement.

**Alice** : Fascinant. Et terrifiant.

**Bob** : Bienvenue dans l'ère des LLMs. 🚀

### 🎯 Points Clés à Retenir

| Concept | Essence |
|---------|---------|
| **Self-Attention** | Q, K, V → softmax(QK^T/√d_k) × V |
| **Multi-Head** | Plusieurs attentions parallèles = patterns multiples |
| **Positional Encoding** | Sin/cos ou learned pour ordre des mots |
| **Feed-Forward** | Transformation non-linéaire par position |
| **Residual + Norm** | Stabilité pour réseaux profonds |
| **Encoder-only** | BERT = compréhension bidirectionnelle |
| **Decoder-only** | GPT = génération causale |
| **Encoder-Decoder** | T5 = seq2seq (traduction, résumé) |

### 📊 Architecture Parameters

**GPT-2 Small** (117M params) :
- 12 layers, 768 dim, 12 heads
- Context: 1024 tokens
- FFN: 3072 dim (4× expansion)

**GPT-3** (175B params) :
- 96 layers, 12288 dim, 96 heads
- Context: 2048 tokens
- FFN: 49152 dim

**Scaling Law** : Performances ∝ √Params (environ)

### 🚀 Prochaines Étapes

Maintenant que vous maîtrisez l'architecture Transformer :

1. **Chapitre 7 : Fine-Tuning** → Adapter un Transformer pré-entraîné
2. **Chapitre 10 : Optimization** → Flash Attention, quantization, etc.
3. **Chapitre 13 : LoRA** → Fine-tuning efficient

---

## 14. Ressources {#14-ressources}

### 📚 Papers Fondamentaux

1. **"Attention is All You Need"** (Vaswani et al., 2017)
   - Le paper original des Transformers

2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
   - Encoder-only, masked language modeling

3. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018)
   - GPT-1, decoder-only

4. **"Language Models are Unsupervised Multitask Learners"** (GPT-2, Radford et al., 2019)

5. **"Language Models are Few-Shot Learners"** (GPT-3, Brown et al., 2020)

6. **"Flash Attention: Fast and Memory-Efficient Exact Attention"** (Dao et al., 2022)

7. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)

### 🛠️ Implémentations de Référence

```bash
# Transformers from scratch (didactique)
https://github.com/karpathy/minGPT
https://github.com/hyunwoongko/transformer

# Production (HuggingFace)
pip install transformers torch

# Optimisations
pip install flash-attn  # Flash Attention
pip install xformers     # Optimized attention variants
```

### 🔗 Tutoriels et Visualisations

- **The Illustrated Transformer** : https://jalammar.github.io/illustrated-transformer/
- **Attention Visualizer** : https://github.com/jessevig/bertviz
- **Tensor2Tensor** (Google) : https://github.com/tensorflow/tensor2tensor

---

**🎓 Bravo !** Vous comprenez maintenant les Transformers de l'intérieur. Dans le prochain chapitre, nous explorerons comment **tokenizer** le texte avant de le passer au Transformer ! 🚀

