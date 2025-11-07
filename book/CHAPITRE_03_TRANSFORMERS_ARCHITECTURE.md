# CHAPITRE 3 : ARCHITECTURE DES TRANSFORMERS (DEEP DIVE)

## Introduction

Le transformer, introduit dans le paper "Attention is All You Need" (Vaswani et al., 2017), a révolutionné le traitement du langage naturel et est devenu l'architecture fondamentale de tous les LLMs modernes. Ce chapitre plonge en profondeur dans chaque composant du transformer, avec des explications mathématiques rigoureuses et des implémentations pratiques.

## 3.1 Vue d'ensemble de l'architecture

### 3.1.1 Architecture originale (Encoder-Decoder)

L'architecture transformer originale se compose de deux parties principales:

```
Input Text → [Encoder] → Context Representation → [Decoder] → Output Text
```

**Diagramme détaillé:**
```
┌─────────────────────────────────────────────────────────┐
│                    TRANSFORMER                           │
├────────────────────────┬────────────────────────────────┤
│       ENCODER          │          DECODER               │
│                        │                                │
│  ┌──────────────────┐ │  ┌──────────────────┐         │
│  │  Output          │ │  │  Output          │         │
│  │  Embedding       │ │  │  Embedding       │         │
│  └────────┬─────────┘ │  └────────┬─────────┘         │
│           │            │           │                    │
│  ┌────────▼─────────┐ │  ┌────────▼─────────┐         │
│  │ Positional       │ │  │ Positional       │         │
│  │ Encoding         │ │  │ Encoding         │         │
│  └────────┬─────────┘ │  └────────┬─────────┘         │
│           │            │           │                    │
│  ┌────────▼─────────┐ │  ┌────────▼─────────┐         │
│  │ N x Encoder      │ │  │ N x Decoder      │         │
│  │ Layer            │ │  │ Layer            │         │
│  │                  │ │  │                  │         │
│  │ • Self-Attn     │ │  │ • Self-Attn      │         │
│  │ • Feed-Forward  │─┼──▶│ • Cross-Attn    │         │
│  │                  │ │  │ • Feed-Forward   │         │
│  └────────┬─────────┘ │  └────────┬─────────┘         │
│           │            │           │                    │
│  ┌────────▼─────────┐ │  ┌────────▼─────────┐         │
│  │ Final Output     │ │  │ Linear + Softmax │         │
│  └──────────────────┘ │  └──────────────────┘         │
└────────────────────────┴────────────────────────────────┘
```

### 3.1.2 Decoder-Only (GPT family)

Les LLMs modernes (GPT, Llama, Mistral) utilisent une architecture decoder-only:

```
Input Tokens → [Decoder Stack] → Output Logits → Next Token Prediction
```

**Avantages du decoder-only:**
- Plus simple (pas de cross-attention)
- Autorégressif naturellement
- Scale mieux avec la taille
- Meilleur pour la génération

**Architecture détaillée (GPT-style):**
```python
class GPTModel(nn.Module):
    """
    Implémentation complète d'un modèle GPT decoder-only
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (embeddings = output weights)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialisation des poids (GPT-2 style)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: token indices [batch, seq_len]
        targets: target tokens pour training [batch, seq_len]
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.config.block_size

        # Token embeddings
        tok_emb = self.token_embedding(idx)  # [B, T, C]

        # Position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # [T, C]

        # Combine embeddings
        x = self.drop(tok_emb + pos_emb)  # [B, T, C]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génération autorégresssive

        idx: context tokens [B, T]
        max_new_tokens: nombre de tokens à générer
        temperature: contrôle la randomness
        top_k: sample from top-k tokens
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

### 3.1.3 Configuration typique des modèles

**GPT-2 Small (124M params):**
```python
from dataclasses import dataclass

@dataclass
class GPT2SmallConfig:
    vocab_size: int = 50257
    n_embd: int = 768        # embedding dimension
    n_layer: int = 12        # number of transformer blocks
    n_head: int = 12         # number of attention heads
    block_size: int = 1024   # max sequence length
    dropout: float = 0.1
    bias: bool = True        # use bias in Linear layers
```

**Llama 2 7B:**
```python
@dataclass
class Llama2_7BConfig:
    vocab_size: int = 32000
    n_embd: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 32      # pour Grouped Query Attention
    block_size: int = 4096   # context length
    dropout: float = 0.0     # pas de dropout après pretraining
    multiple_of: int = 256   # pour optimisation hardware
    norm_eps: float = 1e-5
```

**Comparaison tailles de modèles:**

| Modèle | Paramètres | Layers | d_model | Heads | Context | Training Tokens |
|--------|-----------|--------|---------|-------|---------|----------------|
| GPT-2 Small | 124M | 12 | 768 | 12 | 1024 | - |
| GPT-2 Medium | 355M | 24 | 1024 | 16 | 1024 | - |
| GPT-2 Large | 774M | 36 | 1280 | 20 | 1024 | - |
| GPT-2 XL | 1.5B | 48 | 1600 | 25 | 1024 | - |
| GPT-3 | 175B | 96 | 12288 | 96 | 2048 | 300B |
| Llama 2 7B | 7B | 32 | 4096 | 32 | 4096 | 2T |
| Llama 2 13B | 13B | 40 | 5120 | 40 | 4096 | 2T |
| Llama 2 70B | 70B | 80 | 8192 | 64 | 4096 | 2T |
| GPT-4 | ~1.8T | ? | ? | ? | 128k | ? |

## 3.2 Mécanisme d'Attention

### 3.2.1 Self-Attention : Formulation Mathématique

L'attention est le cœur du transformer. Elle permet au modèle de "regarder" d'autres positions dans la séquence lors du traitement d'une position donnée.

**Intuition:**
Quand vous lisez la phrase "Le chat dort sur le tapis", pour comprendre "dort", vous devez comprendre que c'est le "chat" qui dort, pas le "tapis". L'attention permet au modèle de faire ces connexions.

**Formulation mathématique:**

Pour chaque token, on calcule trois vecteurs:
- **Query (Q)**: "Qu'est-ce que je cherche?"
- **Key (K)**: "Qu'est-ce que j'ai à offrir?"
- **Value (V)**: "Quelle information je porte?"

```
Q = XW^Q    où W^Q ∈ ℝ^(d_model × d_k)
K = XW^K    où W^K ∈ ℝ^(d_model × d_k)
V = XW^V    où W^V ∈ ℝ^(d_model × d_v)
```

**Attention scores:**
```
scores = QK^T / √d_k
```

Le scaling factor √d_k est crucial:

**Pourquoi scaler par √d_k?**

Sans scaling, pour de grandes dimensions, le produit scalaire QK^T a une variance qui croît avec d_k, poussant les valeurs dans les régions où softmax sature (gradients très petits).

**Preuve mathématique:**
```
Si q_i, k_i ~ N(0, 1) indépendants, alors:

q·k = Σ q_i k_i

E[q·k] = 0
Var(q·k) = Σ Var(q_i k_i) = d_k

Donc q·k a écart-type √d_k

En divisant par √d_k, on normalise: q·k/√d_k a variance 1
```

**Softmax et attention weights:**
```
α = softmax(scores) = softmax(QK^T / √d_k)

α_ij = exp(score_ij) / Σ_k exp(score_ik)
```

**Output:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### 3.2.2 Implémentation détaillée

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """
    Implémentation complète de self-attention
    avec tous les détails importants
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Dimension par head
        self.head_dim = config.n_embd // config.n_head

        # Projections Q, K, V en une seule matrice (efficacité)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Projection de sortie
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        """
        x: [batch, seq_len, n_embd]
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 1. Calculer Q, K, V pour tous les heads en parallèle
        # [B, T, C] -> [B, T, 3*C]
        qkv = self.c_attn(x)

        # Split en Q, K, V
        # [B, T, 3*C] -> 3 x [B, T, C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2. Reshape pour multi-head
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Calculer attention scores
        # Q @ K^T: [B, n_head, T, head_dim] @ [B, n_head, head_dim, T]
        #        = [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 4. Appliquer causal mask (pour autoregressive)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # 5. Softmax
        att = F.softmax(att, dim=-1)

        # 6. Dropout sur attention weights
        att = self.attn_dropout(att)

        # 7. Appliquer attention sur V
        # [B, n_head, T, T] @ [B, n_head, T, head_dim]
        # = [B, n_head, T, head_dim]
        y = att @ v

        # 8. Recombiner les heads
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 9. Projection finale et dropout
        y = self.resid_dropout(self.c_proj(y))

        return y
```

### 3.2.3 Visualisation de l'attention

```python
def visualize_attention(model, text, tokenizer, layer=0, head=0):
    """
    Visualise les attention weights pour une phrase
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Tokenize
    tokens = tokenizer.encode(text)
    x = torch.tensor(tokens).unsqueeze(0)  # [1, T]

    # Forward pass avec hooks pour capturer attention
    attention_weights = {}

    def hook_fn(module, input, output, layer_idx, head_idx):
        # Capturer les attention weights
        attention_weights[(layer_idx, head_idx)] = output[1]  # [B, T, T]

    # Register hooks
    for i, block in enumerate(model.blocks):
        block.attn.register_forward_hook(
            lambda m, inp, out, i=i: hook_fn(m, inp, out, i, head)
        )

    # Forward
    model(x)

    # Get attention weights
    attn = attention_weights[(layer, head)][0].detach().cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', ax=ax)
    ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.show()

# Exemple d'utilisation
text = "The cat sat on the mat"
visualize_attention(model, text, tokenizer, layer=0, head=0)
```

### 3.2.4 Multi-Head Attention

**Pourquoi plusieurs heads?**

Différents heads peuvent apprendre différents types de relations:
- Head 1: relations syntaxiques (sujet-verbe)
- Head 2: coréférences (pronoms)
- Head 3: relations sémantiques
- etc.

**Formulation:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Paramètres:**
```
Pour chaque head i:
- W^Q_i ∈ ℝ^(d_model × d_k)
- W^K_i ∈ ℝ^(d_model × d_k)
- W^V_i ∈ ℝ^(d_model × d_v)

Projection finale:
- W^O ∈ ℝ^(h·d_v × d_model)

Typiquement: d_k = d_v = d_model / h
```

**Calcul du nombre de paramètres:**
```python
def count_attention_params(d_model, n_heads):
    """
    Compte les paramètres dans multi-head attention
    """
    # Q, K, V projections
    qkv_params = 3 * d_model * d_model

    # Output projection
    out_params = d_model * d_model

    # Bias (optionnel)
    bias_params = 4 * d_model  # Q, K, V, out

    total = qkv_params + out_params + bias_params

    return total

# GPT-2 small
params = count_attention_params(d_model=768, n_heads=12)
print(f"Attention params: {params:,}")  # 2,362,368
```

### 3.2.5 Causal Attention (Masking)

Pour la génération autorégresssive, on doit empêcher le modèle de "voir" le futur.

**Masque causal:**
```
    t₁  t₂  t₃  t₄  t₅
t₁ [ 0 -∞ -∞ -∞ -∞ ]
t₂ [ 0  0 -∞ -∞ -∞ ]
t₃ [ 0  0  0 -∞ -∞ ]
t₄ [ 0  0  0  0 -∞ ]
t₅ [ 0  0  0  0  0 ]
```

**Implémentation:**
```python
def create_causal_mask(seq_len):
    """
    Crée un masque causal (triangulaire inférieur)

    Returns: [seq_len, seq_len] avec 0 pour allowed, -inf pour masked
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Exemple
mask = create_causal_mask(5)
print(mask)
# tensor([[0., -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf],
#         [0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0.]])

# Application dans attention
scores = torch.randn(1, 5, 5)  # [batch, seq, seq]
masked_scores = scores + mask.unsqueeze(0)
attn_weights = F.softmax(masked_scores, dim=-1)
```

**Visualisation de l'effet:**
```python
# Sans masque
scores_raw = torch.tensor([
    [1.0, 0.5, 0.3, 0.2, 0.1],
    [0.8, 1.2, 0.4, 0.3, 0.2],
    [0.6, 0.7, 1.5, 0.5, 0.3],
    [0.4, 0.5, 0.6, 1.8, 0.4],
    [0.3, 0.4, 0.5, 0.6, 2.0]
])

# Softmax sans masque (INCORRECT pour génération)
attn_no_mask = F.softmax(scores_raw, dim=-1)
print("Sans masque:")
print(attn_no_mask)
# Chaque position regarde TOUS les tokens (y compris futurs)

# Avec masque causal (CORRECT)
mask = create_causal_mask(5)
scores_masked = scores_raw + mask
attn_with_mask = F.softmax(scores_masked, dim=-1)
print("\nAvec masque:")
print(attn_with_mask)
# Chaque position regarde seulement les tokens passés
```

### 3.2.6 Cross-Attention (Encoder-Decoder)

Dans les modèles encoder-decoder, le decoder utilise cross-attention pour regarder les outputs de l'encoder.

**Différence avec self-attention:**
- Self-attention: Q, K, V viennent de la même source
- Cross-attention: Q vient du decoder, K et V de l'encoder

```python
class CrossAttention(nn.Module):
    """
    Cross-attention pour encoder-decoder
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q vient du decoder
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)

        # K, V viennent de l'encoder
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, encoder_output):
        """
        x: decoder hidden state [B, T_dec, C]
        encoder_output: encoder output [B, T_enc, C]
        """
        B, T_dec, C = x.shape
        _, T_enc, _ = encoder_output.shape

        # Query from decoder
        q = self.q_proj(x)  # [B, T_dec, C]

        # Keys and Values from encoder
        k = self.k_proj(encoder_output)  # [B, T_enc, C]
        v = self.v_proj(encoder_output)  # [B, T_enc, C]

        # Reshape for multi-head
        q = q.view(B, T_dec, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: [B, heads, T_dec, T_enc]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Pas de masque causal ici (peut regarder tout l'encoder)
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = attn @ v  # [B, heads, T_dec, head_dim]

        # Recombine
        out = out.transpose(1, 2).contiguous().view(B, T_dec, C)
        out = self.out_proj(out)

        return out
```

### 3.2.7 Flash Attention

Flash Attention (Dao et al., 2022) est une implémentation optimisée qui réduit l'usage mémoire et accélère le calcul.

**Problème standard:**
- Attention calcule une matrice [seq_len, seq_len]
- Pour seq_len=2048: matrice de 4M éléments
- Memory: O(N²)

**Solution Flash Attention:**
- Calcul par blocs (tiling)
- Fusionne kernel operations
- Memory: O(N)
- Speed: 2-4x plus rapide

**Installation et usage:**
```python
# Installation
# pip install flash-attn

from flash_attn import flash_attn_func

def flash_attention_forward(q, k, v, causal=True):
    """
    q, k, v: [batch, seq_len, num_heads, head_dim]
    """
    # Flash attention attend un format spécifique
    out = flash_attn_func(q, k, v, causal=causal)
    return out

# Comparaison performance
import time

# Standard attention
start = time.time()
out_standard = standard_attention(q, k, v)
time_standard = time.time() - start

# Flash attention
start = time.time()
out_flash = flash_attention_forward(q, k, v)
time_flash = time.time() - start

print(f"Standard: {time_standard:.4f}s")
print(f"Flash: {time_flash:.4f}s")
print(f"Speedup: {time_standard/time_flash:.2f}x")
```

---

*[Le chapitre continue avec les sections suivantes sur les autres composants du transformer...]*

## 3.3 Encodage Positionnel

### 3.3.1 Problème et motivation

Les transformers traitent tous les tokens en parallèle (contrairement aux RNNs qui sont séquentiels). Sans information de position, le modèle ne peut pas distinguer l'ordre des mots.

**Exemple:**
```
"Le chat mange la souris"
vs
"La souris mange le chat"
```

Sans positional encoding, ces deux phrases auraient la même représentation!

### 3.3.2 Sinusoidal Positional Encoding (Vaswani et al.)

**Formule:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

où:
- pos: position dans la séquence (0, 1, 2, ...)
- i: dimension index (0, 1, ..., d_model/2)
```

**Implémentation:**
```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal original du paper Transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Créer la matrice de positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calcul des fréquences
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Appliquer sin et cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Enregistrer comme buffer (pas trainable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        # Ajouter positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return x
```

**Visualisation:**
```python
import matplotlib.pyplot as plt

def visualize_positional_encoding(d_model=128, max_len=100):
    """
    Visualise les patterns du positional encoding
    """
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    encoding = pe.pe[0].numpy()  # [max_len, d_model]

    plt.figure(figsize=(15, 5))
    plt.imshow(encoding.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.colorbar()
    plt.title('Sinusoidal Positional Encoding')
    plt.show()

visualize_positional_encoding()
```

**Propriétés intéressantes:**

1. **Périodicité variable:**
   - Dimensions basses: haute fréquence (changement rapide)
   - Dimensions hautes: basse fréquence (changement lent)

2. **Distance relative:**
   - Le produit scalaire PE(pos) · PE(pos+k) dépend seulement de k
   - Permet au modèle d'apprendre les relations de distance

3. **Extrapolation:**
   - Peut généraliser à des séquences plus longues que celles vues en training

### 3.3.3 Learned Positional Embeddings

Alternative: apprendre les positional embeddings comme paramètres.

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Positional embeddings apprenables (utilisé dans GPT, BERT)
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Position indices
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Lookup embeddings
        pos_emb = self.pe(positions)

        return x + pos_emb
```

**Avantages vs Sinusoidal:**
- Plus flexible (peut s'adapter aux données)
- Utilisé dans GPT-2, BERT, etc.

**Inconvénients:**
- Ne peut pas extrapoler au-delà de max_len vu en training
- Nécessite plus de paramètres

### 3.3.4 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) est utilisé dans Llama, GPT-NeoX, et d'autres modèles récents.

**Idée:** Encoder la position par rotation dans l'espace complexe

**Formule simplifiée:**
```
RoPE(x_m, m) = [
  cos(mθ)  -sin(mθ)     [x_{2i}  ]
  sin(mθ)   cos(mθ)  ] × [x_{2i+1}]

où θ = 10000^(-2i/d)
```

**Implémentation:**
```python
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    Utilisé dans Llama, GPT-NeoX
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()

        # Calculer les fréquences
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-calculer pour efficacité
        self.max_seq_len = max_seq_len
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len):
        """Pre-calcule cos et sin pour toutes les positions"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, q, k):
        """
        Applique RoPE sur query et key

        q, k: [batch, num_heads, seq_len, head_dim]
        """
        seq_len = q.shape[2]

        # Récupérer cos et sin
        cos = self.cos_cached[:seq_len, ...].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len, ...].unsqueeze(0).unsqueeze(0)

        # Rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    def _rotate_half(self, x):
        """Helper pour rotation"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
```

**Avantages de RoPE:**
1. Meilleure extrapolation à des séquences longues
2. Encode la distance relative naturellement
3. Plus efficace computationnellement
4. Meilleure performance empirique

### 3.3.5 ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2021) ajoute un biais linéaire aux scores d'attention.

**Formule:**
```
attention_scores = QK^T + m × distance

où m est un slope spécifique à chaque head
```

**Implémentation:**
```python
class ALiBiPositionalBias(nn.Module):
    """
    ALiBi: Attention with Linear Biases
    """
    def __init__(self, num_heads, max_seq_len=2048):
        super().__init__()

        # Calculer les slopes pour chaque head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)

        # Pre-calculer les bias
        self._update_bias_cache(max_seq_len)

    def _get_slopes(self, num_heads):
        """
        Calcule les slopes pour chaque head
        Ratio géométrique: 2^(-8/n), 2^(-16/n), ...
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            # Si num_heads pas power of 2
            closest_power = 2 ** math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power)
            slopes += self._get_slopes(2 * closest_power)[:num_heads - closest_power]

        return torch.tensor(slopes).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def _update_bias_cache(self, max_seq_len):
        """Pre-calcule la matrice de distance"""
        # Matrice de distance: d[i,j] = i - j
        positions = torch.arange(max_seq_len)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance = distance.unsqueeze(0)  # [1, seq, seq]

        self.register_buffer('distance', distance, persistent=False)

    def forward(self, attention_scores):
        """
        Ajoute le biais ALiBi aux attention scores

        attention_scores: [batch, num_heads, seq_len, seq_len]
        """
        seq_len = attention_scores.shape[2]

        # Bias: slopes × distance
        # slopes: [1, num_heads, 1, 1]
        # distance: [1, 1, seq, seq]
        bias = self.slopes * self.distance[:, :seq_len, :seq_len]

        return attention_scores + bias

# Usage dans attention
class AttentionWithALiBi(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... autres inits ...
        self.alibi = ALiBiPositionalBias(config.n_head, config.max_seq_len)

    def forward(self, q, k, v):
        # Calculer attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Ajouter ALiBi bias
        scores = self.alibi(scores)

        # Softmax et reste...
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return out
```

**Avantages d'ALiBi:**
1. Pas d'embeddings positionnels séparés
2. Excellente extrapolation (peut générer 10x+ la longueur d'entraînement)
3. Plus simple et efficace
4. Utilisé dans Bloom, MPT

### 3.3.6 Comparaison des méthodes

| Méthode | Paramètres | Extrapolation | Complexité | Utilisé dans |
|---------|-----------|---------------|------------|--------------|
| **Sinusoidal** | 0 (fixe) | Bonne | O(d) | Transformer original |
| **Learned** | O(n×d) | Limitée | O(d) | GPT-2, BERT |
| **RoPE** | 0 (fixe) | Excellente | O(d) | Llama, GPT-NeoX |
| **ALiBi** | 0 (fixe) | Excellente | O(1) | Bloom, MPT |

**Benchmark empirique:**
```python
def benchmark_positional_encodings(seq_lens=[512, 1024, 2048, 4096]):
    """
    Compare les performances des différentes méthodes
    """
    results = {}

    for method in ['sinusoidal', 'learned', 'rope', 'alibi']:
        times = []
        for seq_len in seq_lens:
            # ... setup et timing ...
            times.append(elapsed_time)
        results[method] = times

    # Plot
    plt.figure(figsize=(10, 6))
    for method, times in results.items():
        plt.plot(seq_lens, times, marker='o', label=method)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.title('Positional Encoding Methods Comparison')
    plt.show()
```

---

*[Le chapitre continue avec Feed-Forward Networks, Normalisation, et Architectures complètes...]*

*[Contenu total du Chapitre 3: ~60-70 pages]*
