# Partie 4 : Architectures de LLM modernes

## Objectifs d'apprentissage

- Maîtriser l'architecture Transformer (attention, positions, normalisation)
- Comprendre les optimisations (FlashAttention, sparse attention, long-context)
- Analyser les Mixture of Experts (MoE) et leurs trade-offs
- Découvrir les architectures post-Transformer (SSM/Mamba, hybrides)
- Explorer les modèles multimodaux

## Prérequis

- Parties 2 et 3 validées
- Compréhension des mécanismes d'attention
- PyTorch avancé

---

## 4.1 Architecture Transformer : fondations

### 4.1.1 Vue d'ensemble

**Paper originel** : "Attention is All You Need" (Vaswani et al., 2017)

**Principe** : Remplacer la récurrence par l'attention pour capturer les dépendances.

**Architecture globale** :

```
Input → Embedding → Positional Encoding
                          ↓
                   [Transformer Blocks] × N
                          ↓
                    Layer Norm → Linear → Softmax
                          ↓
                       Output
```

**Transformer Block** :

```
x → LayerNorm → Multi-Head Attention → + (résiduelle)
                                        ↓
                                  LayerNorm → Feed-Forward → + (résiduelle)
```

### 4.1.2 Multi-Head Attention (MHA)

**Formule de base** :

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

où :
- Q (Query), K (Key), V (Value) sont des projections linéaires de l'input
- d_k est la dimension des clés (pour scaling)

**Multi-Head** :

```
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
```

**Implémentation** :

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Projections linéaires
        Q = self.W_q(query)  # [batch, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape pour multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: [batch, num_heads, seq_len, d_k]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Masking (pour causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Appliquer attention aux valeurs
        attn_output = torch.matmul(attn_weights, V)

        # Concatener les heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Projection de sortie
        output = self.W_o(attn_output)
        return output
```

**Pourquoi multi-head ?**
- Capture différents types de relations (syntaxe, sémantique, long/court range)
- Augmente la capacité du modèle
- Parallélisation efficace

### 4.1.3 Encodages positionnels

**Problème** : L'attention est permutation-invariante → besoin d'encoder la position.

**Sinusoïdal (Vaswani et al.)** :

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]
```

**Learned positional embeddings** :

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
```

**RoPE (Rotary Position Embedding)** (LLaMA, GPT-NeoX) :

Applique une rotation dans l'espace complexe.

```python
# Implémentation simplifiée
def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

**ALiBi (Attention with Linear Biases)** (BLOOM) :

Ajoute un biais linéaire aux scores d'attention.

```
scores += -m × |i - j|
```

où m est un scalaire par head.

**Comparaison** :

| Méthode   | Avantages                          | Inconvénients           |
|-----------|------------------------------------|-------------------------|
| Sinusoïdal| Pas de paramètres, extrapolation   | Moins flexible          |
| Learned   | Adaptatif                          | Fixé à max_len          |
| RoPE      | Relative, bonne extrapolation      | Plus complexe           |
| ALiBi     | Excellente extrapolation           | Nécessite ajustement m  |

### 4.1.4 Feed-Forward Network (FFN)

**Architecture** :

```
FFN(x) = GELU(xW_1 + b_1) W_2 + b_2
```

Typiquement : d_model → 4×d_model → d_model

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(x))))
```

### 4.1.5 Layer Normalization et résidus

**Pre-LN vs Post-LN** :

**Post-LN** (Transformer original) :

```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

**Pre-LN** (GPT-2, GPT-3, LLaMA) :

```
x = LN(x + Attention(x))
x = LN(x + FFN(x))
```

**Avantage Pre-LN** : Plus stable pour l'entraînement de modèles profonds.

### 4.1.6 Transformer Block complet

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN
        attn_out = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
```

---

## 4.2 Optimisations d'efficacité

### 4.2.1 FlashAttention

**Problème** : L'attention standard a une complexité O(n²) en mémoire et compute.

**Solution FlashAttention** (Dao et al., 2022) :
- Réorganise les calculs pour minimiser les accès mémoire
- Fusionne les opérations kernel
- Réduit l'utilisation mémoire de O(n²) à O(n)

**Implémentation** :

```python
# Utiliser FlashAttention via HuggingFace
from transformers.models.llama.modeling_llama import LlamaAttention

# Automatiquement utilisé si disponible
config.attn_implementation = "flash_attention_2"
```

**Gains** :
- 2-4× plus rapide que l'attention standard
- Supporte des contextes plus longs avec la même mémoire

### 4.2.2 Sparse Attention

**Motivation** : Tous les tokens n'ont pas besoin d'attendre à tous les autres.

**Patterns courants** :

1. **Local windowed attention** : Chaque token attends seulement à une fenêtre locale

```
Mask: diagonale + fenêtre de taille w
```

2. **Strided attention** : Attention tous les k tokens

3. **Global + local** : Quelques tokens globaux + attention locale

**Exemple avec Longformer** :

```python
from transformers import LongformerModel

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# Supporte jusqu'à 4096 tokens avec attention sparse
```

### 4.2.3 Long-context strategies

**Approches** :

1. **Extrapolation** (RoPE, ALiBi) : Entraîner sur contexte court, inférer sur long

2. **Sparse attention** : Réduire la complexité

3. **Recurrent patterns** : Combiner attention et récurrence (Transformer-XL)

4. **Hierarchical** : Multi-résolution (attention locale puis globale)

**Exemple : Attention sliding window** (Mistral 7B) :

```python
# Chaque token attends seulement aux w tokens précédents
attention_window = 4096
```

---

## 4.3 Mixture of Experts (MoE)

### 4.3.1 Principe

**Idée** : Au lieu d'une seule FFN, avoir plusieurs FFN (experts) et router chaque token vers un sous-ensemble d'experts.

**Architecture** :

```
Router: x → softmax(xW_r) → top-k experts
Output: Σ_i gate_i × Expert_i(x)
```

**Exemple** :

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()

        # Router scores
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)

        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

        # Appliquer experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # [batch, seq_len]
            expert_weight = top_k_probs[..., i].unsqueeze(-1)  # [batch, seq_len, 1]

            # Batch tous les tokens pour chaque expert
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        return output
```

### 4.3.2 Avantages et trade-offs

**Avantages** :
- Augmente la capacité sans augmenter proportionnellement le compute
- Spécialisation des experts (langues, domaines, tâches)

**Coûts** :
- Complexité d'implémentation
- Déséquilibre de charge (load balancing)
- Mémoire totale élevée (tous les experts stockés)

**Exemples de modèles MoE** :
- Mixtral 8×7B (8 experts, 2 actifs → ~13B params actifs, 47B total)
- GPT-4 (rumeur : MoE massif)
- Switch Transformer (Google, jusqu'à 1.6T params)

### 4.3.3 Load balancing

**Problème** : Le router peut favoriser quelques experts, laissant les autres inutilisés.

**Solutions** :

1. **Auxiliary loss** : Pénaliser le déséquilibre

```python
def load_balancing_loss(router_probs, expert_indices):
    # Encourage une distribution uniforme
    expert_usage = torch.zeros(num_experts)
    for idx in expert_indices.flatten():
        expert_usage[idx] += 1
    return expert_usage.var()  # minimiser la variance
```

2. **Capacity factor** : Limiter le nombre de tokens par expert

3. **Expert dropout** : Désactiver aléatoirement des experts pendant l'entraînement

---

## 4.4 Architectures post-Transformer

### 4.4.1 Motivations

**Limites du Transformer** :
- Complexité quadratique en contexte (O(n²))
- KV cache croît linéairement avec le contexte
- Difficultés sur très longs contextes (>100k tokens)

**Objectifs des alternatives** :
- Complexité linéaire ou sous-quadratique
- Efficacité mémoire
- Vitesse d'inférence

### 4.4.2 State Space Models (SSM)

**Principe** : Modéliser les séquences comme des systèmes dynamiques continus.

**Équations** :

```
x'(t) = Ax(t) + Bu(t)  # état latent
y(t) = Cx(t) + Du(t)   # sortie
```

Discrétisé pour le traitement de séquences.

**Avantages** :
- Complexité linéaire O(n)
- Parallélisable pendant l'entraînement
- Récurrent pendant l'inférence (efficacité mémoire)

### 4.4.3 Mamba

**Paper** : Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

**Innovation** : SSM avec mécanisme de sélection (gates) pour décider quelles informations retenir.

**Architecture** :

```
Selective SSM:
  - Paramètres dépendants de l'input (B, C, Δ)
  - Mécanisme de gating pour filtrer l'information

Mamba block:
  x → [Projection] → [SSM sélectif] → [Activation] → [Projection] → output
```

**Performances** :
- Comparable aux Transformers sur beaucoup de tâches
- Beaucoup plus rapide sur longs contextes
- Inférence en temps constant par token (vs linéaire pour Transformers avec KV cache)

**Code conceptuel** :

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x_proj = self.in_proj(x)
        x_ssm, gate = x_proj.chunk(2, dim=-1)

        # Appliquer SSM sélectif
        ssm_out = self.ssm(x_ssm)

        # Gating
        out = ssm_out * torch.sigmoid(gate)
        return self.out_proj(out)
```

**Note** : Implémentation réelle plus complexe, optimisée avec kernels CUDA custom.

### 4.4.4 Architectures hybrides

**Motivation** : Combiner les forces de Transformer et SSM.

**Exemples** :

1. **Alternance de couches** :
```
[Mamba] → [Mamba] → [Attention] → [Mamba] → [Mamba] → [Attention] ...
```

2. **Attention pour le court range, SSM pour le long range**

3. **Hierarchical** : SSM pour compresser, Attention pour raffiner

**Avantages** :
- Flexibilité : Transformer pour raisonnement complexe, SSM pour efficacité
- Meilleur compromis vitesse/qualité

---

## 4.5 Modèles multimodaux

### 4.5.1 Vision-Language Models

**Architecture typique** :

```
Image → [Vision Encoder] → visual tokens
Text → [Text Embeddings] → text tokens

Concat → [Transformer] → Output
```

**Exemples** :
- **CLIP** (OpenAI) : Alignement vision-texte par contrastive learning
- **Flamingo** (DeepMind) : Few-shot vision-language
- **LLaVA** : LLaMA + vision encoder

**Vision encoders** :
- ViT (Vision Transformer)
- CLIP visual encoder
- ConvNet (ResNet, EfficientNet)

### 4.5.2 Audio-Language Models

**Exemples** :
- **Whisper** (OpenAI) : Speech-to-text robuste
- **AudioLM** : Génération audio
- **MusicLM** : Génération musicale

### 4.5.3 Code Models

**Spécificités** :
- Tokenisation adaptée au code (respects indentation, identifiants)
- Entraînement sur code + docs + commits
- Fine-tuning sur tâches spécifiques (completion, debug, tests)

**Exemples** :
- **Codex** (OpenAI, base de GitHub Copilot)
- **StarCoder** (BigCode)
- **CodeLlama** (Meta)

---

## 4.6 Labs pratiques

### Lab 1 : Implémenter un Transformer minimal

```python
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.output(x)

# Test
model = MiniTransformer(vocab_size=10000)
x = torch.randint(0, 10000, (2, 50))  # batch=2, seq=50
output = model(x)
print(output.shape)  # [2, 50, 10000]
```

### Lab 2 : Comparer dense vs MoE

Mesurer latence, mémoire, et qualité (perplexité) pour :
- Modèle dense 1B
- Modèle MoE 8×125M (1B total, ~250M actif)

### Lab 3 : Benchmarker FlashAttention

```python
import time

# Attention standard
start = time.time()
out_standard = standard_attention(q, k, v)
torch.cuda.synchronize()
time_standard = time.time() - start

# FlashAttention
start = time.time()
out_flash = flash_attention(q, k, v)
torch.cuda.synchronize()
time_flash = time.time() - start

print(f"Standard: {time_standard:.4f}s")
print(f"Flash: {time_flash:.4f}s")
print(f"Speedup: {time_standard / time_flash:.2f}x")
```

---

## 4.7 Références et lectures

### Papers clés

**Transformers** :
- Vaswani et al. (2017) - "Attention is All You Need"

**Optimisations** :
- Dao et al. (2022) - "FlashAttention: Fast and Memory-Efficient Exact Attention"
- Beltagy et al. (2020) - "Longformer: The Long-Document Transformer"

**MoE** :
- Shazeer et al. (2017) - "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
- Fedus et al. (2021) - "Switch Transformers: Scaling to Trillion Parameter Models"

**SSM/Mamba** :
- Gu et al. (2021) - "Efficiently Modeling Long Sequences with Structured State Spaces" (S4)
- Gu & Dao (2023) - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

**Multimodal** :
- Radford et al. (2021) - "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)

### Ressources

- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Mamba implementation](https://github.com/state-spaces/mamba)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Architecture Transformer complète (attention, positions, FFN, normalisation)
- ✅ Optimisations (FlashAttention, sparse attention, long-context)
- ✅ Mixture of Experts (MoE) et trade-offs
- ✅ Architectures post-Transformer (SSM, Mamba, hybrides)
- ✅ Modèles multimodaux (vision, audio, code)

**Prochaine étape** : [Partie 5 - Données](../partie-05/README.md) pour maîtriser la collecte, le nettoyage et la préparation des données.
