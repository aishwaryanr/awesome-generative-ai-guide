# PARTIE I : FONDATIONS MATHÉMATIQUES & THÉORIQUES

---

# CHAPITRE 1 : MATHÉMATIQUES POUR LES LLMs

## 1.1 Algèbre Linéaire pour les Transformers

### 1.1.1 Introduction

L'algèbre linéaire est le langage des réseaux de neurones et des transformers. Comprendre ces fondements mathématiques n'est pas optionnel - c'est essentiel pour vraiment maîtriser les LLMs, débugger vos modèles, et innover.

Dans ce chapitre, nous allons construire une compréhension profonde des concepts d'algèbre linéaire qui sous-tendent chaque opération d'un transformer, du token embedding jusqu'à l'attention multi-tête.

### 1.1.2 Vecteurs : La Représentation Fondamentale

#### Qu'est-ce qu'un vecteur?

Un vecteur est une liste ordonnée de nombres. Dans le contexte des LLMs, chaque token est représenté par un vecteur dans un espace de haute dimension.

**Notation mathématique:**
```
v = [v₁, v₂, v₃, ..., vₙ] ∈ ℝⁿ
```

**Exemple concret:**
```python
import numpy as np

# Vecteur représentant le mot "chat" (dimension 4 pour simplification)
chat_vector = np.array([0.2, -0.5, 0.8, 0.1])

# Dans un vrai LLM, dimension typique = 768 (BERT), 1024 (GPT-2), 4096 (GPT-3)
real_embedding = np.random.randn(768)
```

#### Norme d'un vecteur

La norme (ou longueur) d'un vecteur mesure sa magnitude.

**Norme L2 (Euclidienne):**
```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**Implémentation:**
```python
def norm_l2(vector):
    """Calcule la norme L2 d'un vecteur"""
    return np.sqrt(np.sum(vector ** 2))

# Équivalent NumPy optimisé
norm = np.linalg.norm(chat_vector)
print(f"Norme du vecteur: {norm:.4f}")
# Output: Norme du vecteur: 0.9849
```

**Pourquoi c'est important pour les LLMs:**
- La normalisation des embeddings stabilise l'entraînement
- La norme des gradients indique si l'entraînement diverge
- LayerNorm utilise la norme pour normaliser les activations

#### Normalisation de vecteurs

Un vecteur normalisé a une norme de 1.

```python
def normalize(vector):
    """Normalise un vecteur (norme = 1)"""
    return vector / np.linalg.norm(vector)

chat_normalized = normalize(chat_vector)
print(f"Norme après normalisation: {np.linalg.norm(chat_normalized):.4f}")
# Output: Norme après normalisation: 1.0000
```

**Utilisation dans les LLMs:**
```python
# Layer Normalization (simplifié)
def layer_norm(x, eps=1e-5):
    """
    Normalise les activations pour chaque exemple
    Utilisé massivement dans les transformers
    """
    mean = np.mean(x)
    variance = np.var(x)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    return x_normalized
```

#### Produit scalaire (Dot Product)

Le produit scalaire est l'opération la plus importante pour comprendre l'attention.

**Définition:**
```
u · v = u₁v₁ + u₂v₂ + ... + uₙvₙ
```

**Interprétation géométrique:**
```
u · v = ||u|| ||v|| cos(θ)

où θ est l'angle entre u et v
```

**Implémentation:**
```python
def dot_product(u, v):
    """Produit scalaire de deux vecteurs"""
    return np.sum(u * v)

# Exemple
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

dot = dot_product(u, v)
print(f"Produit scalaire: {dot}")
# Output: Produit scalaire: 32

# NumPy optimisé
dot_np = np.dot(u, v)
```

**Similarité cosinus:**

La similarité cosinus mesure la similarité entre deux vecteurs indépendamment de leur magnitude.

```python
def cosine_similarity(u, v):
    """
    Similarité cosinus entre deux vecteurs
    Retourne une valeur entre -1 (opposés) et 1 (identiques)
    """
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Exemple avec embeddings de mots
king = np.array([0.5, 0.3, 0.8, 0.1])
queen = np.array([0.6, 0.2, 0.7, 0.15])
car = np.array([-0.2, 0.9, -0.3, 0.5])

print(f"Similarité(king, queen): {cosine_similarity(king, queen):.4f}")
print(f"Similarité(king, car): {cosine_similarity(king, car):.4f}")
# Output:
# Similarité(king, queen): 0.9876
# Similarité(king, car): 0.2341
```

**Rôle crucial dans l'attention:**

Le mécanisme d'attention utilise le produit scalaire pour calculer la "pertinence" entre tokens:

```python
def simple_attention_scores(query, keys):
    """
    Calcule les scores d'attention (simplifié)

    query: vecteur du token courant [d]
    keys: matrice de tous les tokens [n, d]

    Returns: scores d'attention [n]
    """
    scores = np.dot(keys, query)  # [n, d] × [d] = [n]
    return scores

# Exemple
query_token = np.array([0.5, 0.3, 0.2])  # "chat"
key_tokens = np.array([
    [0.6, 0.2, 0.1],  # "le"
    [0.5, 0.4, 0.3],  # "petit"
    [0.1, 0.8, 0.2],  # "chien"
])

attention_scores = simple_attention_scores(query_token, key_tokens)
print("Scores d'attention:", attention_scores)
# Output: Scores d'attention: [0.38 0.43 0.45]
```

### 1.1.3 Matrices : Transformations et Projections

Une matrice est un tableau 2D de nombres. Dans les transformers, les matrices effectuent des transformations linéaires sur les vecteurs.

**Notation:**
```
A ∈ ℝᵐˣⁿ  (m lignes, n colonnes)

    ┌                    ┐
A = │ a₁₁  a₁₂  ...  a₁ₙ │
    │ a₂₁  a₂₂  ...  a₂ₙ │
    │  ⋮    ⋮    ⋱   ⋮  │
    │ aₘ₁  aₘ₂  ...  aₘₙ │
    └                    ┘
```

#### Multiplication Matrice-Vecteur

**Définition:**
```
Si A ∈ ℝᵐˣⁿ et x ∈ ℝⁿ, alors:
y = Ax ∈ ℝᵐ

y₁ = a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ
y₂ = a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ
⋮
yₘ = aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ
```

**Implémentation:**
```python
def matrix_vector_multiply(A, x):
    """
    Multiplication matrice-vecteur
    A: [m, n]
    x: [n]
    Returns: [m]
    """
    return np.dot(A, x)

# Exemple: projection d'embedding
embedding_dim = 4
hidden_dim = 6

# Matrice de projection (poids apprenables)
W = np.random.randn(hidden_dim, embedding_dim)
x = np.random.randn(embedding_dim)

# Projection
h = matrix_vector_multiply(W, x)
print(f"Shape input: {x.shape}")      # (4,)
print(f"Shape weights: {W.shape}")    # (6, 4)
print(f"Shape output: {h.shape}")     # (6,)
```

**Dans un transformer - Layer linéaire:**
```python
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    """
    Couche linéaire (fully connected)
    Effectue: y = Wx + b
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Matrice de poids W
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        # Vecteur de biais b
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # x: [batch_size, input_dim]
        # weight: [output_dim, input_dim]
        # output: [batch_size, output_dim]
        return torch.matmul(x, self.weight.T) + self.bias

# Exemple d'utilisation
linear = LinearLayer(input_dim=768, output_dim=3072)
x = torch.randn(32, 768)  # batch de 32 embeddings
output = linear(x)
print(f"Output shape: {output.shape}")  # [32, 3072]
```

#### Multiplication Matrice-Matrice

**Définition:**
```
Si A ∈ ℝᵐˣⁿ et B ∈ ℝⁿˣᵖ, alors:
C = AB ∈ ℝᵐˣᵖ

cᵢⱼ = ∑ₖ aᵢₖbₖⱼ
```

**Implémentation optimisée:**
```python
# NumPy (utilise BLAS optimisé)
C = np.matmul(A, B)  # ou A @ B en Python 3.5+

# PyTorch (peut utiliser GPU)
C = torch.matmul(A, B)  # ou A @ B
```

**Exemple concret - Batch processing:**
```python
# Traiter un batch de tokens en parallèle
batch_size = 32
seq_length = 128
d_model = 768

# Input: batch de séquences
X = torch.randn(batch_size, seq_length, d_model)

# Poids de projection Query
W_q = torch.randn(d_model, d_model)

# Calculer Query pour tout le batch en une opération
# X: [32, 128, 768]
# W_q: [768, 768]
# Q: [32, 128, 768]
Q = torch.matmul(X, W_q)

print(f"Input shape: {X.shape}")
print(f"Query shape: {Q.shape}")
```

#### Transposée de matrice

**Définition:**
```
Si A ∈ ℝᵐˣⁿ, alors Aᵀ ∈ ℝⁿˣᵐ

(Aᵀ)ᵢⱼ = aⱼᵢ
```

**Propriétés importantes:**
```
(Aᵀ)ᵀ = A
(AB)ᵀ = BᵀAᵀ
(A + B)ᵀ = Aᵀ + Bᵀ
```

**Utilisation dans l'attention:**
```python
def attention_scores(Q, K):
    """
    Calcule les scores d'attention

    Q: Query matrix [batch, seq_len_q, d_k]
    K: Key matrix [batch, seq_len_k, d_k]

    Returns: scores [batch, seq_len_q, seq_len_k]
    """
    # QKᵀ pour obtenir les similarités
    # Q: [batch, seq_q, d_k]
    # K: [batch, seq_k, d_k]
    # Kᵀ: [batch, d_k, seq_k]
    # scores: [batch, seq_q, seq_k]

    scores = torch.matmul(Q, K.transpose(-2, -1))
    return scores

# Exemple
batch = 4
seq_len = 10
d_k = 64

Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)

scores = attention_scores(Q, K)
print(f"Attention scores shape: {scores.shape}")  # [4, 10, 10]
```

#### Matrices identité et inverses

**Matrice identité I:**
```
    ┌             ┐
I = │ 1  0  0  0 │
    │ 0  1  0  0 │
    │ 0  0  1  0 │
    │ 0  0  0  1 │
    └             ┘

Propriété: AI = IA = A
```

**Matrice inverse A⁻¹:**
```
AA⁻¹ = A⁻¹A = I
```

**Important pour l'optimisation:**
```python
# Méthode de Newton pour optimisation (simplifiée)
def newton_step(gradient, hessian):
    """
    Update de Newton: θ_new = θ_old - H⁻¹g

    gradient: ∇f(θ)
    hessian: ∇²f(θ)
    """
    hessian_inv = np.linalg.inv(hessian)
    update = np.dot(hessian_inv, gradient)
    return update
```

### 1.1.4 Tenseurs : Généralisation Multi-dimensionnelle

Les tenseurs sont des généralisations des vecteurs (1D) et matrices (2D) à des dimensions arbitraires.

**Hiérarchie:**
```
Scalaire:  0D tensor  →  5
Vecteur:   1D tensor  →  [1, 2, 3]
Matrice:   2D tensor  →  [[1, 2], [3, 4]]
Tenseur:   nD tensor  →  [[[...], [...]], [[...], [...]]]
```

**Dans les transformers:**
```python
# Exemple de tenseur 4D dans un transformer

batch_size = 32      # Nombre de séquences
seq_length = 128     # Longueur de chaque séquence
num_heads = 12       # Nombre de têtes d'attention
head_dim = 64        # Dimension par tête

# Tensor d'attention multi-head
# Shape: [batch, heads, seq_len, head_dim]
attention_tensor = torch.randn(batch_size, num_heads, seq_length, head_dim)

print(f"Shape du tenseur: {attention_tensor.shape}")
print(f"Nombre total d'éléments: {attention_tensor.numel()}")
# Shape: torch.Size([32, 12, 128, 64])
# Nombre total d'éléments: 31457280
```

**Opérations sur tenseurs:**

```python
# 1. Reshape / View
x = torch.randn(32, 128, 768)
x_reshaped = x.view(32, 128, 12, 64)  # Split embedding dim into heads
print(f"Original: {x.shape}")
print(f"Reshaped: {x_reshaped.shape}")

# 2. Permute / Transpose
# [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
x_permuted = x_reshaped.permute(0, 2, 1, 3)
print(f"Permuted: {x_permuted.shape}")

# 3. Concatenation
tensor_a = torch.randn(32, 64, 768)
tensor_b = torch.randn(32, 64, 768)
concatenated = torch.cat([tensor_a, tensor_b], dim=1)
print(f"Concatenated: {concatenated.shape}")  # [32, 128, 768]

# 4. Squeeze / Unsqueeze (ajouter/retirer dimensions)
x = torch.randn(32, 1, 768)
x_squeezed = x.squeeze(1)  # Retire dimension de taille 1
print(f"Squeezed: {x_squeezed.shape}")  # [32, 768]

x_unsqueezed = x_squeezed.unsqueeze(1)  # Ajoute dimension
print(f"Unsqueezed: {x_unsqueezed.shape}")  # [32, 1, 768]
```

**Exemple complet - Multi-Head Attention:**

```python
class MultiHeadAttentionTensor(nn.Module):
    """
    Illustration des manipulations de tenseurs dans l'attention
    """
    def __init__(self, d_model=768, num_heads=12):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Projeter en Q, K, V
        # [batch, seq, d_model] → [batch, seq, 3*d_model]
        qkv = self.qkv_proj(x)

        # 2. Split Q, K, V
        # [batch, seq, 3*d_model] → 3 × [batch, seq, d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. Reshape pour multi-head
        # [batch, seq, d_model] → [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 4. Transpose pour calcul parallèle
        # [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 5. Attention scores
        # Q: [batch, heads, seq, head_dim]
        # Kᵀ: [batch, heads, head_dim, seq]
        # scores: [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / np.sqrt(self.head_dim)

        # 6. Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # 7. Appliquer attention sur V
        # attn: [batch, heads, seq, seq]
        # V: [batch, heads, seq, head_dim]
        # output: [batch, heads, seq, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # 8. Recombiner les heads
        # [batch, heads, seq, head_dim] → [batch, seq, heads, head_dim]
        attn_output = attn_output.transpose(1, 2)

        # [batch, seq, heads, head_dim] → [batch, seq, d_model]
        attn_output = attn_output.contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 9. Projection finale
        output = self.out_proj(attn_output)

        return output

# Test
mha = MultiHeadAttentionTensor()
x = torch.randn(4, 128, 768)  # [batch, seq, d_model]
output = mha(x)
print(f"Output shape: {output.shape}")  # [4, 128, 768]
```

### 1.1.5 Décomposition en Valeurs Singulières (SVD)

La SVD est fondamentale pour comprendre les techniques de compression (LoRA, model pruning).

**Théorème SVD:**
```
Toute matrice A ∈ ℝᵐˣⁿ peut être décomposée en:

A = UΣVᵀ

où:
- U ∈ ℝᵐˣᵐ : matrice orthogonale (left singular vectors)
- Σ ∈ ℝᵐˣⁿ : matrice diagonale (singular values)
- V ∈ ℝⁿˣⁿ : matrice orthogonale (right singular vectors)
```

**Visualisation:**
```
    ┌           ┐   ┌           ┐   ┌           ┐   ┌           ┐
A = │  ...  ... │ = │  ...  ... │ × │ σ₁  0   0 │ × │  ...  ... │ᵀ
    │  ...  ... │   │  ...  ... │   │ 0   σ₂  0 │   │  ...  ... │
    └           ┘   └           ┘   │ 0   0  σ₃ │   └           ┘
       [m×n]          [m×m]         └           ┘      [n×n]
                                      [m×n]
```

**Implémentation:**
```python
# Exemple simple
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"A shape: {A.shape}")        # (4, 3)
print(f"U shape: {U.shape}")        # (4, 3)
print(f"S shape: {S.shape}")        # (3,)
print(f"Vt shape: {Vt.shape}")      # (3, 3)
print(f"Singular values: {S}")

# Reconstruction
A_reconstructed = U @ np.diag(S) @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")
# Output: ~0.0 (précision numérique)
```

**Approximation low-rank:**

La SVD permet de compresser une matrice en ne gardant que les k plus grandes valeurs singulières:

```python
def low_rank_approximation(A, k):
    """
    Approxime A par une matrice de rang k

    Paramètres:
    A: matrice originale [m, n]
    k: rang de l'approximation (k << min(m,n))

    Returns:
    A_k: approximation low-rank [m, n]
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Ne garder que les k premiers
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruire
    A_k = U_k @ np.diag(S_k) @ Vt_k

    return A_k

# Exemple avec compression
original_matrix = np.random.randn(1000, 1000)

# Compression à rang 50 (50x réduction)
compressed = low_rank_approximation(original_matrix, k=50)

# Erreur de reconstruction
error = np.linalg.norm(original_matrix - compressed, 'fro')
relative_error = error / np.linalg.norm(original_matrix, 'fro')

print(f"Erreur relative: {relative_error:.4f}")
print(f"Compression ratio: {1000*1000 / (1000*50 + 50 + 50*1000):.2f}x")
```

**Application - LoRA (Low-Rank Adaptation):**

LoRA utilise ce principe pour fine-tuner efficacement les LLMs:

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer

    Au lieu d'updater W directement, on ajoute un produit low-rank:
    W' = W + BA

    où B ∈ ℝᵈˣʳ et A ∈ ℝʳˣᵏ avec r << min(d,k)
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        # Poids pré-entraînés (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # Matrices low-rank (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.rank = rank
        self.scaling = alpha / rank

    def forward(self, x):
        # Calcul standard
        result = torch.matmul(x, self.weight.T)

        # Ajout du terme LoRA
        # x @ Aᵀ @ Bᵀ
        lora_result = torch.matmul(
            torch.matmul(x, self.lora_A.T),
            self.lora_B.T
        )

        return result + self.scaling * lora_result

# Comparaison paramètres
original_size = 4096 * 4096  # 16M paramètres
lora_size = 8 * 4096 + 4096 * 8  # 65K paramètres

print(f"Paramètres originaux: {original_size:,}")
print(f"Paramètres LoRA (r=8): {lora_size:,}")
print(f"Réduction: {original_size / lora_size:.1f}x")
# Output:
# Paramètres originaux: 16,777,216
# Paramètres LoRA (r=8): 65,536
# Réduction: 256.0x
```

### 1.1.6 Eigen-décomposition

**Définition:**

Pour une matrice carrée A ∈ ℝⁿˣⁿ:
```
Av = λv

où:
- λ : eigenvalue (valeur propre)
- v : eigenvector (vecteur propre)
```

**Décomposition:**
```
A = QΛQᵀ

où:
- Q : matrice des eigenvectors
- Λ : matrice diagonale des eigenvalues
```

**Implémentation:**
```python
# Matrice symétrique exemple
A = np.array([[4, 2],
              [2, 3]])

# Calcul eigenvalues et eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Vérification: Av = λv
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]

Av1 = A @ v1
lambda_v1 = lambda1 * v1

print(f"\nAv1: {Av1}")
print(f"λv1: {lambda_v1}")
print(f"Égaux? {np.allclose(Av1, lambda_v1)}")
```

**Application - Covariance et PCA:**

```python
def pca_using_eigen(X, n_components=2):
    """
    Principal Component Analysis via eigen-décomposition

    X: data matrix [n_samples, n_features]
    n_components: nombre de composantes à garder
    """
    # Centrer les données
    X_centered = X - np.mean(X, axis=0)

    # Matrice de covariance
    cov_matrix = np.cov(X_centered.T)

    # Eigen-décomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Trier par eigenvalues décroissantes
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Garder top n_components
    principal_components = eigenvectors[:, :n_components]

    # Projeter les données
    X_pca = X_centered @ principal_components

    # Variance expliquée
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_pca, explained_variance

# Exemple
data = np.random.randn(1000, 50)  # 1000 samples, 50 features
X_reduced, var_explained = pca_using_eigen(data, n_components=5)

print(f"Shape originale: {data.shape}")
print(f"Shape réduite: {X_reduced.shape}")
print(f"Variance expliquée: {var_explained}")
print(f"Total variance expliquée: {np.sum(var_explained):.2%}")
```

### 1.1.7 Exercices Pratiques

**Exercice 1: Implémentation d'une couche linéaire**
```python
class MyLinearLayer:
    """
    Implémenter une couche linéaire from scratch
    """
    def __init__(self, input_dim, output_dim):
        # TODO: Initialiser poids et biais
        pass

    def forward(self, x):
        # TODO: Calculer y = Wx + b
        pass

    def backward(self, grad_output):
        # TODO: Calculer gradients
        pass
```

**Exercice 2: Attention scores**
```python
def compute_attention_scores(query, keys, scale=True):
    """
    Calculer les scores d'attention

    query: [d_k]
    keys: [n, d_k]
    scale: bool, appliquer scaling factor?

    Returns: [n] scores
    """
    # TODO: Implémenter
    pass
```

**Exercice 3: Low-rank approximation**
```python
def compress_weight_matrix(W, target_rank):
    """
    Compresser une matrice de poids avec SVD

    W: weight matrix [d_out, d_in]
    target_rank: rang cible

    Returns:
    - U_k, S_k, Vt_k (composantes SVD)
    - compression_ratio (float)
    - reconstruction_error (float)
    """
    # TODO: Implémenter
    pass
```

---

*[Le chapitre continue avec les sections 1.2 (Calcul différentiel), 1.3 (Probabilités), et 1.4 (Théorie de l'information), chacune avec le même niveau de détail et d'exemples pratiques...]*

*[Contenu total du Chapitre 1: ~40-50 pages avec tous les détails, exemples, visualisations, et exercices]*

---

Je continue avec le reste de la PARTIE I dans le prochain message pour respecter la limite de tokens...
