# Partie 2 : Fondations mathématiques pour LLM

## Objectifs d'apprentissage

À la fin de cette partie, vous serez capable de :

- Manipuler espaces vectoriels, tenseurs et opérations linéaires fondamentales
- Appliquer les concepts de probabilités, entropie et divergence KL aux LLM
- Comprendre et implémenter des algorithmes d'optimisation (SGD, Adam, AdamW)
- Analyser le compromis biais-variance et les lois d'échelle
- Calculer et interpréter les métriques clés (cross-entropy, perplexité, NLL)

## Prérequis

- Algèbre linéaire niveau licence (vecteurs, matrices, produits)
- Probabilités et statistiques de base
- Calcul différentiel (dérivées partielles)
- Python et NumPy

---

## 2.1 Espaces vectoriels et représentations

### 2.1.1 Vecteurs et tenseurs

**Définition** : Un tenseur est une généralisation multidimensionnelle d'un vecteur ou d'une matrice.

**Rangs** :
- Rang 0 : scalaire (nombre unique)
- Rang 1 : vecteur (liste de nombres)
- Rang 2 : matrice (tableau 2D)
- Rang 3+ : tenseur de rang supérieur

**Exemple en PyTorch** :

```python
import torch

# Scalaire (rang 0)
scalar = torch.tensor(3.14)

# Vecteur (rang 1) - représentation d'un token
token_embed = torch.randn(768)  # embedding de dimension 768

# Matrice (rang 2) - batch de séquences
batch = torch.randn(32, 512, 768)  # [batch_size, seq_len, hidden_dim]

# Tenseur rang 4 - batch d'images
images = torch.randn(32, 3, 224, 224)  # [batch, channels, height, width]
```

### 2.1.2 Opérations fondamentales

**Produit scalaire** :
```
v · w = Σ v_i × w_i
```

Mesure de similarité entre vecteurs (utilisé pour attention).

**Produit matriciel** :
```
C = AB  où  C_ij = Σ_k A_ik × B_kj
```

Transformation linéaire fondamentale dans les réseaux de neurones.

**Broadcasting** :
```python
# Ajout d'un biais à un batch
x = torch.randn(32, 768)  # batch de 32 vecteurs
bias = torch.randn(768)    # biais unique
y = x + bias               # broadcasting automatique sur la dimension batch
```

### 2.1.3 Espaces d'embeddings et similarité

**Embeddings** : Représentations denses de tokens dans un espace vectoriel continu.

**Mesures de similarité** :

1. **Similarité cosinus** :
```
cos(v, w) = (v · w) / (||v|| × ||w||)
```
Invariante à la magnitude, entre -1 et 1.

2. **Distance euclidienne** :
```
d(v, w) = ||v - w|| = sqrt(Σ (v_i - w_i)²)
```

3. **Produit scalaire** :
Utilisé dans l'attention (après scaling).

**Exemple pratique** :

```python
def cosine_similarity(v, w):
    """Similarité cosinus entre deux vecteurs."""
    dot_product = torch.dot(v, w)
    norm_v = torch.norm(v)
    norm_w = torch.norm(w)
    return dot_product / (norm_v * norm_w)

# Mesurer la similarité entre embeddings de tokens
embed_king = model.embed("king")
embed_queen = model.embed("queen")
embed_car = model.embed("car")

print(cosine_similarity(embed_king, embed_queen))  # ~0.7-0.8
print(cosine_similarity(embed_king, embed_car))     # ~0.2-0.3
```

### 2.1.4 Normes et normalisation

**Normes courantes** :

- **L1** : ||v||₁ = Σ |v_i|
- **L2** : ||v||₂ = sqrt(Σ v_i²)
- **L∞** : ||v||∞ = max |v_i|

**LayerNorm** (crucial pour Transformers) :

```python
def layer_norm(x, eps=1e-5):
    """Normalisation par couche."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)
```

**Pourquoi LayerNorm ?**
- Stabilise l'entraînement
- Réduit la sensibilité à l'initialisation
- Accélère la convergence

---

## 2.2 Probabilités et théorie de l'information

### 2.2.1 Variables aléatoires et distributions

**Distribution de probabilité** : Fonction qui associe à chaque événement sa probabilité.

**Distribution discrète** (pour les tokens) :

```
P(X = x_i) = p_i  avec Σ p_i = 1
```

**Distribution catégorielle** (sortie d'un LLM) :

```python
logits = model(input_ids)  # [batch, seq_len, vocab_size]
probs = torch.softmax(logits, dim=-1)  # distribution sur le vocabulaire
```

**Softmax** :
```
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

Transforme des scores (logits) en probabilités.

### 2.2.2 Entropie et incertitude

**Entropie de Shannon** :

```
H(P) = -Σ p_i log p_i
```

Mesure l'incertitude moyenne d'une distribution.

**Interprétation** :
- H élevée : distribution uniforme, grande incertitude
- H faible : distribution concentrée, faible incertitude

**Exemple** :

```python
import numpy as np

def entropy(probs):
    """Entropie d'une distribution de probabilités."""
    return -np.sum(probs * np.log2(probs + 1e-9))

# Distribution uniforme (max entropie)
uniform = np.ones(1000) / 1000
print(f"Entropie uniforme: {entropy(uniform):.2f} bits")  # ~9.97 bits

# Distribution concentrée (faible entropie)
concentrated = np.array([0.9, 0.05, 0.05])
print(f"Entropie concentrée: {entropy(concentrated):.2f} bits")  # ~0.57 bits
```

### 2.2.3 Cross-Entropy et divergence KL

**Cross-Entropy** :

```
H(P, Q) = -Σ p_i log q_i
```

Mesure combien Q est une mauvaise approximation de P.

**Divergence de Kullback-Leibler** :

```
KL(P || Q) = Σ p_i log(p_i / q_i) = H(P, Q) - H(P)
```

Mesure la "distance" entre deux distributions (non symétrique).

**Application aux LLM** :

L'objectif d'entraînement est de minimiser la cross-entropy entre :
- P : distribution vraie (one-hot sur le token suivant)
- Q : distribution prédite par le modèle

```python
def cross_entropy_loss(logits, targets):
    """Cross-entropy pour classification."""
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    return loss.mean()
```

### 2.2.4 Negative Log-Likelihood (NLL)

**Définition** :

```
NLL = -log P(données | modèle)
```

Pour un dataset de N tokens :

```
NLL = -Σ log P(token_i | contexte_i)
```

**Perplexité** :

```
Perplexité = exp(NLL / N)
```

Interprétation : nombre moyen de choix équiprobables que le modèle doit considérer.

**Exemple** :

```python
def perplexity(losses):
    """Calcule la perplexité à partir des losses."""
    return torch.exp(torch.mean(losses))

# Sur un batch
losses = cross_entropy_loss(logits, targets)
ppl = perplexity(losses)
print(f"Perplexité: {ppl:.2f}")
```

---

## 2.3 Optimisation et descente de gradient

### 2.3.1 Fonction de perte et optimisation

**Objectif** : Trouver les paramètres θ qui minimisent une fonction de perte L.

```
θ* = argmin_θ L(θ)
```

Pour les LLM :

```
L(θ) = -Σ log P_θ(token_next | context)
```

### 2.3.2 Gradient et dérivées

**Gradient** : Vecteur des dérivées partielles.

```
∇L = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]
```

**Propriété** : Le gradient pointe dans la direction de plus forte augmentation.

**Descente de gradient** : Se déplacer dans la direction opposée.

```
θ_{t+1} = θ_t - η ∇L(θ_t)
```

où η est le learning rate.

### 2.3.3 SGD et variantes

**Stochastic Gradient Descent (SGD)** :

```python
for batch in dataloader:
    loss = compute_loss(model(batch), targets)
    loss.backward()  # calcule les gradients
    optimizer.step()  # met à jour les paramètres
    optimizer.zero_grad()  # réinitialise les gradients
```

**SGD avec momentum** :

```
v_t = β v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - η v_t
```

Accumule les gradients passés pour accélérer la convergence.

**RMSprop** :

Adapte le learning rate par paramètre en fonction de la variance des gradients.

### 2.3.4 Adam et AdamW

**Adam** (Adaptive Moment Estimation) :

Combine momentum et adaptation du learning rate.

```
m_t = β₁ m_{t-1} + (1-β₁) ∇L       # moment 1 (moyenne)
v_t = β₂ v_{t-1} + (1-β₂) (∇L)²    # moment 2 (variance)

m̂_t = m_t / (1 - β₁^t)             # correction de biais
v̂_t = v_t / (1 - β₂^t)

θ_{t+1} = θ_t - η m̂_t / (sqrt(v̂_t) + ε)
```

**AdamW** (Adam avec Weight Decay découplé) :

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # régularisation L2
)
```

**Pourquoi AdamW pour les LLM ?**
- Convergence stable sur grands modèles
- Régularisation efficace
- Hyperparamètres robustes

### 2.3.5 Learning rate scheduling

**Warmup + Cosine Decay** (standard pour LLM) :

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup linéaire
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Pourquoi le warmup ?**
- Évite les gradients explosifs au début de l'entraînement
- Stabilise la trajectoire d'optimisation

---

## 2.4 Biais-Variance et généralisation

### 2.4.1 Le compromis biais-variance

**Biais** : Erreur due à des hypothèses simplificatrices du modèle.
- Modèle trop simple → biais élevé → underfitting

**Variance** : Sensibilité du modèle aux fluctuations des données d'entraînement.
- Modèle trop complexe → variance élevée → overfitting

**Erreur totale** :

```
Erreur = Biais² + Variance + Bruit irréductible
```

**Pour les LLM** :
- **Sous-paramétrisation** (petit modèle) : biais élevé
- **Sur-paramétrisation** (grand modèle) : variance élevée mais... régularisation implicite !

### 2.4.2 Régularisation

**Techniques courantes** :

1. **Weight decay** (L2 regularization) :
```
L_total = L_data + λ ||θ||²
```

2. **Dropout** :
Désactiver aléatoirement des neurones pendant l'entraînement.

```python
dropout = torch.nn.Dropout(p=0.1)
x = dropout(x)  # met 10% des valeurs à zéro
```

3. **Label smoothing** :
Adoucir les labels one-hot pour réduire la sur-confiance.

```python
# Au lieu de [0, 0, 1, 0] utiliser [0.025, 0.025, 0.925, 0.025]
def label_smoothing(targets, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    one_hot = F.one_hot(targets, num_classes).float()
    return one_hot * confidence + smooth_value
```

### 2.4.3 Validation et early stopping

**Principe** : Surveiller la perte sur un ensemble de validation et arrêter quand elle cesse de diminuer.

```python
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

---

## 2.5 Lois d'échelle (Scaling Laws)

### 2.5.1 Principes fondamentaux

**Observation empirique** (Kaplan et al., 2020) :

Les performances d'un LLM suivent des lois de puissance en fonction de :
- **N** : Nombre de paramètres
- **D** : Taille du dataset (nombre de tokens)
- **C** : Budget de compute (FLOPs)

**Formule simplifiée** :

```
L(N, D) ≈ L_∞ + A/N^α + B/D^β
```

où L est la loss, L_∞ est la loss minimale théorique.

### 2.5.2 Implications pratiques

**Trade-offs** :

1. **Nombre de paramètres vs données** :
   - Modèle plus grand → nécessite plus de données
   - Règle Chinchilla : ~20 tokens par paramètre pour l'optimalité

2. **Compute optimal** :
   - Pour un budget C fixé, comment choisir N et D ?
   - Chinchilla : modèles plus petits, entraînés plus longtemps

**Exemple** :

| Modèle     | Paramètres | Tokens d'entraînement | Ratio tokens/param |
|------------|------------|-----------------------|--------------------|
| GPT-3      | 175B       | 300B                  | 1.7                |
| Chinchilla | 70B        | 1.4T                  | 20                 |
| LLaMA 2    | 70B        | 2T                    | 28.6               |

**Conséquence** : Entraîner plus longtemps sur plus de données est souvent plus efficace que simplement augmenter la taille du modèle.

### 2.5.3 Prédire les performances

**Utilité** : Estimer la loss finale sans entraîner complètement le modèle.

```python
def predict_loss(N, D, L_inf=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Prédit la loss selon les scaling laws (valeurs approx. de Kaplan et al.)."""
    return L_inf + A / (N ** alpha) + B / (D ** beta)

# Exemple
N = 1e9  # 1B paramètres
D = 20e9  # 20B tokens
print(f"Loss prédite: {predict_loss(N, D):.3f}")
```

---

## 2.6 Exercices pratiques

### Lab 1 : Implémentation de cross-entropy

Implémentez la fonction de cross-entropy from scratch et comparez avec PyTorch.

```python
import torch
import torch.nn.functional as F

def my_cross_entropy(logits, targets):
    """
    logits: [batch_size, num_classes]
    targets: [batch_size] (indices de classes)
    """
    # TODO: Implémenter
    pass

# Test
logits = torch.randn(32, 10000)  # batch=32, vocab=10000
targets = torch.randint(0, 10000, (32,))

loss_pytorch = F.cross_entropy(logits, targets)
loss_custom = my_cross_entropy(logits, targets)

print(f"PyTorch: {loss_pytorch:.4f}")
print(f"Custom:  {loss_custom:.4f}")
print(f"Différence: {abs(loss_pytorch - loss_custom):.6f}")
```

### Lab 2 : Visualisation de paysages de perte

Visualisez la surface de perte pour une petite régression.

```python
import matplotlib.pyplot as plt
import numpy as np

# Générer des données
X = np.random.randn(100, 1)
y = 3 * X + 2 + np.random.randn(100, 1) * 0.1

# Grille de paramètres
w_vals = np.linspace(1, 5, 100)
b_vals = np.linspace(0, 4, 100)
W, B = np.meshgrid(w_vals, b_vals)

# Calculer la loss pour chaque combinaison
losses = np.zeros_like(W)
for i in range(len(w_vals)):
    for j in range(len(b_vals)):
        y_pred = w_vals[i] * X + b_vals[j]
        losses[j, i] = np.mean((y - y_pred) ** 2)

# Plot
plt.contour(W, B, losses, levels=20)
plt.xlabel('Weight')
plt.ylabel('Bias')
plt.title('Loss landscape')
plt.colorbar()
plt.show()
```

### Lab 3 : Comparaison d'optimiseurs

Comparez SGD, Adam et AdamW sur un problème simple.

```python
# TODO: Entraîner un petit réseau avec chaque optimiseur
# Tracker et comparer les courbes de convergence
```

---

## 2.7 Lectures recommandées

### Papers fondateurs
- Kaplan et al. (2020) - "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) - "Training Compute-Optimal Large Language Models" (Chinchilla)
- Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"

### Ressources pédagogiques
- [Mathematics for Machine Learning](https://mml-book.github.io/)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
- [Probability and Statistics for Deep Learning](https://probml.github.io/pml-book/)

---

## Résumé de la Partie 2

Vous maîtrisez maintenant :
- ✅ Espaces vectoriels, tenseurs et opérations fondamentales
- ✅ Probabilités, entropie, cross-entropy et divergence KL
- ✅ Optimisation par gradient, Adam/AdamW, learning rate scheduling
- ✅ Biais-variance, régularisation et généralisation
- ✅ Lois d'échelle et prédiction de performances

**Prochaine étape** : [Partie 3 - Bases de deep learning appliquées au texte](../partie-03/README.md) pour construire vos premiers modèles de langage.
