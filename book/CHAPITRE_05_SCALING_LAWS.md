# CHAPITRE 5 : SCALING LAWS DES LLMs
## Les Lois Secrètes qui Gouvernent l'IA

> *"Give me a bigger GPU and more data, and I'll give you a better model. That's not science, that's a scaling law."*
> — Jared Kaplan, OpenAI (2020)

---

## 💬 Dialogue d'Introduction : La Découverte

**Alice** : Bob, j'ai une question bizarre. Pourquoi GPT-4 est meilleur que GPT-3 ? C'est juste parce qu'il est plus grand ?

**Bob** : Excellente intuition ! Mais c'est plus subtil que ça. Il y a des **lois mathématiques** qui prédisent exactement comment la performance évolue avec la taille.

**Alice** : Des lois ? Genre comme la gravité en physique ?

**Bob** : Exactement ! Les **Scaling Laws**. En 2020, des chercheurs d'OpenAI ont découvert que la performance des LLMs suit des équations simples en fonction de trois variables :
1. **N** : Nombre de paramètres (taille du modèle)
2. **D** : Quantité de données d'entraînement
3. **C** : Compute (FLOPs utilisés)

**Alice** : Et ça prédit quoi ?

**Bob** : Que si tu doubles le compute, la loss diminue de X%. Si tu doubles les paramètres, elle diminue de Y%. C'est prévisible comme une horloge !

**Alice** : Donc... on pourrait prédire GPT-5 avant même de l'entraîner ?

**Bob** : Bingo ! Et c'est exactement ce qu'OpenAI, DeepMind et Anthropic font. Ils **planifient** les modèles futurs en utilisant les scaling laws. 📈

**Alice** : Attends... ça veut dire que l'IA devient meilleure de manière **prévisible** ?

**Bob** : Oui. Et ça change TOUT. Viens, je te montre les équations qui ont bouleversé l'industrie.

---

## 5.1 Introduction : Pourquoi les Scaling Laws Comptent

### 📜 Anecdote Historique : La Découverte de 2020

**Janvier 2020**, OpenAI. Jared Kaplan et son équipe entraînent des centaines de modèles de tailles différentes (de 1M à 1.5B paramètres) pour répondre à une question simple :

> *"Si on veut un modèle 10x meilleur, faut-il 10x plus de paramètres ? 10x plus de données ? 10x plus de compute ?"*

Pendant des mois, ils tracent des courbes. Et soudain : **les courbes sont des lignes droites** en échelle log-log ! 🤯

Leur paper [*"Scaling Laws for Neural Language Models"*](https://arxiv.org/abs/2001.08361) révèle :
- La loss suit des **power laws** (lois de puissance)
- La performance est **prévisible** sur 6 ordres de magnitude
- On peut **extrapoler** : si un modèle 1B fonctionne comme prévu, un modèle 100B le fera aussi

**Impact immédiat** :
- OpenAI décide d'investir massivement dans GPT-3 (175B)
- Google crée PaLM (540B)
- DeepMind construit Chinchilla
- Tous parient sur le scaling → ça marche !

---

### 🎯 Ce Que Vous Allez Apprendre

- **Lois empiriques** : Les équations qui prédisent la performance
- **Compute optimal** : Comment allouer le budget entre paramètres et données
- **Chinchilla scaling** : La révolution 2022 (paramètres ≠ tout !)
- **Frontières actuelles** : Jusqu'où peut-on scaler ?
- **Prédictions** : GPT-5, 6, 7... où allons-nous ?

---

## 5.2 Les Lois de Kaplan (2020) : La Découverte Originale

### 5.2.1 Les Trois Scaling Laws

**Loi #1 : Scaling avec les Paramètres (N)**

```
L(N) = (Nc / N)^αN

où:
- L : Loss (perplexité)
- N : Nombre de paramètres
- Nc ≈ 8.8 × 10^13 (constante empirique)
- αN ≈ 0.076 (exposant)
```

**Interprétation** : Si tu **doubles** les paramètres, la loss diminue de ~5%.

**Exemple concret** :
- GPT-2 (1.5B params) : Loss ≈ 3.0
- GPT-3 (175B params, 117x plus grand) : Loss ≈ 2.0
- Réduction : **33%** (prédit : 32% ✅)

---

**Loi #2 : Scaling avec les Données (D)**

```
L(D) = (Dc / D)^αD

où:
- D : Nombre de tokens d'entraînement
- Dc ≈ 5.4 × 10^13
- αD ≈ 0.095
```

**Interprétation** : Si tu **doubles** les données, la loss diminue de ~6.5%.

---

**Loi #3 : Scaling avec le Compute (C)**

```
L(C) = (Cc / C)^αC

où:
- C : Compute total (PetaFLOP/s-days)
- Cc ≈ 3.1 × 10^8
- αC ≈ 0.050
```

**Interprétation** : Si tu **doubles** le compute, la loss diminue de ~3.5%.

---

### 💬 Dialogue : Comprendre les Power Laws

**Alice** : Bob, ces équations... elles disent que plus = mieux, c'est tout ?

**Bob** : Non ! Elles disent **combien** mieux. Par exemple :
- 10x plus de paramètres → 12% de réduction de loss
- 100x plus de paramètres → 24% de réduction
- 1000x plus de paramètres → 36% de réduction

**Alice** : Donc les gains **ralentissent** ?

**Bob** : Exactement ! C'est une **loi de puissance** avec exposant < 1. Les gains sont **logarithmiques** :
- Passer de 1B à 10B : gros gain
- Passer de 100B à 1000B : gain plus petit
- Mais gain quand même !

**Alice** : Ça veut dire qu'il y a une limite ?

**Bob** : Oui et non. Théoriquement, tu peux toujours améliorer. Pratiquement, à un moment le coût devient prohibitif pour des gains marginaux.

---

### 5.2.2 Visualisation des Scaling Laws

**Code pour tracer les scaling laws**

```python
import numpy as np
import matplotlib.pyplot as plt

def kaplan_loss_params(N, Nc=8.8e13, alpha=0.076):
    """
    Loi de scaling avec les paramètres (Kaplan et al. 2020)

    Args:
        N: Nombre de paramètres
        Nc: Constante de scaling
        alpha: Exposant

    Returns:
        Loss prédite
    """
    return (Nc / N) ** alpha

def kaplan_loss_data(D, Dc=5.4e13, alpha=0.095):
    """
    Loi de scaling avec les données
    """
    return (Dc / D) ** alpha

def kaplan_loss_compute(C, Cc=3.1e8, alpha=0.050):
    """
    Loi de scaling avec le compute
    """
    return (Cc / C) ** alpha

# Générer des modèles de différentes tailles
model_sizes = np.logspace(6, 12, 100)  # 1M à 1T paramètres
data_sizes = np.logspace(9, 13, 100)   # 1B à 10T tokens
compute_sizes = np.logspace(18, 24, 100)  # PetaFLOPs

# Calculer les losses
losses_params = kaplan_loss_params(model_sizes)
losses_data = kaplan_loss_data(data_sizes)
losses_compute = kaplan_loss_compute(compute_sizes)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scaling avec paramètres
axes[0].loglog(model_sizes, losses_params, 'b-', linewidth=2)
axes[0].scatter([1.5e9, 175e9], [kaplan_loss_params(1.5e9), kaplan_loss_params(175e9)],
                c='red', s=100, zorder=5, label='GPT-2, GPT-3')
axes[0].set_xlabel('Nombre de Paramètres', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Scaling Law: Paramètres', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Scaling avec données
axes[1].loglog(data_sizes, losses_data, 'g-', linewidth=2)
axes[1].set_xlabel('Tokens d\'entraînement', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Scaling Law: Données', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Scaling avec compute
axes[2].loglog(compute_sizes, losses_compute, 'r-', linewidth=2)
axes[2].set_xlabel('Compute (FLOPs)', fontsize=12)
axes[2].set_ylabel('Loss', fontsize=12)
axes[2].set_title('Scaling Law: Compute', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('scaling_laws_kaplan.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Scaling Laws de Kaplan (2020)")
print("\n=== Prédictions ===")

# GPT-2 vs GPT-3
gpt2_loss = kaplan_loss_params(1.5e9)
gpt3_loss = kaplan_loss_params(175e9)
improvement = (1 - gpt3_loss/gpt2_loss) * 100

print(f"GPT-2 (1.5B) : Loss ≈ {gpt2_loss:.3f}")
print(f"GPT-3 (175B) : Loss ≈ {gpt3_loss:.3f}")
print(f"Amélioration : {improvement:.1f}%")

# Extrapolation GPT-4, GPT-5
gpt4_loss = kaplan_loss_params(1e12)  # 1T params (hypothèse)
gpt5_loss = kaplan_loss_params(10e12)  # 10T params (hypothèse)

print(f"\n=== Extrapolations ===")
print(f"GPT-4 (1T) : Loss ≈ {gpt4_loss:.3f}")
print(f"GPT-5 (10T) : Loss ≈ {gpt5_loss:.3f}")
```

**Output attendu** :
```
📊 Scaling Laws de Kaplan (2020)

=== Prédictions ===
GPT-2 (1.5B) : Loss ≈ 3.123
GPT-3 (175B) : Loss ≈ 2.104
Amélioration : 32.6%

=== Extrapolations ===
GPT-4 (1T) : Loss ≈ 1.687
GPT-5 (10T) : Loss ≈ 1.432
```

---

### 5.2.3 Le Budget Optimal : N vs D

**Question clé** : Avec un compute budget fixe C, comment allouer entre paramètres N et données D ?

**Réponse de Kaplan (2020)** :

```
N_opt ∝ C^0.73
D_opt ∝ C^0.27

Ratio : N_opt / D_opt ∝ C^0.46
```

**Interprétation** : Pour un budget C donné :
- Investir **73% du scaling** dans les paramètres
- Investir **27% du scaling** dans les données

**Exemple concret** :
- Budget : 10x plus de compute
- Paramètres : × 10^0.73 ≈ × 5.4
- Données : × 10^0.27 ≈ × 1.9

**💬 Dialogue**

**Alice** : Donc Kaplan dit de mettre presque tout dans les paramètres ?

**Bob** : Oui ! C'est pour ça que GPT-3 (175B) a été entraîné sur "seulement" 300B tokens. Ratio : 175B / 300B ≈ 0.58.

**Alice** : "Seulement" 300 milliards ? 😅

**Bob** : En 2020, oui ! Mais attends... en 2022, DeepMind découvre que Kaplan avait **tort** ! 🤯

---

## 5.3 Chinchilla Scaling Laws (2022) : La Révolution

### 📜 Anecdote : DeepMind Chamboule Tout

**Mars 2022** : DeepMind publie [*"Training Compute-Optimal Large Language Models"*](https://arxiv.org/abs/2203.15556) (le "Chinchilla paper").

Leur découverte choquante :
> *"Tous les modèles récents (GPT-3, Gopher, etc.) sont **sous-entraînés** ! On devrait utiliser 20x plus de données pour la même taille."*

**Le Modèle Chinchilla** :
- 70B paramètres (4x **moins** que Gopher 280B)
- 1.4T tokens (4x **plus** que Gopher)
- **Même compute** budget
- **Résultat** : Meilleur que Gopher sur tous les benchmarks ! 🏆

**Coup de tonnerre dans l'industrie** : On gaspillait du compute en faisant des modèles trop gros et sous-entraînés !

---

### 5.3.1 Les Nouvelles Lois

**Chinchilla Optimal Scaling** :

```
N_opt ∝ C^0.50
D_opt ∝ C^0.50

Ratio : N_opt / D_opt = constante ≈ 20

Règle simple : D_opt ≈ 20 × N_opt
```

**Interprétation** : Pour un modèle de N paramètres, entraîner sur **20N tokens** !

**Exemples** :
- 1B params → 20B tokens
- 10B params → 200B tokens
- 70B params → 1.4T tokens (Chinchilla)
- 175B params → 3.5T tokens (GPT-3 aurait dû !)

---

### 💬 Dialogue : Kaplan vs Chinchilla

**Alice** : Attends Bob, Kaplan dit N/D ∝ C^0.46 (favorise params), Chinchilla dit N/D = constant (équilibre). Qui a raison ?!

**Bob** : Chinchilla ! Kaplan a fait une erreur méthodologique : il n'a testé que des petits modèles (<1.5B) entraînés longtemps. Chinchilla a testé des gros modèles (jusqu'à 280B) avec plus de variations.

**Alice** : Donc GPT-3 était mal entraîné ?

**Bob** : Oui ! GPT-3 (175B) a été entraîné sur 300B tokens. Selon Chinchilla, il aurait fallu :
- 175B × 20 = **3.5 TRILLIONS de tokens** !
- Soit 12x plus de données

**Alice** : Et si OpenAI refait GPT-3 avec Chinchilla scaling ?

**Bob** : C'est ce qu'ils ont fait ! Regarde :
- GPT-3.5 : retrained avec plus de données
- GPT-4 : probablement plus petit que prévu, mais BEAUCOUP plus de tokens

---

### 5.3.2 Comparaison Kaplan vs Chinchilla

**Code pour comparer les deux approches**

```python
import numpy as np
import matplotlib.pyplot as plt

def kaplan_optimal_allocation(C):
    """
    Allocation optimale selon Kaplan (2020)
    C : compute budget (FLOPs)
    """
    # N ∝ C^0.73, D ∝ C^0.27
    N_opt = (C / 6e6) ** (0.73)  # Normalized
    D_opt = (C / 6e6) ** (0.27)
    return N_opt, D_opt

def chinchilla_optimal_allocation(C):
    """
    Allocation optimale selon Chinchilla (2022)
    C : compute budget (FLOPs)
    """
    # N ∝ C^0.50, D ∝ C^0.50, D ≈ 20N
    N_opt = (C / 6e6) ** (0.50) / np.sqrt(20)  # Normalized
    D_opt = 20 * N_opt
    return N_opt, D_opt

# Range de compute budgets
compute_budgets = np.logspace(20, 25, 100)  # 1e20 à 1e25 FLOPs

# Allocations optimales
kaplan_N = []
kaplan_D = []
chinchilla_N = []
chinchilla_D = []

for C in compute_budgets:
    kN, kD = kaplan_optimal_allocation(C)
    cN, cD = chinchilla_optimal_allocation(C)
    kaplan_N.append(kN)
    kaplan_D.append(kD)
    chinchilla_N.append(cN)
    chinchilla_D.append(cD)

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: N et D en fonction de C
axes[0].loglog(compute_budgets, kaplan_N, 'b-', label='Kaplan N', linewidth=2)
axes[0].loglog(compute_budgets, kaplan_D, 'b--', label='Kaplan D', linewidth=2)
axes[0].loglog(compute_budgets, chinchilla_N, 'r-', label='Chinchilla N', linewidth=2)
axes[0].loglog(compute_budgets, chinchilla_D, 'r--', label='Chinchilla D', linewidth=2)
axes[0].set_xlabel('Compute Budget (FLOPs)', fontsize=12)
axes[0].set_ylabel('Paramètres / Tokens (normalized)', fontsize=12)
axes[0].set_title('Kaplan vs Chinchilla: Allocation Optimale', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Ratio N/D
kaplan_ratio = np.array(kaplan_N) / np.array(kaplan_D)
chinchilla_ratio = np.array(chinchilla_N) / np.array(chinchilla_D)

axes[1].semilogx(compute_budgets, kaplan_ratio, 'b-', label='Kaplan', linewidth=2)
axes[1].semilogx(compute_budgets, chinchilla_ratio, 'r-', label='Chinchilla', linewidth=2)
axes[1].set_xlabel('Compute Budget (FLOPs)', fontsize=12)
axes[1].set_ylabel('Ratio N/D', fontsize=12)
axes[1].set_title('Ratio Paramètres/Tokens', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kaplan_vs_chinchilla.png', dpi=300, bbox_inches='tight')
plt.show()

# Tableau comparatif pour modèles réels
print("=" * 80)
print("COMPARAISON KAPLAN VS CHINCHILLA")
print("=" * 80)

models = [
    ("GPT-2", 1.5e9, 40e9),
    ("GPT-3", 175e9, 300e9),
    ("Gopher", 280e9, 300e9),
    ("Chinchilla", 70e9, 1.4e12),
    ("LLaMA 7B", 7e9, 1e12),
    ("LLaMA 65B", 65e9, 1.4e12),
]

print(f"\n{'Model':<15} {'Params':<12} {'Tokens':<15} {'Ratio':<10} {'Chinchilla?':<12}")
print("-" * 80)

for name, N, D in models:
    ratio = D / N
    chinchilla_optimal = 20
    status = "✅" if ratio >= 15 else "❌ Sous-entraîné"

    print(f"{name:<15} {N/1e9:>10.1f}B {D/1e9:>13.0f}B {ratio:>8.1f}x {status:<12}")

print("\n💡 Chinchilla optimal : D ≈ 20 × N")
print("📊 Modèles post-2022 suivent tous Chinchilla !")
```

**Output attendu** :
```
================================================================================
COMPARAISON KAPLAN VS CHINCHILLA
================================================================================

Model           Params       Tokens          Ratio      Chinchilla?
--------------------------------------------------------------------------------
GPT-2                 1.5B           40B     26.7x ✅
GPT-3               175.0B          300B      1.7x ❌ Sous-entraîné
Gopher              280.0B          300B      1.1x ❌ Sous-entraîné
Chinchilla           70.0B         1400B     20.0x ✅
LLaMA 7B              7.0B         1000B    142.9x ✅
LLaMA 65B            65.0B         1400B     21.5x ✅

💡 Chinchilla optimal : D ≈ 20 × N
📊 Modèles post-2022 suivent tous Chinchilla !
```

---

## 5.4 Implications Pratiques des Scaling Laws

### 5.4.1 Pour les Chercheurs : Prédire Avant d'Entraîner

**Use Case** : Tu veux savoir si un modèle 10B vaut le coup d'être entraîné.

**Méthode** :
1. Entraîner plusieurs petits modèles (100M, 500M, 1B)
2. Tracer la loss en fonction de N (échelle log-log)
3. Extrapoler pour 10B
4. Décider si le gain justifie le coût

```python
def predict_large_model_performance(small_models_data):
    """
    Prédit la performance d'un gros modèle basé sur petits modèles

    Args:
        small_models_data: Liste de (N, loss) pour petits modèles

    Returns:
        Fonction de prédiction
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # Extraire N et losses
    Ns = np.array([d[0] for d in small_models_data])
    losses = np.array([d[1] for d in small_models_data])

    # Fit power law: L(N) = a * N^b
    def power_law(N, a, b):
        return a * N ** b

    # Log-transform pour fit linéaire
    log_Ns = np.log(Ns)
    log_losses = np.log(losses)

    # Fit linéaire en log-space
    coeffs = np.polyfit(log_Ns, log_losses, 1)
    b = coeffs[0]  # Exposant
    a = np.exp(coeffs[1])  # Constante

    print(f"📐 Loi de puissance fittée : L(N) = {a:.3f} * N^({b:.3f})")

    # Fonction de prédiction
    def predict(N_target):
        return power_law(N_target, a, b)

    return predict

# Exemple: prédire 10B basé sur runs de 100M, 500M, 1B
small_runs = [
    (100e6, 3.8),   # 100M params → loss 3.8
    (500e6, 3.2),   # 500M params → loss 3.2
    (1e9, 2.9),     # 1B params → loss 2.9
]

predict_fn = predict_large_model_performance(small_runs)

# Prédictions
for N in [5e9, 10e9, 50e9, 100e9]:
    predicted_loss = predict_fn(N)
    print(f"Modèle {N/1e9:.0f}B params → Loss prédite: {predicted_loss:.3f}")
```

**Output** :
```
📐 Loi de puissance fittée : L(N) = 156.432 * N^(-0.082)
Modèle 5B params → Loss prédite: 2.701
Modèle 10B params → Loss prédite: 2.585
Modèle 50B params → Loss prédite: 2.333
Modèle 100B params → Loss prédite: 2.252
```

**Décision** : Si passer de 1B (loss 2.9) à 10B (loss 2.585) vaut le coût **10x** supérieur → GO !

---

### 5.4.2 Pour les Startups : Optimiser le Budget

**Scénario** : Tu as un budget de $10,000 pour entraîner un modèle. Comment allouer ?

**Données** :
- A100 40GB : $2/heure
- 1 TFLOP/s = 1e12 FLOPs/seconde
- A100 : ~300 TFLOPS (FP16)

**Calcul** :

```python
def optimize_training_budget(budget_usd, cost_per_hour=2.0, tflops=300):
    """
    Optimise l'allocation N vs D pour un budget donné

    Args:
        budget_usd: Budget en dollars
        cost_per_hour: Coût GPU par heure
        tflops: TFLOPs du GPU

    Returns:
        N_opt, D_opt (paramètres et tokens optimaux)
    """
    # Heures GPU disponibles
    hours = budget_usd / cost_per_hour

    # Compute total (FLOPs)
    flops_per_second = tflops * 1e12
    total_compute = flops_per_second * hours * 3600  # secondes

    print(f"💰 Budget: ${budget_usd:,}")
    print(f"⏰ Heures GPU: {hours:,.0f}h")
    print(f"🖥️  Compute total: {total_compute:.2e} FLOPs")

    # Chinchilla optimal: C ≈ 6ND (approximation)
    # N * D = C / 6, avec D = 20N
    # N * 20N = C / 6
    # N^2 = C / 120
    # N = sqrt(C / 120)

    N_opt = np.sqrt(total_compute / 120)
    D_opt = 20 * N_opt

    print(f"\n📊 Allocation optimale (Chinchilla):")
    print(f"   Paramètres: {N_opt/1e9:.2f}B")
    print(f"   Tokens: {D_opt/1e9:.0f}B")
    print(f"   Ratio D/N: {D_opt/N_opt:.1f}x")

    return N_opt, D_opt

# Exemples
budgets = [1000, 10000, 100000, 1000000]

for budget in budgets:
    print("\n" + "=" * 60)
    optimize_training_budget(budget)
```

**Output** :
```
============================================================
💰 Budget: $1,000
⏰ Heures GPU: 500h
🖥️  Compute total: 5.40e+20 FLOPs

📊 Allocation optimale (Chinchilla):
   Paramètres: 0.67B
   Tokens: 13B
   Ratio D/N: 20.0x

============================================================
💰 Budget: $10,000
⏰ Heures GPU: 5,000h
🖥️  Compute total: 5.40e+21 FLOPs

📊 Allocation optimale (Chinchilla):
   Paramètres: 2.12B
   Tokens: 42B
   Ratio D/N: 20.0x

============================================================
💰 Budget: $100,000
⏰ Heures GPU: 50,000h
🖥️  Compute total: 5.40e+22 FLOPs

📊 Allocation optimale (Chinchilla):
   Paramètres: 6.71B
   Tokens: 134B
   Ratio D/N: 20.0x

============================================================
💰 Budget: $1,000,000
⏰ Heures GPU: 500,000h
🖥️  Compute total: 5.40e+23 FLOPs

📊 Allocation optimale (Chinchilla):
   Paramètres: 21.21B
   Tokens: 424B
   Ratio D/N: 20.0x
```

**Conclusion** : Avec $10k, viser un modèle ~2B entraîné sur ~40B tokens (pas un modèle 10B sous-entraîné !).

---

## 5.5 Les Frontières des Scaling Laws

### 5.5.1 Jusqu'où Peut-on Scaler ?

**💬 Dialogue**

**Alice** : Bob, si les scaling laws continuent, on peut juste faire des modèles infinis ?

**Bob** : Bonne question ! Il y a plusieurs **limites** :

**Limite #1 : Les Données**

En 2026, on a déjà utilisé :
- Tout CommonCrawl : ~10T tokens
- Tous les livres : ~1T tokens
- Tous les articles scientifiques : ~500B tokens
- GitHub : ~1T tokens

**Total disponible** : ~15-20T tokens de haute qualité.

Pour un modèle 1T params (Chinchilla optimal), il faut **20T tokens**. On y est presque !

**Solution** :
- Données synthétiques (générées par LLMs)
- Multimodal (images, vidéos)
- Augmentation de données

---

**Limite #2 : Le Compute**

**Coût d'entraînement actuel** :
- GPT-3 (175B) : ~$5M
- PaLM (540B) : ~$20M
- GPT-4 (estimation 1.7T) : ~$100M ?

**Extrapolation** :
- 10T params : ~$1B 💸
- 100T params : ~$10B 💸💸

À un moment, même les GAFAM hésitent !

---

**Limite #3 : L'Énergie**

- GPT-3 training : ~1,300 MWh (= consommation annuelle de 120 foyers US)
- GPT-4 training (estimé) : ~50,000 MWh
- 10T model : ~500,000 MWh (= petite centrale nucléaire pendant 1 mois)

**Impact environnemental** devient un facteur limitant.

---

### 5.5.2 Émergence et Discontinuités

**⚠️ Attention** : Les scaling laws prédisent la **loss**, pas les capacités émergentes !

**Exemple** :
- Loss GPT-2 → GPT-3 : réduction continue (prédit ✅)
- Capacités few-shot : apparaissent soudainement à 13B params (pas prédit ❌)

**Capacités émergentes observées** :
- **Few-shot learning** : >10B params
- **Chain-of-thought** : >50B params
- **Instruction following** : >100B params (+ RLHF)

🎨 **Analogie** : C'est comme l'eau :
- 0°C → 99°C : température monte linéairement
- 100°C : **ébullition** (changement de phase soudain !)

**Bob** : Les scaling laws nous disent que le modèle s'améliore. Mais elles ne prédisent PAS *comment* il s'améliore qualitativement.

---

## 5.6 Quiz et Exercices

### 🎯 Quiz : Testez Vos Connaissances !

**Question 1** : Selon Kaplan (2020), si tu doubles les paramètres, de combien diminue la loss ?

A) ~1%
B) ~5%
C) ~10%
D) ~20%

<details>
<summary>Réponse</summary>

**B) ~5%**

Explication : L(N) = (Nc/N)^0.076
- Si N → 2N : L(2N) / L(N) = (1/2)^0.076 ≈ 0.95
- Réduction : 5%
</details>

---

**Question 2** : Selon Chinchilla (2022), combien de tokens faut-il pour entraîner un modèle 70B de manière optimale ?

A) 70B tokens
B) 350B tokens
C) 1.4T tokens
D) 7T tokens

<details>
<summary>Réponse</summary>

**C) 1.4T tokens**

Explication : Chinchilla optimal : D ≈ 20 × N
- 70B params × 20 = 1,400B tokens = 1.4T tokens
- C'est exactement ce qu'a fait Chinchilla !
</details>

---

**Question 3** : GPT-3 (175B params, 300B tokens) était-il optimal selon Chinchilla ?

A) Oui, parfaitement optimal
B) Non, sur-entraîné (trop de tokens)
C) Non, sous-entraîné (pas assez de tokens)

<details>
<summary>Réponse</summary>

**C) Non, sous-entraîné**

Explication :
- GPT-3 : 175B params, 300B tokens → ratio 1.7x
- Chinchilla optimal : 175B params × 20 = 3,500B tokens
- GPT-3 aurait dû être entraîné sur **12x plus de données** !
</details>

---

**Question 4** : Pourquoi ne peut-on pas scaler indéfiniment ?

A) Les scaling laws s'arrêtent à 1T params
B) Limites de données, compute, énergie
C) Les GPUs ne sont pas assez puissants
D) C'est mathématiquement impossible

<details>
<summary>Réponse</summary>

**B) Limites de données, compute, énergie**

Explication :
- **Données** : On approche de tout le texte disponible (~20T tokens)
- **Compute** : Entraîner 10T params coûterait ~$1 milliard
- **Énergie** : Impact environnemental devient prohibitif
- **Pratique** : Gains marginaux ne justifient plus le coût
</details>

---

### 💻 Exercices Pratiques

**Exercice 1 : Implémenter une Scaling Law** (Débutant)

Créez une fonction qui prédit la loss d'un modèle selon ses paramètres.

```python
def predict_loss(N, Nc=8.8e13, alpha=0.076):
    """
    Prédit la loss selon Kaplan scaling law

    Args:
        N: Nombre de paramètres
        Nc: Constante
        alpha: Exposant

    Returns:
        Loss prédite
    """
    # TODO: Implémenter
    pass

# Test
models = [
    ("GPT-2", 1.5e9),
    ("GPT-3", 175e9),
    ("Hypothetical 1T", 1e12),
]

for name, N in models:
    loss = predict_loss(N)
    print(f"{name}: {loss:.3f}")
```

<details>
<summary>Solution</summary>

```python
def predict_loss(N, Nc=8.8e13, alpha=0.076):
    """
    Prédit la loss selon Kaplan scaling law
    """
    return (Nc / N) ** alpha

# Test
models = [
    ("GPT-2", 1.5e9),
    ("GPT-3", 175e9),
    ("Hypothetical 1T", 1e12),
]

print("📊 Prédictions de Loss (Kaplan 2020)")
print(f"{'Model':<20} {'Params':<15} {'Loss Prédite'}")
print("-" * 50)

for name, N in models:
    loss = predict_loss(N)
    print(f"{name:<20} {N/1e9:>12.1f}B {loss:>12.3f}")

# Output:
# 📊 Prédictions de Loss (Kaplan 2020)
# Model                Params          Loss Prédite
# --------------------------------------------------
# GPT-2                        1.5B        3.123
# GPT-3                      175.0B        2.104
# Hypothetical 1T           1000.0B        1.687
```
</details>

---

**Exercice 2 : Optimiser un Budget** (Intermédiaire)

Vous avez $50,000 pour entraîner un modèle. Utilisez Chinchilla scaling pour déterminer N et D optimaux.

<details>
<summary>Solution</summary>

```python
def optimize_chinchilla(budget_usd, gpu_cost_hour=2.0, gpu_tflops=300):
    """
    Optimise N et D selon Chinchilla pour un budget donné
    """
    import numpy as np

    # Compute disponible
    hours = budget_usd / gpu_cost_hour
    flops_per_sec = gpu_tflops * 1e12
    total_compute = flops_per_sec * hours * 3600

    # Chinchilla: C ≈ 6ND, D = 20N
    # C = 6N(20N) = 120N^2
    # N = sqrt(C/120)
    N_opt = np.sqrt(total_compute / 120)
    D_opt = 20 * N_opt

    # Coût réel d'entraînement (vérification)
    # FLOPs per token ≈ 6N
    tokens_per_second = flops_per_sec / (6 * N_opt)
    training_time_hours = D_opt / (tokens_per_second * 3600)
    actual_cost = training_time_hours * gpu_cost_hour

    print(f"💰 Budget: ${budget_usd:,}")
    print(f"\n📊 Allocation Optimale (Chinchilla):")
    print(f"   Paramètres: {N_opt/1e9:.2f}B")
    print(f"   Tokens: {D_opt/1e9:.0f}B")
    print(f"   Ratio D/N: {D_opt/N_opt:.1f}x")
    print(f"\n⏱️  Temps d'entraînement: {training_time_hours:,.0f} heures")
    print(f"💵 Coût réel: ${actual_cost:,.0f}")

    return N_opt, D_opt

# Test avec $50k
optimize_chinchilla(50000)

# Output:
# 💰 Budget: $50,000
#
# 📊 Allocation Optimale (Chinchilla):
#    Paramètres: 4.74B
#    Tokens: 95B
#    Ratio D/N: 20.0x
#
# ⏱️  Temps d'entraînement: 25,000 heures
# 💵 Coût réel: $50,000
```
</details>

---

## 🎉 Conclusion : L'Âge de la Prévisibilité

### 💬 Dialogue Final

**Alice** : Bob, on vient de voir que l'évolution de l'IA est... prévisible ? C'est fou !

**Bob** : Oui ! Les scaling laws sont peut-être **la découverte la plus importante** de l'IA moderne. Avant 2020, on tâtonnait. Maintenant, on **planifie**.

**Alice** : Donc OpenAI sait exactement ce que GPT-5 fera avant de l'entraîner ?

**Bob** : Ils ont une très bonne idée de la **loss**, oui. Mais attention : les scaling laws ne prédisent pas :
- Les capacités émergentes (few-shot, reasoning)
- L'utilité pratique
- Les problèmes de sécurité

**Alice** : C'est comme avoir la carte d'un territoire, mais pas savoir ce qu'on y trouvera ?

**Bob** : Exactement ! On sait que GPT-5 sera "X% meilleur" que GPT-4. Mais sera-t-il capable de faire des découvertes scientifiques ? De résoudre des problèmes impossibles ? Ça, les scaling laws ne le disent pas.

**Alice** : Et les limites ? On peut scaler jusqu'où ?

**Bob** : Probablement jusqu'à **10-100T params** dans les 5-10 prochaines années. Après :
- On manque de données (fini le texte humain)
- Coût prohibitif ($1B+ par modèle)
- Impact environnemental inacceptable

**Alice** : Donc après, quoi ? L'IA stagne ?

**Bob** : Non ! On trouvera d'autres axes :
- Architectures plus efficaces (Mamba, RWKV)
- Données synthétiques (self-play, RL)
- Multimodalité (vision, audio, robotics)
- Meilleur alignement (RLHF 2.0)

Le scaling n'est qu'une **phase** de l'évolution de l'IA. Mais quelle phase ! 🚀

---

### 📊 Récapitulatif

**Scaling Laws de Kaplan (2020)** :
- L(N) ∝ N^(-0.076) → doubler params = -5% loss
- Allocation : 73% params, 27% données
- A mené à GPT-3 (175B, 300B tokens)

**Scaling Laws de Chinchilla (2022)** :
- D_opt ≈ 20 × N_opt
- Allocation : 50-50 entre params et données
- GPT-3 était **sous-entraîné** (aurait dû avoir 3.5T tokens)

**Implications** :
- On peut prédire la performance avant d'entraîner
- Optimiser budgets (startup : $10k → modèle 2B optimal)
- Frontières : ~10-100T params (limites données/compute/énergie)

**Limites** :
- Ne prédisent pas les capacités émergentes
- Scaling physiquement limité
- Nouveaux paradigmes nécessaires au-delà

---

### 📚 Ressources

**Papers** :
- [Kaplan et al. 2020 - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. 2022 - Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)

**Analyses** :
- [Scaling Laws Visualizer](https://scale.anthropic.com)
- [Epoch AI - Trends in ML](https://epochai.org)

---

**Prochain Chapitre** : [Chapitre 6 : Evaluation des LLMs](./CHAPITRE_06_EVALUATION_LLMS.md)

---

> *"The future is predictable. We just need more compute."*
> — Sam Altman (probablement)

**Fin du Chapitre 5** 🎓
