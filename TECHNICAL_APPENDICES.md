# 📚 ANNEXES TECHNIQUES
## Bible du Développeur AI/LLM 2026

---

# ANNEXE A : FORMULAIRE MATHÉMATIQUE

## A.1 Attention Mechanism

### **Scaled Dot-Product Attention**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

où:
- Q ∈ ℝ^(n×d_k) : Query matrix
- K ∈ ℝ^(m×d_k) : Key matrix
- V ∈ ℝ^(m×d_v) : Value matrix
- d_k : dimension des keys
- n : longueur de la séquence query
- m : longueur de la séquence key
```

### **Multi-Head Attention**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

où head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Paramètres:
- W^Q_i ∈ ℝ^(d_model×d_k)
- W^K_i ∈ ℝ^(d_model×d_k)
- W^V_i ∈ ℝ^(d_model×d_v)
- W^O ∈ ℝ^(hd_v×d_model)
- h : nombre de heads
- d_k = d_v = d_model/h
```

### **Self-Attention (cas particulier)**
```
SelfAttention(X) = Attention(XW^Q, XW^K, XW^V)
où X ∈ ℝ^(n×d_model)
```

### **Causal Attention (masking)**
```
M_{ij} = {
  0   si i >= j (autoriser attention)
  -∞  si i < j  (masquer le futur)
}

Attention_causal(Q, K, V) = softmax((QK^T / √d_k) + M) V
```

## A.2 Positional Encoding

### **Sinusoidal Positional Encoding (Vaswani et al.)**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

où:
- pos : position dans la séquence
- i : dimension index
- d_model : dimension du modèle
```

### **Rotary Position Embedding (RoPE)**
```
RoPE(x_m, m) = [
  cos(mθ_i)  -sin(mθ_i)
  sin(mθ_i)   cos(mθ_i)
] [x_{2i}]
  [x_{2i+1}]

où θ_i = 10000^(-2i/d)
```

### **ALiBi (Attention with Linear Biases)**
```
softmax(q_i K^T + m·(i-j))

où m est un slope spécifique à chaque head
```

## A.3 Loss Functions

### **Cross-Entropy Loss (Language Modeling)**
```
L_CE = -∑_{i=1}^{V} y_i log(ŷ_i)

Pour un batch:
L = -1/N ∑_{n=1}^{N} ∑_{i=1}^{V} y_{n,i} log(ŷ_{n,i})

où:
- V : taille du vocabulaire
- N : batch size
- y : one-hot encoded target
- ŷ : predicted probabilities
```

### **Perplexity**
```
Perplexity = exp(L_CE) = exp(-1/N ∑ log P(x_i))

Interprétation: "En moyenne, le modèle hésite entre perplexity choix"
```

### **KL Divergence**
```
D_KL(P || Q) = ∑_x P(x) log(P(x)/Q(x))

Utilisé dans:
- RLHF (contrainte KL avec policy originale)
- Distillation (match distributions teacher-student)
```

## A.4 Optimization

### **Gradient Descent**
```
θ_{t+1} = θ_t - η ∇_θ L(θ_t)

où:
- θ : paramètres
- η : learning rate
- ∇_θ L : gradient de la loss
```

### **SGD with Momentum**
```
v_t = βv_{t-1} + ∇_θ L(θ_t)
θ_{t+1} = θ_t - η v_t

où β ∈ [0,1] (typiquement 0.9)
```

### **Adam Optimizer**
```
m_t = β_1 m_{t-1} + (1-β_1) g_t              # 1st moment
v_t = β_2 v_{t-1} + (1-β_2) g_t^2            # 2nd moment

m̂_t = m_t / (1-β_1^t)                        # bias correction
v̂_t = v_t / (1-β_2^t)

θ_{t+1} = θ_t - η m̂_t / (√v̂_t + ε)

où:
- β_1 = 0.9 (typiquement)
- β_2 = 0.999
- ε = 1e-8
- g_t : gradient au temps t
```

### **AdamW (Adam with decoupled Weight Decay)**
```
θ_{t+1} = θ_t - η (m̂_t / (√v̂_t + ε) + λθ_t)

où λ est le coefficient de weight decay (typiquement 0.1)
```

### **Learning Rate Schedules**

**Linear Warmup:**
```
η(t) = η_max · min(1, t/t_warmup)
```

**Cosine Decay:**
```
η(t) = η_min + 0.5(η_max - η_min)(1 + cos(πt/T))

où T est le nombre total de steps
```

**Inverse Square Root:**
```
η(t) = η_0 · min(1/√t, t/t_warmup^(3/2))
```

## A.5 Normalization

### **Layer Normalization**
```
LN(x) = γ ⊙ (x - μ)/√(σ^2 + ε) + β

où:
- μ = 1/d ∑_{i=1}^d x_i
- σ^2 = 1/d ∑_{i=1}^d (x_i - μ)^2
- γ, β : paramètres apprenables
- d : feature dimension
```

### **RMSNorm (Root Mean Square Norm)**
```
RMSNorm(x) = x / RMS(x) · γ

où RMS(x) = √(1/d ∑_{i=1}^d x_i^2)
```

## A.6 Information Theory

### **Entropy (Shannon)**
```
H(X) = -∑_x P(x) log P(x)

Unités: bits (log base 2) ou nats (log naturel)
```

### **Cross-Entropy**
```
H(P, Q) = -∑_x P(x) log Q(x)
```

### **Mutual Information**
```
I(X; Y) = H(X) + H(Y) - H(X,Y)
        = ∑∑ P(x,y) log(P(x,y)/(P(x)P(y)))
```

## A.7 Scaling Laws

### **Kaplan Scaling Laws (OpenAI, 2020)**
```
L(N, D) = (N_c/N)^α_N + (D_c/D)^α_D

où:
- L : loss
- N : nombre de paramètres
- D : taille du dataset (tokens)
- N_c, D_c, α_N, α_D : constantes empiriques
```

### **Chinchilla Scaling (DeepMind, 2022)**
```
N_optimal ∝ C^0.50
D_optimal ∝ C^0.50

où C est le compute budget (FLOPs)

Règle: Pour compute optimal, utiliser autant de tokens que de paramètres
Exemple: modèle 70B → entraîner sur 70B tokens (minimum)
```

### **Calcul de FLOPs pour Training**
```
FLOPs ≈ 6ND

où:
- N : nombre de paramètres
- D : nombre de tokens

Pour un forward pass:
FLOPs_forward ≈ 2ND

Pour backward pass (2x forward):
FLOPs_backward ≈ 4ND

Total: 6ND
```

## A.8 Fine-tuning

### **LoRA (Low-Rank Adaptation)**
```
W' = W_0 + ΔW = W_0 + BA

où:
- W_0 ∈ ℝ^(d×k) : poids pré-entraînés (frozen)
- B ∈ ℝ^(d×r) : down-projection (trainable)
- A ∈ ℝ^(r×k) : up-projection (trainable)
- r << min(d,k) : rank (typiquement 8, 16, 32)

Nombre de paramètres:
- Original: d × k
- LoRA: r(d + k)
- Réduction: ~1000x si r=8, d=k=4096
```

### **LoRA scaling**
```
h = W_0 x + (α/r) BA x

où α est un hyperparamètre de scaling (souvent α = r)
```

## A.9 RLHF

### **Reward Model**
```
r_θ(x, y) : score de qualité de la réponse y à la question x

Loss (Bradley-Terry):
L_R(θ) = -E_{(x,y_w,y_l)} [log σ(r_θ(x,y_w) - r_θ(x,y_l))]

où:
- y_w : réponse préférée (winner)
- y_l : réponse rejetée (loser)
- σ : sigmoid
```

### **PPO (Proximal Policy Optimization)**
```
L^{CLIP}(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

où:
- r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
- Â_t : advantage estimé
- ε : clip range (typiquement 0.2)

Objectif LLM complet:
L_PPO(θ) = E[r_θ(x,y) - β·D_KL(π_θ || π_ref)]

où:
- r_θ : reward model
- β : coefficient KL (typiquement 0.01-0.1)
- π_ref : policy de référence (SFT)
```

### **DPO (Direct Preference Optimization)**
```
L_DPO(θ) = -E_{(x,y_w,y_l)} [log σ(β log(π_θ(y_w|x)/π_ref(y_w|x))
                                    - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

Avantage: pas besoin d'entraîner reward model séparé
```

---

# ANNEXE B : MÉTRIQUES & BENCHMARKS

## B.1 Métriques de Génération de Texte

### **BLEU (Bilingual Evaluation Understudy)**
```
BLEU = BP · exp(∑_{n=1}^N w_n log p_n)

où:
- p_n : precision des n-grams
- BP : brevity penalty = min(1, exp(1 - r/c))
- r : longueur référence
- c : longueur candidate
- N : max n-gram order (typiquement 4)

Limites:
- Ne capture pas sémantique
- Sensible à l'ordre des mots
- Peu utilisé pour LLMs modernes
```

### **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
```
ROUGE-N = ∑_{S∈Refs} ∑_{gram_n∈S} Count_match(gram_n) /
          ∑_{S∈Refs} ∑_{gram_n∈S} Count(gram_n)

ROUGE-L : basé sur longest common subsequence

Utilisé pour: résumés, génération de texte
```

### **BERTScore**
```
R_BERT = 1/|x| ∑_{x_i∈x} max_{ŷ_j∈ŷ} x_i^T ŷ_j
P_BERT = 1/|ŷ| ∑_{ŷ_j∈ŷ} max_{x_i∈x} x_i^T ŷ_j
F_BERT = 2·P_BERT·R_BERT / (P_BERT + R_BERT)

où x_i, ŷ_j sont des embeddings BERT

Avantage: capture similarité sémantique
```

## B.2 Benchmarks pour LLMs

### **MMLU (Massive Multitask Language Understanding)**
- **57 tâches** (STEM, humanities, social sciences, etc.)
- **Format**: QCM (4 choix)
- **Métrique**: Accuracy (%)
- **SOTA (2026)**: GPT-4: 86.4%, Claude 3 Opus: 86.8%

### **HellaSwag (Commonsense Reasoning)**
- **Tâche**: Sentence completion
- **Format**: 4 choix
- **Métrique**: Accuracy (%)
- **SOTA**: ~95% (GPT-4, Claude 3)

### **TruthfulQA**
- **Tâche**: Répondre de manière factuelle (éviter hallucinations)
- **Format**: QA
- **Métrique**: % réponses vraies
- **Difficulté**: Même humans ~90%

### **GSM8K (Grade School Math)**
- **8,500 problèmes** de mathématiques niveau primaire
- **Format**: Question → réponse numérique
- **Métrique**: Exact match (%)
- **SOTA**: GPT-4: 92%, o1: 95%+

### **HumanEval (Code Generation)**
- **164 problèmes** de programmation Python
- **Format**: Docstring → fonction complète
- **Métrique**: pass@k (% qui passent tests unitaires)
- **SOTA**: GPT-4: 67% (pass@1), Codex: 72%, AlphaCode: 50%

### **MATH (Competition Mathematics)**
- **12,500 problèmes** niveau compétition
- **Format**: LaTeX → réponse numérique
- **Métrique**: Accuracy (%)
- **SOTA**: GPT-4: 42.5%, Minerva: 50.3%

### **BBHard (BIG-Bench Hard)**
- **23 tâches** difficiles de BIG-Bench
- **Tâches** où CoT aide significativement
- **Métrique**: Accuracy moyenne
- **SOTA**: GPT-4: 86%, PaLM 2: 78%

### **MT-Bench (Multi-Turn Conversations)**
- **80 conversations** multi-tours
- **Catégories**: Writing, Roleplay, Reasoning, Math, Coding, STEM, Humanities
- **Métrique**: Score 1-10 (GPT-4 as judge)
- **SOTA**: GPT-4-Turbo: 9.32, Claude 3 Opus: 9.18

### **AlpacaEval (Instruction Following)**
- **805 instructions** diverses
- **Format**: Instruction → réponse
- **Métrique**: % win vs référence (GPT-4 as judge)
- **SOTA**: GPT-4: 95%, Claude 3: 91%

## B.3 Métriques RAG

### **Retrieval Metrics**

**Recall@k**
```
Recall@k = |relevant docs in top-k| / |total relevant docs|
```

**Precision@k**
```
Precision@k = |relevant docs in top-k| / k
```

**MRR (Mean Reciprocal Rank)**
```
MRR = 1/|Q| ∑_{i=1}^{|Q|} 1/rank_i

où rank_i est le rang du premier doc pertinent pour query i
```

**NDCG (Normalized Discounted Cumulative Gain)**
```
DCG@k = ∑_{i=1}^k (2^{rel_i} - 1) / log_2(i+1)
NDCG@k = DCG@k / IDCG@k

où IDCG est le DCG idéal (ordre optimal)
```

### **Generation Metrics (RAGAS Framework)**

**Faithfulness**
```
Faithfulness = |statements supportés| / |total statements|

Vérifie si la génération est ancrée dans les documents récupérés
```

**Answer Relevancy**
```
Relevancy = similarité_cosine(question, generated_answer)

Utilise embeddings pour mesurer pertinence
```

**Context Precision**
```
Precision = ∑_{k=1}^K (P(k) × rel(k)) / |relevant docs|
```

**Context Recall**
```
Recall = |ground_truth claims in context| / |total ground_truth claims|
```

## B.4 Métriques d'Efficacité

### **Latency**
- **Time to First Token (TTFT)**: Temps avant premier token généré
- **Inter-Token Latency (ITL)**: Temps entre tokens
- **Total Latency**: Temps total génération

**Cibles production**:
- TTFT < 500ms (conversational)
- ITL < 50ms
- Total pour 100 tokens < 5s

### **Throughput**
```
Throughput = nombre de tokens générés / seconde

Batch throughput: tokens/sec avec batching
```

### **Memory Usage**
```
Memory (inference) ≈ 2 × N bytes (FP16)

Exemple:
- 7B params → 14GB VRAM (FP16)
- 13B params → 26GB VRAM
- 70B params → 140GB VRAM

Avec quantization (INT8):
- 7B → 7GB
- 70B → 70GB
```

### **Cost Metrics**
```
Cost per 1M tokens = (inference_time × GPU_cost_per_hour) / throughput

Exemple vLLM sur A100:
- Llama 2 7B: ~$0.20/1M tokens
- Llama 2 70B: ~$2.00/1M tokens
```

## B.5 Comparaison Modèles (2026)

| Modèle | Params | MMLU | HumanEval | Latence (ms/token) | Coût ($/1M tok) |
|--------|--------|------|-----------|-------------------|----------------|
| **GPT-4 Turbo** | ? | 86.4 | 67.0 | 50 | $10.00 |
| **GPT-4o** | ? | 87.2 | 72.0 | 30 | $5.00 |
| **Claude 3 Opus** | ? | 86.8 | 84.9 | 45 | $15.00 |
| **Claude 3.5 Sonnet** | ? | 88.7 | 92.0 | 35 | $3.00 |
| **Gemini 1.5 Pro** | ? | 85.9 | 71.9 | 40 | $7.00 |
| **Llama 3.1 405B** | 405B | 85.2 | 61.0 | 80 | $3.50 |
| **Llama 3.3 70B** | 70B | 82.0 | 58.0 | 25 | $0.60 |
| **Llama 3 8B** | 8B | 68.4 | 48.1 | 8 | $0.10 |
| **Mistral Large** | ? | 81.2 | 45.1 | 35 | $4.00 |
| **DeepSeek-V3** | 671B | 88.5 | 65.0 | 90 | $0.50 |
| **Qwen 2.5 72B** | 72B | 84.2 | 56.0 | 28 | $0.80 |

*(Valeurs indicatives 2026)*

---

# ANNEXE C : GLOSSAIRE COMPLET

## A

**Adapter Layers**: Couches supplémentaires entraînables insérées dans un modèle pré-entraîné (PEFT).

**Adversarial Examples**: Inputs conçus pour tromper un modèle.

**Agent**: Système autonome capable d'utiliser des outils et de raisonner.

**ALiBi** (Attention with Linear Biases): Méthode d'encodage positionnel par biais linéaires.

**Alignment**: Processus de rendre un LLM utile, honnête et inoffensif (RLHF, etc.).

**Attention**: Mécanisme permettant à un modèle de se concentrer sur des parties pertinentes de l'input.

**Autoregressive**: Génération séquentielle où chaque token dépend des précédents.

**AWQ** (Activation-aware Weight Quantization): Méthode de quantization préservant précision.

## B

**Backpropagation**: Algorithme de calcul des gradients pour training.

**Batch Size**: Nombre d'exemples traités simultanément.

**BERT** (Bidirectional Encoder Representations from Transformers): Modèle encoder-only pré-entraîné.

**BF16** (Brain Float 16): Format numérique 16-bit optimisé pour ML.

**Bias**: Dans attention, terme additionnel; aussi biais dans les données.

**BPE** (Byte-Pair Encoding): Algorithme de tokenization.

## C

**Causal Attention**: Attention masquée pour prévenir accès au futur (autoregressive).

**Checkpoint**: Sauvegarde de l'état d'un modèle durant training.

**Chinchilla Scaling**: Loi d'échelle optimale (DeepMind 2022).

**Chunking**: Découpage de documents en morceaux pour RAG.

**CLM** (Causal Language Modeling): Objectif d'entraînement autoregressive.

**Constitutional AI**: Méthode d'alignment par principes (Anthropic).

**Context Length**: Nombre maximum de tokens en input.

**Context Window**: Fenêtre de contexte accessible au modèle.

**CoT** (Chain-of-Thought): Prompting incitant au raisonnement étape par étape.

**Cross-Attention**: Attention entre deux séquences différentes.

**Cross-Entropy**: Loss function pour classification/génération.

**CUDA**: Plateforme de calcul parallèle NVIDIA.

## D

**Decoding Strategy**: Méthode de sélection des tokens (greedy, sampling, beam search).

**DeepSpeed**: Bibliothèque d'optimisation de training distribué (Microsoft).

**Deterministic**: Génération reproductible (temperature=0 ou seed fixe).

**Distillation**: Transfer de connaissances d'un grand modèle vers un petit.

**DPO** (Direct Preference Optimization): Alternative à RLHF sans reward model.

**Dropout**: Régularisation par désactivation aléatoire de neurones.

## E

**Embedding**: Représentation vectorielle dense d'un token/mot.

**Encoder-Decoder**: Architecture avec encoder (compréhension) et decoder (génération).

**Epoch**: Une passe complète sur le dataset d'entraînement.

**EOS** (End of Sequence): Token spécial marquant la fin.

## F

**Few-Shot Learning**: Apprentissage avec quelques exemples en contexte.

**Fine-Tuning**: Entraînement additionnel sur données spécifiques.

**Flash Attention**: Implémentation optimisée de l'attention (IO-aware).

**FLOPs**: Floating Point Operations (mesure de compute).

**FP16/FP32**: Float 16-bit / 32-bit precision.

**FSDP** (Fully Sharded Data Parallel): Stratégie de parallélisme (PyTorch).

**Function Calling**: Capacité du LLM à appeler des fonctions externes.

## G

**GELU** (Gaussian Error Linear Unit): Fonction d'activation.

**Gradient Accumulation**: Accumuler gradients sur plusieurs mini-batches.

**Gradient Clipping**: Limiter la norme des gradients.

**GPTQ**: Méthode de quantization post-training.

**Greedy Decoding**: Sélection du token le plus probable à chaque étape.

## H

**Hallucination**: Génération d'informations fausses ou inventées.

**Head** (Attention): Une des têtes d'attention dans multi-head attention.

**Hidden State**: Représentation interne dans les couches du modèle.

**HuggingFace**: Plateforme et bibliothèques pour ML/NLP.

**Hybrid Search**: Combinaison de dense et sparse retrieval.

**Hyperparameter**: Paramètre de configuration (learning rate, batch size, etc.).

## I

**In-Context Learning**: Apprentissage via exemples dans le prompt.

**Inference**: Utilisation du modèle pour faire des prédictions.

**Instruction Tuning**: Fine-tuning sur instructions/tâches variées.

**INT8/INT4**: Quantization 8-bit ou 4-bit integer.

## J

**Jailbreak**: Contournement des guardrails d'un modèle.

**JSON Mode**: Génération structurée en format JSON.

## K

**KL Divergence** (Kullback-Leibler): Mesure de divergence entre distributions.

**KV Cache**: Cache des Keys et Values pour accélérer inference autoregressive.

## L

**Latent Space**: Espace des représentations internes.

**Layer Normalization**: Normalisation par couche.

**Learning Rate**: Taux d'apprentissage pour l'optimiseur.

**LLM** (Large Language Model): Grand modèle de langage.

**LLMOps**: MLOps appliqué aux LLMs.

**LoRA** (Low-Rank Adaptation): PEFT par matrices low-rank.

**Loss**: Fonction de coût à minimiser.

**LSH** (Locality-Sensitive Hashing): Hashing pour recherche approximative.

## M

**Mamba**: Architecture State Space Model (alternative aux transformers).

**Masked Language Modeling**: Prédire tokens masqués (BERT).

**Maximum Likelihood**: Principe d'optimisation statistique.

**Memory (Agent)**: Système de mémoire court/long terme pour agents.

**MLP** (Multi-Layer Perceptron): Réseau fully-connected.

**MMLU**: Benchmark multitâche.

**MoE** (Mixture of Experts): Architecture avec routage vers experts.

**Multi-Head Attention**: Attention avec plusieurs têtes parallèles.

**Multimodal**: Modèle traitant plusieurs modalités (texte, image, audio).

## N

**nanoGPT**: Implémentation minimaliste de GPT (Karpathy).

**NCCL**: Bibliothèque de communication collective NVIDIA.

**NDCG**: Métrique de ranking.

**Normalization**: Technique de stabilisation (LayerNorm, RMSNorm).

**Nucleus Sampling** (Top-p): Sampling dans le top-p% de probabilité cumulée.

**NumPy**: Bibliothèque Python de calcul numérique.

## O

**One-Shot Learning**: Apprentissage avec un seul exemple.

**ORPO** (Odds Ratio Preference Optimization): Méthode d'alignment (2024).

**Overfitting**: Sur-apprentissage sur les données d'entraînement.

**OpenAI**: Entreprise créatrice de GPT-3, GPT-4, ChatGPT.

## P

**Padding**: Ajout de tokens spéciaux pour uniformiser longueurs.

**Parameter**: Poids apprenables du modèle.

**Parameter-Efficient Fine-Tuning (PEFT)**: Fine-tuning de peu de paramètres.

**Perplexity**: Mesure de performance (exp(loss)).

**PII** (Personally Identifiable Information): Données personnelles sensibles.

**Pipeline Parallelism**: Parallélisme par découpage du modèle en stages.

**Position Embedding**: Encodage de la position des tokens.

**PPO** (Proximal Policy Optimization): Algorithme RL utilisé dans RLHF.

**Prefix Tuning**: PEFT par préfixes entraînables.

**Prompt**: Input textuel donné au modèle.

**Prompt Engineering**: Art de concevoir des prompts efficaces.

**Prompt Injection**: Attaque par manipulation du prompt.

**Pruning**: Suppression de poids/neurones non importants.

**PyTorch**: Framework de deep learning.

## Q

**QLoRA**: LoRA avec quantization 4-bit.

**Quantization**: Réduction de précision numérique (FP16→INT8).

**Query**: Dans attention, vecteur de requête.

## R

**RAG** (Retrieval-Augmented Generation): Génération augmentée par retrieval.

**Rank** (LoRA): Dimension des matrices low-rank.

**ReAct**: Architecture d'agent (Reasoning + Acting).

**Recall**: Métrique de retrieval (proportion de pertinents récupérés).

**Regularization**: Techniques contre l'overfitting.

**Reinforcement Learning**: Apprentissage par récompenses.

**Replay Buffer**: Mémoire de transitions pour RL.

**Re-ranking**: Re-ordonnancement des résultats de retrieval.

**Residual Connection**: Connexion résiduelle (x + F(x)).

**Reward Model**: Modèle de récompense pour RLHF.

**RLHF** (Reinforcement Learning from Human Feedback): Alignment par RL.

**RMSNorm**: Root Mean Square Normalization.

**RoPE** (Rotary Position Embedding): Encodage positionnel rotatif.

## S

**Sampling**: Sélection stochastique de tokens.

**Scaling Laws**: Lois empiriques d'échelle (performance vs taille/data).

**Self-Attention**: Attention d'une séquence sur elle-même.

**Semantic Search**: Recherche par similarité sémantique.

**Sentence Transformers**: Modèles d'embeddings de phrases.

**SGD** (Stochastic Gradient Descent): Descente de gradient stochastique.

**SFT** (Supervised Fine-Tuning): Fine-tuning supervisé sur instructions.

**Softmax**: Fonction de normalisation en probabilités.

**Speculative Decoding**: Génération avec modèle draft + vérification.

**SSM** (State Space Model): Modèle d'espace d'états (Mamba).

**Stop Sequence**: Séquence déclenchant l'arrêt de génération.

**Streaming**: Génération token par token en temps réel.

**Supervised Learning**: Apprentissage avec labels.

## T

**T5**: Modèle encoder-decoder (Google).

**Teacher Forcing**: Utiliser vraies cibles durant training (pas prédictions).

**Temperature**: Hyperparamètre contrôlant randomness de génération.

**Tensor**: Matrice multi-dimensionnelle.

**Tensor Parallelism**: Parallélisme par découpage des tensors.

**Tokenization**: Découpage du texte en tokens.

**Top-k Sampling**: Sampling parmi les k tokens les plus probables.

**Top-p Sampling**: Nucleus sampling.

**TPU** (Tensor Processing Unit): Accélérateur Google.

**Training**: Entraînement du modèle.

**Transfer Learning**: Réutilisation d'un modèle pré-entraîné.

**Transformer**: Architecture "Attention is All You Need" (2017).

**TRL** (Transformer Reinforcement Learning): Bibliothèque HuggingFace pour RLHF.

## U

**Underfitting**: Sous-apprentissage (modèle trop simple).

**Unsloth**: Framework d'entraînement optimisé (vitesse + mémoire).

## V

**Validation Set**: Données pour évaluation durant training.

**Vector Database**: Base de données pour embeddings (Pinecone, Qdrant, etc.).

**vLLM**: Bibliothèque d'inference optimisée (PagedAttention).

**Vocabulary**: Ensemble des tokens connus du modèle.

**VQA** (Visual Question Answering): QA sur images.

## W

**Warmup**: Phase d'augmentation progressive du learning rate.

**Weight Decay**: Régularisation L2 sur les poids.

**Weights**: Paramètres apprenables du modèle.

## Z

**Zero-Shot Learning**: Inférence sans exemples en contexte.

**ZeRO** (Zero Redundancy Optimizer): Optimisation mémoire (DeepSpeed).

---

# ANNEXE D : RESSOURCES & LIENS

## D.1 Papers Fondateurs

### **Transformers**
1. [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
2. [BERT](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
3. [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Radford et al., 2018
4. [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al., 2019
5. [T5](https://arxiv.org/abs/1910.10683) - Raffel et al., 2020
6. [GPT-3](https://arxiv.org/abs/2005.14165) - Brown et al., 2020

### **Scaling & Training**
7. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - Kaplan et al., 2020
8. [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) - Hoffmann et al., 2022
9. [ZeRO](https://arxiv.org/abs/1910.02054) - Rajbhandari et al., 2020
10. [Megatron-LM](https://arxiv.org/abs/1909.08053) - Shoeybi et al., 2020

### **Fine-tuning & Alignment**
11. [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
12. [QLoRA](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
13. [InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
14. [DPO](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023
15. [Constitutional AI](https://arxiv.org/abs/2212.08073) - Bai et al., 2022

### **Open-Source Models**
16. [Llama](https://arxiv.org/abs/2302.13971) - Touvron et al., 2023
17. [Llama 2](https://arxiv.org/abs/2307.09288) - Touvron et al., 2023
18. [Mistral 7B](https://arxiv.org/abs/2310.06825) - Jiang et al., 2023
19. [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) - Jiang et al., 2024

### **Agents & RAG**
20. [ReAct](https://arxiv.org/abs/2210.03629) - Yao et al., 2022
21. [RAG](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
22. [Toolformer](https://arxiv.org/abs/2302.04761) - Schick et al., 2023
23. [Reflexion](https://arxiv.org/abs/2303.11366) - Shinn et al., 2023

### **Multimodal**
24. [CLIP](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
25. [Flamingo](https://arxiv.org/abs/2204.14198) - Alayrac et al., 2022
26. [LLaVA](https://arxiv.org/abs/2304.08485) - Liu et al., 2023
27. [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) - OpenAI, 2023

### **Optimization & Efficiency**
28. [FlashAttention](https://arxiv.org/abs/2205.14135) - Dao et al., 2022
29. [FlashAttention-2](https://arxiv.org/abs/2307.08691) - Dao, 2023
30. [GPTQ](https://arxiv.org/abs/2210.17323) - Frantar et al., 2023
31. [AWQ](https://arxiv.org/abs/2306.00978) - Lin et al., 2023

## D.2 Cours en Ligne

### **Fondations ML/DL**
- [Fast.ai - Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n - CNN](http://cs231n.stanford.edu/)
- [Stanford CS224n - NLP](http://web.stanford.edu/class/cs224n/)
- [MIT 6.S191 - Intro to Deep Learning](http://introtodeeplearning.com/)

### **LLMs Spécifiques**
- [Andrej Karpathy - Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)
- [DeepLearning.AI - LLM Specialization](https://www.deeplearning.ai/courses/)
- [fast.ai - From Deep Learning Foundations to Stable Diffusion](https://www.fast.ai/posts/part2-2022.html)

### **Production & MLOps**
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Made With ML](https://madewithml.com/)

## D.3 Blogs & Newsletters

### **Blogs Techniques**
- [Jay Alammar - Visualizing ML](https://jalammar.github.io/)
- [Lil'Log - Lilian Weng (OpenAI)](https://lilianweng.github.io/)
- [Sebastian Raschka](https://sebastianraschka.com/blog/)
- [Andrej Karpathy](https://karpathy.github.io/)
- [Hugging Face Blog](https://huggingface.co/blog)
- [OpenAI Research](https://openai.com/research)
- [Anthropic Research](https://www.anthropic.com/research)

### **Newsletters**
- [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)
- [Import AI (Jack Clark)](https://jack-clark.net/)
- [TLDR AI](https://tldr.tech/ai)
- [The Gradient](https://thegradient.pub/)

## D.4 Outils & Frameworks

### **Training**
- [PyTorch](https://pytorch.org/) - Framework principal
- [JAX](https://jax.readthedocs.io/) - Alternative fonctionnelle
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [torchtune](https://github.com/pytorch/torchtune)

### **Inference**
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)
- [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)

### **Agents & RAG**
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://www.llamaindex.ai/)
- [Haystack](https://haystack.deepset.ai/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [CrewAI](https://www.crewai.com/)

### **Vector Databases**
- [Pinecone](https://www.pinecone.io/)
- [Qdrant](https://qdrant.tech/)
- [Weaviate](https://weaviate.io/)
- [Milvus](https://milvus.io/)
- [Chroma](https://www.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)

### **Observability**
- [Weights & Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Arize Phoenix](https://phoenix.arize.com/)
- [LangFuse](https://langfuse.com/)

## D.5 Communautés

### **Discord/Slack**
- Hugging Face Discord
- EleutherAI Discord
- LAION Discord
- LangChain Discord

### **Forums**
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)

### **Twitter/X**
- @karpathy (Andrej Karpathy)
- @ylecun (Yann LeCun)
- @goodfellow_ian (Ian Goodfellow)
- @AndrewYNg (Andrew Ng)
- @jackclarkSF (Jack Clark)

## D.6 Datasets

### **Pré-training**
- [The Pile](https://pile.eleuther.ai/)
- [RedPajama](https://www.together.ai/blog/redpajama)
- [C4](https://huggingface.co/datasets/c4)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)

### **Instruction Tuning**
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [UltraChat](https://huggingface.co/datasets/stingning/ultrachat)

### **RLHF**
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1)

---

## 📖 COMMENT UTILISER CES ANNEXES

### **Annexe A (Formules)**
- Référence rapide durant implémentation
- Vérifier formulations mathématiques
- Comprendre intuitions théoriques

### **Annexe B (Métriques)**
- Évaluer vos modèles
- Comparer avec SOTA
- Choisir métriques appropriées

### **Annexe C (Glossaire)**
- Lookup rapide de termes
- Clarifier jargon
- Référence durant lecture de papers

### **Annexe D (Ressources)**
- Approfondir sujets spécifiques
- Rester à jour (papers récents)
- Trouver outils pour projets

---

**Ces annexes sont des compagnons essentiels de votre parcours. Bookmarkez-les!** 📚
