# Partie 12 : Annexes techniques

Cette partie regroupe des ressources de référence pratiques pour l'entraînement et le déploiement de LLM.

---

## 12.1 Glossaire complet LLM

### A

**Adapter** : Module léger fine-tuné sur un modèle base figé (ex: LoRA).

**Alignment** : Processus pour aligner le comportement du modèle avec les intentions humaines (RLHF, DPO).

**Attention** : Mécanisme permettant au modèle de pondérer l'importance de différentes parties de l'input.

**Autoregressive** : Génération token par token, chaque token dépendant des précédents.

### B

**Batch size** : Nombre d'exemples traités simultanément.

**Beam search** : Stratégie de décodage maintenant plusieurs hypothèses.

**BLEU** : Métrique d'évaluation pour génération de texte (comparaison n-grammes).

**BPE (Byte-Pair Encoding)** : Algorithme de tokenisation subword.

### C

**Causal masking** : Masque empêchant l'attention sur les tokens futurs.

**Checkpoint** : Snapshot des poids du modèle à un moment donné.

**Context window** : Longueur maximale de séquence que le modèle peut traiter.

**Cross-entropy** : Fonction de loss mesurant la différence entre distributions.

### D

**Decoding** : Processus de génération de texte depuis les logits.

**Distillation** : Entraîner un petit modèle à imiter un grand.

**DPO (Direct Preference Optimization)** : Méthode d'alignement sans RL explicite.

**Dropout** : Technique de régularisation désactivant aléatoirement des neurones.

### E

**Embedding** : Représentation vectorielle dense d'un token.

**Epoch** : Une passe complète sur tout le dataset d'entraînement.

### F

**Fine-tuning** : Adapter un modèle pré-entraîné à une tâche spécifique.

**FlashAttention** : Implémentation optimisée de l'attention réduisant la mémoire.

**FSDP (Fully Sharded Data Parallel)** : Parallélisme distribué shardant les paramètres.

### G

**Gradient accumulation** : Accumuler gradients sur plusieurs mini-batchs avant update.

**Greedy decoding** : Toujours choisir le token le plus probable.

### H

**Hallucination** : Génération de contenu factuellement incorrect mais fluide.

**Hyperparameter** : Paramètre de configuration de l'entraînement (learning rate, batch size, etc.).

### I

**Inference** : Utilisation du modèle entraîné pour faire des prédictions.

**Instruction tuning** : Fine-tuning sur paires instruction-réponse.

### K

**KL divergence** : Mesure de différence entre deux distributions.

**KV cache** : Cache des clés et valeurs d'attention pour accélérer l'inférence.

### L

**Learning rate** : Taux d'ajustement des poids pendant l'entraînement.

**Logits** : Scores bruts avant softmax.

**LoRA (Low-Rank Adaptation)** : Fine-tuning efficient via matrices low-rank.

### M

**Masking** : Cacher certains tokens (pour causal LM ou padding).

**MoE (Mixture of Experts)** : Architecture avec plusieurs FFN, routage dynamique.

**Multi-head attention** : Attention avec plusieurs têtes parallèles.

### N

**NTP (Next Token Prediction)** : Objectif d'entraînement des LLM.

**Nucleus sampling** : Top-p sampling basé sur probabilité cumulative.

### P

**Perplexity** : Métrique mesurant l'incertitude du modèle (exp de la loss).

**Prompt** : Texte d'entrée donné au modèle.

**PPO (Proximal Policy Optimization)** : Algorithme RL utilisé dans RLHF.

### Q

**Quantization** : Réduction de la précision des poids (float16 → int8/int4).

### R

**RAG (Retrieval-Augmented Generation)** : Combiner retrieval et génération.

**Reward model** : Modèle apprenant à scorer les réponses selon préférences.

**RLHF (RL from Human Feedback)** : Méthode d'alignement via RL.

### S

**Sampling** : Sélection stochastique du prochain token.

**Scaling laws** : Relations empiriques entre taille modèle/données et performance.

**SFT (Supervised Fine-Tuning)** : Fine-tuning supervisé sur paires input-output.

**Softmax** : Fonction transformant logits en probabilités.

### T

**Temperature** : Paramètre contrôlant la randomness du sampling.

**Tokenization** : Découpage du texte en unités (tokens).

**Top-k sampling** : Sampler parmi les k tokens les plus probables.

**Transformer** : Architecture de réseau basée sur l'attention.

### W

**Warmup** : Augmentation progressive du learning rate en début d'entraînement.

**Weight decay** : Régularisation L2 sur les poids.

### Z

**ZeRO** : Optimisation mémoire (sharding optimizer states, gradients, paramètres).

---

## 12.2 Recettes de configuration standard

### 12.2.1 Modèles pédagogiques (125M - 1B params)

**Configuration 125M (GPT-2 small)** :

```python
config_125M = {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,  # 4 × n_embd
    "activation": "gelu",
    "dropout": 0.1,
}

training_config_125M = {
    "batch_size": 512,
    "learning_rate": 6e-4,
    "warmup_steps": 2000,
    "max_steps": 100000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}
```

**Configuration 1.3B** :

```python
config_1.3B = {
    "vocab_size": 50257,
    "n_positions": 2048,
    "n_embd": 1536,
    "n_layer": 36,
    "n_head": 16,
    "n_inner": 6144,
    "activation": "gelu",
    "dropout": 0.1,
}

training_config_1.3B = {
    "batch_size": 256,
    "learning_rate": 2e-4,
    "warmup_steps": 5000,
    "max_steps": 300000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}
```

### 12.2.2 Recette pré-training base (7B params)

**Inspiré de LLaMA/Mistral** :

```python
config_7B = {
    "vocab_size": 32000,
    "hidden_size": 4096,
    "intermediate_size": 11008,  # ~2.7 × hidden
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,  # GQA: peut réduire (ex: 8)
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,  # RoPE
    "rms_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "use_cache": True,
}

training_7B = {
    "global_batch_size": 4_000_000,  # tokens
    "micro_batch_size": 4,
    "gradient_accumulation_steps": "auto",  # dépend du nombre de GPUs
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 2000,
    "total_steps": 100_000,
    "lr_schedule": "cosine",
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "bf16": True,
}

# Données
data_mix_7B = {
    "web_crawl": 0.45,
    "books": 0.15,
    "code": 0.10,
    "wikipedia": 0.05,
    "papers": 0.05,
    "conversations": 0.10,
    "other": 0.10,
}
```

### 12.2.3 Recette SFT

```python
sft_config = {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset_size": 50_000,  # paires instruction-réponse
    "epochs": 3,
    "batch_size": 128,
    "learning_rate": 2e-5,
    "lr_schedule": "cosine",
    "warmup_ratio": 0.03,
    "max_seq_length": 2048,
    "packing": True,  # Packer plusieurs exemples par séquence
    "use_flash_attn": True,
}

# Mix de données SFT
sft_mix = {
    "general_qa": 0.30,
    "reasoning": 0.20,
    "code": 0.20,
    "creative": 0.15,
    "math": 0.10,
    "safety": 0.05,
}
```

### 12.2.4 Recette DPO

```python
dpo_config = {
    "base_model": "llama-2-7b-chat",  # Partir du modèle SFT
    "preference_dataset_size": 10_000,
    "epochs": 1,
    "batch_size": 64,
    "learning_rate": 5e-7,  # Beaucoup plus faible que SFT
    "beta": 0.1,  # Coefficient KL
    "max_seq_length": 2048,
    "max_prompt_length": 1024,
}
```

### 12.2.5 Recette RLHF (PPO)

```python
rlhf_config = {
    "base_model": "llama-2-7b-sft",
    "reward_model": "llama-2-7b-reward",
    "ppo_epochs": 4,
    "batch_size": 128,
    "mini_batch_size": 32,
    "learning_rate": 1.41e-5,
    "init_kl_coef": 0.2,
    "target_kl": 6.0,
    "gamma": 1.0,
    "lam": 0.95,
    "cliprange": 0.2,
}
```

---

## 12.3 Checklists opérationnelles

### 12.3.1 Checklist avant grand training

**Données** :
- [ ] Dataset nettoyé et dédupliqué
- [ ] Validation split séparé (min 1000 exemples)
- [ ] Statistiques de base calculées (longueur moyenne, distribution)
- [ ] PII supprimées ou anonymisées
- [ ] Licences vérifiées

**Infrastructure** :
- [ ] GPUs disponibles et testés
- [ ] Stockage suffisant (3× taille des données minimum)
- [ ] Monitoring configuré (W&B, TensorBoard, Prometheus)
- [ ] Checkpointing automatique activé
- [ ] Backup/recovery plan défini

**Code** :
- [ ] Reproductibilité assurée (seeds, versions figées)
- [ ] Tests unitaires passés
- [ ] Configuration versionnée (git)
- [ ] Script de reprise après crash testé

**Validation** :
- [ ] Benchmarks de référence définis
- [ ] Validation manuelle prévue (échantillon à évaluer humainement)
- [ ] Critères d'arrêt définis (perplexité, accuracy, early stopping)

### 12.3.2 Checklist avant mise en production

**Modèle** :
- [ ] Évalué sur benchmarks standard
- [ ] Testé sur cas edge/adversariaux
- [ ] Red teaming effectué (sécurité, jailbreaks)
- [ ] Taille optimisée (quantization si pertinent)
- [ ] Comportement sur prompts vides/longs vérifié

**Infrastructure** :
- [ ] Load testing effectué (latence, throughput)
- [ ] Autoscaling configuré et testé
- [ ] Fallback/redundancy en place
- [ ] Monitoring et alertes actifs
- [ ] Logs centralisés

**API** :
- [ ] Authentification sécurisée
- [ ] Rate limiting configuré
- [ ] Documentation API à jour
- [ ] Versioning en place (v1, v2...)
- [ ] Quotas et billing configurés

**Légal & Compliance** :
- [ ] Conditions d'utilisation validées
- [ ] RGPD : Droit à l'oubli implémenté
- [ ] Audit trail en place
- [ ] PII masquées dans les logs
- [ ] Modalités de support définies

### 12.3.3 Checklist sécurité

**Modèle** :
- [ ] Pas de données sensibles dans les poids (membership inference testé)
- [ ] Refus appropriés implémentés
- [ ] Garde-fous contre génération de contenu dangereux
- [ ] Watermarking (si applicable)

**API** :
- [ ] HTTPS uniquement
- [ ] CORS correctement configuré
- [ ] Input sanitization (limite taille, format)
- [ ] Output filtering (détection contenu problématique)
- [ ] Rate limiting anti-abus

**Infrastructure** :
- [ ] Secrets stockés sécurisés (vault, secrets manager)
- [ ] Accès basé sur moindre privilège
- [ ] Logs d'accès et audits activés
- [ ] Sauvegarde chiffrée
- [ ] Plan de réponse aux incidents défini

---

## 12.4 Pistes de recherche et évolutions futures

### 12.4.1 Architectures post-Transformer

**SSM/Mamba** :
- Complexité linéaire en longueur de séquence
- Inférence en temps constant par token
- Challenge : Atteindre la qualité des Transformers

**Architectures hybrides** :
- Combiner Transformer (raisonnement) + SSM (efficacité)
- Exemple : Attention locale + SSM global
- Potentiel : Meilleur compromis qualité/coût

**À surveiller** :
- RWKV : RNN avec parallélisation
- Retentive Networks
- Évolutions de Mamba (Mamba-2, etc.)

### 12.4.2 LLM pour mathématiques et raisonnement

**Défis actuels** :
- Hallucinations sur calculs complexes
- Difficulté avec raisonnement multi-étapes
- Manque de vérification formelle

**Approches prometteuses** :
- Tool use systématique (calculateurs, provers)
- Reward models spécialisés en maths
- Génération de preuves formelles (Lean, Coq)
- Chain-of-thought renforcé

### 12.4.3 Intégration symbolique et neuronale

**Neurosymbolic AI** :
- Combiner LLM (flexibilité) + systèmes symboliques (exactitude)
- Exemple : LLM génère code → moteur de règles valide
- Applications : Planification, vérification, contraintes

**Knowledge graphs + LLM** :
- RAG enrichi avec graphes de connaissances
- Raisonnement relationnel explicite
- Traçabilité et explicabilité améliorées

### 12.4.4 Efficacité et green AI

**Challenges** :
- Coût carbone de l'entraînement (GPT-3 : ~500 tonnes CO2)
- Consommation énergétique de l'inférence à grande échelle

**Directions** :
- Modèles small mais capables (distillation avancée)
- Sparsité et pruning
- Optimisations matérielles (TPU, ASICs dédiés)
- Entraînement distribué efficient (moins de redondance)

### 12.4.5 Multimodalité riche

**Au-delà de texte + image** :
- Audio, vidéo, capteurs, robotique
- Génération vidéo cohérente longue
- Contrôle fin et édition

**Challenges** :
- Alignement cross-modal robuste
- Génération haute résolution
- Cohérence temporelle (vidéo, audio)

---

## 12.5 Ressources complémentaires

### 12.5.1 Courses et tutoriels

- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - NLP with Deep Learning
- [Hugging Face NLP Course](https://huggingface.co/course)
- [Fast.ai](https://www.fast.ai/) - Practical Deep Learning
- [DeepLearning.AI courses](https://www.deeplearning.ai/)

### 12.5.2 Frameworks et bibliothèques

**Entraînement** :
- PyTorch, JAX/Flax
- HuggingFace Transformers
- DeepSpeed, FSDP
- Megatron-LM

**Inférence** :
- vLLM, TGI, SGLang
- Ollama (local)
- ONNX Runtime, TensorRT

**Outils** :
- LangChain, LlamaIndex (orchestration)
- Weights & Biases (tracking)
- Prometheus + Grafana (monitoring)

### 12.5.3 Papers foundationnels

Voir [`REFERENCES.md`](../REFERENCES.md) pour la liste complète et annotée.

### 12.5.4 Communautés et conférences

**Conférences** :
- NeurIPS, ICML, ICLR (ML général)
- ACL, EMNLP (NLP)
- MLSys (systèmes ML)

**Communautés** :
- Hugging Face Forums
- EleutherAI Discord
- Reddit r/MachineLearning, r/LocalLLaMA

---

## Conclusion du livre

Félicitations ! Vous avez parcouru un chemin complet de **zéro au LLM de production**.

**Vous êtes maintenant capable de** :
- Concevoir et entraîner des LLM from scratch
- Fine-tuner et aligner avec les meilleures pratiques
- Intégrer outils, agents et RAG
- Optimiser pour l'inférence à grande échelle
- Déployer en production avec observabilité et sécurité
- Mesurer l'impact et itérer en continu

**Le voyage ne s'arrête pas ici** : Le domaine des LLM évolue rapidement. Continuez à :
- Expérimenter avec les nouvelles architectures et techniques
- Partager vos apprentissages avec la communauté
- Mesurer l'impact réel de vos déploiements
- Rester vigilant sur les aspects éthiques et sociétaux

**Bonne construction de LLM ! 🚀**

---

## Index des concepts clés

*(Index alphabétique pointant vers les sections pertinentes dans chaque partie)*

- **Alignement** → Partie 7
- **Attention mechanism** → Partie 4.1
- **Batching continu** → Partie 9.2
- **DPO** → Partie 7.4
- **FlashAttention** → Partie 4.2
- **FSDP** → Partie 6.4
- **KV cache** → Partie 9.2
- **LoRA** → Partie 9.3
- **MoE** → Partie 4.3
- **Quantization** → Partie 9.3
- **RAG** → Partie 8.2
- **RLHF** → Partie 7.2
- **Scaling laws** → Partie 2.5
- **SFT** → Partie 7.1
- **Tokenization** → Partie 3.4
- **Transformer** → Partie 4.1
- **vLLM** → Partie 9.4

---

**Fin des annexes**
