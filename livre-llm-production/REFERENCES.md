# Références bibliographiques annotées

Ce document présente les références principales utilisées dans le livre « De zéro au LLM de production », organisées par thème et avec des annotations sur leur utilisation dans les différentes parties.

---

## Alignement et Post-training

### Wang et al. - A Comprehensive Survey of LLM Alignment Techniques

**Référence** : (1.1, 1.2)

**Portée** : Synthèse récente et organisée des méthodes d'alignement (RLHF, RLAIF, PPO, DPO, etc.). Point de départ pour exposer l'état de l'art et les comparaisons méthodologiques.

**Utilisation dans le livre** :
- Partie 7, sections 7.1-7.4 : Introduction théorique aux méthodes d'alignement
- Comparatif méthodologique DPO vs PPO vs RLHF
- Discussion des limites pratiques et trade-offs

**Messages clés** :
- Panorama complet des techniques d'alignement modernes
- Comparaison empirique des approches (coûts, efficacité, stabilité)
- Problèmes pratiques : collecte de préférences, biais de jugement, dérive de modèle
- Trade-offs entre simplicité (DPO) et contrôle fin (PPO)

---

### Cao et al. - Towards Scalable Automated Alignment of LLMs

**Référence** : (3.1)

**Portée** : Méthodes d'alignement automatisées, usage d'ICL comme signal, boucle juge→critique→refine, contraintes de scalabilité.

**Utilisation dans le livre** :
- Partie 7, sections 7.2-7.4 : RLAIF et préférences synthétiques
- Encadré "Automatisation et échelle"
- Partie 12 : Roadmap recherche sur alignment automatisé

**Messages clés** :
- Pipelines juge-critique-refine pour itération automatique
- In-Context Learning (ICL) comme signal d'alignement
- Scalabilité : réduire la dépendance aux annotations humaines
- Menaces : dérive, amplification de biais, robustesse des juges IA

---

### Pan et al. - A Survey on Training-free Alignment of LLMs

**Référence** : (2.1)

**Portée** : Taxonomie des méthodes d'alignement sans fine-tuning (logit modifications, guidance vectors, stratégies de décodage).

**Utilisation dans le livre** :
- Partie 7, section 7.5 : Sécurité et refus utiles
- Partie 9 : Contrôles au décodage / safety-inference
- Encadré "Training-free alignment"

**Messages clés** :
- Contrôles sur modèles gelés : logit guidance, guidance vectors
- Avantages : pas de ré-entraînement, déploiement rapide, réversibilité
- Stratégies de décodage piloté pour sécurité et conformité
- Intégration dans la stack de serving (vLLM, TGI, etc.)

---

## Serving, Inference et Optimisation

### Park et al. - A Survey on Inference Engines for LLMs

**Référence** : (5.1)

**Portée** : Revue système des moteurs d'inférence (vLLM, SGLang, TGI), patterns d'optimisation (batching, quantization, cache), comparatifs et critères d'évaluation pour production.

**Utilisation dans le livre** :
- Partie 9, sections 9.2 (Accélération) et 9.4 (Serving)
- Partie 10 : Architecture d'API, coûts, autoscaling
- Encadré "Inference engines"

**Messages clés** :
- Comparatif détaillé vLLM vs TGI vs SGLang vs Ollama vs Triton
- Critères de choix : latence, throughput, coût, compatibilité matériel
- Optimisations système : batching dynamique/continu, KV cache partagé, quantization
- Patterns de déploiement : on-prem vs cloud, GPU vs CPU, autoscaling

**Benchmarks** :
- Latence P50, P95, P99
- Throughput (tokens/sec, requêtes/sec)
- Coût par 1M tokens
- Utilisation mémoire et efficacité GPU

---

## RAG et Contexte Transformer

### Gupta et al. - Survey on Retrieval-Augmented Generation

**Référence** : (4.1)

**Portée** : Contexte RAG, rappel historique Transformer→LM, bonnes pratiques d'ingénierie pour systèmes RAG.

**Utilisation dans le livre** :
- Partie 8 : RAG fiable (indexation, chunking, reranking)
- Partie 5 : Construction des datasets et indexation
- Encadré "RAG : métriques et bonnes pratiques"

**Messages clés** :
- Architecture retriever + generator : composants et flux
- Métriques d'évaluation : pertinence, exactitude, fidélité, robustesse
- Pièges courants : chunking inapproprié, reranking inefficace, hallucinations
- Bonnes pratiques : embeddings de qualité, index hybride (dense + sparse), validation humaine

**Techniques avancées** :
- Chunking sémantique vs fixe
- Reranking avec cross-encoders
- Évaluation automatique vs manuelle
- Gestion de la fraîcheur des données

---

## Outils, Agents et Orchestration

### Watson et al. - Towards an end-to-end personal fine-tuning framework

**Référence** : (6.1)

**Portée** : Intégration LLM↔outils (LangChain / Toolformers / SayCan), personnalisation et pipelines d'annotation/feedback.

**Utilisation dans le livre** :
- Partie 8 : Agents, tool calling
- Encadré pratique sur intégration d'outils
- Pipelines de feedback utilisateur

**Messages clés** :
- Function calling et schémas JSON validés
- Frameworks d'orchestration : LangChain, LlamaIndex, AutoGPT
- Séparation contrôleur / outils pour robustesse
- Boucles de feedback et amélioration continue

---

## Lacunes identifiées et compléments à ajouter

### SSM / Mamba et architectures post-Transformer

**Status** : ⚠️ **Références manquantes**

**Action requise** : Ajouter 2-3 papers techniques focalisés sur SSM/Mamba pour étayer la Partie 4.

**Références recommandées à ajouter** :
1. **Gu & Dao (2023)** - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
2. **Gu et al. (2021)** - "Efficiently Modeling Long Sequences with Structured State Spaces"
3. **Survey SSM** - À identifier (état de l'art sur les State Space Models)

**Utilisation prévue** :
- Partie 4, section "Post-Transformers : SSM/Mamba et hybrides"
- Comparaison architecture : Transformer vs SSM vs hybrides
- Trade-offs : complexité linéaire vs expressivité
- Cas d'usage : long-context, streaming, efficacité

---

## Autres références contextuelles

### Transformers et attention

- Vaswani et al. (2017) - "Attention is All You Need"
- Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)
- Touvron et al. (2023) - "LLaMA: Open and Efficient Foundation Language Models"

### Scaling laws

- Kaplan et al. (2020) - "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) - "Training Compute-Optimal Large Language Models" (Chinchilla)

### Optimisation et entraînement distribué

- Rajbhandari et al. (2020) - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- Shoeybi et al. (2019) - "Megatron-LM: Training Multi-Billion Parameter Language Models"

### Tokenisation

- Sennrich et al. (2016) - "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson (2018) - "SentencePiece: A simple and language independent approach"

---

## Guide d'utilisation de cette bibliographie

### Pour les enseignants et formateurs

Utilisez cette bibliographie comme base pour :
- Construire des slides de cours en citant les sources
- Préparer des lectures complémentaires pour les étudiants
- Identifier les papers à étudier en profondeur par thème

### Pour les apprenants

- Commencez par les surveys (Wang, Cao, Pan, Park, Gupta) pour avoir une vue d'ensemble
- Approfondissez ensuite avec les papers originaux selon vos centres d'intérêt
- Utilisez les annotations pour savoir où chaque référence est exploitée dans le livre

### Pour les praticiens

- Concentrez-vous sur les références d'ingénierie (Park, Gupta, Watson)
- Consultez les papers d'alignement pour les choix de méthodes
- Référez-vous aux papers d'optimisation pour les déploiements à grande échelle

---

## Mise à jour et contributions

Cette bibliographie sera mise à jour régulièrement avec :
- Les nouveaux papers de référence dans le domaine
- Les retours d'expérience de production
- Les benchmarks et comparatifs actualisés

Pour suggérer l'ajout d'une référence, ouvrez une issue sur le dépôt GitHub.

---

**Dernière mise à jour** : 2025-11-18
