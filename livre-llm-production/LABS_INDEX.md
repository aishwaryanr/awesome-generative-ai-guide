# Index des Labs et Exercices Pratiques

Ce document répertorie tous les labs et exercices pratiques du livre, organisés par partie.

---

## Partie 1 : Vision, panorama et modèles mentaux

### Exercice 1 : Analyse de cas d'usage
- **Objectif** : Formuler un cas d'usage LLM complet
- **Compétences** : Analyse métier, définition de métriques
- **Localisation** : [Partie 1, Section 1.5](./partie-01/README.md#15-exercices-pratiques)

### Exercice 2 : Formuler des prompts
- **Objectif** : Écrire des prompts efficaces pour différentes tâches
- **Compétences** : Prompt engineering
- **Localisation** : [Partie 1, Section 1.5](./partie-01/README.md#15-exercices-pratiques)

### Exercice 3 : Estimation de ressources
- **Objectif** : Calculer les besoins en mémoire GPU
- **Compétences** : Dimensionnement infrastructure
- **Localisation** : [Partie 1, Section 1.5](./partie-01/README.md#15-exercices-pratiques)

---

## Partie 2 : Fondations mathématiques

### Lab 1 : Implémentation de cross-entropy
- **Objectif** : Implémenter la fonction de loss from scratch
- **Compétences** : NumPy, PyTorch, mathématiques
- **Localisation** : [Partie 2, Section 2.6](./partie-02/README.md#26-exercices-pratiques)
- **Durée estimée** : 1h

### Lab 2 : Visualisation de paysages de perte
- **Objectif** : Visualiser la surface de loss pour comprendre l'optimisation
- **Compétences** : Matplotlib, optimisation
- **Localisation** : [Partie 2, Section 2.6](./partie-02/README.md#26-exercices-pratiques)
- **Durée estimée** : 1-2h

### Lab 3 : Comparaison d'optimiseurs
- **Objectif** : Comparer SGD, Adam et AdamW
- **Compétences** : PyTorch, optimisation
- **Localisation** : [Partie 2, Section 2.6](./partie-02/README.md#26-exercices-pratiques)
- **Durée estimée** : 2-3h

---

## Partie 3 : Bases de deep learning

### Lab 1 : Character-level language model
- **Objectif** : Construire un LM simple prédisant le caractère suivant
- **Compétences** : PyTorch, RNN/LSTM
- **Localisation** : [Partie 3, Section 3.5](./partie-03/README.md#35-labs--construire-un-modèle-de-langage-simple)
- **Durée estimée** : 3-4h

### Lab 2 : Subword language model avec BPE
- **Objectif** : Entraîner un tokenizer BPE et un modèle LSTM
- **Compétences** : Tokenization, LSTM
- **Localisation** : [Partie 3, Section 3.5](./partie-03/README.md#35-labs--construire-un-modèle-de-langage-simple)
- **Durée estimée** : 4-5h

---

## Partie 4 : Architectures modernes

### Lab 1 : Implémenter un Transformer minimal
- **Objectif** : Coder un Transformer from scratch
- **Compétences** : Architecture Transformer, attention
- **Localisation** : [Partie 4, Section 4.6](./partie-04/README.md#46-labs-pratiques)
- **Durée estimée** : 5-8h

### Lab 2 : Comparer dense vs MoE
- **Objectif** : Mesurer latence, mémoire et qualité
- **Compétences** : MoE, benchmarking
- **Localisation** : [Partie 4, Section 4.6](./partie-04/README.md#46-labs-pratiques)
- **Durée estimée** : 3-4h

### Lab 3 : Benchmarker FlashAttention
- **Objectif** : Comparer vitesse attention standard vs FlashAttention
- **Compétences** : Optimisation, profiling
- **Localisation** : [Partie 4, Section 4.6](./partie-04/README.md#46-labs-pratiques)
- **Durée estimée** : 2-3h

---

## Partie 5 : Données

### Lab 1 : Pipeline de déduplication
- **Objectif** : Implémenter déduplication exacte et near-duplicate
- **Compétences** : MinHash, traitement de données
- **Localisation** : [Partie 5, Section 5.7](./partie-05/README.md#57-labs-pratiques)
- **Durée estimée** : 4-6h

### Lab 2 : Analyse de distributions
- **Objectif** : Analyser et visualiser un corpus
- **Compétences** : Statistiques, visualisation
- **Localisation** : [Partie 5, Section 5.7](./partie-05/README.md#57-labs-pratiques)
- **Durée estimée** : 2-3h

### Lab 3 : Construire un mélange
- **Objectif** : Créer un dataset mixte et sharder
- **Compétences** : Data engineering
- **Localisation** : [Partie 5, Section 5.7](./partie-05/README.md#57-labs-pratiques)
- **Durée estimée** : 3-4h

---

## Partie 6 : Pré-training

### Lab : Entraînement complet d'un modèle small
- **Objectif** : Entraîner un modèle 125M from scratch avec FSDP
- **Compétences** : Entraînement distribué, monitoring
- **Localisation** : [Partie 6, Section 6.6](./partie-06/README.md#66-lab--entraînement-complet-dun-modèle-small)
- **Durée estimée** : 1-2 jours (+ compute)
- **Ressources** : 4-8 GPUs recommandés

---

## Partie 7 : Post-training et alignement

### Lab 1 : SFT sur un dataset synthétique
- **Objectif** : Fine-tuner avec des données générées
- **Compétences** : SFT, génération synthétique
- **Localisation** : [Partie 7, Section 7.6](./partie-07/README.md#76-labs-pratiques)
- **Durée estimée** : 1 jour

### Lab 2 : Comparer DPO vs PPO
- **Objectif** : Entraîner et comparer les deux approches
- **Compétences** : DPO, PPO, évaluation
- **Localisation** : [Partie 7, Section 7.6](./partie-07/README.md#76-labs-pratiques)
- **Durée estimée** : 2-3 jours

### Lab 3 : Politique de refus
- **Objectif** : Implémenter des refus appropriés
- **Compétences** : Sécurité, training-free alignment
- **Localisation** : [Partie 7, Section 7.6](./partie-07/README.md#76-labs-pratiques)
- **Durée estimée** : 1 jour

---

## Partie 8 : Outils, agents et RAG

### Lab 1 : Pipeline RAG complet
- **Objectif** : Construire un système RAG de bout en bout
- **Compétences** : Embeddings, retrieval, génération
- **Localisation** : [Partie 8, Section 8.5](./partie-08/README.md#85-labs-pratiques)
- **Durée estimée** : 1-2 jours

### Lab 2 : Agent multi-outils
- **Objectif** : Créer un agent utilisant plusieurs outils
- **Compétences** : Function calling, orchestration
- **Localisation** : [Partie 8, Section 8.5](./partie-08/README.md#85-labs-pratiques)
- **Durée estimée** : 1 jour

### Lab 3 : Système avec mémoire
- **Objectif** : Chatbot avec mémoire persistante
- **Compétences** : Long-term memory, personnalisation
- **Localisation** : [Partie 8, Section 8.5](./partie-08/README.md#85-labs-pratiques)
- **Durée estimée** : 1 jour

---

## Partie 9 : Inference et optimisation

### Lab 1 : Benchmarker vLLM vs TGI
- **Objectif** : Comparer performance des engines
- **Compétences** : Benchmarking, serving
- **Localisation** : [Partie 9, Section 9.5](./partie-09/README.md#95-labs-pratiques)
- **Durée estimée** : 0.5 jour

### Lab 2 : Speculative decoding
- **Objectif** : Implémenter et mesurer le speedup
- **Compétences** : Optimisation inference
- **Localisation** : [Partie 9, Section 9.5](./partie-09/README.md#95-labs-pratiques)
- **Durée estimée** : 1-2 jours

### Lab 3 : Quantization et évaluation
- **Objectif** : Quantifier et évaluer la perte de qualité
- **Compétences** : Compression, benchmarking
- **Localisation** : [Partie 9, Section 9.5](./partie-09/README.md#95-labs-pratiques)
- **Durée estimée** : 1 jour

---

## Partie 10 : Déploiement et LLMOps

### Lab 1 : API complète avec monitoring
- **Objectif** : Déployer une API production-ready
- **Compétences** : FastAPI, Prometheus, Grafana
- **Localisation** : [Partie 10, Section 10.6](./partie-10/README.md#106-labs-pratiques)
- **Durée estimée** : 2-3 jours

### Lab 2 : Détection de dérive
- **Objectif** : Détecter automatiquement le drift
- **Compétences** : Monitoring, ML ops
- **Localisation** : [Partie 10, Section 10.6](./partie-10/README.md#106-labs-pratiques)
- **Durée estimée** : 1 jour

### Lab 3 : Optimisation des coûts
- **Objectif** : Implémenter caching et cascade
- **Compétences** : Optimisation économique
- **Localisation** : [Partie 10, Section 10.6](./partie-10/README.md#106-labs-pratiques)
- **Durée estimée** : 1 jour

---

## Récapitulatif par niveau de difficulté

### Débutant (1-2 jours)
- Partie 1 : Tous les exercices
- Partie 2 : Lab 1 (Cross-entropy)
- Partie 5 : Lab 2 (Analyse de distributions)
- Partie 9 : Lab 1 (Benchmarking)

### Intermédiaire (2-5 jours)
- Partie 2 : Labs 2-3
- Partie 3 : Lab 1 (Char-level LM)
- Partie 4 : Lab 3 (FlashAttention)
- Partie 5 : Labs 1, 3
- Partie 7 : Lab 3 (Refus)
- Partie 8 : Labs 2-3
- Partie 9 : Lab 3 (Quantization)
- Partie 10 : Labs 2-3

### Avancé (5+ jours)
- Partie 3 : Lab 2 (Subword LM)
- Partie 4 : Labs 1-2
- Partie 6 : Lab complet (Pré-training)
- Partie 7 : Labs 1-2
- Partie 8 : Lab 1 (RAG)
- Partie 9 : Lab 2 (Speculative decoding)
- Partie 10 : Lab 1 (API complète)

---

## Parcours suggérés

### Parcours Research (focus qualité modèle)
1. Partie 2 : Tous les labs
2. Partie 3 : Labs 1-2
3. Partie 4 : Lab 1 (Transformer)
4. Partie 6 : Pré-training complet
5. Partie 7 : Labs 1-2 (SFT, DPO/PPO)

### Parcours Engineering (focus déploiement)
1. Partie 5 : Tous les labs (données)
2. Partie 8 : Tous les labs (RAG, agents)
3. Partie 9 : Tous les labs (optimisation)
4. Partie 10 : Tous les labs (production)

### Parcours Full Stack (complet)
Suivre tous les labs dans l'ordre des parties.

**Durée totale estimée** : 6-8 semaines à temps plein

---

## Resources complémentaires pour les labs

### Datasets
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [The Pile](https://pile.eleuther.ai/)
- [OpenWebText](https://openwebtext2.readthedocs.io/)

### Compute
- [Google Colab](https://colab.research.google.com/) - Free GPUs
- [Kaggle Kernels](https://www.kaggle.com/) - Free GPUs/TPUs
- [Lambda Labs](https://lambdalabs.com/) - Cloud GPUs
- [RunPod](https://www.runpod.io/) - Affordable GPU rental

### Outils de monitoring
- [Weights & Biases](https://wandb.ai/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [MLflow](https://mlflow.org/)

---

**Bon apprentissage et bonne pratique !** 🚀
