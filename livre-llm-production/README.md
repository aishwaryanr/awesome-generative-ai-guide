# De zéro au LLM de production

**Guide complet pour la conception, l'entraînement et le déploiement de Large Language Models**

---

## À propos de ce livre

Ce livre vous accompagne dans un parcours complet, de la compréhension théorique des LLM jusqu'à leur mise en production. Il couvre l'ensemble de la chaîne de valeur : fondations mathématiques, architectures modernes, données, entraînement, alignement, optimisation, déploiement et opérations.

### Public cible

- Ingénieurs ML souhaitant maîtriser les LLM de bout en bout
- Data scientists désireux de passer de la recherche à la production
- Architectes techniques concevant des systèmes à base de LLM
- Chercheurs en IA voulant acquérir une vision pratique et industrielle

### Prérequis

- Python de base
- Notions de probabilités et d'algèbre linéaire (niveau licence)
- Familiarité avec les concepts de machine learning (souhaitable mais pas obligatoire)

---

## Table des matières

### [Partie 1. Vision, panorama et modèles mentaux](./partie-01/README.md)
- Historique du NLP et du Deep Learning
- De RNN aux Transformers
- Cas d'usage des LLM (chatbots, agents, code, copilotes, recherche)
- Modèles mentaux d'ingénierie
- **Labs** : Formulation de cas d'usage et définition de métriques

### [Partie 2. Fondations mathématiques pour LLM](./partie-02/README.md)
- Espaces vectoriels et tenseurs
- Probabilités, entropie et divergence KL
- Optimisation et descente de gradient
- Lois d'échelle (scaling laws)
- **Labs** : Implémentation de cross-entropy et visualisation de paysages de perte

### [Partie 3. Bases de deep learning appliquées au texte](./partie-03/README.md)
- Graphes computationnels et backpropagation
- Couches fondamentales (linéaires, normalisation, résidus, dropout)
- Modèles séquentiels (N-gram, RNN, LSTM, GRU)
- Tokenisation (BPE, Unigram, SentencePiece)
- **Labs** : Construction d'un LM char-level puis subword

### [Partie 4. Architectures de LLM modernes](./partie-04/README.md)
- Transformer : Multi-Head Attention, encodages positionnels
- Optimisations : FlashAttention, sparse attention, long-context
- Mixture of Experts (MoE)
- Post-Transformers : SSM/Mamba et hybrides
- Modèles multimodaux
- **Labs** : Comparaison dense vs MoE, analyse latence/mémoire

### [Partie 5. Données : collecte, nettoyage et préparation](./partie-05/README.md)
- Sources de données (publiques, privées, synthétiques)
- Droits, licences et PII
- Déduplication et filtrage
- Construction du mélange final
- **Labs** : Pipeline de déduplication et analyse de distributions

### [Partie 6. Pré-training : entraîner un LLM de zéro](./partie-06/README.md)
- Objectif Next Token Prediction (NTP)
- Configuration modèle et hyperparamètres
- Parallélisme distribué (data, tensor, pipeline, FSDP, ZeRO)
- Monitoring et checkpointing
- **Labs** : Entraînement d'un modèle small en FSDP

### [Partie 7. Post-training : SFT, alignement et préférences](./partie-07/README.md)
- Supervised Fine-Tuning (SFT)
- RLHF classique (reward model, PPO)
- RLAIF et préférences synthétiques
- Méthodes sans RL (DPO et variantes)
- Sécurité et refus utiles
- **Labs** : Dataset de préférences, comparaison DPO vs PPO, politique de refus

### [Partie 8. Outils, agents et intégration avancée](./partie-08/README.md)
- Tool use et function calling
- RAG fiable (indexation, chunking, reranking)
- Agents et orchestration
- Mémoires prolongées et personnalisation
- **Labs** : Pipeline RAG de bout en bout, agent outillé

### [Partie 9. Inference et optimisation modèle](./partie-09/README.md)
- Stratégies de décodage
- Accélération (KV cache, batching dynamique, spéculation)
- Compression et adaptation (quantization, LoRA, distillation)
- Serving (vLLM, TGI, SGLang, Ollama, Triton)
- **Labs** : Benchmarks latence/throughput, pipeline de spéculation

### [Partie 10. Déploiement en production et LLMOps](./partie-10/README.md)
- Architecture API (gateway, auth, rate limiting)
- Observabilité et monitoring
- Détection de dérive et ré-entraînement
- Coûts et optimisation
- Sécurité, privacy et conformité
- **Labs** : API de service avec observabilité, tableau de bord de dérive

### [Partie 11. Étude de cas fil rouge](./partie-11/README.md)
- Projet complet : assistant technique / copilote dev
- De la spécification au déploiement
- Prototype → Échelle → Industrialisation
- A/B testing et roadmap d'évolution

### [Partie 12. Annexes techniques](./partie-12/README.md)
- Glossaire complet LLM
- Recettes de configuration standard
- Checklists (training, production, sécurité)
- Pistes de recherche

---

## Encadrés thématiques

Le livre contient des encadrés approfondis sur des sujets clés :

- **Comparatif DPO vs PPO** : avantages, limites et contextes d'usage
- **Alignment automatisé à l'échelle** : juges IA, boucles critique-refine
- **Inference engines** : critères de choix et patterns de déploiement
- **Training-free alignment** : contrôles au décodage sans fine-tuning
- **RAG** : métriques, pièges et bonnes pratiques

---

## Labs et exercices pratiques

Chaque partie comprend des exercices pratiques et des labs pour mettre en œuvre les concepts. Les notebooks et scripts sont disponibles dans le dossier [`labs/`](./labs/).

---

## Références bibliographiques

Une bibliographie annotée complète est disponible dans [`REFERENCES.md`](./REFERENCES.md), avec les ancrages principaux :

- **Alignement** : Wang et al., Cao et al., Pan et al.
- **Serving/Inference** : Park et al.
- **RAG et contexte Transformer** : Gupta et al.
- **Outils et agents** : Watson et al.
- **SSM/Mamba** : (à compléter avec papers originaux)

---

## Structure du dépôt

```
livre-llm-production/
├── README.md                 # Ce fichier
├── REFERENCES.md             # Bibliographie annotée
├── partie-01/                # Vision et panorama
├── partie-02/                # Fondations mathématiques
├── partie-03/                # Bases deep learning
├── partie-04/                # Architectures modernes
├── partie-05/                # Données
├── partie-06/                # Pré-training
├── partie-07/                # Post-training et alignement
├── partie-08/                # Outils et agents
├── partie-09/                # Inference et optimisation
├── partie-10/                # Déploiement et LLMOps
├── partie-11/                # Étude de cas fil rouge
├── partie-12/                # Annexes
├── labs/                     # Exercices et notebooks
└── assets/                   # Images et ressources
```

---

## Licence

Ce livre est mis à disposition à des fins éducatives. Tous droits réservés.

---

## Contributions et feedback

Pour signaler des erreurs, suggérer des améliorations ou poser des questions, veuillez ouvrir une issue sur le dépôt GitHub.

---

**Bonne lecture et bon apprentissage !** 🚀
