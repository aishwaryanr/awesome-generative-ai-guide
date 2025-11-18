# Partie 1 : Vision, panorama et modèles mentaux

## Objectifs d'apprentissage

À la fin de cette partie, vous serez capable de :

- Situer les LLM dans l'évolution historique du NLP et du deep learning
- Comprendre les grands cas d'usage des LLM et leurs implications business
- Établir des modèles mentaux d'ingénierie pour concevoir et déployer des systèmes basés sur les LLM
- Formuler clairement un cas d'usage LLM avec ses métriques de succès

## Prérequis

- Python de base
- Notions de probabilité et d'algèbre linéaire (niveau licence)
- Curiosité pour l'IA et les systèmes distribués

---

## 1.1 Historique rapide : du NLP classique aux LLM modernes

### 1.1.1 Ère pré-deep learning (1950-2010)

Le traitement automatique du langage naturel (NLP) a connu plusieurs révolutions méthodologiques :

**Années 1950-1980 : Approches symboliques**
- Systèmes à base de règles linguistiques
- Grammaires formelles (Chomsky)
- Analyseurs syntaxiques déterministes
- **Limites** : fragilité, couverture limitée, maintenance complexe

**Années 1990-2000 : Approches statistiques**
- Modèles de langage N-grammes
- Hidden Markov Models (HMM) pour la séquence
- Maximum Entropy et Conditional Random Fields (CRF)
- **Avancée** : apprentissage automatique à partir de corpus
- **Limites** : malédiction de la dimensionnalité, contexte très local

**Années 2000-2010 : Premiers modèles neuronaux**
- Word embeddings : Word2Vec (Mikolov, 2013), GloVe (Pennington, 2014)
- Représentations denses apprises
- **Innovation** : capture de similarités sémantiques (roi - homme + femme ≈ reine)
- **Limites** : représentations statiques, pas de contexte

### 1.1.2 Révolution deep learning pour le NLP (2010-2017)

**Modèles séquentiels récurrents**

Les Recurrent Neural Networks (RNN) ont introduit la capacité de traiter des séquences de longueur variable :

```
RNN simple :
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
y_t = W_hy * h_t
```

**Problèmes** :
- Vanishing gradients : difficulté à capturer des dépendances longues
- Traitement séquentiel : pas de parallélisation

**Solutions : LSTM et GRU (1997/2014)**

Long Short-Term Memory (LSTM) :
- Gates (oubli, entrée, sortie) pour contrôler le flux d'information
- Meilleure capture des dépendances longues
- Applications : traduction automatique, génération de texte

Gated Recurrent Unit (GRU) :
- Version simplifiée de LSTM
- Moins de paramètres, entraînement plus rapide

**Limites persistantes** :
- Coût séquentiel O(n) : difficulté à paralléliser
- Contexte limité en pratique (quelques centaines de tokens)
- Difficultés d'entraînement sur très longues séquences

### 1.1.3 Révolution Transformer (2017-aujourd'hui)

**"Attention is All You Need" (Vaswani et al., 2017)**

Le mécanisme d'attention permet de capturer des dépendances directes entre tous les tokens d'une séquence, quelle que soit leur distance.

**Principe de l'attention** :

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Avantages décisifs** :
- Parallélisation massive : tous les tokens traités simultanément
- Dépendances longues : attention directe sur toute la séquence
- Scalabilité : architecture adaptée aux GPUs/TPUs modernes

**Évolution rapide** :

1. **2018 : BERT (Devlin et al.)**
   - Bidirectionnel, pré-entraîné sur Masked Language Modeling
   - Fine-tuning pour tâches spécifiques
   - Révolutionne la compréhension de langage

2. **2019-2020 : GPT-2 et GPT-3 (OpenAI)**
   - Architecture décodeur-only
   - Pré-entraînement causal (prédiction du token suivant)
   - GPT-3 (175B paramètres) : capacités few-shot spectaculaires

3. **2023 : LLaMA, Falcon, Mistral**
   - Modèles open-source performants
   - Optimisations d'architecture et d'entraînement
   - Démocratisation de l'accès aux LLM

4. **2024-2025 : GPT-4, Claude 3.5, Gemini**
   - Modèles multimodaux (texte + vision + audio)
   - Capacités de raisonnement avancées
   - Fenêtres de contexte étendues (100k+ tokens)

### 1.1.4 Tendances actuelles et futures

**Architectures émergentes** :
- **Mixture of Experts (MoE)** : efficacité computationnelle (Mixtral, GPT-4)
- **State Space Models (SSM/Mamba)** : complexité linéaire pour le contexte long
- **Architectures hybrides** : combiner Transformer et SSM

**Améliorations continues** :
- Fenêtres de contexte plus larges (1M+ tokens)
- Efficacité d'entraînement et d'inférence
- Multimodalité riche (texte, image, audio, vidéo, code)

---

## 1.2 Cas d'usage des LLM : panorama et implications

Les LLM ont ouvert un espace immense de cas d'usage. Voici les principaux domaines d'application.

### 1.2.1 Chatbots et assistants conversationnels

**Description** : Dialogues en langage naturel pour assistance, support client, conseil.

**Exemples** :
- Support client automatisé (résolution de tickets, FAQ)
- Assistants virtuels personnels (planning, recherche d'info)
- Conseillers spécialisés (santé, finance, juridique)

**Capacités LLM mobilisées** :
- Compréhension d'instructions en langage naturel
- Génération de réponses cohérentes et contextuelles
- Mémorisation du contexte conversationnel

**Défis** :
- Hallucinations (génération de fausses informations)
- Gestion de l'ambiguïté et des demandes imprécises
- Conformité et sécurité (données sensibles, refus appropriés)

**Métriques de succès** :
- Taux de résolution automatique
- Satisfaction utilisateur (CSAT)
- Temps de réponse moyen

### 1.2.2 Agents autonomes et workflows orchestrés

**Description** : LLM comme contrôleur capable de planifier, d'exécuter des actions et d'utiliser des outils.

**Exemples** :
- Agents de recherche et analyse (parcourir le web, synthétiser)
- Automatisation de tâches métier (CRM, reporting, workflows)
- Assistants de productivité (gestion emails, agendas, projets)

**Capacités LLM mobilisées** :
- Planification multi-étapes (décomposition de tâches)
- Function calling (appel d'APIs et outils externes)
- Perception-action-observation (boucles de contrôle)

**Défis** :
- Robustesse et gestion d'erreurs
- Sécurité (accès contrôlé aux outils sensibles)
- Coût et latence (multiples appels au LLM)

**Métriques de succès** :
- Taux de complétion de tâches
- Précision des actions exécutées
- Coût par workflow réussi

### 1.2.3 Génération et assistance au code

**Description** : Copilotes de développement, génération de code, debugging, documentation.

**Exemples** :
- GitHub Copilot, Amazon CodeWhisperer
- Génération de tests unitaires
- Refactoring et optimisation de code
- Documentation automatique

**Capacités LLM mobilisées** :
- Compréhension et génération de code multi-langages
- Complétion contextuelle (fichier, projet)
- Traduction de spécifications en code

**Défis** :
- Correction et sécurité du code généré
- Respect des conventions et styles du projet
- Propriété intellectuelle et licences

**Métriques de succès** :
- Pourcentage d'acceptation des suggestions
- Gain de productivité (temps économisé)
- Taux de bugs dans le code généré

### 1.2.4 Recherche d'information et RAG

**Description** : Systèmes de question-réponse sur corpus privés ou étendus.

**Exemples** :
- Search augmenté (Bing Chat, Perplexity AI)
- Q&A sur documentation interne d'entreprise
- Assistants de recherche académique et juridique

**Capacités LLM mobilisées** :
- Retrieval-Augmented Generation (RAG)
- Compréhension de requêtes complexes
- Synthèse de documents multiples

**Défis** :
- Pertinence et exactitude des réponses
- Hallucinations (inventer des sources)
- Fraîcheur et mise à jour des connaissances

**Métriques de succès** :
- Précision et rappel des réponses
- Taux de citations correctes
- Temps de réponse

### 1.2.5 Création de contenu et rédaction

**Description** : Génération de texte marketing, articles, résumés, traductions.

**Exemples** :
- Rédaction publicitaire et marketing
- Résumés automatiques de documents
- Traduction multilingue
- Génération de scripts, scénarios

**Capacités LLM mobilisées** :
- Génération de texte fluide et stylisé
- Adaptation au ton et au public cible
- Multilinguisme

**Défis** :
- Originalité et plagiat
- Respect du style et de la voix de marque
- Fact-checking et exactitude

**Métriques de succès** :
- Qualité perçue (évaluations humaines)
- Engagement (clics, conversions)
- Temps de rédaction économisé

### 1.2.6 Autres cas d'usage émergents

- **Éducation** : tuteurs personnalisés, génération d'exercices
- **Santé** : assistance au diagnostic, synthèse médicale
- **Finance** : analyse de rapports, génération de prévisions
- **Juridique** : analyse de contrats, recherche de jurisprudence
- **Créativité** : musique, art, design assisté par IA

---

## 1.3 Modèles mentaux d'ingénierie pour les LLM

Pour concevoir et déployer des systèmes LLM en production, il est essentiel d'adopter des modèles mentaux clairs.

### 1.3.1 Modèle mental #1 : LLM comme distribution de probabilité

**Concept** : Un LLM est une fonction qui apprend p(texte | contexte).

```
p(token_suivant | token_1, token_2, ..., token_n)
```

**Implications** :
- Le modèle ne "sait" rien : il estime des probabilités sur la base de ses données d'entraînement
- La génération est stochastique (sampling) ou déterministe (greedy, beam search)
- Les hallucinations sont une conséquence naturelle de ce modèle probabiliste

**Conséquences pratiques** :
- Ne jamais faire confiance aveuglément à une sortie LLM
- Toujours mettre en place des validations et garde-fous
- Calibrer la température de génération selon le cas d'usage

### 1.3.2 Modèle mental #2 : LLM comme système distribué

**Composants d'un système LLM en production** :

```
[Données] → [Entraînement/Fine-tuning] → [Modèle]
                                              ↓
[Utilisateurs] ← [API Gateway] ← [Serving engine] ← [Optimisation]
                        ↓
              [Monitoring & Logs]
```

**Caractéristiques d'un système distribué** :
- **Latence** : temps de réponse (P50, P95, P99)
- **Débit** : requêtes par seconde, tokens par seconde
- **Disponibilité** : SLA, redondance, failover
- **Coût** : compute, mémoire, stockage
- **Scalabilité** : horizontal (+ serveurs) vs vertical (+ GPU)

**Trade-offs à gérer** :
- Latence vs coût (modèle small vs large)
- Qualité vs débit (batch size, concurrence)
- Fraîcheur vs stabilité (ré-entraînement, A/B tests)

### 1.3.3 Modèle mental #3 : LLM comme composant logiciel versionné

**Approche DevOps/MLOps** :

Un LLM doit être traité comme du code :
- **Versioning** : git pour les configs, model registry pour les poids
- **Testing** : tests unitaires, tests de régression, benchmarks
- **CI/CD** : pipelines d'entraînement, déploiement progressif (canary, blue/green)
- **Observabilité** : logs, traces, métriques (qualité, coût, dérive)

**Cycle de vie d'un modèle** :

```
Recherche → Prototype → Validation → Staging → Production
              ↓            ↓           ↓           ↓
          [Datasets] [Métriques] [A/B tests] [Monitoring]
                                                  ↓
                                        [Ré-entraînement]
```

**Principes d'ingénierie** :
- Reproductibilité : seeds, configs, environnements figés
- Auditabilité : tracking des datasets, hyperparamètres, résultats
- Rollback : capacité à revenir à une version antérieure rapidement

### 1.3.4 Modèle mental #4 : LLM comme compromis mémoire/compute/IO

**Contraintes matérielles** :

Pour un modèle de N paramètres :
- **Mémoire minimale** : ~2N bytes (en float16) + activations + KV cache
- **Compute** : ~2N FLOPs par token généré (forward pass)
- **IO** : bande passante mémoire, latence réseau (si distribué)

**Exemple** : Modèle 70B paramètres
- Mémoire : 70B × 2 bytes = 140 GB (minimum)
- En pratique : 200-300 GB avec activations et cache
- Nécessite : 4× A100 80GB ou 2× H100 80GB

**Optimisations possibles** :
- **Quantization** : float16 → int8 → int4 (réduction mémoire × 2-4)
- **LoRA/PEFT** : adapter seulement une fraction des paramètres
- **Distillation** : entraîner un modèle small à imiter un modèle large
- **KV cache** : réutiliser les états des tokens déjà traités

### 1.3.5 Modèle mental #5 : LLM comme système humain-machine

**Le LLM n'est qu'un outil** :

Un système LLM efficace combine :
- **Humains** : définir les objectifs, annoter, valider, corriger
- **LLM** : génération, compréhension, raisonnement
- **Outils** : bases de données, APIs, calculs déterministes
- **Garde-fous** : règles métier, validations, refus

**Principes d'intégration** :
- **Human-in-the-loop** : validation humaine sur décisions critiques
- **Confidence scoring** : le LLM indique sa confiance, l'humain décide
- **Feedback loops** : les corrections humaines améliorent le modèle

**Exemple : modération de contenu**

```
Contenu → [LLM classifie] → Si confiance > 0.9 : action automatique
                           → Si 0.5 < confiance < 0.9 : modération humaine
                           → Si confiance < 0.5 : logs et analyse
```

---

## 1.4 Formuler un cas d'usage LLM : méthodologie

### 1.4.1 Étape 1 : Définir le problème et les contraintes

**Questions clés** :
1. Quel est le problème métier à résoudre ?
2. Qui sont les utilisateurs ? Quels sont leurs besoins ?
3. Quelles sont les contraintes (latence, coût, sécurité, conformité) ?
4. Quelles sont les données disponibles ?

**Exemple : Assistant de support client**

- **Problème** : 60% des tickets de support sont répétitifs et simples
- **Utilisateurs** : Clients cherchant une réponse rapide
- **Contraintes** : Latence < 2s, coût < 0.01€/requête, données RGPD
- **Données** : 100k tickets historiques avec réponses

### 1.4.2 Étape 2 : Choisir l'approche et l'architecture

**Approches possibles** :
- **Fine-tuning** : adapter un modèle de base à votre domaine
- **Prompt engineering** : utiliser un modèle générique avec des prompts optimisés
- **RAG** : combiner retrieval et génération pour exploiter une base de connaissances
- **Agents** : orchestrer LLM + outils pour tâches complexes

**Critères de choix** :

| Approche        | Coût initial | Coût récurrent | Flexibilité | Latence |
|-----------------|--------------|----------------|-------------|---------|
| Prompt eng.     | Faible       | Moyen-élevé    | Très haute  | Basse   |
| RAG             | Moyen        | Moyen          | Haute       | Moyenne |
| Fine-tuning     | Élevé        | Faible-moyen   | Moyenne     | Basse   |
| Agents          | Moyen-élevé  | Élevé          | Très haute  | Haute   |

### 1.4.3 Étape 3 : Définir les métriques de succès

**Métriques techniques** :
- **Qualité** : BLEU, ROUGE, perplexité, accuracy sur benchmarks
- **Performance** : latence (P50, P95, P99), throughput (req/s, tokens/s)
- **Coût** : coût par requête, coût par utilisateur actif

**Métriques métier** :
- **Impact** : taux de résolution, gain de productivité, satisfaction utilisateur
- **Adoption** : % d'utilisateurs actifs, fréquence d'utilisation
- **ROI** : économies réalisées, revenus générés

**Métriques de sécurité et conformité** :
- Taux de refus appropriés (prompts malveillants)
- Fuites de données (PII, secrets)
- Hallucinations critiques détectées

### 1.4.4 Étape 4 : Planifier l'itération et l'amélioration

**Cycle d'amélioration continue** :

1. **Prototype** : version minimale sur données restreintes
2. **Test interne** : évaluation par l'équipe, métriques de base
3. **Alpha** : déploiement limité, feedback utilisateurs pilotes
4. **Beta** : déploiement élargi, A/B tests
5. **Production** : monitoring, alertes, ré-entraînement périodique

**Boucles de feedback** :
- Annotations humaines sur erreurs détectées
- Logs de comportements inattendus
- Signalements utilisateurs (thumbs up/down)

---

## 1.5 Exercices pratiques

### Exercice 1 : Analyse de cas d'usage

Choisissez un cas d'usage LLM dans votre domaine et répondez aux questions suivantes :

1. Quel problème métier résout-il ?
2. Quels sont les utilisateurs et leurs besoins ?
3. Quelles sont les contraintes techniques et réglementaires ?
4. Quelle approche choisiriez-vous (prompts, RAG, fine-tuning, agents) et pourquoi ?
5. Quelles métriques de succès proposeriez-vous ?

### Exercice 2 : Formuler des prompts

Pour le cas d'usage suivant : "Assistant de synthèse de réunions", écrivez 3 prompts différents pour :
- Extraire les points clés d'une transcription
- Identifier les décisions prises
- Lister les actions à suivre

Comparez les résultats avec un LLM de votre choix (GPT, Claude, etc.).

### Exercice 3 : Estimation de ressources

Pour un modèle LLaMA 13B (13 milliards de paramètres) :
- Estimez la mémoire GPU minimale nécessaire en float16
- Estimez la mémoire avec KV cache pour 4096 tokens de contexte
- Proposez une stratégie d'optimisation si vous n'avez qu'un GPU 24GB

---

## 1.6 Lectures recommandées

### Articles fondateurs
- Vaswani et al. (2017) - "Attention is All You Need"
- Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)

### Surveys et panoramas
- Gupta et al. - Survey RAG (référence 4.1)
- Watson et al. - Frameworks et outils (référence 6.1)

### Ressources en ligne
- [Hugging Face Course](https://huggingface.co/course) - Introduction aux Transformers
- [Fast.ai NLP](https://www.fast.ai/) - Cours pratique de NLP
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visualisation intuitive

---

## Résumé de la Partie 1

Vous avez maintenant :
- ✅ Une vision historique : de RNN aux Transformers et LLM modernes
- ✅ Une cartographie des cas d'usage : chatbots, agents, code, RAG, création
- ✅ Des modèles mentaux pour penser LLM : probabilités, systèmes distribués, composants versionnés
- ✅ Une méthodologie pour formuler et évaluer un cas d'usage LLM

**Prochaine étape** : [Partie 2 - Fondations mathématiques](../partie-02/README.md) pour maîtriser les outils mathématiques indispensables aux LLM.
