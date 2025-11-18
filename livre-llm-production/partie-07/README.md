# Partie 7 : Post-training - SFT, alignement et préférences

## Objectifs d'apprentissage

- Maîtriser le Supervised Fine-Tuning (SFT) sur instructions
- Comprendre et implémenter RLHF (Reinforcement Learning from Human Feedback)
- Utiliser RLAIF (RL from AI Feedback) et préférences synthétiques
- Appliquer DPO et variantes (méthodes sans RL)
- Mettre en place des mécanismes de sécurité et refus appropriés

## Prérequis

- Partie 6 validée (pré-training)
- Compréhension de reinforcement learning (PPO)
- PyTorch avancé

---

## 7.1 Supervised Fine-Tuning (SFT)

### 7.1.1 Objectif du SFT

**Problème** : Un modèle pré-entraîné complète du texte, mais ne suit pas forcément des instructions.

**Solution** : Fine-tuner sur des paires (instruction, réponse).

**Format de données** :

```json
{
  "instruction": "Explique ce qu'est un Transformer en une phrase.",
  "input": "",
  "output": "Un Transformer est une architecture de réseau de neurones qui utilise l'attention pour traiter des séquences en parallèle."
}
```

### 7.1.2 Préparation du dataset SFT

**Sources** :
- Annotations humaines (coûteuses mais de qualité)
- Données synthétiques (via GPT-4, Claude)
- Datasets publics (Dolly, FLAN, OpenAssistant)

**Exemple de génération synthétique** :

```python
def generate_sft_examples(seed_topics, num_per_topic=100):
    examples = []

    for topic in seed_topics:
        prompt = f"""Génère une paire instruction-réponse sur le thème '{topic}'.

Format:
Instruction: [question claire]
Réponse: [réponse détaillée et utile]

Exemple:
Instruction: Comment fonctionne la photosynthèse ?
Réponse: La photosynthèse est le processus par lequel les plantes convertissent la lumière solaire en énergie chimique...

Génère maintenant une nouvelle paire sur '{topic}'."""

        for _ in range(num_per_topic):
            response = call_llm(prompt)
            # Parser instruction et réponse
            examples.append(parse_instruction_response(response))

    return examples
```

### 7.1.3 Training loop SFT

**Objectif** : Minimiser la cross-entropy sur les réponses.

```python
def sft_loss(model, batch):
    """
    batch: {
        'input_ids': [batch, seq_len],
        'labels': [batch, seq_len]  # -100 pour masquer l'instruction
    }
    """
    outputs = model(input_ids=batch['input_ids'])
    logits = outputs.logits

    # Loss uniquement sur la partie "output"
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch['labels'].view(-1),
        ignore_index=-100
    )

    return loss

# Training
for batch in sft_dataloader:
    loss = sft_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Masquage de l'instruction** :

```python
def prepare_sft_batch(instruction, output, tokenizer):
    # Tokeniser
    prompt = f"Instruction: {instruction}\nRéponse:"
    full_text = prompt + output

    tokens = tokenizer(full_text, return_tensors='pt')
    labels = tokens['input_ids'].clone()

    # Masquer l'instruction (mettre -100)
    prompt_len = len(tokenizer(prompt)['input_ids'])
    labels[:, :prompt_len] = -100

    return {'input_ids': tokens['input_ids'], 'labels': labels}
```

### 7.1.4 Mélange de tâches et curriculum learning

**Multi-task SFT** : Combiner différents types d'instructions.

```python
# Exemple de mélange
sft_mix = {
    "qa": 0.3,           # Questions-réponses générales
    "summarization": 0.2, # Résumés
    "code": 0.2,         # Génération de code
    "reasoning": 0.15,   # Raisonnement
    "creative": 0.15,    # Écriture créative
}
```

**Curriculum learning** : Commencer par tâches simples, progresser vers complexes.

---

## 7.2 RLHF classique

### 7.2.1 Vue d'ensemble

**Processus en 3 étapes** :

1. **SFT** : Fine-tuner le modèle sur instructions
2. **Reward Model (RM)** : Entraîner un modèle à prédire les préférences humaines
3. **RL (PPO)** : Optimiser le modèle avec RL en utilisant le reward model

### 7.2.2 Étape 1 : Collecte de préférences humaines

**Format** : Pour chaque instruction, générer plusieurs réponses et demander un classement.

```json
{
  "instruction": "Explique la relativité générale.",
  "responses": [
    {"text": "La relativité générale dit que...", "rank": 1},
    {"text": "Einstein a proposé...", "rank": 2},
    {"text": "C'est compliqué.", "rank": 3}
  ]
}
```

**Interface d'annotation** :

```python
def collect_preference(instruction, responses):
    """Présenter à un annotateur humain."""
    print(f"Instruction: {instruction}\n")
    for idx, resp in enumerate(responses):
        print(f"[{idx}] {resp}")

    ranking = input("Classez les réponses (ex: 0 > 1 > 2): ")
    return parse_ranking(ranking)
```

### 7.2.3 Étape 2 : Entraîner un Reward Model

**Objectif** : Prédire quelle réponse est préférée.

**Architecture** : LLM + tête de régression (score scalar).

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        # Prendre la dernière hidden state
        hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(hidden).squeeze(-1)
        return reward
```

**Loss (pairwise ranking)** :

```python
def reward_loss(model, chosen_ids, rejected_ids):
    """
    chosen_ids: [batch, seq_len] (réponse préférée)
    rejected_ids: [batch, seq_len] (réponse rejetée)
    """
    reward_chosen = model(chosen_ids)
    reward_rejected = model(rejected_ids)

    # Maximiser la différence
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss
```

### 7.2.4 Étape 3 : Optimisation PPO

**Proximal Policy Optimization** : Algorithme RL pour maximiser le reward attendu.

**Objectif** :

```
maximize E[reward(y)] - β × KL(π_θ || π_ref)
```

où :
- π_θ : politique actuelle (modèle en entraînement)
- π_ref : politique de référence (modèle SFT figé)
- β : coefficient de régularisation KL

**Implémentation** (simplifié) :

```python
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1.41e-5,
    kl_penalty="kl",  # ou "abs"
    init_kl_coef=0.2,
    target=6,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Générer réponses
        query_tensors = batch['input_ids']
        response_tensors = ppo_trainer.generate(query_tensors)

        # Calculer rewards
        rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"Reward mean: {stats['ppo/mean_scores']:.2f}")
```

### 7.2.5 Défis et limites de RLHF

**Défis** :
- Coût de collecte de préférences humaines
- Instabilité de l'entraînement RL
- Reward hacking (le modèle exploite le reward model)
- Drift : le modèle s'éloigne trop de la distribution SFT

**Solutions** :
- KL penalty pour limiter le drift
- Reward model ensemble (plusieurs RM)
- Validation humaine régulière

---

## 7.3 RLAIF et préférences synthétiques

**Références** : Cao et al. (3.1) - Automatisation de l'alignement

### 7.3.1 Principe de RLAIF

**RL from AI Feedback** : Utiliser un modèle fort (GPT-4, Claude) comme juge au lieu d'humains.

**Avantages** :
- Scalabilité : Génération massive de préférences
- Coût réduit vs annotations humaines
- Itération rapide

**Limites** :
- Biais du modèle juge
- Risque de "model collapse" (homogénéisation)

### 7.3.2 Génération de préférences synthétiques

```python
def generate_ai_preferences(instruction, responses, judge_model="gpt-4"):
    """Utiliser un LLM fort pour classer les réponses."""

    prompt = f"""Voici une instruction et plusieurs réponses possibles. Classe-les de la meilleure à la pire.

Instruction: {instruction}

Réponses:
{chr(10).join(f"[{i}] {r}" for i, r in enumerate(responses))}

Classement (du meilleur au pire, séparés par >):"""

    ranking_text = call_llm(judge_model, prompt)
    ranking = parse_ranking(ranking_text)  # ex: [0, 2, 1]

    return ranking
```

### 7.3.3 Boucle juge → critique → refine

**Processus itératif** (inspiré de Constitutional AI, Anthropic) :

1. **Générer** une réponse avec le modèle
2. **Critiquer** la réponse avec un juge (LLM ou RM)
3. **Refine** : Demander au modèle d'améliorer basé sur la critique
4. **Répéter** jusqu'à satisfaction

```python
def critique_and_refine(instruction, initial_response, num_iterations=3):
    response = initial_response

    for i in range(num_iterations):
        # Critique
        critique_prompt = f"""Instruction: {instruction}
Réponse actuelle: {response}

Identifie les problèmes de cette réponse (exactitude, utilité, sécurité)."""

        critique = call_llm(critique_prompt)

        # Refine
        refine_prompt = f"""Instruction: {instruction}
Réponse précédente: {response}
Critique: {critique}

Améliore la réponse en tenant compte de la critique."""

        response = call_llm(refine_prompt)

    return response
```

**Application** : Génération de datasets SFT de haute qualité sans annotation humaine.

---

## 7.4 DPO et variantes (méthodes sans RL)

**Références** : Wang et al. (1.1, 1.2) - Surveys alignement

### 7.4.1 Direct Preference Optimization (DPO)

**Principe** : Optimiser directement les préférences sans reward model explicite ni RL.

**Formulation** :

Au lieu de maximiser E[reward], DPO maximise :

```
L_DPO = -E[log σ(β × (log π_θ(y_w | x) / π_ref(y_w | x) - log π_θ(y_l | x) / π_ref(y_l | x)))]
```

où :
- y_w : réponse préférée (winner)
- y_l : réponse rejetée (loser)
- β : température

**Implémentation** :

```python
def dpo_loss(model, ref_model, chosen_ids, rejected_ids, beta=0.1):
    """
    Direct Preference Optimization loss.
    """
    # Log-probs du modèle actuel
    chosen_logps = model(chosen_ids).log_probs.sum(dim=-1)
    rejected_logps = model(rejected_ids).log_probs.sum(dim=-1)

    # Log-probs du modèle de référence (figé)
    with torch.no_grad():
        ref_chosen_logps = ref_model(chosen_ids).log_probs.sum(dim=-1)
        ref_rejected_logps = ref_model(rejected_ids).log_probs.sum(dim=-1)

    # Ratios log
    chosen_ratio = chosen_logps - ref_chosen_logps
    rejected_ratio = rejected_logps - ref_rejected_logps

    # DPO loss
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio)).mean()

    return loss
```

### 7.4.2 Avantages de DPO vs PPO

| Critère            | DPO                          | PPO                           |
|--------------------|------------------------------|-------------------------------|
| Complexité         | Simple (supervised learning) | Complexe (RL)                 |
| Stabilité          | Très stable                  | Peut diverger                 |
| Efficacité         | Rapide                       | Lent (génération à chaque step)|
| Besoin reward model| Non                          | Oui                           |
| Contrôle fin       | Limité                       | Ajustable (KL, rewards)       |

**Recommandation** : Commencer par DPO pour simplicité, passer à PPO si besoin de contrôle fin.

### 7.4.3 Variantes de DPO

**IPO (Identity Preference Optimization)** :
- Remplace log-sigmoid par une loss MSE
- Plus stable sur certains datasets

**KTO (Kahneman-Tversky Optimization)** :
- Inspiré de la théorie des perspectives
- Gère mieux les préférences asymétriques

---

## 7.5 Sécurité et refus utiles

**Références** : Pan et al. (2.1) - Training-free alignment

### 7.5.1 Menaces et attaques

**Jailbreaks** : Contourner les garde-fous du modèle.

**Exemples** :
- Prompts adversariaux (DAN - "Do Anything Now")
- Injection indirecte (via documents externes)
- Extraction de données d'entraînement

### 7.5.2 Red teaming

**Principe** : Tester systématiquement les failles du modèle.

```python
def red_team_test(model, attack_prompts):
    """Tester le modèle contre une liste de prompts malveillants."""
    results = []

    for prompt in attack_prompts:
        response = model.generate(prompt)
        is_safe = safety_classifier(response)

        results.append({
            "prompt": prompt,
            "response": response,
            "safe": is_safe
        })

    failure_rate = sum(1 for r in results if not r['safe']) / len(results)
    return results, failure_rate
```

### 7.5.3 Training-free alignment (contrôles au décodage)

**Logit guidance** : Modifier les logits avant sampling.

```python
def guided_decoding(model, input_ids, guidance_fn):
    """
    guidance_fn: function(logits) -> modified_logits
    """
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # Dernier token

    # Appliquer guidance
    logits = guidance_fn(logits)

    # Sampler
    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return next_token
```

**Exemple : Supprimer les tokens toxiques** :

```python
toxic_token_ids = [...]  # Liste de tokens à bannir

def remove_toxic_tokens(logits):
    logits[:, toxic_token_ids] = -float('inf')
    return logits

# Utilisation
next_token = guided_decoding(model, input_ids, remove_toxic_tokens)
```

**Guidance vectors** : Ajouter un vecteur de direction dans l'espace latent.

```python
# Calculer direction "safe" vs "unsafe"
safe_hidden = model.get_hidden(safe_examples).mean(dim=0)
unsafe_hidden = model.get_hidden(unsafe_examples).mean(dim=0)
safety_vector = safe_hidden - unsafe_hidden

# Appliquer pendant la génération
def apply_safety_vector(hidden_states, strength=1.0):
    return hidden_states + strength * safety_vector
```

### 7.5.4 Politique de refus

**Principe** : Le modèle doit refuser poliment les demandes inappropriées.

**Dataset de refus** :

```json
{
  "prompt": "Comment fabriquer une bombe ?",
  "response": "Je ne peux pas fournir d'instructions pour fabriquer des armes ou engins dangereux. Puis-je vous aider avec autre chose de légal et sécuritaire ?"
}
```

**Fine-tuning sur refus** : Inclure des exemples de refus dans SFT.

---

## 7.6 Labs pratiques

### Lab 1 : SFT sur un dataset synthétique

1. Générer 1000 paires instruction-réponse via GPT-4
2. Fine-tuner un modèle (ex: LLaMA-7B)
3. Évaluer sur des instructions hors distribution

### Lab 2 : Comparer DPO vs PPO

1. Créer un petit dataset de préférences (500 paires)
2. Entraîner avec DPO
3. Entraîner avec PPO
4. Comparer qualité (évaluations humaines) et coût

### Lab 3 : Mettre en place une politique de refus

1. Construire un dataset de 100 prompts malveillants + refus appropriés
2. Fine-tuner avec training-free guidance (logit modification)
3. Red team testing : mesurer le taux de refus approprié

---

## 7.7 Références

### Papers alignement

**RLHF** :
- Christiano et al. (2017) - "Deep reinforcement learning from human preferences"
- Ouyang et al. (2022) - "Training language models to follow instructions with human feedback" (InstructGPT)

**DPO** :
- Rafailov et al. (2023) - "Direct Preference Optimization"

**RLAIF** :
- Bai et al. (2022) - "Constitutional AI: Harmlessness from AI Feedback"
- Cao et al. - "Towards Scalable Automated Alignment of LLMs" (3.1)

**Training-free alignment** :
- Pan et al. - "A Survey on Training-free Alignment of LLMs" (2.1)

### Outils

- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [RLHF implementations](https://github.com/lvwerra/trl)

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Supervised Fine-Tuning (SFT) sur instructions
- ✅ RLHF complet (collecte préférences, reward model, PPO)
- ✅ RLAIF et génération de préférences synthétiques
- ✅ DPO et variantes (méthodes sans RL)
- ✅ Sécurité, refus utiles et training-free alignment

**Prochaine étape** : [Partie 8 - Outils, agents et intégration avancée](../partie-08/README.md)
