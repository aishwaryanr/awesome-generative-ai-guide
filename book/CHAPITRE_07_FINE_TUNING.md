# CHAPITRE 7 : FINE-TUNING DES LARGE LANGUAGE MODELS

> *« Un modèle pré-entraîné est comme un étudiant brillant qui a lu toute Wikipedia. Le fine-tuning, c'est lui enseigner votre métier spécifique. »*

---

## Introduction : De la Connaissance Générale à l'Expertise

### 🎭 Dialogue : Le Médecin Généraliste

**Alice** : Bob, j'ai téléchargé GPT-2. Il génère du texte cohérent, mais quand je lui demande de diagnostiquer des symptômes médicaux, il hallucine complètement !

**Bob** : Normal. GPT-2 a lu des millions de pages web, mais il n'est pas **spécialisé** en médecine. C'est comme demander à un étudiant en littérature de faire une chirurgie.

**Alice** : Donc je dois ré-entraîner depuis zéro sur des données médicales ?

**Bob** : Non ! Ce serait comme faire refaire toutes les études à ton médecin. Le **fine-tuning** c'est plutôt : il garde ses connaissances générales (grammaire, culture, logique) et tu lui enseignes **en plus** la médecine spécialisée.

**Alice** : Donc je pars du modèle pré-entraîné et je continue l'entraînement sur mes données ?

**Bob** : Exactement ! Coût : 1000× moins cher que pré-entraîner. Temps : quelques heures au lieu de semaines. Résultats : modèle expert dans ton domaine.

### 📊 Pré-Training vs Fine-Tuning

| Aspect | Pré-Training | Fine-Tuning |
|--------|-------------|-------------|
| **Données** | Énormes (TB) : Common Crawl, Wikipedia | Petites (MB-GB) : domaine spécifique |
| **Objectif** | Apprendre le langage général | Spécialiser pour une tâche |
| **Coût** | $1M-$100M (GPT-3) | $100-$10k |
| **Durée** | Semaines-mois | Heures-jours |
| **Hardware** | Clusters GPU (100s-1000s) | 1-8 GPUs |
| **Exemples** | GPT-3, LLaMA, BERT | ChatGPT, Med-PaLM, CodeLlama |

### 🎯 Anecdote : La Naissance du Fine-Tuning Moderne

**2018, Google AI**

L'équipe BERT vient de pré-entraîner un modèle sur 3.3B mots pendant des semaines. Coût : ~$50,000 en compute.

*Chercheur 1* : "Maintenant, pour chaque tâche (classification, NER, QA), on doit ré-entraîner depuis zéro ?"

*Jacob Devlin (lead BERT)* : "Non, regardez : on garde tous les poids BERT, on ajoute juste une petite couche finale, et on ré-entraîne **seulement** sur la tâche cible."

**Résultat** : Fine-tuning BERT sur SQuAD (QA) prend 30 minutes sur 1 GPU et bat tous les records !

**Impact** : Paradigme shift en NLP
- Avant : Entraîner un modèle par tâche (cher, lent)
- Après : 1 modèle pré-entraîné → fine-tuner pour N tâches (rapide, efficace)

Aujourd'hui, 99% des applications LLM utilisent des modèles fine-tunés.

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Comprendre quand et pourquoi fine-tuner
- ✅ Préparer vos données pour le fine-tuning
- ✅ Implémenter le fine-tuning complet avec HuggingFace
- ✅ Maîtriser les hyperparamètres critiques
- ✅ Éviter l'overfitting et la catastrophic forgetting
- ✅ Évaluer et déployer votre modèle fine-tuné

**Difficulté** : 🔴🔴🔴⚪⚪ (Avancé)
**Prérequis** : PyTorch, Transformers (Chapitre 4), GPU recommandé
**Temps de lecture** : ~120 minutes

---

## Les Trois Approches du Fine-Tuning

### 1. Full Fine-Tuning

**Principe** : Ré-entraîner **tous les paramètres** du modèle.

```python
# Tous les paramètres sont modifiables
for param in model.parameters():
    param.requires_grad = True

# Entraîner normalement
optimizer = AdamW(model.parameters(), lr=5e-5)
```

**✅ Avantages** :
- Performance maximale
- Adaptation complète au domaine

**❌ Inconvénients** :
- Très coûteux (GPU 40GB+ pour LLaMA-7B)
- Risque de catastrophic forgetting
- Lent

**Cas d'usage** : Domaine très différent du pré-training (ex: langage médical technique).

### 2. Feature-Based (Frozen Backbone)

**Principe** : **Geler** le modèle pré-entraîné, entraîner seulement la tête de classification.

```python
# Geler le backbone
for param in model.base_model.parameters():
    param.requires_grad = False

# Entraîner seulement la tête
optimizer = AdamW(model.classifier.parameters(), lr=1e-3)
```

**✅ Avantages** :
- Très rapide
- Peu de mémoire
- Pas de forgetting

**❌ Inconvénients** :
- Performance limitée
- Pas d'adaptation du backbone

**Cas d'usage** : Classification simple, peu de données.

### 3. Partial Fine-Tuning (Progressive Unfreezing)

**Principe** : Dégeler progressivement les couches du modèle.

```python
# Phase 1 : Seulement la tête (2 epochs)
# Phase 2 : Dégeler les dernières 2 couches (2 epochs)
# Phase 3 : Dégeler tout (1 epoch)

def unfreeze_layers(model, num_layers):
    """Dégèle les N dernières couches."""
    layers = list(model.encoder.layer)
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
```

**✅ Avantages** :
- Bon compromis performance/coût
- Stabilité accrue

**❌ Inconvénients** :
- Plus complexe à implémenter
- Nécessite plusieurs runs

---

## Préparer les Données de Fine-Tuning

### Format des Données

**Pour la classification** :
```json
[
  {"text": "Ce film est génial !", "label": "positive"},
  {"text": "Quelle déception...", "label": "negative"},
  {"text": "Pas mal mais sans plus.", "label": "neutral"}
]
```

**Pour la génération (instruction-following)** :
```json
[
  {
    "instruction": "Traduis en anglais :",
    "input": "Bonjour, comment allez-vous ?",
    "output": "Hello, how are you?"
  },
  {
    "instruction": "Résume ce texte :",
    "input": "Les LLMs sont des modèles...[long texte]",
    "output": "Les LLMs sont des réseaux de neurones entraînés sur du texte."
  }
]
```

### Quantité de Données Nécessaire

| Tâche | Minimum | Recommandé | Optimal |
|-------|---------|------------|---------|
| **Classification binaire** | 100 | 1,000 | 10,000+ |
| **Classification multi-classe** | 50/classe | 500/classe | 5,000/classe |
| **NER** | 1,000 phrases | 10,000 | 100,000+ |
| **Génération** | 500 | 5,000 | 50,000+ |
| **QA** | 1,000 paires | 10,000 | 100,000+ |

### 💡 Analogie : L'Apprentissage Culinaire

- **Pré-training** : École de cuisine (5 ans) - Apprendre toutes les bases
- **Fine-tuning avec 10,000 exemples** : Stage de 6 mois en pâtisserie française
- **Fine-tuning avec 100 exemples** : Masterclass d'1 journée sur les macarons
- **Few-shot prompting** : Regarder 3 vidéos YouTube et essayer

Plus vous avez d'exemples, plus l'expertise est profonde !

### Préparation du Dataset

```python
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# 1. Charger vos données
data = [
    {"text": "Ce produit est excellent", "label": 1},
    {"text": "Très déçu de cet achat", "label": 0},
    # ... plus d'exemples
]

# 2. Créer un Dataset HuggingFace
dataset = Dataset.from_list(data)

# 3. Split train/validation/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Split validation
split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"Train: {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")

# 4. Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 5. Format PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

---

## Fine-Tuning Complet : Exemple de Classification

### Configuration

```python
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

# 1. Charger le modèle pré-entraîné
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # Classification binaire
)

# 2. Vérifier qu'on peut entraîner
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
```

### Hyperparamètres

```python
training_args = TrainingArguments(
    output_dir="./results",

    # Epochs et batch size
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    # Learning rate
    learning_rate=5e-5,  # Typique pour fine-tuning : 1e-5 à 5e-5
    warmup_ratio=0.1,    # 10% des steps en warmup

    # Optimisation
    weight_decay=0.01,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,

    # Évaluation
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Garder seulement les 2 meilleurs checkpoints

    # Logging
    logging_steps=100,
    logging_dir="./logs",

    # Reproductibilité
    seed=42,

    # Mixed precision (économise mémoire)
    fp16=True if torch.cuda.is_available() else False,

    # Meilleur modèle
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)
```

### Métriques

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(pred):
    """
    Calcule accuracy, precision, recall, F1.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### Entraînement

```python
# Créer le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Entraîner !
print("Starting training...")
trainer.train()

# Évaluation finale sur test set
print("\nEvaluating on test set...")
results = trainer.evaluate(test_dataset)
print(results)

# Sauvegarder le modèle
trainer.save_model("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")
```

### Résultats Typiques

```
Epoch 1/3: 100%|██████████| 625/625 [05:23<00:00, 1.93it/s]
Evaluation: {'accuracy': 0.87, 'f1': 0.86, 'loss': 0.34}

Epoch 2/3: 100%|██████████| 625/625 [05:21<00:00, 1.94it/s]
Evaluation: {'accuracy': 0.91, 'f1': 0.90, 'loss': 0.25}

Epoch 3/3: 100%|██████████| 625/625 [05:20<00:00, 1.95it/s]
Evaluation: {'accuracy': 0.92, 'f1': 0.91, 'loss': 0.23}

Test Results: {'accuracy': 0.915, 'f1': 0.908}
```

---

## Fine-Tuning pour la Génération (GPT-Style)

### Format Instruction-Tuning

**Structure Alpaca** (Stanford) :
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### Préparation des Données

```python
def format_instruction(example):
    """
    Formate un exemple au format instruction-following.
    """
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""

    # Pour training : prompt + output
    full_text = prompt + example['output']

    return {"text": full_text}

# Appliquer à tout le dataset
formatted_dataset = dataset.map(format_instruction)
```

### Tokenization avec Padding Gauche

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Important pour génération : padding à gauche
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """
    Tokenize avec attention mask.
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    # Labels = input_ids (pour causal LM)
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized

train_dataset = formatted_dataset.map(tokenize_function, batched=True)
train_dataset.set_format("torch")
```

### Fine-Tuning GPT

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Plus petit pour GPT
    gradient_accumulation_steps=4,  # Simuler batch size 16
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=1000,
    eval_steps=500,
    logging_steps=100,
    fp16=True,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

### Génération avec le Modèle Fine-Tuné

```python
from transformers import pipeline

# Charger le modèle fine-tuné
generator = pipeline(
    "text-generation",
    model="./gpt2-finetuned",
    tokenizer=tokenizer
)

# Tester
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Traduis en anglais :

### Input:
Bonjour, comment allez-vous ?

### Response:
"""

output = generator(
    prompt,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(output[0]['generated_text'])
# Expected: "Hello, how are you?"
```

---

## Hyperparamètres Critiques

### Learning Rate

**Règle d'or** : Fine-tuning nécessite LR **plus petit** que pré-training.

```python
# Pré-training : 1e-4 à 1e-3
# Fine-tuning : 1e-5 à 5e-5

# Trop haut : catastrophic forgetting
# Trop bas : sous-apprentissage
```

**Learning Rate Scheduling** :

```python
from transformers import get_linear_schedule_with_warmup

num_training_steps = len(train_dataloader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Utilisation
for epoch in range(num_epochs):
    for batch in train_dataloader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update LR
        optimizer.zero_grad()
```

### Nombre d'Epochs

| Taille Dataset | Epochs Recommandés |
|----------------|-------------------|
| < 1,000 | 10-20 |
| 1,000 - 10,000 | 3-5 |
| 10,000 - 100,000 | 2-3 |
| > 100,000 | 1-2 |

**Piège** : Plus de données → moins d'epochs nécessaires !

### Batch Size et Gradient Accumulation

```python
# Si GPU mémoire limitée, utiliser gradient accumulation

# Équivalent batch size 32 avec 4GB GPU:
per_device_train_batch_size = 4  # Ce qui tient en mémoire
gradient_accumulation_steps = 8  # 4 × 8 = 32 effective batch size

# Le gradient est accumulé sur 8 steps avant update
```

### Weight Decay

**Régularisation L2** pour éviter l'overfitting.

```python
weight_decay = 0.01  # Typique : 0.01 à 0.1

# Appliqué à tous les paramètres sauf biases et layer norms
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)
```

### Warmup

**Pourquoi ?** Éviter les gradients explosifs au début.

```python
warmup_ratio = 0.1  # 10% des steps en warmup

# LR augmente linéairement de 0 → learning_rate pendant warmup
# Puis décroît selon le schedule (linear, cosine, etc.)
```

---

## Éviter les Pièges Courants

### 1. Catastrophic Forgetting

**Problème** : Le modèle oublie ses connaissances générales.

**Exemple** :
```python
# Avant fine-tuning
prompt = "La capitale de la France est"
model.generate(prompt)  # "Paris"

# Après fine-tuning agressif sur domaine médical
model.generate(prompt)  # Génère du charabia ou texte médical
```

**Solutions** :

#### A) Learning Rate Faible
```python
learning_rate = 2e-5  # Au lieu de 5e-5
```

#### B) Moins d'Epochs
```python
num_epochs = 2  # Au lieu de 5
```

#### C) Mixte Training Data
```python
# 80% données domaine cible
# 20% données générales (échantillon du pré-training)

mixed_dataset = concatenate_datasets([
    target_domain_data,
    general_data.shuffle().select(range(len(target_domain_data) // 4))
])
```

#### D) Elastic Weight Consolidation (EWC)

Technique avancée : pénaliser les changements des poids importants.

```python
# Pseudo-code (implémentation complexe)
for name, param in model.named_parameters():
    # Calculer importance de chaque poids (Fisher Information)
    fisher_info = compute_fisher_information(param, old_dataset)

    # Loss = cross_entropy + λ × ∑(fisher × (θ_new - θ_old)²)
    ewc_loss = fisher_info * (param - old_param) ** 2
    total_loss = cross_entropy_loss + lambda_ewc * ewc_loss.sum()
```

### 2. Overfitting

**Symptômes** :
```
Epoch 1: Train Loss 0.5, Val Loss 0.6  ✅
Epoch 2: Train Loss 0.3, Val Loss 0.55 ✅
Epoch 3: Train Loss 0.1, Val Loss 0.7  ⚠️ Overfitting!
```

**Solutions** :

#### A) Early Stopping
```python
training_args = TrainingArguments(
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Stop si pas d'amélioration pendant 3 évals
    early_stopping_patience=3
)
```

#### B) Dropout
```python
# Augmenter dropout
model.config.hidden_dropout_prob = 0.2  # Défaut: 0.1
model.config.attention_probs_dropout_prob = 0.2
```

#### C) Data Augmentation
```python
import nlpaug.augmenter.word as naw

# Augmentation par synonymes
aug = naw.SynonymAug(aug_src='wordnet')

def augment_text(text):
    return aug.augment(text)

# Appliquer au dataset
augmented_data = [augment_text(ex['text']) for ex in data]
```

### 3. Class Imbalance

**Problème** : 90% classe A, 10% classe B → modèle prédit toujours A.

**Solutions** :

#### A) Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculer poids automatiquement
labels = [ex['label'] for ex in train_dataset]
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# Créer tenseur
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Loss avec poids
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

#### B) Oversampling / Undersampling
```python
from imblearn.over_sampling import RandomOverSampler

# Oversampler la classe minoritaire
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

#### C) Focal Loss
```python
class FocalLoss(nn.Module):
    """
    Focal Loss pour déséquilibre de classes (Lin et al. 2017).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## Fine-Tuning Avancé : Multi-Task Learning

**Idée** : Fine-tuner sur **plusieurs tâches** simultanément.

### Configuration Multi-Task

```python
# Dataset format
multi_task_data = [
    {"task": "sentiment", "text": "Ce film est super", "label": "positive"},
    {"task": "ner", "text": "Apple Inc. est à Cupertino", "entities": [...]},
    {"task": "qa", "context": "...", "question": "...", "answer": "..."}
]

# Modèle avec plusieurs têtes
class MultiTaskModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.sentiment_head = nn.Linear(768, 3)  # 3 classes sentiment
        self.ner_head = nn.Linear(768, 9)        # 9 tags NER
        self.qa_head = nn.Linear(768, 2)         # start/end positions

    def forward(self, input_ids, task_type, **kwargs):
        outputs = self.base(input_ids, **kwargs)
        hidden = outputs.last_hidden_state

        if task_type == "sentiment":
            return self.sentiment_head(hidden[:, 0, :])  # [CLS]
        elif task_type == "ner":
            return self.ner_head(hidden)  # Tous les tokens
        elif task_type == "qa":
            return self.qa_head(hidden)
```

**Avantages** :
- ✅ Meilleure généralisation
- ✅ Partage de connaissances entre tâches
- ✅ Un seul modèle pour plusieurs usages

---

## Évaluation du Modèle Fine-Tuné

### Métriques par Tâche

| Tâche | Métriques Principales |
|-------|----------------------|
| **Classification** | Accuracy, F1, Precision, Recall |
| **NER** | F1 par entité, Exact Match |
| **QA** | Exact Match, F1 token-level |
| **Génération** | BLEU, ROUGE, BERTScore, Humaine |
| **Résumé** | ROUGE-1/2/L |

### Test sur Distribution OOD (Out-of-Distribution)

```python
# Tester sur données jamais vues (autre domaine)
ood_test_set = load_dataset("different_domain")

ood_results = trainer.evaluate(ood_test_set)
print(f"In-domain accuracy: {in_domain_acc:.2%}")
print(f"Out-of-domain accuracy: {ood_results['accuracy']:.2%}")

# Si gap énorme → overfitting au domaine
```

### Tests d'Adversarialité

```python
from textattack import Attacker, Attack
from textattack.attack_recipes import TextFoolerJin2019

# Créer attaquant
attack = TextFoolerJin2019.build(model_wrapper)

# Tester robustesse
results = attack.attack_dataset(test_dataset, num_examples=100)
print(f"Attack success rate: {results.success_rate:.2%}")
# Si > 50% → modèle fragile
```

---

## Déploiement du Modèle Fine-Tuné

### Export et Optimisation

```python
# 1. Sauvegarder
model.save_pretrained("./final-model")
tokenizer.save_pretrained("./final-model")

# 2. Quantization (réduire taille 4×)
from transformers import AutoModelForSequenceClassification
import torch

model_int8 = AutoModelForSequenceClassification.from_pretrained(
    "./final-model",
    load_in_8bit=True,
    device_map="auto"
)

# 3. ONNX export (pour production)
from transformers.onnx import export

export(
    preprocessor=tokenizer,
    model=model,
    config=model.config.to_diff_dict(),
    opset=13,
    output=Path("model.onnx")
)
```

### API Inference

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Charger modèle au démarrage
classifier = pipeline(
    "text-classification",
    model="./final-model",
    device=0 if torch.cuda.is_available() else -1
)

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Endpoint de prédiction.
    """
    result = classifier(request.text)[0]
    return {
        "label": result['label'],
        "confidence": result['score']
    }

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Quiz Interactif

### Question 1 : Learning Rate

**Pourquoi utiliser un learning rate plus petit pour fine-tuning que pré-training ?**

A) Pour économiser GPU
B) Pour éviter catastrophic forgetting
C) C'est une erreur, LR devrait être identique
D) Pour accélérer la convergence

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Pour éviter catastrophic forgetting**

Le modèle pré-entraîné a déjà de bonnes représentations. Un LR élevé causerait des changements brusques → perte des connaissances générales.

**Typique** :
- Pré-training : 1e-4
- Fine-tuning : 5e-5 (2× plus petit)
</details>

---

### Question 2 : Nombre d'Epochs

**Vous avez 500 exemples d'entraînement. Combien d'epochs ?**

A) 1-2
B) 3-5
C) 10-20
D) 50+

<details>
<summary>Voir la réponse</summary>

**Réponse : C) 10-20**

Avec peu de données (< 1000), plusieurs epochs sont nécessaires pour que le modèle apprenne. Avec beaucoup de données (> 100k), 1-2 epochs suffisent.
</details>

---

### Question 3 : Catastrophic Forgetting

**Votre modèle fine-tuné sur jurisprudence française ne répond plus correctement à "Quelle est la capitale de l'Italie ?". Cause ?**

A) Bug du code
B) Catastrophic forgetting
C) Overfitting
D) Learning rate trop bas

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Catastrophic forgetting**

Le fine-tuning a écrasé les connaissances générales. Solutions :
- LR plus faible
- Moins d'epochs
- Mélanger 20% données générales
</details>

---

### Question 4 : Gradient Accumulation

**Votre GPU 8GB ne peut prendre que batch size 2, mais vous voulez effective batch size 16. Solution ?**

A) Impossible
B) Acheter plus de GPU
C) Gradient accumulation steps = 8
D) Réduire la taille du modèle

<details>
<summary>Voir la réponse</summary>

**Réponse : C) Gradient accumulation steps = 8**

```python
per_device_batch_size = 2
gradient_accumulation_steps = 8
# Effective batch size = 2 × 8 = 16
```

Le gradient est accumulé sur 8 forward passes avant l'optimizer step.
</details>

---

### Question 5 : Class Imbalance

**Dataset : 95% négatif, 5% positif. Accuracy 95% après fine-tuning. Est-ce bon ?**

A) Oui, excellent !
B) Non, le modèle prédit probablement toujours "négatif"
C) Impossible à dire
D) C'est le maximum possible

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Non, le modèle prédit probablement toujours "négatif"**

95% accuracy = baseline (toujours prédire la classe majoritaire). Vérifier :
- Confusion matrix
- F1 score (prend en compte précision ET rappel)
- Accuracy par classe

Solutions : class weights, focal loss, resampling.
</details>

---

## Exercices Pratiques

### Exercice 1 : Fine-Tuner BERT pour Sentiment Analysis

**Objectif** : Classification d'avis IMDb (positif/négatif).

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# TODO:
# 1. Charger le dataset IMDb
# 2. Tokeniser avec bert-base-uncased
# 3. Fine-tuner pour 3 epochs
# 4. Évaluer sur test set
# 5. Tester sur vos propres phrases

# Starter code
dataset = load_dataset("imdb")
# ...
```

<details>
<summary>Voir la solution</summary>

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Tokenization
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(preprocess, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. Training
training_args = TrainingArguments(
    output_dir="./imdb-bert",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"].shuffle().select(range(5000)),  # Subset pour rapidité
    eval_dataset=tokenized["test"].shuffle().select(range(1000)),
    compute_metrics=compute_metrics
)

# 5. Train
trainer.train()

# 6. Test
test_texts = [
    "This movie was absolutely fantastic! I loved every minute.",
    "Waste of time. Terrible acting and boring plot."
]

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(-1).item()
    label = "Positive" if pred == 1 else "Negative"
    print(f"{text[:50]}... → {label}")
```
</details>

---

### Exercice 2 : Learning Rate Finder

**Objectif** : Trouver le meilleur learning rate automatiquement.

```python
import torch
import matplotlib.pyplot as plt

def find_lr(model, train_dataloader, optimizer, device, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    Implémente LR range test (Smith, 2017).

    Principe : augmenter LR exponentiellement et tracer la loss.
    Le meilleur LR est juste avant que la loss explose.
    """
    # TODO: Implémenter
    pass

# Test
# optimal_lr = find_lr(model, train_dataloader, optimizer, device)
# print(f"Optimal LR: {optimal_lr}")
```

<details>
<summary>Voir la solution</summary>

```python
def find_lr(model, train_dataloader, optimizer, device, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    LR range test (Leslie Smith, 2017).
    """
    model.train()
    lrs = []
    losses = []

    # Sauvegarder état initial
    initial_state = model.state_dict()

    # Augmenter LR exponentiellement
    lr_mult = (max_lr / min_lr) ** (1 / num_steps)
    lr = min_lr

    for step, batch in enumerate(train_dataloader):
        if step >= num_steps:
            break

        # Forward
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log
        lrs.append(lr)
        losses.append(loss.item())

        # Augmenter LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lr *= lr_mult

        # Stop si loss explose
        if loss.item() > losses[0] * 4:
            break

    # Restaurer modèle
    model.load_state_dict(initial_state)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    plt.show()

    # Meilleur LR : pente la plus forte (loss décroît le plus vite)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]

    return optimal_lr

# Exemple d'utilisation
# optimal_lr = find_lr(model, train_dataloader, optimizer, device)
# print(f"Optimal LR: {optimal_lr:.2e}")
```
</details>

---

### Exercice 3 : Implémenter Early Stopping Custom

**Objectif** : Arrêter l'entraînement si pas d'amélioration pendant N epochs.

```python
class EarlyStopping:
    """
    Early stopping pour éviter overfitting.
    """
    def __init__(self, patience=3, min_delta=0.0):
        """
        Args:
            patience: Nombre d'epochs sans amélioration avant stop
            min_delta: Amélioration minimum considérée comme significative
        """
        # TODO: Implémenter
        pass

    def __call__(self, val_loss):
        """
        Returns:
            True si doit arrêter, False sinon
        """
        # TODO: Implémenter
        pass

# Utilisation
# early_stopping = EarlyStopping(patience=3)
# for epoch in range(num_epochs):
#     train(...)
#     val_loss = evaluate(...)
#     if early_stopping(val_loss):
#         print(f"Early stopping at epoch {epoch}")
#         break
```

<details>
<summary>Voir la solution</summary>

```python
class EarlyStopping:
    """
    Early stopping pour éviter overfitting.
    """
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Nombre d'epochs sans amélioration avant stop
            min_delta: Amélioration minimum considérée comme significative
            mode: 'min' (loss) ou 'max' (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Args:
            score: Métrique à surveiller (loss ou accuracy)

        Returns:
            True si doit arrêter, False sinon
        """
        if self.best_score is None:
            self.best_score = score
            return False

        # Vérifier amélioration
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

# Exemple d'utilisation
early_stopping = EarlyStopping(patience=3, mode='min')

for epoch in range(20):
    # Training
    train_loss = train_one_epoch(model, train_loader, optimizer)

    # Validation
    val_loss = evaluate(model, val_loader)

    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")

    # Check early stopping
    if early_stopping(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

print(f"Best validation loss: {early_stopping.best_score:.4f}")
```
</details>

---

## Conclusion

### 🎭 Dialogue Final : Le Fine-Tuning, Clé de la Personnalisation

**Alice** : Maintenant je comprends : le fine-tuning transforme un "étudiant généraliste" en "expert spécialisé" !

**Bob** : Exactement. Et le coût est incroyablement faible comparé au pré-training :
- GPT-3 pré-training : $5M
- Fine-tuning GPT-3 sur ton domaine : $100-1000

**Alice** : Quels sont les choix cruciaux ?

**Bob** :
1. **Données** : Qualité > Quantité (1000 bons exemples > 10000 mauvais)
2. **Learning rate** : Petit (2e-5) pour éviter forgetting
3. **Epochs** : Peu (2-3) pour éviter overfitting
4. **Évaluation** : Métriques domaine + test OOD

**Alice** : Et les alternatives au full fine-tuning ?

**Bob** : On en parlera au chapitre 13 :
- **LoRA** : Fine-tuner seulement 0.1% des paramètres
- **Prompt tuning** : Optimiser les prompts, pas le modèle
- **Adapter layers** : Insérer petites couches entraînables

Le futur, c'est l'efficacité !

### 🎯 Points Clés à Retenir

| Concept | Essence |
|---------|---------|
| **Fine-tuning** | Continuer l'entraînement sur domaine spécifique |
| **LR** | 2e-5 à 5e-5 (10× plus petit que pré-training) |
| **Epochs** | 2-3 pour gros dataset, 10-20 pour petit |
| **Catastrophic forgetting** | Modèle oublie connaissances générales |
| **Solutions forgetting** | LR faible, moins epochs, données mixtes |
| **Overfitting** | Train loss ↓ mais val loss ↑ |
| **Solutions overfitting** | Early stopping, dropout, data augmentation |

### 📊 Checklist Fine-Tuning

**Avant de commencer** :
- [ ] Données nettoyées et labelisées
- [ ] Split train/val/test (70/15/15)
- [ ] Baseline établie (modèle simple)
- [ ] Métriques définies

**Pendant l'entraînement** :
- [ ] Monitor train ET validation loss
- [ ] Sauvegarder checkpoints réguliers
- [ ] Tester sur échantillons durant training
- [ ] Utiliser early stopping

**Après fine-tuning** :
- [ ] Évaluation complète sur test set
- [ ] Test OOD (données hors distribution)
- [ ] Tests adversariaux
- [ ] Vérifier pas de catastrophic forgetting
- [ ] Optimiser pour production (quantization, ONNX)

---

## Ressources

### 📚 Papers Fondamentaux

1. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., 2018)
2. **"Universal Language Model Fine-tuning for Text Classification"** (ULMFiT, Howard & Ruder, 2018)
3. **"Parameter-Efficient Transfer Learning for NLP"** (Houlsby et al., 2019)
4. **"The Power of Scale for Parameter-Efficient Prompt Tuning"** (Lester et al., 2021)

### 🛠️ Code et Tutoriels

```bash
# HuggingFace Transformers
pip install transformers datasets evaluate accelerate

# Fine-tuning rapide
pip install autotrain-advanced
```

**Ressources** :
- HuggingFace Course : https://huggingface.co/course/chapter3
- Fine-Tuning Guide : https://huggingface.co/docs/transformers/training
- Example Scripts : https://github.com/huggingface/transformers/tree/main/examples/pytorch

---

**🎓 Bravo !** Vous maîtrisez maintenant le fine-tuning complet. Prochain chapitre : **Chapter 10 - Optimization Techniques** pour rendre tout ça plus rapide et efficient ! 🚀

