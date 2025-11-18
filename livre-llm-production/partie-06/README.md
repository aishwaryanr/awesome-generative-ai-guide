# Partie 6 : Pré-training - Entraîner un LLM de zéro

## Objectifs d'apprentissage

- Maîtriser l'objectif Next Token Prediction (NTP)
- Concevoir la configuration d'un modèle et ses hyperparamètres
- Orchestrer l'entraînement distribué (FSDP, ZeRO, parallelisme)
- Mettre en place monitoring, checkpointing et reprise
- Conduire un pré-training complet sur un modèle de taille réelle

## Prérequis

- Parties 1-5 validées
- PyTorch avancé et distributed training
- Accès à infrastructure GPU (idéalement multi-GPU)

---

## 6.1 Objectif d'entraînement : Next Token Prediction

### 6.1.1 Formulation mathématique

**Objectif** : Maximiser la probabilité des données observées.

```
L(θ) = -Σ log P_θ(token_i | token_1, ..., token_{i-1})
```

Pour un document de longueur T :

```
L = -1/T Σ_{i=1}^T log P_θ(x_i | x_{<i})
```

**Causal masking** : Le modèle ne peut voir que les tokens précédents.

### 6.1.2 Implémentation en PyTorch

```python
import torch
import torch.nn.functional as F

def compute_loss(model, input_ids):
    """
    input_ids: [batch, seq_len]
    """
    # Forward pass
    logits = model(input_ids)  # [batch, seq_len, vocab_size]

    # Shift pour aligner input et target
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Cross-entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100  # padding
    )

    return loss
```

### 6.1.3 Masquage et padding

**Causal mask** : Empêcher l'attention future.

```python
def create_causal_mask(seq_len):
    """Crée un masque triangulaire inférieur."""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
```

**Padding mask** : Ignorer les tokens de padding dans la loss.

```python
# Labels avec padding à -100 (ignoré par cross_entropy)
labels = input_ids.clone()
labels[labels == tokenizer.pad_token_id] = -100
```

---

## 6.2 Configuration du modèle

### 6.2.1 Choix des hyperparamètres architecturaux

**Dimensions clés** :

```python
config = {
    "vocab_size": 50257,          # Taille du vocabulaire
    "max_seq_len": 2048,          # Longueur max de contexte
    "d_model": 2048,              # Dimension des embeddings
    "num_layers": 24,             # Nombre de blocs Transformer
    "num_heads": 16,              # Têtes d'attention
    "d_ff": 8192,                 # Dimension FFN (généralement 4×d_model)
    "dropout": 0.1,               # Taux de dropout
}
```

**Scaling selon la taille du modèle** :

| Taille    | d_model | num_layers | num_heads | Params totaux |
|-----------|---------|------------|-----------|---------------|
| Small     | 768     | 12         | 12        | ~125M         |
| Base      | 1024    | 24         | 16        | ~350M         |
| Large     | 1536    | 36         | 16        | ~1.3B         |
| XL        | 2048    | 48         | 16        | ~2.7B         |
| XXL       | 4096    | 64         | 32        | ~13B          |

### 6.2.2 Initialisation des poids

**Importance** : Bonne initialisation = convergence stable.

**Xavier/Glorot** (pour couches linéaires) :

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

model.apply(init_weights)
```

**GPT-style initialization** (variance scaled) :

```python
# Pour GPT-2/GPT-3
std = 0.02
nn.init.normal_(module.weight, mean=0.0, std=std)
```

**Scaled initialization for deep networks** :

Pour un modèle de N couches, réduire la variance des couches résiduelles :

```python
# Scaling par √(2N) comme dans GPT-2
for layer_idx, layer in enumerate(model.layers):
    layer.attn.proj.weight.data /= math.sqrt(2 * len(model.layers))
    layer.ffn.proj.weight.data /= math.sqrt(2 * len(model.layers))
```

---

## 6.3 Hyperparamètres d'entraînement

### 6.3.1 Batch size et accumulation

**Global batch size** : Nombre total d'exemples par step.

```
Global batch size = micro_batch × grad_accum_steps × num_gpus
```

**Exemple** :
- 8 GPUs
- Micro-batch = 4 (par GPU)
- Grad accumulation = 8
- Global batch = 4 × 8 × 8 = 256

**Implémentation** :

```python
micro_batch_size = 4
grad_accum_steps = 8

optimizer.zero_grad()

for step in range(num_steps):
    accum_loss = 0

    for accum_step in range(grad_accum_steps):
        batch = next(dataloader)
        loss = compute_loss(model, batch)

        # Normaliser par grad_accum_steps
        loss = loss / grad_accum_steps
        loss.backward()

        accum_loss += loss.item()

    # Update après accumulation
    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {accum_loss:.4f}")
```

### 6.3.2 Learning rate et scheduling

**Warmup + Cosine Decay** (standard LLM) :

```python
import math

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step / warmup_steps)
    elif step < max_steps:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return min_lr

# Scheduler PyTorch
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(step):
    return get_lr(step, warmup_steps=2000, max_steps=100000,
                  max_lr=1.0, min_lr=0.1)

scheduler = LambdaLR(optimizer, lr_lambda)
```

**Valeurs typiques** :
- Learning rate max : 1e-4 à 6e-4 (dépend de la taille du modèle)
- Warmup : 1-5% du total des steps
- Min LR : 10% de max LR

### 6.3.3 Optimiseur et régularisation

**AdamW** (standard pour LLM) :

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),        # β1, β2
    eps=1e-8,
    weight_decay=0.1          # Régularisation L2
)
```

**Gradient clipping** :

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Label smoothing** (optionnel) :

```python
def cross_entropy_with_smoothing(logits, targets, smoothing=0.1):
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=-1)

    # True class proba
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Smooth uniform
    smooth_loss = -log_probs.mean(dim=-1)

    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()
```

---

## 6.4 Parallélisme distribué

### 6.4.1 Types de parallélisme

**Data Parallelism** : Chaque GPU a une copie complète du modèle.

**Tensor Parallelism** : Diviser les tenseurs du modèle sur plusieurs GPUs.

**Pipeline Parallelism** : Diviser les couches du modèle sur plusieurs GPUs.

**FSDP/ZeRO** : Sharding des paramètres, gradients, optimizer states.

### 6.4.2 Fully Sharded Data Parallel (FSDP)

**Principe** : Chaque GPU ne stocke qu'une fraction des paramètres.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
import torch.distributed as dist

# Initialiser process group
dist.init_process_group(backend='nccl')

# Wrapping du modèle avec FSDP
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device(),
)
```

**Avantages** :
- Permet d'entraîner des modèles beaucoup plus grands
- Réduction mémoire ~ N (nombre de GPUs)

### 6.4.3 ZeRO (DeepSpeed)

**ZeRO-1** : Shard optimizer states
**ZeRO-2** : + gradients
**ZeRO-3** : + paramètres (équivalent à FSDP)

```python
import deepspeed

ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},  # Offload sur CPU si nécessaire
        "offload_param": {"device": "cpu"}
    },
    "bf16": {"enabled": True}
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

### 6.4.4 Mixed Precision Training

**bfloat16** (préféré pour LLM) :

```python
from torch.cuda.amp import autocast

with autocast(dtype=torch.bfloat16):
    logits = model(input_ids)
    loss = compute_loss(logits, labels)

loss.backward()
optimizer.step()
```

**Avantages bfloat16 vs float16** :
- Même plage dynamique que float32
- Pas besoin de loss scaling
- Plus stable pour l'entraînement de grands modèles

---

## 6.5 Monitoring et checkpointing

### 6.5.1 Métriques à suivre

**Loss et perplexité** :

```python
def log_metrics(step, loss, learning_rate):
    perplexity = math.exp(loss)
    print(f"Step {step} | Loss: {loss:.4f} | PPL: {perplexity:.2f} | LR: {learning_rate:.2e}")

    # Log vers TensorBoard / W&B
    if use_wandb:
        wandb.log({
            "train/loss": loss,
            "train/perplexity": perplexity,
            "train/learning_rate": learning_rate,
        }, step=step)
```

**Gradient norms** (détecter explosions) :

```python
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

wandb.log({"train/grad_norm": total_norm}, step=step)
```

**Utilisation GPU** :

```python
import torch

allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### 6.5.2 Checkpointing

**Sauvegarder régulièrement** :

```python
def save_checkpoint(model, optimizer, scheduler, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
```

**Reprendre l'entraînement** :

```python
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    step = checkpoint["step"]

    print(f"Resumed from step {step}")
    return step
```

### 6.5.3 Validation et early stopping

**Évaluer périodiquement sur validation set** :

```python
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in val_loader:
        loss = compute_loss(model, batch)
        total_loss += loss.item()
        num_batches += 1

    model.train()
    avg_loss = total_loss / num_batches
    return avg_loss

# Pendant l'entraînement
if step % eval_interval == 0:
    val_loss = evaluate(model, val_loader)
    val_ppl = math.exp(val_loss)
    print(f"Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")
    wandb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)
```

---

## 6.6 Lab : Entraînement complet d'un modèle small

**Objectif** : Entraîner un modèle de 125M paramètres sur un petit corpus.

### 6.6.1 Configuration

```python
config = {
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "dropout": 0.1,
}

training_config = {
    "micro_batch_size": 8,
    "grad_accum_steps": 4,
    "max_steps": 10000,
    "lr": 6e-4,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "eval_interval": 500,
    "save_interval": 1000,
}
```

### 6.6.2 Script d'entraînement

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Initialiser W&B
wandb.init(project="llm-pretraining", config=config)

# Modèle
model = GPTModel(config).cuda()

# Optimiseur
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config["lr"],
    betas=(0.9, 0.95),
    weight_decay=training_config["weight_decay"]
)

# Scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_config["warmup_steps"],
    num_training_steps=training_config["max_steps"]
)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=training_config["micro_batch_size"])
val_loader = DataLoader(val_dataset, batch_size=training_config["micro_batch_size"])

# Training loop
model.train()
step = 0

for epoch in range(100):  # Epochs larges, break via max_steps
    for batch in train_loader:
        # Accumulation de gradients
        loss = compute_loss(model, batch.cuda())
        loss = loss / training_config["grad_accum_steps"]
        loss.backward()

        if (step + 1) % training_config["grad_accum_steps"] == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if step % 10 == 0:
                log_metrics(step, loss.item() * training_config["grad_accum_steps"], scheduler.get_last_lr()[0])

        # Validation
        if step % training_config["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader)
            print(f"Step {step} | Val Loss: {val_loss:.4f}")

        # Checkpointing
        if step % training_config["save_interval"] == 0:
            save_checkpoint(model, optimizer, scheduler, step, "checkpoints/")

        step += 1
        if step >= training_config["max_steps"]:
            break

    if step >= training_config["max_steps"]:
        break

print("Training complete!")
```

---

## 6.7 Références

### Papers
- Kaplan et al. (2020) - "Scaling Laws for Neural Language Models"
- Hoffmann et al. (2022) - "Training Compute-Optimal Large Language Models" (Chinchilla)
- Rajbhandari et al. (2020) - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"

### Outils
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed](https://www.deepspeed.ai/)
- [Weights & Biases](https://wandb.ai/)

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Objectif Next Token Prediction et masquage causal
- ✅ Configuration modèle et hyperparamètres d'entraînement
- ✅ Parallélisme distribué (FSDP, ZeRO, mixed precision)
- ✅ Monitoring, checkpointing, validation
- ✅ Entraînement complet d'un modèle LLM

**Prochaine étape** : [Partie 7 - Post-training et alignement](../partie-07/README.md)
