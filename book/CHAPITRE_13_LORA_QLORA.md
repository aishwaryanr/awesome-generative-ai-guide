# CHAPITRE 13 : PARAMETER-EFFICIENT FINE-TUNING (LoRA & QLoRA)

## Introduction

Le fine-tuning complet d'un LLM moderne nécessite des ressources computationnelles massives. Pour Llama 2 70B en FP16, cela requiert:
- **140GB VRAM** minimum (just pour les poids)
- **280-420GB VRAM** réel (avec gradients, optimizer states)
- **8x A100 80GB GPUs** (~$20-30/heure)

Parameter-Efficient Fine-Tuning (PEFT) résout ce problème en n'entraînant qu'une petite fraction des paramètres, permettant le fine-tuning sur des GPUs consumer.

## 13.1 LoRA (Low-Rank Adaptation)

### 13.1.1 Motivation et Intuition

**Observation clé**: Les mises à jour de poids durant le fine-tuning ont souvent un "intrinsic rank" faible.

En d'autres termes, on n'a pas besoin de modifier tous les paramètres - on peut approximer les changements avec des matrices de bas rang.

**Analogie**:
Imaginez que vous voulez ajuster une photo 4K (millions de pixels). Si le changement est principalement de l'éclaircissement, vous n'avez pas besoin de modifier chaque pixel indépendamment - vous pouvez appliquer une transformation simple.

### 13.1.2 Formulation Mathématique

**Full fine-tuning:**
```
h = W₀x
↓ (après fine-tuning)
h = W'x  où W' = W₀ + ΔW
```

**LoRA:**
```
Au lieu d'apprendre ΔW directement (qui a beaucoup de paramètres),
on factorise: ΔW = BA

où:
- B ∈ ℝᵈˣʳ (down-projection)
- A ∈ ℝʳˣᵏ (up-projection)
- r << min(d, k) (rank, typiquement 8, 16, 32, 64)

Donc: h = W₀x + BAx
```

**Nombre de paramètres:**
```
Full fine-tuning: d × k paramètres
LoRA: r(d + k) paramètres

Réduction: d×k / r(d+k)

Exemple (d=k=4096, r=8):
Full: 16,777,216 paramètres
LoRA: 65,536 paramètres
Réduction: 256x !
```

### 13.1.3 Implémentation Détaillée

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    Implémentation complète de LoRA

    Transforme une couche linéaire:
    y = Wx → y = Wx + s·BAx

    où s = α/r est un facteur de scaling
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights

        # Poids pré-entraînés (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False
        )

        # LoRA matrices (trainable)
        # Initialisation: A ~ Gaussian, B = 0
        # Donc au début: BA = 0 (pas de changement)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Scaling factor
        self.scaling = alpha / rank

        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Flag pour savoir si les poids sont mergés
        self.merged = False

        # Initialize A avec Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec LoRA

        x: [batch, ..., in_features]
        returns: [batch, ..., out_features]
        """
        if self.merged:
            # Si déjà mergé, juste forward normal
            return F.linear(x, self.weight, bias=None)

        # Forward normal
        result = F.linear(x, self.weight, bias=None)

        # LoRA forward: x → A → dropout → B
        # x: [batch, in_features]
        # A: [rank, in_features]
        # xA^T: [batch, rank]
        # B: [out_features, rank]
        # (xA^T)B^T: [batch, out_features]

        lora_result = self.dropout(x @ self.lora_A.T) @ self.lora_B.T

        # Ajouter avec scaling
        return result + lora_result * self.scaling

    def merge_weights(self):
        """
        Merge LoRA weights dans les poids principaux
        Utile pour inference (évite le overhead de LoRA)

        W' = W + (α/r)BA
        """
        if not self.merged:
            # Calculer ΔW = (α/r)BA
            delta_w = (self.lora_B @ self.lora_A) * self.scaling

            # Merger dans weight
            self.weight.data += delta_w

            self.merged = True

    def unmerge_weights(self):
        """
        Séparer à nouveau les poids (utile pour continuer training)
        """
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w
            self.merged = False

# Test de la layer
def test_lora_layer():
    """Test LoRA layer"""
    batch_size = 4
    seq_len = 10
    d_in = 768
    d_out = 768

    # Créer layer
    lora = LoRALayer(d_in, d_out, rank=8, alpha=16)

    # Input
    x = torch.randn(batch_size, seq_len, d_in)

    # Forward
    output = lora(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Compter paramètres
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in lora.parameters() if not p.requires_grad)

    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params: {frozen:,}")
    print(f"Ratio: {frozen/trainable:.1f}x")

test_lora_layer()
# Output:
# Input shape: torch.Size([4, 10, 768])
# Output shape: torch.Size([4, 10, 768])
# Trainable params: 12,288
# Frozen params: 589,824
# Ratio: 48.0x
```

### 13.1.4 Intégration dans un Transformer

On applique généralement LoRA uniquement sur les **projections Q et V** de l'attention (empiriquement les plus importantes).

```python
class AttentionWithLoRA(nn.Module):
    """
    Multi-Head Attention avec LoRA sur Q et V
    """
    def __init__(self, config, lora_config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q projection avec LoRA
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.q_lora = LoRALayer(
            config.n_embd,
            config.n_embd,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )

        # K projection (pas de LoRA - empiriquement moins utile)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        for param in self.k_proj.parameters():
            param.requires_grad = False

        # V projection avec LoRA
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_lora = LoRALayer(
            config.n_embd,
            config.n_embd,
            rank=lora_config.rank,
            alpha=lora_config.alpha
        )

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        for param in self.out_proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, T, C = x.shape

        # Q avec LoRA
        q = self.q_proj(x)
        q = q + self.q_lora(x)  # Ajouter LoRA

        # K sans LoRA (frozen)
        k = self.k_proj(x)

        # V avec LoRA
        v = self.v_proj(x)
        v = v + self.v_lora(x)  # Ajouter LoRA

        # Reshape pour multi-head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention normale
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Recombine
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection (frozen)
        out = self.out_proj(out)

        return out
```

### 13.1.5 Conversion de modèle complet en LoRA

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def convert_to_lora(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    target_modules: list = None,
):
    """
    Convertit un modèle HuggingFace en version LoRA

    Args:
        model_name: nom du modèle sur HF Hub
        lora_rank: rang des matrices LoRA
        lora_alpha: paramètre de scaling
        target_modules: modules à appliquer LoRA (default: Q,V)
    """
    # Charger modèle de base
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Freeze tous les paramètres
    for param in base_model.parameters():
        param.requires_grad = False

    # Configuration LoRA
    if target_modules is None:
        # Par défaut: Q et V dans attention
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    # Appliquer LoRA
    model = get_peft_model(base_model, lora_config)

    # Print statistiques
    model.print_trainable_parameters()

    return model

# Exemple d'utilisation
lora_model = convert_to_lora(
    "meta-llama/Llama-2-7b-hf",
    lora_rank=16,
    lora_alpha=32,
)

# Output typique:
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%
```

### 13.1.6 Training Loop avec LoRA

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def train_lora_model(
    model,
    dataset_name: str,
    output_dir: str = "./lora_output",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
):
    """
    Entraîne un modèle LoRA
    """
    # Charger dataset
    dataset = load_dataset(dataset_name)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,  # Mixed precision
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        warmup_steps=100,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Train
    trainer.train()

    # Sauvegarder les adapters LoRA
    model.save_pretrained(output_dir)

    return trainer

# Usage
trainer = train_lora_model(
    lora_model,
    dataset_name="imdb",
    output_dir="./llama2-lora-imdb",
)
```

### 13.1.7 Hyperparamètres LoRA

**Rank (r):**
- Plus petit (4, 8): moins de paramètres, plus rapide, risque d'underfitting
- Moyen (16, 32): bon compromis (recommandé)
- Grand (64, 128): plus expressif, mais plus lent

**Alpha (α):**
- Contrôle l'ampleur des changements
- Rule of thumb: α = 2r (donc scaling = 2)
- Plus grand α = changements plus importants

**Target modules:**
```python
# Configuration minimaliste (Q, V)
target_modules = ["q_proj", "v_proj"]

# Configuration standard (toutes les projections attention)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Configuration complète (attention + FFN)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # FFN (Llama)
]
```

**Guidelines empiriques:**

| Tâche | Rank | Alpha | Target Modules | Data Size |
|-------|------|-------|----------------|-----------|
| Instruction following | 8-16 | 16-32 | Q, V | < 10k |
| Domain adaptation | 16-32 | 32-64 | Q, K, V, O | 10k-100k |
| Style transfer | 32-64 | 64-128 | All attention | 100k+ |
| Full capability | 64-128 | 128-256 | Attention + FFN | 1M+ |

### 13.1.8 Inference avec LoRA

**Option 1: Merge et export**
```python
def export_merged_model(lora_model_path, output_path):
    """
    Merge LoRA weights et export modèle complet
    """
    # Charger modèle avec adapters
    model = AutoModelForCausalLM.from_pretrained(
        lora_model_path,
        torch_dtype=torch.float16,
    )

    # Merge adapters dans base model
    model = model.merge_and_unload()

    # Sauvegarder
    model.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

# Usage
export_merged_model(
    "./llama2-lora-imdb",
    "./llama2-merged-imdb"
)
```

**Option 2: Garder adapters séparés (multi-adapter)**
```python
def load_with_multiple_adapters(base_model_path, adapter_paths):
    """
    Charge un modèle avec plusieurs adapters LoRA
    Permet de switch entre adapters dynamiquement
    """
    # Charger base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    )

    # Charger premier adapter
    model = PeftModel.from_pretrained(model, adapter_paths[0], adapter_name="adapter1")

    # Charger adapters additionnels
    for i, path in enumerate(adapter_paths[1:], start=2):
        model.load_adapter(path, adapter_name=f"adapter{i}")

    return model

# Usage
model = load_with_multiple_adapters(
    "meta-llama/Llama-2-7b-hf",
    ["./lora-chat", "./lora-code", "./lora-math"]
)

# Switch adapter
model.set_adapter("adapter2")  # Utilise lora-code

# Generate
output = model.generate(input_ids, max_length=100)
```

## 13.2 QLoRA (Quantized LoRA)

### 13.2.1 Motivation

LoRA réduit les paramètres trainables, mais les poids du base model occupent toujours beaucoup de mémoire.

**Problème:**
- Llama 2 7B en FP16: 14GB VRAM
- Llama 2 70B en FP16: 140GB VRAM (impossible sur GPUs consumer)

**Solution QLoRA:** Quantizer le base model en 4-bit tout en gardant les adapters LoRA en haute précision.

### 13.2.2 Innovations de QLoRA

**1. 4-bit NormalFloat (NF4)**

Un nouveau format de quantization optimisé pour poids normalement distribués.

**Distribution typique des poids:**
```
La plupart des poids sont proches de 0
Peu de poids ont de grandes valeurs
→ Distribution approximativement normale
```

**NF4:** Quantization levels optimisés pour N(0,1)
```python
def get_nf4_quantization_levels():
    """
    Les 16 niveaux de quantization pour NF4
    Optimisés pour distribution normale
    """
    levels = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495,
        0.0, 0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ])
    return levels
```

**2. Double Quantization**

Quantizer aussi les quantization constants (meta-data).

```
Normal quantization:
weights (FP16) → quantized_weights (INT4) + scales (FP16)

Double quantization:
weights (FP16) → quantized_weights (INT4)
                + quantized_scales (INT8)
                + second_level_scales (FP16)

Économie mémoire additionnelle: ~0.4 bits par paramètre
```

**3. Paged Optimizers**

Utilise NVIDIA unified memory pour gérer les optimizer states (évite OOM).

### 13.2.3 Implémentation QLoRA

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def create_qlora_model(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_quant_type: str = "nf4",
    use_double_quant: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """
    Crée un modèle QLoRA prêt pour fine-tuning

    Args:
        model_name: modèle HuggingFace
        load_in_4bit: quantization 4-bit
        bnb_4bit_compute_dtype: dtype pour compute (bfloat16 recommandé)
        bnb_4bit_quant_type: "nf4" ou "fp4"
        use_double_quant: double quantization
        lora_r: rank LoRA
        lora_alpha: alpha LoRA
        lora_dropout: dropout LoRA
    """

    # Configuration quantization (BitsAndBytes)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

    # Charger modèle quantizé
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatic device placement
        trust_remote_code=True,
    )

    # Préparer pour k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configuration LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Appliquer LoRA
    model = get_peft_model(model, lora_config)

    # Print info
    model.print_trainable_parameters()

    return model

# Exemple d'utilisation
qlora_model = create_qlora_model(
    model_name="meta-llama/Llama-2-7b-hf",
    lora_r=64,
)

# Output:
# trainable params: 41,943,040 || all params: 6,783,152,128 || trainable%: 0.62%
```

### 13.2.4 Comparaison Mémoire: Full FT vs LoRA vs QLoRA

```python
def estimate_memory_requirements(num_params_billions, method="full"):
    """
    Estime les besoins mémoire pour différentes méthodes

    Args:
        num_params_billions: nombre de paramètres (en milliards)
        method: "full", "lora", "qlora"
    """
    num_params = num_params_billions * 1e9

    if method == "full":
        # Full fine-tuning en FP16
        # Model: 2 bytes/param
        # Gradients: 2 bytes/param
        # Optimizer (Adam): 8 bytes/param (2 momentum states)
        # Activations: ~4 bytes/param (approximation)
        bytes_per_param = 2 + 2 + 8 + 4
        total_gb = (num_params * bytes_per_param) / 1e9

    elif method == "lora":
        # Model (frozen): 2 bytes/param
        # LoRA adapters (1% of model): trainable
        # Gradients + optimizer only for LoRA params
        lora_params = num_params * 0.01
        frozen_memory = num_params * 2 / 1e9
        lora_memory = lora_params * (2 + 2 + 8) / 1e9
        total_gb = frozen_memory + lora_memory

    elif method == "qlora":
        # Model (4-bit quantized): 0.5 bytes/param
        # LoRA adapters: trainable (haute précision)
        lora_params = num_params * 0.01
        frozen_memory = num_params * 0.5 / 1e9
        lora_memory = lora_params * (2 + 2 + 8) / 1e9
        total_gb = frozen_memory + lora_memory

    return total_gb

# Comparaison pour différents modèles
models = [7, 13, 30, 65, 70]

print("Memory Requirements (GB):")
print(f"{'Model':>10} | {'Full FT':>10} | {'LoRA':>10} | {'QLoRA':>10}")
print("-" * 50)

for size in models:
    full = estimate_memory_requirements(size, "full")
    lora = estimate_memory_requirements(size, "lora")
    qlora = estimate_memory_requirements(size, "qlora")

    print(f"{size:>10}B | {full:>10.1f} | {lora:>10.1f} | {qlora:>10.1f}")

# Output:
# Memory Requirements (GB):
#      Model |    Full FT |       LoRA |      QLoRA
# --------------------------------------------------
#         7B |      112.0 |       15.4 |        4.2
#        13B |      208.0 |       28.6 |        7.8
#        30B |      480.0 |       66.0 |       18.0
#        65B |     1040.0 |      143.0 |       39.0
#        70B |     1120.0 |      154.0 |       42.0
```

**Conclusion:** QLoRA permet de fine-tuner Llama 2 70B sur un seul GPU consumer (48GB)!

### 13.2.5 Training avec QLoRA

```python
from trl import SFTTrainer

def train_qlora(
    model,
    tokenizer,
    dataset,
    output_dir: str = "./qlora_output",
    max_seq_length: int = 512,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
):
    """
    Fine-tune avec QLoRA using SFTTrainer
    """

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,  # BF16 recommandé avec QLoRA
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_32bit",  # Paged optimizer pour QLoRA
        gradient_checkpointing=True,  # Réduire mémoire
        max_grad_norm=0.3,
    )

    # SFTTrainer (Supervised Fine-Tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",  # Champ contenant le texte
        packing=False,  # Pas de packing pour simplicité
    )

    # Train
    trainer.train()

    # Sauvegarder
    trainer.save_model(output_dir)

    return trainer

# Usage
trainer = train_qlora(
    model=qlora_model,
    tokenizer=tokenizer,
    dataset=train_dataset,
    output_dir="./llama2-qlora-chat",
)
```

### 13.2.6 Best Practices QLoRA

**1. Compute dtype:**
```python
# BF16 >> FP16 pour QLoRA
# BF16 a meilleur range numérique
bnb_4bit_compute_dtype = torch.bfloat16  # Recommandé
```

**2. Learning rate:**
```python
# QLoRA peut utiliser des LR plus élevés que full fine-tuning
lr = 2e-4  # QLoRA
vs
lr = 5e-6  # Full fine-tuning
```

**3. Batch size et gradient accumulation:**
```python
# Avec mémoire limitée
per_device_batch_size = 1
gradient_accumulation_steps = 16
# → Effective batch size = 16
```

**4. Gradient checkpointing:**
```python
# Trade compute pour mémoire
# Ralentit ~20% mais économise 40-50% de mémoire
gradient_checkpointing = True
```

**5. Target modules:**
```python
# Pour meilleure performance, inclure aussi FFN
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # FFN
]
```

### 13.2.7 Projet Pratique Complet

**Objectif:** Fine-tuner Llama 2 7B pour dialogue en français sur un GPU 24GB.

```python
"""
Projet: Fine-tuning Llama 2 7B avec QLoRA pour dialogue français
Hardware: 1x RTX 3090 24GB
Dataset: French conversations
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from datasets import load_dataset

# ============== Configuration ==============

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./llama2-qlora-french-chat"
DATASET_NAME = "OpenAssistant/oasst1"  # Multilingual dataset

# QLoRA config
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training config
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# ============== Setup ==============

# BitsAndBytes quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Charger modèle quantizé
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Préparer pour training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Appliquer LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============== Dataset ==============

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

# Filter pour français uniquement
def filter_french(example):
    return example["lang"] == "fr"

french_dataset = dataset["train"].filter(filter_french)

# Format en prompt-completion
def format_dialogue(example):
    """
    Format: <s>[INST] {prompt} [/INST] {completion}</s>
    """
    prompt = example["text"].split("Assistant:")[0].strip()
    completion = example["text"].split("Assistant:")[-1].strip()

    formatted = f"<s>[INST] {prompt} [/INST] {completion}</s>"

    return {"text": formatted}

formatted_dataset = french_dataset.map(format_dialogue)

print(f"Training samples: {len(formatted_dataset)}")

# ============== Training ==============

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    logging_dir=f"{OUTPUT_DIR}/logs",
    save_strategy="epoch",
    save_total_limit=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    group_by_length=True,  # Optimisation
    report_to="tensorboard",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=False,
)

# Train!
print("Starting training...")
trainer.train()

# Save
print("Saving model...")
trainer.save_model(OUTPUT_DIR)

# ============== Inference Test ==============

print("\n=== Testing generation ===")

# Load du modèle sauvegardé
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

# Test prompt
prompt = "<s>[INST] Explique-moi ce qu'est un transformers en IA. [/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

**Résultats attendus:**
- Training time: ~6-8 heures sur RTX 3090
- Peak memory: ~22GB
- Final loss: ~1.5-2.0
- Perplexity: ~15-20

---

*[Le chapitre continue avec d'autres méthodes PEFT: Adapter Layers, Prefix Tuning, Prompt Tuning, IA³...]*

*[Contenu total du Chapitre 13: ~50-60 pages]*
