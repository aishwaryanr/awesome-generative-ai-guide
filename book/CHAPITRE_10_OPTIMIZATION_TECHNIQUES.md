# CHAPITRE 10 : TECHNIQUES D'OPTIMISATION DES LLMs

> *« Un LLM non optimisé, c'est comme une Formule 1 avec des pneus de tracteur. La puissance est là, mais l'efficacité... »*

---

## Introduction : La Course à l'Efficacité

### 🎭 Dialogue : Le Mur de la Mémoire

**Alice** : Bob, j'ai essayé de charger LLaMA-65B sur mon GPU 24GB... et ça crash immédiatement avec "CUDA out of memory".

**Bob** : Normal. LLaMA-65B en float32 nécessite 65B × 4 bytes = **260GB** de mémoire !

**Alice** : Donc impossible sans un cluster de GPUs ?

**Bob** : Pas forcément. Avec les bonnes optimisations :
- **Quantization** : 8-bit → 65GB (4× moins)
- **4-bit** : 32GB (8× moins)
- **Flash Attention** : 3× plus rapide, 10× moins de mémoire
- **Gradient checkpointing** : Réduire mémoire training 50%

**Alice** : Et la performance ?

**Bob** : Quantization int8 : perte < 1% accuracy. 4-bit : ~2-3%. Flash Attention : **zéro perte**, juste plus rapide !

### 📊 Le Problème de l'Échelle

| Modèle | Paramètres | RAM FP32 | RAM FP16 | RAM INT8 | RAM INT4 |
|--------|-----------|----------|----------|----------|----------|
| **GPT-2 Small** | 124M | 0.5GB | 0.25GB | 0.125GB | 0.06GB |
| **BERT Large** | 340M | 1.4GB | 0.7GB | 0.35GB | 0.17GB |
| **GPT-3** | 175B | 700GB | 350GB | 175GB | 87GB |
| **LLaMA-65B** | 65B | 260GB | 130GB | 65GB | 32GB |
| **GPT-4** (estimé) | 1.8T | 7.2TB | 3.6TB | 1.8TB | 900GB |

**Constat** : Sans optimisation, seuls les labs avec superclusters peuvent utiliser les grands modèles.

### 🎯 Anecdote : Flash Attention Change la Donne

**Juin 2022, Stanford University**

Tri Dao, un PhD étudiant, publie "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness".

**Problème identifié** : L'attention classique est O(n²) en mémoire ET fait des accès mémoire inefficaces (HBM ↔ SRAM).

**Innovation** : Découper l'attention en tuiles (tiling) + fusionner opérations → réduire accès mémoire.

**Résultats** :
- **3× plus rapide** que PyTorch standard
- **10× moins de mémoire** (contextes plus longs)
- **Exactement identique** mathématiquement (pas d'approximation)

**Impact** :
- Intégré dans PyTorch 2.0 (2023)
- Utilisé par GPT-4, Claude, Gemini
- Permet d'entraîner avec contextes 10× plus longs

Aujourd'hui, ne PAS utiliser Flash Attention est considéré comme une erreur d'ingénierie.

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Implémenter Flash Attention et comprendre pourquoi c'est crucial
- ✅ Quantizer des modèles en 8-bit et 4-bit avec QLoRA
- ✅ Utiliser gradient checkpointing pour économiser mémoire
- ✅ Optimiser l'inférence avec TensorRT, ONNX Runtime
- ✅ Appliquer mixed precision training (FP16/BF16)
- ✅ Paralléliser sur plusieurs GPUs (DDP, FSDP)

**Difficulté** : 🔴🔴🔴🔴⚪ (Expert)
**Prérequis** : CUDA basics, PyTorch avancé, Transformers
**Temps de lecture** : ~130 minutes

---

## Flash Attention : Révolution de l'Efficacité

### Le Problème de l'Attention Standard

**Complexité** :
- Compute : O(n²d) où n = seq_len, d = hidden_dim
- Memory : O(n²) pour stocker la matrice d'attention

**Code classique** :
```python
def standard_attention(Q, K, V):
    """
    Attention standard (PyTorch).
    Problème : Matérialise la matrice attention [n, n] en mémoire.
    """
    d_k = Q.size(-1)

    # [batch, heads, n, n] - ÉNORME pour grands n!
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [batch, heads, n, d_v]
    output = torch.matmul(attn_weights, V)
    return output
```

**Mémoire pour seq_len=2048** :
```
attn_weights : [batch, 8 heads, 2048, 2048] × 2 bytes (FP16)
            = batch × 8 × 2048² × 2 / 1e9
            = batch × 0.067 GB

Pour batch=32 : 2.1 GB juste pour les weights!
```

### Flash Attention : L'Algorithme

**Idées clés** :

1. **Tiling** : Découper Q, K, V en blocs
2. **Fusion** : Calculer softmax et matmul en une seule passe
3. **IO-awareness** : Minimiser transferts HBM ↔ SRAM

**Pseudo-code** :
```python
def flash_attention(Q, K, V, block_size=128):
    """
    Flash Attention (simplifié).

    Ne matérialise JAMAIS la matrice [n, n] complète!
    Traite par blocs de taille [block_size, block_size].
    """
    n, d = Q.shape
    output = torch.zeros_like(Q)

    # Itérer par blocs
    for i in range(0, n, block_size):
        Q_block = Q[i:i+block_size]  # [block_size, d]

        for j in range(0, n, block_size):
            K_block = K[j:j+block_size]  # [block_size, d]
            V_block = V[j:j+block_size]

            # Attention sur ce bloc uniquement
            attn_block = Q_block @ K_block.T / math.sqrt(d)
            attn_block = F.softmax(attn_block, dim=-1)

            # Accumuler
            output[i:i+block_size] += attn_block @ V_block

    return output
```

**Mémoire** : O(n × block_size) au lieu de O(n²)

### Utilisation avec PyTorch 2.0+

```python
import torch
import torch.nn.functional as F

# Activer Flash Attention (automatique si disponible)
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False
    )
```

### Benchmark Flash Attention

```python
import torch
import time
from torch.nn.functional import scaled_dot_product_attention

def benchmark_attention(seq_len, d_model, num_heads, use_flash=True):
    """
    Compare attention standard vs Flash Attention.
    """
    batch_size = 8
    d_head = d_model // num_heads

    # Données aléatoires
    Q = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, d_head, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                _ = scaled_dot_product_attention(Q, K, V)
        else:
            # Standard attention
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            _ = torch.matmul(attn, V)

    torch.cuda.synchronize()

    # Mesure
    start = time.time()
    for _ in range(100):
        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                output = scaled_dot_product_attention(Q, K, V)
        else:
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            output = torch.matmul(attn, V)

    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 100

    # Mémoire
    mem_allocated = torch.cuda.max_memory_allocated() / 1e9

    return elapsed * 1000, mem_allocated  # ms, GB

# Benchmark
for seq_len in [512, 1024, 2048, 4096]:
    time_standard, mem_standard = benchmark_attention(seq_len, 768, 12, use_flash=False)
    time_flash, mem_flash = benchmark_attention(seq_len, 768, 12, use_flash=True)

    speedup = time_standard / time_flash
    mem_reduction = mem_standard / mem_flash

    print(f"Seq Len {seq_len}:")
    print(f"  Standard: {time_standard:.2f}ms, {mem_standard:.2f}GB")
    print(f"  Flash:    {time_flash:.2f}ms, {mem_flash:.2f}GB")
    print(f"  Speedup: {speedup:.2f}×, Memory: {mem_reduction:.2f}× less\n")
```

**Résultats typiques** :
```
Seq Len 512:
  Standard: 5.2ms, 0.8GB
  Flash:    1.8ms, 0.3GB
  Speedup: 2.9×, Memory: 2.7× less

Seq Len 2048:
  Standard: 82.4ms, 12.1GB
  Flash:    24.1ms, 1.2GB
  Speedup: 3.4×, Memory: 10.1× less

Seq Len 4096:
  Standard: OOM (Out of Memory)
  Flash:    95.3ms, 4.8GB
  ✅ Fonctionne!
```

---

## Quantization : Réduire la Précision

### Types de Précision

| Type | Bits | Range | Précision | Usage |
|------|------|-------|-----------|-------|
| **FP32** | 32 | ±3.4×10³⁸ | ~7 décimales | Training (ancien) |
| **FP16** | 16 | ±65,504 | ~3 décimales | Training moderne |
| **BF16** | 16 | ±3.4×10³⁸ | ~2 décimales | Training (meilleur) |
| **INT8** | 8 | -128 à 127 | Entier | Inference |
| **INT4** | 4 | -8 à 7 | Entier | Inference (QLoRA) |

### Quantization Dynamique (Post-Training)

**Principe** : Convertir FP32 → INT8 après entraînement.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Charger modèle en FP32
model = AutoModelForCausalLM.from_pretrained("gpt2")
print(f"Taille FP32: {model.get_memory_footprint() / 1e6:.2f} MB")

# 2. Quantization dynamique (INT8)
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantizer seulement les Linear layers
    dtype=torch.qint8
)

print(f"Taille INT8: {model_int8.get_memory_footprint() / 1e6:.2f} MB")

# Réduction: ~4× (32 bits → 8 bits)
```

**Résultats** :
```
Taille FP32: 548.31 MB
Taille INT8: 142.12 MB
Réduction: 3.86×
```

### BitsAndBytes : 8-bit et 4-bit

**Library** de Tim Dettmers (2022) pour quantization efficace.

#### 8-bit (LLM.int8())

```python
from transformers import AutoModelForCausalLM
import torch

# Charger directement en 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-6.7b",
    load_in_8bit=True,
    device_map="auto"  # Répartir automatiquement sur GPUs disponibles
)

print(f"Mémoire: {model.get_memory_footprint() / 1e9:.2f} GB")
# OPT-6.7B : ~6.7 GB au lieu de ~27 GB en FP32
```

**Technique** : Mixed-precision matrix decomposition
```
W × X = W_outliers × X + W_quantized × X

où W_outliers : colonnes avec valeurs extrêmes (gardées en FP16)
    W_quantized : reste en INT8
```

**Performance** : Perte < 0.5% sur la plupart des benchmarks.

#### 4-bit (QLoRA)

**Innovation** : Quantization + LoRA pour fine-tuning efficient.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configuration 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Double quantization
)

# Charger en 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Mémoire: {model.get_memory_footprint() / 1e9:.2f} GB")
# LLaMA-7B : ~3.5 GB au lieu de ~28 GB en FP32
```

**NormalFloat (NF4)** : Quantization optimale pour poids normalement distribués.

**Double Quantization** : Quantizer aussi les constantes de quantization (économise ~0.5 GB).

### Fine-Tuning avec QLoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Préparer modèle 4-bit pour training
model = prepare_model_for_kbit_training(model)

# 2. Configuration LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Appliquer LoRA
model = get_peft_model(model, lora_config)

# Vérifier paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
# LLaMA-7B avec LoRA : ~4M trainable sur 7B total (0.06%)

# 4. Entraîner normalement
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./qlora-llama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

**Résultats QLoRA** :
- LLaMA-65B fine-tunable sur 1× A100 48GB
- Performance équivalente à full fine-tuning
- Coût : $100 au lieu de $10,000

---

## Mixed Precision Training

### FP16 vs BF16

**FP16 (Float16)** :
```
Sign: 1 bit
Exponent: 5 bits → range ±65,504
Mantissa: 10 bits → précision ~0.001
```

**Problème FP16** : Exponent limité → gradient underflow/overflow fréquent.

**BF16 (BFloat16)** :
```
Sign: 1 bit
Exponent: 8 bits → même range que FP32 (±3.4×10³⁸)
Mantissa: 7 bits → précision réduite mais OK
```

**Avantage BF16** : Pas de overflow, plus stable.

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        # Forward en FP16
        with autocast():
            outputs = model(batch)
            loss = criterion(outputs, batch['labels'])

        # Backward avec gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**HuggingFace Trainer** :
```python
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # ou bf16=True pour BF16
    # ...
)
```

**Gains** :
- **2× plus rapide** (TensorCores utilisés)
- **50% moins de mémoire**
- Performance identique à FP32 (avec scaler)

---

## Gradient Checkpointing

### Le Problème de Mémoire en Training

**Forward pass** : Stocker activations pour backward
```
Layer 1: 100 MB
Layer 2: 100 MB
...
Layer 96: 100 MB

Total: 9.6 GB d'activations stockées!
```

### Gradient Checkpointing : Trade-off Temps vs Mémoire

**Principe** : Ne stocker que certaines activations, recalculer les autres en backward.

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...

    def forward(self, x):
        # Avec checkpointing
        x = checkpoint(self.attention, x)
        x = checkpoint(self.ffn, x)
        return x
```

**HuggingFace** :
```python
model.gradient_checkpointing_enable()

# Ou dans TrainingArguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    # ...
)
```

**Résultats** :
- Mémoire : -50% à -70%
- Temps : +20% à +30% (recalcul en backward)
- Trade-off acceptable pour la plupart des cas

---

## Optimisation de l'Inférence

### TorchScript : Compilation

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

# Exemple input
dummy_input = torch.randint(0, 30522, (1, 128))

# Compiler avec TorchScript
traced_model = torch.jit.trace(model, dummy_input)

# Sauvegarder
traced_model.save("model_traced.pt")

# Charger et utiliser
loaded = torch.jit.load("model_traced.pt")
output = loaded(dummy_input)
```

**Gains** : 10-20% plus rapide, meilleure optimisation.

### ONNX Runtime

```python
from transformers import AutoTokenizer, AutoModel
from optimum.onnxruntime import ORTModelForSequenceClassification

# Exporter en ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Inférence
inputs = tokenizer("This movie is great!", return_tensors="pt")
outputs = model(**inputs)
```

**Gains** : 2-3× plus rapide que PyTorch natif.

### TensorRT : Optimisation NVIDIA

```python
# Nécessite NVIDIA TensorRT installé
from transformers import AutoModel
import torch_tensorrt

model = AutoModel.from_pretrained("bert-base-uncased").eval().cuda()

# Compiler avec TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch.randint(0, 30522, (1, 128), device='cuda')],
    enabled_precisions={torch.float16},  # FP16
    workspace_size=1 << 30  # 1 GB
)

# Inférence ultra-rapide
output = trt_model(input_ids)
```

**Gains** : 5-10× plus rapide que PyTorch, surtout pour petits batch sizes.

### KV-Cache pour Génération

**Problème** : Génération autoregressive recalcule keys et values à chaque token.

```python
# Sans cache (inefficace)
for i in range(max_new_tokens):
    # Recalcule K, V pour TOUTE la séquence
    outputs = model(input_ids)  # input_ids grandit à chaque itération
    next_token = outputs.logits[:, -1, :].argmax(-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
```

**Avec KV-Cache** :
```python
# Avec cache (efficace)
past_key_values = None
for i in range(max_new_tokens):
    outputs = model(
        input_ids if past_key_values is None else input_ids[:, -1:],  # Seulement dernier token
        past_key_values=past_key_values,
        use_cache=True
    )
    past_key_values = outputs.past_key_values  # Réutiliser
    next_token = outputs.logits[:, -1, :].argmax(-1)
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
```

**Gains** : 10-100× plus rapide selon longueur séquence.

---

## Parallélisation Multi-GPU

### Data Parallel (DP) - Simple

```python
import torch
import torch.nn as nn

model = MyModel()

# Data Parallel (ancien, simple mais inefficace)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model = model.cuda()

# Training normal
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

**Problème DP** : GPU 0 est bottleneck (collecte tous les gradients).

### Distributed Data Parallel (DDP) - Moderne

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialiser process group."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train(rank, world_size):
    setup(rank, world_size)

    # Modèle sur ce GPU
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # Sampler distribué
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # Training
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important pour shuffling
        for batch in dataloader:
            batch = batch.to(rank)
            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

# Lancer
import torch.multiprocessing as mp
mp.spawn(train, args=(world_size,), nprocs=world_size)
```

**Avec HuggingFace** :
```python
# Lancer avec torchrun
# torchrun --nproc_per_node=4 train.py

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,  # Par GPU
    # DDP automatique avec torchrun
)
```

### Fully Sharded Data Parallel (FSDP)

**Problème DDP** : Chaque GPU a une copie complète du modèle.

**FSDP** : Sharde le modèle entre GPUs.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel()
model = FSDP(model)

# Training normal, modèle shardu automatiquement
```

**HuggingFace** :
```python
training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard auto_wrap",  # Enable FSDP
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "BertLayer"
    }
)
```

**Gains FSDP** :
- Entraîner modèles 8× plus grands qu'avec DDP
- Efficace pour LLaMA-65B+ sur 8× A100

---

## Optimiseurs Efficaces

### AdamW vs AdaFactor

**AdamW** : Standard, mais coûteux
```
Mémoire : 2× paramètres (momentum + variance)
LLaMA-7B : 7B params → 14B floats stockés = 56 GB
```

**AdaFactor** : Version memory-efficient
```python
from transformers import Adafactor

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    scale_parameter=True,
    relative_step=False,
    warmup_init=False
)
```

**Mémoire** : ~1× paramètres (factorisation low-rank)

### 8-bit Adam

```python
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4
)
```

**Mémoire** : 4× moins qu'AdamW classique.

---

## Optimisation CPU : Quantization pour Inference

### ONNX Runtime CPU

```python
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Exporter en ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    export=True
)

# Quantizer pour CPU
quantizer = ORTQuantizer.from_pretrained(model)

# Configuration quantization
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

# Quantizer
quantizer.quantize(
    save_dir="./quantized_model",
    quantization_config=dqconfig
)

# Charger modèle quantizé
quantized_model = ORTModelForSequenceClassification.from_pretrained("./quantized_model")

# Inférence CPU rapide
output = quantized_model(**inputs)
```

**Gains CPU** : 3-4× plus rapide qu'FP32.

---

## 💡 Analogie : L'Optimisation comme une Formule 1

- **Flash Attention** : Aérodynamisme (réduire résistance/accès mémoire)
- **Quantization** : Matériaux légers (carbone au lieu d'acier)
- **Mixed Precision** : Turbo (boost ponctuel quand nécessaire)
- **Gradient Checkpointing** : Économie de carburant (trade-off vitesse/autonomie)
- **FSDP** : Voiture en kit (chaque mécanicien a une pièce)
- **KV-Cache** : Ne pas refaire un tour complet pour doubler

Résultat : Même puissance, 10× plus efficient !

---

## Quiz Interactif

### Question 1 : Flash Attention

**Flash Attention est plus rapide car :**

A) Elle approxime l'attention (moins précise)
B) Elle utilise moins d'opérations mathématiques
C) Elle optimise les accès mémoire (tiling)
D) Elle skip certains tokens

<details>
<summary>Voir la réponse</summary>

**Réponse : C) Elle optimise les accès mémoire (tiling)**

Flash Attention est **mathématiquement identique** à l'attention standard. L'accélération vient de :
- Tiling : Traiter par blocs → moins d'accès HBM
- Fusion : softmax + matmul en une passe
- IO-awareness : Maximiser réutilisation SRAM

Résultat : 3× plus rapide, 10× moins mémoire, **zéro perte de précision**.
</details>

---

### Question 2 : Quantization

**INT8 quantization réduit la mémoire de combien ?**

A) 2×
B) 4×
C) 8×
D) Ça dépend

<details>
<summary>Voir la réponse</summary>

**Réponse : B) 4×**

- FP32 : 32 bits = 4 bytes
- INT8 : 8 bits = 1 byte

Réduction : 4× moins de mémoire.

**Bonus** :
- FP16 → INT8 : 2×
- FP32 → INT4 : 8×
</details>

---

### Question 3 : Gradient Checkpointing

**Gradient checkpointing réduit la mémoire en :**

A) Quantizant les gradients
B) Ne stockant que certaines activations
C) Utilisant moins de layers
D) Skippant le backward pass

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Ne stockant que certaines activations**

Forward : Stocker seulement checkpoints (ex: 1 activation sur 4)
Backward : Recalculer activations manquantes à la volée

Trade-off :
- Mémoire : -50% à -70%
- Temps : +20% à +30% (recalcul)
</details>

---

### Question 4 : BF16 vs FP16

**Pourquoi BF16 est souvent préféré à FP16 pour training ?**

A) Plus rapide
B) Range d'exponent plus large (pas de overflow)
C) Plus précis
D) Moins de mémoire

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Range d'exponent plus large**

**FP16** : 5 bits exponent → ±65,504 (overflow fréquent)
**BF16** : 8 bits exponent → ±3.4×10³⁸ (même range que FP32)

BF16 = "FP32 tronqué" → stable, pas besoin de gradient scaler.
</details>

---

### Question 5 : KV-Cache

**KV-Cache accélère la génération car :**

A) Il approxime les keys et values
B) Il réutilise K et V des tokens précédents
C) Il réduit la taille du vocabulaire
D) Il skip l'attention

<details>
<summary>Voir la réponse</summary>

**Réponse : B) Il réutilise K et V des tokens précédents**

Génération token i+1 : Besoin de K et V pour tokens 1..i
Sans cache : Recalculer K, V pour 1..i (redondant!)
Avec cache : Réutiliser, calculer seulement K, V pour token i+1

Gain : 10-100× plus rapide selon longueur.
</details>

---

## Exercices Pratiques

### Exercice 1 : Benchmark Quantization

**Objectif** : Mesurer impact de INT8 sur vitesse et précision.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

def benchmark_quantization(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Compare FP32 vs INT8 : vitesse, mémoire, accuracy.
    """
    # TODO:
    # 1. Charger modèle FP32
    # 2. Quantizer en INT8
    # 3. Tester sur dataset (ex: 1000 exemples SST-2)
    # 4. Mesurer : temps inférence, mémoire, accuracy
    # 5. Calculer speedup et différence accuracy
    pass

# benchmark_quantization()
```

<details>
<summary>Voir la solution</summary>

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score

def benchmark_quantization(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Compare FP32 vs INT8.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("sst2", split="validation[:1000]")

    # 1. Modèle FP32
    model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_fp32.eval()

    # 2. Modèle INT8
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Mémoire
    mem_fp32 = model_fp32.get_memory_footprint() / 1e6
    mem_int8 = model_int8.get_memory_footprint() / 1e6

    print(f"Mémoire FP32: {mem_fp32:.2f} MB")
    print(f"Mémoire INT8: {mem_int8:.2f} MB")
    print(f"Réduction: {mem_fp32/mem_int8:.2f}×\n")

    # Fonction d'évaluation
    def evaluate(model, name):
        predictions = []
        labels = []

        start = time.time()
        with torch.no_grad():
            for example in dataset:
                inputs = tokenizer(example["sentence"], return_tensors="pt", truncation=True, max_length=128)
                outputs = model(**inputs)
                pred = outputs.logits.argmax(-1).item()
                predictions.append(pred)
                labels.append(example["label"])

        elapsed = time.time() - start
        acc = accuracy_score(labels, predictions)

        print(f"{name}:")
        print(f"  Temps: {elapsed:.2f}s ({elapsed/len(dataset)*1000:.2f}ms/exemple)")
        print(f"  Accuracy: {acc:.4f}\n")

        return elapsed, acc

    # 3. Évaluer
    time_fp32, acc_fp32 = evaluate(model_fp32, "FP32")
    time_int8, acc_int8 = evaluate(model_int8, "INT8")

    # 4. Comparaison
    speedup = time_fp32 / time_int8
    acc_diff = abs(acc_fp32 - acc_int8)

    print(f"Speedup: {speedup:.2f}×")
    print(f"Accuracy difference: {acc_diff:.4f} ({acc_diff*100:.2f}%)")

benchmark_quantization()
```
</details>

---

### Exercice 2 : Implémenter Gradient Checkpointing Custom

**Objectif** : Comprendre le mécanisme en l'implémentant.

```python
import torch
import torch.nn as nn

class CheckpointedSequential(nn.Sequential):
    """
    Sequential avec gradient checkpointing manuel.
    """
    def __init__(self, *args, checkpoint_every=2):
        super().__init__(*args)
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        # TODO:
        # 1. Itérer sur les layers
        # 2. Appliquer checkpointing tous les N layers
        # 3. Forward normal pour les autres
        pass

# Test
# model = CheckpointedSequential(
#     nn.Linear(512, 512),
#     nn.ReLU(),
#     nn.Linear(512, 512),
#     nn.ReLU(),
#     # ... 20 layers
#     checkpoint_every=4
# )
```

<details>
<summary>Voir la solution</summary>

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedSequential(nn.Sequential):
    """
    Sequential avec gradient checkpointing.
    """
    def __init__(self, *args, checkpoint_every=2):
        super().__init__(*args)
        self.checkpoint_every = checkpoint_every

    def forward(self, x):
        """
        Forward avec checkpointing tous les N layers.
        """
        modules = list(self._modules.values())

        for i, module in enumerate(modules):
            if i % self.checkpoint_every == 0 and self.training:
                # Checkpointing : ne pas stocker activations
                x = checkpoint(module, x)
            else:
                # Forward normal
                x = module(x)

        return x

# Test avec mesure mémoire
def test_checkpointing():
    """
    Compare mémoire avec/sans checkpointing.
    """
    # Créer deux modèles identiques
    layers = [nn.Linear(1024, 1024), nn.ReLU()] * 20  # 40 layers

    model_normal = nn.Sequential(*layers).cuda()
    model_checkpointed = CheckpointedSequential(*layers, checkpoint_every=4).cuda()

    # Dummy input
    x = torch.randn(32, 1024, device='cuda', requires_grad=True)

    # Forward + backward normal
    torch.cuda.reset_peak_memory_stats()
    out_normal = model_normal(x)
    loss_normal = out_normal.sum()
    loss_normal.backward()
    mem_normal = torch.cuda.max_memory_allocated() / 1e9

    # Forward + backward checkpointed
    torch.cuda.reset_peak_memory_stats()
    out_checkpointed = model_checkpointed(x)
    loss_checkpointed = out_checkpointed.sum()
    loss_checkpointed.backward()
    mem_checkpointed = torch.cuda.max_memory_allocated() / 1e9

    print(f"Mémoire normal: {mem_normal:.2f} GB")
    print(f"Mémoire checkpointed: {mem_checkpointed:.2f} GB")
    print(f"Réduction: {mem_normal/mem_checkpointed:.2f}×")

test_checkpointing()
```
</details>

---

### Exercice 3 : Profile GPU avec PyTorch Profiler

**Objectif** : Identifier bottlenecks avec le profiler.

```python
import torch
from torch.profiler import profile, ProfilerActivity

def train_step(model, batch):
    """Un step de training."""
    outputs = model(batch)
    loss = outputs.sum()
    loss.backward()
    return loss

# TODO:
# 1. Wrapper train_step avec profiler
# 2. Identifier opérations les plus coûteuses
# 3. Exporter trace Chrome (chrome://tracing)
```

<details>
<summary>Voir la solution</summary>

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModel

def profile_training():
    """
    Profile un step de training avec PyTorch Profiler.
    """
    model = AutoModel.from_pretrained("bert-base-uncased").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Dummy batch
    batch = {
        'input_ids': torch.randint(0, 30522, (8, 128), device='cuda'),
        'attention_mask': torch.ones(8, 128, device='cuda')
    }

    # Warmup
    for _ in range(10):
        outputs = model(**batch)
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(**batch)
            loss = outputs.last_hidden_state.sum()

        with record_function("backward"):
            loss.backward()

        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()

    # Print résumé
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=10
    ))

    # Export trace Chrome
    prof.export_chrome_trace("trace.json")
    print("\nTrace exportée: trace.json")
    print("Ouvrir dans Chrome: chrome://tracing")

profile_training()
```
</details>

---

## Conclusion

### 🎭 Dialogue Final : L'Optimisation, Clé de la Démocratisation

**Alice** : Maintenant je comprends : sans optimisation, les LLMs resteraient dans les labs !

**Bob** : Exactement. Regarde l'évolution :
- **2020** : GPT-3 nécessite cluster A100 (coût : millions)
- **2023** : LLaMA-65B avec QLoRA sur 1× A100 (coût : milliers)
- **2024** : LLaMA-7B en 4-bit sur laptop (coût : gratuit)

**Alice** : Quelles optimisations sont essentielles ?

**Bob** : **Top 3** :
1. **Flash Attention** : Gratuit (zéro perte), 3× plus rapide
2. **Quantization INT8** : Perte < 1%, 4× moins mémoire
3. **Gradient Checkpointing** : Training possible avec 50% moins GPU

**Alice** : Et le futur ?

**Bob** :
- **Quantization 2-bit** : Recherche active, pourrait atteindre 16× réduction
- **Sparse models** : Activer seulement 10% des paramètres par token
- **Mixture of Experts** : GPT-4 style, efficient scaling
- **Hardware spécialisé** : TPUs v5, Groq LPUs (1000× plus rapide)

L'optimisation transforme l'IA d'une technologie élitiste en outil universel.

### 🎯 Points Clés à Retenir

| Technique | Gain Mémoire | Gain Vitesse | Perte Qualité |
|-----------|-------------|--------------|---------------|
| **Flash Attention** | 10× | 3× | 0% |
| **INT8 Quantization** | 4× | 2× | < 1% |
| **INT4 (QLoRA)** | 8× | 1.5× | 2-3% |
| **Gradient Checkpointing** | 2× | -20% | 0% |
| **Mixed Precision (FP16)** | 2× | 2× | 0% |
| **KV-Cache** | - | 10-100× | 0% |
| **FSDP** | 8× (modèle plus grand) | 1× | 0% |

### 📋 Checklist Optimisation

**Pour Training** :
- [ ] Mixed precision (BF16 recommandé)
- [ ] Gradient checkpointing si mémoire limitée
- [ ] Flash Attention (PyTorch 2.0+)
- [ ] QLoRA si fine-tuning grands modèles
- [ ] FSDP si multi-GPU et très grand modèle

**Pour Inference** :
- [ ] Quantization INT8 (production)
- [ ] KV-Cache activé (génération)
- [ ] Batch requests si possible
- [ ] ONNX/TensorRT pour max vitesse
- [ ] CPU : ONNX Runtime quantizé

**Red Flags** :
- ⚠️ Pas de Flash Attention en 2024+
- ⚠️ Full fine-tuning sans considérer QLoRA
- ⚠️ Génération sans KV-Cache
- ⚠️ Training FP32 (obsolète)

---

## Ressources

### 📚 Papers Fondamentaux

1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022)
2. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"** (Dettmers et al., 2022)
3. **"QLoRA: Efficient Finetuning of Quantized LLMs"** (Dettmers et al., 2023)
4. **"Training Compute-Optimal Large Language Models"** (Hoffmann et al., 2022)

### 🛠️ Bibliothèques

```bash
# Flash Attention
pip install flash-attn

# BitsAndBytes (quantization)
pip install bitsandbytes

# Optimum (ONNX, quantization)
pip install optimum[onnxruntime]

# DeepSpeed (optimisations training)
pip install deepspeed
```

### 🔗 Ressources

- **Flash Attention repo** : https://github.com/Dao-AILab/flash-attention
- **BitsAndBytes** : https://github.com/TimDettmers/bitsandbytes
- **Optimum** : https://huggingface.co/docs/optimum
- **PyTorch Profiler Guide** : https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

---

**🎓 Bravo !** Vous maîtrisez maintenant les techniques d'optimisation des LLMs. Prochain chapitre : **Chapitre 11 - Prompt Engineering** pour maximiser les performances sans ré-entraîner ! 🚀

