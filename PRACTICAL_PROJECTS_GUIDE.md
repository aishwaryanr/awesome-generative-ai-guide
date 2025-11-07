# 🔨 GUIDE DES PROJETS PRATIQUES
## 15 Projets Progressifs : Du Débutant à l'Expert

---

> **Philosophie**: Apprendre en faisant. Chaque projet construit sur les précédents et vous amène progressivement vers la maîtrise complète des LLMs.

---

## 📊 APERÇU DES PROJETS

| # | Projet | Niveau | Durée | Technologies Clés | Compétences |
|---|--------|--------|-------|-------------------|-------------|
| 1 | Transformer from Scratch | 🟢 Débutant | 8-12h | PyTorch, NumPy | Architecture, Math |
| 2 | Data Preparation Pipeline | 🟢 Débutant | 10-15h | Python, Datasets | Data Engineering |
| 3 | Train nanoGPT (124M) | 🔵 Intermédiaire | 15-20h | PyTorch, GPUs | Training Basics |
| 4 | Optimize Training Run | 🔵 Intermédiaire | 8-12h | Profiling, DeepSpeed | Performance |
| 5 | Fine-tune Llama 3 | 🔵 Intermédiaire | 10-15h | HuggingFace, Transformers | Fine-tuning |
| 6 | LoRA on Consumer GPU | 🔵 Intermédiaire | 12-18h | PEFT, bitsandbytes | Efficient Training |
| 7 | RLHF Pipeline | 🟠 Avancé | 20-30h | TRL, PPO | Alignment |
| 8 | Quantize for CPU | 🔵 Intermédiaire | 8-10h | llama.cpp, GPTQ | Optimization |
| 9 | Deploy vLLM API | 🟠 Avancé | 12-18h | vLLM, FastAPI | Serving |
| 10 | RAG System (10k docs) | 🟠 Avancé | 20-25h | LangChain, Qdrant | RAG Architecture |
| 11 | Autonomous Agent | 🟠 Avancé | 25-35h | LangChain, Tools | Agents |
| 12 | Fine-tune Multimodal | 🟠 Avancé | 20-30h | LLaVA, Vision | Multimodal |
| 13 | Eval Pipeline (CI/CD) | 🔴 Expert | 15-20h | Testing, Automation | LLMOps |
| 14 | Enterprise Chatbot | 🔴 Expert | 40-60h | Full Stack | Production App |
| 15 | LLM from Scratch | 🔴 Expert | 100-150h | Tout | End-to-End |

**Total estimé**: ~350-450 heures de pratique

---

## 🟢 PROJET 1 : TRANSFORMER FROM SCRATCH

### **Objectifs d'apprentissage**
- Comprendre en profondeur l'architecture transformer
- Implémenter self-attention, multi-head attention
- Maîtriser les positional encodings
- Créer un modèle entraînable from scratch

### **Spécifications**
```python
# Architecture cible
- Modèle: Decoder-only transformer (GPT-style)
- Paramètres: ~6M (petit pour apprentissage)
- Couches: 6 transformer blocks
- Attention heads: 6
- Embedding dim: 384
- Context length: 256 tokens
- Vocabulaire: 50k tokens (GPT-2 tokenizer)
```

### **Structure du projet**
```
project_01_transformer/
├── model.py              # Architecture du transformer
├── attention.py          # Self-attention mechanism
├── positional.py         # Positional encoding
├── feed_forward.py       # FFN layers
├── train.py              # Training loop
├── tokenizer.py          # Tokenization
├── data.py               # Dataset loading
├── config.py             # Hyperparameters
├── utils.py              # Helper functions
└── notebooks/
    ├── 01_attention_visualization.ipynb
    ├── 02_training_demo.ipynb
    └── 03_generation_demo.ipynb
```

### **Étapes détaillées**

#### **Étape 1: Implémentation de l'attention (3-4h)**
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Implémentation from scratch du mécanisme d'attention
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Projeter et reshaper pour multi-head
        qkv = self.qkv_proj(x)  # [B, T, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        # Masque causal pour autoregressive generation
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax et application sur V
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D_h]

        # Recombiner les heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # Projection finale
        output = self.out_proj(attn_output)

        return output, attn_weights
```

**🛠️ Exercice**:
- Visualiser les attention weights sur une phrase simple
- Tester avec différents nombres de heads
- Comparer avec `torch.nn.MultiheadAttention`

#### **Étape 2: Positional Encoding (2h)**
```python
class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal (Vaswani et al. 2017)
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Créer la matrice d'encodage
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]
```

**🛠️ Alternatives à implémenter**:
- Learned positional embeddings
- RoPE (Rotary Position Embedding)
- ALiBi (Attention with Linear Biases)

#### **Étape 3: Transformer Block (2-3h)**
```python
class TransformerBlock(nn.Module):
    """
    Bloc transformer complet: Attention + FFN + LayerNorm + Residual
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = SelfAttention(embed_dim, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization (pre-norm architecture)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN architecture (GPT-2 style)
        # Attention block
        attn_out, attn_weights = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # FFN block
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, attn_weights
```

#### **Étape 4: Modèle complet (2-3h)**
```python
class GPTModel(nn.Module):
    """
    Modèle GPT complet (decoder-only transformer)
    """
    def __init__(self, vocab_size, embed_dim=384, num_heads=6,
                 num_layers=6, ff_dim=1536, max_len=256, dropout=0.1):
        super().__init__()

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encode = PositionalEncoding(embed_dim, max_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output head
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights (token embeddings = output embeddings)
        self.head.weight = self.token_embed.weight

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Créer le masque causal
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx: [batch, seq_len]
        B, T = idx.shape

        # Embeddings
        x = self.token_embed(idx)  # [B, T, D]
        x = self.pos_encode(x)
        x = self.dropout(x)

        # Masque causal
        mask = self.causal_mask[:, :, :T, :T]

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, mask)

        # Final norm
        x = self.ln_f(x)

        # Output logits
        logits = self.head(x)  # [B, T, vocab_size]

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Génération autoregreessive
        """
        for _ in range(max_new_tokens):
            # Crop context si trop long
            idx_cond = idx if idx.size(1) <= self.config.max_len else idx[:, -self.config.max_len:]

            # Forward pass
            logits = self(idx_cond)

            # Prendre le dernier token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling (optionnel)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Softmax et sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

#### **Étape 5: Training Loop (2-3h)**
```python
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # Data
        inputs = batch['input_ids'].to(device)  # [B, T]
        targets = batch['target_ids'].to(device)  # [B, T]

        # Forward
        logits = model(inputs)  # [B, T, vocab_size]

        # Loss (cross-entropy)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Training loop complet
def train(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Learning rate scheduler (cosine avec warmup)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs
    )

    # Training
    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = evaluate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        # Logging
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pt')
```

### **Dataset utilisé**
- **TinyStories** (petites histoires pour enfants, ~2GB)
- Alternative: WikiText-103, OpenWebText

### **Résultats attendus**
- ✅ Loss converge vers ~3.5-4.0
- ✅ Génération de texte cohérent (3-4 mots consécutifs)
- ✅ Attention weights montrent des patterns sensés

### **Extensions possibles**
1. Implémenter Flash Attention
2. Ajouter KV caching pour l'inference
3. Tester différents positional encodings
4. Visualiser les embeddings avec t-SNE
5. Comparer avec GPT-2 (HuggingFace)

### **Ressources**
- 📄 Paper: "Attention is All You Need" (Vaswani et al., 2017)
- 💻 Code: nanoGPT de Karpathy (référence)
- 📹 Vidéo: Andrej Karpathy - Let's build GPT

---

## 🟢 PROJET 2 : DATA PREPARATION PIPELINE

### **Objectifs**
- Maîtriser le preprocessing de données textuelles à grande échelle
- Créer un pipeline reproductible et scalable
- Comprendre les enjeux de qualité des données

### **Scope**
```
Input: 100GB de texte brut (Common Crawl, Wikipedia, Books, Code)
Output: Dataset nettoyé, dédupliqué, tokenizé (HuggingFace format)
```

### **Pipeline complet**

#### **Étape 1: Data Collection (2-3h)**
```python
from datasets import load_dataset

# Télécharger datasets publics
datasets_to_download = [
    ("mc4", "en"),               # Multilingual C4 (English)
    ("wikipedia", "20231101.en"), # Wikipedia dump
    ("bookcorpus", None),         # Books
    ("the_pile", "all"),          # The Pile (subset)
]

for dataset_name, config in datasets_to_download:
    print(f"Downloading {dataset_name}...")
    if config:
        dataset = load_dataset(dataset_name, config, streaming=True)
    else:
        dataset = load_dataset(dataset_name, streaming=True)

    # Sauvegarder localement
    dataset.save_to_disk(f"data/raw/{dataset_name}")
```

#### **Étape 2: Quality Filtering (4-5h)**
```python
import re
from ftlangdetect import detect  # Fast language detection
from typing import Dict

class QualityFilter:
    """
    Implémentation des Gopher Rules (DeepMind)
    """

    def __init__(self):
        self.min_words = 50
        self.max_words = 100000
        self.min_avg_word_length = 3
        self.max_avg_word_length = 10
        self.max_repetition_ratio = 0.15
        self.max_symbol_to_word_ratio = 0.1

    def filter_document(self, text: str) -> bool:
        """
        Retourne True si le document passe les filtres
        """
        # Détecter la langue
        try:
            lang = detect(text)['lang']
            if lang != 'en':
                return False
        except:
            return False

        # Nombre de mots
        words = text.split()
        if len(words) < self.min_words or len(words) > self.max_words:
            return False

        # Longueur moyenne des mots
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < self.min_avg_word_length or avg_word_len > self.max_avg_word_length:
            return False

        # Ratio de répétition (détection de spam)
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / len(words))
        if repetition_ratio > self.max_repetition_ratio:
            return False

        # Ratio symboles/mots
        symbols = re.findall(r'[^a-zA-Z0-9\s]', text)
        symbol_ratio = len(symbols) / len(words) if len(words) > 0 else 1
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False

        # Filtre de contenu adulte/toxique (utiliser library dédiée)
        if self.contains_toxic_content(text):
            return False

        return True

    def contains_toxic_content(self, text: str) -> bool:
        # Implémenter avec: detoxify, perspective API, ou liste de mots
        from detoxify import Detoxify
        results = Detoxify('original').predict(text)
        return max(results.values()) > 0.7  # Threshold
```

#### **Étape 3: Deduplication (3-4h)**
```python
from datasketch import MinHash, MinHashLSH

class Deduplicator:
    """
    Déduplication avec MinHash LSH (scalable à 100GB+)
    """

    def __init__(self, threshold=0.85, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_ids = set()

    def get_minhash(self, text: str) -> MinHash:
        """Créer MinHash pour un document"""
        minhash = MinHash(num_perm=self.num_perm)
        # Shingles de 3 mots
        words = text.lower().split()
        shingles = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
        return minhash

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """Vérifie si le document est un duplicate"""
        minhash = self.get_minhash(text)

        # Chercher des duplicates existants
        result = self.lsh.query(minhash)

        if len(result) > 0:
            return True  # Duplicate trouvé

        # Ajouter à l'index
        self.lsh.insert(doc_id, minhash)
        self.seen_ids.add(doc_id)

        return False

# Usage
dedup = Deduplicator(threshold=0.85)

unique_docs = []
for idx, doc in enumerate(documents):
    if not dedup.is_duplicate(doc['text'], str(idx)):
        unique_docs.append(doc)

print(f"Kept {len(unique_docs)}/{len(documents)} unique documents")
```

#### **Étape 4: Tokenization (2-3h)**
```python
from transformers import AutoTokenizer

# Option 1: Utiliser tokenizer existant (GPT-2, Llama)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Option 2: Entraîner tokenizer custom
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

def train_custom_tokenizer(files, vocab_size=50000):
    """
    Entraîner un tokenizer BPE custom
    """
    # Créer tokenizer BPE
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    )

    # Entraîner
    tokenizer.train(files, trainer)

    # Sauvegarder
    tokenizer.save("custom_tokenizer.json")

    return tokenizer

# Tokenize datasets
def tokenize_dataset(dataset, tokenizer, max_length=2048):
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=16,  # Parallélisation
    )

    return tokenized
```

#### **Étape 5: Final Dataset Creation (2h)**
```python
from datasets import Dataset, DatasetDict

def create_final_dataset(processed_docs, test_size=0.01, val_size=0.01):
    """
    Créer train/val/test splits et sauvegarder
    """
    # Créer dataset
    dataset = Dataset.from_dict({'text': [doc['text'] for doc in processed_docs]})

    # Split train/temp
    train_test = dataset.train_test_split(test_size=test_size + val_size, seed=42)

    # Split temp → val/test
    test_val = train_test['test'].train_test_split(
        test_size=test_size/(test_size+val_size),
        seed=42
    )

    # Créer DatasetDict
    final_dataset = DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test'],
    })

    # Tokenize
    final_dataset = final_dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, max_length=2048),
        batched=True,
        num_proc=16,
    )

    # Sauvegarder
    final_dataset.save_to_disk("data/processed/final_dataset")

    # Upload vers HuggingFace Hub (optionnel)
    final_dataset.push_to_hub("your_username/your_dataset")

    return final_dataset
```

### **Métriques de qualité**
```python
def compute_dataset_stats(dataset):
    """
    Statistiques du dataset final
    """
    stats = {
        'num_examples': len(dataset),
        'total_tokens': 0,
        'avg_tokens_per_doc': 0,
        'vocab_coverage': 0,
    }

    token_counts = [len(ex['input_ids']) for ex in dataset]
    stats['total_tokens'] = sum(token_counts)
    stats['avg_tokens_per_doc'] = stats['total_tokens'] / len(dataset)

    print(f"Dataset Statistics:")
    print(f"  Examples: {stats['num_examples']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")

    return stats
```

### **Résultats attendus**
- ✅ 100GB brut → ~60GB après filtering
- ✅ ~30GB après déduplication
- ✅ Dataset HuggingFace prêt pour training

---

## 🔵 PROJET 3 : TRAIN NANOGPT (124M PARAMS)

### **Objectifs**
- Entraîner un vrai modèle de langage from scratch
- Maîtriser le training loop complet
- Comprendre les métriques (loss, perplexité)

### **Spécifications**
```yaml
Model:
  architecture: GPT-2 style (decoder-only)
  parameters: 124M
  layers: 12
  heads: 12
  embedding_dim: 768
  context_length: 1024

Training:
  dataset: OpenWebText (~8GB)
  batch_size: 12
  gradient_accumulation: 4  # effective batch = 48
  learning_rate: 6e-4
  warmup_steps: 2000
  max_steps: 100000
  fp16: true

Hardware:
  min: 1x RTX 3090 (24GB)
  recommended: 1x A100 (40GB)
  time: ~3-4 days
```

### **Code complet**
```python
# train.py - basé sur nanoGPT de Karpathy

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Configuration
out_dir = 'out'
eval_interval = 2000
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

# Data
dataset = 'openwebtext'
gradient_accumulation_steps = 4
batch_size = 12
block_size = 1024  # context length

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# Optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile_model = True  # PyTorch 2.0

# -----------------------------------------------------------------------------

# Setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=False, vocab_size=None, dropout=dropout)

print("Initializing model...")
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.to(device)

# Compile model (PyTorch 2.0)
if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Training loop
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate scheduler
def get_lr(it):
    # Warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Training
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

for iter_num in range(max_iters):

    # Learning rate scheduling
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # Forward backward update
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        X, Y = get_batch('train')
        loss.backward()

    # Clip gradients
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    local_iter_num += 1
```

### **Résultats attendus**
- ✅ Val loss converge vers ~3.0-3.2
- ✅ Perplexité: ~20-25
- ✅ Génération cohérente sur 20-30 tokens

---

*(Les 12 autres projets suivent avec le même niveau de détail...)*

---

**[Projets 4-15 continueraient ici avec la même structure détaillée...]**

Pour raisons de concision, je liste les outlines:

## 🔵 PROJET 4 : OPTIMIZE TRAINING RUN
- Profiling avec PyTorch Profiler
- Optimisations mémoire (gradient checkpointing)
- DeepSpeed ZeRO stage 2
- Mixed precision (BF16)
- Target: 2x speedup

## 🔵 PROJET 5 : FINE-TUNE LLAMA 3
- Supervised Fine-Tuning sur dataset custom
- HuggingFace Trainer API
- LoRA (r=16, alpha=32)
- Evaluation metrics

## 🔵 PROJET 6 : LORA ON CONSUMER GPU
- QLoRA (4-bit quantization)
- Fine-tuner Llama 2 7B sur RTX 3090 (24GB)
- bitsandbytes + PEFT
- Merge adapters et deploy

## 🟠 PROJET 7 : RLHF PIPELINE
- Étape 1: SFT
- Étape 2: Reward Model training
- Étape 3: PPO training
- TRL library
- Human preference dataset

## 🔵 PROJET 8 : QUANTIZE FOR CPU
- GPTQ quantization
- llama.cpp conversion (GGUF)
- CPU inference (MacBook M1/M2)
- Benchmark (latency, throughput)

## 🟠 PROJET 9 : DEPLOY VLLM API
- vLLM serving
- FastAPI wrapper
- Load balancing
- Monitoring (Prometheus + Grafana)
- Docker deployment

## 🟠 PROJET 10 : RAG SYSTEM (10K DOCS)
- Document ingestion pipeline
- Chunking (semantic, recursive)
- Embeddings (sentence-transformers)
- Vector DB (Qdrant)
- Re-ranking (cross-encoder)
- Evaluation (RAGAS)

## 🟠 PROJET 11 : AUTONOMOUS AGENT
- ReAct architecture
- 10+ tools (web search, calculator, code execution, etc.)
- Long-term memory (vector DB)
- LangChain + MCP
- Multi-step reasoning

## 🟠 PROJET 12 : FINE-TUNE MULTIMODAL
- LLaVA architecture
- Vision encoder fine-tuning
- Custom vision-language dataset
- VQA evaluation

## 🔴 PROJET 13 : EVAL PIPELINE (CI/CD)
- Automated benchmark suite
- GitHub Actions integration
- Regression testing
- Statistical significance tests
- Cost-aware evaluation

## 🔴 PROJET 14 : ENTERPRISE CHATBOT
- Full-stack application
- RAG + Fine-tuning hybrid
- Multi-tenancy
- Security (auth, PII redaction)
- Monitoring et logging
- Frontend (React)
- Backend (FastAPI)
- Database (PostgreSQL + Qdrant)

## 🔴 PROJET 15 : LLM FROM SCRATCH
- **Durée**: 3 mois
- **Scope complet**:
  - Data collection (200GB)
  - Custom tokenizer training
  - Model architecture (1.5B params)
  - Distributed training (multi-GPU)
  - Checkpointing et reprise
  - Evaluation benchmarks
  - Instruction tuning
  - RLHF
  - Quantization (GPTQ + llama.cpp)
  - Deployment (vLLM)
  - Monitoring production
  - Documentation complète

---

## 📊 PROGRESSION RECOMMANDÉE

### **Track Débutant → Intermédiaire** (3-4 mois)
```
Projets 1 → 2 → 3 → 4 → 5 → 6 → 8
```

### **Track Praticien Rapide** (2 mois)
```
Projets 5 → 6 → 9 → 10
```

### **Track Expert Production** (4-6 mois)
```
Projets 1 → 3 → 5 → 7 → 9 → 10 → 11 → 13 → 14 → 15
```

---

## 🎯 REPOSITORIES GITHUB

Tous les projets auront des repositories dédiés:

```
github.com/ai-bible-2026/project-01-transformer-from-scratch
github.com/ai-bible-2026/project-02-data-preparation-pipeline
...
github.com/ai-bible-2026/project-15-llm-from-scratch
```

Chaque repo contient:
- ✅ Code source complet et commenté
- ✅ README détaillé
- ✅ Requirements.txt / environment.yml
- ✅ Notebooks Jupyter de démonstration
- ✅ Datasets (ou instructions de téléchargement)
- ✅ Checkpoints pré-entraînés (si applicable)
- ✅ Documentation API
- ✅ Tests unitaires

---

## 💡 CONSEILS GÉNÉRAUX

### **Avant de commencer**
1. Setup environnement (conda/venv)
2. Vérifier hardware requirements
3. Lire le chapitre théorique correspondant
4. Cloner le repository du projet

### **Pendant le projet**
1. Suivre les étapes dans l'ordre
2. Comprendre chaque ligne de code (ne pas copier-coller)
3. Expérimenter avec les hyperparamètres
4. Documenter vos observations
5. Débugger méthodiquement

### **Après le projet**
1. Comparer résultats avec benchmarks
2. Créer un notebook de démonstration
3. Partager sur LinkedIn/Twitter
4. Ajouter au portfolio

---

## 🆘 SUPPORT

- **Discord**: #projet-X-help
- **GitHub Issues**: Pour bugs/questions
- **Office Hours**: Hebdomadaires (live coding)

---

**Prêt à construire? Let's code! 🚀**
