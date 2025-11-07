# CHAPITRE 9 : PRÉ-TRAINING D'UN LLM FROM SCRATCH

> *« Pré-entraîner un LLM, c'est comme élever un enfant prodige : des années d'apprentissage général avant toute spécialisation. »*

---

## Introduction : L'Apprentissage à Grande Échelle

### 🎭 Dialogue : Pourquoi Tout Reprendre à Zéro ?

**Alice** : Bob, on peut fine-tuner GPT-3, pourquoi vouloir pré-entraîner un nouveau modèle from scratch ?

**Bob** : Question légitime ! Pré-entraîner coûte des **millions de dollars**. Mais parfois c'est nécessaire :

1. **Domaine spécifique** : Modèle médical avec terminologie technique
2. **Langue rare** : Modèle pour une langue peu représentée (swahili, quechua)
3. **Données propriétaires** : Entreprise avec corpus unique
4. **Contrôle total** : Architecture custom, pas de boîte noire
5. **Recherche** : Tester nouvelles architectures/objectifs

**Alice** : Mais c'est faisable pour une startup ?

**Bob** : Aujourd'hui, oui !
- **2020** : GPT-3 = $5M, cluster de 10,000 GPUs
- **2023** : LLaMA-7B = $50k, 2048 GPUs pendant 21 jours
- **2024** : Modèles 1-3B = $5k-10k, possible sur 8× A100

Avec les bonnes techniques (efficient attention, mixed precision, etc.), le pré-training se démocratise.

### 📊 Coûts et Échelle

| Modèle | Paramètres | Tokens | GPUs | Durée | Coût Estimé |
|--------|-----------|--------|------|-------|-------------|
| **GPT-2** | 1.5B | 40B | 256 TPU | ~1 mois | ~$50k |
| **GPT-3** | 175B | 300B | 10,000 V100 | ~1 mois | ~$5M |
| **LLaMA** | 7B | 1T | 2048 A100 | 21 jours | ~$50k |
| **LLaMA** | 65B | 1.4T | 2048 A100 | 21 jours | ~$500k |
| **Bloom** | 176B | 366B | 384 A100 | 3.5 mois | ~$2M |
| **GPT-4** (estimé) | 1.8T | 13T | ~25,000 A100 | ~3 mois | ~$100M |

### 🎯 Anecdote : La Naissance de BERT

**Octobre 2018, Google AI**

Jacob Devlin et son équipe ont une idée radicale : au lieu de prédire le prochain mot (GPT), **masquer** des mots aléatoires et les prédire (MLM - Masked Language Modeling).

**Corpus** : BooksCorpus (800M mots) + Wikipedia anglaise (2,500M mots) = 3.3B mots

**Infrastructure** :
- BERT-Base : 4 TPUs (16 GB chacune) pendant 4 jours
- BERT-Large : 16 TPUs pendant 4 jours
- **Coût total** : ~$7,000 (TPU pricing 2018)

**Résultat** : BERT explose tous les records sur 11 tâches NLP. Le paradigme "pré-training + fine-tuning" devient le standard.

**Impact** : Démocratisation du NLP. Avant BERT, il fallait des millions d'exemples labelisés par tâche. Après, quelques milliers suffisent (fine-tuning).

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Décider quand pré-entraîner vs fine-tuner
- ✅ Préparer un corpus de pré-training (nettoyage, déduplication)
- ✅ Choisir l'objectif de pré-training (CLM, MLM, etc.)
- ✅ Configurer l'infrastructure (multi-GPU, mixed precision)
- ✅ Implémenter le training loop complet
- ✅ Monitorer et debugger l'entraînement
- ✅ Estimer les coûts et optimiser

**Difficulté** : 🔴🔴🔴🔴🔴 (Expert)
**Prérequis** : Chapitres 4, 7, 10, PyTorch avancé, expérience distributed training
**Temps de lecture** : ~150 minutes

---

## Quand Pré-Entraîner vs Fine-Tuner ?

### Arbre de Décision

```
Avez-vous besoin d'un modèle custom ?
├─ NON → Utilisez modèle pré-entraîné existant (GPT, LLaMA, etc.)
│
└─ OUI → Pourquoi ?
    ├─ Domaine très spécifique (médical, légal, code)
    │   ├─ Corpus < 10B tokens → Fine-tune modèle existant
    │   └─ Corpus > 100B tokens → Pré-entraîner from scratch
    │
    ├─ Langue rare/peu représentée
    │   └─ Pré-entraîner (modèles existants inefficaces)
    │
    ├─ Données propriétaires/sensibles
    │   └─ Pré-entraîner (contrôle total, pas de fuite)
    │
    ├─ Architecture custom
    │   └─ Pré-entraîner (recherche, innovation)
    │
    └─ Budget ?
        ├─ < $10k → Fine-tune ou modèle petit (1-3B)
        ├─ $10k-100k → Modèle 7-13B
        └─ > $1M → Modèle 70B+
```

### Exemples Concrets

**Cas 1 : Startup FinTech**
- Besoin : Chatbot analyse financière
- Corpus : Rapports financiers (50B tokens)
- **Décision** : Fine-tune LLaMA-7B sur données financières ($500)

**Cas 2 : Gouvernement (langue bretonne)**
- Besoin : Modèle langue bretonne (peu de données en ligne)
- Corpus : Documents collectés (10B tokens)
- **Décision** : Pré-entraîner petit modèle 1B from scratch ($5k)

**Cas 3 : Big Tech (nouveau modèle SOTA)**
- Besoin : Battre GPT-4
- Corpus : Web scraping + datasets propriétaires (10T tokens)
- **Décision** : Pré-entraîner 500B+ modèle ($50M+)

---

## Préparation du Corpus de Pré-Training

### Les Trois Étapes Cruciales

#### 1. Collecte des Données

**Sources communes** :

| Source | Taille | Qualité | Accès |
|--------|--------|---------|-------|
| **Common Crawl** | 250TB+ | Variable | Public |
| **Wikipedia** | 20GB | Élevée | Public |
| **Books** | 100GB+ | Élevée | Légal complexe |
| **GitHub** | 1TB+ (code) | Moyenne | Public (filtrer licences) |
| **Reddit** | 500GB+ | Variable | Public (Pushshift) |
| **ArXiv** | 100GB (papers) | Élevée | Public |
| **Propriétaire** | Variable | Variable | Privé |

**Exemple : Télécharger Common Crawl**

```python
import requests
from bs4 import BeautifulSoup
import gzip
import io

def download_common_crawl_sample(num_files=10):
    """
    Télécharge un échantillon de Common Crawl.

    Common Crawl structure:
    - Monthly crawls
    - WARC files (Web ARChive format)
    """
    base_url = "https://data.commoncrawl.org"

    # Liste des crawls disponibles
    crawls_url = f"{base_url}/crawl-data/CC-MAIN-2024-10/warc.paths.gz"

    # Télécharger liste
    response = requests.get(crawls_url)
    paths = gzip.decompress(response.content).decode('utf-8').split('\n')

    documents = []

    for path in paths[:num_files]:
        if not path:
            continue

        warc_url = f"{base_url}/{path}"
        print(f"Downloading {warc_url}...")

        # Télécharger WARC
        response = requests.get(warc_url, stream=True)

        # Parser WARC (simplifié)
        # En production : utiliser warcio library
        content = gzip.decompress(response.content).decode('utf-8', errors='ignore')

        # Extraire texte HTML
        # ... parsing logic

    return documents

# Note: Common Crawl fait 250TB+, télécharger tout est impraticable
# En production: utiliser AWS EMR ou Spark cluster
```

#### 2. Nettoyage et Filtrage

**Problèmes du web** :
- Spam, publicités, contenu dupliqué
- HTML, JavaScript, CSS
- Langues multiples mélangées
- Contenu de basse qualité

**Pipeline de nettoyage** :

```python
import re
from langdetect import detect
import unicodedata

class TextCleaner:
    """
    Pipeline de nettoyage pour corpus web.
    """
    def __init__(self, target_language='en', min_words=50):
        self.target_language = target_language
        self.min_words = min_words

    def clean(self, text):
        """Nettoie un document."""
        # 1. Normalisation Unicode
        text = unicodedata.normalize('NFKC', text)

        # 2. Supprimer HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Supprimer URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # 4. Supprimer emails
        text = re.sub(r'\S+@\S+', '', text)

        # 5. Normaliser whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def filter(self, text):
        """Filtre les documents de basse qualité."""
        # 1. Longueur minimum
        words = text.split()
        if len(words) < self.min_words:
            return False

        # 2. Détection de langue
        try:
            lang = detect(text)
            if lang != self.target_language:
                return False
        except:
            return False

        # 3. Ratio caractères alphanumériques
        alphanum = sum(c.isalnum() for c in text)
        if alphanum / len(text) < 0.8:
            return False  # Trop de caractères spéciaux

        # 4. Répétition excessive
        # Détecter "aaaaaaa" ou "test test test test"
        if self._has_excessive_repetition(text):
            return False

        return True

    def _has_excessive_repetition(self, text, threshold=0.3):
        """Détecte répétitions excessives."""
        words = text.lower().split()
        if len(words) < 10:
            return False

        # Ratio mots uniques
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < threshold

# Utilisation
cleaner = TextCleaner(target_language='en', min_words=50)

documents_raw = [...]  # Documents téléchargés
documents_clean = []

for doc in documents_raw:
    cleaned = cleaner.clean(doc)
    if cleaner.filter(cleaned):
        documents_clean.append(cleaned)

print(f"Kept {len(documents_clean)} / {len(documents_raw)} documents")
```

#### 3. Déduplication

**Problème** : Le web contient ~50% de contenu dupliqué !

**Impact** : Modèle mémorise au lieu d'apprendre.

**Solution : MinHash + LSH**

```python
from datasketch import MinHash, MinHashLSH

class Deduplicator:
    """
    Déduplication avec MinHash LSH.
    """
    def __init__(self, threshold=0.8, num_perm=128):
        """
        Args:
            threshold: Similarité Jaccard minimum pour considérer dupliqué
            num_perm: Nombre de permutations MinHash
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_hashes = set()

    def get_minhash(self, text):
        """Calcule MinHash d'un document."""
        m = MinHash(num_perm=self.num_perm)

        # Tokenize en mots (simplifié)
        words = text.lower().split()

        # Shingles (n-grams de mots)
        shingles = [' '.join(words[i:i+3]) for i in range(len(words)-2)]

        for shingle in shingles:
            m.update(shingle.encode('utf-8'))

        return m

    def is_duplicate(self, text, doc_id):
        """
        Vérifie si document est dupliqué.

        Returns:
            True si dupliqué, False sinon
        """
        minhash = self.get_minhash(text)

        # Chercher duplicates
        duplicates = self.lsh.query(minhash)

        if duplicates:
            return True  # Dupliqué trouvé

        # Ajouter à l'index
        self.lsh.insert(doc_id, minhash)
        return False

# Utilisation
dedup = Deduplicator(threshold=0.8)

documents_unique = []

for i, doc in enumerate(documents_clean):
    if not dedup.is_duplicate(doc, doc_id=f"doc_{i}"):
        documents_unique.append(doc)

print(f"Removed {len(documents_clean) - len(documents_unique)} duplicates")
print(f"Final corpus: {len(documents_unique)} documents")
```

### Tokenization du Corpus

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_tokenizer(corpus, vocab_size=50000):
    """
    Entraîne un tokenizer BPE sur le corpus.
    """
    # Initialiser BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )

    # Entraîner sur corpus
    tokenizer.train_from_iterator(corpus, trainer)

    # Sauvegarder
    tokenizer.save("tokenizer.json")

    return tokenizer

# Entraîner tokenizer
tokenizer = train_tokenizer(documents_unique, vocab_size=50000)

# Tokeniser le corpus
def tokenize_corpus(corpus, tokenizer, max_length=2048):
    """Tokenise le corpus entier."""
    tokenized = []

    for doc in corpus:
        encoding = tokenizer.encode(doc)

        # Découper en chunks de max_length
        tokens = encoding.ids
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i+max_length]
            if len(chunk) >= max_length // 2:  # Garder seulement chunks assez longs
                tokenized.append(chunk)

    return tokenized

tokenized_corpus = tokenize_corpus(documents_unique, tokenizer)
print(f"Total chunks: {len(tokenized_corpus)}")
```

---

## Objectifs de Pré-Training

### 1. Causal Language Modeling (CLM) - GPT Style

**Objectif** : Prédire le token suivant.

```
Input:  "The cat sat on the"
Target: "The cat sat on the mat"
         ^   ^   ^   ^   ^   ^
         Prédire chaque token à partir du contexte gauche
```

**Loss** : Cross-entropy sur chaque position.

```python
class CausalLMTrainer:
    """
    Trainer pour Causal Language Modeling.
    """
    def compute_loss(self, model, input_ids):
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            loss: Scalar
        """
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        # Shift pour prédiction
        # Input:  [BOS, The, cat, sat]
        # Target: [The, cat, sat, EOS]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return loss
```

### 2. Masked Language Modeling (MLM) - BERT Style

**Objectif** : Prédire tokens masqués.

```
Original: "The cat sat on the mat"
Masked:   "The [MASK] sat on the [MASK]"
Target:   Prédire "cat" et "mat"
```

**Stratégie de masquage (BERT)** :
- 80% : Remplacer par [MASK]
- 10% : Remplacer par token aléatoire
- 10% : Garder original

```python
import random
import torch

class MLMDataCollator:
    """
    Data collator pour MLM (BERT-style).
    """
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.token_to_id("[MASK]")

    def mask_tokens(self, inputs):
        """
        Masque tokens pour MLM.

        Args:
            inputs: [batch, seq_len]

        Returns:
            inputs: [batch, seq_len] avec tokens masqués
            labels: [batch, seq_len] (-100 pour tokens non-masqués)
        """
        labels = inputs.clone()

        # Probabilité de masquage
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Ne pas masquer tokens spéciaux (PAD, BOS, EOS)
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_id in [self.tokenizer.token_to_id(t) for t in ["<PAD>", "<BOS>", "<EOS>"]]:
            special_tokens_mask |= (labels == special_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Échantillonner tokens à masquer
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels : -100 pour tokens non-masqués (ignorés dans loss)
        labels[~masked_indices] = -100

        # 80% : [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% : token aléatoire
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.get_vocab()), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% : garder original (déjà fait, rien à changer)

        return inputs, labels
```

### 3. Autres Objectifs

| Objectif | Description | Modèles |
|----------|-------------|---------|
| **NSP** (Next Sentence Prediction) | Prédire si phrase B suit phrase A | BERT (abandonné dans RoBERTa) |
| **SOP** (Sentence Order Prediction) | Prédire ordre des phrases | ALBERT |
| **Denoising** | Reconstruire texte avec bruit | T5, BART |
| **Contrastive** | Embeddings similaires pour augmentations | SimCLR NLP |

---

## Configuration et Architecture

### Hyperparamètres de Pré-Training

```python
from dataclasses import dataclass

@dataclass
class PretrainingConfig:
    """
    Configuration pour pré-training.
    """
    # Modèle
    vocab_size: int = 50000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072  # 4 × d_model
    max_seq_len: int = 2048
    dropout: float = 0.1

    # Training
    batch_size: int = 256  # Per GPU
    gradient_accumulation_steps: int = 4  # Effective batch: 256 × 4 = 1024
    learning_rate: float = 6e-4
    warmup_steps: int = 10000
    max_steps: int = 500000  # ~1 epoch sur 512B tokens
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Optimisation
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 100
    eval_steps: int = 5000
    save_steps: int = 10000

    # Hardware
    num_gpus: int = 8
    num_workers: int = 4  # DataLoader workers

config = PretrainingConfig()
```

### Instanciation du Modèle

```python
from transformers import GPT2Config, GPT2LMHeadModel

def create_model(config):
    """
    Crée un modèle GPT-2 style.
    """
    model_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_len,
        n_embd=config.d_model,
        n_layer=config.n_layers,
        n_head=config.n_heads,
        n_inner=config.d_ff,
        resid_pdrop=config.dropout,
        embd_pdrop=config.dropout,
        attn_pdrop=config.dropout,
    )

    model = GPT2LMHeadModel(model_config)

    # Nombre de paramètres
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    return model

model = create_model(config)
```

---

## Training Loop Complet

### Distributed Training Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_distributed():
    """Initialise distributed training."""
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    return local_rank

def cleanup_distributed():
    """Nettoie distributed training."""
    dist.destroy_process_group()

# Setup
local_rank = setup_distributed()
device = torch.device(f'cuda:{local_rank}')
```

### DataLoader

```python
from torch.utils.data import Dataset

class PretrainingDataset(Dataset):
    """
    Dataset pour pré-training.
    """
    def __init__(self, tokenized_corpus, max_seq_len=2048):
        self.data = tokenized_corpus
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]

        # Padding si nécessaire
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

# Dataset
dataset = PretrainingDataset(tokenized_corpus, max_seq_len=config.max_seq_len)

# Sampler distribué
sampler = DistributedSampler(dataset, num_replicas=config.num_gpus, rank=local_rank)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    sampler=sampler,
    num_workers=config.num_workers,
    pin_memory=True
)
```

### Optimizer et Scheduler

```python
from transformers import get_linear_schedule_with_warmup

# Model sur GPU
model = model.to(device)
model = DDP(model, device_ids=[local_rank])

# Optimizer (AdamW)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),  # GPT-3 values
    eps=1e-8
)

# Scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=config.max_steps
)

# GradScaler pour mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler() if config.fp16 else None
```

### Main Training Loop

```python
import wandb
from tqdm import tqdm

def train(model, dataloader, optimizer, scheduler, scaler, config):
    """
    Boucle de training principale.
    """
    model.train()
    global_step = 0

    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Training loop
    for epoch in range(100):  # Assez d'epochs pour atteindre max_steps
        sampler.set_epoch(epoch)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}") if local_rank == 0 else dataloader

        for step, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # Forward pass
            with autocast() if config.fp16 else nullcontext():
                outputs = model(batch, labels=batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            # Backward pass
            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if config.fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                if config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % config.logging_steps == 0 and local_rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({
                        'loss': loss.item() * config.gradient_accumulation_steps,
                        'learning_rate': lr,
                        'global_step': global_step
                    })

                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

                # Evaluation
                if global_step % config.eval_steps == 0:
                    eval_loss = evaluate(model, val_dataloader, device)
                    if local_rank == 0:
                        wandb.log({'eval_loss': eval_loss, 'global_step': global_step})
                    model.train()

                # Checkpoint
                if global_step % config.save_steps == 0 and local_rank == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, config)

                # Max steps
                if global_step >= config.max_steps:
                    return

def evaluate(model, dataloader, device):
    """Évalue le modèle."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches

def save_checkpoint(model, optimizer, scheduler, global_step, config):
    """Sauvegarde un checkpoint."""
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'global_step': global_step
    }

    path = f"checkpoint-{global_step}.pt"
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
```

---

## Monitoring et Debugging

### Métriques Clés

```python
class MetricsTracker:
    """
    Track métriques durant training.
    """
    def __init__(self):
        self.metrics = {
            'loss': [],
            'perplexity': [],
            'gradient_norm': [],
            'learning_rate': []
        }

    def log(self, loss, model, optimizer, scheduler):
        """Log métriques."""
        # Loss et perplexity
        self.metrics['loss'].append(loss)
        ppl = torch.exp(torch.tensor(loss))
        self.metrics['perplexity'].append(ppl.item())

        # Gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.metrics['gradient_norm'].append(total_norm)

        # Learning rate
        self.metrics['learning_rate'].append(scheduler.get_last_lr()[0])

    def plot(self):
        """Affiche graphiques."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.metrics['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')

        # Perplexity
        axes[0, 1].plot(self.metrics['perplexity'])
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('PPL')

        # Gradient norm
        axes[1, 0].plot(self.metrics['gradient_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')

        # Learning rate
        axes[1, 1].plot(self.metrics['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        plt.show()
```

### Problèmes Communs et Solutions

| Problème | Symptôme | Solution |
|----------|----------|----------|
| **Loss NaN** | Loss devient NaN après quelques steps | Réduire LR, vérifier données, gradient clipping |
| **Loss stagnante** | Pas d'amélioration après 10k steps | Augmenter LR, vérifier données, réduire batch size |
| **OOM** | Out of Memory CUDA | Réduire batch size, activer gradient checkpointing, FP16 |
| **Slow training** | < 100 samples/sec | Optimiser DataLoader (num_workers), pin_memory, prefetch |
| **Divergence** | Loss explose après warmup | Augmenter warmup steps, réduire LR max |

---

## Estimation des Coûts

### Calcul Théorique

**FLOPs par forward pass** :
```
FLOPs ≈ 6 × N × D

où N = nombre de paramètres
    D = nombre de tokens dans le batch
```

**Exemple : GPT-3 (175B params)**
```
Batch size: 3.2M tokens
FLOPs/forward: 6 × 175B × 3.2M = 3.36 × 10^18 FLOPs

GPU A100: 312 TFLOPS (FP16)
Temps/forward: 3.36 × 10^18 / 312 × 10^12 = 10.8 secondes

Pour 300B tokens:
Nombre de forwards: 300B / 3.2M ≈ 94,000
Temps total: 94,000 × 10.8s = 1,015,200s ≈ 280 heures sur 1 A100
```

**Avec 10,000 A100** : 280h / 10,000 = ~1.7 minutes... **MAIS** :
- Overhead communication : ×3
- Inefficiencies (idle time) : ×1.5
- Total réel : ~30-40 jours

### Coûts Cloud

| Provider | GPU | Prix/heure | 8× GPUs/h | 1 mois (720h) |
|----------|-----|------------|-----------|---------------|
| **AWS** | A100 40GB | $4.00 | $32 | $23,040 |
| **GCP** | A100 40GB | $3.67 | $29.36 | $21,139 |
| **Azure** | A100 40GB | $3.40 | $27.20 | $19,584 |
| **Lambda Labs** | A100 40GB | $1.10 | $8.80 | $6,336 |

**LLaMA-7B (1T tokens, 21 jours)** :
- 2048× A100 = 256 nodes × 8 GPUs
- Lambda Labs : $1.10 × 8 × 256 × 24 × 21 = ~$900k... **réduction avec contrats**
- Coût réel estimé : **$50k-100k**

---

## 💡 Analogie : Pré-Training comme l'Éducation

- **Corpus = Bibliothèque** : Plus large et diverse, meilleure culture générale
- **Nettoyage = Curation** : Retirer livres de mauvaise qualité
- **Déduplication = Éviter redondance** : Lire 10 fois le même livre n'apporte rien
- **CLM = Apprendre en lisant** : Prédire suite de l'histoire
- **MLM = Textes à trous** : Deviner mots manquants (exercice scolaire classique)
- **Fine-tuning = Spécialisation** : Médecine, droit, etc. après culture générale

---

## Conclusion

### 🎭 Dialogue Final : De l'Ambition à la Réalité

**Alice** : Pré-entraîner un LLM from scratch, c'est vraiment accessible maintenant ?

**Bob** : Ça dépend de l'échelle :
- **1-3B params** : $5k-10k → accessible startups/universités
- **7-13B params** : $50k-100k → grosses startups, labs de recherche
- **70B+ params** : $500k+ → Big Tech uniquement

**Alice** : Quels sont les facteurs de succès ?

**Bob** :
1. **Données de qualité** > Quantité brute
2. **Compute efficient** : Flash Attention, mixed precision, FSDP
3. **Monitoring rigoureux** : Détecter problèmes tôt
4. **Patience** : Pré-training prend des semaines, pas de shortcuts

**Alice** : Alternatives au full pré-training ?

**Bob** : **Continued pre-training** :
- Partir d'un modèle existant (LLaMA, GPT-2)
- Continuer pré-training sur votre corpus
- **10× moins cher** que from scratch
- Bonne option si domaine pas trop différent

L'avenir : pré-training devient plus accessible, mais reste un investissement majeur.

---

## Ressources

### 📚 Papers Fondamentaux

1. **"Language Models are Unsupervised Multitask Learners"** (GPT-2, Radford et al., 2019)
2. **"BERT: Pre-training of Deep Bidirectional Transformers"** (Devlin et al., 2018)
3. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)
4. **"Training Compute-Optimal Large Language Models"** (Chinchilla, Hoffmann et al., 2022)

### 🛠️ Outils

```bash
# Training à grande échelle
pip install deepspeed accelerate

# Monitoring
pip install wandb tensorboard

# Déduplication
pip install datasketch

# Distributed
pip install torch torchrun
```

### 🔗 Ressources

- **Common Crawl** : https://commoncrawl.org/
- **The Pile** : https://pile.eleuther.ai/ (open dataset 825GB)
- **Megatron-LM** : https://github.com/NVIDIA/Megatron-LM
- **DeepSpeed** : https://www.deepspeed.ai/

---

**🎓 Bravo !** Vous comprenez maintenant le pré-training from scratch, le processus le plus coûteux mais fondamental de l'IA moderne. Prochain chapitre : **Chapitre 11 - Prompt Engineering** pour maximiser l'utilisation des modèles pré-entraînés ! 🚀

