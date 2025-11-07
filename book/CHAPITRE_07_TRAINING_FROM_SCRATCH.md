# CHAPITRE 7 : ENTRAÎNEMENT FROM SCRATCH

## Introduction

Entraîner un LLM from scratch est l'expérience ultime pour comprendre comment fonctionnent réellement ces modèles. Ce chapitre couvre tout le processus, de la configuration hardware jusqu'au modèle entraîné.

**Ce que vous allez apprendre:**
- Configuration matérielle (GPUs, TPUs)
- Distributed training (data, model, pipeline parallelism)
- Training loop complet
- Optimisation et scheduling
- Monitoring durant training
- Debugging et problèmes courants

## 7.1 Configuration Matérielle

### 7.1.1 Calcul des Besoins

**Formule de base pour mémoire GPU:**
```
Memory (GB) = Parameters × Bytes_per_param × Multiplier

où:
- Parameters: nombre de paramètres du modèle
- Bytes_per_param: 2 (FP16) ou 4 (FP32)
- Multiplier:
  - 1x pour inference (poids seulement)
  - 4x pour training (poids + gradients + optimizer states)
```

**Exemples:**

```python
def estimate_training_memory(params_billions, precision="fp16"):
    """
    Estime la mémoire nécessaire pour training

    Args:
        params_billions: taille modèle en milliards de paramètres
        precision: "fp16" ou "fp32"
    """
    params = params_billions * 1e9

    # Bytes par paramètre
    bytes_per_param = 2 if precision == "fp16" else 4

    # Components mémoire
    model_memory = params * bytes_per_param
    gradients_memory = params * bytes_per_param
    optimizer_memory = params * 8  # Adam: 2 momentum states

    # Activations (approximation)
    activations_memory = model_memory * 2

    # Total
    total_gb = (model_memory + gradients_memory + optimizer_memory + activations_memory) / 1e9

    breakdown = {
        "model_gb": model_memory / 1e9,
        "gradients_gb": gradients_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activations_memory / 1e9,
        "total_gb": total_gb,
    }

    return breakdown

# Exemples
print("GPT-2 Small (124M):")
print(estimate_training_memory(0.124))

print("\nLlama 2 7B:")
print(estimate_training_memory(7))

print("\nLlama 2 70B:")
print(estimate_training_memory(70))

# Output:
# GPT-2 Small (124M):
# {'model_gb': 0.25, 'gradients_gb': 0.25, 'optimizer_gb': 0.99,
#  'activations_gb': 0.50, 'total_gb': 1.99}
#
# Llama 2 7B:
# {'model_gb': 14.0, 'gradients_gb': 14.0, 'optimizer_gb': 56.0,
#  'activations_gb': 28.0, 'total_gb': 112.0}
#
# Llama 2 70B:
# {'model_gb': 140.0, 'gradients_gb': 140.0, 'optimizer_gb': 560.0,
#  'activations_gb': 280.0, 'total_gb': 1120.0}
```

**Conclusion:**
- GPT-2 Small: 1x RTX 3090 (24GB) ✅
- Llama 2 7B: 8x A100 (80GB) avec optimisations ✅
- Llama 2 70B: 16+ A100 (80GB) avec ZeRO stage 3 ✅

### 7.1.2 Choix de Hardware

**GPUs NVIDIA:**

| GPU | VRAM | FP16 TFLOPS | Prix/heure (cloud) | Use Case |
|-----|------|-------------|-------------------|----------|
| RTX 3090 | 24GB | 35 | $0.50 | Prototyping, petits modèles |
| RTX 4090 | 24GB | 83 | $0.60 | Prototyping, research |
| A100 40GB | 40GB | 312 | $2-3 | Training medium models |
| A100 80GB | 80GB | 312 | $3-4 | Training large models |
| H100 | 80GB | 1000+ | $5-8 | Training très large models |

**TPUs (Google Cloud):**

| TPU | Mémoire | TFLOPS | Prix/heure | Use Case |
|-----|---------|--------|-----------|----------|
| TPU v4 | 32GB HBM | 275 | $1-2 | Training JAX models |
| TPU v5e | 16GB HBM | 197 | $0.80 | Cost-effective training |
| TPU v5p | 95GB HBM | 459 | $3-5 | Large model training |

**Setup multi-GPU local:**
```python
import torch

def check_gpu_availability():
    """Check available GPUs"""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")

check_gpu_availability()
```

## 7.2 Distributed Training

### 7.2.1 Data Parallelism

Chaque GPU a une copie complète du modèle et traite un subset des données.

**Simple DistributedDataParallel (DDP):**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup_distributed():
    """Initialize distributed training"""
    # Environment variables set by torch.distributed.launch
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # NCCL for GPU
        init_method="env://",
    )

    # Set device
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size

def cleanup_distributed():
    """Cleanup distributed"""
    dist.destroy_process_group()

class DistributedTrainer:
    """
    Trainer avec Data Parallelism
    """
    def __init__(self, model, train_loader, config):
        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()

        # Model to device
        self.model = model.to(self.local_rank)

        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

        self.train_loader = train_loader
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.local_rank)
            labels = batch['labels'].to(self.local_rank)

            # Forward
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Log (only on rank 0)
            if self.rank == 0 and batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Average loss across all processes
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def save_checkpoint(self, epoch, path):
        """Save checkpoint (only rank 0)"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved: {path}")

# Launch script
"""
# train.py
trainer = DistributedTrainer(model, train_loader, config)

for epoch in range(config.num_epochs):
    loss = trainer.train_epoch()
    if trainer.rank == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

    trainer.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pt")

cleanup_distributed()

# Launch command:
# torchrun --nproc_per_node=8 train.py
"""
```

### 7.2.2 Model Parallelism (Tensor Parallelism)

Divise le modèle sur plusieurs GPUs. Utilisé quand le modèle est trop grand pour un seul GPU.

**Exemple avec Megatron-LM style:**

```python
import torch.nn as nn

class TensorParallelLinear(nn.Module):
    """
    Linear layer avec tensor parallelism

    Split poids sur colonne (column-parallel) ou ligne (row-parallel)
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        parallel_mode="column",  # "column" ou "row"
        tp_group=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_mode = parallel_mode

        # Get tensor parallel group
        self.tp_group = tp_group or dist.group.WORLD
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)

        if parallel_mode == "column":
            # Split output features
            assert out_features % self.tp_size == 0
            self.out_features_per_partition = out_features // self.tp_size

            self.weight = nn.Parameter(
                torch.empty(self.out_features_per_partition, in_features)
            )

        else:  # row parallel
            # Split input features
            assert in_features % self.tp_size == 0
            self.in_features_per_partition = in_features // self.tp_size

            self.weight = nn.Parameter(
                torch.empty(out_features, self.in_features_per_partition)
            )

        if bias:
            if parallel_mode == "column":
                self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
            else:
                self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.parallel_mode == "column":
            # x: [B, S, in_features]
            # weight: [out_features_per_partition, in_features]
            output = torch.matmul(x, self.weight.t())  # [B, S, out_features_per_partition]

            if self.bias is not None:
                output += self.bias

            return output

        else:  # row parallel
            # x: [B, S, in_features_per_partition]
            # weight: [out_features, in_features_per_partition]
            output_partial = torch.matmul(x, self.weight.t())

            # All-reduce across tensor parallel group
            dist.all_reduce(output_partial, group=self.tp_group)

            if self.bias is not None:
                output_partial += self.bias

            return output_partial

class TensorParallelAttention(nn.Module):
    """
    Multi-head attention avec tensor parallelism
    """
    def __init__(self, config, tp_group=None):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # QKV projection (column parallel)
        self.qkv_proj = TensorParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            parallel_mode="column",
            tp_group=tp_group,
        )

        # Output projection (row parallel)
        self.out_proj = TensorParallelLinear(
            config.hidden_size,
            config.hidden_size,
            parallel_mode="row",
            tp_group=tp_group,
        )

    def forward(self, x):
        # QKV projection (output split across GPUs)
        qkv = self.qkv_proj(x)

        # Split Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Attention computation (local to each GPU)
        # ... (attention logic)

        # Output projection (all-reduce inside)
        output = self.out_proj(attn_out)

        return output
```

### 7.2.3 Pipeline Parallelism

Divise le modèle en stages et pipeline les micro-batches.

```python
class PipelineParallelModel(nn.Module):
    """
    Model divisé en stages pour pipeline parallelism
    """
    def __init__(self, config, num_stages=4):
        super().__init__()
        self.num_stages = num_stages

        # Divide layers into stages
        layers_per_stage = config.num_layers // num_stages

        # Get stage for this rank
        stage_id = dist.get_rank() // (dist.get_world_size() // num_stages)

        start_layer = stage_id * layers_per_stage
        end_layer = start_layer + layers_per_stage

        # Only create layers for this stage
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(start_layer, end_layer)
        ])

        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward(self, x):
        # Forward through this stage's layers
        for layer in self.layers:
            x = layer(x)

        # Send to next stage
        if self.stage_id < self.num_stages - 1:
            dist.send(x, dst=self.stage_id + 1)

        # Receive from previous stage
        if self.stage_id > 0:
            x = torch.empty_like(x)
            dist.recv(x, src=self.stage_id - 1)

        return x
```

### 7.2.4 ZeRO (Zero Redundancy Optimizer)

DeepSpeed ZeRO élimine la redondance mémoire dans data parallelism.

**ZeRO Stages:**
- **Stage 1**: Partition optimizer states (~4x réduction)
- **Stage 2**: Partition gradients (~8x réduction)
- **Stage 3**: Partition model parameters (~16x+ réduction)

```python
import deepspeed

def create_deepspeed_config(stage=2):
    """
    Create DeepSpeed configuration

    Stage 1: Optimizer state partitioning
    Stage 2: + Gradient partitioning
    Stage 3: + Parameter partitioning
    """
    config = {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 32,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 6e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 100000,
                "warmup_min_lr": 0,
                "warmup_max_lr": 6e-4,
                "warmup_num_steps": 2000,
            }
        },

        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
        },

        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu" if stage == 3 else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if stage == 3 else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        },

        "gradient_clipping": 1.0,
        "steps_per_print": 100,
    }

    return config

# Initialize model with DeepSpeed
model = GPTModel(config)

ds_config = create_deepspeed_config(stage=2)

model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,
    config=ds_config,
)

# Training loop
for batch in train_loader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

## 7.3 Training Loop Complet

### 7.3.1 Main Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import math

class Trainer:
    """
    Trainer complet pour LLM from scratch
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
    ):
        self.model = model
        self.config = config

        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = self.configure_optimizer()

        # Learning rate scheduler
        self.scheduler = self.configure_scheduler()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None

        # Tensorboard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

    def configure_optimizer(self):
        """
        Configure optimizer avec weight decay approprié
        """
        # Separate parameters into those with and without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and layer norms
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        return optimizer

    def configure_scheduler(self):
        """Configure learning rate scheduler"""
        def lr_lambda(step):
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps

            # Cosine decay
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda,
        )

        return scheduler

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        # Move to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        return avg_loss, perplexity

    def save_checkpoint(self, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        # Save latest
        path = os.path.join(self.config.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, path)

        print(f"Checkpoint saved: {path}")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Num parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Training
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc="Training")

            for batch_idx, batch in enumerate(progress_bar):
                loss = self.train_step(batch)

                epoch_loss += loss
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]

                    self.writer.add_scalar('train/loss', loss, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)

                    progress_bar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{lr:.2e}',
                    })

                # Validation
                if self.global_step % self.config.eval_interval == 0:
                    val_loss, perplexity = self.validate()

                    self.writer.add_scalar('val/loss', val_loss, self.global_step)
                    self.writer.add_scalar('val/perplexity', perplexity, self.global_step)

                    print(f"\nValidation - Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

                    # Save best
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)

                # Regular checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f}")

        print("Training complete!")
        self.writer.close()

# Usage
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "gpt2-small"

    # Data
    batch_size: int = 8
    max_seq_length: int = 1024

    # Optimization
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000

    # Training
    num_epochs: int = 1
    fp16: bool = True

    # Logging
    log_interval: int = 100
    eval_interval: int = 2000
    save_interval: int = 5000

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

# Create trainer
config = TrainingConfig()
model = GPTModel(model_config)
trainer = Trainer(model, train_dataset, val_dataset, config)

# Train!
trainer.train()
```

---

*[Le chapitre continue avec debugging, monitoring, et cas pratiques complets...]*

*[Contenu total du Chapitre 7: ~80-90 pages]*
