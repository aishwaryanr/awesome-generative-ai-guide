# CHAPITRE 14 : REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)

## Introduction

RLHF est la technique qui a transformé les LLMs de simples "text completion engines" en assistants utiles et sûrs. C'est ce qui différencie GPT-3 de ChatGPT, Llama 2 de Llama 2 Chat.

**Pipeline RLHF:**
1. **Supervised Fine-Tuning (SFT)** - Base model → instruction-following
2. **Reward Model Training** - Apprendre les préférences humaines
3. **RL Fine-Tuning (PPO)** - Optimiser selon le reward model

## 14.1 Vue d'ensemble du Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF PIPELINE COMPLET                         │
└─────────────────────────────────────────────────────────────────┘

ÉTAPE 1: SUPERVISED FINE-TUNING (SFT)
┌──────────────┐
│ Base Model   │ (GPT, Llama pré-entraîné)
│ (Pretrained) │
└──────┬───────┘
       │ + High-quality demonstrations
       │   (prompt → desired response)
       ▼
┌──────────────┐
│  SFT Model   │ (Suit les instructions basiques)
└──────┬───────┘

ÉTAPE 2: REWARD MODEL TRAINING
       │ + Human preference data
       │   (response_A vs response_B rankings)
       ▼
┌──────────────────────────────────────────┐
│  Reward Model (RM)                       │
│  Input: (prompt, response) → score      │
│  Learns: Human preferences              │
└──────┬───────────────────────────────────┘

ÉTAPE 3: RL OPTIMIZATION (PPO)
       │
       ▼
┌──────────────────────────────────────────┐
│  PPO Training Loop                       │
│  1. Generate response with policy       │
│  2. Score with reward model             │
│  3. Compute PPO loss                    │
│  4. Update policy                       │
│  5. Repeat                              │
└──────┬───────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Final Model  │ (Aligned, helpful, harmless)
│  (RLHF'd)    │
└──────────────┘
```

## 14.2 Étape 1 : Supervised Fine-Tuning (SFT)

### 14.2.1 Création du Dataset SFT

**Format des données:**
```json
[
  {
    "prompt": "What is machine learning?",
    "completion": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and learn patterns from it."
  },
  {
    "prompt": "How do I sort a list in Python?",
    "completion": "You can sort a list in Python using the sorted() function or the .sort() method:\n\n```python\n# Using sorted() - returns new list\nnumbers = [3, 1, 4, 1, 5]\nsorted_numbers = sorted(numbers)\n\n# Using .sort() - sorts in place\nnumbers.sort()\n```"
  }
]
```

**Collecting demonstrations:**
```python
import json
from typing import List, Dict

class SFTDatasetCreator:
    """
    Outil pour créer dataset SFT
    """
    def __init__(self):
        self.examples = []

    def add_example(self, prompt: str, completion: str, metadata: Dict = None):
        """Add single example"""
        example = {
            "prompt": prompt,
            "completion": completion,
        }
        if metadata:
            example["metadata"] = metadata

        self.examples.append(example)

    def add_conversation(self, messages: List[Dict]):
        """
        Add conversation format

        messages: [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
        """
        # Convert to prompt-completion pairs
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                self.add_example(
                    prompt=messages[i]["content"],
                    completion=messages[i+1]["content"]
                )

    def save(self, path: str):
        """Save dataset"""
        with open(path, 'w') as f:
            json.dump(self.examples, f, indent=2)

        print(f"Saved {len(self.examples)} examples to {path}")

    def load(self, path: str):
        """Load dataset"""
        with open(path, 'r') as f:
            self.examples = json.load(f)

        print(f"Loaded {len(self.examples)} examples from {path}")

# Usage
creator = SFTDatasetCreator()

# Add examples
creator.add_example(
    prompt="Explain quantum entanglement simply",
    completion="Quantum entanglement is a phenomenon where two particles become connected in such a way that the state of one instantly influences the other, no matter how far apart they are. Einstein called it 'spooky action at a distance.'"
)

# Add conversation
creator.add_conversation([
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"},
    {"role": "assistant", "content": "Paris has a population of approximately 2.2 million people within the city proper, and about 12 million in the metropolitan area."}
])

creator.save("sft_dataset.json")
```

### 14.2.2 SFT Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

def prepare_sft_dataset(examples, tokenizer, max_length=2048):
    """
    Format dataset pour SFT

    Format: <|user|> {prompt} <|assistant|> {completion}
    """
    formatted_texts = []

    for ex in examples:
        text = f"<|user|>\n{ex['prompt']}\n\n<|assistant|>\n{ex['completion']}<|endoftext|>"
        formatted_texts.append(text)

    # Tokenize
    encodings = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Labels = input_ids (causal LM)
    encodings["labels"] = encodings["input_ids"].clone()

    return Dataset.from_dict(encodings)

def train_sft(
    model_name: str,
    train_dataset,
    output_dir: str,
    num_epochs: int = 3,
):
    """
    Train SFT model
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,  # Lower than pretraining
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        warmup_steps=100,
        lr_scheduler_type="cosine",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model(output_dir)

    return model, tokenizer
```

## 14.3 Étape 2 : Reward Model Training

### 14.3.1 Collection de Préférences Humaines

**Format pairwise comparison:**
```json
[
  {
    "prompt": "Explain relativity theory",
    "response_a": "Relativity theory, developed by Einstein, describes how space and time are interconnected and relative to the observer's motion. It consists of special relativity (1905) dealing with constant velocities, and general relativity (1915) incorporating gravity.",
    "response_b": "Relativity is Einstein's theory. It's complicated physics stuff about space and time.",
    "preferred": "a",
    "reason": "More informative, accurate, and detailed"
  }
]
```

**Collecting preferences:**
```python
class PreferenceCollector:
    """
    Collect human preference data
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.preferences = []

    def generate_pair(self, prompt: str, temperature_a=0.7, temperature_b=1.2):
        """
        Generate two different responses pour comparison
        """
        # Generate response A (lower temperature)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output_a = self.model.generate(
            **inputs,
            max_length=256,
            temperature=temperature_a,
            do_sample=True,
        )
        response_a = self.tokenizer.decode(output_a[0], skip_special_tokens=True)

        # Generate response B (higher temperature)
        output_b = self.model.generate(
            **inputs,
            max_length=256,
            temperature=temperature_b,
            do_sample=True,
        )
        response_b = self.tokenizer.decode(output_b[0], skip_special_tokens=True)

        return response_a, response_b

    def collect_preference(self, prompt: str, response_a: str, response_b: str):
        """
        Present pair to human and collect preference
        """
        print(f"\nPrompt: {prompt}\n")
        print(f"Response A:\n{response_a}\n")
        print(f"Response B:\n{response_b}\n")

        while True:
            choice = input("Which response is better? (a/b/tie): ").lower()
            if choice in ['a', 'b', 'tie']:
                break

        reason = input("Why? (optional): ")

        preference = {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "preferred": choice,
            "reason": reason,
        }

        self.preferences.append(preference)

        return preference

    def save_preferences(self, path: str):
        """Save collected preferences"""
        with open(path, 'w') as f:
            json.dump(self.preferences, f, indent=2)

        print(f"Saved {len(self.preferences)} preferences to {path}")
```

### 14.3.2 Reward Model Architecture

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Reward Model pour RLHF

    Input: (prompt, response) text
    Output: scalar reward score
    """
    def __init__(self, base_model, config):
        super().__init__()

        # Base LLM (frozen ou fine-tunable)
        self.base_model = base_model

        # Reward head (map hidden states → scalar)
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize
        nn.init.zeros_(self.reward_head.weight)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        return_dict=True,
    ):
        """
        Forward pass

        Returns: reward scores [batch_size]
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        # [batch, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Take last token's hidden state
        # [batch, hidden_size]
        if attention_mask is not None:
            # Find last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                sequence_lengths,
            ]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Compute reward
        # [batch, 1] → [batch]
        rewards = self.reward_head(last_hidden).squeeze(-1)

        if not return_dict:
            return (rewards,)

        return {
            "rewards": rewards,
            "hidden_states": hidden_states,
        }
```

### 14.3.3 Reward Model Training

```python
class RewardModelTrainer:
    """
    Train reward model on preference data
    """
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

    def compute_loss(self, batch):
        """
        Compute pairwise ranking loss (Bradley-Terry model)

        L = -log(σ(r_chosen - r_rejected))

        où σ est sigmoid
        """
        # Tokenize chosen and rejected
        chosen_inputs = self.tokenizer(
            batch["chosen"],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.model.device)

        rejected_inputs = self.tokenizer(
            batch["rejected"],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.model.device)

        # Get rewards
        chosen_rewards = self.model(**chosen_inputs)["rewards"]
        rejected_rewards = self.model(**rejected_inputs)["rewards"]

        # Compute loss
        # log(sigmoid(r_chosen - r_rejected))
        loss = -torch.nn.functional.logsigmoid(
            chosen_rewards - rejected_rewards
        ).mean()

        # Accuracy (pour monitoring)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, accuracy

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        loss, accuracy = self.compute_loss(batch)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def train(self, train_loader, num_epochs=1):
        """Training loop"""
        for epoch in range(num_epochs):
            total_loss = 0
            total_accuracy = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                metrics = self.train_step(batch)

                total_loss += metrics["loss"]
                total_accuracy += metrics["accuracy"]

            avg_loss = total_loss / len(train_loader)
            avg_accuracy = total_accuracy / len(train_loader)

            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

# Prepare preference dataset
def prepare_preference_dataset(preferences):
    """
    Convert preferences to training format
    """
    data = {
        "chosen": [],
        "rejected": [],
    }

    for pref in preferences:
        prompt = pref["prompt"]

        if pref["preferred"] == "a":
            chosen = f"{prompt}\n\n{pref['response_a']}"
            rejected = f"{prompt}\n\n{pref['response_b']}"
        else:
            chosen = f"{prompt}\n\n{pref['response_b']}"
            rejected = f"{prompt}\n\n{pref['response_a']}"

        data["chosen"].append(chosen)
        data["rejected"].append(rejected)

    return Dataset.from_dict(data)
```

## 14.4 Étape 3 : PPO Training

### 14.4.1 PPO Algorithm pour LLMs

**Proximal Policy Optimization** limite les changements de policy pour stabilité.

**Objectif PPO:**
```
L_PPO(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

où:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
- Â_t : advantage estimate
- ε : clip range (typiquement 0.2)
```

**Pour LLMs:**
```
L_RLHF(θ) = E[r_RM(x,y) - β·D_KL(π_θ || π_ref)]

où:
- r_RM : reward model score
- β : KL penalty coefficient
- π_ref : reference policy (SFT model frozen)
```

### 14.4.2 Implémentation PPO

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

class RLHFTrainer:
    """
    RLHF training avec PPO
    """
    def __init__(
        self,
        policy_model,  # SFT model
        reward_model,  # Trained reward model
        ref_model,     # Reference model (frozen SFT)
        tokenizer,
        config,
    ):
        # Wrap policy with value head
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            policy_model
        )

        self.reward_model = reward_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer

        # PPO configuration
        ppo_config = PPOConfig(
            model_name=config.model_name,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            mini_batch_size=config.mini_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=0.1,  # KL divergence target
            ppo_epochs=4,
            max_grad_norm=1.0,
            seed=config.seed,
        )

        # PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    @torch.no_grad()
    def compute_rewards(self, prompts, responses):
        """
        Compute rewards pour generated responses

        Returns: tensor of rewards [batch_size]
        """
        # Format as (prompt, response) pairs
        texts = [f"{p}\n\n{r}" for p, r in zip(prompts, responses)]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.reward_model.device)

        # Get rewards
        outputs = self.reward_model(**inputs)
        rewards = outputs["rewards"]

        return rewards

    def train_step(self, batch):
        """
        Single PPO training step
        """
        prompts = batch["prompt"]

        # Generate responses with current policy
        prompt_tensors = [
            self.tokenizer.encode(p, return_tensors="pt")[0]
            for p in prompts
        ]

        response_tensors = self.ppo_trainer.generate(
            prompt_tensors,
            max_length=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # Decode responses
        responses = [
            self.tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]

        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # PPO step
        stats = self.ppo_trainer.step(
            prompt_tensors,
            response_tensors,
            rewards,
        )

        return stats

    def train(self, prompts_dataset, num_steps=10000):
        """
        Main training loop
        """
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            prompts_dataset,
            batch_size=self.ppo_trainer.config.batch_size,
            shuffle=True,
        )

        step = 0

        for epoch in range(100):  # Large number
            for batch in dataloader:
                # Training step
                stats = self.train_step(batch)

                step += 1

                # Logging
                if step % 10 == 0:
                    print(f"Step {step}: {stats}")

                # Save checkpoint
                if step % 1000 == 0:
                    self.ppo_trainer.save_pretrained(f"checkpoint-{step}")

                if step >= num_steps:
                    print("Training complete!")
                    return

# Usage
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
sft_model = AutoModelForCausalLM.from_pretrained("./sft_model")
reward_model = RewardModel.from_pretrained("./reward_model")
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")  # Frozen copy
tokenizer = AutoTokenizer.from_pretrained("./sft_model")

# Freeze reference model
for param in ref_model.parameters():
    param.requires_grad = False

# Create trainer
trainer = RLHFTrainer(
    policy_model=sft_model,
    reward_model=reward_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=training_config,
)

# Train
trainer.train(prompts_dataset, num_steps=10000)
```

## 14.5 Alternatives à RLHF

### 14.5.1 DPO (Direct Preference Optimization)

DPO simplifie RLHF en éliminant le reward model et PPO.

**Loss function:**
```
L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[
    log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
]

où:
- y_w : winning response
- y_l : losing response
- σ : sigmoid
- β : temperature parameter
```

**Implémentation:**

```python
class DPOTrainer:
    """
    Direct Preference Optimization
    """
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-7,
        )

    def compute_loss(self, batch):
        """
        Compute DPO loss
        """
        # Tokenize chosen and rejected
        chosen_inputs = self.tokenizer(
            batch["chosen"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        rejected_inputs = self.tokenizer(
            batch["rejected"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Get log probs from current policy
        chosen_logps = self.get_log_probs(self.model, chosen_inputs)
        rejected_logps = self.get_log_probs(self.model, rejected_inputs)

        # Get log probs from reference policy
        with torch.no_grad():
            chosen_ref_logps = self.get_log_probs(self.ref_model, chosen_inputs)
            rejected_ref_logps = self.get_log_probs(self.ref_model, rejected_inputs)

        # Compute log ratios
        chosen_logratios = chosen_logps - chosen_ref_logps
        rejected_logratios = rejected_logps - rejected_ref_logps

        # DPO loss
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Metrics
        accuracy = (chosen_logratios > rejected_logratios).float().mean()

        return loss, accuracy

    def get_log_probs(self, model, inputs):
        """
        Get log probabilities for responses
        """
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute log probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        labels = inputs["input_ids"][:, 1:]  # Shift
        log_probs = log_probs[:, :-1, :]    # Align

        # Gather
        gathered_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # Average over sequence
        return gathered_log_probs.mean(dim=1)

    def train(self, train_loader, num_epochs=1):
        """Training loop"""
        for epoch in range(num_epochs):
            for batch in tqdm(train_loader):
                loss, accuracy = self.compute_loss(batch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Log
                print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
```

### 14.5.2 RLAIF (RL from AI Feedback)

Utilise un LLM pour générer les préférences au lieu d'humains.

```python
class RLAIFPreferenceGenerator:
    """
    Generate preferences using AI instead of humans
    """
    def __init__(self, judge_model, judge_tokenizer):
        self.judge = judge_model
        self.tokenizer = judge_tokenizer

    def generate_preference(self, prompt, response_a, response_b):
        """
        Use AI to judge which response is better
        """
        judge_prompt = f"""You are an expert evaluator. Compare these two responses and determine which is better.

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response is better (A or B)? Consider accuracy, helpfulness, and safety.
Answer with just 'A' or 'B':"""

        inputs = self.tokenizer(judge_prompt, return_tensors="pt").to(self.judge.device)

        output = self.judge.generate(
            **inputs,
            max_length=10,
            temperature=0.0,  # Deterministic
        )

        judgment = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Parse judgment
        if 'A' in judgment:
            return 'a'
        elif 'B' in judgment:
            return 'b'
        else:
            return 'tie'
```

---

*[Le chapitre continue avec Constitutional AI, Iterated DPO, et cas pratiques complets...]*

*[Contenu total du Chapitre 14: ~90-100 pages]*
