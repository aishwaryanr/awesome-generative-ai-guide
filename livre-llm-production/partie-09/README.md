# Partie 9 : Inference et optimisation modèle

## Objectifs d'apprentissage

- Maîtriser les stratégies de décodage
- Optimiser latence et throughput (KV cache, batching, spéculation)
- Appliquer compression et adaptation (quantization, LoRA, distillation)
- Déployer avec les meilleurs serving engines (vLLM, TGI, SGLang)

## Prérequis

- Parties 1-8 validées
- Compréhension des trade-offs latence/throughput/mémoire

**Référence** : Park et al. (5.1) - Survey inference engines

---

## 9.1 Stratégies de décodage

### 9.1.1 Greedy decoding

**Principe** : Toujours choisir le token le plus probable.

```python
def greedy_decode(model, input_ids, max_length=50):
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Dernier token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return input_ids
```

**Avantages** : Rapide, déterministe
**Inconvénients** : Répétitif, manque de diversité

### 9.1.2 Sampling avec température

**Principe** : Sampler selon la distribution de probabilité.

```python
def sample_with_temperature(logits, temperature=1.0):
    """Temperature < 1 → plus déterministe, > 1 → plus aléatoire."""
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

**Effet de la température** :
- T = 0 : équivalent à greedy
- T = 1 : distribution originale
- T > 1 : plus de diversité (mais risque de non-sens)
- T < 1 : plus focalisé (mais répétitif)

### 9.1.3 Top-k et Top-p (nucleus) sampling

**Top-k** : Sampler seulement parmi les k tokens les plus probables.

```python
def top_k_sampling(logits, k=50):
    top_k_logits, top_k_indices = torch.topk(logits, k)
    probs = torch.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = top_k_indices.gather(-1, next_token_idx)
    return next_token
```

**Top-p (nucleus)** : Sampler dans le plus petit ensemble dont la probabilité cumulative ≥ p.

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Retirer tokens au-delà du seuil p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_logits[sorted_indices_to_remove] = -float('inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token_idx)
    return next_token
```

### 9.1.4 Beam search

**Principe** : Maintenir les k séquences les plus probables.

```python
def beam_search(model, input_ids, beam_width=5, max_length=50):
    from queue import PriorityQueue

    # Initialiser avec l'input
    beams = PriorityQueue()
    beams.put((0, input_ids, False))  # (score, sequence, finished)

    for _ in range(max_length):
        candidates = PriorityQueue()

        for _ in range(min(beam_width, beams.qsize())):
            score, seq, finished = beams.get()

            if finished:
                candidates.put((score, seq, finished))
                continue

            # Générer prochains tokens
            outputs = model(seq)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                new_score = score + top_k_probs[0, i].item()
                new_seq = torch.cat([seq, top_k_indices[0, i].unsqueeze(0).unsqueeze(0)], dim=1)
                is_finished = (top_k_indices[0, i].item() == tokenizer.eos_token_id)

                candidates.put((new_score, new_seq, is_finished))

        # Garder les beam_width meilleurs
        beams = PriorityQueue()
        for _ in range(beam_width):
            if not candidates.empty():
                beams.put(candidates.get())

    # Retourner la meilleure séquence
    best_score, best_seq, _ = beams.get()
    return best_seq
```

### 9.1.5 Contrôles additionnels

**Répétition penalty** :

```python
def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    """Pénaliser les tokens déjà générés."""
    for token_id in input_ids[0].unique():
        logits[0, token_id] /= penalty
    return logits
```

**Length penalty** : Encourager/décourager les séquences longues.

```python
def length_penalty(score, length, alpha=0.6):
    """Alpha > 0 favorise les séquences plus longues."""
    return score / (length ** alpha)
```

---

## 9.2 Accélération d'inférence

**Référence** : Park et al. (5.1)

### 9.2.1 KV Cache

**Problème** : Recalculer toutes les clés et valeurs à chaque step est coûteux.

**Solution** : Garder en mémoire les KV des tokens précédents.

```python
class TransformerWithKVCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ...

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        if past_key_values is None:
            past_key_values = [None] * self.num_layers

        present_key_values = []

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx]

            # Utiliser le cache si disponible
            hidden_states, new_kv = layer(
                hidden_states,
                past_key_value=layer_past,
                use_cache=use_cache
            )

            if use_cache:
                present_key_values.append(new_kv)

        if use_cache:
            return hidden_states, present_key_values
        return hidden_states
```

**Gain** : Réduction de O(n²) à O(n) en compute par token.

**Coût mémoire** : Croît linéairement avec la longueur de séquence.

### 9.2.2 Batching dynamique et continu

**Continuous batching** (vLLM) :

**Problème** : Les séquences dans un batch finissent à des moments différents.

**Solution** : Ajouter de nouvelles requêtes dès qu'une séquence termine.

```
Batch initial: [seq1, seq2, seq3]
seq2 termine → remplacée par seq4
Batch: [seq1, seq4, seq3]
seq3 termine → remplacée par seq5
Batch: [seq1, seq4, seq5]
...
```

**Implémentation** (conceptuel) :

```python
class ContinuousBatchScheduler:
    def __init__(self, max_batch_size):
        self.max_batch_size = max_batch_size
        self.active_requests = []
        self.pending_requests = queue.Queue()

    def add_request(self, request):
        self.pending_requests.put(request)

    def get_batch(self):
        """Construire le prochain batch."""
        # Retirer les requêtes terminées
        self.active_requests = [r for r in self.active_requests if not r.is_finished()]

        # Ajouter de nouvelles requêtes pour remplir le batch
        while len(self.active_requests) < self.max_batch_size and not self.pending_requests.empty():
            self.active_requests.append(self.pending_requests.get())

        return self.active_requests
```

### 9.2.3 Speculative Decoding

**Principe** : Utiliser un modèle petit et rapide pour proposer des tokens, validés par le grand modèle.

```
1. Draft model génère k tokens rapidement
2. Target model valide en parallèle
3. Accepter les tokens corrects, rejeter les autres
4. Répéter
```

**Implémentation** :

```python
def speculative_decode(draft_model, target_model, input_ids, k=5):
    """k = nombre de tokens spéculés."""

    # Draft: générer k tokens
    draft_tokens = []
    draft_input = input_ids
    for _ in range(k):
        draft_logits = draft_model(draft_input).logits[:, -1, :]
        next_token = torch.argmax(draft_logits, dim=-1).unsqueeze(-1)
        draft_tokens.append(next_token)
        draft_input = torch.cat([draft_input, next_token], dim=1)

    # Validation par le grand modèle (parallèle)
    full_draft = torch.cat(draft_tokens, dim=1)
    target_input = torch.cat([input_ids, full_draft], dim=1)
    target_logits = target_model(target_input).logits

    # Vérifier token par token
    accepted = 0
    for i, draft_token in enumerate(draft_tokens):
        target_next = torch.argmax(target_logits[:, input_ids.size(1) + i - 1, :], dim=-1)
        if target_next == draft_token.squeeze():
            accepted += 1
        else:
            break  # Rejeter le reste

    # Retourner les tokens acceptés
    return torch.cat([input_ids] + draft_tokens[:accepted], dim=1)
```

**Speedup** : 2-3× si le draft model est bon.

---

## 9.3 Compression et adaptation

### 9.3.1 Quantization

**Principe** : Réduire la précision des poids (float16 → int8 → int4).

**Post-training quantization (PTQ)** :

```python
import torch.quantization

# Quantization int8
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Couches à quantifier
    dtype=torch.qint8
)

# Mesurer la taille
def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

print(f"Original: {model_size_mb(model):.2f} MB")
print(f"Quantized: {model_size_mb(model_int8):.2f} MB")
```

**GPTQ (int4 quantization)** :

```python
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    quantization_config=quantization_config
)
```

**Trade-off** : Réduction mémoire ÷2-4, légère perte de qualité.

### 9.3.2 LoRA (Low-Rank Adaptation)

**Principe** : Adapter seulement des matrices low-rank au lieu de tout le modèle.

```
W' = W + AB  où A ∈ ℝ^(d×r), B ∈ ℝ^(r×k), r << d, k
```

**Implémentation** :

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # Rang
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Modules à adapter
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())
# trainable params: 4M / 7000M → 0.05%
```

**Avantages** :
- Fine-tuning 10-100× plus rapide
- Mémoire réduite
- Multiples adaptateurs pour différentes tâches

### 9.3.3 Distillation

**Principe** : Entraîner un modèle petit à imiter un grand modèle.

```python
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    alpha: poids de la distillation vs loss classique
    """
    # Soft targets (distillation)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Hard targets (cross-entropy classique)
    hard_loss = F.cross_entropy(student_logits, labels)

    # Combinaison
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training loop
for batch in dataloader:
    with torch.no_grad():
        teacher_logits = teacher_model(batch['input_ids']).logits

    student_logits = student_model(batch['input_ids']).logits
    loss = distillation_loss(student_logits, teacher_logits, batch['labels'])

    loss.backward()
    optimizer.step()
```

---

## 9.4 Serving engines

**Référence** : Park et al. (5.1)

### 9.4.1 Comparatif des engines

| Engine     | Avantages                              | Inconvénients                    |
|------------|----------------------------------------|----------------------------------|
| **vLLM**   | PagedAttention, continuous batching, très rapide | Moins de modèles supportés       |
| **TGI**    | HuggingFace natif, feature-rich        | Moins optimisé que vLLM          |
| **SGLang** | Optimisé pour structured generation    | Plus récent, moins mature        |
| **Ollama** | Simple, local, CPU/GPU                 | Moins de fonctionnalités avancées|

### 9.4.2 Déploiement avec vLLM

**Installation** :

```bash
pip install vllm
```

**Serveur API** :

```python
from vllm import LLM, SamplingParams

# Initialiser
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Paramètres de génération
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Générer
prompts = ["Tell me about AI", "What is machine learning?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**Serveur HTTP** :

```bash
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000
```

**Appeler l'API** :

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.8
})

print(response.json()["text"])
```

### 9.4.3 Déploiement avec TGI (Text Generation Inference)

**Docker** :

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf
```

**Client Python** :

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="http://localhost:8080")

response = client.text_generation(
    "What is the capital of France?",
    max_new_tokens=50
)
print(response)
```

### 9.4.4 Choix d'un engine

**Critères** :

1. **Latence** : vLLM > SGLang > TGI pour throughput élevé
2. **Compatibilité modèles** : TGI supporte plus de modèles
3. **Features** : TGI a plus de fonctionnalités (streaming, safety, etc.)
4. **Facilité** : Ollama pour usage local simple

**Benchmarks** (référence 5.1) :

Comparer sur votre workload spécifique :
- Latence P50, P95, P99
- Throughput (tokens/sec)
- Coût par 1M tokens

---

## 9.5 Labs pratiques

### Lab 1 : Benchmarker vLLM vs TGI

1. Déployer le même modèle sur vLLM et TGI
2. Mesurer latence et throughput avec différents batch sizes
3. Tracer les courbes de performance

### Lab 2 : Implémenter speculative decoding

1. Choisir un draft model (petit) et target model (grand)
2. Implémenter le pipeline de spéculation
3. Mesurer le speedup réel

### Lab 3 : Quantization et évaluation

1. Quantifier un modèle en int8 et int4
2. Évaluer la perte de qualité (perplexité, benchmarks)
3. Mesurer le gain de vitesse et mémoire

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Stratégies de décodage (greedy, sampling, beam search, top-k/p)
- ✅ Accélération (KV cache, batching continu, spéculation)
- ✅ Compression (quantization, LoRA, distillation)
- ✅ Serving engines (vLLM, TGI, SGLang, Ollama)

**Prochaine étape** : [Partie 10 - Déploiement et LLMOps](../partie-10/README.md)
