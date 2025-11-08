# CHAPITRE 15 : DÉPLOIEMENT ET PRODUCTION

> *« In theory, there is no difference between theory and practice. In practice, there is. »*
> — Yogi Berra

---

## Introduction : Du Notebook au Monde Réel

Vous avez entraîné un LLM. Vous avez fine-tuné, optimisé, évalué. Dans votre notebook Jupyter, tout fonctionne parfaitement. Le modèle génère des réponses brillantes. Les métriques sont excellentes.

**Et maintenant ?**

Comment passer de "ça marche sur mon laptop" à "ça sert 10 000 requêtes par seconde en production avec une latence < 200ms et un SLA de 99.9%" ?

C'est tout l'enjeu du **déploiement en production** : transformer un prototype en un système robuste, scalable, observable, et rentable.

Dans ce chapitre, nous couvrirons :
- **Architectures de serving** : API REST, streaming, batching
- **Frameworks d'inférence** : vLLM, Text Generation Inference, TensorRT-LLM
- **Optimisations** : quantization, KV-cache, batching continu
- **Infrastructure** : GPU, Kubernetes, autoscaling
- **Monitoring** : métriques, tracing, alerting
- **Coûts** : calcul, optimisation, pricing

Bienvenue dans le monde réel des LLMs en production.

---

## 1. Architecture d'un Système LLM en Production

### 🎭 Dialogue : Les Défis de la Production

**Alice** : Bob, j'ai fine-tuné GPT-2 pour mon use case. Comment je le mets en production pour que mes utilisateurs puissent y accéder ?

**Bob** : Excellente question ! Déployer un LLM en production, c'est beaucoup plus que "lancer un serveur".

**Alice** : C'est-à-dire ?

**Bob** : Réfléchis aux contraintes :
- **Latence** : Les utilisateurs attendent < 2 secondes, pas 30 secondes
- **Throughput** : Tu dois gérer 100, 1000, peut-être 10 000 requêtes par seconde
- **Coût** : Les GPUs coûtent cher, chaque milliseconde compte
- **Disponibilité** : 99.9% uptime minimum (< 9 heures de downtime par an)
- **Scalabilité** : Pic de trafic le lundi matin ? Il faut scaler automatiquement
- **Observabilité** : Quand ça casse (et ça cassera), il faut savoir pourquoi

**Alice** : Wow, c'est beaucoup ! Par où commencer ?

**Bob** : Commençons par l'architecture de base, puis nous optimiserons.

---

### 1.1 Architecture Simplifiée

```
┌──────────────────────────────────────────────────┐
│                   CLIENT                         │
│  (App mobile, Web, CLI, etc.)                    │
└─────────────────┬────────────────────────────────┘
                  │
                  │ HTTPS
                  ▼
┌──────────────────────────────────────────────────┐
│              LOAD BALANCER                       │
│  (NGINX, AWS ALB, GCP Load Balancer)             │
└─────────────────┬────────────────────────────────┘
                  │
                  │ Distribue les requêtes
                  ▼
┌──────────────────────────────────────────────────┐
│          API GATEWAY / BACKEND                   │
│  (FastAPI, Flask, Express.js)                    │
│  • Authentification                              │
│  • Rate limiting                                 │
│  • Validation des inputs                        │
│  • Logging                                       │
└─────────────────┬────────────────────────────────┘
                  │
                  │ gRPC / HTTP
                  ▼
┌──────────────────────────────────────────────────┐
│         INFERENCE SERVICE                        │
│  (vLLM, TGI, TensorRT-LLM, Custom)               │
│  • KV-Cache optimization                         │
│  • Batching continu                              │
│  • Quantization                                  │
└─────────────────┬────────────────────────────────┘
                  │
                  │ CUDA
                  ▼
┌──────────────────────────────────────────────────┐
│                GPU(s)                            │
│  (A100, H100, L4, T4, etc.)                      │
└──────────────────────────────────────────────────┘
```

**Composants clés** :

1. **Load Balancer** : Distribue le trafic sur plusieurs instances
2. **API Gateway** : Gère l'authentification, le rate limiting, la validation
3. **Inference Service** : Exécute le modèle (le cœur du système)
4. **GPU** : Calcul parallèle pour l'inférence

---

### 📜 Anecdote Historique : Le Lancement de ChatGPT (30 novembre 2022)

**OpenAI, San Francisco** : Le 30 novembre 2022, OpenAI lance ChatGPT en "research preview". L'équipe s'attend à quelques milliers d'utilisateurs.

**5 jours plus tard** : 1 million d'utilisateurs.
**2 mois plus tard** : 100 millions d'utilisateurs actifs (record absolu).

**Le défi** : Scaler l'infrastructure pour supporter cette croissance explosive.

**Solutions mises en place** :
- **Autoscaling agressif** sur Azure (partenariat OpenAI-Microsoft)
- **File d'attente** : "ChatGPT is at capacity right now"
- **Throttling** : Limitation du nombre de messages par heure
- **Geographic distribution** : Serveurs en Amérique du Nord, Europe, Asie
- **Modèles optimisés** : Passage de GPT-3.5 initial à GPT-3.5-turbo (2x plus rapide, 10x moins cher)

**Leçon** : Même avec une infrastructure de classe mondiale, la production réserve des surprises. Il faut **over-engineer** pour la scalabilité.

---

## 2. Frameworks d'Inférence

### 2.1 Comparaison des Frameworks

| Framework | Développeur | Spécialité | Throughput | Latence | Ease of Use |
|-----------|-------------|------------|------------|---------|-------------|
| **vLLM** | UC Berkeley | Batching continu, PagedAttention | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Text Generation Inference (TGI)** | Hugging Face | Intégration HF, streaming | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TensorRT-LLM** | NVIDIA | Performance maximale, FP8 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **llama.cpp** | Georgi Gerganov | CPU inference, quantization | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **FastAPI + Transformers** | Custom | Flexibilité maximale | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

### 2.2 vLLM : Le Standard de Facto

**vLLM** est devenu le framework de référence pour servir des LLMs en production grâce à **PagedAttention** et au **continuous batching**.

#### Installation et Démarrage

```bash
# Installation
pip install vllm

# Lancer le serveur (API compatible OpenAI)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype auto \
    --api-key sk-my-secret-key

# Le serveur démarre sur http://localhost:8000
```

#### Client Python

```python
from openai import OpenAI

# vLLM expose une API compatible OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-my-secret-key"
)

# Requête standard
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

#### Streaming

```python
# Streaming pour une meilleure UX
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

### 2.3 Text Generation Inference (TGI)

**TGI** de Hugging Face offre une intégration parfaite avec l'écosystème HF et un excellent support du streaming.

#### Lancement avec Docker

```bash
# Lancer TGI avec Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --num-shard 1 \
    --max-total-tokens 4096 \
    --max-batch-prefill-tokens 4096
```

#### Client Python

```python
import requests

url = "http://localhost:8080/generate"

headers = {"Content-Type": "application/json"}

data = {
    "inputs": "What is the capital of France?",
    "parameters": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json()["generated_text"])
```

#### Streaming

```python
# Streaming avec TGI
data = {
    "inputs": "Write a story about a robot",
    "parameters": {"max_new_tokens": 500},
    "stream": True
}

with requests.post(url + "_stream", headers=headers, json=data, stream=True) as r:
    for line in r.iter_lines():
        if line:
            import json
            chunk = json.loads(line.decode('utf-8').replace('data:', ''))
            if 'token' in chunk:
                print(chunk['token']['text'], end='', flush=True)
```

---

### 2.4 Service Custom avec FastAPI

Pour un contrôle total, créer un service custom :

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List

app = FastAPI(title="Custom LLM API")

# Charger le modèle au démarrage
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str):
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully")

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    model_manager.load_model("meta-llama/Llama-2-7b-chat-hf")


# Modèles de requête/réponse
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class GenerationResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    latency_ms: float


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Génère du texte à partir d'un prompt."""
    import time

    start_time = time.time()

    try:
        # Tokenization
        inputs = model_manager.tokenizer(
            request.prompt,
            return_tensors="pt"
        ).to(model_manager.device)

        # Génération
        with torch.no_grad():
            outputs = model_manager.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=model_manager.tokenizer.eos_token_id
            )

        # Décodage
        generated_text = model_manager.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        latency = (time.time() - start_time) * 1000  # ms

        return GenerationResponse(
            generated_text=generated_text,
            tokens_generated=len(outputs[0]) - inputs['input_ids'].shape[1],
            latency_ms=latency
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": model_manager.device
    }


@app.get("/metrics")
async def metrics():
    """Métriques du service."""
    import torch

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
    else:
        gpu_memory = 0
        gpu_memory_max = 0

    return {
        "gpu_memory_allocated_gb": gpu_memory,
        "gpu_memory_max_gb": gpu_memory_max,
        "device": model_manager.device
    }


# Lancer avec : uvicorn app:app --host 0.0.0.0 --port 8000
```

**Utilisation** :

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Once upon a time in a land far away,",
        "max_tokens": 150,
        "temperature": 0.8
    }
)

result = response.json()
print(f"Generated text: {result['generated_text']}")
print(f"Latency: {result['latency_ms']:.2f}ms")
print(f"Tokens: {result['tokens_generated']}")
```

---

## 3. Optimisations d'Inférence

### 🎭 Dialogue : Pourquoi Mon Modèle Est Si Lent ?

**Alice** : Bob, mon LLM en production prend 5 secondes par requête. C'est beaucoup trop lent ! Pourquoi ?

**Bob** : Plusieurs raisons possibles. Regarde ce qui se passe pendant l'inférence :

**Bob** : 1. **Loading du modèle** : Si tu recharges le modèle à chaque requête, ça peut prendre des secondes.

**Alice** : Ah oui, je le charge en mémoire une seule fois au démarrage.

**Bob** : Bien. 2. **Tokenization** : C'est généralement rapide, mais vérifie quand même.

**Bob** : 3. **Forward passes** : C'est là que ça peut être lent. Pour générer 100 tokens, tu fais 100 forward passes !

**Alice** : Attends, un forward pass par token ?

**Bob** : Oui ! Les LLMs sont **autorégressifs** : ils génèrent un token, puis utilisent ce token comme input pour générer le suivant, etc.

**Alice** : Je vois... Et comment accélérer ?

**Bob** : Plusieurs techniques :
- **KV-Cache** : éviter de recalculer l'attention pour les tokens déjà générés
- **Batching** : traiter plusieurs requêtes en parallèle
- **Quantization** : réduire la précision (FP16, INT8, INT4)
- **Flash Attention** : optimisation de l'attention
- **Compilation** : TorchScript, ONNX, TensorRT

**Alice** : Par où commencer ?

**Bob** : KV-Cache et quantization, ce sont les quick wins.

---

### 3.1 KV-Cache : L'Optimisation Essentielle

**Problème** : Sans KV-cache, on recalcule l'attention pour **tous** les tokens à chaque étape.

```python
# Sans KV-cache (inefficace)
tokens_generated = []

for i in range(max_tokens):
    # À chaque itération, on recalcule l'attention pour TOUS les tokens
    # (prompt + tokens déjà générés)
    output = model(tokens_prompt + tokens_generated)  # ❌ LENT
    next_token = sample(output[-1])
    tokens_generated.append(next_token)
```

**Solution** : **KV-Cache** stocke les clés (K) et valeurs (V) de l'attention pour les tokens déjà traités.

```python
# Avec KV-cache (efficace)
past_key_values = None
tokens_generated = []

for i in range(max_tokens):
    if i == 0:
        # Premier passage : traiter tout le prompt
        input_ids = tokens_prompt
    else:
        # Passages suivants : seulement le dernier token
        input_ids = [tokens_generated[-1]]

    output = model(
        input_ids,
        past_key_values=past_key_values,  # ✅ Réutiliser le cache
        use_cache=True
    )

    past_key_values = output.past_key_values  # Mettre à jour le cache
    next_token = sample(output.logits[-1])
    tokens_generated.append(next_token)
```

**Gain** : 5x-10x plus rapide pour la génération.

---

### 3.2 Quantization : Réduire la Précision

La **quantization** réduit la précision des poids pour économiser mémoire et calcul.

| Précision | Mémoire (7B modèle) | Performance | Qualité |
|-----------|---------------------|-------------|---------|
| **FP32** | 28 GB | Baseline | 100% |
| **FP16** | 14 GB | 1.5-2x faster | ~99.9% |
| **INT8** | 7 GB | 2-3x faster | ~99% |
| **INT4** | 3.5 GB | 3-4x faster | ~95-98% |

#### Quantization avec bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configuration pour quantization INT8
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Charger le modèle en INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Model size in memory: {model.get_memory_footprint() / 1024**3:.2f} GB")
```

#### Quantization INT4 (GPTQ)

```python
# Quantization INT4 avec GPTQ (encore plus agressif)
from transformers import GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",  # Dataset de calibration
    tokenizer=tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

# Un modèle 7B tient maintenant dans ~3.5 GB !
```

---

### 3.3 Batching Continu (Continuous Batching)

**Problème du batching classique** : Attendre que toutes les requêtes du batch se terminent.

```
Batch 1:
Request A: ████████████████████████████ (28 tokens, 2.8s)
Request B: ██████ (6 tokens, 0.6s) ... attente 2.2s ❌
Request C: ████████████ (12 tokens, 1.2s) ... attente 1.6s ❌
```

**Continuous Batching** (vLLM) : Ajouter/retirer des requêtes du batch dynamiquement.

```
Request A: ████████████████████████████ (28 tokens)
Request B: ██████ → terminé, remplacé par Request D immédiatement ✅
Request C: ████████████ → terminé, remplacé par Request E ✅
Request D: ██████████████████
Request E: ████████
```

**Résultat** : Throughput amélioré de 2-3x.

**vLLM gère ça automatiquement** — c'est pourquoi il est si performant !

---

### 3.4 Compilation et Optimisation

#### TorchScript

```python
# Compiler le modèle avec TorchScript
model_scripted = torch.jit.script(model)
model_scripted.save("model_scripted.pt")

# Charger le modèle compilé
model_loaded = torch.jit.load("model_scripted.pt")

# Généralement 10-20% plus rapide
```

#### torch.compile (PyTorch 2.0+)

```python
# PyTorch 2.0 : compilation automatique
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
model = torch.compile(model)  # ✨ Magic

# Premier appel : lent (compilation)
# Appels suivants : 30-50% plus rapides !
```

---

## 4. Infrastructure et Scaling

### 4.1 Choix du GPU

| GPU | VRAM | FP16 Throughput | Prix/h (cloud) | Use Case |
|-----|------|-----------------|----------------|----------|
| **T4** | 16 GB | ~6 TFLOPS | $0.35 | Petits modèles (< 7B) |
| **L4** | 24 GB | ~60 TFLOPS | $0.70 | 7B-13B modèles |
| **A10G** | 24 GB | ~35 TFLOPS | $1.00 | 7B-13B modèles |
| **A100 40GB** | 40 GB | ~312 TFLOPS | $3.00 | 13B-30B modèles |
| **A100 80GB** | 80 GB | ~312 TFLOPS | $4.50 | 30B-70B modèles |
| **H100** | 80 GB | ~1000 TFLOPS | $8.00+ | 70B+ modèles |

**Règle empirique** : Vous avez besoin de ~2x la taille du modèle en VRAM (pour FP16 + KV-cache + overhead).

**Exemple** :
- **LLaMA-2 7B** en FP16 : ~14 GB → T4/L4 suffisent
- **LLaMA-2 13B** en FP16 : ~26 GB → A100 40GB ou L4 avec quantization
- **LLaMA-2 70B** en FP16 : ~140 GB → A100 80GB x2 ou H100

---

### 4.2 Déploiement Kubernetes

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
  namespace: ml-services
spec:
  replicas: 3  # 3 instances pour haute disponibilité
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-2-7b-chat-hf
          - --dtype
          - float16
          - --max-model-len
          - "4096"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        ports:
        - containerPort: 8000
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: llm-inference-service
  namespace: ml-services
spec:
  selector:
    app: llm-inference
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
  namespace: ml-services
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

### 4.3 Autoscaling Basé sur la Queue

Pour des coûts optimaux, utilisez un système de queue avec autoscaling :

```python
# Architecture avec Celery + Redis

from celery import Celery
import redis

# Configuration Celery
app = Celery('llm_tasks', broker='redis://localhost:6379/0')

# Task de génération
@app.task(bind=True)
def generate_text(self, prompt: str, max_tokens: int = 100):
    """Tâche de génération de texte."""
    import time
    start = time.time()

    # Appel au modèle
    result = model.generate(prompt, max_tokens=max_tokens)

    duration = time.time() - start

    return {
        "generated_text": result,
        "duration": duration,
        "task_id": self.request.id
    }


# API Frontend
from fastapi import FastAPI, BackgroundTasks

api = FastAPI()

@api.post("/generate_async")
async def generate_async(prompt: str):
    """Enqueue une tâche de génération."""
    task = generate_text.delay(prompt)

    return {
        "task_id": task.id,
        "status": "queued"
    }

@api.get("/result/{task_id}")
async def get_result(task_id: str):
    """Récupère le résultat d'une tâche."""
    task = generate_text.AsyncResult(task_id)

    if task.ready():
        return {
            "status": "completed",
            "result": task.result
        }
    else:
        return {
            "status": "processing"
        }
```

**Autoscaling** : Scaler les workers Celery en fonction de la longueur de la queue.

```bash
# Kubernetes HPA basé sur la métrique custom (queue length)
kubectl autoscale deployment llm-workers \
    --cpu-percent=50 \
    --min=2 \
    --max=20 \
    --custom-metric queue-length:10
```

---

## 5. Monitoring et Observabilité

### 🎭 Dialogue : Pourquoi Le Monitoring Est Crucial

**Alice** : Bob, mon service LLM est en production depuis 2 semaines. Tout a l'air de fonctionner. Pourquoi tu insistes autant sur le monitoring ?

**Bob** : Parce que "ça a l'air de marcher" n'est pas suffisant en production. Tu as besoin de savoir :
- **Performance** : Quelle est la latence P50, P95, P99 ?
- **Throughput** : Combien de requêtes par seconde ?
- **Erreurs** : Quel est le taux d'erreur ? Quels types d'erreurs ?
- **Coûts** : Combien coûte chaque requête en GPU time ?
- **Qualité** : Les réponses sont-elles bonnes ?

**Alice** : D'accord, mais comment mesurer tout ça ?

**Bob** : Plusieurs niveaux :
1. **Métriques système** : CPU, GPU, mémoire
2. **Métriques applicatives** : latence, throughput, erreurs
3. **Métriques métier** : coût par requête, satisfaction utilisateur
4. **Tracing** : suivre une requête de bout en bout

---

### 5.1 Métriques avec Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from fastapi import FastAPI
import time

app = FastAPI()

# Métriques Prometheus
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

TOKENS_GENERATED = Counter(
    'llm_tokens_generated_total',
    'Total number of tokens generated'
)

GPU_MEMORY = Gauge(
    'llm_gpu_memory_allocated_bytes',
    'GPU memory allocated in bytes'
)

QUEUE_SIZE = Gauge(
    'llm_queue_size',
    'Number of requests in queue'
)


@app.middleware("http")
async def monitor_requests(request, call_next):
    """Middleware pour monitorer toutes les requêtes."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # Enregistrer les métriques
    REQUEST_COUNT.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response


@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 100):
    """Endpoint de génération avec monitoring."""
    # Génération
    result = model.generate(prompt, max_tokens=max_tokens)

    # Métriques
    TOKENS_GENERATED.inc(len(result.tokens))

    if torch.cuda.is_available():
        GPU_MEMORY.set(torch.cuda.memory_allocated())

    return {"generated_text": result.text}


# Exposer les métriques Prometheus sur le port 9090
start_http_server(9090)
```

**Grafana Dashboard** : Visualiser les métriques

```promql
# Latence P95
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))

# Throughput (requêtes/seconde)
rate(llm_requests_total[1m])

# Taux d'erreur
rate(llm_requests_total{status=~"5.."}[5m]) / rate(llm_requests_total[5m])

# Tokens générés par seconde
rate(llm_tokens_generated_total[1m])
```

---

### 5.2 Tracing avec OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configuration OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)


@app.post("/generate")
async def generate_with_tracing(prompt: str):
    """Génération avec tracing distribué."""
    with tracer.start_as_current_span("llm_generation") as span:
        span.set_attribute("prompt_length", len(prompt))

        # Tokenization
        with tracer.start_as_current_span("tokenization"):
            tokens = tokenizer(prompt)
            span.set_attribute("num_tokens", len(tokens['input_ids'][0]))

        # Inference
        with tracer.start_as_current_span("model_inference"):
            output = model.generate(**tokens, max_new_tokens=100)

        # Decoding
        with tracer.start_as_current_span("decoding"):
            result = tokenizer.decode(output[0])

        span.set_attribute("output_length", len(result))

        return {"generated_text": result}
```

**Visualisation dans Jaeger** : Voir exactement où le temps est passé (tokenization 5ms, inference 1.2s, decoding 8ms).

---

### 5.3 Logging Structuré

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Logger structuré pour faciliter l'analyse."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_request(self, request_id: str, prompt: str, user_id: str):
        """Log une requête entrante."""
        self.logger.info(json.dumps({
            "event": "request_received",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "user_id": user_id,
            "prompt_length": len(prompt)
        }))

    def log_generation(self, request_id: str, tokens: int, latency: float):
        """Log une génération terminée."""
        self.logger.info(json.dumps({
            "event": "generation_completed",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "tokens_generated": tokens,
            "latency_ms": latency * 1000,
            "tokens_per_second": tokens / latency if latency > 0 else 0
        }))

    def log_error(self, request_id: str, error: str):
        """Log une erreur."""
        self.logger.error(json.dumps({
            "event": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "error_message": str(error)
        }))


# Utilisation
logger = StructuredLogger("llm_service")

@app.post("/generate")
async def generate(prompt: str, user_id: str):
    request_id = str(uuid.uuid4())
    logger.log_request(request_id, prompt, user_id)

    start = time.time()
    try:
        result = model.generate(prompt)
        latency = time.time() - start

        logger.log_generation(request_id, len(result.tokens), latency)

        return {"text": result.text}

    except Exception as e:
        logger.log_error(request_id, str(e))
        raise
```

**Analyse avec ELK Stack** : Chercher, filtrer, agréger les logs JSON.

---

## 6. Gestion des Coûts

### 6.1 Calculer le Coût par Requête

```python
class CostCalculator:
    """Calcule le coût de chaque requête."""

    def __init__(self, gpu_cost_per_hour: float):
        """
        Args:
            gpu_cost_per_hour: Coût du GPU en $/heure (ex: A100 = $3.00/h)
        """
        self.gpu_cost_per_second = gpu_cost_per_hour / 3600

    def calculate_cost(self, latency_seconds: float, gpu_utilization: float = 1.0):
        """
        Calcule le coût d'une requête.

        Args:
            latency_seconds: Temps de génération en secondes
            gpu_utilization: Utilisation du GPU (0.0 à 1.0)

        Returns:
            Coût en dollars
        """
        cost = latency_seconds * self.gpu_cost_per_second * gpu_utilization
        return cost


# Exemple
calculator = CostCalculator(gpu_cost_per_hour=3.00)  # A100

# Requête qui prend 2 secondes
cost = calculator.calculate_cost(latency_seconds=2.0)
print(f"Coût par requête : ${cost:.6f}")  # $0.001667

# Si on sert 10 000 requêtes/jour
daily_cost = cost * 10000
print(f"Coût quotidien : ${daily_cost:.2f}")  # $16.67
```

---

### 6.2 Stratégies d'Optimisation des Coûts

**1. Batching Agressif**
```python
# Au lieu de traiter les requêtes une par une
# → Accumuler pendant 50ms et traiter en batch

import asyncio
from collections import deque

class BatchingService:
    def __init__(self, max_batch_size=32, max_wait_ms=50):
        self.queue = deque()
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

    async def add_request(self, prompt: str):
        """Ajoute une requête à la queue."""
        future = asyncio.Future()
        self.queue.append((prompt, future))

        # Si le batch est plein, traiter immédiatement
        if len(self.queue) >= self.max_batch_size:
            await self.process_batch()

        return await future

    async def process_batch(self):
        """Traite un batch de requêtes."""
        if not self.queue:
            return

        batch = []
        futures = []

        while self.queue and len(batch) < self.max_batch_size:
            prompt, future = self.queue.popleft()
            batch.append(prompt)
            futures.append(future)

        # Inférence en batch (beaucoup plus efficace !)
        results = model.generate_batch(batch)

        # Retourner les résultats
        for future, result in zip(futures, results):
            future.set_result(result)

# Réduction du coût : 3-5x grâce au batching
```

**2. Utiliser des Modèles Plus Petits**
```python
# Cascade de modèles : petit modèle d'abord, grand modèle si nécessaire

async def smart_generate(prompt: str):
    """Utilise un petit modèle, puis un grand si besoin."""

    # Essayer avec un petit modèle (GPT-3.5, Llama-2 7B)
    small_result = await small_model.generate(prompt)

    # Vérifier la qualité (heuristique simple)
    confidence = calculate_confidence(small_result)

    if confidence > 0.8:
        # Le petit modèle est confiant → utiliser sa réponse
        return small_result  # Coût : 10x moins cher

    else:
        # Faible confiance → utiliser le grand modèle
        large_result = await large_model.generate(prompt)
        return large_result

# 70% des requêtes traitées par le petit modèle
# → Réduction de coût globale : ~7x
```

**3. Caching Intelligent**
```python
import hashlib
from functools import lru_cache

class SemanticCache:
    """Cache basé sur la similarité sémantique."""

    def __init__(self, similarity_threshold=0.95):
        self.cache = {}
        self.embeddings_cache = {}
        self.threshold = similarity_threshold

    def get(self, prompt: str):
        """Cherche dans le cache."""
        # Calculer l'embedding du prompt
        emb = get_embedding(prompt)

        # Chercher un prompt similaire
        for cached_prompt, cached_emb in self.embeddings_cache.items():
            similarity = cosine_similarity(emb, cached_emb)

            if similarity > self.threshold:
                # Cache hit !
                return self.cache[cached_prompt]

        return None

    def set(self, prompt: str, result: str):
        """Ajoute au cache."""
        emb = get_embedding(prompt)
        self.embeddings_cache[prompt] = emb
        self.cache[prompt] = result

# 30-40% de cache hit rate sur des queries similaires
# → Coût réduit de 30-40%
```

---

## 🧠 Quiz Interactif

### Question 1
**Qu'est-ce que le KV-Cache ?**

A) Un cache pour stocker les résultats des requêtes
B) Un cache qui stocke les clés et valeurs de l'attention pour éviter les recalculs
C) Un cache de tokenization
D) Un système de mise en cache des embeddings

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Le **KV-Cache** stocke les matrices de **clés (K)** et **valeurs (V)** de l'attention pour les tokens déjà traités.

Sans KV-cache, à chaque génération de token, le modèle doit recalculer l'attention pour **tous** les tokens (prompt + tokens générés).

Avec KV-cache, on réutilise les K et V déjà calculés, et on calcule seulement pour le nouveau token.

**Gain** : 5-10x plus rapide pour la génération autoregressive.
</details>

---

### Question 2
**Quel est l'avantage principal du continuous batching (vLLM) par rapport au batching classique ?**

A) Utilise moins de mémoire
B) Permet d'ajouter/retirer des requêtes du batch dynamiquement
C) Plus simple à implémenter
D) Fonctionne seulement avec les GPUs NVIDIA

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Le **batching classique** attend que toutes les requêtes du batch se terminent avant de traiter le batch suivant. Si une requête génère 100 tokens et une autre 10 tokens, les 9 slots attendent inutilement.

Le **continuous batching** (PagedAttention dans vLLM) permet de :
- Retirer les requêtes terminées du batch
- Ajouter de nouvelles requêtes immédiatement
- Maximiser l'utilisation du GPU

**Résultat** : Throughput amélioré de 2-3x sans augmenter la latence.
</details>

---

### Question 3
**Quelle quantization offre le meilleur rapport qualité/coût pour la plupart des cas d'usage ?**

A) FP32
B) FP16
C) INT8
D) INT4

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : C (INT8)**

**INT8** offre généralement le meilleur compromis :
- **Mémoire** : Réduction de 4x vs FP32, 2x vs FP16
- **Performance** : 2-3x plus rapide que FP16
- **Qualité** : ~99% de la qualité originale (perte minime)
- **Compatibilité** : Supporté par la plupart des frameworks

**FP16** : Bon si vous avez assez de VRAM et voulez la qualité maximale
**INT4** : Utile pour les très grands modèles (70B+) mais qualité dégradée (~95-98%)

**Best practice** : Commencer avec INT8, downgrade vers INT4 seulement si nécessaire.
</details>

---

### Question 4
**Pourquoi le monitoring est-il crucial en production ?**

A) Pour impressionner les managers avec des dashboards
B) Pour détecter les problèmes avant qu'ils n'impactent les utilisateurs
C) C'est obligatoire par la loi
D) Pour réduire les coûts de 90%

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

Le monitoring permet de :
1. **Détecter les problèmes proactivement** : Latence qui augmente, taux d'erreur qui monte
2. **Diagnostiquer rapidement** : Où est le bottleneck ? GPU saturé ? Queue qui déborde ?
3. **Optimiser les coûts** : Identifier les requêtes coûteuses, optimiser les patterns
4. **Garantir les SLAs** : P99 latency < 2s, 99.9% uptime
5. **Comprendre l'usage** : Quels prompts ? Quelle charge ? Quels patterns ?

**Sans monitoring** : Vous êtes aveugle. Vous découvrez les problèmes quand les utilisateurs se plaignent.

**Avec monitoring** : Vous voyez les problèmes arriver et pouvez agir avant l'impact.
</details>

---

### Question 5
**Quelle stratégie permet de réduire les coûts de 70% en moyenne ?**

A) Utiliser des GPUs moins chers
B) Cascade de modèles (petit modèle → grand modèle si nécessaire)
C) Réduire la qualité des réponses
D) Limiter le nombre de tokens générés

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

La **cascade de modèles** utilise un modèle petit et rapide (GPT-3.5, Llama-2 7B) pour la majorité des requêtes, et ne fait appel au grand modèle (GPT-4, Llama-2 70B) que pour les cas complexes.

**Exemple** :
- 70% des requêtes : GPT-3.5 (10x moins cher)
- 30% des requêtes : GPT-4

**Coût moyen** : 0.7 × coût_GPT35 + 0.3 × coût_GPT4
Si GPT-4 coûte 10x plus cher : 0.7 × 1 + 0.3 × 10 = 3.7
**Réduction** : ~63% vs utiliser GPT-4 pour tout

**Avec batching + cache** : Réduction totale > 80%
</details>

---

### Question 6
**Quelle est la règle empirique pour la VRAM nécessaire ?**

A) Taille du modèle × 1
B) Taille du modèle × 2
C) Taille du modèle × 4
D) Ça dépend uniquement du batch size

<details>
<summary>👉 Voir la réponse</summary>

**Réponse : B**

**Règle empirique** : VRAM nécessaire ≈ **2× taille du modèle** (en FP16)

**Détail** :
- **Poids du modèle** : Taille nominale (ex: 7B × 2 bytes = 14 GB)
- **KV-Cache** : ~20-30% de la taille du modèle (dépend de la longueur de contexte)
- **Activations** : ~10-20% pendant le forward pass
- **Overhead** : ~10% (CUDA, PyTorch, etc.)

**Total** : ~1.4-2× la taille du modèle

**Exemples** :
- LLaMA-2 7B (FP16): ~14 GB → **besoin de 24-32 GB** (L4, A10G, A100 40GB)
- LLaMA-2 70B (FP16): ~140 GB → **besoin de 280 GB** (4× A100 80GB ou 2× H100)

**Avec quantization INT8** : Divisez par 2
**Avec quantization INT4** : Divisez par 4
</details>

---

## 💻 Exercices Pratiques

### Exercice 1 : Déployer un Service avec vLLM

**Objectif** : Déployer LLaMA-2 7B avec vLLM et mesurer les performances.

**Consignes** :
1. Installer vLLM
2. Lancer le serveur
3. Créer un client de test qui mesure latence et throughput
4. Comparer performances avec/sans batching

<details>
<summary>👉 Voir la solution</summary>

```bash
# Installation
pip install vllm

# Lancer le serveur
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --dtype float16 \
    --max-model-len 4096
```

```python
import time
import asyncio
from openai import AsyncOpenAI
import statistics

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

async def test_single_request():
    """Test une seule requête."""
    start = time.time()

    response = await client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": "What is AI?"}],
        max_tokens=100
    )

    latency = time.time() - start
    return latency

async def test_concurrent_requests(num_requests=10):
    """Test plusieurs requêtes en parallèle."""
    start = time.time()

    tasks = [test_single_request() for _ in range(num_requests)]
    latencies = await asyncio.gather(*tasks)

    total_time = time.time() - start
    throughput = num_requests / total_time

    return {
        "total_time": total_time,
        "throughput": throughput,
        "mean_latency": statistics.mean(latencies),
        "p95_latency": statistics.quantiles(latencies, n=20)[18],  # P95
        "p99_latency": statistics.quantiles(latencies, n=100)[98]  # P99
    }

# Exécuter les tests
async def main():
    print("Test 1: Single request")
    latency = await test_single_request()
    print(f"Latency: {latency:.3f}s\n")

    print("Test 2: 10 concurrent requests")
    results = await test_concurrent_requests(10)
    for key, value in results.items():
        print(f"{key}: {value:.3f}")

    print("\nTest 3: 50 concurrent requests")
    results = await test_concurrent_requests(50)
    for key, value in results.items():
        print(f"{key}: {value:.3f}")

asyncio.run(main())
```

</details>

---

### Exercice 2 : Implémenter un Cache Sémantique

**Objectif** : Créer un système de cache basé sur la similarité sémantique pour réduire les coûts.

<details>
<summary>👉 Voir la solution dans le code de la section 6.2</summary>

Utilisez la classe `SemanticCache` fournie et mesurez le cache hit rate sur vos données réelles.
</details>

---

## 📚 Résumé du Chapitre

### Points Clés

1. **Architecture** : Load Balancer → API Gateway → Inference Service → GPU

2. **Frameworks** :
   - **vLLM** : Meilleur throughput (continuous batching)
   - **TGI** : Meilleure intégration HF
   - **TensorRT-LLM** : Performance maximale (NVIDIA)

3. **Optimisations** :
   - **KV-Cache** : 5-10x plus rapide
   - **Quantization INT8** : 2x plus rapide, 2x moins de VRAM
   - **Batching continu** : 2-3x meilleur throughput

4. **Infrastructure** :
   - Choix GPU basé sur taille modèle
   - Kubernetes pour orchestration
   - Autoscaling basé sur métriques

5. **Monitoring** :
   - Métriques (Prometheus + Grafana)
   - Tracing (OpenTelemetry + Jaeger)
   - Logging structuré (ELK Stack)

6. **Coûts** :
   - Batching, caching, cascade de modèles
   - Réduction typique : 70-80%

---

## 🚀 Prochaine Étape

Dans le **Chapitre 16 : Sécurité et Éthique**, nous explorerons :
- Sécurisation des LLMs (prompt injection, jailbreaking)
- Filtrage de contenu toxique
- Privacy et données sensibles
- Biais et fairness
- Réglementations (RGPD, AI Act)

**À très bientôt !** 🎉

---

## 📖 Références

### Frameworks
- **vLLM** : https://github.com/vllm-project/vllm
- **Text Generation Inference** : https://github.com/huggingface/text-generation-inference
- **TensorRT-LLM** : https://github.com/NVIDIA/TensorRT-LLM

### Papers
- PagedAttention (vLLM) : https://arxiv.org/abs/2309.06180
- Continuous Batching : Orca paper

### Outils
- **Prometheus** : Monitoring et alerting
- **Grafana** : Visualisation de métriques
- **Jaeger** : Distributed tracing
- **ELK Stack** : Elasticsearch + Logstash + Kibana

---

*Fin du Chapitre 15*
