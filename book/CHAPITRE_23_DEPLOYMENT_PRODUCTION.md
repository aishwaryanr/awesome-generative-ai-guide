# CHAPITRE 23 : ARCHITECTURE DE SYSTÈMES LLM EN PRODUCTION

## Introduction

Déployer un LLM en production est radicalement différent d'un prototype. Les défis incluent:
- **Latence**: < 500ms pour TTFT (Time To First Token)
- **Throughput**: 100+ requêtes/seconde
- **Coût**: Optimiser $/token
- **Fiabilité**: 99.9%+ uptime
- **Sécurité**: Protection contre injections, DDoS
- **Monitoring**: Tracking performance en temps réel

Ce chapitre couvre l'architecture complète pour déployer des LLMs à grande échelle.

## 23.1 Architecture Haute Niveau

### 23.1.1 Composants d'un Système LLM Production

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION LLM SYSTEM                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Client     │
│  (Web/API)   │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│  LOAD BALANCER (nginx/HAProxy/CloudFlare)                      │
│  - Rate limiting                                               │
│  - DDoS protection                                             │
│  - SSL termination                                             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  API GATEWAY (Kong/AWS API Gateway)                            │
│  - Authentication (JWT, API keys)                              │
│  - Request validation                                          │
│  - Rate limiting per user                                      │
│  - Caching (Redis)                                             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER (FastAPI/Flask)                             │
│  - Request preprocessing                                       │
│  - Prompt engineering                                          │
│  - Context assembly (RAG)                                      │
│  - Post-processing                                             │
└────────┬───────────────────────────────────────────────────────┘
         │
         ├────────────┐
         ▼            ▼
    ┌────────┐  ┌────────────┐
    │ Vector │  │  Cache     │
    │   DB   │  │  (Redis)   │
    │(Qdrant)│  └────────────┘
    └────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  INFERENCE LAYER (vLLM/TensorRT-LLM)                          │
│  - Model serving                                               │
│  - Batch processing                                            │
│  - GPU optimization                                            │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│  OBSERVABILITY (Prometheus/Grafana/DataDog)                   │
│  - Metrics collection                                          │
│  - Logging (ELK stack)                                         │
│  - Tracing (Jaeger)                                            │
│  - Alerting                                                    │
└────────────────────────────────────────────────────────────────┘
```

### 23.1.2 Implémentation API Layer avec FastAPI

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
from datetime import datetime
import hashlib

app = FastAPI(title="LLM API", version="1.0.0")

# ============== Models ==============

class GenerationRequest(BaseModel):
    """Request model pour génération"""
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

class GenerationResponse(BaseModel):
    """Response model"""
    id: str
    text: str
    model: str
    usage: dict
    created: int

# ============== Dependencies ==============

async def verify_api_key(x_api_key: str = Header(...)):
    """Vérifie l'API key"""
    # Dans production, vérifier contre DB
    valid_keys = {"test-key-123", "prod-key-456"}

    if x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return x_api_key

# ============== Rate Limiting ==============

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============== Caching ==============

import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_cache_key(prompt: str, params: dict) -> str:
    """Generate cache key"""
    cache_string = f"{prompt}:{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(cache_string.encode()).hexdigest()

async def get_from_cache(key: str) -> Optional[str]:
    """Get from cache"""
    try:
        return redis_client.get(key)
    except:
        return None

async def set_cache(key: str, value: str, ttl: int = 3600):
    """Set cache with TTL"""
    try:
        redis_client.setex(key, ttl, value)
    except:
        pass

# ============== Model Inference ==============

from vllm import LLM, SamplingParams

# Initialize model (done once at startup)
llm_model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global llm_model
    llm_model = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=1,  # Number of GPUs
        dtype="half",  # FP16
    )
    print("Model loaded successfully")

async def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]] = None
) -> str:
    """Generate text using vLLM"""
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
    )

    # Generate (async with vLLM)
    outputs = llm_model.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text

# ============== Endpoints ==============

@app.post("/v1/generate", response_model=GenerationResponse)
@limiter.limit("100/minute")  # Rate limit
async def generate(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Generate text endpoint

    - Caching enabled
    - Rate limited: 100 requests/minute
    - Requires API key
    """
    # Generate cache key
    cache_key = get_cache_key(
        request.prompt,
        {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
    )

    # Check cache
    cached_response = await get_from_cache(cache_key)
    if cached_response:
        return GenerationResponse(**json.loads(cached_response))

    # Generate
    try:
        generated_text = await generate_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop_sequences,
        )

        # Build response
        response = GenerationResponse(
            id=f"gen-{datetime.now().timestamp()}",
            text=generated_text,
            model="llama-2-7b-chat",
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(request.prompt.split()) + len(generated_text.split()),
            },
            created=int(datetime.now().timestamp())
        )

        # Cache response
        await set_cache(cache_key, response.json(), ttl=3600)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Metrics endpoint (Prometheus format)"""
    # Return metrics for Prometheus scraping
    return {
        "requests_total": 1234,
        "requests_success": 1200,
        "requests_failed": 34,
        "avg_latency_ms": 450,
        "gpu_utilization": 0.85,
    }

# ============== Streaming Support ==============

from fastapi.responses import StreamingResponse

@app.post("/v1/generate/stream")
@limiter.limit("50/minute")
async def generate_stream(
    request: GenerationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Streaming generation endpoint
    """
    async def generate_stream_response():
        """Generator pour streaming"""
        # Simulate streaming (dans production, utiliser vLLM streaming)
        full_text = await generate_text(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Stream token by token
        words = full_text.split()
        for word in words:
            chunk = {
                "text": word + " ",
                "done": False,
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.05)  # Simulate streaming delay

        # Final chunk
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    return StreamingResponse(
        generate_stream_response(),
        media_type="text/event-stream"
    )

# ============== Error Handling ==============

from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "code": exc.status_code,
            }
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )

# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers
        log_level="info",
    )
```

### 23.1.3 Configuration Docker

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API service
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
    restart: unless-stopped

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # Nginx load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: unless-stopped

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped

  # Grafana dashboard
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

## 23.2 Optimisation des Performances

### 23.2.1 Batching Dynamique

vLLM implémente automatic batching pour maximiser throughput.

```python
from vllm import LLM, SamplingParams
import asyncio
from collections import deque
from typing import List, Tuple

class BatchingInferenceEngine:
    """
    Engine avec batching dynamique
    """
    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 32,
        max_wait_time: float = 0.05,  # 50ms
    ):
        self.llm = LLM(model=model_name)
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        # Queue de requêtes en attente
        self.request_queue = deque()
        self.processing = False

    async def add_request(self, prompt: str, params: dict) -> str:
        """
        Ajoute une requête à la queue

        Returns: generated text
        """
        # Create future pour cette requête
        future = asyncio.Future()

        # Add to queue
        self.request_queue.append((prompt, params, future))

        # Start processing si pas déjà en cours
        if not self.processing:
            asyncio.create_task(self._process_batch())

        # Wait for result
        result = await future
        return result

    async def _process_batch(self):
        """Process batch de requêtes"""
        self.processing = True

        while self.request_queue:
            # Collect batch
            batch = []
            futures = []

            # Wait pour accumuler requêtes
            await asyncio.sleep(self.max_wait_time)

            # Collect up to max_batch_size
            while self.request_queue and len(batch) < self.max_batch_size:
                prompt, params, future = self.request_queue.popleft()
                batch.append((prompt, params))
                futures.append(future)

            if not batch:
                break

            # Process batch
            try:
                results = await self._generate_batch(batch)

                # Set results
                for future, result in zip(futures, results):
                    future.set_result(result)

            except Exception as e:
                # Set exception pour toutes les requêtes
                for future in futures:
                    future.set_exception(e)

        self.processing = False

    async def _generate_batch(self, batch: List[Tuple[str, dict]]) -> List[str]:
        """Generate pour un batch"""
        prompts = [item[0] for item in batch]
        params = [item[1] for item in batch]

        # Créer sampling params (assume tous pareils pour simplicité)
        sampling_params = SamplingParams(**params[0])

        # Generate
        outputs = self.llm.generate(prompts, sampling_params)

        # Extract texts
        results = [output.outputs[0].text for output in outputs]

        return results

# Usage
engine = BatchingInferenceEngine("meta-llama/Llama-2-7b-chat-hf")

# Multiple concurrent requests
async def make_requests():
    tasks = [
        engine.add_request("Hello, how are you?", {"temperature": 0.7, "max_tokens": 100}),
        engine.add_request("What is AI?", {"temperature": 0.7, "max_tokens": 100}),
        engine.add_request("Explain quantum physics", {"temperature": 0.7, "max_tokens": 100}),
    ]

    results = await asyncio.gather(*tasks)
    return results

# Run
results = asyncio.run(make_requests())
```

### 23.2.2 KV Cache Optimization

Réutiliser les KV caches pour les prefixes communs.

```python
class KVCacheManager:
    """
    Manage KV cache pour réutilisation
    """
    def __init__(self):
        self.cache = {}  # prefix -> (key_cache, value_cache)

    def get_cache(self, prefix: str):
        """Get cached KV pour un prefix"""
        return self.cache.get(prefix)

    def set_cache(self, prefix: str, key_cache, value_cache):
        """Cache KV pour un prefix"""
        self.cache[prefix] = (key_cache, value_cache)

    def clear_cache(self):
        """Clear all caches"""
        self.cache.clear()

# Usage avec système prompt
system_prompt = "You are a helpful AI assistant specialized in Python programming."

kv_manager = KVCacheManager()

# First request: compute et cache
user_query_1 = "How do I read a file?"
full_prompt_1 = f"{system_prompt}\n\nUser: {user_query_1}\nAssistant:"

# Generate (calcule KV cache pour system_prompt)
output_1, kv_cache = model.generate_with_cache(full_prompt_1)

# Cache le system prompt
kv_manager.set_cache(system_prompt, kv_cache["keys"], kv_cache["values"])

# Second request: réutilise cache
user_query_2 = "How do I write to a file?"
full_prompt_2 = f"{system_prompt}\n\nUser: {user_query_2}\nAssistant:"

# Réutilise cached KV pour system_prompt (plus rapide!)
cached_kv = kv_manager.get_cache(system_prompt)
output_2 = model.generate_with_cached_kv(full_prompt_2, cached_kv)
```

### 23.2.3 Prompt Caching Sémantique

Cache basé sur similarité sémantique plutôt que exact match.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCache:
    """
    Cache avec similarité sémantique
    """
    def __init__(self, similarity_threshold=0.95):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = []  # List of (embedding, prompt, response)
        self.threshold = similarity_threshold

    def get(self, prompt: str):
        """
        Get cached response si prompt similaire existe
        """
        if not self.cache:
            return None

        # Encode prompt
        prompt_emb = self.encoder.encode(prompt)

        # Calculer similarité avec tous les cached prompts
        for cached_emb, cached_prompt, cached_response in self.cache:
            similarity = np.dot(prompt_emb, cached_emb) / (
                np.linalg.norm(prompt_emb) * np.linalg.norm(cached_emb)
            )

            if similarity >= self.threshold:
                print(f"Cache hit! Similarity: {similarity:.3f}")
                return cached_response

        return None

    def set(self, prompt: str, response: str):
        """Cache prompt-response pair"""
        embedding = self.encoder.encode(prompt)
        self.cache.append((embedding, prompt, response))

        # Limit cache size
        if len(self.cache) > 1000:
            self.cache.pop(0)

# Usage
semantic_cache = SemanticCache(similarity_threshold=0.95)

# First query
prompt1 = "What is machine learning?"
response1 = model.generate(prompt1)
semantic_cache.set(prompt1, response1)

# Similar query (cache hit)
prompt2 = "Can you explain machine learning?"
cached_response = semantic_cache.get(prompt2)  # Returns response1

if cached_response:
    response2 = cached_response  # Use cached (pas de génération)
else:
    response2 = model.generate(prompt2)
    semantic_cache.set(prompt2, response2)
```

## 23.3 Monitoring & Observability

### 23.3.1 Metrics Collection avec Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# Define metrics
requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['endpoint', 'status']
)

request_duration = Histogram(
    'llm_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf'))
)

tokens_generated = Counter(
    'llm_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

active_requests = Gauge(
    'llm_active_requests',
    'Number of active requests'
)

# Middleware pour tracking
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for each request"""

    # Increment active requests
    active_requests.inc()

    # Start timer
    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        request_duration.labels(endpoint=request.url.path).observe(duration)
        requests_total.labels(
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        return response

    except Exception as e:
        # Record error
        requests_total.labels(
            endpoint=request.url.path,
            status=500
        ).inc()
        raise

    finally:
        # Decrement active requests
        active_requests.dec()

# Metrics endpoint
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Update GPU metrics (background task)
import pynvml

async def update_gpu_metrics():
    """Update GPU utilization metrics"""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    while True:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)

        await asyncio.sleep(5)  # Update every 5 seconds

# Start background task
@app.on_event("startup")
async def start_metrics_updater():
    asyncio.create_task(update_gpu_metrics())
```

### 23.3.2 Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Format logs en JSON pour meilleure parsing
    """
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

# Setup logger
def setup_logger():
    logger = logging.getLogger("llm_api")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler("logs/api.log")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# Usage
@app.post("/v1/generate")
async def generate(request: GenerationRequest):
    request_id = str(uuid.uuid4())

    logger.info(
        "Generation request received",
        extra={
            "request_id": request_id,
            "prompt_length": len(request.prompt),
            "max_tokens": request.max_tokens,
        }
    )

    try:
        result = await generate_text(request.prompt)

        logger.info(
            "Generation completed",
            extra={
                "request_id": request_id,
                "tokens_generated": len(result.split()),
            }
        )

        return result

    except Exception as e:
        logger.error(
            "Generation failed",
            extra={"request_id": request_id},
            exc_info=True
        )
        raise
```

---

*[Le chapitre continue avec Load Balancing, Auto-scaling, Disaster Recovery, et case studies...]*

*[Contenu total du Chapitre 23: ~70-80 pages]*
