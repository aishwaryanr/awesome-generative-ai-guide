# Partie 10 : Déploiement en production et LLMOps

## Objectifs d'apprentissage

- Architecturer une API LLM robuste et scalable
- Mettre en place observabilité et monitoring complets
- Détecter la dérive et organiser le ré-entraînement
- Optimiser les coûts et la capacité
- Assurer sécurité, privacy et conformité

**Référence** : Park et al. (5.1) pour choix d'engines et optimisations

---

## 10.1 Architecture API

### 10.1.1 Gateway et routing

**Stack typique** :

```
Users → [Load Balancer] → [API Gateway] → [LLM Service]
                                 ↓
                          [Auth / Rate Limiting]
                                 ↓
                          [Logging / Monitoring]
```

**API Gateway avec FastAPI** :

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

app = FastAPI()
security = HTTPBearer()

# Rate limiting (simple in-memory)
from collections import defaultdict
request_counts = defaultdict(lambda: {"count": 0, "reset_time": time.time() + 60})

def rate_limit(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Limite à 10 requêtes/minute par clé API."""
    api_key = credentials.credentials
    user_data = request_counts[api_key]

    if time.time() > user_data["reset_time"]:
        user_data["count"] = 0
        user_data["reset_time"] = time.time() + 60

    if user_data["count"] >= 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    user_data["count"] += 1
    return api_key

@app.post("/v1/completions")
async def generate(request: dict, api_key: str = Depends(rate_limit)):
    """Endpoint de génération."""
    prompt = request.get("prompt")
    max_tokens = request.get("max_tokens", 100)

    # Appeler le modèle
    response = llm_service.generate(prompt, max_tokens=max_tokens)

    return {"text": response, "model": "llama-2-7b", "usage": {"tokens": len(response.split())}}
```

### 10.1.2 Authentification et autorisation

**API Keys** :

```python
import secrets
import hashlib

class APIKeyManager:
    def __init__(self):
        self.keys = {}  # En pratique: base de données

    def create_key(self, user_id, permissions=None):
        """Générer une nouvelle clé API."""
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        self.keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions or ["read", "write"],
            "created_at": time.time()
        }

        return key  # Retourner seulement une fois, ensuite utiliser le hash

    def validate_key(self, key):
        """Valider et récupérer les permissions."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.keys.get(key_hash)
```

**JWT pour auth avancée** :

```python
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"

def create_jwt_token(user_id, expires_delta=timedelta(hours=24)):
    """Créer un JWT."""
    expire = datetime.utcnow() + expires_delta
    payload = {"user_id": user_id, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_jwt_token(token):
    """Vérifier et décoder un JWT."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 10.1.3 Streaming responses

**SSE (Server-Sent Events)** :

```python
from fastapi.responses import StreamingResponse

@app.post("/v1/completions/stream")
async def generate_stream(request: dict):
    """Génération en streaming."""

    async def event_generator():
        prompt = request.get("prompt")

        for token in llm_service.generate_stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 10.2 Observabilité et monitoring

### 10.2.1 Logging structuré

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_request(self, user_id, prompt, response, latency, tokens_used):
        """Logger une requête LLM."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "llm_request",
            "user_id": user_id,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency_ms": latency * 1000,
            "tokens_used": tokens_used
        }

        self.logger.info(json.dumps(log_entry))

# Utilisation
logger = StructuredLogger("llm_api")

start = time.time()
response = llm.generate(prompt)
latency = time.time() - start

logger.log_request(user_id, prompt, response, latency, tokens_used=150)
```

### 10.2.2 Métriques avec Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge
import prometheus_client

# Définir métriques
request_count = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
active_requests = Gauge('llm_active_requests', 'Currently active requests')
token_count = Counter('llm_tokens_generated_total', 'Total tokens generated')

@app.post("/v1/completions")
async def generate(request: dict):
    active_requests.inc()
    start = time.time()

    try:
        response = llm.generate(request["prompt"])
        request_count.labels(model="llama-2-7b", status="success").inc()
        token_count.inc(len(response.split()))

        return {"text": response}

    except Exception as e:
        request_count.labels(model="llama-2-7b", status="error").inc()
        raise

    finally:
        latency = time.time() - start
        request_latency.observe(latency)
        active_requests.dec()

# Exposer les métriques
@app.get("/metrics")
async def metrics():
    return prometheus_client.generate_latest()
```

### 10.2.3 Dashboards (Grafana)

**Métriques clés à visualiser** :

1. **Requêtes** : Total, par minute, par utilisateur
2. **Latence** : P50, P95, P99
3. **Erreurs** : Taux d'erreur, types d'erreurs
4. **Tokens** : Tokens/seconde, coût estimé
5. **Ressources** : GPU utilization, mémoire

**Alertes** :

```yaml
# Exemple de règle Prometheus
groups:
  - name: llm_alerts
    rules:
      - alert: HighLatency
        expr: llm_request_latency_seconds{quantile="0.95"} > 2
        for: 5m
        annotations:
          summary: "Latence P95 > 2s pendant 5 minutes"

      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) > 0.05
        for: 2m
        annotations:
          summary: "Taux d'erreur > 5%"
```

---

## 10.3 Qualité et dérive

### 10.3.1 Métriques de qualité perçue

**Feedback utilisateur** :

```python
class FeedbackCollector:
    def __init__(self):
        self.feedback_db = []

    def collect(self, request_id, rating, comment=None):
        """Collecter un thumbs up/down."""
        self.feedback_db.append({
            "request_id": request_id,
            "rating": rating,  # 1 ou -1
            "comment": comment,
            "timestamp": time.time()
        })

    def get_satisfaction_rate(self, window_hours=24):
        """Calculer le taux de satisfaction."""
        cutoff = time.time() - window_hours * 3600
        recent = [f for f in self.feedback_db if f["timestamp"] > cutoff]

        if not recent:
            return None

        positive = sum(1 for f in recent if f["rating"] == 1)
        return positive / len(recent)

# API endpoint
@app.post("/v1/feedback")
async def submit_feedback(request_id: str, rating: int):
    feedback_collector.collect(request_id, rating)
    return {"status": "ok"}
```

### 10.3.2 Détection de dérive

**Drift de distribution des prompts** :

```python
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self):
        self.baseline_embeddings = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def set_baseline(self, prompts):
        """Établir une distribution baseline."""
        self.baseline_embeddings = self.embedder.encode(prompts)

    def detect_drift(self, recent_prompts, threshold=0.05):
        """Détecter si la distribution a changé."""
        recent_embeddings = self.embedder.encode(recent_prompts)

        # Test KS sur chaque dimension
        p_values = []
        for dim in range(self.baseline_embeddings.shape[1]):
            _, p_value = ks_2samp(
                self.baseline_embeddings[:, dim],
                recent_embeddings[:, dim]
            )
            p_values.append(p_value)

        # Si beaucoup de dimensions ont changé
        drift_detected = sum(p < threshold for p in p_values) > len(p_values) * 0.1

        return drift_detected, min(p_values)
```

**Monitoring continu** :

```python
# Tous les jours, vérifier drift
import schedule

def check_drift():
    recent_prompts = get_prompts_last_24h()
    drift_detected, p_value = drift_detector.detect_drift(recent_prompts)

    if drift_detected:
        logger.warning(f"Drift détecté ! p-value min = {p_value:.4f}")
        # Déclencher alerte

schedule.every().day.at("02:00").do(check_drift)
```

### 10.3.3 Pipelines de ré-entraînement

**Critères de ré-entraînement** :

1. Dérive détectée
2. Taux de satisfaction < seuil
3. Nouvelles données substantielles disponibles
4. Période régulière (ex: tous les 3 mois)

```python
def should_retrain():
    """Décider s'il faut ré-entraîner."""
    drift = drift_detector.detect_drift(recent_prompts)[0]
    satisfaction = feedback_collector.get_satisfaction_rate() or 1.0
    days_since_last_train = (time.time() - last_train_timestamp) / 86400

    return (
        drift or
        satisfaction < 0.7 or
        days_since_last_train > 90
    )

if should_retrain():
    trigger_retraining_pipeline()
```

---

## 10.4 Optimisation des coûts

### 10.4.1 Coût par token

**Calculer le coût** :

```python
class CostTracker:
    def __init__(self, cost_per_1m_tokens=0.002):  # $2 / 1M tokens
        self.cost_per_token = cost_per_1m_tokens / 1_000_000
        self.total_tokens = 0
        self.total_cost = 0.0

    def track(self, num_tokens):
        self.total_tokens += num_tokens
        cost = num_tokens * self.cost_per_token
        self.total_cost += cost
        return cost

    def report(self):
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "cost_per_request": self.total_cost / max(request_count, 1)
        }
```

### 10.4.2 Caching intelligent

**Semantic caching** : Mettre en cache les réponses pour prompts similaires.

```python
class SemanticCache:
    def __init__(self, embedder, similarity_threshold=0.95):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.cache = []  # [(embedding, prompt, response), ...]

    def get(self, prompt):
        """Chercher dans le cache."""
        query_emb = self.embedder.encode([prompt])[0]

        for cached_emb, cached_prompt, cached_response in self.cache:
            similarity = cosine_similarity(query_emb, cached_emb)

            if similarity > self.threshold:
                logger.info(f"Cache hit! Similarity: {similarity:.3f}")
                return cached_response

        return None

    def set(self, prompt, response):
        """Ajouter au cache."""
        embedding = self.embedder.encode([prompt])[0]
        self.cache.append((embedding, prompt, response))

        # Limiter la taille
        if len(self.cache) > 10000:
            self.cache = self.cache[-10000:]

# Utilisation
@app.post("/v1/completions")
async def generate(request: dict):
    prompt = request["prompt"]

    # Vérifier cache
    cached = semantic_cache.get(prompt)
    if cached:
        return {"text": cached, "cached": True}

    # Générer
    response = llm.generate(prompt)
    semantic_cache.set(prompt, response)

    return {"text": response, "cached": False}
```

### 10.4.3 Stratégie cascade (small → large)

**Principe** : Utiliser un petit modèle rapide, escalader au grand modèle si nécessaire.

```python
def cascade_generate(prompt, small_model, large_model, confidence_threshold=0.8):
    """Essayer small model d'abord."""

    # Générer avec petit modèle
    small_response, confidence = small_model.generate_with_confidence(prompt)

    if confidence > confidence_threshold:
        logger.info("Small model suffisant")
        return small_response, "small"

    # Escalader au grand modèle
    logger.info("Escalade vers grand modèle")
    large_response = large_model.generate(prompt)
    return large_response, "large"
```

---

## 10.5 Sécurité et conformité

### 10.5.1 PII et anonymisation

**Détecter et masquer PII** :

```python
import re

def anonymize_pii(text):
    """Masquer emails, téléphones, etc."""

    # Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)

    # Téléphones (simple)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Numéros de carte
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CREDIT_CARD]', text)

    return text

# Appliquer avant logging
logged_prompt = anonymize_pii(user_prompt)
logger.log_request(user_id, logged_prompt, response, latency, tokens)
```

### 10.5.2 Audit et compliance

**Audit trail** :

```python
class AuditLogger:
    def __init__(self, db_connection):
        self.db = db_connection

    def log_access(self, user_id, resource, action, result):
        """Logger chaque accès pour audit."""
        self.db.execute("""
            INSERT INTO audit_log (timestamp, user_id, resource, action, result, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow(), user_id, resource, action, result, request.client.host))

# Utilisation
audit_logger.log_access(user_id, "llm_completion", "generate", "success")
```

**RGPD : Droit à l'oubli** :

```python
def delete_user_data(user_id):
    """Supprimer toutes les données d'un utilisateur."""
    # Logs
    logging_db.execute("DELETE FROM logs WHERE user_id = ?", (user_id,))

    # Feedback
    feedback_db.execute("DELETE FROM feedback WHERE user_id = ?", (user_id,))

    # Audit (garder anonymisé)
    audit_db.execute("UPDATE audit_log SET user_id = 'DELETED' WHERE user_id = ?", (user_id,))

    logger.info(f"Données utilisateur {user_id} supprimées (RGPD)")
```

### 10.5.3 Menaces spécifiques LLM

**Prompt injection** :

```python
def detect_prompt_injection(prompt):
    """Détecter des patterns d'injection."""
    injection_patterns = [
        r"ignore (previous|above|all) instructions",
        r"disregard.*rules",
        r"new instructions",
        r"system.*override"
    ]

    for pattern in injection_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True

    return False

# Bloquer si détecté
if detect_prompt_injection(prompt):
    raise HTTPException(status_code=400, detail="Prompt injection detected")
```

**Rate limiting adaptatif** (contre abus) :

```python
class AdaptiveRateLimiter:
    def __init__(self):
        self.user_behavior = defaultdict(lambda: {"requests": [], "suspicious": 0})

    def check(self, user_id):
        """Limiter selon le comportement."""
        now = time.time()
        user = self.user_behavior[user_id]

        # Nettoyer vieux événements
        user["requests"] = [t for t in user["requests"] if now - t < 3600]

        # Limites adaptatives
        if user["suspicious"] > 3:
            limit = 10  # Très restrictif
        elif user["suspicious"] > 0:
            limit = 50
        else:
            limit = 100

        if len(user["requests"]) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit")

        user["requests"].append(now)
```

---

## 10.6 Labs pratiques

### Lab 1 : API complète avec monitoring

1. Déployer une API FastAPI avec auth, rate limiting
2. Intégrer Prometheus metrics
3. Créer des dashboards Grafana
4. Configurer des alertes

### Lab 2 : Détection de dérive

1. Collecter un baseline de prompts
2. Simuler un drift (changer la distribution)
3. Détecter automatiquement le drift
4. Déclencher un pipeline de ré-entraînement

### Lab 3 : Optimisation des coûts

1. Implémenter semantic caching
2. Comparer coût avec/sans cache
3. Tester la stratégie cascade small→large
4. Mesurer le trade-off coût/qualité

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Architecture API (gateway, auth, rate limiting, streaming)
- ✅ Observabilité (logging structuré, métriques Prometheus, dashboards)
- ✅ Détection de dérive et pipelines de ré-entraînement
- ✅ Optimisation des coûts (tracking, caching, cascade)
- ✅ Sécurité et conformité (PII, audit, RGPD, injection)

**Prochaine étape** : [Partie 11 - Étude de cas fil rouge](../partie-11/README.md)
