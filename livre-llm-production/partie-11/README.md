# Partie 11 : Étude de cas fil rouge - Assistant technique de bout en bout

## Objectifs d'apprentissage

- Conduire un projet LLM complet de l'idée au déploiement
- Appliquer toutes les techniques apprises dans les parties précédentes
- Gérer les itérations et l'amélioration continue
- Mesurer l'impact business et technique

---

## 11.1 Spécification du projet

### 11.1.1 Cahier des charges

**Projet** : Assistant technique pour développeurs Python

**Fonctionnalités** :
1. Répondre aux questions de programmation Python
2. Générer du code à partir de descriptions
3. Debugger et expliquer des erreurs
4. Suggérer des optimisations
5. Rechercher dans la documentation officielle (RAG)

**Utilisateurs** :
- Développeurs juniors à seniors
- ~10 000 utilisateurs potentiels
- Utilisation quotidienne (5-20 requêtes/jour/utilisateur)

**Contraintes** :
- Latence < 2s (P95)
- Disponibilité > 99.5%
- Coût < $0.01 par requête
- Conformité RGPD (données en Europe)
- Sécurité : Pas d'injection de code malveillant

### 11.1.2 Métriques de succès

**Business** :
- Adoption : 30% des développeurs utilisent l'outil après 3 mois
- Satisfaction : CSAT > 4/5
- Productivité : Gain estimé de 15% sur tâches ciblées

**Technique** :
- Qualité : Score BLEU > 0.4 sur génération de code
- Latence : P95 < 2s, P50 < 500ms
- Disponibilité : 99.5%
- Coût : $0.008 par requête en moyenne

---

## 11.2 Phase 1 : Prototype de recherche

### 11.2.1 Dataset initial

**Sources** :
- StackOverflow Python Q&A : 100k paires filtrées (score > 10)
- Documentation Python officielle : Chunked et indexé
- GitHub Python : Exemples de code high-quality
- Synthetic data : 10k paires générées via GPT-4

**Préparation** :

```python
# 1. Scraper StackOverflow
import praw  # Reddit/StackExchange API

def scrape_stackoverflow_python(min_score=10, limit=100000):
    # TODO: Implémenter avec StackExchange API
    pass

# 2. Chunk documentation
docs = load_python_docs()
chunks = []
for doc in docs:
    chunks.extend(chunk_document(doc, chunk_size=500))

# 3. Générer synthetic
def generate_python_qa_pairs(num=10000):
    prompt_template = """Génère une question Python et sa réponse avec un exemple de code.

Format:
Question: [question technique précise]
Réponse: [explication] + [code example]

Exemple:
Question: Comment lire un fichier CSV en Python ?
Réponse: Utilisez le module csv ou pandas.
```python
import csv
with open('file.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)
```

Génère maintenant une nouvelle paire :"""

    # Générer avec GPT-4
    pass
```

### 11.2.2 Modèle de base

**Choix** : Fine-tuner CodeLlama-7B (spécialisé code, open-source)

**Raison** :
- Taille gérable (7B paramètres)
- Pré-entraîné sur code → meilleur point de départ
- License permissive (commercial OK)

**SFT** :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

# Préparer dataset
train_dataset = prepare_sft_dataset(stackoverflow_qa + synthetic_qa)

# Training
training_args = TrainingArguments(
    output_dir="./codellama-python-assistant",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 11.2.3 Évaluation initiale

**Benchmarks** :
- HumanEval (génération de code) : 35% pass@1
- MBPP (problèmes Python basiques) : 42% pass@1
- Custom eval (Q&A internes) : 68% exact match

**Feedback qualitatif** :
- 10 développeurs testent pendant 1 semaine
- 70% de satisfaction globale
- Points d'amélioration : Hallucinations sur API récentes, manque de contexte

---

## 11.3 Phase 2 : Passage à l'échelle

### 11.3.1 RAG pour documentation à jour

**Problème** : Le modèle ne connaît pas les dernières versions de bibliothèques.

**Solution** : RAG avec docs officielles + StackOverflow récent.

```python
# Indexer la documentation
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-mpnet-base-v2')

# Chunks de documentation
docs_chunks = load_and_chunk_docs([
    "Python 3.12 docs",
    "NumPy latest",
    "Pandas latest",
    "Requests, Flask, FastAPI..."
])

# Créer index vectoriel
embeddings = embedder.encode(docs_chunks)
index = create_faiss_index(embeddings)

# Pipeline RAG
def answer_with_rag(query):
    # 1. Retrieval
    relevant_docs = retrieve(query, index, docs_chunks, k=5)

    # 2. Construire prompt augmenté
    context = "\n\n".join(relevant_docs)
    prompt = f"""Documentation pertinente:
{context}

Question: {query}

Réponse précise basée sur la documentation ci-dessus, avec exemple de code si pertinent:"""

    # 3. Générer
    response = model.generate(prompt, max_tokens=500)
    return response, relevant_docs  # Retourner sources pour traçabilité
```

### 11.3.2 Tool use pour exécution de code

**Fonctionnalité** : Exécuter le code Python pour valider les suggestions.

```python
import subprocess
import tempfile

def safe_execute_python(code, timeout=5):
    """Exécuter du code Python en sandbox."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PYTHONDONTWRITEBYTECODE": "1"}  # Sécurité basique
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0
        }

    except subprocess.TimeoutExpired:
        return {"error": "Timeout", "success": False}

    finally:
        os.remove(temp_file)

# Agent avec validation
def generate_and_validate_code(task_description):
    # 1. Générer code
    code = model.generate(f"Écris du code Python pour : {task_description}")

    # 2. Exécuter pour vérifier
    exec_result = safe_execute_python(code)

    if exec_result["success"]:
        return code, "✓ Code testé avec succès"
    else:
        # 3. Refine basé sur l'erreur
        error = exec_result["stderr"]
        refined_code = model.generate(f"""Le code suivant a une erreur:
{code}

Erreur:
{error}

Corrige le code:""")

        return refined_code, "Code corrigé après erreur"
```

### 11.3.3 Alignement avec DPO

**Dataset de préférences** :

Collecter 5000 paires (meilleure/pire réponse) via :
- Annotations humaines (500 paires)
- AI feedback (4500 paires via GPT-4 comme juge)

```python
# Génération de préférences synthétiques
def generate_preference_pair(query):
    # Générer 2 réponses différentes
    response_A = model.generate(query, temperature=0.7)
    response_B = model.generate(query, temperature=0.9)

    # Juge IA (GPT-4)
    judge_prompt = f"""Quelle réponse est meilleure pour cette question Python ?

Question: {query}

Réponse A:
{response_A}

Réponse B:
{response_B}

Choix (A ou B) et justification courte:"""

    judgment = call_gpt4(judge_prompt)

    if "A" in judgment:
        return {"chosen": response_A, "rejected": response_B}
    else:
        return {"chosen": response_B, "rejected": response_A}

# Entraîner DPO
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    beta=0.1,
)

dpo_trainer.train()
```

**Résultats après DPO** :
- HumanEval : 35% → 42% pass@1
- Satisfaction utilisateurs : 70% → 85%
- Réduction des réponses trop verbeuses

---

## 11.4 Phase 3 : Industrialisation

### 11.4.1 Containerisation et CI/CD

**Dockerfile** :

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model/ /app/model/
COPY app/ /app/

WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**GitHub Actions CI/CD** :

```yaml
name: Deploy LLM Assistant

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements-dev.txt
          pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t llm-assistant:${{ github.sha }} .

      - name: Push to registry
        run: docker push llm-assistant:${{ github.sha }}

      - name: Deploy to Kubernetes
        run: kubectl set image deployment/llm-assistant llm-assistant=llm-assistant:${{ github.sha }}
```

### 11.4.2 Déploiement progressif

**Canary deployment** :

```yaml
# k8s/canary.yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-assistant
spec:
  selector:
    app: llm-assistant
  ports:
    - port: 80
      targetPort: 8000

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-assistant-stable
spec:
  replicas: 9  # 90% du trafic
  selector:
    matchLabels:
      app: llm-assistant
      version: stable
  template:
    metadata:
      labels:
        app: llm-assistant
        version: stable
    spec:
      containers:
      - name: llm-assistant
        image: llm-assistant:v1.2.0

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-assistant-canary
spec:
  replicas: 1  # 10% du trafic
  selector:
    matchLabels:
      app: llm-assistant
      version: canary
  template:
    metadata:
      labels:
        app: llm-assistant
        version: canary
    spec:
      containers:
      - name: llm-assistant
        image: llm-assistant:v1.3.0-rc
```

**Surveiller canary** :

```python
def monitor_canary(canary_version, stable_version, duration_minutes=60):
    """Surveiller les métriques pendant le canary."""
    start_time = time.time()

    while time.time() - start_time < duration_minutes * 60:
        # Métriques canary vs stable
        canary_metrics = get_metrics(version=canary_version)
        stable_metrics = get_metrics(version=stable_version)

        # Comparer
        if canary_metrics['error_rate'] > stable_metrics['error_rate'] * 1.5:
            logger.error("Canary error rate trop élevé, rollback!")
            rollback_canary()
            return False

        if canary_metrics['latency_p95'] > stable_metrics['latency_p95'] * 1.2:
            logger.warning("Canary latence élevée")

        time.sleep(60)  # Check toutes les minutes

    # Succès : promouvoir canary
    promote_canary_to_stable()
    return True
```

### 11.4.3 A/B testing

```python
class ABTestManager:
    def __init__(self):
        self.experiments = {}

    def assign_variant(self, user_id, experiment_name):
        """Assigner un utilisateur à une variante."""
        # Hash déterministe pour cohérence
        hash_val = int(hashlib.md5(f"{user_id}{experiment_name}".encode()).hexdigest(), 16)

        experiment = self.experiments[experiment_name]
        threshold = experiment['split']  # ex: 0.5 pour 50/50

        if (hash_val % 100) / 100 < threshold:
            return "A"
        else:
            return "B"

    def log_metric(self, user_id, experiment_name, metric_name, value):
        """Logger une métrique pour analyse."""
        variant = self.assign_variant(user_id, experiment_name)
        # Enregistrer dans DB pour analyse ultérieure
        metrics_db.insert({
            "experiment": experiment_name,
            "variant": variant,
            "user_id": user_id,
            "metric": metric_name,
            "value": value,
            "timestamp": time.time()
        })

# Utilisation
ab_test = ABTestManager()
ab_test.experiments["rag_vs_no_rag"] = {"split": 0.5}

@app.post("/v1/query")
async def query(request: dict, user_id: str):
    variant = ab_test.assign_variant(user_id, "rag_vs_no_rag")

    start = time.time()

    if variant == "A":
        # Avec RAG
        response = answer_with_rag(request["query"])
    else:
        # Sans RAG
        response = model.generate(request["query"])

    latency = time.time() - start

    # Logger métriques
    ab_test.log_metric(user_id, "rag_vs_no_rag", "latency", latency)

    return {"response": response, "variant": variant}
```

---

## 11.5 Résultats et roadmap

### 11.5.1 Résultats après 3 mois

**Adoption** :
- 4200 utilisateurs actifs (42% de la cible)
- 60 000 requêtes/jour

**Satisfaction** :
- CSAT : 4.3/5
- Thumbs up : 82%
- NPS : +45

**Technique** :
- Latence P95 : 1.8s ✓
- Latence P50 : 420ms ✓
- Disponibilité : 99.7% ✓
- Coût : $0.0065/requête ✓

**Impact productivité** :
- Gain estimé : 18% sur tâches ciblées (sondage auto-déclaré)
- Réduction temps de recherche : 12 min → 3 min (moyenne)

### 11.5.2 Roadmap future

**Court terme (3-6 mois)** :
- Support multi-langages (JavaScript, Go, Rust)
- Intégration IDE (VSCode extension)
- Amélioration RAG (reranking plus intelligent)

**Moyen terme (6-12 mois)** :
- Fine-tuning spécialisé par domaine (data science, web dev, etc.)
- Agents autonomes pour debugging multi-fichiers
- Mémoire utilisateur et personnalisation

**Long terme (12+ mois)** :
- Modèles hybrides (Transformer + SSM pour contexte long)
- Training continu avec feedback
- Multi-modalité (screenshots → code)

---

## 11.6 Leçons apprises

### 11.6.1 Techniques

1. **RAG est essentiel** pour connaissances à jour
2. **DPO > RLHF** pour notre use case (simplicité, efficacité)
3. **Validation automatique** (exécution code) réduit hallucinations
4. **Canary + A/B** permettent itération rapide avec confiance

### 11.6.2 Organisation

1. **Commencer simple** : Prototype rapide > perfection initiale
2. **Mesurer tôt** : Métriques dès le début pour piloter
3. **Feedback utilisateur** : Intégrer directement dans le produit
4. **Automatisation** : CI/CD et monitoring indispensables à l'échelle

### 11.6.3 Coûts

**Breakdown mensuel** :

| Poste                 | Coût/mois |
|-----------------------|-----------|
| Compute (inference)   | $2,600    |
| Storage (embeddings)  | $150      |
| Monitoring/logs       | $200      |
| Bandwidth             | $100      |
| **Total**             | **$3,050**|

**Coût par utilisateur actif** : $0.73/mois

**ROI** : Positif dès mois 4 (basé sur gain de productivité estimé)

---

## Résumé

Ce cas fil rouge a démontré :
- ✅ Cycle complet : Idée → Prototype → Scale → Production
- ✅ Intégration de multiples techniques (SFT, RAG, DPO, tool use)
- ✅ Déploiement industriel (CI/CD, canary, A/B tests, monitoring)
- ✅ Mesure d'impact (technique + business)

**Prochaine étape** : [Partie 12 - Annexes techniques](../partie-12/README.md)
