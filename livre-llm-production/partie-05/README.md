# Partie 5 : Données - Collecte, nettoyage et préparation

## Objectifs d'apprentissage

- Identifier et collecter des sources de données appropriées
- Maîtriser les techniques de nettoyage et filtrage
- Implémenter la déduplication à grande échelle
- Construire un mélange de données optimal
- Mettre en place un pipeline de streaming efficace

## Prérequis

- Compréhension des formats de données (JSON, Parquet, Arrow)
- Python, pandas, et outils big data (Spark optionnel)
- Notions de traitement distribué

---

## 5.1 Sources de données

### 5.1.1 Données publiques

**Common Crawl** :
- Crawl complet du web (~250TB par dump)
- Fichiers WARC (Web ARChive)
- Nécessite filtrage intensif

```python
# Télécharger un segment de Common Crawl
from warcio.archiveiterator import ArchiveIterator

def extract_text_from_warc(warc_path):
    with open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                content = record.content_stream().read()
                # Extraire le texte HTML
                yield extract_clean_text(content)
```

**Autres sources publiques** :
- **Wikipedia** : Connaissances factuelles, multilingue
- **Books** (Gutenberg, Books3) : Texte long-forme de qualité
- **Code** (GitHub, StackOverflow) : Pour modèles de code
- **Papers** (arXiv, PubMed) : Contenu scientifique
- **Reddit, Twitter** : Conversations et style informel

### 5.1.2 Données privées et propriétaires

**Sources internes** :
- Documents d'entreprise
- Historiques de conversations (support, chat)
- Bases de connaissances, wikis internes
- Logs et traces d'utilisation

**Considérations** :
- Droits d'utilisation et propriété intellectuelle
- Confidentialité et données sensibles (PII)
- Qualité variable, nécessite curation

### 5.1.3 Données synthétiques

**Génération par LLM** :
- Utiliser un modèle fort (GPT-4, Claude) pour générer des exemples
- Instructions + few-shot examples → nouveaux exemples

**Exemple** :

```python
def generate_synthetic_data(prompt_template, num_examples=1000):
    examples = []
    for i in range(num_examples):
        prompt = prompt_template.format(seed=i)
        response = call_llm(prompt)
        examples.append(response)
    return examples

# Prompt pour générer des paires Q&A
template = """Génère une question et sa réponse sur le thème de la programmation Python.

Question:"""

synthetic_data = generate_synthetic_data(template, 10000)
```

**Avantages** :
- Contrôle du contenu et du style
- Couverture de cas spécifiques
- Augmentation de données pour domaines rares

**Limites** :
- Risque de biais et d'homogénéisation
- Qualité dépendante du modèle générateur
- Coût (appels API)

### 5.1.4 Droits, licences et considérations légales

**Points de vigilance** :

1. **Licences** : Vérifier les termes d'utilisation (CC-BY, CC-BY-SA, MIT, Apache, propriétaire)
2. **PII (Personally Identifiable Information)** : Emails, numéros de téléphone, adresses
3. **Contenu sensible** : Données médicales (HIPAA), financières (PCI DSS), RGPD
4. **Copyright** : Éviter le contenu sous copyright strict
5. **Biais et représentativité** : Diversité linguistique, culturelle, démographique

---

## 5.2 Formats et preprocessing

### 5.2.1 Formats courants

**Texte brut (.txt)** :
- Simple mais inefficace pour grands volumes
- Pas de métadonnées

**JSON / JSONL** :
```json
{"text": "Exemple de document", "url": "https://...", "date": "2024-01-01"}
```

Avantages : Métadonnées, structuré.

**Parquet** :
- Format colonnaire compressé
- Lecture rapide et sélective
- Standard pour big data

```python
import pyarrow.parquet as pq

# Écrire
table = pa.Table.from_pandas(df)
pq.write_table(table, 'data.parquet')

# Lire
table = pq.read_table('data.parquet')
df = table.to_pandas()
```

**Arrow** :
- Format en mémoire compatible Parquet
- Interopérabilité entre langages (Python, C++, Rust)

### 5.2.2 Extraction de texte depuis HTML

**BeautifulSoup** :

```python
from bs4 import BeautifulSoup

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Supprimer scripts et styles
    for script in soup(["script", "style"]):
        script.decompose()

    # Extraire le texte
    text = soup.get_text(separator='\n')

    # Nettoyer les espaces
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text
```

**Trafilatura** (plus robuste) :

```python
import trafilatura

def extract_clean_text(html):
    return trafilatura.extract(html, include_comments=False, include_tables=False)
```

---

## 5.3 Nettoyage et filtrage

### 5.3.1 Filtres de qualité de base

**Longueur** :
- Trop court : Probablement spam ou peu informatif
- Trop long : Peut-être du bruit (logs, dumps)

```python
def filter_by_length(text, min_length=100, max_length=100000):
    return min_length <= len(text) <= max_length
```

**Ratio alphanumérique** :

```python
import re

def alphanum_ratio(text):
    alphanum = len(re.findall(r'[a-zA-Z0-9]', text))
    total = len(text)
    return alphanum / total if total > 0 else 0

def filter_alphanum(text, min_ratio=0.7):
    return alphanum_ratio(text) >= min_ratio
```

**Filtres de langue** :

```python
from langdetect import detect

def filter_by_language(text, target_lang='en'):
    try:
        return detect(text) == target_lang
    except:
        return False
```

### 5.3.2 Filtres de contenu

**Toxicité** :

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="unitary/toxic-bert")

def is_toxic(text, threshold=0.5):
    result = classifier(text[:512])[0]  # Limiter longueur
    return result['label'] == 'toxic' and result['score'] > threshold
```

**Spam et contenu de faible qualité** :

```python
# Heuristiques simples
def is_spam(text):
    # Trop de répétitions
    if len(set(text.split())) / len(text.split()) < 0.3:
        return True

    # Mots-clés spam
    spam_keywords = ['click here', 'buy now', 'limited offer']
    if any(kw in text.lower() for kw in spam_keywords):
        return True

    return False
```

**PII (Personally Identifiable Information)** :

```python
import re

def contains_pii(text):
    # Email
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        return True

    # Téléphone (simple)
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
        return True

    # Numéro de carte de crédit (basique)
    if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
        return True

    return False
```

### 5.3.3 Perplexité filtering

**Principe** : Utiliser un modèle de langue pour scorer la "normalité" du texte.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()

def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def filter_by_perplexity(text, max_ppl=1000):
    ppl = compute_perplexity(text)
    return ppl < max_ppl
```

Filtre les textes anormaux ou très mal formés.

---

## 5.4 Déduplication

### 5.4.1 Importance de la déduplication

**Problèmes du contenu dupliqué** :
- Mémorisation et overfitting
- Biais vers contenu répété
- Gaspillage de compute

**Types de duplications** :
- Exacte (copie identique)
- Near-duplicate (quasi-identique)
- Substring matches (paragraphes répétés)

### 5.4.2 Déduplication exacte

**Hash de documents** :

```python
import hashlib

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def deduplicate_exact(documents):
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        doc_hash = hash_text(doc)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs
```

### 5.4.3 Déduplication near-duplicate avec MinHash

**Principe** : Approximation rapide de la similarité Jaccard via hashing.

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

def deduplicate_near_duplicates(documents, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Indexer
    for idx, doc in enumerate(documents):
        minhash = create_minhash(doc)
        lsh.insert(f"doc_{idx}", minhash)

    # Trouver les duplicatas
    duplicates = set()
    for idx, doc in enumerate(documents):
        minhash = create_minhash(doc)
        results = lsh.query(minhash)
        if len(results) > 1:  # Lui-même + autres
            duplicates.add(idx)

    # Garder seulement les uniques
    unique_docs = [doc for idx, doc in enumerate(documents) if idx not in duplicates]
    return unique_docs
```

**Scalabilité** :
- Pour très grands datasets : utiliser Spark ou Dask
- MinHashLSH permet de passer à des milliards de documents

### 5.4.4 Déduplication au niveau paragraphe

**Problème** : Certains documents partagent des sections (boilerplate, disclaimers).

**Solution** : Dédup par n-grammes

```python
def extract_ngrams(text, n=13):
    """Extrait tous les n-grammes de mots."""
    words = text.split()
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))

def contains_substantial_overlap(text1, text2, threshold=0.5):
    """Vérifie si deux textes ont un chevauchement substantiel."""
    ngrams1 = extract_ngrams(text1)
    ngrams2 = extract_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return False

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    jaccard = intersection / union if union > 0 else 0

    return jaccard > threshold
```

---

## 5.5 Construction du mélange de données

### 5.5.1 Stratégie de mélange

**Objectifs** :
- Diversité : Couvrir plusieurs domaines, styles, langues
- Équilibre : Éviter qu'une source domine
- Qualité : Favoriser les sources de haute qualité

**Exemple de composition** (inspiré de LLaMA, GPT) :

| Source          | Proportion | Raison                              |
|-----------------|------------|-------------------------------------|
| Common Crawl    | 40%        | Diversité web                       |
| Wikipedia       | 5%         | Connaissances factuelles            |
| Books           | 15%        | Texte long-forme, qualité           |
| GitHub          | 10%        | Code                                |
| arXiv           | 5%         | Connaissances scientifiques         |
| StackOverflow   | 5%         | Q&A technique                       |
| Reddit          | 10%        | Conversations                       |
| News            | 10%        | Actualités, style journalistique    |

### 5.5.2 Sampling et pondération

**Sampling proportionnel** :

```python
import random

def sample_mixed_dataset(sources, proportions, total_size):
    """
    sources: dict {nom: liste_de_documents}
    proportions: dict {nom: proportion}
    """
    mixed = []
    for source_name, proportion in proportions.items():
        num_samples = int(total_size * proportion)
        samples = random.sample(sources[source_name], num_samples)
        mixed.extend(samples)

    random.shuffle(mixed)
    return mixed
```

**Temperature sampling** (pour sur/sous-échantillonner) :

```
p_i = (count_i)^T / Σ (count_j)^T
```

- T < 1 : Favorise les sources rares
- T = 1 : Proportionnel
- T > 1 : Favorise les sources dominantes

### 5.5.3 Upsampling de données de qualité

Pour renforcer l'apprentissage sur données de haute qualité :

```python
def upsample_quality_data(documents, quality_scores, factor=2):
    """Répète les documents de haute qualité."""
    upsampled = []
    for doc, score in zip(documents, quality_scores):
        upsampled.append(doc)
        if score > 0.8:  # Seuil de qualité
            for _ in range(factor - 1):
                upsampled.append(doc)
    return upsampled
```

---

## 5.6 Sharding, indexation et streaming

### 5.6.1 Sharding pour traitement distribué

**Principe** : Diviser le dataset en chunks pour paralléliser.

```python
import os

def shard_dataset(documents, output_dir, shard_size=10000):
    os.makedirs(output_dir, exist_ok=True)

    for shard_idx, i in enumerate(range(0, len(documents), shard_size)):
        shard = documents[i:i + shard_size]
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.jsonl")

        with open(shard_path, 'w') as f:
            for doc in shard:
                f.write(json.dumps(doc) + '\n')

    print(f"Created {shard_idx + 1} shards in {output_dir}")
```

### 5.6.2 Streaming avec HuggingFace Datasets

**Avantages** : Pas besoin de tout charger en mémoire.

```python
from datasets import load_dataset

# Charger en streaming
dataset = load_dataset("c4", "en", split="train", streaming=True)

# Itérer
for example in dataset:
    text = example['text']
    # Traiter...
```

**Custom streaming dataset** :

```python
class StreamingTextDataset:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    yield json.loads(line)

# Utilisation
dataset = StreamingTextDataset(glob.glob("data/shards/*.jsonl"))
for example in dataset:
    print(example['text'][:100])
```

### 5.6.3 Indexation pour retrieval rapide

**Elasticsearch** (pour recherche full-text) :

```python
from elasticsearch import Elasticsearch

es = Elasticsearch()

# Indexer
for idx, doc in enumerate(documents):
    es.index(index="corpus", id=idx, body={"text": doc})

# Rechercher
results = es.search(index="corpus", body={"query": {"match": {"text": "python programming"}}})
```

**FAISS** (pour recherche vectorielle) :

```python
import faiss
import numpy as np

# Créer index
dimension = 768  # taille des embeddings
index = faiss.IndexFlatL2(dimension)

# Ajouter vecteurs
embeddings = np.random.rand(10000, dimension).astype('float32')
index.add(embeddings)

# Rechercher les k plus proches voisins
query = np.random.rand(1, dimension).astype('float32')
k = 10
distances, indices = index.search(query, k)
```

---

## 5.7 Labs pratiques

### Lab 1 : Pipeline de déduplication

Implémentez un pipeline complet de déduplication :

1. Dédup exacte (hash)
2. Dédup near-duplicate (MinHash)
3. Mesurer le taux de duplication avant/après

```python
# TODO: Implémenter
def full_dedup_pipeline(documents):
    print(f"Documents initiaux: {len(documents)}")

    # Étape 1
    docs = deduplicate_exact(documents)
    print(f"Après dédup exacte: {len(docs)}")

    # Étape 2
    docs = deduplicate_near_duplicates(docs, threshold=0.8)
    print(f"Après dédup near-duplicate: {len(docs)}")

    return docs
```

### Lab 2 : Analyse de distributions

Analysez la distribution d'un corpus :
- Longueur des documents
- Distribution de langues
- Scores de toxicité
- Perplexité

Visualisez avec matplotlib/seaborn.

### Lab 3 : Construire un mélange

Créez un dataset mixte à partir de 3 sources avec proportions spécifiées, puis shardez pour entraînement distribué.

---

## Résumé

Vous maîtrisez maintenant :
- ✅ Identification et collecte de sources de données (publiques, privées, synthétiques)
- ✅ Nettoyage et filtrage (qualité, toxicité, PII, perplexité)
- ✅ Déduplication (exacte, near-duplicate, n-grammes)
- ✅ Construction de mélanges de données optimaux
- ✅ Sharding, indexation et streaming

**Prochaine étape** : [Partie 6 - Pré-training](../partie-06/README.md) pour entraîner votre LLM de zéro.
