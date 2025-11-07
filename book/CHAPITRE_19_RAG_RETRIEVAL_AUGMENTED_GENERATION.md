# CHAPITRE 19 : RETRIEVAL-AUGMENTED GENERATION (RAG)

## Introduction

Les LLMs ont une connaissance limitée:
- **Cutoff date**: connaissances gelées au moment du training
- **Hallucinations**: génèrent des faits plausibles mais faux
- **Pas de sources**: difficile de vérifier l'information
- **Domain-specific knowledge**: manque d'expertise dans des domaines spécialisés

**RAG (Retrieval-Augmented Generation)** résout ces problèmes en:
1. **Récupérant** des documents pertinents d'une base de connaissances
2. **Augmentant** le prompt avec ces documents
3. **Générant** une réponse basée sur les documents fournis

## 19.1 Architecture RAG : Vue d'ensemble

### 19.1.1 Pipeline RAG Complet

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
│                                                              │
│  1. INDEXING (Offline)                                       │
│  ┌────────────┐   ┌──────────┐   ┌──────────────┐         │
│  │ Documents  │──▶│ Chunking │──▶│  Embedding   │──┐      │
│  └────────────┘   └──────────┘   └──────────────┘  │      │
│                                                       ▼      │
│                                          ┌──────────────────┐│
│                                          │ Vector Database  ││
│                                          └────────┬─────────┘│
│  2. RETRIEVAL (Online)                            │          │
│  ┌────────────┐   ┌──────────┐                  │          │
│  │User Query  │──▶│ Embedding│───────────────────┘          │
│  └────────────┘   └──────────┘                 │            │
│                                                  ▼            │
│                                     ┌──────────────────────┐ │
│                                     │ Similarity Search    │ │
│                                     │ (Top-k retrieval)    │ │
│                                     └──────────┬───────────┘ │
│                                                 │             │
│  3. GENERATION                                 │             │
│  ┌──────────────┐                             │             │
│  │   Query +    │◀────────────────────────────┘             │
│  │   Retrieved  │                                            │
│  │   Docs       │                                            │
│  └──────┬───────┘                                            │
│         ▼                                                     │
│  ┌──────────────┐   ┌──────────────┐                       │
│  │   LLM        │──▶│   Response   │                       │
│  └──────────────┘   └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### 19.1.2 Implémentation Basique

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

class SimpleRAG:
    """
    Implémentation basique d'un système RAG
    """
    def __init__(self, documents, model_name="gpt-3.5-turbo"):
        self.documents = documents
        self.model_name = model_name

        # Setup components
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.llm = OpenAI(model_name=model_name)

        # Build vector store
        self.vectorstore = self._build_vectorstore()

        # Create RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

    def _build_vectorstore(self):
        """Construit la vector database"""
        # Split documents
        chunks = self.text_splitter.split_documents(self.documents)

        # Create embeddings et store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        return vectorstore

    def query(self, question):
        """
        Exécute une requête RAG

        Returns: (answer, source_documents)
        """
        result = self.qa_chain({"query": question})

        return result["result"], result["source_documents"]

# Usage
from langchain.document_loaders import TextLoader

# Charger documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# Créer RAG system
rag = SimpleRAG(documents)

# Query
answer, sources = rag.query("What is machine learning?")
print(f"Answer: {answer}")
print(f"\nSources:")
for i, doc in enumerate(sources):
    print(f"{i+1}. {doc.page_content[:100]}...")
```

## 19.2 Document Ingestion & Chunking

### 19.2.1 Chargement de Documents

**Supports multiples formats:**
```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    Docx2txtLoader,
)

def load_document(file_path):
    """
    Charge un document basé sur son extension
    """
    extension = file_path.split('.')[-1].lower()

    loaders = {
        'pdf': PyPDFLoader,
        'txt': TextLoader,
        'md': UnstructuredMarkdownLoader,
        'csv': CSVLoader,
        'html': UnstructuredHTMLLoader,
        'docx': Docx2txtLoader,
    }

    if extension not in loaders:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = loaders[extension](file_path)
    documents = loader.load()

    return documents

# Exemples
pdf_docs = load_document("research_paper.pdf")
txt_docs = load_document("notes.txt")
html_docs = load_document("webpage.html")
```

**Chargement de répertoires:**
```python
from langchain.document_loaders import DirectoryLoader

def load_directory(directory_path, glob_pattern="**/*.pdf"):
    """
    Charge tous les fichiers d'un répertoire
    """
    loader = DirectoryLoader(
        directory_path,
        glob=glob_pattern,
        show_progress=True,
        use_multithreading=True,
    )

    documents = loader.load()

    return documents

# Charger tous les PDFs
docs = load_directory("./knowledge_base/", glob_pattern="**/*.pdf")
print(f"Loaded {len(docs)} documents")
```

### 19.2.2 Stratégies de Chunking

Le chunking est **crucial** pour la performance RAG. Trop grand = contexte dilué, trop petit = perte de contexte.

**1. Fixed-Size Chunking**

Le plus simple: découper en taille fixe avec overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

def fixed_size_chunking(documents, chunk_size=1000, overlap=200):
    """
    Découpage à taille fixe
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separator="\n\n",  # Split sur paragraphes de préférence
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

# Usage
chunks = fixed_size_chunking(documents, chunk_size=1000, overlap=200)
```

**2. Recursive Character Text Splitting**

Plus sophistiqué: essaie de préserver la structure sémantique.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def semantic_chunking(documents, chunk_size=1000, overlap=200):
    """
    Découpage récursif préservant la structure

    Essaie de split sur (dans l'ordre):
    1. Paragraphes (\n\n)
    2. Lignes (\n)
    3. Phrases (. )
    4. Mots ( )
    5. Caractères
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

chunks = semantic_chunking(documents)
```

**3. Markdown/Code-Aware Splitting**

Pour documentation technique et code.

```python
from langchain.text_splitter import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)

def code_aware_chunking(code_text, language="python"):
    """
    Découpage respectant la structure du code
    """
    splitters = {
        "python": PythonCodeTextSplitter,
        "markdown": MarkdownTextSplitter,
    }

    splitter = splitters[language](
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(code_text)

    return chunks

# Usage
python_code = """
def calculate_sum(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0
    ...
"""

chunks = code_aware_chunking(python_code, language="python")
```

**4. Semantic Chunking (Advanced)**

Utilise embeddings pour découper là où le sens change.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def advanced_semantic_chunking(
    text,
    embeddings_model,
    similarity_threshold=0.7,
    min_chunk_size=200,
):
    """
    Découpage basé sur similarité sémantique

    Principe:
    1. Split en phrases
    2. Calculer embedding de chaque phrase
    3. Quand similarité < threshold, créer nouveau chunk
    """
    # Split en phrases
    sentences = text.split('. ')

    # Calculer embeddings
    embeddings = embeddings_model.embed_documents(sentences)

    # Détecter les ruptures sémantiques
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = [embeddings[0]]

    for i in range(1, len(sentences)):
        # Similarité avec chunk actuel
        avg_emb = np.mean(current_embedding, axis=0)
        sim = cosine_similarity([avg_emb], [embeddings[i]])[0][0]

        if sim < similarity_threshold and len(' '.join(current_chunk)) > min_chunk_size:
            # Rupture sémantique → nouveau chunk
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i]]
            current_embedding = [embeddings[i]]
        else:
            # Continuer chunk actuel
            current_chunk.append(sentences[i])
            current_embedding.append(embeddings[i])

    # Dernier chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
chunks = advanced_semantic_chunking(long_text, embeddings)
```

**5. Parent-Child Chunking**

Stocke de petits chunks pour retrieval, mais fournit le contexte parent au LLM.

```python
class ParentChildChunker:
    """
    Crée des chunks parent-enfant

    - Petits chunks (child): meilleur retrieval
    - Grands chunks (parent): meilleur contexte pour LLM
    """
    def __init__(self, parent_size=2000, child_size=400, overlap=100):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap,
        )

    def chunk(self, document):
        """
        Returns: [(child_chunk, parent_chunk), ...]
        """
        # Créer parents
        parent_chunks = self.parent_splitter.split_text(document)

        # Pour chaque parent, créer children
        all_pairs = []
        for parent in parent_chunks:
            children = self.child_splitter.split_text(parent)
            for child in children:
                all_pairs.append((child, parent))

        return all_pairs

chunker = ParentChildChunker()
pairs = chunker.chunk(long_document)

# Indexer children, mais stocker référence au parent
for child, parent in pairs:
    vectorstore.add_texts(
        texts=[child],
        metadatas=[{"parent": parent}]
    )
```

### 19.2.3 Stratégies de Chunking : Comparaison

| Méthode | Avantages | Inconvénients | Use Cases |
|---------|-----------|---------------|-----------|
| **Fixed-Size** | Simple, rapide | Coupe arbitrairement | Documents non-structurés |
| **Recursive** | Préserve structure | Plus complexe | Documentation, articles |
| **Semantic** | Cohérence sémantique | Lent, coûteux | Content de haute qualité |
| **Code-Aware** | Respect syntaxe | Spécifique langage | Code source, notebooks |
| **Parent-Child** | Meilleur des deux | Complexe, plus stockage | Knowledge bases |

**Benchmark empirique:**
```python
def benchmark_chunking_strategies(document, query, top_k=3):
    """
    Compare différentes stratégies de chunking
    """
    strategies = {
        "fixed": fixed_size_chunking,
        "recursive": semantic_chunking,
        "semantic": advanced_semantic_chunking,
    }

    results = {}

    for name, chunker in strategies.items():
        # Chunk
        chunks = chunker(document)

        # Build vectorstore
        vectorstore = Chroma.from_texts(chunks, embeddings)

        # Retrieve
        docs = vectorstore.similarity_search(query, k=top_k)

        # Measure relevance (manual or with LLM)
        relevance_score = evaluate_relevance(docs, query)

        results[name] = {
            "num_chunks": len(chunks),
            "avg_chunk_size": np.mean([len(c) for c in chunks]),
            "relevance": relevance_score,
        }

    return results
```

## 19.3 Embeddings & Vector Search

### 19.3.1 Modèles d'Embeddings

**Options populaires:**

```python
# 1. OpenAI Embeddings (proprietary)
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # ou text-embedding-3-large
)

# 2. Sentence Transformers (open-source)
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Cohere Embeddings
from langchain.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# 4. Local model (e.g., BAAI/bge-large)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)
```

**Comparaison des modèles:**

| Modèle | Dimension | Performance | Coût | Latence |
|--------|-----------|-------------|------|---------|
| **OpenAI text-embedding-3-small** | 1536 | Excellent | $0.02/1M tokens | Bas |
| **OpenAI text-embedding-3-large** | 3072 | Meilleur | $0.13/1M tokens | Moyen |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | Bon | Gratuit | Très bas |
| **BAAI/bge-large-en-v1.5** | 1024 | Excellent | Gratuit | Bas |
| **Cohere embed-english-v3.0** | 1024 | Excellent | $0.10/1M tokens | Bas |

**Benchmark personnalisé:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

def benchmark_embedding_models(texts, queries):
    """
    Compare vitesse et qualité de différents modèles
    """
    models = {
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "BGE-large": "BAAI/bge-large-en-v1.5",
        "MPNet": "sentence-transformers/all-mpnet-base-v2",
    }

    results = {}

    for name, model_name in models.items():
        print(f"Testing {name}...")

        # Load model
        model = SentenceTransformer(model_name)

        # Embed documents
        start = time.time()
        doc_embeddings = model.encode(texts, show_progress_bar=False)
        embed_time = time.time() - start

        # Embed queries
        query_embeddings = model.encode(queries, show_progress_bar=False)

        # Compute similarities
        similarities = cosine_similarity(query_embeddings, doc_embeddings)

        results[name] = {
            "embed_time": embed_time,
            "dim": doc_embeddings.shape[1],
            "avg_similarity": similarities.mean(),
        }

    return results

# Test
texts = ["Sample document 1", "Sample document 2", ...]
queries = ["query 1", "query 2"]

results = benchmark_embedding_models(texts, queries)
```

### 19.3.2 Vector Databases

**Choix de vector database:**

```python
# 1. Chroma (simple, local)
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 2. Pinecone (managed, scalable)
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

vectorstore = Pinecone.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# 3. Qdrant (open-source, production-ready)
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

vectorstore = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="my_collection"
)

# 4. Weaviate (GraphQL, multi-modal)
from langchain.vectorstores import Weaviate
import weaviate

client = weaviate.Client(url="http://localhost:8080")

vectorstore = Weaviate.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=client,
    by_text=False
)

# 5. FAISS (in-memory, très rapide)
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

**Comparaison vector databases:**

| Database | Type | Performance | Scalabilité | Filtrage | Use Case |
|----------|------|-------------|-------------|----------|----------|
| **Chroma** | Local | Moyen | Petite | Basique | Dev, prototyping |
| **FAISS** | In-memory | Très rapide | Moyenne | Limité | Haute performance |
| **Pinecone** | Managed | Rapide | Excellente | Avancé | Production |
| **Qdrant** | Self-hosted | Rapide | Excellente | Avancé | Production |
| **Weaviate** | Self-hosted | Rapide | Excellente | GraphQL | Multi-modal |
| **Milvus** | Self-hosted | Très rapide | Excellente | Avancé | Large scale |

### 19.3.3 Search Algorithms

**1. Similarity Search (basique)**

```python
def similarity_search(query, vectorstore, k=3):
    """
    Recherche les k documents les plus similaires
    """
    results = vectorstore.similarity_search(query, k=k)
    return results

# Usage
results = similarity_search("What is machine learning?", vectorstore)
```

**2. MMR (Maximal Marginal Relevance)**

Équilibre entre relevance et diversity.

```python
def mmr_search(query, vectorstore, k=3, fetch_k=20, lambda_mult=0.5):
    """
    MMR: sélectionne documents pertinents ET diversifiés

    lambda_mult=1: pure relevance
    lambda_mult=0: pure diversity
    """
    results = vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,  # Récupère plus de candidats
        lambda_mult=lambda_mult
    )
    return results

# Usage
results = mmr_search("machine learning", vectorstore, lambda_mult=0.7)
```

**3. Similarity Search with Score**

Obtenir les scores de similarité.

```python
def search_with_scores(query, vectorstore, k=3, score_threshold=0.7):
    """
    Recherche avec scores, filtre par threshold
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Filtrer par score
    filtered = [(doc, score) for doc, score in results if score >= score_threshold]

    return filtered

# Usage
results = search_with_scores("AI ethics", vectorstore, score_threshold=0.75)
for doc, score in results:
    print(f"Score: {score:.3f} | Content: {doc.page_content[:100]}...")
```

**4. Hybrid Search (Dense + Sparse)**

Combine vector search avec BM25 (keyword matching).

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

def hybrid_search(query, documents, vectorstore, k=3):
    """
    Hybrid search: vector similarity + keyword matching

    - Vector search: sémantique
    - BM25: keywords exacts
    """
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Ensemble (weighted combination)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 70% vector, 30% BM25
    )

    # Search
    results = ensemble_retriever.get_relevant_documents(query)

    return results

results = hybrid_search("machine learning algorithms", docs, vectorstore)
```

## 19.4 Re-ranking

Le retrieval initial peut manquer de précision. Le re-ranking affine les résultats.

### 19.4.1 Cross-Encoder Re-ranking

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Re-rank documents using cross-encoder
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=3):
        """
        Re-rank documents par relevance

        Returns: top_k documents re-ranked
        """
        # Préparer paires (query, doc)
        pairs = [[query, doc.page_content] for doc in documents]

        # Score avec cross-encoder
        scores = self.model.predict(pairs)

        # Trier par score décroissant
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Retourner top-k
        return scored_docs[:top_k]

# Usage dans RAG
reranker = Reranker()

# 1. Initial retrieval (top-20)
initial_docs = vectorstore.similarity_search(query, k=20)

# 2. Re-rank (keep top-3)
reranked_docs = reranker.rerank(query, initial_docs, top_k=3)

# 3. Use reranked docs for generation
for doc, score in reranked_docs:
    print(f"Score: {score:.3f} | {doc.page_content[:100]}...")
```

### 19.4.2 LLM-based Re-ranking

Utilise un LLM pour évaluer la pertinence.

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class LLMReranker:
    """
    Re-rank using LLM relevance scoring
    """
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""Given the query and document, rate the relevance on a scale of 0-10.
            Query: {query}
            Document: {document}

            Relevance score (0-10):""",
            input_variables=["query", "document"]
        )

    def rerank(self, query, documents, top_k=3):
        """Score each document with LLM"""
        scored_docs = []

        for doc in documents:
            # Generate prompt
            prompt_text = self.prompt.format(
                query=query,
                document=doc.page_content[:500]  # Limit size
            )

            # Get score from LLM
            response = self.llm(prompt_text)
            try:
                score = float(response.strip())
            except:
                score = 0.0

            scored_docs.append((doc, score))

        # Sort and return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

# Usage
llm = OpenAI(temperature=0)
reranker = LLMReranker(llm)

reranked = reranker.rerank(query, initial_docs, top_k=3)
```

## 19.5 Query Transformation

Améliorer la requête avant retrieval.

### 19.5.1 Query Expansion

```python
def query_expansion(query, llm):
    """
    Génère des variations de la query
    """
    prompt = f"""Given the question: "{query}"

    Generate 3 alternative phrasings that would help retrieve relevant information:

    1.
    2.
    3."""

    response = llm(prompt)

    # Parse expansions
    expansions = [query] + response.strip().split('\n')

    return expansions

# Usage
original_query = "How does attention work?"
expanded = query_expansion(original_query, llm)

# Retrieve pour chaque expansion
all_docs = []
for q in expanded:
    docs = vectorstore.similarity_search(q, k=2)
    all_docs.extend(docs)

# Deduplicate et rerank
unique_docs = list(set(all_docs))
final_docs = reranker.rerank(original_query, unique_docs, top_k=3)
```

### 19.5.2 Hypothetical Document Embeddings (HyDE)

Génère un document hypothétique, l'embed, et cherche des documents similaires.

```python
def hyde_retrieval(query, llm, vectorstore, k=3):
    """
    HyDE: génère document hypothétique pour meilleur retrieval
    """
    # Générer document hypothétique
    prompt = f"""Write a detailed passage that would answer this question:
    "{query}"

    Passage:"""

    hypothetical_doc = llm(prompt)

    # Embed et chercher documents similaires au doc hypothétique
    results = vectorstore.similarity_search(hypothetical_doc, k=k)

    return results

# Usage
results = hyde_retrieval(
    "What are the best practices for prompt engineering?",
    llm,
    vectorstore
)
```

---

*[Le chapitre continue avec Advanced RAG patterns, Agentic RAG, Evaluation, et cas pratiques...]*

*[Contenu total du Chapitre 19: ~70-80 pages]*
