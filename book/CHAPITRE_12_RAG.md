# CHAPITRE 12 : RAG - RETRIEVAL-AUGMENTED GENERATION

> *« Un LLM seul hallucine. Un LLM avec RAG cite des sources. La différence entre créatif et fiable. »*

---

## Introduction : Résoudre le Problème de la Connaissance

### 🎭 Dialogue : Les Limites de la Mémoire

**Alice** : Bob, ChatGPT ne connaît rien après septembre 2021. Comment le rendre utile pour des infos récentes ?

**Bob** : Trois options :
1. **Fine-tuning** : Ré-entraîner sur nouvelles données → Coûteux ($10k+), lent
2. **Contexte dans prompt** : Copier-coller l'info → Limite de tokens (8k-32k)
3. **RAG** : Récupérer automatiquement l'info pertinente → Optimal !

**Alice** : C'est quoi exactement RAG ?

**Bob** : **Retrieval-Augmented Generation**. Le workflow :
```
1. Question utilisateur
   ↓
2. Recherche dans base documentaire (embedding similarity)
   ↓
3. Récupération top-k documents pertinents
   ↓
4. Injection dans prompt avec question
   ↓
5. LLM génère réponse basée sur documents
```

**Alice** : Avantages vs fine-tuning ?

**Bob** :
- ✅ **Pas de ré-entraînement** : Mise à jour = ajouter docs
- ✅ **Citations vérifiables** : LLM cite les sources
- ✅ **Moins d'hallucinations** : Grounded sur faits réels
- ✅ **Scalable** : Millions de documents possibles
- ✅ **Coût** : $100 setup vs $10k+ fine-tuning

### 📊 RAG vs Alternatives

| Approche | Coût Setup | MAJ Données | Hallucinations | Citations | Scalabilité |
|----------|-----------|-------------|----------------|-----------|-------------|
| **Prompt seul** | $0 | N/A | Élevées | ❌ | Limite tokens |
| **Fine-tuning** | $10k+ | Ré-entraîner | Moyennes | ❌ | Coûteux |
| **RAG** | $100-1k | Ajouter docs | Faibles | ✅ | Excellente |
| **Hybrid** | $10k+ | Mix | Très faibles | ✅ | Excellente |

### 🎯 Anecdote : Facebook RAG (2020)

**Septembre 2020, Meta AI**

Lewis et al. publient "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks".

**Innovation** : Combiner dense retrieval (DPR) + génération (BART) end-to-end.

**Résultats** :
- Natural Questions : 44.5% → **56.8%** (+12.3 points)
- TriviaQA : 68.0% → **68.0%** (équivalent avec 10× moins params)

**Impact** : RAG devient l'architecture standard pour chatbots d'entreprise, assistants documentaires, etc.

**Aujourd'hui** : ChatGPT Plugins, Bing Chat, Perplexity.ai utilisent tous RAG.

### 🎯 Objectifs du Chapitre

À la fin de ce chapitre, vous saurez :

- ✅ Architecturer un système RAG complet
- ✅ Créer et gérer une vector database
- ✅ Implémenter retrieval avec embeddings
- ✅ Optimiser la récupération (reranking, hybrid search)
- ✅ Construire un chatbot RAG production-ready
- ✅ Évaluer et debugger RAG
- ✅ Gérer les cas limites (multi-hop, contradictions)

**Difficulté** : 🔴🔴🔴⚪⚪ (Avancé)
**Prérequis** : Embeddings (Ch. 3), Prompt Engineering (Ch. 11), bases de données
**Temps de lecture** : ~120 minutes

---

## Architecture RAG : Les Composants

### Pipeline Complet

```
┌─────────────────────────────────────────────────────┐
│                 INDEXATION (Offline)                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Documents → Chunking → Embeddings → Vector DB     │
│                                                     │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               QUERY TIME (Online)                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Question → Embedding → Similarity Search           │
│      ↓                                              │
│  Top-k Docs → Prompt Construction → LLM → Answer   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1. Document Chunking

**Problème** : Documents longs (100 pages) ne rentrent pas dans contexte LLM.

**Solution** : Découper en chunks gérables.

```python
from typing import List

class DocumentChunker:
    """
    Découpe documents en chunks.
    """
    def __init__(self, chunk_size=512, overlap=50):
        """
        Args:
            chunk_size: Taille max d'un chunk (en tokens)
            overlap: Overlap entre chunks consécutifs
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_by_tokens(self, text: str, tokenizer) -> List[str]:
        """
        Chunking par tokens.
        """
        # Tokeniser
        tokens = tokenizer.encode(text)

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            # Décoder en texte
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Avancer avec overlap
            start += (self.chunk_size - self.overlap)

        return chunks

    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Chunking par phrases (préserve sémantique).
        """
        import nltk
        nltk.download('punkt', quiet=True)

        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.chunk_size:
                # Sauvegarder chunk actuel
                chunks.append(' '.join(current_chunk))

                # Nouveau chunk avec overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Dernier chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_by_semantic(self, text: str, model) -> List[str]:
        """
        Chunking sémantique (préserve cohérence thématique).
        """
        sentences = nltk.sent_tokenize(text)

        # Embeddings de chaque phrase
        embeddings = model.encode(sentences)

        # Détection de ruptures sémantiques
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            for i in range(len(embeddings)-1)
        ]

        # Seuil de rupture
        threshold = 0.7
        breakpoints = [0] + [i+1 for i, sim in enumerate(similarities) if sim < threshold] + [len(sentences)]

        # Créer chunks
        chunks = []
        for i in range(len(breakpoints)-1):
            start, end = breakpoints[i], breakpoints[i+1]
            chunk = ' '.join(sentences[start:end])
            chunks.append(chunk)

        return chunks

# Utilisation
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
chunker = DocumentChunker(chunk_size=512, overlap=50)

document = """
Long document here...
Multiple paragraphs...
Etc.
"""

# Chunking par tokens
chunks_tokens = chunker.chunk_by_tokens(document, tokenizer)

# Chunking par phrases
chunks_sentences = chunker.chunk_by_sentences(document)

print(f"Nombre de chunks: {len(chunks_tokens)}")
```

### 2. Embedding Generation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    """
    Génère embeddings pour documents et queries.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Modèle Sentence-BERT
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents en embeddings.

        Returns:
            embeddings: [num_docs, embedding_dim]
        """
        embeddings = self.model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode une query.

        Returns:
            embedding: [embedding_dim]
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding

# Utilisation
embedder = EmbeddingGenerator()

# Embeddings des chunks
chunks = ["chunk 1 text", "chunk 2 text", ...]
chunk_embeddings = embedder.embed_documents(chunks)

print(f"Embedding shape: {chunk_embeddings.shape}")  # [num_chunks, 384]
```

### 3. Vector Database

```python
import faiss
import pickle

class VectorStore:
    """
    Vector database avec FAISS.
    """
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.documents = []

    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """
        Ajoute documents à l'index.
        """
        # Vérifier dimensions
        assert embeddings.shape[1] == self.embedding_dim

        # Ajouter à FAISS
        self.index.add(embeddings.astype('float32'))

        # Stocker documents
        self.documents.extend(documents)

        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")

    def search(self, query_embedding: np.ndarray, top_k=5):
        """
        Recherche top-k documents similaires.

        Returns:
            documents: Liste de (document, score) tuples
        """
        # FAISS search
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        # Récupérer documents
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = 1 / (1 + distances[0][i])  # Convertir distance en similarité
                results.append((doc, score))

        return results

    def save(self, path):
        """Sauvegarde l'index."""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)

    def load(self, path):
        """Charge l'index."""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)

# Utilisation
vector_store = VectorStore(embedding_dim=384)

# Ajouter documents
vector_store.add_documents(chunks, chunk_embeddings)

# Recherche
query = "What is machine learning?"
query_embedding = embedder.embed_query(query)
results = vector_store.search(query_embedding, top_k=3)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Document: {doc[:100]}...\n")
```

---

## RAG Complet : Implémentation

### Classe RAG Pipeline

```python
from openai import OpenAI

class RAGPipeline:
    """
    Pipeline RAG complet.
    """
    def __init__(self, vector_store, embedder, llm_model="gpt-4"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_model = llm_model
        self.client = OpenAI()

    def retrieve(self, query: str, top_k=3):
        """
        Récupère documents pertinents.
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results

    def generate(self, query: str, context_docs: List[tuple]):
        """
        Génère réponse avec LLM.
        """
        # Construire contexte
        context = "\n\n".join([
            f"Document {i+1} (score: {score:.2f}):\n{doc}"
            for i, (doc, score) in enumerate(context_docs)
        ])

        # Prompt
        prompt = f"""
Réponds à la question suivante en te basant UNIQUEMENT sur les documents fournis.
Si l'information n'est pas dans les documents, dis "Je ne trouve pas l'information dans les documents fournis."

Documents:
{context}

Question: {query}

Réponse:"""

        # Appel LLM
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "Tu es un assistant qui répond basé sur des documents fournis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content
        return answer

    def query(self, query: str, top_k=3, return_sources=True):
        """
        Pipeline complet : retrieve + generate.
        """
        # Retrieval
        context_docs = self.retrieve(query, top_k=top_k)

        # Generation
        answer = self.generate(query, context_docs)

        if return_sources:
            return {
                "answer": answer,
                "sources": [{"text": doc, "score": score} for doc, score in context_docs]
            }
        else:
            return answer

# Utilisation
rag = RAGPipeline(vector_store, embedder, llm_model="gpt-4")

# Query
result = rag.query("What are the main benefits of transformers?", top_k=3)

print("Answer:", result["answer"])
print("\nSources:")
for i, source in enumerate(result["sources"], 1):
    print(f"{i}. (Score: {source['score']:.3f}) {source['text'][:100]}...")
```

---

## Optimisations Avancées

### 1. Hybrid Search (Dense + Sparse)

**Principe** : Combiner embeddings (dense) + BM25 (sparse).

```python
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Hybrid retrieval : Dense (embeddings) + Sparse (BM25).
    """
    def __init__(self, vector_store, embedder, documents, alpha=0.5):
        """
        Args:
            alpha: Poids dense vs sparse (0.5 = égal)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.documents = documents
        self.alpha = alpha

        # BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query: str, top_k=5):
        """
        Hybrid retrieval.
        """
        # Dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=top_k*2)

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normaliser scores
        dense_scores = {i: score for i, (doc, score) in enumerate(dense_results)}
        bm25_scores_dict = {i: score for i, score in enumerate(bm25_scores)}

        # Normalisation min-max
        def normalize(scores_dict):
            if not scores_dict:
                return {}
            min_score = min(scores_dict.values())
            max_score = max(scores_dict.values())
            if max_score == min_score:
                return {k: 0.5 for k in scores_dict}
            return {k: (v - min_score) / (max_score - min_score) for k, v in scores_dict.items()}

        dense_norm = normalize(dense_scores)
        bm25_norm = normalize(bm25_scores_dict)

        # Combiner scores
        hybrid_scores = {}
        all_indices = set(dense_norm.keys()) | set(bm25_norm.keys())

        for idx in all_indices:
            dense_score = dense_norm.get(idx, 0)
            bm25_score = bm25_norm.get(idx, 0)
            hybrid_scores[idx] = self.alpha * dense_score + (1 - self.alpha) * bm25_score

        # Top-k
        top_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = [(self.documents[idx], score) for idx, score in top_indices]
        return results

# Utilisation
hybrid_retriever = HybridRetriever(vector_store, embedder, chunks, alpha=0.5)
results = hybrid_retriever.retrieve("machine learning applications", top_k=3)
```

### 2. Reranking

**Principe** : Récupérer top-100 avec retrieval rapide, puis reranker avec modèle cross-encoder.

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Reranking avec cross-encoder.
    """
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[str], top_k=5):
        """
        Rerank documents.
        """
        # Créer paires (query, doc)
        pairs = [(query, doc) for doc in documents]

        # Score avec cross-encoder
        scores = self.model.predict(pairs)

        # Trier
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

# Pipeline avec reranking
class RAGWithReranking(RAGPipeline):
    """
    RAG avec reranking.
    """
    def __init__(self, vector_store, embedder, llm_model="gpt-4"):
        super().__init__(vector_store, embedder, llm_model)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k=3, initial_k=20):
        """
        Retrieve avec reranking.
        """
        # Retrieval initial (large)
        query_embedding = self.embedder.embed_query(query)
        initial_results = self.vector_store.search(query_embedding, top_k=initial_k)

        # Extraire documents
        docs = [doc for doc, score in initial_results]

        # Rerank
        reranked = self.reranker.rerank(query, docs, top_k=top_k)

        return reranked

# Utilisation
rag_rerank = RAGWithReranking(vector_store, embedder)
result = rag_rerank.query("What is attention mechanism?")
```

### 3. Query Expansion

**Principe** : Générer plusieurs variations de la query pour améliorer recall.

```python
class QueryExpander:
    """
    Expansion de queries avec LLM.
    """
    def __init__(self, client):
        self.client = client

    def expand(self, query: str, num_variations=3):
        """
        Génère variations de la query.
        """
        prompt = f"""
Génère {num_variations} reformulations de la question suivante pour améliorer la recherche documentaire.
Les reformulations doivent capturer différents aspects de la question.

Question originale: {query}

Reformulations:
1."""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        variations_text = response.choices[0].message.content
        variations = [query] + [line.strip() for line in variations_text.split('\n') if line.strip()]

        return variations[:num_variations+1]

# Pipeline avec expansion
class RAGWithExpansion(RAGPipeline):
    """
    RAG avec query expansion.
    """
    def __init__(self, vector_store, embedder, llm_model="gpt-4"):
        super().__init__(vector_store, embedder, llm_model)
        self.expander = QueryExpander(self.client)

    def retrieve(self, query: str, top_k=3):
        """
        Retrieve avec expansion.
        """
        # Expand query
        queries = self.expander.expand(query, num_variations=2)

        # Retrieve pour chaque variation
        all_results = []
        for q in queries:
            query_embedding = self.embedder.embed_query(q)
            results = self.vector_store.search(query_embedding, top_k=top_k)
            all_results.extend(results)

        # Déduplication et tri
        seen = set()
        unique_results = []
        for doc, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if doc not in seen:
                seen.add(doc)
                unique_results.append((doc, score))

        return unique_results[:top_k]
```

---

## Chatbot RAG avec Mémoire

### Conversation Multi-Turn

```python
class ConversationalRAG:
    """
    RAG avec historique de conversation.
    """
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.history = []

    def query(self, user_message: str, top_k=3):
        """
        Query avec contexte conversationnel.
        """
        # Réécrire query avec contexte si nécessaire
        if self.history:
            rewritten_query = self._rewrite_query(user_message)
        else:
            rewritten_query = user_message

        # RAG
        result = self.rag.query(rewritten_query, top_k=top_k)

        # Ajouter à historique
        self.history.append({
            "user": user_message,
            "assistant": result["answer"],
            "sources": result["sources"]
        })

        return result

    def _rewrite_query(self, current_query: str):
        """
        Réécrire query en intégrant contexte conversationnel.
        """
        # Contexte des N derniers échanges
        context = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in self.history[-3:]  # 3 derniers tours
        ])

        prompt = f"""
Contexte de la conversation:
{context}

Question actuelle: {current_query}

Réécrire la question actuelle de manière autonome (standalone) en intégrant le contexte si nécessaire.
Si la question est déjà autonome, la retourner telle quelle.

Question réécrite:"""

        response = self.rag.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        rewritten = response.choices[0].message.content.strip()
        return rewritten

    def clear_history(self):
        """Réinitialise l'historique."""
        self.history = []

# Utilisation
conv_rag = ConversationalRAG(rag)

# Tour 1
result1 = conv_rag.query("What is a transformer?")
print("Answer 1:", result1["answer"])

# Tour 2 (référence au tour 1)
result2 = conv_rag.query("How does it compare to RNNs?")
# "it" sera résolu en "transformer" grâce au contexte
print("Answer 2:", result2["answer"])
```

---

## Évaluation de RAG

### Métriques

```python
class RAGEvaluator:
    """
    Évalue un système RAG.
    """
    def __init__(self, test_set):
        """
        Args:
            test_set: Liste de {"question": ..., "answer": ..., "context": ...}
        """
        self.test_set = test_set

    def evaluate_retrieval(self, rag_pipeline):
        """
        Évalue qualité du retrieval.

        Métriques: Recall@k, MRR
        """
        recalls = []
        mrrs = []

        for example in self.test_set:
            query = example["question"]
            relevant_context = example["context"]

            # Retrieve
            retrieved = rag_pipeline.retrieve(query, top_k=5)
            retrieved_texts = [doc for doc, score in retrieved]

            # Recall@k : % de contexte pertinent récupéré
            recall = self._compute_recall(relevant_context, retrieved_texts)
            recalls.append(recall)

            # MRR : Mean Reciprocal Rank
            mrr = self._compute_mrr(relevant_context, retrieved_texts)
            mrrs.append(mrr)

        return {
            "recall@5": np.mean(recalls),
            "mrr": np.mean(mrrs)
        }

    def _compute_recall(self, relevant, retrieved):
        """Calcule recall."""
        # Simplifié : vérifier si texte pertinent dans top-k
        for r in retrieved:
            if relevant in r or r in relevant:
                return 1.0
        return 0.0

    def _compute_mrr(self, relevant, retrieved):
        """Mean Reciprocal Rank."""
        for i, r in enumerate(retrieved, 1):
            if relevant in r or r in relevant:
                return 1.0 / i
        return 0.0

    def evaluate_generation(self, rag_pipeline):
        """
        Évalue qualité de la génération.

        Métriques: Accuracy, F1, ROUGE
        """
        from sklearn.metrics import f1_score
        from rouge import Rouge

        predictions = []
        references = []

        for example in self.test_set:
            query = example["question"]
            expected_answer = example["answer"]

            # Generate
            result = rag_pipeline.query(query, return_sources=False)

            predictions.append(result)
            references.append(expected_answer)

        # ROUGE scores
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)

        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"]
        }

# Utilisation
test_set = [
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of AI...",
        "context": "Machine learning (ML) is a field of study..."
    },
    # ... plus d'exemples
]

evaluator = RAGEvaluator(test_set)

# Évaluer retrieval
retrieval_metrics = evaluator.evaluate_retrieval(rag)
print("Retrieval:", retrieval_metrics)

# Évaluer generation
generation_metrics = evaluator.evaluate_generation(rag)
print("Generation:", generation_metrics)
```

---

## Problèmes Courants et Solutions

### 1. Retrieval Échoue (Rien de Pertinent)

**Causes** :
- Embeddings de mauvaise qualité
- Chunking trop grossier/fin
- Query mal formulée

**Solutions** :
```python
# A) Améliorer embeddings
embedder = EmbeddingGenerator("sentence-transformers/all-mpnet-base-v2")  # Meilleur modèle

# B) Chunking adaptatif
chunker = DocumentChunker(chunk_size=256, overlap=50)  # Plus petit

# C) Query expansion
expanded_queries = expander.expand(query)
```

### 2. Contradictions entre Sources

**Solution** : Synthèse multi-documents

```python
SYNTHESIS_PROMPT = """
Les documents suivants contiennent des informations potentiellement contradictoires:

Document 1: {doc1}
Document 2: {doc2}

Question: {query}

Tâche:
1. Identifier les contradictions
2. Expliquer les différences (dates, contexte, etc.)
3. Fournir une réponse nuancée

Réponse:"""
```

### 3. Hallucinations malgré RAG

**Solution** : Contrainte stricte + vérification

```python
STRICT_RAG_PROMPT = """
RÈGLE ABSOLUE: Réponds UNIQUEMENT avec des informations présentes dans les documents.
Si tu dois inventer ou inférer, commence par "D'après mon raisonnement général..."

Documents:
{context}

Question: {query}

Réponse (avec citations):"""
```

---

## 💡 Analogie : RAG comme une Bibliothèque Intelligente

- **Vector DB** = Catalogue (index par thème, auteur)
- **Embeddings** = Système de classification sémantique
- **Retrieval** = Bibliothécaire qui trouve les livres pertinents
- **LLM** = Expert qui lit les livres et répond à votre question
- **Reranking** = Trier les livres du plus au moins pertinent
- **Hybrid Search** = Chercher par titre (BM25) ET par sujet (embeddings)

RAG transforme un LLM généraliste en expert de VOTRE domaine avec VOTRE documentation.

---

## Conclusion

### 🎭 Dialogue Final : RAG, Futur des Applications LLM

**Alice** : RAG semble être LA solution pour la plupart des cas d'usage !

**Bob** : Exactement. En 2024, **80% des applications LLM en entreprise** utilisent RAG :
- Chatbots documentaires (support client, RH)
- Assistants de code (sur codebase privée)
- Analyse de rapports (finance, légal)
- FAQ intelligentes

**Alice** : Limites ?

**Bob** :
- **Multi-hop reasoning** : Questions nécessitant plusieurs documents
- **Informations numériques** : LLM fait des erreurs de calcul
- **Mise à jour temps réel** : Latence indexation

**Alice** : Solutions ?

**Bob** :
- **Multi-hop** : Chain-of-Thought + retrieval itératif
- **Calculs** : Intégrer calculatrice/Python
- **Temps réel** : Streaming ingestion

**Alice** : Coût ?

**Bob** : Setup RAG typique :
- Vector DB (Pinecone/Weaviate) : $100-500/mois
- Embeddings : $0.0001/1k tokens
- LLM calls : $0.03/1k tokens (GPT-4)
- **Total** : $500-2000/mois pour PME

Le futur : **RAG multimodal** (texte + images + tables + graphiques).

---

## Ressources

### 📚 Papers Fondamentaux

1. **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"** (Lewis et al., 2020)
2. **"Dense Passage Retrieval for Open-Domain Question Answering"** (Karpukhin et al., 2020)
3. **"REALM: Retrieval-Augmented Language Model Pre-Training"** (Guu et al., 2020)

### 🛠️ Bibliothèques

```bash
# RAG frameworks
pip install langchain llama-index

# Vector databases
pip install faiss-cpu chromadb pinecone-client weaviate-client

# Embeddings
pip install sentence-transformers

# Retrieval
pip install rank-bm25
```

### 🔗 Ressources

- **LangChain RAG** : https://python.langchain.com/docs/use_cases/question_answering/
- **LlamaIndex** : https://docs.llamaindex.ai/
- **Pinecone Learning Center** : https://www.pinecone.io/learn/
- **Weaviate Docs** : https://weaviate.io/developers/weaviate

---

**🎓 Bravo !** Vous maîtrisez maintenant RAG, l'architecture qui rend les LLMs utiles en production. Vous avez maintenant une base solide couvrant 12 chapitres majeurs du livre ! 🚀

