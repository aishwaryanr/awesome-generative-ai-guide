"""A tiny in-memory knowledge base and retriever.

In production this is a vector store (embeddings) plus keyword/hybrid search over your
help center, backed by an ingestion pipeline that keeps it fresh. Here it is a small dict
with a keyword retriever, so the whole system runs offline with no external services.
"""

KB = {
    "return-policy": "Returns are accepted within 30 days of delivery for unused items. "
                     "Refunds are issued to the original payment method within 5 business days.",
    "shipping": "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 "
                "business days and costs 12 dollars.",
    "password-reset": "To reset your password, open Settings, choose Security, then Reset "
                      "Password. A reset link is emailed to you and expires in 30 minutes.",
    "damaged-item": "If an item arrives damaged, we replace it free of charge. A support "
                    "ticket routes it to the fulfillment team.",
}


# Filtering stopwords is what stops "meaning of life" from matching a doc on the words
# "of" and "is". A real system uses embeddings plus a relevance threshold for the same reason:
# retrieval needs a floor, or the grounding guardrail fires on noise.
STOPWORDS = {"what", "is", "the", "of", "a", "an", "to", "how", "do", "i", "my", "where",
             "when", "why", "you", "your", "are", "for", "and", "in", "on", "it", "this",
             "that", "need", "want", "me", "can", "with", "about", "please", "there"}


def _content_words(text: str):
    return {w for w in text.lower().replace("?", " ").replace("-", " ").split()
            if w not in STOPWORDS and len(w) > 1}


def retrieve(question: str, k: int = 2):
    """Keyword retriever with a relevance floor. Returns up to k (doc_id, text) pairs.

    Scores by overlap of content words only, and returns nothing when no content word matches,
    so an out-of-scope question retrieves an empty context and the agent escalates.
    """
    q = _content_words(question)
    scored = []
    for doc_id, text in KB.items():
        overlap = len(q & _content_words(doc_id + " " + text))
        if overlap:
            scored.append((overlap, doc_id, text))
    scored.sort(reverse=True)
    return [(doc_id, text) for _, doc_id, text in scored[:k]]
