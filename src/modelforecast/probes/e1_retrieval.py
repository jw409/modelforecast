"""E1 Retrieval: Can the model retrieve the right document for a query?

MTEB-inspired but opinionated: Tests retrieval quality for RAG/search use cases.
Uses real-world programming questions with obvious right/wrong document pairs.
"""

import math
import time
from typing import Any

from openai import OpenAI

from .base import EmbeddingResult


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vectors must have same length: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class E1RetrievalProbe:
    """
    E1 Retrieval: Query-Document Matching

    MTEB-inspired retrieval test, opinionated for RAG/search scenarios.
    Tests: Given a query, does the model correctly rank the relevant document
    above irrelevant ones?

    Test Design (coding domain - practical for agent use):
    - Query: "How do I handle async errors in Python?"
    - Doc A (relevant): Python try/except with asyncio explanation
    - Doc B (distractor): JavaScript Promise error handling
    - Doc C (irrelevant): Database connection pooling config

    Pass: similarity(Query, DocA) > similarity(Query, DocB) > similarity(Query, DocC)
    Fail: Wrong ranking - model confused languages or missed relevance

    This is harder than generic STS because:
    1. Query is a question, docs are answers (asymmetric)
    2. Distractor (DocB) has keyword overlap (errors, async) but wrong domain
    3. Tests practical RAG retrieval, not just paraphrase detection
    """

    def __init__(self):
        self.level = 1
        self.name = "Retrieval Quality"
        self.dimension = "E"  # Embedding dimension

        # MTEB-style retrieval: query -> documents ranking
        self.query = "How do I handle async errors in Python?"

        # Doc A: Correct answer (Python asyncio error handling)
        self.doc_relevant = """To handle async errors in Python, wrap your await calls in try/except blocks:

```python
async def fetch_data():
    try:
        result = await some_async_operation()
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
```

You can also use asyncio.gather with return_exceptions=True to collect errors without stopping."""

        # Doc B: Distractor - same keywords (async, errors) but JavaScript, not Python
        self.doc_distractor = """In JavaScript, handle async errors with .catch() or try/catch in async functions:

```javascript
async function fetchData() {
    try {
        const result = await fetch('/api/data');
    } catch (error) {
        console.error('Failed to fetch:', error);
    }
}

// Or with promises:
fetch('/api/data').catch(err => console.error(err));
```"""

        # Doc C: Irrelevant - different topic entirely
        self.doc_irrelevant = """Database connection pooling improves performance by reusing connections:

```yaml
pool:
  min_size: 5
  max_size: 20
  timeout: 30
```

Configure max_overflow for burst traffic. Monitor pool exhaustion metrics."""

        # Strict margin: relevant doc must beat distractor by significant margin
        # This is opinionated - we want CLEAR winners, not marginal ones
        self.min_margin_relevant_vs_distractor = 0.08
        self.min_margin_distractor_vs_irrelevant = 0.05

    def run(self, model: str, client: OpenAI) -> EmbeddingResult:
        """
        Execute the probe against the specified embedding model.

        Args:
            model: Embedding model identifier (e.g., "openai/text-embedding-3-small")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            EmbeddingResult with success status and retrieval ranking scores
        """
        try:
            start_time = time.time()

            # Get embeddings for query and all documents in one call (batch)
            response = client.embeddings.create(
                model=model,
                input=[self.query, self.doc_relevant, self.doc_distractor, self.doc_irrelevant],
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            raw_response = response.model_dump()

            # Check if embeddings were returned
            if not response.data or len(response.data) < 4:
                return EmbeddingResult(
                    success=False,
                    embedding_returned=False,
                    dimensions=None,
                    embedding=None,
                    raw_response=raw_response,
                    latency_ms=latency_ms,
                    error=f"Expected 4 embeddings, got {len(response.data) if response.data else 0}",
                )

            # Extract embeddings
            emb_query = response.data[0].embedding
            emb_relevant = response.data[1].embedding
            emb_distractor = response.data[2].embedding
            emb_irrelevant = response.data[3].embedding

            if not all([emb_query, emb_relevant, emb_distractor, emb_irrelevant]):
                return EmbeddingResult(
                    success=False,
                    embedding_returned=False,
                    dimensions=None,
                    embedding=None,
                    raw_response=raw_response,
                    latency_ms=latency_ms,
                    error="One or more embeddings empty",
                )

            dimensions = len(emb_query)

            # Calculate query-to-document similarities
            sim_relevant = cosine_similarity(emb_query, emb_relevant)
            sim_distractor = cosine_similarity(emb_query, emb_distractor)
            sim_irrelevant = cosine_similarity(emb_query, emb_irrelevant)

            # Pass criteria (opinionated, strict):
            # 1. Relevant doc must beat distractor by significant margin
            # 2. Distractor should still beat irrelevant (it has keyword overlap)
            margin_relevant_distractor = sim_relevant - sim_distractor
            margin_distractor_irrelevant = sim_distractor - sim_irrelevant

            # Both margins must meet thresholds
            ranking_correct = (
                margin_relevant_distractor >= self.min_margin_relevant_vs_distractor
                and margin_distractor_irrelevant >= self.min_margin_distractor_vs_irrelevant
            )

            # Also check basic ordering (relevant > distractor > irrelevant)
            ordering_correct = sim_relevant > sim_distractor > sim_irrelevant

            success = ranking_correct and ordering_correct

            # Store embedding sample (query, first 10 dims)
            embedding_sample = emb_query[:10] if len(emb_query) > 10 else emb_query

            return EmbeddingResult(
                success=success,
                embedding_returned=True,
                dimensions=dimensions,
                embedding=embedding_sample,
                raw_response={
                    # Don't include full response, just metrics
                    "model": model,
                    "sim_relevant": round(sim_relevant, 4),
                    "sim_distractor": round(sim_distractor, 4),
                    "sim_irrelevant": round(sim_irrelevant, 4),
                    "margin_relevant_vs_distractor": round(margin_relevant_distractor, 4),
                    "margin_distractor_vs_irrelevant": round(margin_distractor_irrelevant, 4),
                    "ordering_correct": ordering_correct,
                    "margins_sufficient": ranking_correct,
                },
                latency_ms=latency_ms,
                similarity_score=margin_relevant_distractor,  # Primary metric: how well it discriminates
            )

        except Exception as e:
            return EmbeddingResult(
                success=False,
                embedding_returned=False,
                dimensions=None,
                embedding=None,
                raw_response={},
                latency_ms=0,
                error=str(e),
            )

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"


# Keep backward compatibility alias
E1SimilarityProbe = E1RetrievalProbe
