"""E2 Rerank: Can the reranker correctly order documents by relevance?

Tests reranking quality for RAG retrieval pipelines.
Uses same test scenario as E1 but with dedicated reranker model.
"""

import time
from dataclasses import dataclass
from typing import Any

from .base import EmbeddingResult


@dataclass
class RerankResult:
    """Result from running a rerank probe trial."""

    success: bool
    """Whether the probe passed its criteria."""

    ranking_correct: bool
    """Whether documents were ranked in correct order."""

    scores: list[float] | None
    """Relevance scores for each document."""

    raw_response: dict[str, Any]
    """Full API response from the reranker."""

    latency_ms: int
    """Time taken for the API call in milliseconds."""

    margin: float | None = None
    """Score margin between relevant and distractor docs."""

    error: str | None = None
    """Error message if the probe encountered an exception."""

    def __repr__(self) -> str:
        """Compact representation for logging."""
        status = "PASS" if self.success else "FAIL"
        margin_info = f", margin={self.margin:.3f}" if self.margin is not None else ""
        return f"RerankResult({status}{margin_info}, {self.latency_ms}ms)"


class E2RerankProbe:
    """
    E2 Rerank: Document Reranking Quality

    Tests whether a reranker model can correctly order documents by relevance.
    Uses same test scenario as E1 (Python vs JavaScript distractor).

    Pass criteria:
    1. Relevant doc (Python) ranked #1
    2. Distractor doc (JavaScript) ranked #2
    3. Irrelevant doc (Database) ranked #3
    4. Score margin between #1 and #2 >= threshold

    This tests reranker-specific capabilities that embedding models lack:
    - Cross-encoder attention between query and document
    - Fine-grained relevance scoring
    - Typically higher accuracy but slower than bi-encoder
    """

    def __init__(self):
        self.level = 2
        self.name = "Reranking Quality"
        self.dimension = "E"  # Embedding dimension (reranking is related)

        # Same test case as E1 for comparability
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

        # Minimum score margin between relevant and distractor
        # Rerankers should show CLEAR separation (higher than embedding margin)
        self.min_margin = 0.10

    def run(self, model: str, client: Any) -> RerankResult:
        """
        Execute the probe against the specified reranker.

        Args:
            model: Reranker model name (e.g., "qwen3_reranker")
            client: LocalRerankClient instance

        Returns:
            RerankResult with success status and ranking scores
        """
        try:
            start_time = time.time()

            # Call reranker
            documents = [self.doc_relevant, self.doc_distractor, self.doc_irrelevant]
            response = client.rerank(
                query=self.query,
                documents=documents,
                model=model,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract results
            # Expected format: {"results": [{"index": 0, "score": 0.95}, ...]}
            results = response.get("results", [])
            if not results:
                return RerankResult(
                    success=False,
                    ranking_correct=False,
                    scores=None,
                    raw_response=response,
                    latency_ms=latency_ms,
                    error="No results returned from reranker",
                )

            # Sort by score descending to get ranking
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

            # Extract scores in original document order
            scores_by_index = {r["index"]: r["score"] for r in results}
            scores = [scores_by_index.get(i, 0.0) for i in range(len(documents))]

            # Get ranking order (indices of docs in ranked order)
            ranking_order = [r["index"] for r in sorted_results]

            # Check if ranking is correct: [0, 1, 2] = relevant, distractor, irrelevant
            ranking_correct = ranking_order == [0, 1, 2]

            # Calculate margin between relevant and distractor
            score_relevant = scores_by_index.get(0, 0.0)
            score_distractor = scores_by_index.get(1, 0.0)
            margin = score_relevant - score_distractor

            # Success requires correct ranking AND sufficient margin
            success = ranking_correct and margin >= self.min_margin

            return RerankResult(
                success=success,
                ranking_correct=ranking_correct,
                scores=scores,
                raw_response={
                    "model": model,
                    "ranking_order": ranking_order,
                    "scores": {
                        "relevant": round(score_relevant, 4),
                        "distractor": round(score_distractor, 4),
                        "irrelevant": round(scores_by_index.get(2, 0.0), 4),
                    },
                    "margin_relevant_vs_distractor": round(margin, 4),
                },
                latency_ms=latency_ms,
                margin=margin,
            )

        except Exception as e:
            return RerankResult(
                success=False,
                ranking_correct=False,
                scores=None,
                raw_response={},
                latency_ms=0,
                error=str(e),
            )

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
