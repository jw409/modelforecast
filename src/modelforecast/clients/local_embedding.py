"""Local embedding client for TalentOS localhost:8765 service.

Provides OpenAI-compatible interface for local embedding service.
"""

import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LocalEmbeddingData:
    """Mimics OpenAI embedding response structure."""
    embedding: list[float]
    index: int = 0
    object: str = "embedding"


@dataclass
class LocalEmbeddingResponse:
    """Mimics OpenAI embeddings.create() response."""
    data: list[LocalEmbeddingData]
    model: str
    usage: dict[str, int]

    def model_dump(self) -> dict[str, Any]:
        """Serialize to dict (OpenAI compatibility)."""
        return {
            "data": [{"embedding": d.embedding, "index": d.index, "object": d.object} for d in self.data],
            "model": self.model,
            "usage": self.usage,
        }


class LocalEmbeddingClient:
    """Client for localhost:8765 TalentOS embedding service.

    Provides OpenAI-compatible interface so existing probes work unchanged.

    Usage:
        client = LocalEmbeddingClient()
        # Use exactly like OpenAI client
        response = client.embeddings.create(model="qwen3_4b", input="hello")
    """

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0):
        """Initialize local embedding client.

        Args:
            base_url: Base URL of embedding service (default: localhost:8765)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.embeddings = LocalEmbeddingsAPI(self)

    def health_check(self) -> bool:
        """Check if local service is running."""
        try:
            with httpx.Client(timeout=5.0) as http:
                response = http.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


class LocalEmbeddingsAPI:
    """Embeddings API wrapper (mimics client.embeddings interface)."""

    def __init__(self, client: LocalEmbeddingClient):
        self._client = client

    def create(
        self,
        model: str,
        input: str | list[str],
        **kwargs,  # Ignore extra params for compatibility
    ) -> LocalEmbeddingResponse:
        """Create embeddings via local service.

        Args:
            model: Model name (e.g., "qwen3_4b")
            input: Single text or list of texts to embed

        Returns:
            LocalEmbeddingResponse compatible with OpenAI format
        """
        # Normalize input to list
        texts = [input] if isinstance(input, str) else input

        # Build request for local API
        # Local API uses: {"texts": [...], "model": "..."}
        payload = {
            "texts": texts,
            "model": model,
        }

        with httpx.Client(timeout=self._client.timeout) as http:
            response = http.post(
                f"{self._client.base_url}/embed",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        # Local service returns: {"embeddings": [[...], [...]], "model": "...", "dimensions": N}
        embeddings = data.get("embeddings", [])
        dimensions = data.get("dimensions", len(embeddings[0]) if embeddings else 0)

        # Convert to OpenAI-compatible format
        embedding_data = [
            LocalEmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ]

        return LocalEmbeddingResponse(
            data=embedding_data,
            model=model,
            usage={"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": sum(len(t.split()) for t in texts)},
        )


class LocalRerankClient:
    """Client for localhost:8765 TalentOS reranking service.

    Usage:
        client = LocalRerankClient()
        scores = client.rerank(query="...", documents=["...", "..."])
    """

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 30.0):
        """Initialize local rerank client.

        Args:
            base_url: Base URL of reranking service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "qwen3_reranker",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Rerank documents by relevance to query.

        Args:
            query: Query text
            documents: List of documents to rerank (max 3 per API spec)
            model: Reranker model name
            top_k: Optional limit on returned results

        Returns:
            Dict with 'results' containing ranked documents with scores
        """
        payload = {
            "query": query,
            "documents": documents,
            "model": model,
        }
        if top_k is not None:
            payload["top_k"] = top_k

        with httpx.Client(timeout=self.timeout) as http:
            response = http.post(
                f"{self.base_url}/rerank",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def health_check(self) -> bool:
        """Check if local service is running."""
        try:
            with httpx.Client(timeout=5.0) as http:
                response = http.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
