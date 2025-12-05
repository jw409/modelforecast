"""E0 Invoke: Basic Embedding - Can the model produce embeddings at all?"""

import time
from typing import Any

from openai import OpenAI

from .base import EmbeddingResult


class E0InvokeProbe:
    """
    E0 Invoke: Basic Embedding

    Tests whether the embedding model can produce a vector embedding at all.

    Input: "The quick brown fox jumps over the lazy dog"
    Pass: Response contains embedding array with > 0 dimensions
    Fail: Error, empty response, or malformed data
    """

    def __init__(self):
        self.level = 0
        self.name = "Basic Embedding"
        self.dimension = "E"  # Embedding dimension
        self.input_text = "The quick brown fox jumps over the lazy dog"

    def run(self, model: str, client: OpenAI) -> EmbeddingResult:
        """
        Execute the probe against the specified embedding model.

        Args:
            model: Embedding model identifier (e.g., "openai/text-embedding-3-small")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            EmbeddingResult with success status and metadata
        """
        try:
            start_time = time.time()

            response = client.embeddings.create(
                model=model,
                input=self.input_text,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            raw_response = response.model_dump()

            # Check if embedding was returned
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                embedding_returned = embedding is not None and len(embedding) > 0

                if embedding_returned:
                    dimensions = len(embedding)
                    # Store first 10 elements for verification (not full vector)
                    embedding_sample = embedding[:10] if len(embedding) > 10 else embedding

                    return EmbeddingResult(
                        success=True,
                        embedding_returned=True,
                        dimensions=dimensions,
                        embedding=embedding_sample,
                        raw_response=raw_response,
                        latency_ms=latency_ms,
                    )

            # No embedding returned
            return EmbeddingResult(
                success=False,
                embedding_returned=False,
                dimensions=None,
                embedding=None,
                raw_response=raw_response,
                latency_ms=latency_ms,
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
