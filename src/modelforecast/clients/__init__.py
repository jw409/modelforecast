"""Embedding, reranking, and API clients for local and remote services."""

from .local_embedding import (
    LocalEmbeddingClient,
    LocalRerankClient,
    LocalEmbeddingResponse,
    LocalEmbeddingData,
)
from .openrouter import (
    OpenRouterClient,
    ChatCompletionResponse,
    EmbeddingResponse,
    WitnessReport,
    APICall,
)

__all__ = [
    # Local embedding clients
    "LocalEmbeddingClient",
    "LocalRerankClient",
    "LocalEmbeddingResponse",
    "LocalEmbeddingData",
    # OpenRouter client with witness tracking
    "OpenRouterClient",
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "WitnessReport",
    "APICall",
]
