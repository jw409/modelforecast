"""Probe implementations for each capability dimension.

Dimensions:
- T (Tool Calling): T0 Invoke, T1 Schema, T2 Selection
- R (Restraint): R0 Abstain
- A (Agency): A1 Linear
- E (Embedding): E0 Invoke, E1 Retrieval (MTEB-inspired)
"""

from .base import ProbeResult, EmbeddingResult
from .t0_invoke import T0InvokeProbe
from .t1_schema import T1SchemaProbe
from .t2_selection import T2SelectionProbe
from .r0_abstain import R0AbstainProbe
from .a1_linear import A1LinearProbe
from .e0_invoke import E0InvokeProbe
from .e1_retrieval import E1RetrievalProbe, E1SimilarityProbe

# Backwards compatibility aliases
Level0BasicProbe = T0InvokeProbe
Level1SchemaProbe = T1SchemaProbe
Level2SelectionProbe = T2SelectionProbe
Level3MultiTurnProbe = A1LinearProbe
Level4AdversarialProbe = R0AbstainProbe

__all__ = [
    "ProbeResult",
    "EmbeddingResult",
    # Tool Calling (T)
    "T0InvokeProbe",
    "T1SchemaProbe",
    "T2SelectionProbe",
    # Restraint (R)
    "R0AbstainProbe",
    # Agency (A)
    "A1LinearProbe",
    # Embedding (E) - MTEB-inspired
    "E0InvokeProbe",
    "E1RetrievalProbe",
    "E1SimilarityProbe",  # Backwards compatibility alias
    # Legacy names
    "Level0BasicProbe",
    "Level1SchemaProbe",
    "Level2SelectionProbe",
    "Level3MultiTurnProbe",
    "Level4AdversarialProbe",
]
