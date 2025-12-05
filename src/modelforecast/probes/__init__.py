"""Probe implementations for each capability dimension.

Dimensions:
- T (Tool Calling): T0 Invoke, T1 Schema, T2 Selection
- R (Restraint): R0 Abstain
- A (Agency): A1 Linear
"""

from .base import ProbeResult
from .t0_invoke import T0InvokeProbe
from .t1_schema import T1SchemaProbe
from .t2_selection import T2SelectionProbe
from .r0_abstain import R0AbstainProbe
from .a1_linear import A1LinearProbe

# Backwards compatibility aliases
Level0BasicProbe = T0InvokeProbe
Level1SchemaProbe = T1SchemaProbe
Level2SelectionProbe = T2SelectionProbe
Level3MultiTurnProbe = A1LinearProbe
Level4AdversarialProbe = R0AbstainProbe

__all__ = [
    "ProbeResult",
    # New names
    "T0InvokeProbe",
    "T1SchemaProbe",
    "T2SelectionProbe",
    "R0AbstainProbe",
    "A1LinearProbe",
    # Backwards compatibility
    "Level0BasicProbe",
    "Level1SchemaProbe",
    "Level2SelectionProbe",
    "Level3MultiTurnProbe",
    "Level4AdversarialProbe",
]
