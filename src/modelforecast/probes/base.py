"""Base classes and types for probe implementations."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProbeResult:
    """Result from running a single probe trial."""

    success: bool
    """Whether the probe passed its criteria."""

    tool_called: bool
    """Whether any tool was called (vs text-only response)."""

    tool_name: str | None
    """Name of the tool that was called, if any."""

    parameters: dict[str, Any] | None
    """Parameters passed to the tool, if any."""

    raw_response: dict[str, Any]
    """Full API response from the model."""

    latency_ms: int
    """Time taken for the API call in milliseconds."""

    error: str | None = None
    """Error message if the probe encountered an exception."""

    def __repr__(self) -> str:
        """Compact representation for logging."""
        status = "PASS" if self.success else "FAIL"
        tool_info = f"{self.tool_name}({self.parameters})" if self.tool_called else "no_tool"
        return f"ProbeResult({status}, {tool_info}, {self.latency_ms}ms)"
