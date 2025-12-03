"""Level 1: Schema Compliance - Does the model respect parameter types?"""

import time
from typing import Any

from openai import OpenAI

from .base import ProbeResult
from ..tools.mock_tools import get_search_tool_with_limit


class Level1SchemaProbe:
    """
    Level 1: Schema Compliance

    Tests whether the model respects parameter types and required fields.

    Prompt: "Search for authentication files, limit results to 5"
    Tools: search(query: string, limit: integer)
    Pass: Tool called with correct types (query is string, limit is integer)
    Fail: Wrong types (limit="5" instead of limit=5), missing required fields
    """

    def __init__(self):
        self.level = 1
        self.name = "Schema Compliance"
        self.prompt = "Search for authentication files, limit results to 5"
        self.tools = [get_search_tool_with_limit()]

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        """
        Execute the probe against the specified model.

        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            ProbeResult with success status and schema validation details
        """
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.prompt}],
                tools=self.tools,
                temperature=0.1,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            raw_response = response.model_dump()
            choice = response.choices[0]
            message = choice.message

            # Check if tool was called
            tool_called = message.tool_calls is not None and len(message.tool_calls) > 0

            if not tool_called:
                return ProbeResult(
                    success=False,
                    tool_called=False,
                    tool_name=None,
                    parameters=None,
                    raw_response=raw_response,
                    latency_ms=latency_ms,
                    error="No tool called",
                )

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name

            # Parse arguments
            import json

            try:
                parameters = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                return ProbeResult(
                    success=False,
                    tool_called=True,
                    tool_name=tool_name,
                    parameters=None,
                    raw_response=raw_response,
                    latency_ms=latency_ms,
                    error="Malformed JSON arguments",
                )

            # Validate schema compliance
            success, error = self._validate_schema(parameters)

            return ProbeResult(
                success=success,
                tool_called=tool_called,
                tool_name=tool_name,
                parameters=parameters,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error=error if not success else None,
            )

        except Exception as e:
            return ProbeResult(
                success=False,
                tool_called=False,
                tool_name=None,
                parameters=None,
                raw_response={},
                latency_ms=0,
                error=str(e),
            )

    def _validate_schema(self, parameters: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Validate that parameters match the expected schema.

        Pass criteria:
        - 'query' parameter present and is a string
        - 'limit' parameter (if present) is an integer (not string "5")
        - No hallucinated extra fields that would break the tool

        Args:
            parameters: Parsed parameters from tool call

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required field: query
        if "query" not in parameters:
            return (False, "Missing required field 'query'")

        # Check query is string
        if not isinstance(parameters["query"], str):
            return (False, f"Field 'query' must be string, got {type(parameters['query']).__name__}")

        # Check limit type if present
        if "limit" in parameters:
            if not isinstance(parameters["limit"], int):
                return (
                    False,
                    f"Field 'limit' must be integer, got {type(parameters['limit']).__name__}",
                )

        # All checks passed
        return (True, None)

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
