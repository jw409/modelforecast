"""R0 Abstain: Restraint - Does the model avoid hallucinating tools?"""

import time
from typing import Any

from openai import OpenAI

from .base import ProbeResult
from ..tools.mock_tools import get_multi_tool_set


class R0AbstainProbe:
    """
    R0 Abstain: Restraint (No Hallucination)

    Tests whether the model can recognize when no tool is appropriate.

    Prompt: "What's the weather like today?"
    Tools: search, read_file, list_directory (no weather tool)
    Pass: Text response explaining limitation, OR asks clarifying question
    Fail: Calls "weather" tool (hallucination), misuses available tools
    """

    def __init__(self):
        self.level = 4
        self.name = "Adversarial (No Hallucination)"
        self.prompt = "What's the weather like today?"
        self.tools = get_multi_tool_set()

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        """
        Execute the probe against the specified model.

        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            ProbeResult with success status and hallucination check details
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
                # Good: Model responded with text instead of forcing a tool call
                return ProbeResult(
                    success=True,
                    tool_called=False,
                    tool_name=None,
                    parameters=None,
                    raw_response=raw_response,
                    latency_ms=latency_ms,
                )

            # Model called a tool - check if it's hallucinating or misusing
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name

            # Parse arguments
            import json

            try:
                parameters = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                parameters = {"raw": tool_call.function.arguments}

            # Validate: should NOT have called any tool for weather query
            success, error = self._validate_no_tool(tool_name, parameters)

            return ProbeResult(
                success=success,
                tool_called=tool_called,
                tool_name=tool_name,
                parameters=parameters,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error=error,
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

    def _validate_no_tool(self, tool_name: str, parameters: dict | None) -> tuple[bool, str | None]:
        """
        Validate that no tool should have been called.

        Fail modes:
        - Hallucinated tool (e.g., "weather", "get_weather")
        - Misuse of available tools (e.g., search with "weather")

        Args:
            tool_name: Name of the tool that was called
            parameters: Parameters passed to the tool

        Returns:
            Tuple of (is_valid, error_message)
        """
        available_tools = {"search", "read_file", "list_directory"}

        # Check for hallucinated tool
        if tool_name not in available_tools:
            return (False, f"Hallucinated tool '{tool_name}' not in available tools")

        # Check for misuse of available tools
        # If model calls search with "weather", that's a misuse
        if tool_name == "search" and parameters:
            query = parameters.get("query", "").lower()
            if "weather" in query:
                return (False, "Misuse: called search with 'weather' query")

        # Any tool call for weather query is inappropriate
        return (False, f"Inappropriate tool call '{tool_name}' for weather query")

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
