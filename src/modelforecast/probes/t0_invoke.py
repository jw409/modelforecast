"""T0 Invoke: Basic Tool Calling - Can the model produce a tool_call at all?"""

import time
from typing import Any

from openai import OpenAI

from .base import ProbeResult
from ..tools.mock_tools import get_search_tool_basic


class T0InvokeProbe:
    """
    T0 Invoke: Basic Tool Calling

    Tests whether the model can produce a tool_call response at all.

    Prompt: "Use the search tool to find files containing 'authentication'"
    Tools: search(query: string)
    Pass: Response contains tool_calls array with at least one entry
    Fail: Text response, empty tool_calls, or malformed JSON
    """

    def __init__(self):
        self.level = 0
        self.name = "Basic Tool Calling"
        self.prompt = "Use the search tool to find files containing 'authentication'"
        self.tools = [get_search_tool_basic()]

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        """
        Execute the probe against the specified model.

        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            ProbeResult with success status and metadata
        """
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": self.prompt}],
                tools=self.tools,
                temperature=0.1,  # Low temperature for consistency
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract response data
            raw_response = response.model_dump()
            choice = response.choices[0]
            message = choice.message

            # Check if tool was called
            tool_called = message.tool_calls is not None and len(message.tool_calls) > 0

            if tool_called:
                tool_call = message.tool_calls[0]
                tool_name = tool_call.function.name
                # Parse arguments (may be JSON string)
                import json

                try:
                    parameters = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    parameters = {"raw": tool_call.function.arguments}

                # Pass criteria: tool_calls array has at least one entry
                success = True
            else:
                tool_name = None
                parameters = None
                success = False

            return ProbeResult(
                success=success,
                tool_called=tool_called,
                tool_name=tool_name,
                parameters=parameters,
                raw_response=raw_response,
                latency_ms=latency_ms,
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

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
