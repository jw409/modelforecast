"""Level 2: Tool Selection - Does the model choose the right tool?"""

import time
from typing import Any

from openai import OpenAI

from .base import ProbeResult
from ..tools.mock_tools import get_multi_tool_set


class Level2SelectionProbe:
    """
    Level 2: Tool Selection

    Tests whether the model can choose the appropriate tool given multiple options.

    Prompt: "I need to understand what the auth module does"
    Tools: search, read_file, list_directory
    Pass: Chooses 'search' (understanding requires finding relevant files first)
    Acceptable: 'list_directory' (looking for auth/ folder)
    Fail: Random selection, hallucinated tool, or no tool call
    """

    def __init__(self):
        self.level = 2
        self.name = "Tool Selection"
        self.prompt = "I need to understand what the auth module does"
        self.tools = get_multi_tool_set()

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        """
        Execute the probe against the specified model.

        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            ProbeResult with success status and tool selection details
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
                parameters = {"raw": tool_call.function.arguments}

            # Validate tool selection
            success = self._validate_selection(tool_name)

            return ProbeResult(
                success=success,
                tool_called=tool_called,
                tool_name=tool_name,
                parameters=parameters,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error=None if success else f"Inappropriate tool choice: {tool_name}",
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

    def _validate_selection(self, tool_name: str) -> bool:
        """
        Validate that the selected tool is appropriate for the task.

        For "I need to understand what the auth module does":
        - Best choice: 'search' (find files related to auth)
        - Acceptable: 'list_directory' (explore directory structure)
        - Wrong: 'read_file' (can't read a file without knowing which one)
        - Wrong: Hallucinated tool name

        Args:
            tool_name: Name of the tool that was called

        Returns:
            True if appropriate tool was selected
        """
        appropriate_tools = {"search", "list_directory"}
        return tool_name in appropriate_tools

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
