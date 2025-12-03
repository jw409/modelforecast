"""Level 3: Multi-Turn Coherence - Can the model follow up appropriately?"""

import json
import time
from typing import Any

from openai import OpenAI

from .base import ProbeResult
from ..tools.mock_tools import get_multi_tool_set


class Level3MultiTurnProbe:
    """
    Level 3: Multi-Turn Coherence

    Tests whether the model can process tool results and make appropriate follow-up calls.

    Turn 1 Prompt: "Find files related to authentication"
    Turn 1 Expected: Model calls 'search'
    Turn 2: Inject tool result with file paths
    Turn 2 Expected: Model calls 'read_file' on one of the returned files

    Pass: Calls read_file on a file from the results
    Fail: Calls search again, calls unrelated tool, text-only response, hallucinated file
    """

    def __init__(self):
        self.level = 3
        self.name = "Multi-Turn Coherence"
        self.turn1_prompt = "Find files related to authentication and read the most relevant one."
        self.prompt = self.turn1_prompt  # For provenance tracking
        self.tools = get_multi_tool_set()

        # Mock tool result to inject after turn 1
        self.mock_search_result = ["src/auth/middleware.ts", "src/auth/jwt.ts"]

    def run(self, model: str, client: OpenAI) -> ProbeResult:
        """
        Execute the multi-turn probe against the specified model.

        Args:
            model: Model identifier (e.g., "x-ai/grok-4.1-fast:free")
            client: Configured OpenAI client (OpenRouter compatible)

        Returns:
            ProbeResult for Turn 2 (follow-up behavior)
        """
        try:
            # Turn 1: Model should call search
            turn1_result = self._execute_turn1(model, client)

            if not turn1_result.success:
                # If turn 1 fails, return that result
                return turn1_result

            # Turn 2: Inject search result and check follow-up
            turn2_result = self._execute_turn2(
                model, client, turn1_result.tool_name, turn1_result.parameters
            )

            return turn2_result

        except Exception as e:
            return ProbeResult(
                success=False,
                tool_called=False,
                tool_name=None,
                parameters=None,
                raw_response={},
                latency_ms=0,
                error=f"Multi-turn error: {str(e)}",
            )

    def _execute_turn1(self, model: str, client: OpenAI) -> ProbeResult:
        """Execute turn 1: prompt for file search."""
        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": self.turn1_prompt}],
            tools=self.tools,
            temperature=0.1,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        raw_response = response.model_dump()
        choice = response.choices[0]
        message = choice.message

        tool_called = message.tool_calls is not None and len(message.tool_calls) > 0

        if not tool_called:
            return ProbeResult(
                success=False,
                tool_called=False,
                tool_name=None,
                parameters=None,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error="Turn 1: No tool called",
            )

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name

        try:
            parameters = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            parameters = {"raw": tool_call.function.arguments}

        # Turn 1 should call search or list_directory
        if tool_name not in {"search", "list_directory"}:
            return ProbeResult(
                success=False,
                tool_called=True,
                tool_name=tool_name,
                parameters=parameters,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error=f"Turn 1: Wrong tool '{tool_name}', expected search",
            )

        return ProbeResult(
            success=True,
            tool_called=True,
            tool_name=tool_name,
            parameters=parameters,
            raw_response=raw_response,
            latency_ms=latency_ms,
        )

    def _execute_turn2(
        self, model: str, client: OpenAI, turn1_tool: str | None, turn1_params: dict | None
    ) -> ProbeResult:
        """Execute turn 2: inject tool result and check follow-up."""
        start_time = time.time()

        # Build conversation history with injected tool result
        messages = [
            {"role": "user", "content": self.turn1_prompt},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_turn1",
                        "type": "function",
                        "function": {
                            "name": turn1_tool or "search",
                            "arguments": json.dumps(turn1_params or {"query": "authentication"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_turn1",
                "content": json.dumps(self.mock_search_result),
            },
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=self.tools,
            temperature=0.1,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        raw_response = response.model_dump()
        choice = response.choices[0]
        message = choice.message

        tool_called = message.tool_calls is not None and len(message.tool_calls) > 0

        if not tool_called:
            return ProbeResult(
                success=False,
                tool_called=False,
                tool_name=None,
                parameters=None,
                raw_response=raw_response,
                latency_ms=latency_ms,
                error="Turn 2: No tool called (should read file)",
            )

        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name

        try:
            parameters = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            parameters = {"raw": tool_call.function.arguments}

        # Validate turn 2 behavior
        success, error = self._validate_turn2(tool_name, parameters)

        return ProbeResult(
            success=success,
            tool_called=True,
            tool_name=tool_name,
            parameters=parameters,
            raw_response=raw_response,
            latency_ms=latency_ms,
            error=error,
        )

    def _validate_turn2(self, tool_name: str, parameters: dict | None) -> tuple[bool, str | None]:
        """
        Validate turn 2 behavior: should call read_file on a returned file.

        Pass criteria:
        - Tool is 'read_file'
        - Path parameter matches one of the mock results

        Args:
            tool_name: Tool called in turn 2
            parameters: Parameters passed to the tool

        Returns:
            Tuple of (is_valid, error_message)
        """
        if tool_name != "read_file":
            return (False, f"Turn 2: Wrong tool '{tool_name}', expected read_file")

        if not parameters or "path" not in parameters:
            return (False, "Turn 2: read_file missing 'path' parameter")

        file_path = parameters["path"]

        # Check if path is one of the returned files
        if file_path not in self.mock_search_result:
            return (
                False,
                f"Turn 2: Hallucinated file '{file_path}', not in search results",
            )

        return (True, None)

    def __str__(self) -> str:
        return f"Level {self.level}: {self.name}"
