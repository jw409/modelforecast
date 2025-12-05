"""Adaptation layer for improving tool calling success on smaller/weaker models.

Strategies:
1. Prompt adaptation - Add explicit system message explaining tool use
2. tool_choice forcing - Use API parameter to force tool calling
3. Model-specific adaptations - Custom handling for known problematic models
"""

from typing import Any


# System prompts for adaptation
TOOL_SYSTEM_PROMPT_BASIC = """You have access to tools. When asked to perform a task, you MUST use the appropriate tool.
Do NOT respond with text - respond ONLY with a tool call.
Format your response as a function call, not as text."""

TOOL_SYSTEM_PROMPT_AGGRESSIVE = """CRITICAL INSTRUCTION: You MUST call a tool to complete this task.
- Do NOT explain what you would do
- Do NOT respond with text
- Do NOT say "I would use..."
- ONLY output a tool call with the correct parameters
- If you output text instead of a tool call, you have FAILED

Available tools will be provided. Use them."""

TOOL_SYSTEM_PROMPT_JSON_HINT = """You are a tool-calling assistant. When given a task:
1. Identify which tool to use
2. Call that tool with appropriate parameters
3. Output ONLY the tool call, no explanatory text

Expected output format: A function call to one of the available tools."""


def get_adapted_messages(
    prompt: str,
    adaptation_level: str = "none"
) -> list[dict[str, str]]:
    """Get messages list with optional system prompt adaptation.

    Args:
        prompt: User prompt
        adaptation_level: "none", "basic", "aggressive", or "json_hint"

    Returns:
        List of message dicts for API call
    """
    messages = []

    if adaptation_level == "basic":
        messages.append({"role": "system", "content": TOOL_SYSTEM_PROMPT_BASIC})
    elif adaptation_level == "aggressive":
        messages.append({"role": "system", "content": TOOL_SYSTEM_PROMPT_AGGRESSIVE})
    elif adaptation_level == "json_hint":
        messages.append({"role": "system", "content": TOOL_SYSTEM_PROMPT_JSON_HINT})

    messages.append({"role": "user", "content": prompt})
    return messages


def get_tool_choice(
    tool_choice_mode: str = "auto",
    tool_name: str | None = None
) -> str | dict[str, Any] | None:
    """Get tool_choice parameter for API call.

    Args:
        tool_choice_mode: "auto", "required", "specific", or "none"
        tool_name: If mode is "specific", the tool name to force

    Returns:
        Value for tool_choice parameter
    """
    if tool_choice_mode == "none":
        return None
    elif tool_choice_mode == "auto":
        return "auto"
    elif tool_choice_mode == "required":
        return "required"
    elif tool_choice_mode == "specific" and tool_name:
        return {
            "type": "function",
            "function": {"name": tool_name}
        }
    return "auto"


# Model-specific adaptation profiles
MODEL_ADAPTATIONS = {
    # Models that need aggressive prompting
    "qwen": {"prompt": "aggressive", "tool_choice": "required"},
    "gemini": {"prompt": "basic", "tool_choice": "auto"},
    "llama": {"prompt": "aggressive", "tool_choice": "required"},
    "mistral": {"prompt": "basic", "tool_choice": "required"},

    # Default for unknown models
    "default": {"prompt": "none", "tool_choice": "auto"},
}


def get_model_adaptation(model_id: str) -> dict[str, str]:
    """Get recommended adaptation for a specific model.

    Args:
        model_id: Full model ID

    Returns:
        Dict with "prompt" and "tool_choice" keys
    """
    model_lower = model_id.lower()

    for key, config in MODEL_ADAPTATIONS.items():
        if key != "default" and key in model_lower:
            return config

    return MODEL_ADAPTATIONS["default"]


class AdaptiveProbeRunner:
    """Runs probes with multiple adaptation strategies to find what works."""

    STRATEGIES = [
        {"prompt": "none", "tool_choice": "auto", "name": "baseline"},
        {"prompt": "none", "tool_choice": "required", "name": "required_only"},
        {"prompt": "basic", "tool_choice": "auto", "name": "basic_prompt"},
        {"prompt": "basic", "tool_choice": "required", "name": "basic+required"},
        {"prompt": "aggressive", "tool_choice": "required", "name": "aggressive"},
    ]

    @classmethod
    def find_working_strategy(
        cls,
        model: str,
        client: "OpenAI",
        tools: list[dict],
        prompt: str,
        max_attempts: int = 3,
    ) -> dict[str, Any] | None:
        """Try different strategies to find one that works.

        Args:
            model: Model ID
            client: OpenAI client
            tools: Tool definitions
            prompt: User prompt
            max_attempts: Attempts per strategy

        Returns:
            Dict with working strategy details, or None if all fail
        """
        for strategy in cls.STRATEGIES:
            messages = get_adapted_messages(prompt, strategy["prompt"])
            tool_choice = get_tool_choice(strategy["tool_choice"])

            successes = 0
            for _ in range(max_attempts):
                try:
                    kwargs = {
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "temperature": 0.1,
                    }
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice

                    response = client.chat.completions.create(**kwargs)

                    # Check if tool was called
                    choice = response.choices[0]
                    if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
                        successes += 1
                except Exception:
                    pass

            if successes > 0:
                return {
                    "strategy": strategy["name"],
                    "prompt_adaptation": strategy["prompt"],
                    "tool_choice": strategy["tool_choice"],
                    "success_rate": successes / max_attempts,
                }

        return None
