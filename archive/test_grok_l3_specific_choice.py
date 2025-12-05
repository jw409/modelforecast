#!/usr/bin/env python3
"""Test different tool_choice strategies for Grok L3."""

import json
import os

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from modelforecast.tools.mock_tools import get_multi_tool_set


def test_turn2_behavior(client: OpenAI, model: str, tool_choice_param, messages: list, tools: list) -> dict:
    """Test turn 2 with specific tool_choice parameter."""
    kwargs = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.1,
    }

    if tool_choice_param is not None:
        kwargs["tool_choice"] = tool_choice_param

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    result = {
        "tool_called": message.tool_calls is not None and len(message.tool_calls) > 0,
        "tool_name": None,
        "arguments": None,
        "is_text": message.content is not None and len(message.content) > 0 if message.content else False,
        "text_preview": message.content[:100] if message.content else None,
    }

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        result["tool_name"] = tool_call.function.name
        try:
            result["arguments"] = json.loads(tool_call.function.arguments)
        except Exception:
            result["arguments"] = {"raw": tool_call.function.arguments}

    return result


def main():
    """Test different tool_choice strategies."""
    console = Console()

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
        return

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    model = "x-ai/grok-4.1-fast:free"
    tools = get_multi_tool_set()

    console.print(f"[bold blue]Testing tool_choice strategies for {model}[/bold blue]\n")

    # Setup conversation history
    turn1_prompt = "Find files related to authentication"
    mock_search_result = ["src/auth/middleware.ts", "src/auth/jwt.ts"]

    messages = [
        {"role": "user", "content": turn1_prompt},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_turn1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "authentication"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_turn1",
            "content": json.dumps(mock_search_result),
        },
    ]

    # Test different strategies
    strategies = [
        ("auto (default)", None),
        ("required (any tool)", "required"),
        ("specific (read_file)", {"type": "function", "function": {"name": "read_file"}}),
    ]

    results = []

    for strategy_name, tool_choice_value in strategies:
        console.print(f"[yellow]Testing: {strategy_name}[/yellow]")

        try:
            result = test_turn2_behavior(client, model, tool_choice_value, messages, tools)
            results.append((strategy_name, result))

            if result["tool_called"]:
                console.print(f"  ✓ Tool: {result['tool_name']}")
                console.print(f"    Args: {json.dumps(result['arguments'])}")
            elif result["is_text"]:
                console.print(f"  ✗ Text response: {result['text_preview']}...")
            else:
                console.print(f"  ✗ No tool called, no text")

        except Exception as e:
            console.print(f"  ✗ Error: {e}")
            results.append((strategy_name, {"error": str(e)}))

        console.print()

    # Summary table
    console.print("[bold]Summary[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strategy", style="dim")
    table.add_column("Behavior", justify="center")
    table.add_column("Tool Called", justify="center")
    table.add_column("Correct?", justify="center")

    for strategy_name, result in results:
        if "error" in result:
            table.add_row(strategy_name, "Error", "N/A", "✗")
        elif result["tool_called"]:
            is_correct = result["tool_name"] == "read_file" and result["arguments"].get("path") in ["src/auth/middleware.ts", "src/auth/jwt.ts"]
            table.add_row(
                strategy_name,
                "Tool call",
                result["tool_name"],
                "✓" if is_correct else "✗"
            )
        else:
            table.add_row(strategy_name, "Text", "-", "✗")

    console.print(table)

    # Analysis
    console.print("\n[bold]Analysis[/bold]\n")

    # Check if specific tool_choice works
    specific_result = results[2][1] if len(results) > 2 else None

    if specific_result and specific_result["tool_called"]:
        if specific_result["tool_name"] == "read_file":
            path = specific_result["arguments"].get("path", "")
            if path in ["src/auth/middleware.ts", "src/auth/jwt.ts"]:
                console.print("[green]✓ Success: tool_choice with specific function forces correct tool selection[/green]")
                console.print("[green]  Recommendation: Use tool_choice={'type': 'function', 'function': {'name': 'read_file'}}[/green]")
                console.print("[green]  for L3 multi-turn with Grok when next tool is known.[/green]")
            else:
                console.print("[yellow]⚠ Partial: Correct tool but wrong file path[/yellow]")
        else:
            console.print("[red]✗ Failed: Specific tool_choice didn't force read_file[/red]")
    else:
        console.print("[red]✗ Failed: Specific tool_choice strategy errored or returned text[/red]")

    # Check if "required" helps
    required_result = results[1][1] if len(results) > 1 else None

    if required_result and required_result["tool_called"]:
        console.print(f"\n[yellow]Note: tool_choice='required' forces a tool call but doesn't guarantee correct selection[/yellow]")
        console.print(f"[yellow]  Grok chose {required_result['tool_name']} instead of read_file[/yellow]")


if __name__ == "__main__":
    main()
