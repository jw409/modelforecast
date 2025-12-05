#!/usr/bin/env python3
"""Debug Grok L3 behavior to understand what's happening."""

import json
import os

from openai import OpenAI
from rich.console import Console
from rich.syntax import Syntax

from modelforecast.tools.mock_tools import get_multi_tool_set


def main():
    """Debug Grok's L3 multi-turn behavior."""
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

    console.print(f"[bold blue]Debugging {model} L3 Behavior[/bold blue]\n")

    # Turn 1: User asks to find files
    console.print("[bold yellow]Turn 1: User prompt[/bold yellow]")
    turn1_prompt = "Find files related to authentication"
    console.print(f"  Prompt: {turn1_prompt}\n")

    response1 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": turn1_prompt}],
        tools=tools,
        temperature=0.1,
    )

    tool_call1 = response1.choices[0].message.tool_calls[0] if response1.choices[0].message.tool_calls else None

    if tool_call1:
        console.print(f"[green]  ✓ Turn 1: Model called {tool_call1.function.name}[/green]")
        console.print(f"    Arguments: {tool_call1.function.arguments}\n")
    else:
        console.print("[red]  ✗ Turn 1: No tool called[/red]\n")
        return

    # Turn 2: Inject search results
    mock_search_result = ["src/auth/middleware.ts", "src/auth/jwt.ts"]

    console.print("[bold yellow]Turn 2: Tool result injected[/bold yellow]")
    console.print(f"  Tool result: {json.dumps(mock_search_result)}\n")

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
                        "name": tool_call1.function.name,
                        "arguments": tool_call1.function.arguments,
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

    # Try baseline (auto)
    console.print("[bold]Baseline: tool_choice='auto'[/bold]")
    response2_auto = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0.1,
    )

    message2_auto = response2_auto.choices[0].message

    if message2_auto.tool_calls:
        tool_call2_auto = message2_auto.tool_calls[0]
        console.print(f"[yellow]  Tool called: {tool_call2_auto.function.name}[/yellow]")
        console.print(f"  Arguments: {tool_call2_auto.function.arguments}")
    else:
        console.print(f"[red]  No tool called[/red]")
        if message2_auto.content:
            console.print(f"  Text response: {message2_auto.content[:200]}...")

    console.print()

    # Try with tool_choice="required"
    console.print("[bold]Fixed: tool_choice='required'[/bold]")
    response2_required = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=0.1,
        tool_choice="required",
    )

    message2_required = response2_required.choices[0].message

    if message2_required.tool_calls:
        tool_call2_required = message2_required.tool_calls[0]
        console.print(f"[green]  Tool called: {tool_call2_required.function.name}[/green]")
        console.print(f"  Arguments: {tool_call2_required.function.arguments}")

        # Analyze why wrong tool
        try:
            params = json.loads(tool_call2_required.function.arguments)
            console.print(f"\n[bold]Analysis:[/bold]")
            console.print(f"  Expected: read_file(path='src/auth/middleware.ts' OR 'src/auth/jwt.ts')")
            console.print(f"  Got: {tool_call2_required.function.name}(path='{params.get('path', 'N/A')}')")

            if tool_call2_required.function.name == "list_directory":
                console.print("\n[yellow]  Issue: Model is listing directory instead of reading specific file[/yellow]")
                console.print("[yellow]  Hypothesis: Model wants to explore directory structure[/yellow]")
        except Exception as e:
            console.print(f"  Error parsing arguments: {e}")
    else:
        console.print(f"[red]  No tool called[/red]")

    console.print("\n[bold]Available tools:[/bold]")
    for i, tool in enumerate(tools, 1):
        console.print(f"  {i}. {tool['function']['name']}: {tool['function']['description']}")

    console.print("\n[bold]Expected behavior:[/bold]")
    console.print("  After receiving file paths, model should call read_file on one of them")
    console.print("  to examine the authentication-related code.")


if __name__ == "__main__":
    main()
