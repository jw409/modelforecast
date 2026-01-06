#!/usr/bin/env python3
"""Validate L3 Multi-Turn behavior with raw response capture.

Usage:
    OPENROUTER_API_KEY=... uv run python scripts/validate_l3.py --model "x-ai/grok-4.1-fast:free"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI
from rich.console import Console
from rich.json import JSON

console = Console()


def validate_l3(client: OpenAI, model: str, trials: int = 3):
    """Run L3 multi-turn probe with full raw response capture."""
    from modelforecast.probes.a1_linear import A1LinearProbe

    probe = A1LinearProbe()
    console.print(f"\n[bold cyan]L3 Multi-Turn Validation: {model}[/bold cyan]")
    console.print(f"Prompt: {probe.turn1_prompt}")
    console.print(f"Expected: Turn 1 → search, Turn 2 → read_file")
    console.print()

    results = []

    for i in range(trials):
        console.print(f"[bold]─── Trial {i+1}/{trials} ───[/bold]")

        try:
            # Turn 1: Should call search
            console.print("[dim]Turn 1: Sending initial prompt...[/dim]")
            start = time.time()

            response1 = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": probe.turn1_prompt}],
                tools=probe.tools,
                temperature=0.1,
            )

            latency1 = int((time.time() - start) * 1000)
            raw1 = response1.model_dump()

            # Check Turn 1 response
            msg1 = response1.choices[0].message
            t1_tool_called = msg1.tool_calls is not None and len(msg1.tool_calls) > 0

            if t1_tool_called:
                t1_tool = msg1.tool_calls[0].function.name
                t1_args = msg1.tool_calls[0].function.arguments
                console.print(f"[green]Turn 1 PASS: Called {t1_tool}[/green]")
                console.print(f"  Args: {t1_args}")
            else:
                t1_tool = None
                console.print(f"[red]Turn 1 FAIL: No tool called[/red]")
                console.print(f"  Content: {msg1.content[:200] if msg1.content else '(empty)'}")

            # Turn 2: Inject search result and check follow-up
            if t1_tool_called:
                console.print("[dim]Turn 2: Injecting search results...[/dim]")

                messages = [
                    {"role": "user", "content": probe.turn1_prompt},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_turn1",
                            "type": "function",
                            "function": {
                                "name": t1_tool,
                                "arguments": t1_args,
                            },
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_turn1",
                        "content": json.dumps(["src/auth/middleware.ts", "src/auth/jwt.ts"]),
                    },
                ]

                start = time.time()
                response2 = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=probe.tools,
                    temperature=0.1,
                )
                latency2 = int((time.time() - start) * 1000)
                raw2 = response2.model_dump()

                msg2 = response2.choices[0].message
                t2_tool_called = msg2.tool_calls is not None and len(msg2.tool_calls) > 0

                if t2_tool_called:
                    t2_tool = msg2.tool_calls[0].function.name
                    t2_args = msg2.tool_calls[0].function.arguments

                    if t2_tool == "read_file":
                        # Check if file path is from results
                        try:
                            args = json.loads(t2_args)
                            path = args.get("path", "")
                            if path in ["src/auth/middleware.ts", "src/auth/jwt.ts"]:
                                console.print(f"[green]Turn 2 PASS: read_file({path})[/green]")
                                results.append({"success": True, "error": None})
                            else:
                                console.print(f"[yellow]Turn 2 PARTIAL: read_file({path}) - not from results[/yellow]")
                                results.append({"success": False, "error": f"hallucinated path: {path}"})
                        except json.JSONDecodeError:
                            console.print(f"[red]Turn 2 FAIL: Invalid JSON args[/red]")
                            results.append({"success": False, "error": "invalid json"})
                    else:
                        console.print(f"[red]Turn 2 FAIL: Wrong tool {t2_tool}, expected read_file[/red]")
                        results.append({"success": False, "error": f"wrong tool: {t2_tool}"})
                else:
                    console.print(f"[red]Turn 2 FAIL: No tool called[/red]")
                    console.print(f"  Content: {msg2.content[:200] if msg2.content else '(empty)'}")
                    results.append({"success": False, "error": "no tool called in turn 2"})

                    # Dump raw response for diagnosis
                    console.print("[dim]Raw Turn 2 response:[/dim]")
                    console.print(JSON(json.dumps(raw2, indent=2)[:500]))
            else:
                results.append({"success": False, "error": "turn 1 failed"})

        except Exception as e:
            console.print(f"[red]Error: {type(e).__name__}: {e}[/red]")
            results.append({"success": False, "error": str(e)})

        console.print()
        if i < trials - 1:
            time.sleep(1)

    # Summary
    successes = sum(1 for r in results if r["success"])
    console.print(f"\n[bold]SUMMARY: {model}[/bold]")
    console.print(f"L3 Success Rate: {successes}/{trials} ({100*successes/trials:.0f}%)")

    error_counts = {}
    for r in results:
        if r["error"]:
            error_counts[r["error"]] = error_counts.get(r["error"], 0) + 1

    if error_counts:
        console.print("[bold]Failure breakdown:[/bold]")
        for err, count in error_counts.items():
            console.print(f"  - {err}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate L3 multi-turn behavior")
    parser.add_argument("--model", required=True, help="Model ID to test")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENROUTER_API_KEY not set[/red]")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    validate_l3(client, args.model, args.trials)


if __name__ == "__main__":
    main()
