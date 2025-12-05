#!/usr/bin/env python3
"""Test Grok L3 performance with and without tool_choice='required' fix."""

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from modelforecast.probes.base import ProbeResult
from modelforecast.probes.level3_multiturn import Level3MultiTurnProbe


class Level3WithToolChoice(Level3MultiTurnProbe):
    """Modified L3 probe that supports tool_choice parameter."""

    def __init__(self, use_tool_choice: bool = False):
        super().__init__()
        self.use_tool_choice = use_tool_choice
        self.name = f"Multi-Turn Coherence (tool_choice={'required' if use_tool_choice else 'auto'})"

    def _execute_turn1(self, model: str, client: OpenAI) -> ProbeResult:
        """Execute turn 1: prompt for file search."""
        start_time = time.time()

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": self.turn1_prompt}],
            "tools": self.tools,
            "temperature": 0.1,
        }

        if self.use_tool_choice:
            kwargs["tool_choice"] = "required"

        response = client.chat.completions.create(**kwargs)

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

        kwargs = {
            "model": model,
            "messages": messages,
            "tools": self.tools,
            "temperature": 0.1,
        }

        if self.use_tool_choice:
            kwargs["tool_choice"] = "required"

        response = client.chat.completions.create(**kwargs)

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


def run_trials(probe: Level3WithToolChoice, model: str, client: OpenAI, num_trials: int) -> dict[str, Any]:
    """Run multiple trials and collect statistics."""
    results = []
    successes = 0

    for i in range(num_trials):
        result = probe.run(model, client)
        results.append(result)
        if result.success:
            successes += 1

        # Print progress
        print(f"  Trial {i+1}/{num_trials}: {'✓' if result.success else '✗'} {result.error or ''}")

    success_rate = successes / num_trials

    return {
        "successes": successes,
        "trials": num_trials,
        "rate": success_rate,
        "results": results,
    }


def main():
    """Run L3 tests on Grok with and without the fix."""
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
    num_trials = 10

    console.print(f"[bold blue]Testing {model} L3 Multi-Turn Performance[/bold blue]\n")
    console.print(f"Running {num_trials} trials per configuration\n")

    # Run baseline (no fix)
    console.print("[bold yellow]Baseline: tool_choice='auto' (default)[/bold yellow]")
    probe_baseline = Level3WithToolChoice(use_tool_choice=False)
    baseline_results = run_trials(probe_baseline, model, client, num_trials)

    console.print()

    # Run with fix
    console.print("[bold green]Fixed: tool_choice='required'[/bold green]")
    probe_fixed = Level3WithToolChoice(use_tool_choice=True)
    fixed_results = run_trials(probe_fixed, model, client, num_trials)

    # Print summary table
    console.print("\n[bold]Summary[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Configuration", style="dim")
    table.add_column("Success Rate", justify="right")
    table.add_column("Successes", justify="center")
    table.add_column("Interpretation", style="cyan")

    baseline_pct = int(baseline_results["rate"] * 100)
    fixed_pct = int(fixed_results["rate"] * 100)

    table.add_row(
        "Baseline (auto)",
        f"{baseline_pct}%",
        f"{baseline_results['successes']}/{baseline_results['trials']}",
        "Expected: ~0%" if baseline_pct <= 10 else "Unexpected: should be ~0%"
    )

    table.add_row(
        "Fixed (required)",
        f"{fixed_pct}%",
        f"{fixed_results['successes']}/{fixed_results['trials']}",
        "Expected: ~100%" if fixed_pct >= 90 else "Unexpected: should be ~100%"
    )

    console.print(table)

    # Recommendation
    console.print("\n[bold]Recommendation[/bold]\n")

    if fixed_pct >= 90 and baseline_pct <= 10:
        console.print("[green]✓ Fix confirmed working. Recommend using tool_choice='required' for Grok in L3+ scenarios.[/green]")
        console.print("[green]  This should be the default for multi-turn tool-calling workflows with Grok.[/green]")
    elif fixed_pct >= 90:
        console.print("[yellow]⚠ Fix works, but baseline unexpectedly high. May indicate model update.[/yellow]")
    elif baseline_pct >= 90:
        console.print("[yellow]⚠ Baseline already working. Fix may not be needed. Model may have improved.[/yellow]")
    else:
        console.print("[red]✗ Fix not effective. Further investigation needed.[/red]")

    # Save detailed results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "grok_l3_fix_comparison.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model,
            "trials": num_trials,
            "baseline": {
                "use_tool_choice": False,
                "success_rate": baseline_results["rate"],
                "successes": baseline_results["successes"],
            },
            "fixed": {
                "use_tool_choice": True,
                "success_rate": fixed_results["rate"],
                "successes": fixed_results["successes"],
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    console.print(f"\n[dim]Detailed results saved to: {output_file}[/dim]")


if __name__ == "__main__":
    main()
