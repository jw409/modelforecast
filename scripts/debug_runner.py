#!/usr/bin/env python3
"""Debug runner for ModelForecast with retry handling and verbose logging.

Usage:
    uv run python scripts/debug_runner.py --model "anthropic/claude-3.5-haiku" --level 0
    uv run python scripts/debug_runner.py --model "google/gemini-2.5-flash:free" --explore
    uv run python scripts/debug_runner.py --model "deepseek/deepseek-chat-v3-0324:free" --explore
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from rich.console import Console
from rich.table import Table

console = Console()


def retry_with_backoff(func, max_retries=5, initial_delay=1.0):
    """Execute function with exponential backoff on rate limits."""
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            last_error = e
            console.print(f"[yellow]Rate limited (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...[/yellow]")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except APIError as e:
            last_error = e
            if "rate" in str(e).lower() or "429" in str(e):
                console.print(f"[yellow]API error (rate related), attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s...[/yellow]")
                time.sleep(delay)
                delay *= 2
            else:
                raise  # Re-raise non-rate-limit API errors
        except APIConnectionError as e:
            last_error = e
            console.print(f"[yellow]Connection error, attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s...[/yellow]")
            time.sleep(delay)
            delay *= 2

    raise last_error


def run_single_probe(client: OpenAI, model: str, level: int = 0):
    """Run a single probe with detailed output."""
    from modelforecast.probes.t0_invoke import T0InvokeProbe

    probes = {0: T0InvokeProbe}

    try:
        from modelforecast.probes.t1_schema import T1SchemaProbe
        from modelforecast.probes.t2_selection import T2SelectionProbe
        probes[1] = T1SchemaProbe
        probes[2] = T2SelectionProbe
    except ImportError:
        pass

    if level not in probes:
        console.print(f"[red]Level {level} not available. Available: {list(probes.keys())}[/red]")
        return None

    probe = probes[level]()
    console.print(f"\n[bold cyan]Running {probe.name} (Level {level})[/bold cyan]")
    console.print(f"Model: {model}")
    console.print(f"Prompt: {probe.prompt[:80]}...")

    def execute():
        return probe.run(model, client)

    start = time.time()
    result = retry_with_backoff(execute)
    elapsed = time.time() - start

    # Display result
    status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
    console.print(f"\nResult: {status}")
    console.print(f"Latency: {result.latency_ms}ms (total with retries: {elapsed*1000:.0f}ms)")

    if result.tool_name:
        console.print(f"Tool called: {result.tool_name}")
    if result.parameters:
        console.print(f"Tool params: {json.dumps(result.parameters, indent=2)}")
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")

    return result


def explore_model(client: OpenAI, model: str, trials: int = 3):
    """Exploratory run with multiple trials and detailed diagnostics."""
    console.print(f"\n[bold magenta]EXPLORATORY MODE: {model}[/bold magenta]")
    console.print(f"Running {trials} trials at level 0 to diagnose behavior\n")

    results = []
    errors = []

    for i in range(trials):
        console.print(f"[dim]--- Trial {i+1}/{trials} ---[/dim]")
        try:
            result = run_single_probe(client, model, level=0)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Trial {i+1} failed: {type(e).__name__}: {e}[/red]")
            errors.append(str(e))

        # Brief pause between trials
        if i < trials - 1:
            time.sleep(1)

    # Summary
    console.print("\n[bold]SUMMARY[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    successes = sum(1 for r in results if r and r.success)
    table.add_row("Model", model)
    table.add_row("Trials", str(trials))
    table.add_row("Successes", f"{successes}/{len(results)}")
    table.add_row("Errors", str(len(errors)))

    if results:
        latencies = [r.latency_ms for r in results if r]
        table.add_row("Avg Latency", f"{sum(latencies)/len(latencies):.0f}ms")

    console.print(table)

    if errors:
        console.print("\n[bold red]Errors encountered:[/bold red]")
        for err in set(errors):
            console.print(f"  - {err}")

    # Save diagnostic report
    report = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
        "successes": successes,
        "errors": errors,
        "results": [
            {
                "success": r.success,
                "latency_ms": r.latency_ms,
                "tool_name": r.tool_name,
                "parameters": r.parameters,
                "error": r.error,
            }
            for r in results if r
        ]
    }

    report_path = Path("var/debug_reports")
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / f"{model.replace('/', '_').replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.write_text(json.dumps(report, indent=2))
    console.print(f"\n[dim]Report saved: {report_file}[/dim]")

    return report


def main():
    parser = argparse.ArgumentParser(description="Debug runner for ModelForecast")
    parser.add_argument("--model", required=True, help="Model ID to test")
    parser.add_argument("--level", type=int, default=0, help="Probe level (0-2)")
    parser.add_argument("--explore", action="store_true", help="Exploratory mode with multiple trials")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials in explore mode")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]ERROR: OPENROUTER_API_KEY not set[/red]")
        console.print("Set it with: export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    console.print(f"[bold]ModelForecast Debug Runner[/bold]")
    console.print(f"API Key: {api_key[:12]}...{api_key[-4:]}")

    if args.explore:
        explore_model(client, args.model, args.trials)
    else:
        run_single_probe(client, args.model, args.level)


if __name__ == "__main__":
    main()
