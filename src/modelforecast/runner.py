"""Main probe runner for ModelForecast benchmarks."""

import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from modelforecast.output.json_report import write_individual_result, write_json_report
from modelforecast.output.markdown_report import write_markdown_report
from modelforecast.probes.base import ProbeResult
from modelforecast.probes.t0_invoke import T0InvokeProbe
from modelforecast.stats.confidence import wilson_interval
from modelforecast.verification.provenance import ProvenanceTracker

# Default models to test (from METHODOLOGY.md)
DEFAULT_MODELS = [
    "google/gemma-3-27b-it:free",
    "google/gemini-2.5-flash-lite-preview-09-2025:free",
    "meta-llama/llama-4-maverick:free",
    "microsoft/mai-ds-r1:free",
    "nousresearch/deephermes-3-llama-3-8b-preview:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-30b-a3b:free",
    "qwen/qwen3-32b:free",
    "x-ai/grok-4.1-fast:free",
]

# Level threshold for skipping higher levels
LEVEL_0_THRESHOLD = 0.20  # If L0 < 20%, skip higher levels


class ProbeRunner:
    """Main orchestrator for running model capability probes."""

    def __init__(
        self,
        output_dir: Path,
        models: list[str] | None = None,
        contributor: str | None = None,
    ):
        """Initialize probe runner.

        Args:
            output_dir: Directory to write results to
            models: List of models to test (defaults to all free models)
            contributor: GitHub username for provenance (defaults to env var or "unknown")
        """
        self.output_dir = Path(output_dir)
        self.models = models or DEFAULT_MODELS
        self.contributor = contributor or os.getenv("GITHUB_USERNAME", "unknown")
        self.console = Console()

        # Initialize OpenAI client with OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Initialize probe levels
        # Import all probe classes using new T/R/A naming
        try:
            from modelforecast.probes.t1_schema import T1SchemaProbe
            from modelforecast.probes.t2_selection import T2SelectionProbe
            from modelforecast.probes.a1_linear import A1LinearProbe
            from modelforecast.probes.r0_abstain import R0AbstainProbe

            self.probes = {
                0: T0InvokeProbe(),   # T0 Invoke (was L0 Basic)
                1: T1SchemaProbe(),   # T1 Schema (was L1)
                2: T2SelectionProbe(), # T2 Selection (was L2)
                3: A1LinearProbe(),   # A1 Linear (was L3 Multi-turn)
                4: R0AbstainProbe(),  # R0 Abstain (was L4 Adversarial)
            }
        except ImportError:
            # Fall back to T0 only if other probes not yet implemented
            self.probes = {
                0: T0InvokeProbe(),
            }

    def run_level(
        self,
        model: str,
        level: int,
        trials: int = 10,
    ) -> dict[str, Any] | None:
        """Run all trials for a specific model and level.

        Args:
            model: Model identifier
            level: Probe level (0-4)
            trials: Number of trials to run

        Returns:
            Complete result dictionary with provenance, or None if probe not available
        """
        if level not in self.probes:
            self.console.print(f"[yellow]Level {level} probe not implemented yet[/yellow]")
            return None

        probe = self.probes[level]
        tracker = ProvenanceTracker(contributor=self.contributor)

        self.console.print(f"\n[bold]Testing {model} - {probe}[/bold]")

        results: list[ProbeResult] = []
        trial_records = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {trials} trials...", total=trials)

            for trial_num in range(trials):
                result = probe.run(model, self.client)
                results.append(result)

                # Create trial record for provenance with full data for schema-on-read
                request_data = {
                    "model": model,
                    "messages": [{"role": "user", "content": probe.prompt}],
                    "tools": probe.tools,
                    "temperature": 0.1,
                }
                trial_record = tracker.create_trial_record(
                    prompt=probe.prompt,
                    response=str(result.raw_response),
                    tool_called=result.tool_called,
                    schema_valid=result.success,  # For L0, success == tool_called
                    latency_ms=result.latency_ms,
                    openrouter_request_id=result.raw_response.get("id"),
                    # Full data for efficiency analysis
                    request_data=request_data,
                    response_data=result.raw_response,
                )
                trial_records.append(trial_record)

                # Update progress
                status = "✓" if result.success else "✗"
                progress.update(
                    task,
                    advance=1,
                    description=f"Trial {trial_num + 1}/{trials} {status}",
                )

        # Calculate statistics
        successes = sum(1 for r in results if r.success)
        ci = wilson_interval(successes, trials)

        # Create complete result with provenance
        complete_result = tracker.create_result(
            model=model,
            level=level,
            trials=trial_records,
            successes=successes,
            wilson_ci=ci,
        )

        # Print summary
        rate_pct = int(complete_result["summary"]["rate"] * 100)
        ci_lower = int(ci[0] * 100)
        ci_upper = int(ci[1] * 100)
        self.console.print(
            f"[bold green]Result: {successes}/{trials} ({rate_pct}%) "
            f"CI=[{ci_lower},{ci_upper}][/bold green]"
        )

        # Write individual result
        write_individual_result(self.output_dir, model, level, complete_result)

        return complete_result

    def run_model(
        self,
        model: str,
        trials: int = 10,
        max_level: int = 4,
    ) -> dict[int, dict[str, Any]]:
        """Run all levels for a specific model.

        Args:
            model: Model identifier
            trials: Number of trials per level
            max_level: Maximum level to test (0-4)

        Returns:
            Dictionary mapping level -> result data
        """
        model_results = {}

        # Always run Level 0 first
        level_0_result = self.run_level(model, 0, trials)
        if level_0_result:
            model_results[0] = level_0_result

            # Check if Level 0 performance warrants testing higher levels
            l0_rate = level_0_result["summary"]["rate"]
            if l0_rate < LEVEL_0_THRESHOLD:
                self.console.print(
                    f"[yellow]Skipping higher levels (L0 rate {l0_rate:.1%} < "
                    f"{LEVEL_0_THRESHOLD:.0%})[/yellow]"
                )
                return model_results

        # Run higher levels if L0 passed threshold
        for level in range(1, max_level + 1):
            if level in self.probes:
                result = self.run_level(model, level, trials)
                if result:
                    model_results[level] = result

        return model_results

    def run_all(
        self,
        trials: int = 10,
        max_level: int = 4,
    ) -> dict[str, dict[str, Any]]:
        """Run all configured models and levels.

        Args:
            trials: Number of trials per (model, level) combination
            max_level: Maximum level to test (0-4)

        Returns:
            Dictionary mapping result_key -> complete result data
        """
        self.console.print("[bold blue]ModelForecast Benchmark Suite[/bold blue]")
        self.console.print(f"Testing {len(self.models)} models, up to Level {max_level}")
        self.console.print(f"Trials per level: {trials}")
        self.console.print(f"Output directory: {self.output_dir}")

        all_results = {}

        for model_idx, model in enumerate(self.models, 1):
            self.console.print(
                f"\n[bold cyan]Model {model_idx}/{len(self.models)}: {model}[/bold cyan]"
            )

            model_results = self.run_model(model, trials, max_level)

            # Add to all_results with unique keys
            for level, result in model_results.items():
                result_key = f"{model}__level_{level}"
                all_results[result_key] = result

        # Write summary reports
        self.console.print("\n[bold]Generating reports...[/bold]")
        json_file = write_json_report(self.output_dir, all_results)
        md_file = write_markdown_report(self.output_dir, all_results)

        self.console.print(f"[green]✓ JSON report: {json_file}[/green]")
        self.console.print(f"[green]✓ Markdown report: {md_file}[/green]")

        return all_results
