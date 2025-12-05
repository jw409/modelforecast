#!/usr/bin/env python3
"""Smart tiered benchmarking with async parallel execution.

Implements three-phase testing strategy:
- Phase 1 (T0): Test basic tool calling on all models
- Phase 2 (expand): If T0 >= 80%, run T1, T2, R0, A1
- Phase 3 (retest): Re-run previously broken models

Usage:
    # Run T0 on all untested models
    uv run python scripts/batch_benchmark.py --phase t0

    # Expand testing on T0 passers
    uv run python scripts/batch_benchmark.py --phase expand

    # Re-test broken models
    uv run python scripts/batch_benchmark.py --phase retest

    # Full run with custom concurrency
    uv run python scripts/batch_benchmark.py --phase t0 --concurrency 5

    # Force re-run
    uv run python scripts/batch_benchmark.py --phase t0 --force

    # Test specific models
    uv run python scripts/batch_benchmark.py --models "moonshotai/kimi-k2:free,allenai/olmo-3-32b-think:free"
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelforecast.models import get_free_models
from modelforecast.runner import ProbeRunner

# Thresholds
T0_EXPAND_THRESHOLD = 0.80  # If T0 >= 80%, run higher probes
CONCURRENCY_DEFAULT = 3  # Avoid rate limits

# Level mapping
PROBE_LEVELS = {
    "t0": 0,
    "t1": 1,
    "t2": 2,
    "a1": 3,
    "r0": 4,
}


def get_existing_results(results_dir: Path) -> dict[str, set[int]]:
    """Scan results directory for existing (model, level) combinations.

    Args:
        results_dir: Directory containing result JSON files

    Returns:
        Dict mapping model_id -> set of completed levels
    """
    completed = {}

    if not results_dir.exists():
        return completed

    for result_file in results_dir.glob("*.json"):
        # Parse filename: {model_id}__level_{level}.json
        stem = result_file.stem
        if "__level_" not in stem:
            continue

        model_id, level_part = stem.rsplit("__level_", 1)
        try:
            level = int(level_part)
            if model_id not in completed:
                completed[model_id] = set()
            completed[model_id].add(level)
        except ValueError:
            continue

    return completed


def get_t0_passers(results_dir: Path, threshold: float = T0_EXPAND_THRESHOLD) -> list[str]:
    """Find models that passed T0 with rate >= threshold.

    Args:
        results_dir: Directory containing result JSON files
        threshold: Minimum success rate to qualify

    Returns:
        List of model IDs that passed T0
    """
    passers = []

    for result_file in results_dir.glob("*__level_0.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            rate = data.get("summary", {}).get("rate", 0.0)
            if rate >= threshold:
                # Extract model ID from filename
                model_id = result_file.stem.replace("__level_0", "")
                passers.append(model_id)
        except (json.JSONDecodeError, KeyError):
            continue

    return passers


def get_t0_failers(results_dir: Path, threshold: float = T0_EXPAND_THRESHOLD) -> list[str]:
    """Find models that failed T0 (rate < threshold).

    Args:
        results_dir: Directory containing result JSON files
        threshold: Minimum success rate to qualify

    Returns:
        List of model IDs that failed T0
    """
    failers = []

    for result_file in results_dir.glob("*__level_0.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            rate = data.get("summary", {}).get("rate", 0.0)
            if rate < threshold:
                # Extract model ID from filename
                model_id = result_file.stem.replace("__level_0", "")
                failers.append(model_id)
        except (json.JSONDecodeError, KeyError):
            continue

    return failers


async def run_model_level_async(
    model: str,
    level: int,
    runner: ProbeRunner,
    trials: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, dict[str, Any] | None]:
    """Run a single (model, level) combination asynchronously.

    Args:
        model: Model identifier
        level: Probe level (0-4)
        runner: ProbeRunner instance
        trials: Number of trials to run
        semaphore: Concurrency control

    Returns:
        Tuple of (model, level, result_dict)
    """
    async with semaphore:
        # Run in executor since ProbeRunner is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, runner.run_level, model, level, trials
        )
        return (model, level, result)


async def run_phase_t0(
    models: list[str],
    results_dir: Path,
    trials: int,
    concurrency: int,
    force: bool,
) -> None:
    """Phase 1: Run T0 on all models.

    Args:
        models: List of model IDs to test
        results_dir: Output directory
        trials: Number of trials per model
        concurrency: Max parallel executions
        force: If True, re-run existing results
    """
    print(f"\n{'='*60}")
    print("PHASE T0: Basic Tool Calling")
    print(f"{'='*60}\n")

    # Filter out already-tested models
    if not force:
        completed = get_existing_results(results_dir)
        models = [m for m in models if m not in completed or 0 not in completed[m]]

    if not models:
        print("No models to test (all already completed)")
        return

    print(f"Testing {len(models)} models with {concurrency} concurrent executions\n")

    # Initialize runner
    runner = ProbeRunner(
        output_dir=results_dir,
        models=models,
        skip_validation=False,
    )

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Create async tasks
    tasks = [
        run_model_level_async(model, 0, runner, trials, semaphore)
        for model in models
    ]

    # Run with progress tracking
    for idx, task in enumerate(asyncio.as_completed(tasks), 1):
        model, level, result = await task

        if result:
            rate = result["summary"]["rate"]
            successes = result["summary"]["successes"]
            status = "PASS" if rate >= T0_EXPAND_THRESHOLD else "FAIL"
            print(f"[{idx}/{len(models)}] {model} - T0: {successes}/{trials} ({rate:.0%}) - {status}")
        else:
            print(f"[{idx}/{len(models)}] {model} - T0: ERROR")


async def run_phase_expand(
    results_dir: Path,
    trials: int,
    concurrency: int,
    force: bool,
) -> None:
    """Phase 2: Expand testing on T0 passers.

    Runs T1, T2, A1, R0 on models that passed T0.

    Args:
        results_dir: Output directory
        trials: Number of trials per (model, level)
        concurrency: Max parallel executions
        force: If True, re-run existing results
    """
    print(f"\n{'='*60}")
    print("PHASE EXPAND: Higher Probes for T0 Passers")
    print(f"{'='*60}\n")

    # Find T0 passers
    passers = get_t0_passers(results_dir)

    if not passers:
        print("No models passed T0 - run phase t0 first")
        return

    print(f"Found {len(passers)} models that passed T0\n")

    # Determine which (model, level) combinations to run
    completed = get_existing_results(results_dir) if not force else {}
    expand_levels = [1, 2, 3, 4]  # T1, T2, A1, R0

    tasks_to_run = []
    for model in passers:
        for level in expand_levels:
            if force or model not in completed or level not in completed[model]:
                tasks_to_run.append((model, level))

    if not tasks_to_run:
        print("No new (model, level) combinations to test")
        return

    print(f"Testing {len(tasks_to_run)} (model, level) combinations\n")

    # Initialize runner (with all passers as valid models)
    runner = ProbeRunner(
        output_dir=results_dir,
        models=passers,
        skip_validation=False,
    )

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create async tasks
    async_tasks = [
        run_model_level_async(model, level, runner, trials, semaphore)
        for model, level in tasks_to_run
    ]

    # Run with progress tracking
    probe_names = {0: "T0", 1: "T1", 2: "T2", 3: "A1", 4: "R0"}
    for idx, task in enumerate(asyncio.as_completed(async_tasks), 1):
        model, level, result = await task

        probe_name = probe_names.get(level, f"L{level}")
        if result:
            rate = result["summary"]["rate"]
            successes = result["summary"]["successes"]
            print(f"[{idx}/{len(tasks_to_run)}] {model} - {probe_name}: {successes}/{trials} ({rate:.0%})")
        else:
            print(f"[{idx}/{len(tasks_to_run)}] {model} - {probe_name}: NOT IMPLEMENTED")


async def run_phase_retest(
    results_dir: Path,
    trials: int,
    concurrency: int,
) -> None:
    """Phase 3: Re-test models that failed T0.

    Args:
        results_dir: Output directory
        trials: Number of trials per model
        concurrency: Max parallel executions
    """
    print(f"\n{'='*60}")
    print("PHASE RETEST: Re-run T0 Failers")
    print(f"{'='*60}\n")

    # Find T0 failers
    failers = get_t0_failers(results_dir)

    if not failers:
        print("No models failed T0 - all passed!")
        return

    print(f"Re-testing {len(failers)} models that failed T0\n")

    # Initialize runner
    runner = ProbeRunner(
        output_dir=results_dir,
        models=failers,
        skip_validation=False,
    )

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create async tasks
    tasks = [
        run_model_level_async(model, 0, runner, trials, semaphore)
        for model in failers
    ]

    # Run with progress tracking
    for idx, task in enumerate(asyncio.as_completed(tasks), 1):
        model, level, result = await task

        if result:
            rate = result["summary"]["rate"]
            successes = result["summary"]["successes"]
            status = "NOW PASS" if rate >= T0_EXPAND_THRESHOLD else "STILL FAIL"
            print(f"[{idx}/{len(failers)}] {model} - T0: {successes}/{trials} ({rate:.0%}) - {status}")
        else:
            print(f"[{idx}/{len(failers)}] {model} - T0: ERROR")


async def run_specific_models(
    models: list[str],
    results_dir: Path,
    trials: int,
    concurrency: int,
    force: bool,
) -> None:
    """Run T0 on specific models.

    Args:
        models: List of model IDs to test
        results_dir: Output directory
        trials: Number of trials per model
        concurrency: Max parallel executions
        force: If True, re-run existing results
    """
    print(f"\n{'='*60}")
    print(f"TESTING {len(models)} SPECIFIC MODELS")
    print(f"{'='*60}\n")

    # Filter out already-tested models
    if not force:
        completed = get_existing_results(results_dir)
        models = [m for m in models if m not in completed or 0 not in completed[m]]

    if not models:
        print("No models to test (all already completed)")
        return

    print(f"Testing {len(models)} models with {concurrency} concurrent executions\n")

    # Initialize runner
    runner = ProbeRunner(
        output_dir=results_dir,
        models=models,
        skip_validation=False,
    )

    # Create semaphore
    semaphore = asyncio.Semaphore(concurrency)

    # Create async tasks
    tasks = [
        run_model_level_async(model, 0, runner, trials, semaphore)
        for model in models
    ]

    # Run with progress tracking
    for idx, task in enumerate(asyncio.as_completed(tasks), 1):
        model, level, result = await task

        if result:
            rate = result["summary"]["rate"]
            successes = result["summary"]["successes"]
            status = "PASS" if rate >= T0_EXPAND_THRESHOLD else "FAIL"
            print(f"[{idx}/{len(models)}] {model} - T0: {successes}/{trials} ({rate:.0%}) - {status}")
        else:
            print(f"[{idx}/{len(models)}] {model} - T0: ERROR")


def main():
    parser = argparse.ArgumentParser(
        description="Smart tiered benchmarking with async parallel execution"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["t0", "expand", "retest"],
        help="Testing phase (t0, expand, retest)",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of specific models to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory (default: ./results)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per (model, level) (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_DEFAULT,
        help=f"Max parallel executions (default: {CONCURRENCY_DEFAULT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of existing results",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.phase and not args.models:
        parser.error("Either --phase or --models must be specified")

    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key from: https://openrouter.ai/keys")
        return 1

    # Initialize output directory
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run appropriate phase
    try:
        if args.models:
            # Test specific models
            model_list = [m.strip() for m in args.models.split(",")]
            asyncio.run(run_specific_models(
                models=model_list,
                results_dir=results_dir,
                trials=args.trials,
                concurrency=args.concurrency,
                force=args.force,
            ))
        elif args.phase == "t0":
            # Phase 1: T0 on all free models
            print("Fetching free models from OpenRouter...")
            models = get_free_models()
            print(f"Found {len(models)} free models\n")
            asyncio.run(run_phase_t0(
                models=models,
                results_dir=results_dir,
                trials=args.trials,
                concurrency=args.concurrency,
                force=args.force,
            ))
        elif args.phase == "expand":
            # Phase 2: Expand on T0 passers
            asyncio.run(run_phase_expand(
                results_dir=results_dir,
                trials=args.trials,
                concurrency=args.concurrency,
                force=args.force,
            ))
        elif args.phase == "retest":
            # Phase 3: Re-test T0 failers
            asyncio.run(run_phase_retest(
                results_dir=results_dir,
                trials=args.trials,
                concurrency=args.concurrency,
            ))

        print(f"\n{'='*60}")
        print(f"Results written to: {results_dir}")
        print(f"{'='*60}\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
