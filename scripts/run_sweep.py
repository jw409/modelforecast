#!/usr/bin/env python3
"""Smart sweep runner with fail-fast canary and progress tracking.

Usage:
    uv run python scripts/run_sweep.py              # Full sweep
    uv run python scripts/run_sweep.py --canary     # Test 3 models first
    uv run python scripts/run_sweep.py --resume     # Resume interrupted sweep
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modelforecast.models import get_free_models
from modelforecast.runner import ProbeRunner


def run_canary(api_key: str, output_dir: Path) -> bool:
    """Run T0 on 3 diverse models. If >50% fail, abort."""
    # Use models known to support tool calling well for canary
    canary_models = [
        "nvidia/nemotron-nano-9b-v2:free",
        "openai/gpt-oss-20b:free",
    ]

    print("\n=== CANARY: Testing 2 models (T0 only, 3 trials each) ===")
    print("  Goal: verify API connectivity and probe pipeline work\n")

    runner = ProbeRunner(
        output_dir=output_dir,
        models=canary_models,
        skip_validation=True,
    )

    successes = 0
    for model in canary_models:
        try:
            result = runner.run_level(model, level=0, trials=3)
            if result is not None and result["summary"]["rate"] > 0:
                rate = int(result["summary"]["rate"] * 100)
                print(f"  OK:   {model} — {rate}% on T0")
                successes += 1
            else:
                print(f"  FAIL: {model} — 0% on T0 (model may not support tools)")
        except Exception as e:
            print(f"  ERROR: {model} — {e}")

    if successes == 0:
        print("\n=== CANARY FAILED: No models succeeded. API or pipeline broken. ===")
        return False

    print(f"\n=== CANARY PASSED: {successes}/{len(canary_models)} models OK. Pipeline works. ===\n")
    return True


def run_full_sweep(api_key: str, output_dir: Path, resume: bool = False) -> dict:
    """Run full sweep across all tool-capable free models."""
    models = get_free_models(api_key, tools_only=True)
    print(f"Found {len(models)} tool-capable free models")

    checkpoint_file = output_dir / "checkpoint.json"
    completed = set()

    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = set(json.load(f).get("completed_models", []))
        print(f"Resuming: {len(completed)} models already done")

    pending = [m for m in models if m not in completed]
    print(f"Pending: {len(pending)} models")

    # Single runner — shared rate limiter across all models
    runner = ProbeRunner(
        output_dir=output_dir,
        models=pending,
        skip_validation=True,
    )

    consecutive_failures = 0
    max_consecutive_failures = 3
    all_results = {}

    for i, model in enumerate(pending, 1):
        print(f"\n--- [{i}/{len(pending)}] {model} ---")

        try:
            model_results = runner.run_model(model, trials=10, max_level=4)

            if model_results:
                for level, result in model_results.items():
                    key = f"{model}__level_{level}"
                    all_results[key] = result
                consecutive_failures = 0
            else:
                print(f"  WARNING: No results for {model}")
                consecutive_failures += 1

        except Exception as e:
            print(f"  ERROR: {model} — {e}")
            consecutive_failures += 1

        # Update checkpoint
        completed.add(model)
        with open(checkpoint_file, "w") as f:
            json.dump({
                "completed_models": list(completed),
                "last_updated": datetime.now().isoformat(),
                "total_models": len(models),
            }, f, indent=2)

        # Fail-fast: 3 consecutive failures = something is wrong
        if consecutive_failures >= max_consecutive_failures:
            print(f"\n=== ABORT: {max_consecutive_failures} consecutive failures. ===")
            print("Check API key, rate limits, or model availability.")
            break

    return all_results


def main():
    parser = argparse.ArgumentParser(description="ModelForecast Smart Sweep")
    parser.add_argument("--canary", action="store_true", help="Run canary test only")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted sweep")
    parser.add_argument("--skip-canary", action="store_true", help="Skip canary, go straight to sweep")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    sweep_id = f"sweep_{datetime.now().strftime('%Y%m%d')}"
    output_dir = Path("results") / sweep_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.canary:
        success = run_canary(api_key, output_dir)
        sys.exit(0 if success else 1)

    if not args.skip_canary and not args.resume:
        if not run_canary(api_key, output_dir):
            sys.exit(1)

    results = run_full_sweep(api_key, output_dir, resume=args.resume)

    # Write manifest
    manifest = {
        "sweep_id": sweep_id,
        "date": datetime.now().isoformat(),
        "models_tested": len(set(k.split("__")[0] for k in results)),
        "total_results": len(results),
    }
    with open(output_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== SWEEP COMPLETE: {manifest['models_tested']} models, {manifest['total_results']} results ===")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
