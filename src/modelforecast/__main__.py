"""Entry point for python -m modelforecast."""

import argparse
import os
import sys
from pathlib import Path

from modelforecast import __version__
from modelforecast.models import (
    DEFAULT_ROSTER,
    ROSTER_LAST_VERIFIED,
    get_models,
    validate_model,
    validate_roster,
)
from modelforecast.runner import ProbeRunner
from modelforecast.sweep.orchestrator import SweepOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="ModelForecast - Tool-calling capability benchmarks for OpenRouter models"
    )
    parser.add_argument("--version", action="version", version=f"modelforecast {__version__}")

    # Subparsers — 'sweep' is the new subcommand; no subcommand = single-model/run-all mode
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.default = None

    # ------------------------------------------------------------------
    # sweep subcommand
    # ------------------------------------------------------------------
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run a full sweep of all models with checkpoint-resume and timestamped output",
    )
    sweep_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint.json if sweep was interrupted",
    )
    sweep_parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per (model, level) combination (default: 10)",
    )
    sweep_parser.add_argument(
        "--max-level",
        type=int,
        default=4,
        help="Maximum probe level to run (default: 4)",
    )
    sweep_parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Base results directory (default: ./results)",
    )
    sweep_parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Override auto-generated sweep ID (e.g., sweep_20260318)",
    )
    sweep_parser.add_argument(
        "--contributor",
        type=str,
        help="GitHub username for provenance (default: GITHUB_USERNAME env var)",
    )
    sweep_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model ID validation against OpenRouter API",
    )
    sweep_parser.add_argument(
        "--validate-roster",
        action="store_true",
        help="Validate the curated model roster against OpenRouter and exit",
    )
    sweep_parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="Confirm that the default roster may use paid OpenRouter endpoints",
    )

    # ------------------------------------------------------------------
    # Top-level flags (single-model / run-all mode — backward compatible)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument("--model", type=str, help="Specific model to test")

    # Mutually exclusive group for level selection
    level_group = parser.add_mutually_exclusive_group()
    level_group.add_argument(
        "--level", type=int, choices=[0, 1, 2, 3, 4, 5], help="Specific level to test (0-5)"
    )
    level_group.add_argument(
        "--probe",
        type=str,
        choices=["T0", "T1", "T2", "A1", "R0", "DAG"],
        help="Specific probe to test (T0=0, T1=1, T2=2, A1=3, R0=4, DAG=5)",
    )

    parser.add_argument(
        "--trials", type=int, default=10, help="Number of trials per probe (default: 10)"
    )
    parser.add_argument(
        "--contributor",
        type=str,
        help="GitHub username for provenance (default: GITHUB_USERNAME env var)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model ID validation against OpenRouter API",
    )
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="Confirm that an implicit full-roster run may use paid OpenRouter endpoints",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available tool-capable models and exit",
    )
    parser.add_argument(
        "--validate",
        type=str,
        metavar="MODEL_ID",
        help="Validate a model ID exists on OpenRouter and exit",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # sweep subcommand dispatch
    # ------------------------------------------------------------------
    if args.command == "sweep":
        print(f"ModelForecast v{__version__}")

        if not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY environment variable not set")
            print("Get your API key from: https://openrouter.ai/keys")
            return 1

        if args.validate_roster:
            print(
                "Validating the curated model roster against OpenRouter "
                f"(last verified {ROSTER_LAST_VERIFIED})..."
            )
            try:
                status = validate_roster()
                for model_id, (ok, message) in status.items():
                    marker = "OK" if ok else "FAIL"
                    print(f"  {marker:4} {model_id} — {message}")
                failures = sum(not ok for ok, _ in status.values())
                print(f"\n{len(status) - failures}/{len(status)} roster models ready")
                return 1 if failures else 0
            except Exception as e:
                print(f"ERROR: {e}")
                return 1

        if not args.confirm_spend:
            dimensions = args.max_level + 1
            extra_a1_calls = 1 if args.max_level >= 3 else 0
            max_requests = len(DEFAULT_ROSTER) * args.trials * (
                dimensions + extra_a1_calls
            )
            print("ERROR: A default-roster sweep may use paid OpenRouter endpoints.")
            print(
                f"This configuration can make up to approximately {max_requests} "
                "model requests."
            )
            print("Review the roster, then rerun with --confirm-spend.")
            return 2

        orchestrator = SweepOrchestrator(
            base_results_dir=Path(args.output),
            sweep_id=args.sweep_id,
        )
        runner = ProbeRunner(
            output_dir=orchestrator.sweep_dir,
            contributor=args.contributor,
            skip_validation=args.skip_validation,
        )
        orchestrator.run(
            runner=runner,
            trials=args.trials,
            max_level=args.max_level,
            resume=args.resume,
        )
        print(f"\nSweep complete. Results in: {orchestrator.sweep_dir}")
        return 0

    # ------------------------------------------------------------------
    # Single-model / run-all mode (backward compatible)
    # ------------------------------------------------------------------

    # Map --probe to --level internally
    probe_to_level = {
        "T0": 0,
        "T1": 1,
        "T2": 2,
        "A1": 3,
        "R0": 4,
        "DAG": 5,
    }
    if args.probe:
        args.level = probe_to_level[args.probe]

    print(f"ModelForecast v{__version__}")

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key from: https://openrouter.ai/keys")
        return 1

    # Handle --list-models
    if args.list_models:
        print("\nFetching available tool-capable models from OpenRouter...")
        try:
            models = get_models(tools_only=True)
            print(f"\nFound {len(models)} tool-capable models:\n")
            for model in models:
                print(f"  {model}")
            return 0
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    # Handle --validate
    if args.validate:
        print(f"\nValidating model: {args.validate}")
        is_valid, message = validate_model(args.validate)
        print(f"  {message}")
        return 0 if is_valid else 1

    # Initialize output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize runner
    models = [args.model] if args.model else None
    if models is None and not args.confirm_spend:
        max_requests = len(DEFAULT_ROSTER) * args.trials * 6
        print("ERROR: A default-roster run may use paid OpenRouter endpoints.")
        print(
            f"This configuration can make up to approximately {max_requests} "
            "model requests."
        )
        print("Select --model explicitly or rerun with --confirm-spend.")
        return 2

    runner = ProbeRunner(
        output_dir=output_dir,
        models=models,
        contributor=args.contributor,
        skip_validation=args.skip_validation,
    )

    try:
        if args.level is not None:
            # Run specific level only
            if args.model:
                # Single model, single level
                result = runner.run_level(args.model, args.level, args.trials)
                if result:
                    print(f"\nResults written to: {output_dir}")
                    return 0
                else:
                    print(f"\nERROR: Level {args.level} probe not implemented")
                    return 1
            else:
                # All models, single level
                all_results = {}
                for model in runner.models:
                    result = runner.run_level(model, args.level, args.trials)
                    if result:
                        result_key = f"{model}__level_{args.level}"
                        all_results[result_key] = result

                # Write reports
                from modelforecast.output.json_report import write_json_report
                from modelforecast.output.markdown_report import write_markdown_report

                write_json_report(output_dir, all_results)
                write_markdown_report(output_dir, all_results)
                print(f"\nResults written to: {output_dir}")
                return 0
        else:
            # Run all levels
            max_level = 5
            runner.run_all(trials=args.trials, max_level=max_level)
            print(f"\nResults written to: {output_dir}")
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
