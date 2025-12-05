"""Entry point for python -m modelforecast."""

import argparse
import os
import sys
from pathlib import Path

from modelforecast import __version__
from modelforecast.models import get_available_models, get_free_models, validate_model
from modelforecast.runner import ProbeRunner


def main():
    parser = argparse.ArgumentParser(
        description="ModelForecast - Tool-calling capability benchmarks for free LLM models"
    )
    parser.add_argument("--version", action="version", version=f"modelforecast {__version__}")
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
        "--probe", type=str, choices=["T0", "T1", "T2", "A1", "R0", "DAG"],
        help="Specific probe to test (T0=0, T1=1, T2=2, A1=3, R0=4, DAG=5)"
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
        "--list-models",
        action="store_true",
        help="List available free models and exit",
    )
    parser.add_argument(
        "--validate",
        type=str,
        metavar="MODEL_ID",
        help="Validate a model ID exists on OpenRouter and exit",
    )

    args = parser.parse_args()

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
        print("\nFetching available free models from OpenRouter...")
        try:
            free_models = get_free_models()
            print(f"\nFound {len(free_models)} free models:\n")
            for model in free_models:
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
