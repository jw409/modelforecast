"""Entry point for python -m modelforecast."""

import argparse
import sys

from modelforecast import __version__


def main():
    parser = argparse.ArgumentParser(
        description="ModelForecast - Tool-calling capability benchmarks for free LLM models"
    )
    parser.add_argument("--version", action="version", version=f"modelforecast {__version__}")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4], help="Specific level to test")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per probe")

    args = parser.parse_args()

    print(f"ModelForecast v{__version__}")
    print("Probes not yet implemented - scaffold only")
    print(f"Would run with: output={args.output}, model={args.model}, level={args.level}")

    # TODO: Implement actual probe runner in Phase 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
