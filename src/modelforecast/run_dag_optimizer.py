import argparse
import os
import sys
from pathlib import Path
from openai import OpenAI
from modelforecast.optimization.optimizer import DagOptimizer

def main():
    parser = argparse.ArgumentParser(
        description="Run DAG Optimizer for a specific DAG definition."
    )
    parser.add_argument(
        "--dag",
        type=str,
        required=True,
        help="Path to the DAG definition JSON file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of optimization iterations (default: 3).",
    )
    args = parser.parse_args()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        print("Get your API key from: https://openrouter.ai/keys")
        return 1

    # Initialize OpenAI client with OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    dag_path = Path(args.dag)
    if not dag_path.exists():
        print(f"ERROR: DAG file not found at {dag_path}")
        return 1
    
    optimizer = DagOptimizer(dag_path, client)
    optimizer.optimize_loop(args.iterations)

if __name__ == "__main__":
    sys.exit(main())

