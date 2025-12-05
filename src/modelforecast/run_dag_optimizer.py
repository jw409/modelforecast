import argparse
import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from modelforecast.optimization.optimizer import DagOptimizer
from modelforecast.dags.interpreter import DagInterpreter, DagResult
from typing import List, Tuple

def run_trials(dag_def: dict, client: OpenAI, trials: int) -> Tuple[float, float, int]:
    """Runs a DAG definition multiple times and returns average cost and success rate."""
    total_cost = 0.0
    successful_runs = 0
    total_latency_ms = 0

    print(f"Running {trials} trials for DAG: {dag_def.get('workflow_id', 'unknown')}")
    for i in range(trials):
        print(f"  Trial {i+1}/{trials}...", end='', flush=True)
        interpreter = DagInterpreter(dag_def, client)
        result = interpreter.run()
        total_cost += result.total_cost
        total_latency_ms += result.total_latency_ms
        if result.success:
            successful_runs += 1
            print("SUCCESS")
        else:
            print(f"FAILED: {result.error}")
    
    avg_cost = total_cost / trials
    accuracy = (successful_runs / trials) * 100
    avg_latency = total_latency_ms / trials
    return avg_cost, accuracy, avg_latency

def evaluate_optimization(dag_path: Path, client: OpenAI, trials: int, optimization_iterations: int):
    """
    Evaluates the prompt optimization loop against baseline performance. 
    
    Success criteria:
    - 15% reduction in average cost per query.
    - 98% baseline accuracy maintained.
    """
    
    print("\n--- Starting Optimization Evaluation ---")
    print(f"DAG: {dag_path.name}")
    print(f"Trials per phase: {trials}")
    print(f"Optimization iterations: {optimization_iterations}")

    # 1. Baseline Phase
    print("\n--- Baseline Phase ---")
    with open(dag_path) as f:
        baseline_dag_def = json.load(f)
    
    baseline_avg_cost, baseline_accuracy, baseline_avg_latency = run_trials(baseline_dag_def, client, trials)
    print(f"\nBaseline Avg Cost: ${baseline_avg_cost:.6f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Baseline Avg Latency: {baseline_avg_latency:.2f}ms")

    # 2. Optimization Phase
    print("\n--- Optimization Phase ---")
    optimizer = DagOptimizer(baseline_dag_def, client) # Use a copy of the original DAG
    optimizer.optimize_loop(optimization_iterations)
    optimized_dag_def = optimizer.dag_def # Get the optimized DAG definition
    
    # Save optimized DAG for inspection (optional)
    optimized_dag_path = dag_path.parent / f"optimized_{dag_path.name}"
    with open(optimized_dag_path, 'w') as f:
        json.dump(optimized_dag_def, f, indent=2)
    print(f"\nOptimized DAG saved to: {optimized_dag_path}")

    # 3. Optimized Runs Phase
    print("\n--- Optimized Runs Phase ---")
    optimized_avg_cost, optimized_accuracy, optimized_avg_latency = run_trials(optimized_dag_def, client, trials)
    print(f"\nOptimized Avg Cost: ${optimized_avg_cost:.6f}")
    print(f"Optimized Accuracy: {optimized_accuracy:.2f}%")
    print(f"Optimized Avg Latency: {optimized_avg_latency:.2f}ms")

    # 4. Report and Validation
    print("\n--- Evaluation Report ---")
    cost_reduction_pct = ((baseline_avg_cost - optimized_avg_cost) / baseline_avg_cost) * 100 if baseline_avg_cost > 0 else 0
    accuracy_maintained = optimized_accuracy >= 98.0
    
    print(f"Cost Reduction: {cost_reduction_pct:.2f}% (Target: 15%)")
    print(f"Accuracy Maintained: {optimized_accuracy:.2f}% (Target: 98%)")

    success_cost = cost_reduction_pct >= 15.0
    success_accuracy = accuracy_maintained

    if success_cost and success_accuracy:
        print("\n✅ SUCCESS: Optimization met both cost reduction and accuracy targets!")
    elif success_cost:
        print("\n⚠️ PARTIAL SUCCESS: Cost reduction target met, but accuracy fell below 98%.")
    elif success_accuracy:
        print("\n⚠️ PARTIAL SUCCESS: Accuracy maintained, but cost reduction target not met.")
    else:
        print("\n❌ FAILURE: Neither cost reduction nor accuracy targets were met.")
        
    print("\n--- End Optimization Evaluation ---")


def main():
    parser = argparse.ArgumentParser(
        description="Run DAG Optimizer or evaluate optimization for a specific DAG definition."
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
    parser.add_argument(
        "--evaluate-optimization",
        action="store_true",
        help="Run full evaluation against baseline with cost and accuracy targets.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials for baseline and optimized runs during evaluation (default: 5).",
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
    
    if args.evaluate_optimization:
        evaluate_optimization(dag_path, client, args.trials, args.iterations)
    else:
        # Default behavior: just run the optimizer without full evaluation
        optimizer = DagOptimizer(dag_path, client)
        optimizer.optimize_loop(args.iterations)

if __name__ == "__main__":
    sys.exit(main())