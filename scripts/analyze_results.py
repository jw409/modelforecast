#!/usr/bin/env python3
"""
Analyze benchmark results and output structured JSON for content generation.

Reads results/*.json files, aggregates metrics per model, generates:
- analysis/metrics.json: Raw aggregated metrics
- analysis/rankings.json: Rankings and production readiness
- analysis/insights.json: Pattern detection (agency cliff, hidden champions, etc.)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict


# Map old level names to new dimension names (T0-T2, A1, R0)
LEVEL_TO_DIMENSION = {
    0: "T0",  # Tool invocation
    1: "T1",  # Schema compliance
    2: "T2",  # Tool selection
    3: "A1",  # Agency (multi-step)
    4: "R0",  # Restraint
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all result files and group by model.

    Returns: {
        "model_name": {
            "T0": {"rate": 1.0, "trials": 10, "latencies": [...]},
            "T1": {...},
            ...
        }
    }
    """
    models = defaultdict(lambda: defaultdict(dict))

    for result_file in results_dir.glob("*.json"):
        # Skip non-result files
        if result_file.name in [".gitkeep", "phase3_summary.csv", "grok_l3_final_comparison.json", "grok_l3_fix_comparison.json"]:
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Extract model and level
            if "probes" not in data:
                continue

            model = data["probes"]["model"]
            level = data["probes"]["level"]
            dimension = LEVEL_TO_DIMENSION.get(level, f"L{level}")

            # Extract metrics
            summary = data.get("summary", {})
            trials_data = data["probes"].get("trials", [])

            # Calculate latencies
            latencies = [t.get("latency_ms", 0) for t in trials_data if "latency_ms" in t]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            models[model][dimension] = {
                "rate": summary.get("rate", 0.0),
                "trials": summary.get("trials", 0),
                "avg_latency_ms": int(avg_latency),
                "wilson_ci_95": summary.get("wilson_ci_95", [0.0, 1.0])
            }

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {result_file.name}: {e}")
            continue

    return dict(models)


def generate_metrics(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate metrics.json output."""
    return {
        "models": models,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }


def calculate_rankings(models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate rankings.json output.

    Rankings by:
    - Overall: Weighted average of all dimensions
    - By dimension: Separate ranking for each T0, T1, T2, A1, R0
    - Production ready: All probes >= 90%
    - Marginal: T0 >= 50% but not production ready
    - Broken: T0 < 50%
    """
    all_dimensions = ["T0", "T1", "T2", "A1", "R0"]

    # Calculate overall scores (weighted average)
    # Weights: T0=2x (prerequisite), others 1x
    overall_scores = {}
    for model, dims in models.items():
        weights = {"T0": 2.0, "T1": 1.0, "T2": 1.0, "A1": 1.0, "R0": 1.0}
        total_weight = 0
        weighted_sum = 0

        for dim in all_dimensions:
            if dim in dims:
                rate = dims[dim]["rate"]
                weight = weights.get(dim, 1.0)
                weighted_sum += rate * weight
                total_weight += weight

        overall_scores[model] = weighted_sum / total_weight if total_weight > 0 else 0.0

    overall_ranking = sorted(overall_scores.keys(), key=lambda m: overall_scores[m], reverse=True)

    # Rankings by dimension
    by_dimension = {}
    for dim in all_dimensions:
        dim_scores = {m: dims[dim]["rate"] for m, dims in models.items() if dim in dims}
        by_dimension[dim] = sorted(dim_scores.keys(), key=lambda m: dim_scores[m], reverse=True)

    # Production ready: all tested dimensions >= 90%
    production_ready = []
    for model, dims in models.items():
        if all(dims[d]["rate"] >= 0.90 for d in dims):
            production_ready.append(model)
    production_ready.sort(key=lambda m: overall_scores[m], reverse=True)

    # Marginal: T0 >= 50% but not production ready
    marginal = []
    for model, dims in models.items():
        if "T0" in dims and dims["T0"]["rate"] >= 0.50 and model not in production_ready:
            marginal.append(model)
    marginal.sort(key=lambda m: overall_scores[m], reverse=True)

    # Broken: T0 < 50%
    broken = []
    for model, dims in models.items():
        if "T0" in dims and dims["T0"]["rate"] < 0.50:
            broken.append(model)
    broken.sort(key=lambda m: overall_scores[m], reverse=True)

    return {
        "overall": overall_ranking,
        "by_dimension": by_dimension,
        "production_ready": production_ready,
        "marginal": marginal,
        "broken": broken
    }


def detect_patterns(models: Dict[str, Dict[str, Any]], rankings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect interesting patterns:
    - Agency cliff: Passes T0-T2 >= 90% but A1 < 50%
    - Hidden champion: Low-star model outperforms high-star models
    - Speed leader: Lowest latency among production_ready
    - Graveyard: T0 = 0%
    """
    patterns = []

    # Agency cliff detection
    for model, dims in models.items():
        t0_pass = dims.get("T0", {}).get("rate", 0) >= 0.90
        t1_pass = dims.get("T1", {}).get("rate", 0) >= 0.90
        t2_pass = dims.get("T2", {}).get("rate", 0) >= 0.90
        a1_fail = dims.get("A1", {}).get("rate", 1.0) < 0.50

        if t0_pass and t1_pass and t2_pass and a1_fail:
            patterns.append({
                "type": "agency_cliff",
                "model": model,
                "description": f"Passes T0-T2 (>= 90%) but fails A1 (< 50%)",
                "metrics": {
                    "T0": dims.get("T0", {}).get("rate", 0),
                    "T1": dims.get("T1", {}).get("rate", 0),
                    "T2": dims.get("T2", {}).get("rate", 0),
                    "A1": dims.get("A1", {}).get("rate", 0)
                }
            })

    # Speed leader (lowest latency among production ready)
    production_ready = rankings["production_ready"]
    if production_ready:
        latencies = {}
        for model in production_ready:
            # Average latency across all dimensions
            all_latencies = [dims["avg_latency_ms"] for dims in models[model].values() if dims["avg_latency_ms"] > 0]
            if all_latencies:
                latencies[model] = sum(all_latencies) / len(all_latencies)

        if latencies:
            speed_leader = min(latencies.keys(), key=lambda m: latencies[m])
            patterns.append({
                "type": "speed_leader",
                "model": speed_leader,
                "latency_ms": int(latencies[speed_leader]),
                "description": f"Lowest latency among production-ready models ({int(latencies[speed_leader])}ms avg)"
            })

    # Graveyard (T0 = 0%)
    for model, dims in models.items():
        if "T0" in dims and dims["T0"]["rate"] == 0.0:
            patterns.append({
                "type": "graveyard",
                "model": model,
                "description": "Cannot invoke tools (T0 = 0%)"
            })

    # Hidden champion: Detect if kat-coder-pro beats larger models
    # (This is domain-specific, looking for kat-coder-pro specifically)
    kat_coder_models = [m for m in models.keys() if "kat-coder" in m.lower()]
    if kat_coder_models:
        kat_model = kat_coder_models[0]
        # Check if it beats grok or deepseek in overall ranking
        overall_ranking = rankings["overall"]
        kat_position = overall_ranking.index(kat_model) if kat_model in overall_ranking else -1

        beaten_models = []
        for other_model in overall_ranking[kat_position + 1:] if kat_position >= 0 else []:
            if "grok" in other_model.lower() or "deepseek" in other_model.lower():
                beaten_models.append(other_model)

        if beaten_models:
            patterns.append({
                "type": "hidden_champion",
                "model": kat_model,
                "beats": beaten_models,
                "description": f"Outperforms {len(beaten_models)} higher-profile models"
            })

    return {"patterns": patterns}


def main():
    """Main analysis pipeline."""
    # Paths
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    analysis_dir = repo_root / "analysis"

    # Create analysis directory
    analysis_dir.mkdir(exist_ok=True)

    # Load results
    print("Loading results...")
    models = load_results(results_dir)
    print(f"Loaded {len(models)} models")

    # Generate metrics.json
    print("Generating metrics.json...")
    metrics = generate_metrics(models)
    with open(analysis_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {len(models)} models analyzed")

    # Generate rankings.json
    print("Generating rankings.json...")
    rankings = calculate_rankings(models)
    with open(analysis_dir / "rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)
    print(f"  ✓ Production ready: {len(rankings['production_ready'])}")
    print(f"  ✓ Marginal: {len(rankings['marginal'])}")
    print(f"  ✓ Broken: {len(rankings['broken'])}")

    # Generate insights.json
    print("Generating insights.json...")
    insights = detect_patterns(models, rankings)
    with open(analysis_dir / "insights.json", "w") as f:
        json.dump(insights, f, indent=2)
    print(f"  ✓ Patterns detected: {len(insights['patterns'])}")

    print(f"\nAnalysis complete! Files written to {analysis_dir}/")


if __name__ == "__main__":
    main()
