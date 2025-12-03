"""JSON report generation for machine-readable results."""

import json
from pathlib import Path
from typing import Any


def write_json_report(
    output_dir: Path,
    results: dict[str, Any],
) -> Path:
    """Write machine-readable JSON report.

    Args:
        output_dir: Directory to write reports to
        results: Complete results dictionary with all model/level data

    Returns:
        Path to written JSON file

    The output structure is:
    {
        "metadata": {
            "version": "0.1.0",
            "generated_at": "2025-12-02T...",
            "total_models": 9,
            "total_levels": 5
        },
        "results": [
            {
                "submission_id": "sub_...",
                "model": "x-ai/grok-4.1-fast:free",
                "level": 0,
                "rate": 0.9,
                "wilson_ci_95": [0.55, 0.99],
                "trials": 10,
                "successes": 9
            },
            ...
        ]
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    # Build metadata
    unique_models = set()
    unique_levels = set()
    formatted_results = []

    # Process each result entry
    for result_key, result_data in results.items():
        if not isinstance(result_data, dict) or "summary" not in result_data:
            continue

        model = result_data.get("probes", {}).get("model", "unknown")
        level = result_data.get("probes", {}).get("level", -1)
        summary = result_data.get("summary", {})

        unique_models.add(model)
        unique_levels.add(level)

        formatted_results.append(
            {
                "submission_id": result_data.get("submission_id", "unknown"),
                "timestamp": result_data.get("timestamp", ""),
                "contributor": result_data.get("contributor", "unknown"),
                "model": model,
                "level": level,
                "rate": summary.get("rate", 0.0),
                "wilson_ci_95": summary.get("wilson_ci_95", [0.0, 1.0]),
                "trials": summary.get("trials", 0),
                "successes": summary.get("successes", 0),
            }
        )

    # Create report structure
    report = {
        "metadata": {
            "version": "0.1.0",
            "generated_at": timestamp,
            "total_models": len(unique_models),
            "total_levels": len(unique_levels),
        },
        "results": formatted_results,
    }

    # Write to file
    output_file = output_dir / "summary.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    return output_file


def write_individual_result(
    output_dir: Path,
    model: str,
    level: int,
    result: dict[str, Any],
) -> Path:
    """Write individual model+level result to separate JSON file.

    Args:
        output_dir: Directory to write results to
        model: Model identifier
        level: Probe level (0-4)
        result: Complete result dictionary with provenance

    Returns:
        Path to written JSON file

    Files are named: {model_slug}__level_{level}.json
    Example: grok-4-1-fast_free__level_0.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from model name
    model_slug = model.replace("/", "_").replace(":", "_").replace(".", "-")
    filename = f"{model_slug}__level_{level}.json"

    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    return output_file
