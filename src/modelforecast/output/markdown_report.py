"""Markdown report generation for human-readable results."""

from pathlib import Path
from typing import Any


def calculate_grade(level_results: dict[int, dict[str, Any]]) -> str:
    """Calculate letter grade based on performance across levels.

    Grading rubric:
    - A: L0 >= 80%, L1 >= 70%, no level below 50%
    - B: L0 >= 60%, L1 >= 50%, no level below 30%
    - C: L0 >= 40%, at least one level above 50%
    - D: L0 >= 20%, or any success at higher levels
    - F: L0 < 20% (cannot reliably call tools at all)

    Args:
        level_results: Dictionary mapping level -> result data

    Returns:
        Letter grade (A, B, C, D, or F)
    """
    if not level_results or 0 not in level_results:
        return "F"

    # Extract rates (as percentages 0-100)
    rates = {
        level: result["summary"]["rate"] * 100 for level, result in level_results.items()
    }

    l0 = rates.get(0, 0)
    l1 = rates.get(1, 0)
    min_rate = min(rates.values()) if rates else 0
    max_rate = max(rates.values()) if rates else 0

    # Apply grading rubric
    if l0 >= 80 and l1 >= 70 and min_rate >= 50:
        return "A"
    elif l0 >= 60 and l1 >= 50 and min_rate >= 30:
        return "B"
    elif l0 >= 40 and max_rate >= 50:
        return "C"
    elif l0 >= 20 or any(rate > 0 for level, rate in rates.items() if level > 0):
        return "D"
    else:
        return "F"


def format_percentage_with_ci(
    rate: float, ci: tuple[float, float], trials: int
) -> str:
    """Format success rate with confidence interval.

    Args:
        rate: Success rate (0.0 to 1.0)
        ci: Wilson confidence interval (lower, upper) as 0.0-1.0
        trials: Number of trials

    Returns:
        Formatted string like "90% [76,97]"
    """
    rate_pct = int(rate * 100)
    ci_lower = int(ci[0] * 100)
    ci_upper = int(ci[1] * 100)

    return f"{rate_pct}% [{ci_lower},{ci_upper}]"


def write_markdown_report(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
) -> Path:
    """Write human-readable markdown report.

    Args:
        output_dir: Directory to write report to
        results: Complete results dictionary with all model/level data

    Returns:
        Path to written markdown file

    Output format:
    | Model | L0 Basic | L1 Schema | L2 Select | L3 Multi | L4 Advers | Grade |
    |-------|----------|-----------|-----------|----------|-----------|-------|
    | grok-4.1-fast:free | 90% [76,97] | 85% [62,96] | ... | ... | ... | **A** |
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Organize results by model
    model_results: dict[str, dict[int, dict[str, Any]]] = {}

    for result_key, result_data in results.items():
        if not isinstance(result_data, dict) or "probes" not in result_data:
            continue

        model = result_data["probes"]["model"]
        level = result_data["probes"]["level"]

        if model not in model_results:
            model_results[model] = {}

        model_results[model][level] = result_data

    # Build markdown table
    lines = [
        "# ModelForecast Results",
        "",
        "Tool-calling capability benchmarks for free LLM models.",
        "",
        "| Model | L0 Basic | L1 Schema | L2 Select | L3 Multi | L4 Advers | Grade |",
        "|-------|----------|-----------|-----------|----------|-----------|-------|",
    ]

    # Sort models by name
    for model in sorted(model_results.keys()):
        level_data = model_results[model]

        # Create short model name (remove :free suffix for readability)
        short_model = model.replace(":free", "")

        row = [short_model]

        # Add columns for each level
        for level in range(5):
            if level in level_data:
                result = level_data[level]
                summary = result["summary"]
                cell = format_percentage_with_ci(
                    summary["rate"],
                    tuple(summary["wilson_ci_95"]),
                    summary["trials"],
                )
                row.append(cell)
            else:
                row.append("-")

        # Calculate and add grade
        grade = calculate_grade(level_data)
        row.append(f"**{grade}**")

        lines.append("| " + " | ".join(row) + " |")

    # Add footer notes
    lines.extend(
        [
            "",
            "*Percentages show success rate. Brackets show 95% Wilson CI.*",
            "*n=10 per cell. \"-\" indicates not tested (prerequisite level failed).*",
            "",
            "## Grading Rubric",
            "",
            "- **A**: L0 >= 80%, L1 >= 70%, no level below 50%",
            "- **B**: L0 >= 60%, L1 >= 50%, no level below 30%",
            "- **C**: L0 >= 40%, at least one level above 50%",
            "- **D**: L0 >= 20%, or any success at higher levels",
            "- **F**: L0 < 20% (cannot reliably call tools at all)",
        ]
    )

    # Write to file
    output_file = output_dir / "RESULTS.md"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return output_file
