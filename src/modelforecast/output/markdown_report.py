"""Markdown report generation for human-readable results."""

from pathlib import Path
from typing import Any


def find_latest_sweep_dir(base_results_dir: Path) -> Path:
    """Find the most recent sweep directory in base_results_dir.

    Args:
        base_results_dir: Parent directory containing sweep_* subdirectories

    Returns:
        Path to the latest sweep directory (lexicographic sort on sweep_YYYYMMDD names)

    Raises:
        FileNotFoundError: If no sweep directory exists in base_results_dir
    """
    sweep_dirs = sorted(base_results_dir.glob("sweep_*"))
    if not sweep_dirs:
        raise FileNotFoundError(f"No sweep directory found in {base_results_dir}")
    return sweep_dirs[-1]


def cis_overlap(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Check whether two 95% Wilson confidence intervals overlap.

    Args:
        ci_a: First CI as (lower, upper) in 0.0-1.0
        ci_b: Second CI as (lower, upper) in 0.0-1.0

    Returns:
        True if the intervals overlap
    """
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def calculate_grade(level_results: dict[int, dict[str, Any]]) -> str:
    """Calculate letter grade based on performance across probes.

    Grading rubric:
    - A: T0 >= 80%, T1 >= 70%, no probe below 50%
    - B: T0 >= 60%, T1 >= 50%, no probe below 30%
    - C: T0 >= 40%, at least one probe above 50%
    - D: T0 >= 20%, or any success at higher probes
    - F: T0 < 20% (cannot reliably call tools at all)

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
    sweep_date: str = "",
    model_count: int = 0,
    trials_per_level: int = 10,
) -> Path:
    """Write human-readable markdown report.

    Args:
        output_dir: Directory to write report to
        results: Complete results dictionary with all model/level data
        sweep_date: ISO date string for the sweep (e.g. "2026-03-18"), or "" to omit
        model_count: Number of models in sweep, used in metadata line
        trials_per_level: Trial count per level, used in metadata line

    Returns:
        Path to written markdown file

    Output format:
    | Model | T0 Invoke | T1 Schema | T2 Select | A1 Linear | R0 Abstain | Grade |
    |-------|-----------|-----------|-----------|-----------|------------|-------|
    | [grok-4.1-fast](https://openrouter.ai/models/x-ai/grok-4.1-fast) | 90% [76,97] | ... | **A** |
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

    # Calculate grades for all models
    model_grades: dict[str, str] = {
        model: calculate_grade(level_data) for model, level_data in model_results.items()
    }

    # Detect statistical ties within each grade group at T0 (level 0)
    # A model is marked as a tie if its T0 CI overlaps with any other model in the same grade
    tied_models: set[str] = set()
    grade_groups: dict[str, list[str]] = {}
    for model, grade in model_grades.items():
        grade_groups.setdefault(grade, []).append(model)

    for grade, models_in_grade in grade_groups.items():
        if len(models_in_grade) < 2:
            continue
        # Collect T0 CIs for models that have level-0 data
        t0_cis: dict[str, tuple[float, float]] = {}
        for model in models_in_grade:
            level_data = model_results[model]
            if 0 in level_data:
                raw_ci = level_data[0]["summary"]["wilson_ci_95"]
                t0_cis[model] = (raw_ci[0], raw_ci[1])
        # Mark models whose T0 CI overlaps with any other in the same grade
        models_with_ci = list(t0_cis.keys())
        for i, model_a in enumerate(models_with_ci):
            for model_b in models_with_ci[i + 1 :]:
                if cis_overlap(t0_cis[model_a], t0_cis[model_b]):
                    tied_models.add(model_a)
                    tied_models.add(model_b)

    # Build markdown table
    lines = [
        "# ModelForecast Results",
        "",
        "Tool-calling capability benchmarks for free LLM models.",
        "",
    ]

    # Metadata line (only if sweep_date provided)
    if sweep_date:
        meta_parts = [f"Sweep: {sweep_date}"]
        if model_count:
            meta_parts.append(f"{model_count} models")
        meta_parts.append(f"n={trials_per_level} trials per level")
        lines.append(f"*{' · '.join(meta_parts)}*")
        lines.append("")

    lines.extend(
        [
            "| Model | T0 Invoke | T1 Schema | T2 Select | A1 Linear | R0 Abstain | Grade |",
            "|-------|-----------|-----------|-----------|-----------|------------|-------|",
        ]
    )

    has_ties = False

    # Sort models by name
    for model in sorted(model_results.keys()):
        level_data = model_results[model]

        # Build OpenRouter link — strip :free suffix for URL
        short_model = model.replace(":free", "")
        model_id_for_link = short_model
        model_link = f"[{short_model}](https://openrouter.ai/models/{model_id_for_link})"

        row = [model_link]

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

        # Calculate grade cell with optional tie marker
        grade = model_grades[model]
        if model in tied_models:
            grade_cell = f"**{grade}†**"
            has_ties = True
        else:
            grade_cell = f"**{grade}**"
        row.append(grade_cell)

        lines.append("| " + " | ".join(row) + " |")

    # Tie footnote (only if any ties found)
    if has_ties:
        lines.append("")
        lines.append("*† Statistical tie: overlapping 95% CI at T0.*")

    # Add footer notes
    lines.extend(
        [
            "",
            "*Percentages show success rate. Brackets show 95% Wilson CI.*",
            "*n=10 per cell. \"-\" indicates not tested (prerequisite probe failed).*",
            "",
            "## Grading Rubric",
            "",
            "- **A**: T0 >= 80%, T1 >= 70%, no probe below 50%",
            "- **B**: T0 >= 60%, T1 >= 50%, no probe below 30%",
            "- **C**: T0 >= 40%, at least one probe above 50%",
            "- **D**: T0 >= 20%, or any success at higher probes",
            "- **F**: T0 < 20% (cannot reliably call tools at all)",
        ]
    )

    # Write to file
    output_file = output_dir / "RESULTS.md"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return output_file
