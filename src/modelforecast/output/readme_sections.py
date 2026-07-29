"""README section generators for ModelForecast sweep results.

Builds consumer-facing markdown sections from aggregated model results and
injects them into README.md between HTML comment markers. All functions are
idempotent — running them repeatedly produces identical output.
"""

import re
from pathlib import Path

GRADE_COLORS: dict[str, str] = {
    "A": "brightgreen",
    "B": "green",
    "C": "yellow",
    "D": "orange",
    "F": "red",
}

BADGE_BASE = "https://img.shields.io/badge"

# HTML comment markers for each section
MARKERS = {
    "QUICK-ANSWER": (
        "<!-- MODELFORECAST:QUICK-ANSWER:START -->",
        "<!-- MODELFORECAST:QUICK-ANSWER:END -->",
    ),
    "CATEGORY-WINNERS": (
        "<!-- MODELFORECAST:CATEGORY-WINNERS:START -->",
        "<!-- MODELFORECAST:CATEGORY-WINNERS:END -->",
    ),
    "AVOID": (
        "<!-- MODELFORECAST:AVOID:START -->",
        "<!-- MODELFORECAST:AVOID:END -->",
    ),
    "GRADE-BADGES": (
        "<!-- MODELFORECAST:GRADE-BADGES:START -->",
        "<!-- MODELFORECAST:GRADE-BADGES:END -->",
    ),
}


def _short_name(model_id: str) -> str:
    """Extract human-readable short name from a model ID.

    e.g. "openai/gpt-5.6-sol" -> "gpt-5.6-sol"
    """
    return model_id.split("/")[-1].replace(":free", "")


def _model_url(model_id: str) -> str:
    """Build OpenRouter model page URL (strips :free suffix)."""
    return f"https://openrouter.ai/models/{model_id.replace(':free', '')}"


def grade_badge_markdown(model_id: str, grade: str) -> str:
    """Return inline clickable badge markdown for a single model.

    Args:
        model_id: Full model identifier
        grade: Letter grade (A, B, C, D, F)

    Returns:
        Markdown string: [![Grade X](badge-url)](model-url)
    """
    color = GRADE_COLORS.get(grade, "lightgrey")
    short = _short_name(model_id)
    # Shields.io: hyphens in label need double-dash, dots need escaping
    label = short.replace("-", "--").replace(".", ".")
    badge_url = f"{BADGE_BASE}/{label}-Grade_{grade}-{color}"
    model_url = _model_url(model_id)
    return f"[![{short}: Grade {grade}]({badge_url})]({model_url})"


def _get_level_rate(model_data: dict, level: int) -> float | None:
    """Extract success rate for a given level from model_data dict."""
    lr = model_data.get("level_results", {})
    if level not in lr:
        return None
    summary = lr[level].get("summary", {})
    return summary.get("rate")


def _pick_top_model(model_summary: dict) -> tuple[str, dict] | None:
    """Select the top model from model_summary.

    Prefers grade A, then B. Among ties, picks highest T0 rate.
    Falls back to highest grade if no A/B exists.
    """
    preferred_grades = ["A", "B", "C", "D", "F"]
    for grade in preferred_grades:
        candidates = [
            (mid, data)
            for mid, data in model_summary.items()
            if data.get("grade") == grade
        ]
        if candidates:
            # Sort by T0 rate descending; use 0.0 as fallback
            candidates.sort(
                key=lambda x: _get_level_rate(x[1], 0) or 0.0,
                reverse=True,
            )
            return candidates[0]
    return None


def build_quick_answer_section(
    model_summary: dict,
    sweep_metadata: dict | None = None,
) -> str:
    """Build the QUICK-ANSWER section markdown.

    Identifies the top-graded model and emits a one-liner recommendation
    with a shields.io grade badge.

    Args:
        model_summary: Dict mapping model_id -> {grade, level_results, failure_modes}
        sweep_metadata: Optional sweep date, model count, and trial count

    Returns:
        Markdown string for injection between QUICK-ANSWER markers.
    """
    if not model_summary:
        return "_No sweep data available._"

    top = _pick_top_model(model_summary)
    if top is None:
        return "_No models evaluated._"

    model_id, model_data = top
    grade = model_data.get("grade", "F")
    color = GRADE_COLORS.get(grade, "lightgrey")
    short = _short_name(model_id)
    badge_url = f"{BADGE_BASE}/Grade-{grade}-{color}"
    model_id_for_link = model_id.replace(":free", "")
    metadata = sweep_metadata or {}
    sweep_date = metadata.get("sweep_date", "this")
    model_count = metadata.get("model_count")
    cohort = f" from a {model_count}-model cohort" if model_count else ""

    lines = [
        f"### Best model in the {sweep_date} sweep: {short}",
        "",
        f"![Grade {grade}]({badge_url}) — scored {grade} across all tool-calling dimensions.",
        "",
        f"> Benchmark result{cohort}: "
        f"[`{model_id}`](https://openrouter.ai/models/{model_id_for_link}). "
        "Validate current availability and rerun before deployment.",
        "",
        "Full results: [results/RESULTS.md](results/RESULTS.md)"
        " · [Methodology](docs/METHODOLOGY.md)",
    ]
    return "\n".join(lines)


def build_category_winners_section(model_summary: dict) -> str:
    """Build the CATEGORY-WINNERS section markdown.

    Derives winners from existing level scores — no new probes needed.

    Categories:
    - Best for tool calling (T0+T1): highest avg of level 0+1 rates, A/B only
    - Best for schema compliance (T1): highest level 1 rate
    - Best for restraint (R0): highest level 4 rate
    - Best for multi-turn agency (A1): highest level 3 rate

    Args:
        model_summary: Dict mapping model_id -> {grade, level_results, failure_modes}

    Returns:
        Markdown string for injection between CATEGORY-WINNERS markers.
    """
    if not model_summary:
        return "_No sweep data available._"

    def _best_by(
        level: int,
        *,
        require_ab: bool = False,
        secondary: int | None = None,
    ) -> tuple[str, float] | None:
        """Find model with best rate for given level.

        Returns (model_id, rate) or None if fewer than 2 models have data.
        """
        scored: list[tuple[str, float]] = []
        for mid, data in model_summary.items():
            if require_ab and data.get("grade") not in ("A", "B"):
                continue
            rate = _get_level_rate(data, level)
            if rate is None:
                continue
            if secondary is not None:
                sec_rate = _get_level_rate(data, secondary)
                if sec_rate is None:
                    continue
                score = (rate + sec_rate) / 2
            else:
                score = rate
            scored.append((mid, score))

        if len(scored) < 2:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    categories = [
        (
            "tool calling (T0+T1)",
            _best_by(0, require_ab=True, secondary=1),
            "T0+T1",
        ),
        (
            "schema compliance (T1)",
            _best_by(1),
            "T1",
        ),
        (
            "restraint (R0)",
            _best_by(4),
            "R0",
        ),
        (
            "multi-turn agency (A1)",
            _best_by(3),
            "A1",
        ),
    ]

    lines = ["### Category winners"]
    lines.append("")
    any_winner = False
    for category, result, probe_label in categories:
        if result is None:
            continue
        mid, score = result
        short = _short_name(mid)
        url = _model_url(mid)
        rate_pct = int(score * 100)
        lines.append(
            f"- **Best for {category}**: [{short}]({url}) — {rate_pct}% {probe_label}"
        )
        any_winner = True

    if not any_winner:
        return "_Insufficient data for category winners._"

    return "\n".join(lines)


def build_avoid_section(model_summary: dict) -> str:
    """Build the AVOID section markdown.

    Lists models graded F (T0 < 20%) with their failure mode.

    Args:
        model_summary: Dict mapping model_id -> {grade, level_results, failure_modes}

    Returns:
        Markdown string for injection between AVOID markers.
    """
    header = [
        "### Avoid these models for tool calling",
        "",
        "These models fail the basic tool invocation test (T0 < 20%)."
        " They will silently fail your agent pipeline.",
    ]

    f_models = [
        (mid, data)
        for mid, data in model_summary.items()
        if data.get("grade") == "F"
    ]

    if not f_models:
        return "\n".join(
            [
                "### Avoid these models for tool calling",
                "",
                "No models scored F in this sweep.",
            ]
        )

    f_models.sort(key=lambda x: x[0])

    lines = header[:]
    lines.append("")
    for mid, data in f_models:
        short = _short_name(mid)
        rate = _get_level_rate(data, 0)
        rate_pct = int((rate or 0.0) * 100)

        failure_modes = data.get("failure_modes", [])
        if failure_modes:
            from collections import Counter

            most_common = Counter(failure_modes).most_common(1)[0][0]
        else:
            most_common = "no tool_call produced"

        lines.append(f"- **{short}**: {most_common} (T0={rate_pct}%)")

    return "\n".join(lines)


def build_grade_badges_section(model_summary: dict) -> str:
    """Build the GRADE-BADGES section markdown.

    Emits one clickable badge per grade-A and grade-B model.

    Args:
        model_summary: Dict mapping model_id -> {grade, level_results, failure_modes}

    Returns:
        Markdown string for injection between GRADE-BADGES markers.
    """
    top_models = [
        (mid, data)
        for mid, data in sorted(model_summary.items())
        if data.get("grade") in ("A", "B")
    ]

    if not top_models:
        return "_No grade-A or grade-B models in this sweep._"

    badges = [grade_badge_markdown(mid, data["grade"]) for mid, data in top_models]

    lines = [
        "### Top performers",
        "",
        " ".join(badges),
        "",
        "*Click a badge to view the model on OpenRouter.*",
    ]
    return "\n".join(lines)


def _inject_section(content: str, section_key: str, new_body: str) -> str:
    """Replace content between a pair of HTML comment markers.

    If markers are not found in content, appends the full block (marker +
    body + end marker) at the end of the document.

    Args:
        content: Full README text
        section_key: Key from MARKERS dict (e.g. "QUICK-ANSWER")
        new_body: Replacement markdown to place between markers

    Returns:
        Updated README text
    """
    start_marker, end_marker = MARKERS[section_key]

    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )

    replacement = f"{start_marker}\n{new_body}\n{end_marker}"

    if pattern.search(content):
        return pattern.sub(replacement, content)

    # Markers not found — append at end
    return content.rstrip("\n") + f"\n\n{replacement}\n"


def update_readme_sections(
    readme_path: Path,
    model_summary: dict,
    sweep_metadata: dict,
) -> None:
    """Inject all generated sections into README.md between HTML comment markers.

    Idempotent — re-running replaces existing section content.

    Args:
        readme_path: Path to README.md
        model_summary: Dict mapping model_id -> {grade, level_results, failure_modes}
        sweep_metadata: Dict with sweep_date, model_count, trials_per_level
    """
    content = readme_path.read_text(encoding="utf-8")

    content = _inject_section(
        content,
        "QUICK-ANSWER",
        build_quick_answer_section(model_summary, sweep_metadata),
    )
    content = _inject_section(
        content, "GRADE-BADGES", build_grade_badges_section(model_summary)
    )
    content = _inject_section(
        content, "CATEGORY-WINNERS", build_category_winners_section(model_summary)
    )
    content = _inject_section(
        content, "AVOID", build_avoid_section(model_summary)
    )

    readme_path.write_text(content, encoding="utf-8")
