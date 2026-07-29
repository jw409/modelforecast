#!/usr/bin/env python3
"""Generate results/RESULTS.md from the latest sweep directory.

Usage:
    uv run python scripts/generate_readme_results.py
    uv run python scripts/generate_readme_results.py --sweep results/sweep_20260318
    uv run python scripts/generate_readme_results.py --out results/RESULTS.md
"""
import argparse
import json
from pathlib import Path

from rich.console import Console

from modelforecast.output.markdown_report import (
    calculate_grade,
    find_latest_sweep_dir,
    write_markdown_report,
)
from modelforecast.output.readme_sections import update_readme_sections

console = Console()


def parse_sweep_date(sweep_dir: Path) -> str:
    """Extract ISO date from sweep directory name.

    e.g. sweep_20260318 -> "2026-03-18"
    """
    name = sweep_dir.name
    if name.startswith("sweep_"):
        raw = name[len("sweep_"):]
        # Expect YYYYMMDD format
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return name


def load_sweep_results(
    sweep_dir: Path,
) -> tuple[dict[str, dict], int, int]:
    """Load all result JSON files from a sweep directory.

    Args:
        sweep_dir: Path to sweep_YYYYMMDD directory

    Returns:
        Tuple of (results dict, model_count, trials_per_level)
    """
    results: dict[str, dict] = {}
    skip_files = {"sweep_manifest.json", "checkpoint.json"}

    for json_file in sorted(sweep_dir.glob("*.json")):
        if json_file.name in skip_files:
            continue
        with open(json_file) as f:
            data = json.load(f)
        results[json_file.stem] = data

    # Count distinct models
    model_ids = set()
    for result in results.values():
        if isinstance(result, dict) and "probes" in result:
            model_ids.add(result["probes"]["model"])
    model_count = len(model_ids)

    # Try manifest for trials_per_level; fall back to first result
    trials_per_level = 10
    manifest_path = sweep_dir / "sweep_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if "trials_per_level" in manifest:
            trials_per_level = manifest["trials_per_level"]
    elif results:
        # Infer from first result's summary.trials
        first = next(iter(results.values()))
        if isinstance(first, dict) and "summary" in first:
            trials_per_level = first["summary"].get("trials", 10)

    return results, model_count, trials_per_level


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate results/RESULTS.md from sweep JSON files."
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Explicit sweep directory path (default: auto-detect latest)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/RESULTS.md"),
        help="Output file path (default: results/RESULTS.md)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("results"),
        help="Base results directory for auto-detection (default: results/)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="README file to update (default: README.md)",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Generate the results report without updating a README",
    )
    args = parser.parse_args()

    # Resolve sweep directory
    if args.sweep is not None:
        sweep_dir = args.sweep
        if not sweep_dir.exists():
            console.print(f"[red]Error:[/] Sweep directory not found: {sweep_dir}")
            raise SystemExit(1)
    else:
        sweep_dir = find_latest_sweep_dir(args.base)

    console.print(f"[dim]Reading sweep:[/] {sweep_dir}")

    # Load results
    results, model_count, trials_per_level = load_sweep_results(sweep_dir)
    sweep_date = parse_sweep_date(sweep_dir)

    console.print(
        f"[dim]Loaded {len(results)} result files, "
        f"{model_count} models, {trials_per_level} trials/level[/]"
    )

    # Write report
    out_path = args.out
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written = write_markdown_report(
        output_dir=out_dir,
        results=results,
        sweep_date=sweep_date,
        model_count=model_count,
        trials_per_level=trials_per_level,
    )

    # If --out differs from the default RESULTS.md location, rename
    default_output = out_dir / "RESULTS.md"
    if out_path != default_output and written != out_path:
        written.rename(out_path)
        written = out_path

    console.print(f"[green]Wrote {written}[/]")

    # Build model_summary from the results dict already loaded
    model_summary: dict[str, dict] = {}
    for result_data in results.values():
        if not isinstance(result_data, dict) or "probes" not in result_data:
            continue
        model = result_data["probes"]["model"]
        level = result_data["probes"]["level"]
        if model not in model_summary:
            model_summary[model] = {"level_results": {}, "failure_modes": []}
        model_summary[model]["level_results"][level] = result_data
        # Collect failure modes from trial data if present
        for trial in result_data.get("probes", {}).get("trials", []):
            if fm := trial.get("failure_mode"):
                model_summary[model]["failure_modes"].append(fm)

    # Calculate grades
    for model_data in model_summary.values():
        model_data["grade"] = calculate_grade(model_data["level_results"])

    sweep_metadata = {
        "sweep_date": sweep_date,
        "model_count": model_count,
        "trials_per_level": trials_per_level,
    }

    readme_path = args.readme
    if args.no_readme:
        console.print("[dim]README update disabled[/dim]")
    elif readme_path.exists():
        update_readme_sections(readme_path, model_summary, sweep_metadata)
        console.print(f"[green]✓ Updated {readme_path} sections[/green]")
    else:
        console.print(f"[yellow]{readme_path} not found — skipping README update[/yellow]")


if __name__ == "__main__":
    main()
