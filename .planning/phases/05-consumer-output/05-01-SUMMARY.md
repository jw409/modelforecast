---
phase: 05-consumer-output
plan: "01"
subsystem: output
tags: [markdown, reporting, sweep-results, consumer-output]
dependency_graph:
  requires: [results/sweep_20260318/*.json]
  provides: [scripts/generate_readme_results.py, src/modelforecast/output/markdown_report.py, results/RESULTS.md]
  affects: [README.md consumers]
tech_stack:
  added: [ruff (dev dependency)]
  patterns: [Wilson CI overlap detection, OpenRouter model links, sweep metadata injection]
key_files:
  created:
    - scripts/generate_readme_results.py
    - results/RESULTS.md
  modified:
    - src/modelforecast/output/markdown_report.py
    - pyproject.toml
decisions:
  - "ruff added as dev dependency — was in pyproject.toml tool config but missing from dependency-groups"
  - "sweep manifest in sweep_20260318 has models_tested/total_results but no trials_per_level — infer from first result summary.trials"
  - "All models in same grade receive tie marker when any pair overlaps at T0 — marks the entire grade cohort as statistically indistinct"
metrics:
  duration: 174s
  completed: "2026-03-19"
  tasks_completed: 2
  files_modified: 4
---

# Phase 05 Plan 01: Consumer Output Pipeline Summary

Sweep-to-RESULTS.md pipeline: Wilson CI table with OpenRouter links, tie detection, and sweep metadata from 48 JSON files across 16 models.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Enhance markdown_report with links, CI overlap, metadata | d6e0995 | src/modelforecast/output/markdown_report.py, pyproject.toml |
| 2 | Create generate_readme_results.py and regenerate RESULTS.md | f03c382 | scripts/generate_readme_results.py, results/RESULTS.md |

## What Was Built

### `src/modelforecast/output/markdown_report.py` (enhanced)

Three new additions to the existing module:

- `find_latest_sweep_dir(base_results_dir)` — globs `sweep_*` dirs, sorts lexicographically (ISO date in name), returns last or raises `FileNotFoundError`
- `cis_overlap(ci_a, ci_b)` — overlap predicate: `ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]`
- `write_markdown_report` updated signature with `sweep_date`, `model_count`, `trials_per_level` params; adds metadata line, OpenRouter links, statistical tie markers with dagger symbol

### `scripts/generate_readme_results.py`

CLI entry point with `--sweep`, `--out`, `--base` flags. Auto-detects latest sweep, reads manifest for metadata, counts distinct models from `probes.model` fields, calls `write_markdown_report`. Prints summary via rich console.

### `results/RESULTS.md` (regenerated)

16 models from sweep_20260318. All models marked as ties at T0 (n=10 trials produces wide CIs that overlap across all grade groups). Four A-grade models: nvidia/nemotron-3-nano-30b-a3b, nvidia/nemotron-3-super-120b-a12b, stepfun/step-3.5-flash, z-ai/glm-4.5-air.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing dependency] ruff not installed in venv**
- **Found during:** Task 1 verification
- **Issue:** `pyproject.toml` had `[tool.ruff]` config but ruff was not in `dependency-groups.dev`; `uv run ruff check` failed with "No such file or directory"
- **Fix:** `uv add --dev ruff` — added ruff==0.14.7 to dev dependencies
- **Files modified:** pyproject.toml
- **Commit:** d6e0995

**2. [Rule 1 - Schema mismatch] Sweep manifest missing trials_per_level**
- **Found during:** Task 2 implementation
- **Issue:** Plan interface spec showed manifest with `trials_per_level` field; actual manifest has `models_tested` and `total_results` but no `trials_per_level`
- **Fix:** `load_sweep_results()` falls back to `summary.trials` from first result when manifest lacks `trials_per_level`
- **Files modified:** scripts/generate_readme_results.py

## Self-Check: PASSED

- `/home/jw/dev/modelforecast/scripts/generate_readme_results.py` — FOUND
- `/home/jw/dev/modelforecast/results/RESULTS.md` — FOUND
- `/home/jw/dev/modelforecast/src/modelforecast/output/markdown_report.py` — FOUND (enhanced)
- Commit d6e0995 — FOUND
- Commit f03c382 — FOUND
