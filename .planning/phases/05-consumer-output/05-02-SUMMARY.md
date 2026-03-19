---
phase: 05-consumer-output
plan: "02"
subsystem: output
tags: [readme, badges, shields.io, idempotent, consumer-output]
dependency_graph:
  requires: [05-01]
  provides: [readme_sections, readme_injection]
  affects: [README.md, scripts/generate_readme_results.py]
tech_stack:
  added: []
  patterns: [html-comment-markers, idempotent-injection, shields.io-badges]
key_files:
  created:
    - src/modelforecast/output/readme_sections.py
  modified:
    - scripts/generate_readme_results.py
    - README.md
decisions:
  - "HTML comment markers chosen for idempotent injection — invisible in rendered GitHub view, survive editor reformats"
  - "Top model selection prefers grade A then B by T0 rate — deterministic given same sweep data"
  - "Failure mode defaults to 'no tool_call produced' when no trial-level failure_mode field present in sweep JSON"
metrics:
  duration: 146s
  completed: "2026-03-19"
  tasks_completed: 2
  files_modified: 3
---

# Phase 05 Plan 02: README Consumer Sections Summary

**One-liner:** Idempotent HTML-comment-marker injection of quick answer, grade badges, category winners, and avoid list into README.md, driven by sweep data.

## What Was Built

`src/modelforecast/output/readme_sections.py` — new module with five public functions:

- `build_quick_answer_section(model_summary)` — top-graded model + shields.io badge + one-liner recommendation
- `build_category_winners_section(model_summary)` — per-category best model (tool calling, schema, restraint, multi-turn)
- `build_avoid_section(model_summary)` — all F-grade models with failure mode and T0 rate
- `build_grade_badges_section(model_summary)` — clickable shields.io badges for all A/B models
- `update_readme_sections(readme_path, model_summary, sweep_metadata)` — idempotent injection via `re.sub` on HTML comment marker pairs

`scripts/generate_readme_results.py` — updated to:
1. Build `model_summary` dict from loaded sweep results
2. Calculate grades via `calculate_grade()`
3. Call `update_readme_sections()` to inject all four sections

`README.md` — updated with four marker pairs injected and populated:
- QUICK-ANSWER section: `### Best free model right now: nemotron-3-nano-30b-a3b`
- GRADE-BADGES section: four brightgreen A-grade badges
- CATEGORY-WINNERS section: four category one-liners
- AVOID section: eight F-grade models named with failure modes

## Decisions Made

- HTML comment markers (`<!-- MODELFORECAST:SECTION:START/END -->`) for injection — invisible when rendered, survive editor reformats
- Top model selection: prefer grade A then B, tie-break by T0 rate — gives deterministic output
- Failure mode field: defaults to `"no tool_call produced"` when `trial.get("failure_mode")` returns nothing — current sweep JSON has `text-instead-of-tool` from probe harness

## Deviations from Plan

None — plan executed exactly as written.

## Verification Results

- `uv run ruff check src/modelforecast/output/readme_sections.py scripts/generate_readme_results.py` — all checks passed
- `uv run pytest tests/ -x -q` — 35 passed, 9 skipped
- `grep "Best free model right now" README.md` — match found
- `grep "Best for tool calling" README.md` — match found
- `grep "img.shields.io/badge/Grade" README.md | wc -l` — 2 (passes > 1 criterion)
- Idempotence confirmed: running script 3 times produces identical README.md output

## Self-Check: PASSED

- src/modelforecast/output/readme_sections.py: FOUND
- Commit 2dde555 (Task 1): FOUND
- Commit f489d11 (Task 2): FOUND
