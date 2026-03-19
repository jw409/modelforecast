---
phase: 03-codebase-and-documentation-cleanup
plan: 02
subsystem: documentation
tags: [docs, cleanup, claude-md, methodology]
completed: "2026-03-19T05:05:13Z"
duration: 116s

dependency_graph:
  requires: []
  provides: [accurate-claude-md, current-methodology-roster]
  affects: [future-sessions, sweep-execution-docs]

tech_stack:
  added: []
  patterns: [probe-sweep-architecture, T/R/A-dimensions, Wilson-CI-grading]

key_files:
  created: []
  modified:
    - CLAUDE.md
    - docs/METHODOLOGY.md

decisions:
  - "google/gemma-3-27b-it:free excluded from METHODOLOGY.md roster — confirmed in GRAVEYARD.md"
  - "Inline grading rubric added to Output Format section without removing standalone Grading Rubric section"
  - "CLAUDE.md free models examples use only non-graveyard models (qwen3-32b, llama-4-maverick, grok-4.1-fast)"

metrics:
  duration: 116s
  tasks_completed: 2
  files_modified: 2
---

# Phase 03 Plan 02: Documentation Accuracy — CLAUDE.md + METHODOLOGY.md Summary

**One-liner**: Rewrote CLAUDE.md to remove Dagster/CoreWars/Playwright fiction and updated METHODOLOGY.md with table-format roster (excluding graveyard models) plus inline grading rubric.

## What Was Built

### Task 1: CLAUDE.md Rewrite
Wholesale replacement of the stale CLAUDE.md bootloader. The file previously described a fictional Dagster/CoreWars tournament system. The new version accurately documents:
- Probe sweep runner architecture (SweepOrchestrator + ProbeRunner)
- T/R/A probe dimensions
- Correct CLI commands (`sweep`, `sweep --validate-roster`, `sweep --resume`)
- Removed: Dagster, CoreWars, Playwright, chart generation, `games/` directory

### Task 2: METHODOLOGY.md Updates
Two targeted edits:
1. **Models Tested section**: Replaced 9-item bullet list with a `| Model | Last Verified |` table. Excluded `google/gemma-3-27b-it:free` (confirmed defunct in GRAVEYARD.md). 8 live models remain with `2026-03-19` verification dates. Added link to GRAVEYARD.md and update instructions.
2. **Inline grading rubric**: Added `**Grading rubric**` table immediately after the Output Format example table (after `*n=10 per cell.*` line). The standalone Grading Rubric section in the Statistical Framework area was preserved as specified.

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written, with one note:

**Graveyard cross-check result**: `google/gemma-3-27b-it:free` was confirmed in GRAVEYARD.md (removed 2026-03-18). This model appeared in both the plan's proposed 9-model table AND the METHODOLOGY.md file. It was correctly excluded from the final roster, leaving 8 models. The plan's proposed table included it as a candidate — the cross-check step worked as designed.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | aff5a4f | docs(03-02): rewrite CLAUDE.md to reflect probe-sweep architecture |
| 2 | 4dc3ff0 | docs(03-02): update METHODOLOGY.md model roster and add inline grading rubric |

## Verification

All success criteria met:
- `grep -i "dagster\|corewars\|playwright" CLAUDE.md` → CLEAN
- `grep "uv run python -m modelforecast sweep" CLAUDE.md` → 12 matches
- `grep "Last Verified" docs/METHODOLOGY.md` → match (table column header)
- `grep "Grading rubric" docs/METHODOLOGY.md` → match (inline rubric present)
- `grep "GRAVEYARD" docs/METHODOLOGY.md` → link to GRAVEYARD.md
- `uv run pytest tests/ -q` → 35 passed, 9 skipped, 0 failures

## Self-Check: PASSED
