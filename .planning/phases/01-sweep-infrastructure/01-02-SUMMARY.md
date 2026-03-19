---
phase: 01-sweep-infrastructure
plan: "02"
subsystem: sweep
tags: [sweep, checkpoint, resume, cli, orchestrator]
dependency_graph:
  requires: [01-01]
  provides: [SweepOrchestrator, sweep-cli]
  affects: [runner, __main__]
tech_stack:
  added: []
  patterns: [checkpoint-resume, timestamped-output-dirs, circular-import-TYPE_CHECKING]
key_files:
  created:
    - src/modelforecast/sweep/orchestrator.py
  modified:
    - src/modelforecast/sweep/__init__.py
    - src/modelforecast/__main__.py
decisions:
  - "Use TYPE_CHECKING guard + string annotation to break runner<->sweep circular import"
  - "SweepOrchestrator.run() overrides runner.output_dir to sweep_dir at call time"
  - "sweep subcommand added via subparsers while preserving all top-level flags unchanged"
metrics:
  duration: ~180s
  completed: "2026-03-19"
  tasks_completed: 2
  files_changed: 3
---

# Phase 01 Plan 02: SweepOrchestrator and CLI Sweep Subcommand Summary

**One-liner:** Interrupt-safe sweep orchestration with date-stamped output dirs, checkpoint-resume, and a `sweep` CLI subcommand.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create SweepOrchestrator | 6d17a7f | sweep/orchestrator.py, sweep/__init__.py |
| 2 | Wire sweep subcommand into CLI | db82649 | __main__.py, sweep/orchestrator.py |

## What Was Built

### SweepOrchestrator (`src/modelforecast/sweep/orchestrator.py`)

- `_generate_sweep_id()`: produces `sweep_YYYYMMDD`, appends `_2`, `_3`... if directory already exists
- `read_checkpoint()`: reads `results/{sweep_id}/checkpoint.json`, returns `[]` if absent
- `write_checkpoint(completed_models)`: overwrites checkpoint after each model completes
- `write_manifest(...)`: writes `results/{sweep_id}/sweep_manifest.json` with full schema on sweep completion
- `run(runner, trials, max_level, resume)`: orchestrates sweep — overrides `runner.output_dir` to sweep-stamped dir, skips completed models when `resume=True`, writes checkpoint after each model, writes manifest at end

### CLI (`src/modelforecast/__main__.py`)

- `sweep` subcommand added via `add_subparsers`
- Flags: `--resume`, `--trials`, `--max-level`, `--output`, `--sweep-id`, `--contributor`, `--skip-validation`
- All existing top-level flags (`--model`, `--level`, `--probe`, `--list-models`, `--validate`) unchanged

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Circular import between runner.py and sweep/__init__.py**
- **Found during:** Task 2 verification
- **Issue:** `runner.py` imports `from modelforecast.sweep.rate_limiter import RateLimiter` (directly, not via `__init__`). When `sweep/__init__.py` was updated to re-export `SweepOrchestrator`, importing `__init__` triggered `orchestrator.py`, which imported `runner.py` — creating a circular initialization.
- **Fix:** Replaced `from modelforecast.runner import ProbeRunner` with `from typing import TYPE_CHECKING` guard + string annotation `"ProbeRunner"` in method signature. Runtime behavior unchanged; type checkers still see the correct type.
- **Files modified:** `src/modelforecast/sweep/orchestrator.py`
- **Commit:** db82649

## Verification Results

```
uv run python -m modelforecast sweep --help   → exit 0, shows all flags
uv run python -m modelforecast --help         → shows 'sweep' as subcommand
uv run pytest tests/ -x -q                   → 29 passed, 9 skipped
SweepOrchestrator checkpoint round-trip       → PASSED
Manifest schema validation                    → PASSED
Resume skip logic                             → PASSED
```

## Self-Check: PASSED

Files exist:
- src/modelforecast/sweep/orchestrator.py — FOUND
- src/modelforecast/sweep/__init__.py — FOUND
- src/modelforecast/__main__.py — FOUND

Commits exist:
- 6d17a7f — FOUND (feat(01-02): add SweepOrchestrator...)
- db82649 — FOUND (feat(01-02): wire sweep subcommand...)
