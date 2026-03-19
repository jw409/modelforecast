---
phase: 03-codebase-and-documentation-cleanup
plan: "01"
subsystem: dependencies-and-runner
tags: [cleanup, dependencies, retry, openai-sdk]
dependency_graph:
  requires: []
  provides: [clean-dependency-tree, sdk-managed-retry]
  affects: [pyproject.toml, uv.lock, runner.py]
tech_stack:
  added: []
  patterns: [openai-sdk-max-retries]
key_files:
  created:
    - tests/test_runner_retry.py
  modified:
    - pyproject.toml
    - src/modelforecast/runner.py
decisions:
  - "openai>=2.0.0 bumped from 1.x — 2.x SDK has stable max_retries on constructor"
  - "httpx bumped to >=0.27.0 per research recommendation"
  - "self.max_retries removed from ProbeRunner — no longer needed as SDK owns retry state"
  - "import time removed from runner.py — only used in deleted retry loop"
metrics:
  duration_seconds: 154
  completed_date: "2026-03-19"
  tasks_completed: 2
  files_modified: 3
---

# Phase 03 Plan 01: Dependency Cleanup and SDK Retry Migration Summary

Remove dead dependencies (dagster, matplotlib, pandas, playwright) and migrate ProbeRunner from a manual exponential-backoff retry loop to openai SDK built-in `max_retries` on the OpenAI constructor.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Strip dead deps, bump openai to >=2.0.0 | 8fc92d2 | pyproject.toml, uv.lock |
| 2 | Migrate runner retry to SDK max_retries | 4f450eb + 60efb72 | runner.py, tests/test_runner_retry.py |

## What Was Built

**Task 1 — Dependency cleanup:**

- Removed `dagster>=1.12.4`, `dagster-pipes>=1.12.4`, `matplotlib>=3.10.7`, `pandas>=2.3.3`, `playwright>=1.56.0` from `pyproject.toml`
- Bumped `openai>=1.0.0` → `openai>=2.0.0` (installed: 2.8.1)
- Bumped `httpx>=0.25.0` → `httpx>=0.27.0`
- `uv sync` removed 51 packages from the environment

**Task 2 — SDK retry migration (TDD):**

- Wrote 3 tests (RED): constructor kwarg check, source-scan for manual loop, RateLimitError handling
- Added `max_retries=max_retries` to `OpenAI(...)` constructor in `runner.py`
- Removed 40-line manual `for attempt in range(self.max_retries + 1)` loop
- Removed `import time` and `self.max_retries` (no longer used)
- All 35 tests pass (3 new + 32 existing)

## Verification

```
PASS: no dead deps
"openai>=2.0.0"
PASS: manual loop gone
37:        max_retries: int = 3,
60:            max_retries=max_retries,
35 passed, 9 skipped
```

## Deviations from Plan

None - plan executed exactly as written. The ruff issues found in runner.py (unused `validate_model` import, trailing whitespace) are pre-existing and out of scope for this task.

## Self-Check: PASSED

- [x] pyproject.toml modified — 3 core deps only
- [x] src/modelforecast/runner.py modified — max_retries on constructor, no manual loop
- [x] tests/test_runner_retry.py created — 3 tests, all pass
- [x] commit 8fc92d2 exists (dep cleanup)
- [x] commit 60efb72 exists (RED tests)
- [x] commit 4f450eb exists (GREEN implementation)
