---
phase: 02-model-roster-validation
plan: "01"
subsystem: api
tags: [openrouter, model-discovery, tool-calling, validation, pytest]

# Dependency graph
requires:
  - phase: 01-sweep-infrastructure
    provides: ProbeRunner, SweepOrchestrator, get_free_models signature with tools_only param
provides:
  - ProbeRunner auto-discovery restricted to tool-capable free models only
  - --validate-roster flag on sweep subcommand for pre-sweep live roster check
  - 3 unit tests covering tools_only filter, explicit model validation, skip_validation bypass
affects:
  - 02-model-roster-validation plans 02+
  - Any plan that uses ProbeRunner auto-discovery or sweep subcommand

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD: write failing tests first, fix implementation to pass"
    - "Pre-sweep exit path: --validate-roster dispatches before SweepOrchestrator construction"

key-files:
  created:
    - tests/test_roster_validation.py
  modified:
    - src/modelforecast/runner.py
    - src/modelforecast/__main__.py

key-decisions:
  - "Auto-discovery always uses tools_only=True — prevents wasting quota on models that cannot call tools"
  - "--validate-roster exits before orchestrator/runner construction — zero quota cost for roster check"

patterns-established:
  - "Pattern: pre-flight exit flags placed before expensive object construction in sweep dispatch"
  - "Pattern: monkeypatch.setenv + patch openai.OpenAI for unit-testing ProbeRunner without network calls"

requirements-completed: [MODEL-01, MODEL-02]

# Metrics
duration: 2min
completed: 2026-03-18
---

# Phase 02 Plan 01: Model Roster Validation Summary

**ProbeRunner restricted to tool-capable free models via tools_only=True, plus --validate-roster flag for zero-cost live roster inspection**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-18T~04:53Z
- **Completed:** 2026-03-18T~04:55Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Fixed ProbeRunner auto-discovery: `get_free_models(api_key, tools_only=True)` — non-tool-capable models no longer included in sweeps
- Added `--validate-roster` flag to sweep subcommand that fetches live tool-capable free models and exits with a count (zero quota cost, exits before any probe runs)
- 3 unit tests with mocked `get_available_models`: tools_only filter, explicit model validation with warning for invalid, skip_validation bypass

## Task Commits

1. **Task 1: Fix auto-discovery to tools_only=True and add pre-sweep validation** - `b78f9ef` (feat + test — TDD)
2. **Task 2: Add --validate-roster flag to sweep subcommand** - `e24efac` (feat)

## Files Created/Modified
- `tests/test_roster_validation.py` — 3 new tests: test_auto_discovery_uses_tools_only, test_explicit_models_validated_before_run, test_skip_validation_bypasses_check
- `src/modelforecast/runner.py` — line 78: `get_free_models(api_key)` → `get_free_models(api_key, tools_only=True)`; console message updated
- `src/modelforecast/__main__.py` — `--validate-roster` arg added to sweep_parser; dispatch block added before SweepOrchestrator construction

## Decisions Made
- `--validate-roster` exits before `SweepOrchestrator(...)` construction: no sweep state created, zero API quota used beyond the models endpoint
- Used `monkeypatch.setenv + patch("openai.OpenAI")` pattern for unit tests — no API key required, no network calls

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness
- ProbeRunner now only sweeps tool-capable models, prerequisite for MODEL-01 and MODEL-02
- `sweep --validate-roster` provides operator tooling for verifying current roster before committing to a full sweep
- Full test suite: 32 passed, 9 skipped (no regressions)

---
*Phase: 02-model-roster-validation*
*Completed: 2026-03-18*
