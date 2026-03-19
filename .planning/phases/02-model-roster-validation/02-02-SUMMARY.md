---
phase: 02-model-roster-validation
plan: "02"
subsystem: infra
tags: [openrouter, free-models, graveyard, maintenance, model-tracking]

# Dependency graph
requires:
  - phase: 02-model-roster-validation
    provides: get_available_models() function for live roster lookup
provides:
  - GRAVEYARD.md with documented format and 4 seeded defunct model entries
  - scripts/update_graveyard.py CLI for detecting and recording newly-absent models
affects: [sweep, phase-03, phase-04, model-roster-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Graveyard pattern: Markdown table tracking defunct models with last-known and removed dates"
    - "sys.path.insert for running scripts without package install"
    - "Idempotent append: check already_buried set before writing new row"

key-files:
  created:
    - GRAVEYARD.md
    - scripts/update_graveyard.py
  modified: []

key-decisions:
  - "Use 2026-03-01 as conservative Last Known Available for seed entries (exact date unknown)"
  - "append_grave writes after last row via rstrip + newline to avoid double-blank-line drift"
  - "already_buried parsed via regex from graveyard text — no external state file needed"

patterns-established:
  - "Graveyard pattern: Append-only Markdown table for historical model tracking"
  - "Script imports via sys.path.insert to work without package installation"

requirements-completed: [MODEL-03]

# Metrics
duration: 2min
completed: 2026-03-19
---

# Phase 02 Plan 02: Model Graveyard Summary

**GRAVEYARD.md + update_graveyard.py: append-only defunct model tracker seeded with 4 confirmed-absent free OpenRouter models**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-19T04:53:32Z
- **Completed:** 2026-03-19T04:55:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created GRAVEYARD.md with documented Markdown table format and 4 seeded entries for models confirmed absent from current live roster
- Created scripts/update_graveyard.py that checks a known model list against the live OpenRouter API and appends new graves idempotently
- Full test suite (32 tests) continues to pass after both changes

## Task Commits

Each task was committed atomically:

1. **Task 1: Create GRAVEYARD.md with format and seed entries** - `d7181e8` (feat)
2. **Task 2: Create scripts/update_graveyard.py** - `b91f361` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `GRAVEYARD.md` - Human-readable graveyard of defunct free models with Format section and seeded Graveyard table
- `scripts/update_graveyard.py` - CLI script: accepts `--known` or `--known-file`, fetches live roster, appends new graves without duplicates, prints summary

## Decisions Made

- Used 2026-03-01 as conservative "Last Known Available" for seed entries — exact removal date unknown, conservative estimate avoids false precision
- append_grave uses rstrip + newline approach to avoid blank-line drift at end of file
- Already-buried check uses regex extraction from file text — avoids need for separate state file or database

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- GRAVEYARD.md and update_graveyard.py are ready for use before any sweep
- update_graveyard.py can be run with the current live model list to capture any models that disappear between planning and execution
- Requirement MODEL-03 (defunct models documented) is satisfied

---
*Phase: 02-model-roster-validation*
*Completed: 2026-03-19*

## Self-Check: PASSED

- FOUND: GRAVEYARD.md
- FOUND: scripts/update_graveyard.py
- FOUND: .planning/phases/02-model-roster-validation/02-02-SUMMARY.md
- FOUND: commit d7181e8 (Task 1)
- FOUND: commit b91f361 (Task 2)
