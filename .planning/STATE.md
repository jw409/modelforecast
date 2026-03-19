---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
stopped_at: Completed 05-02-PLAN.md
last_updated: "2026-03-19T08:36:19.098Z"
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 8
  completed_plans: 8
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-18)

**Core value:** Empirical, reproducible answers to "which free OpenRouter model actually works for tool calling?" — data with confidence intervals, not vibes
**Current focus:** Phase 02 — Model Roster Validation

## Current Position

Phase: 02 (Model Roster Validation) — EXECUTING
Plan: 1 of N (Plan 01 complete)

## Performance Metrics

**Velocity:**

- Total plans completed: 1
- Average duration: 139s
- Total execution time: 139s

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| Phase 01 | 1 | 139s | 139s |

**Recent Trend:**

- Last 5 plans: 139s (01-01)
- Trend: baseline

*Updated after each plan completion*
| Phase 01 P02 | 180 | 2 tasks | 3 files |
| Phase 02 P01 | 74 | 2 tasks | 3 files |
| Phase 02-model-roster-validation P02 | 66 | 2 tasks | 2 files |
| Phase 03-codebase-and-documentation-cleanup P02 | 116 | 2 tasks | 2 files |
| Phase 03-codebase-and-documentation-cleanup P01 | 154 | 2 tasks | 3 files |
| Phase 05-consumer-output P01 | 174 | 2 tasks | 4 files |
| Phase 05-consumer-output P02 | 146 | 2 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Init]: Tool probes primary, CoreWars secondary — probes answer the consumer question directly
- [Init]: README table over web app — simplest distribution, least maintenance
- [Init]: Manual sweep cadence — avoids CI secret management
- [Init]: Free models only — zero cost, biggest audience
- [Phase 01]: openai SDK 1.x does not expose raw response headers; x-openrouter-provider stored from raw_response only, with per-trial warning when absent
- [Phase 01]: Retry budget: 3 attempts with exponential backoff (2^attempt seconds) for RateLimitError and 5xx APIStatusError
- [Phase 01]: Use TYPE_CHECKING guard + string annotation to break runner<->sweep circular import
- [Phase 01]: sweep subcommand added via subparsers while preserving all top-level flags
- [Phase 02]: Auto-discovery always uses tools_only=True — prevents wasting quota on models that cannot call tools
- [Phase 02]: --validate-roster exits before orchestrator/runner construction — zero quota cost for roster check
- [Phase 02-model-roster-validation]: 2026-03-01 used as conservative Last Known Available for seed entries; already_buried detection uses regex from file text avoiding separate state file
- [Phase 03-codebase-and-documentation-cleanup]: google/gemma-3-27b-it:free excluded from METHODOLOGY.md — confirmed in GRAVEYARD.md (removed 2026-03-18)
- [Phase 03-codebase-and-documentation-cleanup]: CLAUDE.md probe-sweep architecture documented — no Dagster/CoreWars/Playwright; sweep subcommand is canonical entry point
- [Phase 03-01]: openai>=2.0.0 bumped from 1.x; SDK max_retries replaces 40-line manual backoff loop; dead deps (dagster, matplotlib, pandas, playwright) stripped
- [Phase 05-consumer-output]: ruff added as dev dependency — was in pyproject.toml tool config but missing from dependency-groups
- [Phase 05-consumer-output]: All models in same grade receive tie marker when any CI pair overlaps at T0 — marks the entire grade cohort as statistically indistinct
- [Phase 05-consumer-output]: HTML comment markers used for idempotent README section injection — invisible when rendered

### Pending Todos

None yet.

### Blockers/Concerns

- **Account quota risk**: Sweep calculation (~800 calls) nearly exhausts the 1,000/day verified-account limit. Verify account status via `/api/v1/auth/key` before planning Phase 4 timeline. If unverified (50/day), sweep must be split across multiple days.
- **Provider header capture**: `x-openrouter-provider` header format should be verified with a live test call during Phase 1 — openai SDK may not expose raw response headers, requiring an httpx workaround.

## Session Continuity

Last session: 2026-03-19T08:36:19.096Z
Stopped at: Completed 05-02-PLAN.md
Resume file: None
