# Roadmap: ModelForecast Revival

## Overview

ModelForecast's methodology is already sound — Wilson CI, per-trial provenance, deterministic pass/fail grading. What's missing is fresh data and consumer framing. This milestone works in build-order: harden the sweep runner so it can reliably complete 800 API calls, validate the model roster against live OpenRouter data, clean up the codebase and documentation, run the actual sweep, then publish consumer-friendly output. Five phases, each delivering one complete, verifiable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Sweep Infrastructure** - Harden the runner with retry, checkpoint-resume, rate limiting, and provider header capture (completed 2026-03-19)
- [x] **Phase 2: Model Roster Validation** - Validate all model IDs against live OpenRouter API and maintain a graveyard for defunct models (completed 2026-03-19)
- [ ] **Phase 3: Codebase and Documentation Cleanup** - Remove dead dependencies, upgrade SDK, and update METHODOLOGY.md and CLAUDE.md
- [ ] **Phase 4: Fresh Sweep Execution** - Run complete probe sweep across all validated free models with tool support
- [ ] **Phase 5: Consumer Output** - Generate tiered README, grade badges, category winners, and "best for X" recommendations

## Phase Details

### Phase 1: Sweep Infrastructure
**Goal**: The sweep runner can be safely started, interrupted, and resumed; it captures provider variance data; it handles 429s without losing progress
**Depends on**: Nothing (first phase)
**Requirements**: SWEEP-01, SWEEP-02, SWEEP-03, SWEEP-04, SWEEP-05
**Success Criteria** (what must be TRUE):
  1. A sweep interrupted mid-run resumes from the last completed model/level without re-running earlier probes
  2. Every API response record contains the `x-openrouter-provider` header value
  3. The runner retries 429 responses with backoff and does not fail the sweep on transient errors
  4. Failed probes are stored with a classified failure mode (text-instead-of-tool, malformed-JSON, wrong-tool, hallucinated-tool, missing-required-param)
  5. Results are written to a timestamped directory (`results/sweep_YYYYMMDD/`) with a manifest file
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Runner hardening: RateLimiter, failure classification, provider header storage (SWEEP-01, SWEEP-03, SWEEP-04)
- [ ] 01-02-PLAN.md — Sweep orchestration: SweepOrchestrator with checkpoint-resume and timestamped output (SWEEP-02, SWEEP-05)

### Phase 2: Model Roster Validation
**Goal**: Every model ID used in the sweep is confirmed live on OpenRouter; defunct models are documented rather than silently skipped
**Depends on**: Phase 1
**Requirements**: MODEL-01, MODEL-02, MODEL-03
**Success Criteria** (what must be TRUE):
  1. Running the sweep against a stale or nonexistent model ID produces a clear error before any API quota is consumed
  2. The runner auto-discovers all current free models with tool support from the live OpenRouter `/api/v1/models` endpoint
  3. A graveyard section documents models that were tested but are no longer available, with the date they disappeared
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Fix auto-discovery to tools_only=True, add pre-sweep validation, add --validate-roster flag (MODEL-01, MODEL-02)
- [ ] 02-02-PLAN.md — Create GRAVEYARD.md and scripts/update_graveyard.py for defunct model tracking (MODEL-03)

### Phase 3: Codebase and Documentation Cleanup
**Goal**: The project installs cleanly, runs against the current OpenRouter API, and its documentation reflects reality
**Depends on**: Phase 1
**Requirements**: MAINT-01, MAINT-02, MAINT-03, MAINT-04, METH-01, METH-02
**Success Criteria** (what must be TRUE):
  1. `uv sync` completes without installing dagster, matplotlib, pandas, or playwright
  2. The runner uses openai SDK 2.x with built-in retry (no tenacity or custom retry wrapper)
  3. CLAUDE.md describes the current project state accurately (probe dimensions, run command, output structure)
  4. METHODOLOGY.md lists only models that currently exist on OpenRouter free tier, with grading rubric documented inline
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — Strip dead deps, bump openai to >=2.0.0, migrate runner to SDK built-in retry (MAINT-01, MAINT-02)
- [ ] 03-02-PLAN.md — Rewrite CLAUDE.md, update METHODOLOGY.md model roster and inline rubric (MAINT-03, MAINT-04, METH-01, METH-02)

### Phase 4: Fresh Sweep Execution
**Goal**: Complete, fresh probe results exist for all current free models with tool support
**Depends on**: Phase 2, Phase 3
**Requirements**: EXEC-01, EXEC-02
**Success Criteria** (what must be TRUE):
  1. `results/sweep_YYYYMMDD/` directory exists with individual JSON result files for every (model, level) combination across all validated models
  2. The sweep manifest records provider distribution data and total trial count
  3. Results are committed and pushed to the public repo with a clear commit message stating the sweep date and model count
**Plans**: TBD

### Phase 5: Consumer Output
**Goal**: Any developer who visits the repo immediately knows which free OpenRouter model to use for their task, backed by data they can verify
**Depends on**: Phase 4
**Requirements**: OUTPUT-01, OUTPUT-02, OUTPUT-03, OUTPUT-04, OUTPUT-05, OUTPUT-06, OUTPUT-07, OUTPUT-08
**Success Criteria** (what must be TRUE):
  1. The README opens with a one-line best-model recommendation backed by a grade badge — visible without scrolling
  2. Category winner one-liners ("Best for tool calling", "Best for restraint") appear in the README before the full matrix
  3. The full grade matrix shows Wilson CI alongside each grade; models with overlapping CIs are marked as statistical ties
  4. An "Avoid these" section actively names models that fail T0 (below 20%) with their failure mode
  5. Each model row links directly to its OpenRouter model page; shields.io badges and a sweep metadata badge (date, n=, model count) are present
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5
Note: Phases 2 and 3 can run in parallel after Phase 1 completes (no dependency between them).

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Sweep Infrastructure | 2/2 | Complete   | 2026-03-19 |
| 2. Model Roster Validation | 2/2 | Complete   | 2026-03-19 |
| 3. Codebase and Documentation Cleanup | 0/2 | Not started | - |
| 4. Fresh Sweep Execution | 0/TBD | Not started | - |
| 5. Consumer Output | 0/TBD | Not started | - |
