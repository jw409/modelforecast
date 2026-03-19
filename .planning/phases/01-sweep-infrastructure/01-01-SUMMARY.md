---
phase: 01-sweep-infrastructure
plan: "01"
subsystem: sweep-infrastructure
tags: [rate-limiting, retry, failure-classification, provenance]
dependency_graph:
  requires: []
  provides: [RateLimiter, ProbeResult.failure_mode, ProbeRunner retry loop, ProvenanceTracker.openrouter_provider]
  affects: [src/modelforecast/runner.py, src/modelforecast/probes/base.py, src/modelforecast/verification/provenance.py]
tech_stack:
  added: [src/modelforecast/sweep/]
  patterns: [per-model token bucket with jitter, exponential backoff retry, failure mode classification]
key_files:
  created:
    - src/modelforecast/sweep/__init__.py
    - src/modelforecast/sweep/rate_limiter.py
  modified:
    - src/modelforecast/probes/base.py
    - src/modelforecast/runner.py
    - src/modelforecast/verification/provenance.py
decisions:
  - "Store openrouter_provider from raw_response only (not raw HTTP headers) — openai SDK 1.x does not expose headers; httpx workaround deferred"
  - "Log warning per trial when x-openrouter-provider is absent rather than aborting"
  - "Retry budget: max 3 retries with 2^attempt exponential backoff (1s, 2s, 4s)"
metrics:
  duration_seconds: 139
  completed_date: "2026-03-19T04:40:46Z"
  tasks_completed: 2
  files_changed: 5
---

# Phase 01 Plan 01: Sweep Infrastructure Hardening Summary

Per-model token bucket rate limiter, ProbeResult failure classification, and ProbeRunner retry loop with x-openrouter-provider header capture path.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create RateLimiter and add failure_mode to ProbeResult | ff566b7 | sweep/__init__.py, sweep/rate_limiter.py, probes/base.py |
| 2 | Add retry, provider header capture, failure classification to ProbeRunner | 8eb8e3d | runner.py, verification/provenance.py |

## What Was Built

**RateLimiter** (`src/modelforecast/sweep/rate_limiter.py`): Per-model token bucket enforcing minimum call interval. Default 8 calls/minute with up to 0.5s jitter per wait. First call per model is immediate (no false wait).

**ProbeResult.failure_mode** (`src/modelforecast/probes/base.py`): Optional field added after `error`. Five named values: `text-instead-of-tool`, `malformed-json`, `wrong-tool`, `hallucinated-tool`, `missing-required-param`.

**ProbeRunner retry loop** (`src/modelforecast/runner.py`): Wraps `probe.run()` for `openai.RateLimitError` and `openai.APIStatusError` (5xx). Up to 3 retries with 2^attempt backoff. Final failure creates a synthetic `ProbeResult` with `success=False`. Rate limiter is called before each trial.

**ProbeRunner._classify_failure()**: Static method classifying failed trials. Called after each successful run, sets `result.failure_mode` in place.

**ProvenanceTracker.create_trial_record** (`src/modelforecast/verification/provenance.py`): Added `openrouter_provider` and `failure_mode` parameters. Both stored in the returned record dict. Provider value comes from `result.raw_response.get("x-openrouter-provider")` — None if not captured.

## Deviations from Plan

None — plan executed exactly as written.

**Note on provider header capture**: The plan acknowledged the openai SDK 1.x limitation. Per plan instruction, the storage path is wired (provider field in record dict) but actual header extraction depends on probe implementation injecting it into `raw_response`. This is the intended state for this plan.

## Verification Results

All integration checks passed:
- `RateLimiter(calls_per_minute=600).acquire('m')` completes in < 0.2s
- `ProbeResult.failure_mode` field present and assignable
- `ProvenanceTracker.create_trial_record(openrouter_provider='Google', failure_mode='text-instead-of-tool')` returns matching dict
- `ProbeRunner._classify_failure(no-tool-result, tools)` returns `'text-instead-of-tool'`
- 29 tests passed, 9 skipped — no regressions

## Self-Check: PASSED

- sweep/__init__.py: FOUND
- sweep/rate_limiter.py: FOUND
- probes/base.py (failure_mode): FOUND
- runner.py (rate_limiter, retry, _classify_failure): FOUND
- verification/provenance.py (openrouter_provider, failure_mode): FOUND
- Commit ff566b7: FOUND
- Commit 8eb8e3d: FOUND
