# Project Research Summary

**Project:** ModelForecast — LLM evaluation benchmark for free OpenRouter models
**Domain:** Consumer-facing LLM evaluation platform / tool-calling capability measurement
**Researched:** 2026-03-18
**Confidence:** HIGH

## Executive Summary

ModelForecast occupies a uniquely uncontested niche: empirical, probe-based tool-calling evaluation of free OpenRouter models, presented for developer consumption rather than academic audiences. No competitor (BFCL, Artificial Analysis, Chatbot Arena, OpenRouter Rankings) answers the specific question "which free OpenRouter model should I use for agentic tool calling?" — and ModelForecast's existing methodology (Wilson CI, per-trial provenance, deterministic pass/fail grading) is technically sounder than most leaderboards for this domain. The gap is not methodology; it is execution (stale data) and presentation (results are buried without consumer framing).

The recommended approach for this milestone is strictly additive: extend the existing probe runner with retry robustness and checkpoint-resume, run a fresh sweep across the current 26 free models, then layer on consumer-facing output (tiered README, availability flags, "best for X" guides). The core architecture already handles the hard parts correctly — individual JSON results, Wilson CI computation, T0 skip-ahead gate, provenance per trial. Three new components are needed: a `SweepOrchestrator` (thin coordinator with checkpoint support), a `RateLimiter` (per-model token bucket), and an `OutputPipeline` (reads stored results, generates README and badges). Nothing should be rewritten from scratch.

The most consequential risks are operational, not architectural. Provider variance on OpenRouter free tier can shift a model's score by 40-60 percentage points depending on which backend handles the request — this must be captured as a data dimension (record `x-openrouter-provider` header) before any results are published. Free model churn is severe: models already in the methodology doc may be gone. Validate the roster against the live `/api/v1/models` endpoint before spending any quota. CI overlap misread as ranking is a presentation risk: with n=10, two models at 80% and 70% are not statistically distinguishable and must not be displayed as if they are.

---

## Key Findings

### Recommended Stack

The existing Python 3.11+ / uv / openai SDK stack is correct and should not change structurally. The primary upgrade is the OpenAI SDK from `>=1.0.0` to `>=2.0.0` — v2 has built-in retry with exponential backoff for 429 and 5xx responses, eliminating the need for tenacity or any custom retry wrapper. For the free-tier rate limit (20 rpm), three retries with SDK-managed backoff is sufficient.

Statistics should remain on the existing hand-rolled Wilson CI (`stats/confidence.py`) for all probe calls — it has no dependencies and is correct. Add `statsmodels>=0.14.6` and `scipy>=1.15.0` as optional `[stats]` extras for anyone who wants Cohen's h effect sizes or bootstrap CIs for aggregate scores. Importantly, do not add any pairwise significance testing without Holm-Bonferroni correction — at 80 cells across 16 models, uncorrected pairwise tests will produce ~158 false positives at α=0.05.

Clean up `pyproject.toml` aggressively: remove `matplotlib`, `dagster`, `dagster-pipes`, `pandas`, `playwright` — none are used in the probe path and add unnecessary install weight.

**Core technologies:**
- Python 3.11+ / uv: already locked in, keep as-is
- openai SDK `>=2.0.0`: upgrade from 1.x for built-in retry (replaces need for tenacity)
- httpx `>=0.27.0`: keep for model discovery calls, bump minimum
- statsmodels `>=0.14.6` (optional): proportion tests, Cohen's h for pairwise comparison
- scipy `>=1.15.0` (optional): bootstrap CIs for aggregate scores
- shields.io static badges: no library needed, URL pattern generated in markdown writer

### Expected Features

The methodology and probe framework are sound. What is missing is fresh data and consumer framing. The highest-value work this milestone is not new features — it is running the sweep and presenting results in a way that immediately answers the question.

**Must have (table stakes):**
- Quick answer at top of README — one recommended model with grade badge, two sentences
- Timestamped sweep metadata — show n= and run date visibly, not in footnotes
- Model availability flags — mark deprecated models with date removed; KAT Coder Pro is already a known casualty
- Grade per model displayed prominently — A-F collapses the dimension matrix into a scannable signal
- Confidence intervals shown visibly — not buried in docs; they are the trust signal
- Reproducibility link — prominent link to METHODOLOGY.md from results table

**Should have (competitive differentiators):**
- "Best free model for X" one-liners — best overall, best for agentic pipelines (A1 focus), best T1/T2 for structured output
- "Avoid these" section — active negative recommendation for T0 failures, not passive "broken models" table
- OpenRouter direct link per model row — reduces click friction for adoption
- Sweep metadata badge — auto-generated shields.io badge showing n=10, date, model count
- Pairwise tie notation — mark overlapping CIs as "~" (statistical tie) rather than implying rank order

**Defer (v2+):**
- Failure taxonomy logging — requires new probe instrumentation before the sweep; high value but blocks fresh data
- Rate limit / reliability notes per model — requires separate timing instrumentation
- HTML dimension-filtering page — GitHub Pages page exists; add after results are fresh
- CoreWars adaptation rate as structured metric — medium complexity, second milestone
- Contributor trust tiers — premature at current scale (one maintainer)

### Architecture Approach

The architecture should add three thin components around the existing `ProbeRunner` without rewriting it. `SweepOrchestrator` coordinates the full sweep, reads/writes a `checkpoint.json` to support resume-from-partial, and routes results into sweep-ID-stamped directories. `RateLimiter` is injected into `ProbeRunner` as a per-model token bucket with jitter — a pure Python class with no external dependencies. `OutputPipeline` reads stored JSON results and generates the consumer output (README, RESULTS.md, badges) as a separate, re-runnable command that never touches the API.

Result storage stays as JSON files organized by sweep ID (`results/sweep_20260318/`), not SQLite or DuckDB. The project has no query workload — only aggregation for output. JSON files are git-diffable, human-readable, and already the format in use.

**Major components:**
1. `SweepOrchestrator` (new) — sweep config, checkpoint read/write, model iteration, manifest writing
2. `ProbeRunner` (extend) — add RateLimiter injection, retry wrapper in `run_level()`, sweep_id propagation
3. `RateLimiter` (new) — per-model token bucket with jitter, pure Python, no external deps
4. `ResultStore` (new) — reads and aggregates all JSONs from a sweep directory
5. `OutputPipeline` (new) — generates README, RESULTS.md, and badges from ResultStore; never calls API
6. `CoreWarsBridge` (new, optional) — wraps `games/corewars/` for game benchmarks; skippable with `--skip-games`

### Critical Pitfalls

1. **Provider variance invisibility** — The same free model routed to different OpenRouter backends can score 36.7% vs 93.3% on identical probes. Record `x-openrouter-provider` response header on every request before publishing any results. This is retroactively unfixable — it must be captured during the sweep.

2. **tool_choice vs tools distinction** — Models claiming tool support may 404 on `tool_choice: required`. Use `tool_choice: auto` as the default. The probe runner must log the distinction between "API rejected the request" (routing failure) and "model responded but did not call a tool" (capability failure).

3. **Free model churn** — Models disappear without notice. Validate every model ID against `/api/v1/models` before spending quota. Add a `GRAVEYARD.md` section for graded-but-removed models. Every result row must show "last verified available" date.

4. **CI overlap misread as ranking** — At n=10, a 90% score has a CI of roughly [57%, 98%]. Two models at 90% and 70% are not distinguishable. Do not sort by point estimate when CIs overlap. Consider grading on the lower CI bound, not the point estimate.

5. **Benchmark contamination** — The T0 probe text, once public, becomes training data for future model fine-tuning. Parameterize probe keywords before going public (e.g., "find files containing '{keyword}'" with varied keywords per run). This is retroactively hard to fix without breaking historical comparability.

---

## Implications for Roadmap

Based on combined research findings, the architecture's own build-order analysis, and pitfall phase mappings, a 5-phase structure is recommended:

### Phase 1: Sweep Infrastructure Hardening

**Rationale:** Everything downstream depends on a reliable sweep runner. Without retry, checkpoint-resume, and proper rate limiting, the fresh sweep will fail partway through 800 API calls and produce misleading partial results. Provider variance capture and probe runner error distinction (API rejection vs model failure) must also be baked in before any data is collected — both are retroactively unfixable.

**Delivers:** A sweep runner that can be safely started, stopped, and resumed; provider header capture; 429 handling; checkpoint persistence; sweep-ID-stamped output directories; probe fingerprinting for contamination tracking.

**Addresses (from FEATURES.md):** Timestamped sweep metadata, reproducibility statement foundation, availability validation.

**Avoids (from PITFALLS.md):** Provider variance invisibility (P1), tool_choice vs tools distinction (P2), daily quota exhaustion (P7), retry logic skewing results (P9).

### Phase 2: Model Roster Validation

**Rationale:** The current model list references models that no longer exist. Running probes against stale IDs wastes daily quota and produces results that consumers cannot reproduce. Roster validation is a prerequisite for the sweep, not a concurrent task.

**Delivers:** Validated current list of free models with tool support from live `/api/v1/models` endpoint; identification of deprecated models; updated METHODOLOGY.md with current roster; `GRAVEYARD.md` for removed models.

**Addresses (from FEATURES.md):** Model availability flags, freshness of results.

**Avoids (from PITFALLS.md):** Free model churn making results stale immediately (P3), OpenRouter model ID instability (P10).

### Phase 3: Fresh Sweep Execution

**Rationale:** Without current data, no presentation work matters. This is the milestone's core deliverable. The sweep should run sequentially (not async — free tier rate limits make concurrent calls counterproductive) against all validated models, writing individual result JSONs with sweep_id, provider header, and retry metadata.

**Delivers:** Complete sweep results in `results/sweep_20260318/` — individual JSONs per (model, level), sweep manifest, provider distribution data.

**Addresses (from FEATURES.md):** Fresh data is the prerequisite for every table stakes and differentiator feature.

**Avoids (from PITFALLS.md):** Partial results from interrupted sweeps (P7), provider variance invisibility (P1 — capture is live during sweep).

### Phase 4: Consumer Output and Results Display

**Rationale:** Fresh data without consumer framing leaves ModelForecast in its current state — technically sound but invisible to its target audience. The `OutputPipeline` reads stored results (no API calls) and generates the tiered README structure. This phase also addresses CI display and grading logic — both of which require the fresh data to be in hand before they can be validated.

**Delivers:** Tiered README (quick answer → category winners → full matrix), RESULTS.md with visible CIs and tie notation for overlapping intervals, shields.io badges, "best for X" one-liners, "avoid these" active anti-recommendations, OpenRouter direct links per model row.

**Addresses (from FEATURES.md):** Quick answer at top, grade per model, dimension breakdown, CI display, sweep metadata badge, differentiator category guides, anti-feature "avoid" section.

**Avoids (from PITFALLS.md):** CI overlap misread as ranking (P4), grade rubric hiding CI width (P8), integer rounding display issues (P12).

### Phase 5: CoreWars Integration (optional)

**Rationale:** CoreWars game benchmarks provide the "adaptation rate" differentiator that no competitor offers. However, it depends on Phase 4 output to know which models are worth running games against (top-grade models only). It is explicitly optional and skippable — the sweep infrastructure should support `--skip-games`.

**Delivers:** CoreWars leaderboard section in the unified README; adaptation rate as a structured metric; `CoreWarsBridge` wrapping existing `games/corewars/`; game result JSON schema compatible with the storage layer.

**Addresses (from FEATURES.md):** CoreWars adaptation story as structured metric (currently deferred to v2, but available if Phase 4 completes early).

**Avoids (from PITFALLS.md):** Merging CoreWars into ProbeRunner (game results are win/loss/draw, not binomial — separate schemas required).

### Phase Ordering Rationale

- Phase 1 before Phase 3: A broken sweep runner produces invalid data. Infrastructure correctness is a hard dependency.
- Phase 2 before Phase 3: Stale model IDs waste the daily quota cap (50-1000 req/day). Roster validation is cheap; re-running the sweep is expensive.
- Phase 3 before Phase 4: OutputPipeline reads stored results. It cannot generate the README without them.
- Phase 5 optional: CoreWars is a differentiator, not table stakes. Deferring it doesn't block any consumer-facing deliverable.

### Research Flags

Phases needing deeper research during planning:
- **Phase 1:** Rate limiting behavior — the OpenRouter free tier limits are documented as 20 rpm but may vary per model or account tier. Validate against the live `/api/v1/auth/key` endpoint during Phase 1 build.
- **Phase 1:** Provider header capture — the `x-openrouter-provider` header behavior should be verified against a live test call before building the capture path into the sweep runner.

Phases with standard patterns (skip research-phase):
- **Phase 2:** Roster validation is a single API call pattern, well-documented.
- **Phase 4:** OutputPipeline is pure data transformation; shields.io badge format is stable and well-documented.
- **Phase 5:** CoreWars infrastructure already exists in `games/corewars/`; the bridge is a thin wrapper.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Core stack verified against existing codebase; SDK version from PyPI; rate limit behavior MEDIUM (subject to change) |
| Features | HIGH | Table stakes verified by direct fetch of BFCL, Artificial Analysis, OpenRouter leaderboard pages; differentiators MEDIUM (gap analysis inference) |
| Architecture | HIGH | Existing codebase directly inspected; proposed new components follow established patterns with no novel dependencies |
| Pitfalls | HIGH | Provider variance pitfall from official OpenRouter announcement + Simon Willison write-up; tool_choice issue from live GitHub issues; CI statistics from published literature |

**Overall confidence:** HIGH

### Gaps to Address

- **Actual current daily quota for the account:** The sweep calculation (800 calls) nearly exhausts the 1,000/day verified-account limit. Verify account status via `/api/v1/auth/key` before planning the sweep timeline. If unverified (50/day limit), the sweep must be split across multiple days.
- **Provider header format:** `x-openrouter-provider` header existence and format should be verified with a live test call during Phase 1. The openai SDK may not expose raw response headers, requiring a workaround (httpx interceptor or switch specific calls to raw httpx).
- **CoreWars game interface:** The existing `games/corewars/` infrastructure interface was not directly inspected during this research. Phase 5 planning should include a read of `arena_session.py` and `model_benchmark.py` to confirm the bridge design is valid.
- **Probe contamination baseline:** No baseline exists for measuring contamination. The parameterization recommendation (variable keywords per run) should be decided before going public — the architecture change is small but the methodology documentation must reflect it.

---

## Sources

### Primary (HIGH confidence)
- Existing codebase at `/home/jw/dev/modelforecast/src/modelforecast/` — direct inspection
- statsmodels PyPI `>=0.14.6` — current stable
- openai SDK PyPI v2.29 — built-in retry behavior
- scipy 1.15.0 release notes — bootstrap BCa method
- OpenRouter Provider Variance: Introducing Exacto announcement — provider variance pitfall
- BFCL V4 Leaderboard — directly fetched for feature comparison
- Artificial Analysis LLM Leaderboard — directly fetched for feature comparison
- OpenRouter Rankings and Free Models Collection — directly fetched

### Secondary (MEDIUM confidence)
- OpenRouter free tier rate limits (20 rpm, 50/1000 req/day) — documented but subject to change
- Goose issue #3054 — tool_choice vs tools distinction in OpenRouter routing
- Simon Willison — Open weight LLMs exhibit inconsistent performance (provider variance data point)
- Goodeye Labs 2025 LLM evaluation review — anti-feature and contamination rationale
- CostGoat free model availability list — current March 2026 roster reference

### Tertiary (LOW confidence)
- Rate limit behavior per individual model — inferred from documentation; verify with live calls
- CoreWars interface design — not directly inspected; planned in Phase 5

---
*Research completed: 2026-03-18*
*Ready for roadmap: yes*
