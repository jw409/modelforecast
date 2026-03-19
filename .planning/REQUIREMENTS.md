# Requirements: ModelForecast Revival

**Defined:** 2026-03-18
**Core Value:** Empirical, reproducible answers to "which free OpenRouter model actually works for tool calling?"

## v1 Requirements

### Sweep Infrastructure

- [x] **SWEEP-01**: Runner handles OpenRouter 429s with backoff and SDK built-in retry
- [x] **SWEEP-02**: Sweep can be interrupted and resumed from last completed model/level
- [x] **SWEEP-03**: Each API response captures `x-openrouter-provider` header for backend tracking
- [x] **SWEEP-04**: Failed probes are classified by failure mode (text-instead-of-tool, malformed-JSON, wrong-tool, hallucinated-tool, missing-required-param)
- [x] **SWEEP-05**: Sweep results write to timestamped directory (`results/sweep_YYYYMMDD/`)

### Model Management

- [x] **MODEL-01**: Runner validates all model IDs against live OpenRouter API before starting sweep
- [x] **MODEL-02**: Runner auto-discovers current free models with tool support from OpenRouter API
- [x] **MODEL-03**: Removed/defunct models are tracked in a graveyard section

### Consumer Output

- [ ] **OUTPUT-01**: README shows quick answer at top ("Best free model right now: X")
- [ ] **OUTPUT-02**: README shows category winners ("Best for tool calling", "Best for restraint", etc.)
- [ ] **OUTPUT-03**: README shows full grade matrix with Wilson CI for all tested models
- [ ] **OUTPUT-04**: README shows "Avoid these" section for models that fail T0 (<20%)
- [ ] **OUTPUT-05**: Each model row links to its OpenRouter model page
- [ ] **OUTPUT-06**: Shields.io grade badges for top models
- [ ] **OUTPUT-07**: Sweep metadata badge showing date and model count
- [ ] **OUTPUT-08**: "Best for X" one-liner recommendations by use case (tool calling, coding, chat)

### Methodology

- [ ] **METH-01**: METHODOLOGY.md updated with current model list and dimension descriptions
- [ ] **METH-02**: Grading rubric documented inline in results table

### Maintenance

- [ ] **MAINT-01**: Dead dependencies removed from pyproject.toml (dagster, matplotlib, pandas, playwright)
- [ ] **MAINT-02**: openai SDK upgraded to 2.x with built-in retry
- [ ] **MAINT-03**: CLAUDE.md updated to reflect current project state
- [ ] **MAINT-04**: models listed in METHODOLOGY.md match current OpenRouter free roster

### Execution

- [ ] **EXEC-01**: Fresh sweep completed across all free models with tool support
- [ ] **EXEC-02**: Results committed and pushed to public repo

## v2 Requirements

### Advanced Statistics

- **STATS-01**: Grade on CI lower bound instead of point estimate
- **STATS-02**: Pairwise comparison tests (proportions_ztest) when CIs overlap
- **STATS-03**: Cohen's h effect sizes for model comparison

### Game Integration

- **GAME-01**: CoreWars results surface adaptation rate as a metric
- **GAME-02**: Game results cross-referenced with probe grades

### Ecosystem

- **ECO-01**: Cross-validation with game1 dispatcher observed data
- **ECO-02**: Auto-detect new free models since last sweep

## Out of Scope

| Feature | Reason |
|---------|--------|
| DOOM/Angband game benchmarks | Design exists but unimplemented; CoreWars sufficient |
| Automated weekly CI sweeps | Author runs manually when inspired |
| Full web app or interactive tool | README table is the output |
| Paid model testing | Free models only — zero cost, biggest audience |
| Probe prompt parameterization | Contamination risk negligible at current visibility |
| Multi-tier contributor trust system | Premature at current scale (one maintainer) |
| Database storage (SQLite/DuckDB) | JSON-per-file is correct for <1MB sweep data |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| SWEEP-01 | Phase 1 | Complete |
| SWEEP-02 | Phase 1 | Complete |
| SWEEP-03 | Phase 1 | Complete |
| SWEEP-04 | Phase 1 | Complete |
| SWEEP-05 | Phase 1 | Complete |
| MODEL-01 | Phase 2 | Complete |
| MODEL-02 | Phase 2 | Complete |
| MODEL-03 | Phase 2 | Complete |
| MAINT-01 | Phase 3 | Pending |
| MAINT-02 | Phase 3 | Pending |
| MAINT-03 | Phase 3 | Pending |
| MAINT-04 | Phase 3 | Pending |
| METH-01 | Phase 3 | Pending |
| METH-02 | Phase 3 | Pending |
| EXEC-01 | Phase 4 | Pending |
| EXEC-02 | Phase 4 | Pending |
| OUTPUT-01 | Phase 5 | Pending |
| OUTPUT-02 | Phase 5 | Pending |
| OUTPUT-03 | Phase 5 | Pending |
| OUTPUT-04 | Phase 5 | Pending |
| OUTPUT-05 | Phase 5 | Pending |
| OUTPUT-06 | Phase 5 | Pending |
| OUTPUT-07 | Phase 5 | Pending |
| OUTPUT-08 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-18 — traceability validated against ROADMAP.md*
