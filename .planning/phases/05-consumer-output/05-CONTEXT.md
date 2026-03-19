# Phase 5: Consumer Output - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate a tiered README that answers "which free OpenRouter model should I use?" at a glance. Quick answer at top, category winners below, full grade matrix with Wilson CI, "avoid these" section, shields.io badges, OpenRouter direct links.

</domain>

<decisions>
## Implementation Decisions

### Result Display Structure
- README opens with one-line best-model recommendation + grade badge
- Category winners: "Best for tool calling", "Best for restraint", "Best for multi-step"
- Full grade matrix table with all models, all dimensions, Wilson CI
- "Avoid these" section listing models that fail T0 (<20%) with failure mode
- Sweep metadata badge: date, n=trials, model count

### Badge Design
- Shields.io static badges — no external service needed
- Grade colors: A=brightgreen, B=green, C=yellow, D=orange, F=red
- Badge URL format: `https://img.shields.io/badge/Grade-A-brightgreen`

### Links
- Each model row links to `https://openrouter.ai/models/{model_id}`
- Models with `:free` suffix link correctly

### Script Design
- `scripts/generate_readme_results.py` reads from `results/sweep_YYYYMMDD/`
- Outputs markdown sections that get inserted into README.md
- Also generates `results/RESULTS.md` with full data
- Idempotent — can re-run after each sweep

### Claude's Discretion
- Table formatting details
- Exact wording of recommendations
- How to handle statistical ties (overlapping CIs)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/modelforecast/output/markdown_report.py` — existing report generator, needs enhancement
- `src/modelforecast/stats/confidence.py` — wilson_interval
- `results/sweep_YYYYMMDD/` — sweep output with per-model JSON files

### Established Patterns
- Rich console for output
- JSON result files per (model, level) combination
- Grading rubric: A (T0>=80%, T1>=70%, none below 50%), B, C, D, F

### Integration Points
- README.md — results section needs to be insertable/replaceable
- results/RESULTS.md — standalone results file

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the tiered display structure agreed in questioning.

</specifics>

<deferred>
## Deferred Ideas

- Interactive web comparison tool
- JSON API for embedding results
- Automated weekly sweeps

</deferred>
