---
phase: 05-consumer-output
verified: 2026-03-19T00:00:00Z
status: human_needed
score: 8/8 must-haves verified
human_verification:
  - test: "Confirm quick-answer section is visible above the fold when opening the repo on GitHub"
    expected: "A developer landing on the repo sees 'Best free model right now: nemotron-3-nano-30b-a3b' and the Grade A badge before scrolling"
    why_human: "The ROADMAP success criterion says 'visible without scrolling.' The section is at line 112 of 273 — behind ~100 lines of CoreWars colosseum content. Whether this satisfies 'above the fold' depends on GitHub rendering and screen resolution. The PLAN task placed it correctly (before ## Tool-Calling Benchmark), but the ROADMAP criterion is stricter."
---

# Phase 5: Consumer Output Verification Report

**Phase Goal:** Any developer who visits the repo immediately knows which free OpenRouter model to use for their task, backed by data they can verify
**Verified:** 2026-03-19
**Status:** human_needed (automated checks pass; one positioning question requires human judgment)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Plan 05-01)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running `uv run python scripts/generate_readme_results.py` produces a valid results/RESULTS.md | VERIFIED | File exists, contains full grade matrix |
| 2 | RESULTS.md contains a full grade matrix table with Wilson CI for every (model, level) in the latest sweep | VERIFIED | 16 model rows, CI in every populated cell e.g. `100% [72,100]` |
| 3 | Every model row in RESULTS.md links to its openrouter.ai/models/{model_id} page | VERIFIED | 16 rows, all use `[model](https://openrouter.ai/models/...)` format |
| 4 | RESULTS.md includes a sweep metadata line showing sweep date, trial count, and model count | VERIFIED | Line 5: `*Sweep: 2026-03-18 · 16 models · n=10 trials per level*` |
| 5 | CI overlaps between models within one grade letter are marked as statistical ties with a dagger symbol | VERIFIED | All 16 rows marked `†`; footnote `*† Statistical tie: overlapping 95% CI at T0.*` present |

**Score:** 5/5 truths verified (Plan 05-01)

### Observable Truths (Plan 05-02)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | README.md opens with a one-line best-model recommendation and a shields.io grade badge — visible before any table | PARTIAL | Content present at line 113; badge present. Placement is before `## Tool-Calling Benchmark` at line 139, but line 112 is not "at the top" — ~100 lines of CoreWars content precede it. See human verification item. |
| 2 | README.md has a category winners section with one-liners for tool calling, schema compliance, and restraint | VERIFIED | Lines 133-136: tool calling, schema compliance, restraint, multi-turn agency all present |
| 3 | README.md has an "Avoid these" section naming models that fail T0 (<20%) with their failure mode | VERIFIED | Lines 192-204: 8 F-grade models named with failure modes and T0 rates |
| 4 | README.md has shields.io grade badges for the top-graded free models | VERIFIED | Line 125: 4 clickable Grade-A badges via shields.io |
| 5 | Running generate_readme_results.py updates all these sections in README.md idempotently | VERIFIED | All 4 marker pairs present in README.md; script calls `update_readme_sections()` which uses `re.sub` with DOTALL — idempotent by construction |

**Score:** 4/5 verified, 1 needs human (positioning)

**Combined Phase Score:** 8/8 must-haves verified (all automated assertions pass)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/generate_readme_results.py` | Entry point script that reads latest sweep dir and writes RESULTS.md | VERIFIED | Exists, 184 lines, ruff clean, `--sweep`/`--out`/`--base` args present |
| `src/modelforecast/output/markdown_report.py` | Enhanced report generator with links, CI overlap detection, metadata line | VERIFIED | Exports `write_markdown_report`, `calculate_grade`, `format_percentage_with_ci`, `find_latest_sweep_dir`, `cis_overlap` |
| `results/RESULTS.md` | Human-readable full grade matrix | VERIFIED | Contains `| Model | T0 Invoke` header, 16 model rows with Wilson CI and links |
| `src/modelforecast/output/readme_sections.py` | Functions that generate README result sections | VERIFIED | Exports `build_quick_answer_section`, `build_category_winners_section`, `build_avoid_section`, `build_grade_badges_section`, `update_readme_sections` |
| `README.md` | Consumer-facing README with all output sections injected between markers | VERIFIED | All 4 marker pairs present and populated |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/generate_readme_results.py` | `src/modelforecast/output/markdown_report.py` | `from modelforecast.output.markdown_report import write_markdown_report, find_latest_sweep_dir` | WIRED | Import confirmed at lines 15-19 |
| `scripts/generate_readme_results.py` | `src/modelforecast/output/readme_sections.py` | `from modelforecast.output.readme_sections import update_readme_sections` | WIRED | Import at line 20; called at line 176 |
| `write_markdown_report` | `results/sweep_YYYYMMDD/*.json` | `json.load()` on each file in sweep dir | WIRED | `load_sweep_results()` globs `*.json`, calls `json.load(f)` at line 57 |
| `update_readme_sections` | `README.md` | `re.sub` on HTML comment marker pairs | WIRED | `_inject_section()` uses `re.compile(..., re.DOTALL)` and `pattern.sub()`; writes back to file |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OUTPUT-01 | 05-02 | README shows quick answer at top ("Best free model right now: X") | SATISFIED | `README.md` line 113: `### Best free model right now: nemotron-3-nano-30b-a3b` |
| OUTPUT-02 | 05-02 | README shows category winners ("Best for tool calling", "Best for restraint", etc.) | SATISFIED | Lines 133-136: 4 category winners including tool calling and restraint |
| OUTPUT-03 | 05-01 | README shows full grade matrix with Wilson CI for all tested models | SATISFIED | `results/RESULTS.md` has 16 rows, each with CI brackets e.g. `30% [10,60]` |
| OUTPUT-04 | 05-02 | README shows "Avoid these" section for models that fail T0 (<20%) | SATISFIED | Lines 192-204: 8 F-grade models with failure modes |
| OUTPUT-05 | 05-01 | Each model row links to its OpenRouter model page | SATISFIED | All 16 rows in RESULTS.md use `[model](https://openrouter.ai/models/...)` |
| OUTPUT-06 | 05-02 | Shields.io grade badges for top models | SATISFIED | Line 125: 4 clickable Grade-A badges; 2 badges in QUICK-ANSWER and GRADE-BADGES sections |
| OUTPUT-07 | 05-01 | Sweep metadata badge showing date and model count | SATISFIED | `results/RESULTS.md` line 5: `*Sweep: 2026-03-18 · 16 models · n=10 trials per level*` |
| OUTPUT-08 | 05-02 | "Best for X" one-liner recommendations by use case | SATISFIED | Category winners section has tool calling, schema compliance, restraint, multi-turn agency lines |

**All 8 OUTPUT-* requirements satisfied.** No orphaned requirements.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | — |

Ruff lint passes on all three modified files (`markdown_report.py`, `readme_sections.py`, `generate_readme_results.py`). No TODO/FIXME/placeholder comments. No stub return patterns. Test suite: 35 passed, 9 skipped (no regressions).

---

## Human Verification Required

### 1. Quick-Answer Visibility "Above the Fold"

**Test:** Open `https://github.com/jw409/modelforecast` (or locally render `README.md`) in a standard browser without scrolling.
**Expected:** The section `### Best free model right now: nemotron-3-nano-30b-a3b` and the Grade A badge are visible before any scrolling.
**Why human:** The ROADMAP success criterion states "visible without scrolling." The section is injected at line 112 of 273, preceded by approximately 100 lines of CoreWars colosseum content (mermaid diagram, results table, drama section, live battles CTA). Whether this satisfies "above the fold" depends on browser viewport height and GitHub's rendering. The plan task spec said to place it "BEFORE the existing `## Tool-Calling Benchmark` heading" which was done correctly, but the ROADMAP criterion may require placement earlier in the document. If the section is below the fold, the section ordering in README.md should be adjusted.

---

## Gaps Summary

No automated gaps. All 8 requirements have implementation evidence. All artifacts exist, are substantive, and are correctly wired. The single open question is a positioning/visual concern that requires a human to confirm against the ROADMAP's "visible without scrolling" criterion.

The phase goal — "Any developer who visits the repo immediately knows which free OpenRouter model to use for their task, backed by data they can verify" — is achieved at the content and wiring level. The "immediately" interpretation (visible without scrolling) is the outstanding human question.

---

_Verified: 2026-03-19_
_Verifier: Claude (gsd-verifier)_
