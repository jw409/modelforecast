---
plan_version: 1.0
plan_id: "2025-12-04-capability-dimensions-migration"
created: 2025-12-04
mode: implementation

# Complexity signals
estimated_complexity: medium
estimated_scope: "~500 lines across 8 files (README, runner, charts, 5 probes)"
context_sensitive: true

# Context loading strategy
context_files:
  phase1: [README.md, scripts/generate_charts.py]
  phase2: [src/modelforecast/__main__.py, src/modelforecast/runner.py]
  phase3: [src/modelforecast/probes/level*.py]
  shared: [contexts/modelforecast/private/plans/2025-12-04-modelforecast-v2-redesign.md]

# Verification
verification:
  phase1: "Visual review of README.md + regenerate charts"
  phase2: "uv run python -m modelforecast --help (check new flags)"
  phase3: "uv run python -m modelforecast --model kwaipilot/kat-coder-pro:free --probe M0 --trials 3"

# Execution hints
parallelization_potential: high  # Phases 1a/1b/1c can run in parallel
checkpoint_after_phases: true
rollback_safe: true
resumable: true
---

# Migrate ModelForecast from L0-L4 to Capability Dimensions

## Objective

Replace the misleading "Level 0-4" nomenclature with three orthogonal capability dimensions:
- **TOOL CALLING (T)**: T0 Invoke, T1 Schema, T2 Selection
- **RESTRAINT (R)**: R0 Abstain (was L4 Adversarial) - with falsification test
- **AGENCY (A)**: A1 Linear (was L3 Multi-turn)

**Why this matters**: Current "levels" imply difficulty progression, but L4 (restraint) is empirically easier than L3 (agency). People argue "why is L3 harder than L4?" instead of understanding they test different capabilities. The new taxonomy makes this crystal clear.

## Context

**Current state**:
- README uses L0-L4 terminology throughout
- Charts reference "L0 Basic", "L3 Multi-turn", etc.
- CLI uses `--level N` flag
- Probe files named `level0_basic.py`, etc.
- JSON results store `level: N`

**Mapping (old → new)**:
| Old | New | Category |
|-----|-----|----------|
| L0 Basic | T0 Invoke | TOOL CALLING |
| L1 Schema | T1 Schema | TOOL CALLING |
| L2 Selection | T2 Selection | TOOL CALLING |
| L3 Multi-turn | A1 Linear | AGENCY |
| L4 Adversarial | R0 Abstain | RESTRAINT |

**Dimension Naming Decision**:
- **TOOL CALLING (T)** instead of "Mechanics" - more direct, says what it tests
- **RESTRAINT (R)** - kept, but with falsification requirement
- **AGENCY (A)** - kept, aligns with industry "agentic workflows" terminology

**Restraint Falsification** (critical - avoid "dumb silence" false positives):
- R0 must be a **paired test**, not just "didn't call tools":
  - **R0a (Abstention)**: Given code tools + "What's the capital of France?" → model should NOT call tools
  - **R0b (Helpfulness)**: Same prompt → model MUST provide correct answer via text
  - **Pass criteria**: Both R0a AND R0b pass. Silence fails R0b. Wrong tool call fails R0a.
- This prevents a broken model from getting "restraint credit" for being unresponsive

**Constraints**:
- Backwards compatibility: Keep `--level` flag working (map internally)
- No data loss: Existing results stay valid, just relabeled
- Atomic commit: All changes in one commit (no intermediate broken state)

---

## Gemini Advisory Feedback (Cross-Check)

**Endorsement**: "Proceed with the migration. The M/R/A taxonomy transforms the project from a 'Can it use tools?' check into a 'How sophisticated is this model?' analysis."

**Key Insights Incorporated**:

1. **"Safe but Stupid" vs "Smart but Hallucinatory"** framing:
   - High R, Low A = safe but can't do complex tasks
   - High A, Low R = capable but might hallucinate tool calls
   - This insight should be in README to help developers choose models

2. **A2 Parallel might be Mechanics, not Agency**: Gemini notes parallel tool calling is often API-level (OpenAI `tool_calls` array) rather than requiring agentic planning. Consider whether A2 belongs in M or A dimension.

3. **Numbering consistency**: Consider A0 = None/Single-turn (baseline), A1 = Linear multi-turn. This makes the progression cleaner.

4. **Exit Gates**: Define strict gates - if M0 fails, can't test Agency. Can test Restraint even with poor Mechanics.

5. **Restraint covers discernment**: R0 tests whether model knows when to answer from internal knowledge vs call tools.

---

## Phases

### Phase 1: Public-Facing Updates (README + Charts)
**Goal**: User-visible changes that communicate new taxonomy

**1a. README.md rewrite**:
- Replace all "L0/L1/L2/L3/L4" with "T0/T1/T2/R0/A1"
- Add "Capability Dimensions" glossary section explaining the three axes (T/R/A)
- Update tables: "Level | Test" → "Dimension | Test"
- Update "Multi-Turn Cliff" → "The Agency Gap" (emphasizes capability not difficulty)
- Add "Safe but Stupid vs Smart but Hallucinatory" framing (from Gemini advisory)
- Explain R0 falsification: must abstain from tools AND provide helpful answer
- Update chart alt-text to reference new names

**1b. Chart regeneration** (`scripts/generate_charts.py`):
- Change x-axis labels: "L0 Basic" → "T0 Invoke", "L3 Multi-turn" → "A1 Linear", "L4 Adversarial" → "R0 Abstain"
- Update `level_names` list in `generate_multi_level_comparison()`
- Update legend text
- Update chart titles

**1c. Add METHODOLOGY.md glossary**:
- Full definitions of T/R/A dimensions
- Explain why this taxonomy vs levels
- Future-proof: mention A2/A3/A4 coming (parallel, tree, diamond)
- Document R0 falsification methodology

### Phase 2: CLI Interface Updates
**Goal**: New `--probe` flag, backwards-compatible `--level` mapping

**Changes to `__main__.py`**:
- Add `--probe` flag: `--probe T0|T1|T2|R0|A1` (mutually exclusive with `--level`)
- Add `--dimension` flag: `--dimension T|R|A` (run all probes in dimension)
- Keep `--level` working: internally map to new probe codes
- Update help text with new terminology

**Changes to `runner.py`**:
- Update `PROBES` dict keys: `0` → `"T0"`, `3` → `"A1"`, `4` → `"R0"`, etc.
- Add mapping dict for backwards compatibility
- Update console output messages

### Phase 3: Internal Code Updates (Probe Files)
**Goal**: Rename files and update internal naming

**File renames**:
```
level0_basic.py → t0_invoke.py
level1_schema.py → t1_schema.py
level2_selection.py → t2_selection.py
level3_multiturn.py → a1_linear.py
level4_adversarial.py → r0_abstain.py
```

**Class renames**:
```
Level0BasicProbe → T0InvokeProbe
Level1SchemaProbe → T1SchemaProbe
Level2SelectionProbe → T2SelectionProbe
Level3MultiTurnProbe → A1LinearProbe
Level4AdversarialProbe → R0AbstainProbe
```

**Update imports** in `runner.py` and `probes/__init__.py`

---

## Execution Strategy for Subagents

**For Claude Sonnet 4.5 subagents**: This is a refactoring task. The tests already pass - we're renaming, not changing logic. Use search-and-replace for consistent changes, verify each file compiles.

**Parallelization**:
- Phase 1a (README) + Phase 1b (charts) + Phase 1c (METHODOLOGY) can run in parallel
- Phase 2 depends on Phase 1 (need to know final terminology)
- Phase 3 depends on Phase 2 (imports need new names)

**Subagent assignments**:
1. **Subagent A**: README.md rewrite (Phase 1a)
2. **Subagent B**: generate_charts.py update (Phase 1b)
3. **Subagent C**: CLI + runner.py update (Phase 2)
4. **Subagent D**: Probe file renames + class updates (Phase 3)

---

## Success Criteria

- [ ] README uses T0/T1/T2/R0/A1 terminology consistently
- [ ] Charts regenerate with new axis labels
- [ ] `--probe T0` and `--level 0` both work
- [ ] `--dimension T` runs T0+T1+T2
- [ ] All probe files renamed to new convention
- [ ] `uv run python -m modelforecast --help` shows new terminology
- [ ] Existing results/ JSON files still parseable (backwards compat)
- [ ] Git commit atomic with all changes

---

## Resources

- Design doc: `contexts/modelforecast/private/plans/2025-12-04-modelforecast-v2-redesign.md`
- Current README: `/home/jw/dev/modelforecast/README.md`
- Chart generator: `/home/jw/dev/modelforecast/scripts/generate_charts.py`
- CLI entry: `/home/jw/dev/modelforecast/src/modelforecast/__main__.py`
- Runner: `/home/jw/dev/modelforecast/src/modelforecast/runner.py`
- Probes: `/home/jw/dev/modelforecast/src/modelforecast/probes/level*.py`

---

## Timeline Estimate

- Phase 1 (parallel): 30-45 minutes total
- Phase 2: 20-30 minutes
- Phase 3: 30-40 minutes
- Integration + testing: 15-20 minutes

**Total**: ~90-135 minutes with parallel execution
