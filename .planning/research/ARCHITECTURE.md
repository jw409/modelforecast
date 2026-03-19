# Architecture Patterns

**Domain:** LLM evaluation benchmark platform (tool-calling probes + game-based benchmarks)
**Project:** ModelForecast
**Researched:** 2026-03-18

---

## Current Architecture (What Exists)

The codebase has working, well-structured foundations. Architecture decisions should extend, not replace.

```
ProbeRunner
  ├── ModelDiscovery (models.py — lru_cache, OpenRouter /models)
  ├── ProbeSet {T0, T1, T2, A1, R0, DagProbe} (probes/)
  ├── ProvancenceTracker (verification/provenance.py — per-run, per-trial)
  ├── OpenRouterClient (clients/openrouter.py — urllib, witness pattern)
  └── OutputWriters
        ├── write_individual_result() → results/{model_slug}__level_{N}.json
        └── write_json_report() + write_markdown_report() → results/
```

**What works well:**
- Per-trial provenance with OpenRouter request IDs already captured
- Wilson CI already computed per (model, level) result
- T0 skip-ahead gate (skip higher probes if T0 < 20%) — correct design
- Individual JSON per (model, level) avoids losing partial sweeps
- `lru_cache` on model list fetch — no repeated API calls during a sweep

**What is missing for a fresh sweep of 26 models:**
- Rate limiting / backoff on 429s (OpenRouter enforces per-model rate limits)
- Retry with jitter for transient failures
- Resume-from-checkpoint (rerunning after partial failure re-runs completed models)
- CoreWars integration into the sweep pipeline
- Consumer output (tiered README, badges)
- Result versioning (no date/sweep-ID on result files — overwriting risk)

---

## Recommended Architecture

### Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│ CLI Entry Point (run_sweep.py / __main__.py)                     │
│   --models [all|list]  --trials N  --levels [0-4]  --resume     │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  SweepOrchestrator    │  ← NEW (thin coordinator)
        │  - loads sweep config │
        │  - reads checkpoint   │
        │  - iterates models    │
        │  - writes checkpoint  │
        └───┬───────────────────┘
            │
    ┌───────▼──────────────────────────────────────────────────┐
    │  ProbeRunner (existing — extend, do not rewrite)          │
    │  + RateLimiter injected as dependency                     │
    │  + retry wrapping in run_level()                          │
    └───┬──────────────────────────────────────────────────────┘
        │
   ┌────▼──────────┐    ┌──────────────────┐    ┌────────────────┐
   │  RateLimiter  │    │  ResultStore     │    │  CoreWarsBridge│
   │  (NEW)        │    │  (NEW)           │    │  (NEW)         │
   └───────────────┘    └──────────────────┘    └────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  Storage Layer            │
                    │  results/{sweep_id}/      │
                    │    *.json (existing)      │
                    │    sweep_manifest.json    │
                    │    checkpoint.json        │
                    └───────────────────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  OutputPipeline (NEW)      │
                    │  - grade all models        │
                    │  - render README table     │
                    │  - emit badges JSON        │
                    │  - write RESULTS.md        │
                    └───────────────────────────┘
```

---

## Component Boundaries

| Component | Responsibility | Reads From | Writes To | Communicates With |
|-----------|---------------|------------|-----------|-------------------|
| SweepOrchestrator | Coordinates a full sweep run | sweep config, checkpoint | checkpoint, sweep_manifest | ProbeRunner, ResultStore |
| ProbeRunner | Runs probes for one model (existing) | probe definitions | individual result JSONs | RateLimiter, ProvenanceTracker |
| RateLimiter | Enforces per-model call pacing | nothing | nothing | ProbeRunner (injected) |
| ResultStore | Reads/aggregates sweep results | results/{sweep_id}/ | nothing | OutputPipeline, SweepOrchestrator |
| CoreWarsBridge | Wraps CoreWars game execution | games/corewars/ | game result JSON | SweepOrchestrator |
| OutputPipeline | Generates consumer output from results | ResultStore | README.md, RESULTS.md, badges.json | nothing |

**Boundary rule:** OutputPipeline is read-only against results. It never calls the API. It can be re-run at any time from stored results without network access.

---

## Data Flow

### 1. Sweep Execution Flow

```
CLI args
  → SweepOrchestrator.load_config()
  → SweepOrchestrator.read_checkpoint()   # skip completed models
  → for each pending model:
      ProbeRunner.run_model(model)
        → for each level:
            RateLimiter.acquire()          # blocking wait if needed
            probe.run(model, client)       # single API call
            ProvenanceTracker.record()     # trial record with request_id
        → write_individual_result()        # results/{sweep_id}/{model}__level_{N}.json
      SweepOrchestrator.update_checkpoint(model)
  → SweepOrchestrator.write_manifest()
```

### 2. Output Generation Flow (separate command, re-runnable)

```
ResultStore.load_sweep(sweep_id)       # reads all *.json from results/{sweep_id}/
  → aggregate by model
  → calculate_grade() per model
  → sort by grade desc, then T0 rate desc
OutputPipeline.render_readme()         # tiered table: quick answer → full matrix
OutputPipeline.render_badges_json()    # shields.io endpoint data
OutputPipeline.render_results_md()     # detailed table with CIs
```

### 3. Data Shape at Rest

Individual result file (already correct format, extend not change):
```json
{
  "submission_id": "sub_abc123",
  "sweep_id": "sweep_20260318",        // ADD: links to sweep manifest
  "timestamp": "2026-03-18T...",
  "contributor": "jw",
  "environment": { "python_version": "..." },
  "probes": {
    "model": "google/gemini-2.0-flash-exp:free",
    "level": 0,
    "trials": [...]                    // per-trial provenance preserved
  },
  "summary": {
    "successes": 9, "trials": 10,
    "rate": 0.9,
    "wilson_ci_95": [0.597, 0.997]
  }
}
```

Sweep manifest (new):
```json
{
  "sweep_id": "sweep_20260318",
  "started_at": "2026-03-18T10:00:00Z",
  "completed_at": "2026-03-18T11:23:00Z",
  "models_attempted": 26,
  "models_completed": 26,
  "trials_per_level": 10,
  "max_level": 4,
  "tool_support_snapshot": {
    "google/gemini-2.0-flash-exp:free": true,
    ...
  }
}
```

---

## Answers to the Five Architecture Questions

### 1. Running Sweeps Across 20+ Models with Rate Limiting and Error Recovery

**Rate limiting:** Inject a `RateLimiter` into `ProbeRunner`. Use a token bucket with per-model tracking. OpenRouter free models typically enforce ~10 req/min per model. A simple `time.sleep` with jitter is sufficient — this is a manual, author-run tool, not a production service. Async is unnecessary complexity.

```python
class RateLimiter:
    """Per-model token bucket with jitter."""
    def __init__(self, calls_per_minute: int = 8, jitter_range: float = 0.5):
        self._last_call: dict[str, float] = {}
        self._min_interval = 60.0 / calls_per_minute

    def acquire(self, model: str) -> None:
        now = time.time()
        last = self._last_call.get(model, 0)
        wait = self._min_interval - (now - last)
        if wait > 0:
            jitter = random.uniform(0, 0.5)
            time.sleep(wait + jitter)
        self._last_call[model] = time.time()
```

**Error recovery:** Wrap `probe.run()` in a retry loop with exponential backoff. Catch `429 Too Many Requests` and `5xx` errors. After 3 failures on a model+level, record a null result and continue — don't abort the sweep.

**Resume from checkpoint:** `SweepOrchestrator` writes `results/{sweep_id}/checkpoint.json` after each completed model. On restart with `--resume`, it reads checkpoint and skips models already in it. This means individual result files are the source of truth, not the checkpoint — checkpoint is just a skip index.

### 2. Storing and Versioning Results

**Recommended: JSON files organized by sweep ID.** Not SQLite or DuckDB.

Rationale:
- Results are already individual JSON files per (model, level) — this is correct
- SQLite adds a dependency and complicates git history (binary file)
- DuckDB would enable SQL queries but this project has no query workload — only aggregation for output generation
- JSON files are human-readable, git-diffable, and the existing format is sound
- Sweep versioning comes from directory structure, not DB rows

Directory structure:
```
results/
  sweep_20260318/               # one directory per sweep run
    sweep_manifest.json          # metadata about the sweep
    checkpoint.json              # resume state (gitignored)
    google_gemini-2-0-flash-exp_free__level_0.json
    google_gemini-2-0-flash-exp_free__level_1.json
    ...
  sweep_20260201/               # previous sweep preserved
    ...
  latest -> sweep_20260318/     # symlink to most recent
  RESULTS.md                    # generated, points to latest
```

`checkpoint.json` should be gitignored. All other result files should be committed — they are the reproducibility artifact.

**Migration note:** The current code writes flat into `results/` without sweep IDs. The `write_individual_result()` function needs an `output_dir` that includes the sweep subdirectory. This is a 1-line change at the call site.

### 3. Generating Consumer-Friendly Output

**Tiered README structure** (in order):

```
# ModelForecast

> Which free OpenRouter model actually works for tool calling?

## Quick Answer
[Grade A models highlighted] — 2-3 sentences, no tables

## Category Winners
| Category | Winner | Grade |
|----------|--------|-------|
| Best overall | model-x | A |
| Best tool schema | model-y | B |
| Most consistent | model-z | A |

## Full Results Matrix
[existing RESULTS.md table format — all models, all probe columns]

## Methodology
[brief, link to CONTRIBUTING.md for full detail]
```

**Generation:** `OutputPipeline.render_readme()` reads the latest sweep, computes rankings, injects into a Jinja2 template (or string template — no new deps needed). Writes to `README.md` directly at project root.

**Badges:** Generate `var/badges.json` in shields.io endpoint format:
```json
{"schemaVersion": 1, "label": "top model", "message": "gemini-flash", "color": "green"}
```
If the repo is public, these can be served via a static hosting solution. For now, static badges via shields.io with hardcoded values are sufficient.

**Key design rule:** Output generation is a separate command (`python -m modelforecast report`) that reads from stored results. It never calls the API. This means the README can be regenerated without re-running the sweep.

### 4. Supporting Both Tool-Calling Probes and Game-Based Benchmarks

**Separate pipeline, unified output.** Do not force CoreWars results through `ProbeRunner`.

CoreWars games produce a different result shape (win/loss/draw over N rounds, code quality metrics) that doesn't map cleanly onto the probe (successes/trials/wilson_ci) structure. Forcing them into the same runner creates false symmetry.

Architecture:
```
SweepOrchestrator
  ├── runs ProbeRunner for all models → probe results
  └── runs CoreWarsBridge for subset of models → game results

OutputPipeline
  ├── reads probe results → tool-calling table
  └── reads game results → CoreWars leaderboard section
  └── merges into unified README
```

`CoreWarsBridge` wraps the existing `games/corewars/` infrastructure. It should:
- Accept a list of models to pit against each other
- Delegate to existing `arena_session.py` or `model_benchmark.py`
- Return a structured game result JSON compatible with the storage layer
- Be entirely skippable (`--skip-games` flag) since CoreWars is secondary to tool probes

Game result schema (new):
```json
{
  "sweep_id": "sweep_20260318",
  "game": "corewars",
  "timestamp": "...",
  "matchups": [
    {"model_a": "...", "model_b": "...", "rounds": 10,
     "wins_a": 6, "wins_b": 3, "draws": 1}
  ]
}
```

### 5. Making Results Reproducible

**What already exists and is correct:**
- OpenRouter request ID captured per trial (`openrouter_request_id` in trial record)
- SHA256 hash of prompt and response stored per trial
- `submission_id` per (model, level) run
- Environment capture (Python version, openai SDK version, OS)

**What to add:**

Seed tracking: The current implementation uses `temperature=0.1` but no explicit seed. OpenRouter does not guarantee seeded outputs — this is a limitation of the API, not a bug. Document it in methodology rather than trying to solve it.

Provenance at sweep level (not just per-result): Add `sweep_id` field to every individual result so they can be traced back to a specific run. Currently, results are isolated — there's no way to know which sweep produced them.

Probe version hash: Hash the probe source (prompt + tool definition) and store it in result JSON. This catches silent probe mutations that would make results non-comparable across sweeps.

```python
def probe_fingerprint(probe) -> str:
    content = json.dumps({
        "prompt": probe.prompt,
        "tools": probe.tools
    }, sort_keys=True)
    return hash_content(content)[:12]
```

Request ID coverage: The current `trial_record` captures `openrouter_request_id` but only as optional. Make it required — if it's missing, the trial is not auditable. Log a warning when the API response lacks an ID.

---

## Build Order (Phase Dependencies)

This is the recommended implementation sequence based on dependencies:

**Phase 1: Fix the Sweep Runner (unblocks everything)**
1. Add `RateLimiter` class — no external deps, pure logic
2. Add retry wrapper in `run_level()` — wraps existing code
3. Add `sweep_id` to `SweepOrchestrator` and propagate to output dir
4. Add `checkpoint.json` write/read in `SweepOrchestrator`
5. Add `probe_fingerprint()` to provenance — 5 lines

Dependency: None. Can be built and tested without running a full sweep.

**Phase 2: Update Model Roster**
1. Run `models.py:get_free_models()` to get current 26 models
2. Identify which have tool support via `get_tool_support_matrix()`
3. Update `METHODOLOGY.md` with current roster

Dependency: Phase 1 complete (need working sweep runner to validate models).

**Phase 3: Run the Fresh Sweep**
1. Execute `python -m modelforecast sweep --resume` against all 26 models
2. Results land in `results/sweep_20260318/`

Dependency: Phase 1 + 2 complete.

**Phase 4: Consumer Output**
1. Implement `OutputPipeline` (reads Phase 3 results)
2. Implement tiered README template
3. Write `RESULTS.md` and update root `README.md`

Dependency: Phase 3 results exist.

**Phase 5: CoreWars Integration (optional)**
1. Implement `CoreWarsBridge` wrapping existing `games/corewars/`
2. Run game benchmarks for top-grade models from Phase 3
3. Add CoreWars section to OutputPipeline

Dependency: Phase 4 complete (need to know which models to run games against).

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Migrating to a Database
**What:** Moving from JSON files to SQLite or DuckDB for result storage
**Why bad:** Adds dependency, breaks git-readable results, introduces migration burden, no query workload justifies it. JSON files can be queried with `jq` when needed.
**Instead:** Keep JSON files, use sweep-ID directories for versioning.

### Anti-Pattern 2: Async Sweep Execution
**What:** Using `asyncio` to run multiple models concurrently
**Why bad:** OpenRouter free tier rate limits are per-model already tight. Concurrent calls create more 429s, not fewer. The sweep takes ~1-2 hours; making it parallel saves minutes at the cost of significant complexity.
**Instead:** Sequential with rate limiting. Simple, debuggable, correct.

### Anti-Pattern 3: Merging CoreWars into ProbeRunner
**What:** Forcing game results through the probe (successes/trials/wilson_ci) schema
**Why bad:** Game results are win/loss/draw — not binomial. Shoehorning creates meaningless confidence intervals. Grade calculation breaks.
**Instead:** Separate `CoreWarsBridge` with its own result schema. Merge at OutputPipeline level only.

### Anti-Pattern 4: Generating README During the Sweep
**What:** Writing `README.md` inline as models complete
**Why bad:** Partial results produce misleading output. Interrupted sweeps leave README in inconsistent state.
**Instead:** README generation is always a post-sweep step, run explicitly.

### Anti-Pattern 5: Hardcoding Model Lists
**What:** Maintaining a static list of model IDs in code
**Why bad:** Models appear and disappear from OpenRouter frequently. The Jan 2026 sweep showed several models from the methodology doc no longer exist.
**Instead:** Always discover from `get_free_models()` at sweep time. Store the snapshot in `sweep_manifest.json` for reproducibility.

---

## Scalability Considerations

This is a manual-cadence tool. Scalability is not a concern. Document the non-requirements explicitly:

| Concern | At current scope | Notes |
|---------|-----------------|-------|
| Concurrent sweeps | Not needed | Single author, manual cadence |
| Result database | Not needed | <1MB of JSON per full sweep |
| API authentication | Single key | OpenRouter free tier |
| Distributed runners | Not needed | One machine per sweep |
| Automated CI | Not needed | Author-driven |

If the project ever adds automated CI sweeps, the only architectural change needed is: move `checkpoint.json` to an artifact store and add a GH Actions workflow. The sweep code itself doesn't need to change.

---

## Sources

- Existing source code: `/home/jw/dev/modelforecast/src/modelforecast/` (HIGH confidence — direct inspection)
- OpenRouter API behavior: observed from `clients/openrouter.py` implementation patterns and `models.py` (MEDIUM confidence)
- Rate limiting design: conservative estimate based on OpenRouter documented free tier limits (MEDIUM confidence — verify against live API during Phase 1)
- shields.io badge endpoint format: standard public API, stable for years (HIGH confidence)
