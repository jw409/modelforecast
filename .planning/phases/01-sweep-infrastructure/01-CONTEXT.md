# Phase 1: Sweep Infrastructure - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Harden the ProbeRunner so it can reliably complete a full sweep of 26 free models (800+ API calls) with retry on 429s, checkpoint-resume for interrupted runs, provider header capture, failure mode classification, and timestamped output directories.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Key constraints from research:
- Use openai SDK 2.x built-in retry (max_retries=N) rather than tenacity
- Sequential API calls, no async (20rpm free tier makes concurrency pointless)
- JSON-per-file storage (no database)
- Provider header may need httpx interceptor if openai SDK doesn't expose raw headers

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/modelforecast/runner.py` — ProbeRunner class, extend don't replace
- `src/modelforecast/verification/provenance.py` — ProvenanceTracker, already captures per-trial records
- `src/modelforecast/output/json_report.py` — write_individual_result, write_json_report
- `src/modelforecast/stats/confidence.py` — wilson_interval

### Established Patterns
- Probes return `ProbeResult` with `success`, `tool_called`, `raw_response`, `latency_ms`
- Results are JSON files organized by model provider directory
- Rich console for progress display

### Integration Points
- `ProbeRunner.__init__` creates OpenAI client — retry config goes here
- `ProbeRunner.run_level` calls `probe.run(model, client)` — failure classification wraps this
- `write_individual_result` writes JSON — provider header goes into this output

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
