# Phase 2: Model Roster Validation - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Ensure every model ID used in a sweep is confirmed live on OpenRouter before consuming API quota. Auto-discover current free models with tool support. Track removed/defunct models in a graveyard.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Key constraints from research:
- Use existing `models.py` `get_free_models(tools_only=True)` for auto-discovery
- Validate before sweep starts, not during (fail fast)
- Graveyard is a simple markdown file or JSON, not a database
- Current free models with tool support (from live API check in this conversation): arcee-ai/trinity-large-preview:free, arcee-ai/trinity-mini:free, meta-llama/llama-3.3-70b-instruct:free, minimax/minimax-m2.5:free, mistralai/mistral-small-3.1-24b-instruct:free, nvidia/nemotron-3-nano-30b-a3b:free, nvidia/nemotron-3-super-120b-a12b:free, nvidia/nemotron-nano-12b-v2-vl:free, nvidia/nemotron-nano-9b-v2:free, openai/gpt-oss-120b:free, openai/gpt-oss-20b:free, qwen/qwen3-4b:free, qwen/qwen3-coder:free, qwen/qwen3-next-80b-a3b-instruct:free, stepfun/step-3.5-flash:free, z-ai/glm-4.5-air:free

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/modelforecast/models.py` — `get_free_models()`, `validate_model()`, `filter_valid_models()`, `supports_tool_calling()` all exist
- `src/modelforecast/runner.py` — ProbeRunner already calls `get_free_models()` in __init__ when no models specified

### Established Patterns
- Model validation uses httpx to call OpenRouter `/api/v1/models`
- Results cached with `@lru_cache`
- Warning printed for invalid models, valid ones returned

### Integration Points
- `SweepOrchestrator` should call validation before starting sweep
- `models.py` needs a `get_tool_capable_free_models()` convenience or the existing `get_free_models(tools_only=True)` is sufficient

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>
