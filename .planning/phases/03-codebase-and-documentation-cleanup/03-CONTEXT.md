# Phase 3: Codebase and Documentation Cleanup - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Clean install, current SDK, accurate documentation. Remove dead dependencies, upgrade openai SDK to 2.x, update CLAUDE.md and METHODOLOGY.md to reflect current project state and OpenRouter free model roster.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — cleanup/maintenance phase.

Key constraints from research:
- Strip dagster, matplotlib, pandas, playwright from pyproject.toml dependencies
- Upgrade openai to >=2.0.0 (currently uses 1.x, latest is 2.29)
- Drop tenacity if present (openai 2.x has built-in retry)
- METHODOLOGY.md must list only current free models (see Phase 2 CONTEXT.md for live list)
- CLAUDE.md should reflect: T/R/A dimension names, sweep CLI command, output structure
- Fix Pyright type warnings from Phase 1 (optional member access on ProbeResult | None)

</decisions>

<code_context>
## Existing Code Insights

### Files to Update
- `pyproject.toml` — remove dead deps, bump openai version
- `CLAUDE.md` — outdated (last updated 2025-12-10), references wrong model list
- `docs/METHODOLOGY.md` — lists 9 defunct models, needs current roster

### Established Patterns
- `uv sync` for dependency management
- Rich console for output
- ruff for linting/formatting

### Integration Points
- `uv.lock` will regenerate after pyproject.toml changes
- Runner already uses openai client — SDK upgrade should be backward compatible

</code_context>

<specifics>
## Specific Ideas

No specific requirements — cleanup phase

</specifics>

<deferred>
## Deferred Ideas

None

</deferred>
