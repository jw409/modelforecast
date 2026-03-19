# ModelForecast

## What This Is

A competitive LLM evaluation platform that answers "which free model on OpenRouter should I use?" with empirical data. Tests tool-calling capabilities across 5 dimensions (T0 Invoke, T1 Schema, T2 Selection, A1 Linear Agency, R0 Abstain) and runs CoreWars game battles where LLMs write assembly code. Consumer-friendly leaderboard backed by open, reproducible methodology with Wilson CI provenance.

## Core Value

Empirical, reproducible answers to "which free OpenRouter model actually works for tool calling?" — not vibes, not usage rankings, not star counts. Data with confidence intervals.

## Requirements

### Validated

- ✓ T/R/A probe framework (T0-T2, A1, R0) — existing
- ✓ Wilson score confidence intervals — existing
- ✓ Provenance tracking with OpenRouter request IDs — existing
- ✓ CoreWars GPU MARS battles with 10-round adaptation — existing
- ✓ GitHub Pages arena visualization — existing
- ✓ OpenRouter model discovery and validation — existing
- ✓ Grading rubric (A-F) — existing

### Active

- [ ] Fresh sweep of all current free models (26 models, 16 with tool support)
- [ ] Consumer-friendly README with tiered results (quick answer → category winners → full matrix)
- [ ] Model roster update (remove defunct models, add new ones)
- [ ] Working run script that handles current OpenRouter API
- [ ] Results table in README showing all free models graded
- [ ] CLAUDE.md update to reflect current state
- [ ] Alignment with state-of-the-art eval patterns (BFCL V4, tau-bench, GameArena insights)

### Out of Scope

- DOOM/Angband game benchmarks — design exists but not implemented, defer
- Automated weekly CI sweeps — author runs manually when inspired
- Full web app or API — README table is the output
- Multi-agent coordination or MCP integration — standalone project
- Paid model testing — free models only
- Real-time model change detection — manual cadence

## Context

**Current State (March 2026):**
- Last commit: Feb 2026 (submodule extraction)
- RESULTS.md shows only 1 model tested (nemotron-3-nano-30b, grade C)
- Methodology doc lists 9 models, several no longer exist on OpenRouter
- OpenRouter now has 26 free models (16 with tool support)
- Notable new free models: gpt-oss-120b, nemotron-3-super-120b, qwen3-coder, minimax-m2.5, qwen3-next-80b
- Weekly probe CI configured but never activated (no secrets)

**Competitive Landscape:**
- BFCL V4 (Berkeley): de facto tool-calling benchmark, now agentic eval. Academic, not consumer-friendly.
- tau-bench / tau2-bench (Sierra): real-world domain policy compliance. pass^k reliability metric.
- GameArena (ICLR 2025): dynamic game-based eval. Prevents benchmark saturation.
- ProxyWar: dynamic LLM code generation in game arenas.
- OpenRouter compare/rankings: usage-based, not capability-tested.

**ModelForecast's lane:** The only project that empirically tests free OpenRouter models for tool-calling capability with reproducible methodology and consumer-friendly output.

**Marketing angle:** "Don't trust vibes. Trust data." — answers the question OpenRouter's own tools don't: which free model actually works.

## Constraints

- **Stack**: Pure Python 3.11+ with uv, OpenRouter API only, no local inference
- **Cost**: All testing uses free models (zero API cost for sweeps)
- **Visibility**: Repo will be made public (author decision)
- **Cadence**: Manual sweeps by author, not automated
- **Independence**: Standalone project, no TalentOS/game1 dependencies

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Tool probes primary, CoreWars secondary | Probes answer the consumer question directly; CoreWars is marketing | — Pending |
| README table over web app | Simplest distribution, least maintenance | — Pending |
| Manual sweep cadence | Author preference, avoids CI secret management | — Pending |
| Free models only | Zero cost, biggest audience ("what's the best free model?") | — Pending |
| Tiered results display | Quick answer at top, drill-down below for power users | — Pending |

---
*Last updated: 2026-03-18 after initialization*
