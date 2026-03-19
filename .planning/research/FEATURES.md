# Feature Landscape

**Domain:** Consumer-facing LLM evaluation / free model recommendation platform
**Researched:** 2026-03-18
**Milestone context:** Subsequent milestone — T/R/A probe dimensions and CoreWars battles already exist. Research gap: what does the consumer expect that ModelForecast is missing?

---

## Competitive Reference Points

| Platform | Core Value | Consumer Orientation |
|----------|-----------|----------------------|
| **BFCL V4** (Berkeley) | Academic function-calling benchmark with agentic categories | Low — academic, requires domain knowledge to interpret |
| **tau-bench / tau2-bench** (Sierra) | Real-world domain policy compliance, pass^k reliability | Low — no leaderboard UI, raw research tool |
| **Chatbot Arena / LMArena** (LMSYS) | Human pairwise preference via blind battles | Medium — engaging UI, vibes-driven not task-driven |
| **Artificial Analysis** | Independent measurement of intelligence, speed, price, latency | High — rich filtering, side-by-side comparison, daily updates |
| **OpenRouter Rankings** | Usage-based rankings from real developer traffic | Medium — shows what's popular, not what's capable |

**ModelForecast's gap:** None of the above answer "which free OpenRouter model should I use for agentic tool calling?" — that specific intersection is ModelForecast's territory. But ModelForecast currently delivers results in a static README table without discoverability, filtering, or recommendation framing.

---

## Table Stakes

Features users expect from a leaderboard tool. Missing any of these causes immediate distrust or abandonment.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Quick answer at the top** | Users want a recommendation, not a matrix to decode. Chatbot Arena, Artificial Analysis, OpenRouter all lead with a "winner." | Low | One sentence: "For tool calling, use X. It's free and scores A on all 5 dimensions." Already planned per PROJECT.md tiered results. |
| **Grade / score per model** | A-F or numeric score collapses dimensions into a scannable decision signal. All major leaderboards do this. | Low | Already implemented in grading rubric. |
| **Dimension breakdown (T/R/A)** | Power users need to know which capability failed. BFCL exposes per-category scores. Artificial Analysis exposes per-metric rows. | Low | Already implemented. |
| **Confidence intervals displayed** | Trust signal. Without CIs, any 80% score is meaningless at n=5. BFCL and credible academic leaderboards always show them. | Low | Already implemented (Wilson CI). Display them visibly, not buried in docs. |
| **When results were collected** | Freshness matters — OpenRouter models change rapidly. Any leaderboard older than 3 months loses trust. | Low | Timestamp each sweep in the results table. |
| **Model availability status** | Users click through to OpenRouter only to find a model deprecated. Frustrating and erodes trust. | Low | Flag "no longer available" with date removed. KAT Coder Pro is already an example of this failure mode. |
| **Reproducibility statement** | "How do I verify this?" is the first question a skeptical developer asks. BFCL publishes full code + data. | Low | METHODOLOGY.md already addresses this. Prominently link it from results. |
| **Provenance / request IDs** | Distinguishes ModelForecast from vibes-based rankings. The cryptographic provenance design exists in methodology — it needs to actually appear in results. | Medium | Current RESULTS.md does not expose request IDs. Wire up the existing design. |

---

## Differentiators

Features ModelForecast could build that no competitor currently provides for the free-OpenRouter-model consumer.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **"Best free model for X" guides** | Explicit category wins: "best for coding agents," "best for structured output," "best for multi-turn chat." OpenRouter's collections are usage-based; ModelForecast would be empirical. | Low | Requires tagging result rows with capability categories, then surfacing winners. No new probes needed for MVP — derive from existing T/R/A scores. |
| **Dimension-first filtering** | "Show me only models that pass A1 (multi-turn agency)." Users building agentic pipelines don't care about T0 — they need A1. No competitor offers this for free-model-only scope. | Low | Markdown tables support this poorly; a lightweight HTML page (already has GitHub Pages) handles it well. |
| **Failure taxonomy** | Surface exactly how broken models fail: "Qwen free tier: calls wrong tool 80% of the time" vs "Gemma: produces text instead of tool_call." BFCL does not describe failure modes for consumers. | Medium | Requires logging fail modes per probe run, not just pass/fail. New instrumentation. |
| **Rate limit / reliability notes** | Free models have wildly different rate limits and availability. OpenRouter free tier is 20 req/min, 200 req/day. A model with A+ capability but 50% timeout rate is not usable. | Medium | Requires measuring latency and timeout rates during sweep. |
| **CoreWars adaptation story** | The narrative "model improved +19% over 10 rounds" is unique — no other benchmark measures learning rate under pressure. Currently buried in README prose, not structured data. | Medium | Extract improvement delta as a structured metric: rank models by adaptation rate, not just final score. |
| **"Don't use this" section** | Explicit negative recommendation. The current "Broken T0" table is close but framed passively. Active framing: "These 12 models will silently fail your agent pipeline. Avoid them." Actionable anti-recommendations differentiate from mealy-mouthed academic leaderboards. | Low | Editorial tone change, no new data. |
| **OpenRouter direct link per model** | Each model row links directly to `openrouter.ai/models/[model-id]`. Reduces friction from "I want to try this" to using it. No competitor does this for the free-model audience. | Low | Pure link addition. |
| **Sweep metadata badge** | A simple badge: "n=10 trials per dimension, last run 2026-03-18, 16 models tested." Shows at-a-glance data quality. Chatbot Arena shows vote counts; this is the equivalent signal for ModelForecast. | Low | Auto-generate from sweep output. |

---

## Anti-Features

Features to explicitly NOT build. Each one represents a trap that would consume resources without serving the audience.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Web app / React frontend** | PROJECT.md already ruled this out, correctly. A web app for a manually-run monthly benchmark is maintenance overhead that delivers no additional value over a well-structured README. Artificial Analysis needs a live app because they update 8x/day. ModelForecast sweeps manually. | README table + GitHub Pages static page (already exists for CoreWars visualization). |
| **Automated CI sweeps** | OpenRouter rate limits (200 req/day free tier) make automated CI fragile. CI secrets management adds complexity. The manual cadence is honest — don't pretend to be a live dashboard. | Clearly date-stamp each sweep. Run when inspired. |
| **Paid model testing** | Scope creep. The free-model audience is well-defined. Adding paid models immediately puts ModelForecast in direct competition with Artificial Analysis, BFCL, and every other major leaderboard — all with more resources. | Keep free-only positioning. It's the one dimension where ModelForecast has a clear, uncontested lane. |
| **Human preference / ELO battles** | Chatbot Arena already does this at massive scale. Pairwise preference voting requires ongoing user traffic to be statistically valid. ModelForecast's strength is empirical capability measurement, not preference aggregation. | Stick to probe-based measurement. |
| **LLM-as-judge grading** | Using an LLM to grade other LLMs introduces circular dependency and trust questions. The current deterministic pass/fail grading (did the tool_call happen? was the schema correct?) is more trustworthy for the tool-calling domain. | Keep deterministic grading for T/R/A dimensions. |
| **Benchmark saturation arms race** | BFCL adds new categories every 6 months to stay ahead of contamination. That's appropriate for an academic benchmark with a team. ModelForecast should not try to match BFCL's breadth — it will always lose. | Stay focused on the 5 existing dimensions plus CoreWars. Depth over breadth. |
| **Contributor community / tiered trust system** | The tiered trust design in METHODOLOGY.md (Unverified/Verified/Trusted/Core badges) is architecturally sound but requires significant community management. At current scale (one maintainer, manually-run sweeps), this is premature complexity. | Implement only when the project has demonstrated community interest. |

---

## Feature Dependencies

```
Sweep execution (new data)
  → Timestamped results table           (table stakes: freshness)
    → Model availability flags          (table stakes: availability status)
    → Sweep metadata badge              (differentiator: data quality signal)
    → Failure taxonomy (fail modes)     (differentiator: requires new instrumentation)
    → Rate limit / reliability notes    (differentiator: requires new instrumentation)

Existing T/R/A scores
  → "Best for X" guides                 (differentiator: editorial synthesis)
  → "Don't use this" section            (differentiator: editorial tone)
  → Dimension-first filtering           (differentiator: HTML page, low complexity)
  → OpenRouter direct links             (differentiator: link addition)

Existing CoreWars adaptation data
  → Adaptation rate as structured metric (differentiator: requires extraction)
```

---

## MVP Recommendation

**Current state:** The methodology and probe framework are solid. What's missing is execution (fresh data) and presentation (consumer framing). The highest-value work is not new features — it's running the sweep and presenting results in a way that answers the question immediately.

Prioritize for this milestone:

1. **Fresh sweep** — Run all 26 free models (16 with tool support). Without data, no presentation matters.
2. **Tiered results display** — Quick answer at top, grade table below, broken models at the bottom. Already planned in PROJECT.md.
3. **Model availability flags** — Visibly mark deprecated models. Prevents user trust erosion.
4. **Timestamped sweep metadata** — Show n= and run date prominently. Converts "vibes" into "data."
5. **"Best for X" one-liners** — Three editorial sentences derived from results: best overall, best for agentic pipelines (A1 focus), best T1/T2 for structured output work.
6. **"Avoid these" section** — Active negative recommendation for the T0 failures.

Defer:

- Failure taxonomy logging: useful but requires new instrumentation before the sweep runs
- Rate limit / reliability notes: useful but requires separate timing instrumentation
- HTML dimension-filtering: GitHub Pages page exists; add after results are fresh
- Adaptation rate metric extraction from CoreWars: medium complexity, second milestone

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table stakes features | HIGH | Consistent across Artificial Analysis, BFCL, OpenRouter, Chatbot Arena — all verified via direct page fetch |
| Differentiators | MEDIUM | Derived from gap analysis of competitor pages; OpenRouter and BFCL pages fetched directly, rest via WebSearch |
| Anti-features | MEDIUM | Contamination / saturation findings from multiple sources (Goodeye Labs 2025 review, ACL Anthology papers); LLM-as-judge risk well-documented |
| Competitor feature set | HIGH | Artificial Analysis and BFCL leaderboard pages directly fetched; OpenRouter rankings page directly fetched |

---

## Sources

- [BFCL V4 Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — directly fetched
- [BFCL V4 Agentic: Web Search](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [BFCL V4 Memory](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html)
- [BFCL V4 Format Sensitivity](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/leaderboards/models) — directly fetched
- [OpenRouter Rankings](https://openrouter.ai/rankings) — directly fetched
- [OpenRouter Free Models Collection](https://openrouter.ai/collections/free-models)
- [OpenRouter Tool Calling Models](https://openrouter.ai/collections/tool-calling-models)
- [tau2-bench: Sierra Research](https://sierra.ai/uk/blog/benchmarking-agents-in-collaborative-real-world-scenarios)
- [2025 Year in Review for LLM Evaluation: When the Scorecard Broke](https://www.goodeyelabs.com/insights/llm-evaluation-2025-review)
- [Chatbot Arena LMSYS Review 2025: Is the LLM Leaderboard Reliable?](https://skywork.ai/blog/chatbot-arena-lmsys-review-2025/)
- [OpenRouter Free Models: Which Work for AI Agents](https://brainroad.com/openrouter-free-models-which-ones-actually-work-for-ai-agents/)
- [Best LLM Leaderboard 2026 — Dextra Labs](https://dextralabs.com/blog/best-llm-leaderboard/)
- [Benchmarking is Broken — arXiv 2510.07575](https://arxiv.org/html/2510.07575v1)
