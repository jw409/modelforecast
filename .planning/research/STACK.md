# Technology Stack

**Project:** ModelForecast — LLM evaluation benchmark for free OpenRouter models
**Researched:** 2026-03-18
**Scope:** Subsequent milestone — fresh model sweep + consumer-facing results layer

---

## Context: What Already Exists (Do Not Re-implement)

The existing codebase has working implementations that should be **extended, not replaced**:

| Component | Status | Location |
|-----------|--------|----------|
| Wilson CI (hand-rolled) | Working, correct | `stats/confidence.py` |
| OpenRouter model discovery | Working | `models.py` |
| OpenRouter client (urllib3 + witness) | Working, no retry | `clients/openrouter.py` |
| Probe runner (T0-R0) | Working | `runner.py` |
| Rich console progress | Working | `runner.py` |
| JSON + Markdown report writers | Working | `output/` |
| Provenance tracking | Working | `verification/provenance.py` |

The milestone adds: retry robustness, optional statistical depth (effect sizes, comparison tests), and consumer-facing output layer (tiered README table, shields.io badges).

---

## Recommended Stack

### Core Runtime

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11+ | Runtime | Already locked in `pyproject.toml`. 3.11 is the floor; 3.12 preferred for perf. Do not upgrade to 3.13+ yet — scipy/statsmodels wheel availability lags. |
| uv | latest | Package manager | Already used. Keep. Faster than pip, lock-file reproducible. |

### API Layer — Existing (Keep, Add Retry)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| openai (SDK) | `>=2.0.0` | OpenRouter chat/tools API | Existing dependency pinned `>=1.0.0`. The OpenAI SDK v2.x was released (latest ~2.29 as of March 2026). The SDK's `OpenAI(base_url=..., max_retries=N)` built-in retry handles 429/5xx automatically via exponential backoff — no need for tenacity. Upgrade pin to `>=2.0.0`. |
| httpx | `>=0.27.0` | Model discovery HTTP calls | Already used in `models.py`. The openai SDK uses httpx internally. Keep as the standalone HTTP client for discovery calls. Bump minimum to 0.27+ for better timeout controls. |

**OpenAI SDK retry pattern (prefer over tenacity for this use case):**
```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    max_retries=3,           # retries on 429, 5xx, network errors
    timeout=httpx.Timeout(60.0, connect=5.0),
)
```

The SDK retries automatically on 429 with exponential backoff + jitter. This is better than wrapping with tenacity because the SDK also handles 5xx and connection errors in the same retry budget. For the free-tier rate limit (20 rpm), 3 retries with SDK-managed backoff is sufficient.

**Do NOT add tenacity.** The openai SDK already implements the OpenAI cookbook's recommended retry pattern. Adding tenacity creates a double-retry system with unpredictable stacking behavior.

**Do NOT use `clients/openrouter.py`'s urllib3 client for probe calls.** The existing custom client (witness pattern) uses raw `urllib.request` with no retry. Keep it for cost tracking, but route probe calls through the openai SDK client which has retry built in.

### Statistics Layer — Extend, Not Replace

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| statsmodels | `>=0.14.6` | Wilson CI (authoritative implementation), proportion tests, effect sizes | 0.14.6 is current stable (Dec 2025). Provides `proportion_confint(method='wilson')` — same math as the hand-rolled `stats/confidence.py` but no maintenance burden. Also provides `proportions_ztest` for pairwise model comparison and `proportion_effectsize` for Cohen's h. Add as optional dependency, not required. |
| scipy | `>=1.15.0` | Bootstrap CIs for aggregate scores, permutation tests | 1.15.0 (Jan 2025). `scipy.stats.bootstrap` added BCa method in 1.7, `rng` keyword replaces `random_state` in 1.15. Use BCa bootstrap for overall grade confidence when aggregating across dimensions — Wilson CI is per-cell only. |

**Keep the hand-rolled Wilson CI** in `stats/confidence.py` as the default path. It has no dependencies and is demonstrably correct. Add statsmodels as `[stats]` optional extra for anyone who wants to run deeper analysis.

**Add for model comparison (new, needed for fresh sweep):**
```python
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize

# Compare two models on T0: is the difference real?
stat, pvalue = proportions_ztest([s1, s2], [n1, n2])
effect_h = proportion_effectsize(p1, p2)  # Cohen's h
# h >= 0.2 small, h >= 0.5 medium, h >= 0.8 large
```

This answers "is Model A actually better than Model B at T0, or is it noise?" — especially important at n=10 where Wilson CIs overlap substantially. Report in RESULTS.md when CIs overlap.

**Do NOT add pass^k reliability metric** (tau-bench's innovation). ModelForecast runs n=10 independent trials per cell, which is equivalent — pass^k = (success_rate)^k is derivable post-hoc from the existing rate. No framework change needed.

### Output Layer — New for This Milestone

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| shields.io (static badges) | CDN | Model grade badges in README | No library needed — static URL pattern `https://img.shields.io/badge/T0-90%25-brightgreen`. Generate in `output/markdown_report.py`. Cache-safe: embed sweep date in label. |
| Shields.io endpoint badge | CDN | Live "last updated" badge | `https://img.shields.io/badge/last_sweep-2026--03--18-blue` generated from results JSON. Static string, no external JSON endpoint needed for this scope. |

**Badge color mapping (use for grade column):**
```
A → brightgreen
B → green
C → yellow
D → orange
F → red
```

**Tiered README table format** (consumer-first, power-user drill-down):
```
Quick Answer → single recommended model with grade badge
Category Winners → best T0, best T2, best overall
Full Matrix → all 16 tool-capable models × 5 dimensions
```

This pattern mirrors what BFCL's leaderboard does for academics but flattened for consumer GitHub browsing. Do NOT build a web app — the README table is the correct output for this project's scope.

### Removed / Do Not Add

| Technology | Reason to Exclude |
|------------|-------------------|
| Inspect AI (`inspect-ai`) | Designed for structured eval task suites with scorer functions and multi-model parallel execution. Appropriate if building a general eval framework. ModelForecast already has a custom probe runner that does exactly what's needed. Adding Inspect AI would require rewriting all probes to its DSL with no user-facing benefit. |
| tau-bench / tau2-bench | Evaluates domain policy compliance in customer-service scenarios. Pass^k metric is derivable from existing trial data. The framework itself doesn't apply to tool-calling probes. Reference as competitive context only. |
| BFCL evaluation harness | Academic AST-checking framework. ModelForecast's probe approach (live API calls, Wilson CI) is the differentiator vs BFCL. Don't conflate them. |
| dagster / dagster-pipes | Already in `pyproject.toml` but not used in any probes. These are heavy pipeline orchestration dependencies adding 50MB+ to installs. The `run_all()` method in `runner.py` is sufficient. Remove in next cleanup pass. |
| matplotlib / pandas | In `pyproject.toml` but not used in probe path. Archive scripts use them. Do not add to probe runner path. |
| playwright | In `pyproject.toml`, not used. Remove. |
| aiometer / self-limiters | Async rate limiting libraries. Free-tier OpenRouter already limits to 20rpm. Sequential calls with SDK retry handle this without concurrency machinery. Don't add async complexity for a CLI tool that runs once manually. |
| tenacity | See above — redundant with openai SDK built-in retry. |

### Development Dependencies

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytest | `>=9.0` | Test runner | Already in dev group. Keep. |
| ruff | `>=0.9.0` | Linting + formatting | Already in dev. Bump minimum — 0.9.x (released 2025) has significant rule improvements. |

---

## Dependency Changes for This Milestone

**pyproject.toml changes needed:**

```toml
[project]
dependencies = [
    "openai>=2.0.0",          # was >=1.0.0, SDK v2 for built-in retry
    "httpx>=0.27.0",           # was >=0.25.0
    "rich>=13.0.0",            # keep
    # REMOVE: matplotlib, dagster, dagster-pipes, pandas, playwright
]

[project.optional-dependencies]
stats = [
    "statsmodels>=0.14.6",    # Wilson CI, proportion tests, Cohen's h
    "scipy>=1.15.0",          # Bootstrap CIs for aggregates
]
dev = [
    "pytest>=9.0.1",
    "ruff>=0.9.0",
]
```

Install for full run:
```bash
uv sync --extra stats
```

Install for CI/probe only:
```bash
uv sync   # no extras needed for sweep
```

---

## OpenRouter API Patterns (Current as of March 2026)

**Model discovery** — `GET /api/v1/models` returns `supported_parameters` array. Filter `"tools" in supported_parameters` to find tool-capable free models. The `models.py` implementation is correct and current.

**Free tier limits** — 20 requests/minute, 50 requests/day for new accounts. With 16 tool-capable models × 5 dimensions × 10 trials = 800 calls. At 20rpm sequential this takes ~40 minutes. Do not parallelize — sequential is fine for a manual CLI sweep and avoids rate limit complexity entirely.

**Request IDs** — OpenRouter returns `x-request-id` in response headers. The openai SDK doesn't expose raw headers; use `response.model_extra` or switch discovery calls to httpx where raw header access matters. For provenance, the existing approach of capturing `response.id` from the body is sufficient (this is the `id` field in the chat completion object, not the header).

**Tool support detection pitfall** — `supported_parameters` is declarative self-reported by providers. A model can claim tool support but return malformed JSON or text responses. This is why T0 exists as an empirical probe — it catches models that claim but fail. Do not trust `supported_parameters` alone for filtering.

---

## Sources

| Claim | Source | Confidence |
|-------|--------|------------|
| statsmodels 0.14.6 current stable | [statsmodels PyPI](https://pypi.org/project/statsmodels/) | HIGH |
| scipy 1.15.0 bootstrap `rng` kwarg | [scipy 1.15 release notes](https://docs.scipy.org/doc/scipy-1.16.0/release/1.15.0-notes.html) | HIGH |
| openai SDK v2.29 latest | [openai PyPI](https://pypi.org/project/openai/) + GitHub releases | HIGH |
| openai SDK auto-retries 429/5xx | [OpenAI rate limits guide](https://platform.openai.com/docs/guides/rate-limits) | HIGH |
| tenacity 9.1.4 latest | [tenacity PyPI](https://pypi.org/project/tenacity/) | HIGH |
| BFCL V4 agentic eval + AST method | [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), [ICML 2025 paper](https://icml.cc/virtual/2025/poster/46593) | HIGH |
| tau2-bench pass^k metric | [tau2-bench GitHub](https://github.com/sierra-research/tau2-bench) | HIGH |
| Inspect AI framework | [inspect-ai GitHub](https://github.com/UKGovernmentBEIS/inspect_ai), [PyPI](https://pypi.org/project/inspect-ai/) | HIGH |
| OpenRouter free tier 20rpm | [OpenRouter rate limits docs](https://openrouter.ai/docs/api/reference/limits) | MEDIUM (rate limits subject to change) |
| shields.io static badge pattern | [shields.io docs](https://shields.io/docs/static-badges) | HIGH |
| proportion_effectsize Cohen's h | [statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_effectsize.html) | HIGH |
