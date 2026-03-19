# Domain Pitfalls: LLM Evaluation Benchmark Platform

**Domain:** Competitive LLM evaluation platform targeting free OpenRouter models
**Researched:** 2026-03-18
**Confidence:** MEDIUM-HIGH (OpenRouter specifics from official docs + community issues; stats from literature)

---

## Critical Pitfalls

These mistakes cause invalid results, broken sweeps, or loss of consumer trust. Each has hit production benchmarking projects.

---

### Pitfall 1: Provider Variance Invisibility

**What goes wrong:** A free model's tool-calling performance varies significantly depending on which backend OpenRouter routes the request to. The same model ID, same weights, different provider infrastructure — different results. OpenRouter's own data shows tool-call propensity varying by 40-60 percentage points across providers for the same model. The benchmark records the model ID, but not the provider, so results are unreproducible and the variance looks like noise.

**Why it happens:** OpenRouter routes `:free` requests across whatever providers are available at that moment. Different providers run different vLLM versions, different quantization, and different inference parameter handling. OpenRouter publicly documented this problem in their "Provider Variance: Introducing Exacto" announcement (August 2025), noting that tool calling was specifically vulnerable. A Microsoft Azure incident showed GPT-OSS-120B scoring 36.7% vs 93.3% depending purely on which vLLM version was running.

**Consequences:**
- Run A gives model X a grade of B. Run B gives the same model a grade of D. Both are "correct."
- Confidence intervals become meaningless — they capture statistical noise from sampling, not the underlying systematic variance from provider routing.
- Cherry-picking accusations become valid even when you're not cherry-picking.

**Prevention:**
- Record the `x-openrouter-provider` response header on every request. OpenRouter includes this in API responses.
- Treat provider as a dimension, not noise. A model's score is "70% on provider CloudA, 40% on provider CloudB" not "55% with high variance."
- Consider using the `:exacto` variant for tool-calling evals — OpenRouter routes these to providers they've benchmarked as reliable for tool use.
- Flag runs where provider distribution is heavily skewed (>70% requests to one provider).

**Warning signs:**
- High variance across separate runs that exceeds expected binomial variance given n
- Model scores shifting dramatically between sweep dates without model updates
- CI bands that are technically correct but contain the "real" performance difference from provider routing

**Phase:** Address in MVP sweep phase. Add provider-header capture before first public result. Retroactively unfixable.

---

### Pitfall 2: The tool_choice vs tools Distinction

**What goes wrong:** A model advertised as "supports tool calling" on OpenRouter may support the `tools` parameter but not the `tool_choice` parameter. Probes that rely on `tool_choice: required` to force a tool invocation will silently 404 or return unexpected behavior on these models, making them appear to fail T0 when they would pass with a different invocation strategy.

**Why it happens:** OpenRouter's capability metadata distinguishes `tools` from `tool_choice` support. The `:free` variant documentation does not prominently surface which specific `tool_choice` values each backend accepts. This was documented in a 2025 Goose issue (#3054): models like `nvidia/nemotron-3-nano-30b-a3b:free` support `tools` but fail when `tool_choice` is passed. The error message is a 404: "No endpoints found that support the provided 'tool_choice' value."

**Consequences:**
- T0 Invoke results for affected models measure "does this model accept our probe format" rather than "can this model call tools."
- Models are incorrectly graded F and excluded from further testing, creating systematic bias against models with partial tool-calling support.

**Prevention:**
- Check the `/api/v1/models` endpoint capability flags: `supported_parameters` array. Validate both `tools` and `tool_choice` support before probe design.
- Use `tool_choice: auto` as default (not `required`) unless the test specifically requires forced invocation — this is more compatible across backends.
- Distinguish test failure modes: "rejected by OpenRouter routing layer (404)" vs "model responded but did not call tool" in probe result logging.

**Warning signs:**
- 404 errors on T0 probes (should be 0%)
- Models that pass T0 in one run and 404 in another (provider changed)
- Error body contains "tool_choice" in the message

**Phase:** Address before any sweep. The probe runner must distinguish API errors from model failures.

---

### Pitfall 3: Free Model Churn Makes Results Stale Immediately

**What goes wrong:** Free models on OpenRouter have no stability SLA. Models are added, removed, renamed, or version-bumped without notice. The methodology doc already shows this: the models list references models that no longer exist. Results published with a specific model ID may be pointing at a model the reader cannot reproduce.

**Why it happens:** OpenRouter's free tier is subsidized and discretionary. Model providers remove models, free access is withdrawn, version identifiers change (e.g., `deepseek-chat-v3:free` was replaced by `deepseek-chat-v3-0324:free`). The daily limit policy was overhauled in 2025, dropping unverified accounts from 200 to 50 requests/day. These are structural features of the platform, not bugs.

**Consequences:**
- A leaderboard showing "Model X grades A" is misleading if Model X was removed six weeks after the test.
- Consumers who follow the recommendation hit 404s.
- The "reproducibility via request IDs" provenance system cannot compensate for models that are gone.

**Prevention:**
- Every result row in the README table must include a "Tested on" date and an explicit "Last verified available" date.
- Add a `model_still_exists` validation step at the top of every sweep: hit `/api/v1/models`, verify each model ID is in the response before spending quota on it.
- Display a staleness warning on results older than 60 days.
- Keep a `GRAVEYARD.md` or similar section for models that were graded but have since been removed — this is honest reporting and prevents "where did that result go" confusion.

**Warning signs:**
- Project has been dormant (ModelForecast has been dormant since Feb 2026 — this is already in effect)
- Any model ID in results that uses a date-versioned suffix (v3-0324) is especially likely to have rotated

**Phase:** Phase 1 (roster update). Do not run any probe suite against an unvalidated roster.

---

### Pitfall 4: Confidence Intervals Interpreted as Rankings

**What goes wrong:** The leaderboard shows "Model A: 80% [62,95]" and "Model B: 75% [58,91]" and someone reads this as "A is better than B." With n=10, the CIs overlap substantially. The actual conclusion should be "we cannot distinguish A from B at this sample size." Reporting point estimates as rankings without calling out overlap misleads consumers.

**Why it happens:** Wilson intervals are reported (correctly), but the display design leads consumers to rank-order by the point estimate. With n=10 per cell, a 90% success rate has a CI of roughly [57%, 98%] — a 41-point range. Two models with 90% and 70% success rates at n=10 are not distinguishable. The existing output format (`90% [76,97]`) uses integer rounding that narrows the apparent interval and makes it look tighter than it is.

**Consequences:**
- Consumer makes a decision based on "A ranked #1" when A and B are statistically identical.
- Reverse recommendation: Model B might actually be better but looks worse due to sampling luck.
- Damages credibility when two runs produce different orderings.

**Prevention:**
- Add explicit "overlapping intervals = not distinguishable" language to the README methodology section.
- Visual: use a horizontal CI plot rather than a point-estimate table for the top results. The overlap is immediately visible.
- Add a pairwise distinguishability note to results: mark pairs where CIs overlap with "~" (tie) vs pairs where they are clearly separated.
- Consider n=20 as the minimum (not n=10) for models that pass T0. At n=20, a 90% model has CI [69%, 98%] — still wide, but halved width vs n=10.
- Do not sort the table by point estimate when intervals overlap. Sort by lower CI bound, which is the pessimistic guarantee.

**Warning signs:**
- Any two models with CIs that share >50% of their range are being presented as ranked
- Grading rubric uses thresholds (A: T0>=80%) without acknowledging that 80% [52,95] at n=10 contains 52% — which is a B by the rubric's own criteria

**Phase:** Phase 2 (results display design). The grading rubric thresholds need CI-awareness.

---

### Pitfall 5: Multiple Comparisons Without Correction

**What goes wrong:** With 16 models × 5 dimensions = 80 cells, running pairwise comparisons generates up to 3,160 possible model-vs-model tests. At α=0.05, expect ~158 false positive "significant differences" by chance alone. The current methodology has no correction for this.

**Why it happens:** Benchmarks naturally invite "is Model A significantly better than Model B on T2?" questions. Without Holm-Bonferroni or Benjamini-Hochberg correction, the answer "yes, p<0.05" is meaningless at this comparison scale.

**Consequences:**
- Any specific pairwise claim ("Model X is significantly better than Model Y at schema compliance") is likely to be false positive noise.
- The project is not set up to make these claims currently (no pairwise test in the code), so this is a future pitfall as the project matures.

**Prevention:**
- Do not add pairwise significance testing to the codebase without FWER/FDR correction.
- If pairwise claims are made in the README, use Holm-Bonferroni: order p-values, apply stepwise correction. The `scipy.stats` library supports this directly.
- Prefer visual CI overlap communication over formal hypothesis testing for consumer-facing output. Consumers understand "these bars overlap" better than "p=0.12 after Holm correction."
- Be explicit that n=10 is powered for detecting large effects (>30 percentage point differences) not subtle ones.

**Warning signs:**
- Any future PR adds `scipy.stats.ttest_ind` or similar without correction
- README starts making claims like "significantly outperforms" between models with similar point estimates

**Phase:** Phase 3+ (if comparison features are added). Document the constraint in METHODOLOGY.md now.

---

### Pitfall 6: Benchmark Contamination via Probe Reuse

**What goes wrong:** The T0 probe is "Use the search tool to find files containing 'authentication'". This exact prompt, once published in a public repo, becomes training data for future model fine-tuning. Models released after the project goes public may have seen this probe verbatim. Their T0 score measures "did this model memorize the correct response to this specific prompt" not "can this model call tools."

**Why it happens:** ModelForecast's repo will be public (per PROJECT.md). The probes are static and small (5 total). Large training corpus scrapers index GitHub continuously. Models fine-tuned on "function calling datasets" routinely include GitHub code and issues. LiveBench (ICLR 2025 Spotlight) addresses this at massive scale; ModelForecast faces the same problem at small scale.

**Consequences:**
- A model that scores A on T0 may have memorized ModelForecast's probe specifically.
- The "empirical" result is actually a contamination measurement, not a capability measurement.
- Differentiation from BFCL disappears — BFCL has the same problem but with 2,000 test cases, making memorization less likely per individual case.

**Prevention:**
- Keep a set of "canonical probes" (publicly documented) and a set of "canary probes" (withheld, rotated annually). Compare model scores on canonical vs canary. Divergence flags contamination.
- Parameterize probe templates: "find files containing '{keyword}'" where the keyword varies per run. Log which variation was used. Contamination is harder when prompt varies.
- Add a probe rotation mechanism to the roadmap — even if not implemented in v1, the architecture should support it.
- Surface scores as "tested on probe version 1.0" so that future score comparisons account for probe changes.

**Warning signs:**
- A model scores significantly better on probes that appear verbatim in its Hugging Face training dataset description
- Perfect T0 scores (10/10) on a model known to have poor general tool-calling capability

**Phase:** Phase 1 (probe design). Parameterize before going public. Retroactively hard to fix without breaking historical comparability.

---

## Moderate Pitfalls

These cause degraded reliability or maintenance headaches, but not invalid results.

---

### Pitfall 7: Daily Quota Exhaustion Silently Truncates Sweeps

**What goes wrong:** OpenRouter free model limits are 50 requests/day (unverified account) or 1,000/day (account with $10+ credit). A full sweep is 16 models × 5 dimensions × 10 trials = 800 requests. This nearly exhausts the 1,000/day limit and completely blows past the 50/day limit for unverified accounts. If the sweep runner doesn't handle 429 responses gracefully, it fails partway through, producing partial results that are easy to mistake for complete results.

**Prevention:**
- Check account status and daily remaining quota before starting a sweep via `/api/v1/auth/key`.
- Implement explicit 429 handling with exponential backoff (not suppress-error-and-continue).
- Write partial results to disk after each model completes, with a `sweep_complete: false` flag.
- The sweep runner should refuse to publish results if any model has fewer than n trials due to rate limiting.

**Warning signs:** Sweep completing in suspiciously short time, or results table showing some models with n<10.

**Phase:** Phase 1 (sweep runner). Non-negotiable for correctness.

---

### Pitfall 8: The Grade Rubric Hides CI Width

**What goes wrong:** The rubric grades on point estimates ("T0 >= 80% = A"). A model with T0 = 80% [52%, 95%] gets the same grade as a model with T0 = 80% [71%, 88%]. The first model's actual performance could be as low as 52% — a C grade by the same rubric. The grade is not a grade; it is an optimistic guess.

**Prevention:**
- Grade on the lower Wilson CI bound, not the point estimate. "A" means "we are 95% confident this model achieves at least X%." This is conservative but honest.
- Alternatively, require higher minimum n before awarding top grades: grade A requires n=20, grade B requires n=10.
- Document this choice explicitly in METHODOLOGY.md. Either approach is defensible; undocumented optimistic grading is not.

**Phase:** Phase 2 (grading logic). The confidence.py code is correct; the rubric application is the problem.

---

### Pitfall 9: Rate Limit Interacts with Retry Logic to Skew Results

**What goes wrong:** When a request is rate-limited (429) and retried, the retry is served at a different time, possibly from a different provider backend. If retries are not logged as retries, the probe data conflates original requests with retried requests that may have different provider characteristics. This inflates apparent variance.

**Prevention:**
- Log `is_retry: true` on all retried requests, including the retry count and the delay.
- Consider excluding retried requests from the pass/fail calculation, or logging them separately.
- Do not count a probe trial as complete until a non-429 response is received.

**Phase:** Phase 1 (probe runner infrastructure).

---

### Pitfall 10: OpenRouter Model IDs Are Not Stable Aliases

**What goes wrong:** `qwen/qwen3-14b:free` may serve QWen 3 14B today and QWen 3 14B-instruct-v2 next month with no API change. The provenance system captures the model ID, not the underlying model version. Two results for the same model ID may be measuring different models.

**Why it happens:** OpenRouter treats model IDs as routing targets, not immutable content addresses. Providers update the served weights behind the same ID. This has been documented for DeepSeek variants (v3 → v3-0324) and Gemini flash variants.

**Prevention:**
- Capture the `model` field from the API response (not just the request), which may reveal version differences.
- Check if OpenRouter returns any version metadata in the response headers or body.
- When a model is heavily updated, treat the old and new results as separate data series even if the ID is the same.

**Phase:** Phase 2 (provenance tracking). The existing provenance JSON structure should add a `response_model_id` field distinct from `request_model_id`.

---

## Minor Pitfalls

Low-severity issues that affect polish and trust but not fundamental validity.

---

### Pitfall 11: T2 Selection Has an Ambiguous Ground Truth

**What goes wrong:** The T2 probe ("I need to understand what the auth module does" with search, read_file, list_directory) accepts both `search` and `list_directory` as valid. This ambiguity is already acknowledged in METHODOLOGY.md. However, if a model systematically chooses list_directory where the test expects search, it is marked correct even though the behavior may be less useful. Consumers see "80% T2" without knowing how many passed by the less-useful path.

**Prevention:**
- Track which tool was selected per trial, not just pass/fail. Report "80% T2 (60% via search, 20% via list_directory)" or equivalent.
- Consider whether list_directory should be downweighted in the T2 rubric rather than fully accepted.

**Phase:** Phase 2 (probe grading). Low priority but adds analytical value.

---

### Pitfall 12: Error Bar Display Format Rounds Away Information

**What goes wrong:** The existing format `[76,97]` truncates to integers. A CI of [76.3%, 97.2%] is displayed the same as [76.4%, 97.4%] — fine for readability. But the rounding can make the interval appear narrower than it is when both bounds round inward. At n=10, this creates false precision.

**Prevention:**
- This is a display choice, not a correctness issue. Document that integers are rounded.
- Consider one decimal place (`[76.3, 97.2]`) for the methodology doc while keeping integers in the consumer table.

**Phase:** Phase 3 (polish). Low priority.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Model roster update | Stale IDs, churn (P3) | Validate against `/api/v1/models` before any run |
| First sweep run | Daily quota exhaustion (P7) | Implement 429 handling and partial-results persistence first |
| Probe runner build | tool_choice vs tools (P2) | Check capability flags; use `tool_choice: auto` |
| Results display | CI misread as ranking (P4) | CI overlap visualization, not just point-estimate table |
| README table publishing | Provider variance invisible (P1) | Capture `x-openrouter-provider` header per request |
| Grading logic | Rubric on point estimates (P8) | Grade on lower CI bound |
| Going public | Contamination risk (P6) | Parameterize probe keywords before public release |
| Future pairwise claims | Multiple comparisons (P5) | Document correction requirement in METHODOLOGY.md now |

---

## Sources

- OpenRouter Provider Variance announcement: [Provider Variance: Introducing Exacto](https://openrouter.ai/announcements/provider-variance-introducing-exacto)
- OpenRouter free model rate limits: [OpenRouter FAQ](https://openrouter.ai/docs/faq) and [Oreate AI analysis](https://www.oreateai.com/blog/indepth-analysis-of-openrouters-free-policy-adjustments-daily-quota-changes-and-response-strategies/d450d1aa56b67882c0100e68510fac55)
- tool_choice vs tools distinction: [Goose issue #3054](https://github.com/block/goose/issues/3054)
- Zed editor 404 on tool_choice: [zed-industries/zed #36094](https://github.com/zed-industries/zed/issues/36094)
- Provider performance variance (93.3% vs 36.7%): [Simon Willison — Open weight LLMs exhibit inconsistent performance](https://simonwillison.net/2025/Aug/15/inconsistent-performance/)
- Benchmark contamination research: [LiveBench](https://livebench.ai/livebench.pdf), [awesome-data-contamination](https://github.com/lyy1994/awesome-data-contamination)
- Benchmark leaderboard trust issues and cherry-picking: [Goodeye Labs 2025 review](https://www.goodeyelabs.com/insights/llm-evaluation-2025-review), [Collinear — Goodhart's Law in AI leaderboards](https://blog.collinear.ai/p/gaming-the-system-goodharts-law-exemplified-in-ai-leaderboard-controversy)
- CI overlapping interval interpretation: [Cameron Wolfe — Applying Statistics to LLM Evaluations](https://cameronrwolfe.substack.com/p/stats-llm-evals)
- Multiple comparisons correction: [Holm-Bonferroni method](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method)
- Wilson interval at n=10: [Wilson CI literature](https://insightful-data-lab.com/2025/08/20/wilson-score-interval/) — confirmed adequate coverage at n=10, not n<5
- BFCL V4 methodology: [BFCL paper proceedings](https://proceedings.mlr.press/v267/patil25a.html)
- Free model availability list as of March 2026: [CostGoat](https://costgoat.com/pricing/openrouter-free-models)
