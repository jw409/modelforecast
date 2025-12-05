# ModelForecast

**Check the forecast before you deploy.**

Do free LLM models actually support tool calling? Marketing says yes. We test it.

---

## Today's Forecast (2025-12-04)

**Only 7 of 32 tested models are production-ready for tool calling.**

![Reliability vs Latency](charts/reliability_vs_latency.png)

*Upper-left quadrant = fast AND reliable. That's where you want to be.*

### The Headline Numbers

| Category | Count |
|----------|------:|
| Production Ready (≥90%) | 7 |
| Unreliable (50-89%) | 4 |
| Broken (<50%) | 21 |
| **Total Tested** | **32** |

---

## Understanding the Dimensions

We measure three orthogonal capabilities:

### TOOL CALLING (T): Can it use tools correctly?
- **T0 (Invoke)**: Can it call a tool at all?
- **T1 (Schema)**: Does it respect parameter types (string vs int, required vs optional)?
- **T2 (Selection)**: Can it choose the right tool from a set?

### RESTRAINT (R): Does it know when NOT to use tools?
- **R0 (Abstain)**: Will it refuse to call tools when none are appropriate AND still provide a helpful answer?
- Not just silence - the model must explain why it's not using tools while still being useful.

### AGENCY (A): Can it orchestrate multi-step workflows?
- **A1 (Linear)**: After receiving tool results, can it continue using tools to complete a task?
- This tests whether the model can chain tool calls together, not just invoke once and stop.

### Safe but Stupid vs Smart but Hallucinatory

These dimensions create tradeoffs:

- **High R, Low A** = Safe but can't do complex tasks (won't hallucinate, but limited capability)
- **High A, Low R** = Capable but might hallucinate tool calls (powerful but dangerous in production)

The sweet spot: High scores across all three dimensions.

---

## The Agency Gap

Most models pass basic tests. Then they hit A1 (agency) and fall off a cliff.

![Multi-Dimension Comparison](charts/multi_level_comparison.png)

**A1 is the differentiator.** After receiving tool results, can the model continue using tools appropriately?

- **KAT Coder Pro**: 100% - continues tool use correctly
- **DeepSeek V3.2-exp**: 60% - sometimes stops or picks wrong tool
- **Grok 4.1**: 0% - returns text instead of calling next tool
- **Gemini (free)**: Never gets there - fails T0 Invoke

---

## Full Results

### Production Ready (≥90% T0 Invoke)

| Model | T0 Pass | Latency | Tokens | AES | Grade |
|-------|--------:|--------:|-------:|----:|-------|
| **gemini-2.5-flash-preview** | 100% | **0.5s** | 7 | **0.95** | **A+** |
| claude-haiku-4.5 | 100% | 1.1s | 26 | 0.81 | A |
| kwaipilot/kat-coder-pro:free | 100% | 1.3s | 19 | 0.80 | A |
| claude-sonnet-4.5 | 100% | 2.0s | 26 | 0.68 | B+ |
| deepseek-v3.2-exp | 100% | 2.6s | 8 | 0.64 | B+ |
| claude-opus-4.5 | 100% | 3.4s | 26 | 0.56 | B |
| x-ai/grok-4.1-fast:free | 100% | 6.8s | 8 | 0.38 | C+ |

*AES = Agent Efficiency Score. Higher is better. See Efficiency Metrics section.*

![Efficiency Comparison](charts/efficiency_comparison.png)

### The Full Picture

![Success Rates with Confidence Intervals](charts/success_rates_with_ci.png)

*Error bars show 95% Wilson confidence intervals. Wide bars = few trials or high variance.*

### Unreliable (50-89%)

| Model | T0 Invoke | CI (95%) |
|-------|----------:|----------|
| nvidia/nemotron-nano-12b-v2-vl:free | 67% | [21%, 94%] |
| amazon/nova-2-lite-v1:free | 67% | [21%, 94%] |
| nvidia/nemotron-nano-9b-v2:free | 60% | [31%, 83%] |
| alibaba/tongyi-deepresearch-30b-a3b:free | 50% | [24%, 76%] |

### Broken (0-30%)

14 models claim tool support but failed most or all trials:

- **All Qwen variants** (qwen3-32b, qwen3-30b-a3b, qwen3-14b, qwen3-4b, qwen3-coder, qwen3-235b-a22b)
- **All Google free variants** (gemini-2.0-flash-exp, gemini-2.5-flash-lite, gemma-3-27b-it)
- meta-llama/llama-4-maverick, llama-3.3-70b-instruct
- microsoft/mai-ds-r1, mistralai/mistral-small-3.1-24b-instruct
- nousresearch/deephermes-3-llama-3-8b-preview

---

## Free vs Paid: The Real Comparison

| Model | T0 Pass | Latency | AES | Cost/1M | Grade |
|-------|--------:|--------:|----:|--------:|-------|
| **gemini-2.5-flash-preview** | 100% | **0.5s** | **0.95** | $0.30/$1.25 | **A+** |
| claude-haiku-4.5 | 100% | 1.1s | 0.81 | $0.80/$4 | A |
| **kwaipilot/kat-coder-pro:free** | 100% | 1.3s | 0.80 | **$0** | A |
| claude-sonnet-4.5 | 100% | 2.0s | 0.68 | $3/$15 | B+ |
| deepseek-v3.2-exp | 100% | 2.6s | 0.64 | $0.21/$0.32 | B+ |
| claude-opus-4.5 | 100% | 3.4s | 0.56 | $15/$75 | B |
| x-ai/grok-4.1-fast:free | 100% | 6.8s | 0.38 | $0 | C+ |

**Bottom line**: Gemini 2.5 Flash dominates on efficiency. KAT Coder Pro is the best free option - fast, efficient, and $0.

---

## Don't Believe Me? Go Outside.

Everything here is reproducible. Run it yourself:

```bash
# Clone and install
git clone https://github.com/jw409/modelforecast
cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Set your OpenRouter API key
export OPENROUTER_API_KEY=your_key_here

# Run probes (takes ~15 minutes)
uv run python -m modelforecast

# Regenerate charts
uv run python scripts/generate_charts.py

# View raw results
cat results/phase3_summary.csv
```

---

## The 3-Trial Trap

**8 models passed 3/3 quick tests but failed extended testing.**

| Model | 3 trials | 10 trials | Reality |
|-------|----------|-----------|---------|
| meta-llama/llama-3.3-70b-instruct | 100% | 0% | Broken |
| nvidia/nemotron-nano-9b-v2 | 100% | 60% | Unreliable |
| alibaba/tongyi-deepresearch-30b-a3b | 100% | 50% | Unreliable |

Small sample sizes give false confidence. That's why we use Wilson score intervals.

---

## What We Test

| Dimension | Code | Test | Question |
|-----------|------|------|----------|
| TOOL CALLING | T0 | Invoke | Can it call a tool at all? |
| TOOL CALLING | T1 | Schema | Does it respect parameter types? |
| TOOL CALLING | T2 | Selection | Can it choose the right tool from a set? |
| AGENCY | A1 | Linear | After getting results, can it continue using tools? |
| RESTRAINT | R0 | Abstain | Will it NOT call tools when none are appropriate? |

---

## Efficiency Metrics

When multiple models hit 100% pass rate, how do you choose? We measure **efficiency** - how well a model uses tokens and time.

### Agent Efficiency Score (AES)

AES = 0.4 × Token Efficiency + 0.4 × Latency Score + 0.2 × Strictness

| Component | What it measures | Best score |
|-----------|------------------|:----------:|
| **Token Efficiency** | Fewer completion tokens = less cost | 1.0 (optimal tokens) |
| **Latency Score** | Faster response = better UX | 1.0 (< 500ms) |
| **Strictness** | Uses `tool_calls` finish reason | 1.0 (always strict) |

**Why this matters**: Two models both pass T0 with 100%. But one uses 7 tokens in 500ms, the other uses 50 tokens in 6 seconds. For real applications, that's a massive difference in cost and user experience.

---

## Methodology

We use **Wilson score intervals** for confidence - the gold standard for binomial proportions with small samples.

- Each probe runs multiple trials (default: 10 for extended, 3 for triage)
- Results include 95% confidence intervals
- Grading based on lower bound of CI (conservative)

Full methodology: [METHODOLOGY.md](METHODOLOGY.md)

---

## Contributing

We welcome community contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

**To submit results:**
1. Fork this repo
2. Run `uv run python -m modelforecast`
3. Commit your `results/` folder
4. Open a PR

Automated verification checks your results. If they match within tolerance, they'll be merged.

---

## License

MIT

---

*ModelForecast is maintained by [@jw409](https://github.com/jw409) and contributors.*
*Not affiliated with OpenRouter.*
