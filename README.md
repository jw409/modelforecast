# ModelForecast

**Check the forecast before you deploy.**

Do free LLM models actually support tool calling? Marketing says yes. We test it.

## Latest Results (2025-12-02)

**Only 2 of 25 free models achieve 100% tool-calling reliability.**

### Full Probe Results (L0-L4)

| Model | L0 | L1 | L2 | L3 Multi-turn | L4 Adversarial | Latency | Grade |
|-------|----|----|----|--------------:|---------------:|--------:|-------|
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | **100%** | 100% | **1.3s** | **A+** |
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | **0%** ❌ | 100% | 6.8s | **B** |
| google/gemini-2.5-flash-lite:free | 0% | 0% | 0% | 0% | 0% | N/A | **F** |

### The Differentiator: Multi-Turn (L3)

**Grok fails multi-turn conversations by default.** After receiving tool results, it returns text instead of continuing to use tools.

**But it's fixable!** Three workarounds tested:

| Fix | Works | Complexity |
|-----|-------|------------|
| `tool_choice="required"` | ✅ | API param only |
| System prompt: "MUST use another tool after results" | ✅ | Prompt change |
| User prompt: "This requires TWO tool calls" | ✅ | Prompt change |

```python
# Fix Grok multi-turn with one line:
response = client.chat.completions.create(
    model="x-ai/grok-4.1-fast:free",
    messages=messages,
    tools=tools,
    tool_choice="required",  # <-- This fixes L3
)
```

| Use Case | Recommended Model |
|----------|-------------------|
| Agentic workflows (ReAct, loops) | **KAT Coder Pro** (works by default) |
| Agentic with Grok | Grok + `tool_choice="required"` |
| Single-shot tool calls | Either (KAT faster) |
| Real-time applications | **KAT Coder Pro** (5x faster) |
| Gemini via OpenRouter | **Broken - don't use** |

### Unreliable (20-60% success)

| Model | L0 Basic | CI (95%) | Grade |
|-------|----------|----------|-------|
| nvidia/nemotron-nano-9b-v2:free | 60% | [31%, 83%] | D |
| alibaba/tongyi-deepresearch-30b-a3b:free | 50% | [24%, 76%] | D |
| arcee-ai/trinity-mini:free | 30% | [11%, 60%] | F |
| z-ai/glm-4.5-air:free | 20% | [6%, 51%] | F |
| tngtech/tng-r1t-chimera:free | 20% | [6%, 51%] | F |
| openai/gpt-oss-20b:free | 20% | [6%, 51%] | F |
| meituan/longcat-flash-chat:free | 20% | [6%, 51%] | F |

### Broken (0% success)

14 models claim tool support but failed all trials:
- All Qwen variants (qwen3-32b, qwen3-30b-a3b, qwen3-14b, qwen3-4b, qwen3-coder, qwen3-235b-a22b)
- All Google variants (gemini-2.0-flash-exp, gemini-2.5-flash-lite, gemma-3-27b-it)
- meta-llama/llama-4-maverick, llama-3.3-70b-instruct
- microsoft/mai-ds-r1, mistralai/mistral-small-3.1-24b-instruct
- nousresearch/deephermes-3-llama-3-8b-preview

*Full results: [PHASE3_RESULTS.md](PHASE3_RESULTS.md) | [Raw CSV](results/phase3_summary.csv)*

## Premium Models: The Baseline

For comparison, premium models show what "actually works" looks like:

| Model | L0 Basic | Notes |
|-------|----------|-------|
| claude-sonnet-4-5-20250929 | 100% | Paid tier, reliable |
| x-ai/grok-4.1-fast:free | 100% | **Free tier winner** |

**Key insight**: Free-tier Grok 4.1 matches premium Claude Sonnet 4.5 at L0. The gap between "broken free" and "working free" is larger than "working free" vs "premium paid."

## The 3-Trial Trap

**8 models passed 3/3 quick tests but failed extended testing.**

| Model | 3 trials | 10 trials | Reality |
|-------|----------|-----------|---------|
| meta-llama/llama-3.3-70b-instruct | 100% | 0% | Broken |
| nvidia/nemotron-nano-9b-v2 | 100% | 60% | Unreliable |
| alibaba/tongyi-deepresearch-30b-a3b | 100% | 50% | Unreliable |

Small sample sizes give false confidence. Always test with sufficient trials.

## Quick Start

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

# View results
cat results/phase3_summary.csv
```

## What We Test

| Level | Test | Question Answered |
|-------|------|-------------------|
| 0 | Basic | Can it call a tool at all? |
| 1 | Schema | Does it respect parameter types? |
| 2 | Selection | Can it choose the right tool? |
| 3 | Multi-turn | Can it follow up appropriately? |
| 4 | Adversarial | Does it hallucinate when no tool fits? |

## The Problem

Free-tier LLM models on OpenRouter claim tool-calling capabilities they may not reliably deliver:

1. **Fail silently** - Output text instead of tool_call JSON
2. **Hallucinate tool names** - Call tools that don't exist
3. **Break on parameters** - Ignore required fields, wrong types
4. **Degrade under forcing** - Work with tool_choice="auto" but break with "required"

Every failed experiment on a model that can't tool-call is wasted compute, wasted electricity, wasted time.

## What We Do

**We test what model cards claim. We publish what we find. With error bars.**

- **Quantitative**: Not "works" vs "doesn't work" but "87% +/- 8% (95% CI, n=10)"
- **Reproducible**: Single command, your API key, same results
- **Opinionated**: We name names. If a model claims tools but fails Level 0, we say so.
- **Community-verified**: Crowdsourced results with cryptographic provenance

## Contributing

We welcome community contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

**To submit results:**
1. Fork this repo
2. Run `uv run python -m modelforecast`
3. Commit your `results/` folder
4. Open a PR

Automated verification will check your results. If they match our reproduction within tolerance, they'll be merged.

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for:
- Detailed probe descriptions
- Statistical approach (Wilson score intervals)
- Grading rubric
- Verification protocol

## License

MIT

---

*ModelForecast is maintained by [@jw409](https://github.com/jw409) and contributors.*
*Not affiliated with OpenRouter.*
