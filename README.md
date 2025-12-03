# ModelForecast

**Check the forecast before you deploy.**

Do free LLM models actually support tool calling? Marketing says yes. We test it.

## Latest Results (Updated Weekly)

| Model | Tool Calling | Schema | Selection | Grade |
|-------|-------------|--------|-----------|-------|
| *Coming soon* | - | - | - | - |

*First probe run in progress. Results will appear here after initial benchmarks.*

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
cat results/latest/summary.md
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

- **Quantitative**: Not "works" vs "doesn't work" but "87% +/- 8% (95% CI, n=15)"
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
