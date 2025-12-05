# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2025-12-04: The Free Tool-Calling Landscape is More Nuanced Than We Thought

**12 of 29 free models don't support tools AT ALL (API returns 404). Of the 17 that do, only 4 reliably work.**

We tested all 29 free models on OpenRouter. The key finding: **many "0%" results aren't failures - they're impossible**.

| Category | Count | What It Means |
|----------|------:|---------------|
| **Tools NOT supported** | 12 | API returns 404 - can't even try |
| **Tools supported** | 17 | Can theoretically call tools |
| **Reliable** (100% T0) | 4 | Production-ready |
| **Partial** (50-99%) | 4 | May work with retries |
| **Broken** (<50%) | 9 | Supports tools but fails |

### The Working Models

| Model | T0 | Full Suite | Notes |
|-------|:--:|:----------:|-------|
| **kwaipilot/kat-coder-pro:free** | 100% | 100% all | **Only perfect free model** |
| **x-ai/grok-4.1-fast:free** | 100% | A1 fails | Single-shot only |
| **nvidia/nemotron-nano-9b-v2:free** | 100%* | - | Improved on retest |
| **nvidia/nemotron-nano-12b-v2-vl:free** | 67% | - | High variance |

*Variance is real - always test your specific use case*

### Can't Support Tools (Not Failures)

These 12 models return `404: No endpoints found that support tool use`:

```
google/gemma-3-*           (5 models)  - Gemma doesn't support tools on OpenRouter
meta-llama/llama-3.2-3b    (1 model)   - Small LLaMA variant
moonshotai/kimi-k2         (1 model)   - Provider limitation
nousresearch/hermes-3      (1 model)   - Provider limitation
allenai/olmo-3-32b-think   (1 model)   - Provider limitation
cognitivecomputations/*    (1 model)   - Provider limitation
tngtech/deepseek-r1t*      (2 models)  - Provider limitation
```

### Why This Matters

1. **Don't blame the model**: Gemma isn't "bad at tools" - it's not offered with tools on OpenRouter
2. **Provider != Model**: Same model can have different capabilities on different providers
3. **Check `supported_parameters`**: OpenRouter API tells you what's actually supported

### The Practical Takeaway

```
Building with free models?
├── Check API first: Does it support tools?
│   └── GET /api/v1/models → supported_parameters includes "tools"
├── Need reliability: KAT Coder Pro (only 100% all-dimension free model)
├── Need speed: Nvidia Nemotron (fast, mostly reliable)
└── Have budget: Claude Haiku ($0.80/1M) beats all free options
```

[Full results](README.md) | [Raw data](results/) | [API check script](scripts/check_tool_support.py)

---

## Archive

Headlines move here after new results are published.

<!-- Format:
### YYYY-MM-DD: Headline
Brief summary of key finding.
[Full article](articles/YYYY-MM-DD-slug.md)
-->