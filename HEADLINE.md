# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2025-12-04: 94% of Free Models Fail Basic Tool Calling

**Only 2 of 37 free models achieve 100% T0 (basic tool invocation). The free model graveyard is bigger than we thought.**

We ran T0 Invoke probes on all 37 free models available on OpenRouter. The results are devastating for the "free AI agent" dream.

| Category | Count | % |
|----------|------:|--:|
| **Perfect T0** (100%) | 2 | 5% |
| Partial (50-99%) | 5 | 14% |
| Broken (<50%) | 30 | 81% |

### The Only Two Winners

| Model | T0 | T1 | T2 | A1 | R0 | Verdict |
|-------|----|----|----|----|----|---------|
| **kwaipilot/kat-coder-pro:free** | 100% | 100% | 100% | 100% | 100% | ✅ **Production Ready** |
| **x-ai/grok-4.1-fast:free** | 100% | 100% | 100% | 0% | 100% | ⚠️ Single-shot only |

**KAT Coder Pro is the only free model with perfect scores across all dimensions.**

### The Graveyard (30 Models)

These models claim tool support but can't reliably call tools:

| Family | Count | All Failed T0 |
|--------|------:|:-------------:|
| Qwen | 6 | 0%, 0%, 0%, 0%, 0%, 0% |
| Google | 6 | 0%, 0%, 0%, 0%, 0%, 0% |
| Meta/LLaMA | 3 | 0%, 0%, 0% |
| DeepSeek/TNG | 3 | 0%, 0%, 20% |
| NousResearch | 2 | 0%, 0% |
| Others | 10 | 0-30% |

**Notable failures**: All Qwen models (0%), all Google free models (0%), all Meta/LLaMA models (0%), DeepSeek chat (0%), Mistral (0%)

### Why Are Most Free Models Broken?

Three patterns explain the 94% failure rate:

1. **Tool calling not enabled**: Many free tiers simply disable tool calling infrastructure
2. **Output truncation**: Free tiers aggressively truncate, breaking JSON tool schemas
3. **Different inference parameters**: Temperature/sampling changes break structured output

### The Practical Takeaway

```
Building with free models?
├── Need tool calling? → KAT Coder Pro (only reliable free option)
├── Need speed? → KAT Coder Pro (1.3s avg)
├── Have budget? → Claude Haiku ($0.80/1M), Gemini Flash ($0.30/1M)
└── Need agency (multi-turn)? → KAT Coder Pro only (Grok free fails A1)
```

[Full results](README.md#full-results) | [Raw data](results/)

---

## Archive

Headlines move here after new results are published.

<!-- Format:
### YYYY-MM-DD: Headline
Brief summary of key finding.
[Full article](articles/YYYY-MM-DD-slug.md)
-->