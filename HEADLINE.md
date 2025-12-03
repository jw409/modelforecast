# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2025-12-03: Only 1 Free Model Survives Full Testing

**Of 25 free models, only KAT Coder Pro passes all 5 probe levels.**

We ran L0-L4 probes (Basic → Schema → Selection → Multi-turn → Adversarial) on every free model. The differentiator was **L3: Multi-turn**.

| Model | L0-L2 | L3 Multi-turn | L4 | Latency | Verdict |
|-------|-------|---------------|----:|--------:|---------|
| kwaipilot/kat-coder-pro:free | 100% | **100%** | 100% | **1.3s** | ✅ Production |
| x-ai/grok-4.1-fast:free | 100% | **0%** ❌ | 100% | 6.8s | ⚠️ Single-shot only |
| google/gemini-2.5-flash-lite:free | 0% | 0% | 0% | N/A | ❌ Broken |

### The Multi-Turn Trap

**Grok passes L0-L2 but fails L3.** After receiving tool results, it returns text instead of continuing tool use. This breaks agentic loops (ReAct patterns, iterative reasoning).

**Practical impact:**
- ✅ **KAT Coder Pro**: Use for agents, loops, multi-step workflows
- ⚠️ **Grok 4.1**: Single-shot only (one tool call, done)
- ❌ **Gemini via OpenRouter**: Completely broken, 0% all levels

### How Do Paid Models Compare?

| Model | L0-L4 | Multi-turn | Latency | Cost/1M tokens |
|-------|-------|------------|--------:|---------------:|
| **claude-sonnet-4-5** | 100% | ✅ 100% | 1.8s | $3/$15 |
| **gpt-4o** | 100% | ✅ 100% | 2.1s | $2.50/$10 |
| **gemini-2.0-flash** (paid) | 100% | ✅ 100% | 0.9s | $0.10/$0.40 |
| kwaipilot/kat-coder-pro:free | 100% | ✅ 100% | 1.3s | **FREE** |
| x-ai/grok-4.1-fast:free | 60% | ❌ 0% | 6.8s | FREE |

**Key insight**: KAT Coder Pro matches paid model reliability at zero cost. The free-vs-paid gap is smaller than the working-vs-broken gap among free models.

### Speed Matters

```
Gemini 2.0 (paid): 0.9s  ███
KAT Coder Pro:     1.3s  ████
Claude Sonnet:     1.8s  ██████
GPT-4o:            2.1s  ███████
Grok 4.1 Fast:     6.8s  ██████████████████████████
```

[Full analysis](PHASE3_RESULTS.md) | [Raw data](results/phase3_summary.csv)

---

## Archive

Headlines move here after new results are published.

<!-- Format:
### YYYY-MM-DD: Headline
Brief summary of key finding.
[Full article](articles/YYYY-MM-DD-slug.md)
-->
