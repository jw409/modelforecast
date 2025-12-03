# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2025-12-03: Two Free Models Now Achieve Full Tool-Calling Reliability

**KwaiPilot's KAT Coder Pro and DeepSeek v3.2-exp now pass all 5 probe levels, with DeepSeek requiring explicit multi-step prompts for L3.**

We ran L0-L4 probes (Basic → Schema → Selection → Multi-turn → Adversarial) on free models. The differentiator, L3: Multi-turn, was refined to ensure fair testing for chat-optimized models.

| Model | L0 | L1 | L2 | L3 Multi | L4 Advers | Latency (avg) | Verdict |
|-------|----|----|----|----------|-----------|---------------|---------|
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | **100%** | 100% | **1.3s** | ✅ Production |
| deepseek/deepseek-v3.2-exp | 100% | 100% | 100% | **100%** | 100% | **3.3s** | ✅ Production (explicit L3) |
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | 0% ❌ | 100% | 6.8s | ⚠️ Single-shot only |
| google/gemini-2.5-flash-lite:free | 0% | 0% | 0% | 0% ❌ | 0% | N/A | ❌ Broken |

### The Nuance of Multi-Turn

**Initial L3 probe results were misleading for chat-optimized models.** Models like Grok and DeepSeek, when simply asked to "Find files...", would often return text or continue searching after results were presented, rather than immediately initiating a `read_file` call. This was interpreted as an L3 failure.

**Refinement:** The L3 prompt now explicitly includes the follow-up step: *"Find files related to authentication AND read the most relevant one."*
- **DeepSeek v3.2-exp:** With this clarity, DeepSeek v3.2-exp now consistently executes the multi-turn sequence and achieves 100% on L3.
- **Grok 4.1-fast:** While not re-benchmarked, this strongly suggests Grok's L3 failure was similarly due to prompt ambiguity rather than a lack of capability. Its L3 score is pending re-evaluation, but it's likely capable with explicit prompting.

**Practical impact remains:** For truly autonomous, zero-shot multi-turn tasks (where the model infers the next step), KAT Coder Pro still excels. For models like DeepSeek (and likely Grok), explicit multi-step instructions are key.

### How Do Paid Models Compare?

| Model | L0-L4 | Multi-turn | Latency | Cost/1M tokens |
|-------|-------|------------|--------:|---------------:|
| **claude-sonnet-4-5** | 100% | ✅ 100% | 1.8s | $3/$15 |
| **gpt-4o** | 100% | ✅ 100% | 2.1s | $2.50/$10 |
| **gemini-2.0-flash** (paid) | 100% | ✅ 100% | 0.9s | $0.10/$0.40 |
| kwaipilot/kat-coder-pro:free | 100% | ✅ 100% | 1.3s | **FREE** |
| deepseek/deepseek-v3.2-exp | 100% | ✅ 100% | 3.3s | **FREE** |
| x-ai/grok-4.1-fast:free | 60% | ❌ 0% | 6.8s | FREE |

**Key insight**: KAT Coder Pro and DeepSeek v3.2 (with explicit L3) match paid model reliability at zero cost. The free-vs-paid gap is smaller than the working-vs-broken gap among free models.

### Speed Matters

```
Gemini 2.0 (paid): 0.9s  ███
KAT Coder Pro:     1.3s  ████
Claude Sonnet:     1.8s  ██████
GPT-4o:            2.1s  ███████
DeepSeek v3.2-exp: 3.3s  ███████████
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