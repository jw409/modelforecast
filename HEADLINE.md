# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2026-03-19: Half of Free "Tool-Capable" Models Can't Actually Call Tools

**8 of 16 free models that advertise tool support score 0% on the basic invocation test. One model scores 100% across all dimensions.**

We swept all 16 free models on OpenRouter that claim tool support. The results split cleanly:

| Category | Count | Models |
|----------|------:|--------|
| **Perfect (100% all dimensions)** | 1 | nemotron-3-nano-30b |
| **Grade A** | 5 | + nemotron-3-super-120b, gpt-oss-120b, step-3.5-flash, glm-4.5-air |
| **Partial (some dimensions work)** | 4 | trinity-large-preview, trinity-mini, nemotron-nano-12b-vl, nemotron-nano-9b-v2 |
| **Broken (0% T0)** | 7 | llama-3.3-70b, minimax-m2.5, mistral-small-3.1, gpt-oss-20b, qwen3-4b, qwen3-coder, qwen3-next-80b |

### The Winner

**nvidia/nemotron-3-nano-30b-a3b:free** — 100% across T0 (invoke), T1 (schema), T2 (selection), A1 (multi-turn), R0 (restraint). The only free model with a perfect score.

### The Surprises

- **gpt-oss-120b (Grade A!)**: Initially scored 0% due to OpenRouter privacy settings blocking the endpoint. After fixing settings: 80% T0, 100% T1, 90% T2, 90% A1, 100% R0. **Your OpenRouter privacy settings affect which free models are reachable.**
- **qwen3-coder (0%)**: Despite being a coding model, zero tool calls
- **llama-3.3-70b (0%)**: Meta's flagship free model — text responses only
- **trinity-large-preview**: Weak at T0 (30%) but **100% on multi-turn agency (A1)** — it struggles to start but excels at chaining

### The Practical Takeaway

```
Building with free models on OpenRouter?
├── Need reliable tool calling: nvidia/nemotron-3-nano-30b-a3b (100% all)
├── Need the biggest model: openai/gpt-oss-120b (Grade A, 120B params)
├── Need fast + cheap: stepfun/step-3.5-flash (Grade A)
├── Check your settings: OpenRouter privacy config blocks some free endpoints
└── Avoid: Qwen, Meta, Mistral in free tier (all 0% or rate-limited)
```

[Full results](results/RESULTS.md) | [Raw data](results/sweep_20260318/) | [Methodology](docs/METHODOLOGY.md)

---

## Archive

### 2025-12-04: The Free Tool-Calling Landscape is More Nuanced Than We Thought

12 of 29 free models didn't support tools at all (API 404). Of the 17 that did, only 4 reliably worked. KAT Coder Pro was the only perfect free model. [Full results from Dec 2025](results/)
