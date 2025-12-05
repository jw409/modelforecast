# ModelForecast Phase 3 - Initial Probe Results

**Date**: 2025-12-02
**Test**: Level 0 - Basic Tool Calling
**Question**: Can the model produce a tool_call at all?

## Summary

Out of 30 free OpenRouter models, 18 claim tool-calling support. We tested all of them.

### Key Findings

1. **Only 2 models are production-ready** (100% success over 10 trials):
   - `x-ai/grok-4.1-fast:free` - 10/10 (100%), CI=[72%, 100%]
   - `kwaipilot/kat-coder-pro:free` - 10/10 (100%), CI=[72%, 100%]

2. **Most models that passed initial triage failed extended testing**:
   - 8 models passed 3/3 trials but failed to maintain reliability over 10 trials
   - This demonstrates the importance of sufficient trial counts

3. **13 models completely failed** basic tool calling (0% success rate)

4. **1 model hangs** on tool calling requests:
   - `mistralai/mistral-7b-instruct:free` - Process hung, had to be killed

## Detailed Results

### Production-Ready Models (100% over 10 trials)

| Model | Success Rate | Confidence Interval | Verdict |
|-------|--------------|---------------------|---------|
| x-ai/grok-4.1-fast:free | 10/10 (100%) | [72%, 100%] | ✓ PRODUCTION READY |
| kwaipilot/kat-coder-pro:free | 10/10 (100%) | [72%, 100%] | ✓ PRODUCTION READY |

### Unreliable Models (Passed triage, failed extended testing)

| Model | Initial (3) | Extended (10) | Confidence Interval |
|-------|-------------|---------------|---------------------|
| nvidia/nemotron-nano-9b-v2:free | 3/3 (100%) | 6/10 (60%) | [31%, 83%] |
| alibaba/tongyi-deepresearch-30b-a3b:free | 3/3 (100%) | 5/10 (50%) | [24%, 76%] |
| arcee-ai/trinity-mini:free | 3/3 (100%) | 3/10 (30%) | [11%, 60%] |
| z-ai/glm-4.5-air:free | 3/3 (100%) | 2/10 (20%) | [6%, 51%] |
| tngtech/tng-r1t-chimera:free | 3/3 (100%) | 2/10 (20%) | [6%, 51%] |
| openai/gpt-oss-20b:free | 3/3 (100%) | 2/10 (20%) | [6%, 51%] |
| meituan/longcat-flash-chat:free | 3/3 (100%) | 2/10 (20%) | [6%, 51%] |
| meta-llama/llama-3.3-70b-instruct:free | 3/3 (100%) | 0/10 (0%) | [0%, 28%] |

**Note**: Llama 3.3 70B is particularly surprising - it passed initial triage perfectly but then failed all 10 extended trials. This suggests possible rate limiting or endpoint instability.

### Partially Working Models (Some success in triage)

| Model | Success Rate | Notes |
|-------|--------------|-------|
| nvidia/nemotron-nano-12b-v2-vl:free | 2/3 (66%) | Vision model, possibly wrong endpoint |
| amazon/nova-2-lite-v1:free | 2/3 (66%) | Inconsistent behavior |

### Completely Broken Models (0% success)

These models claim tool support but failed all trials:

- google/gemini-2.0-flash-exp:free
- google/gemini-2.5-flash-lite-preview-09-2025:free
- google/gemma-3-27b-it:free
- mistralai/mistral-small-3.1-24b-instruct:free
- qwen/qwen3-235b-a22b:free
- qwen/qwen3-32b:free
- qwen/qwen3-30b-a3b:free
- qwen/qwen3-14b:free
- qwen/qwen3-4b:free
- qwen/qwen3-coder:free
- meta-llama/llama-4-maverick:free
- microsoft/mai-ds-r1:free
- nousresearch/deephermes-3-llama-3-8b-preview:free

### Models That Hang

- mistralai/mistral-7b-instruct:free - Process hangs indefinitely on tool call request

## Methodology

### Test Design

- **Prompt**: "Use the search tool to find files containing 'authentication'"
- **Tool**: Simple `search(query: string)` function
- **Pass Criteria**: Response contains `tool_calls` array with at least one entry
- **Fail Criteria**: Text response, empty tool_calls, malformed JSON, or exception

### Testing Approach

1. **Quick Triage** (3 trials): Fast screening of all models claiming tool support
2. **Extended Testing** (10 trials): Full reliability testing on models that passed triage
3. **Statistical Analysis**: Wilson score confidence intervals (95% CI) for all results

### Why 10 Trials Matters

The dramatic difference between 3-trial and 10-trial results demonstrates that:

- **3 trials can give false confidence** - 8 models went from 100% to <60%
- **10+ trials are needed** to detect reliability issues
- **Production systems need evidence-based selection**, not marketing claims

## Implications

### For Users

If you're building with free OpenRouter models that need tool calling:

1. **Use Grok 4.1 Fast or KAT Coder Pro** - Only proven reliable options
2. **Don't trust marketing claims** - "supports tool calling" ≠ "reliably supports tool calling"
3. **Test your specific use case** - Even passing models may fail under different conditions

### For Model Providers

- **Advertised features should work** - 13/18 models claiming tool support don't work at all
- **Reliability matters** - A model that works 20% of the time is not production-ready
- **Transparency helps users** - Clear capability statements prevent wasted time

## Next Steps

### Phase 4: Higher-Level Probes

Now that we know which models can call tools at all, test:

- **Level 1**: Schema compliance (parameter types, required fields)
- **Level 2**: Tool selection (choosing correct tool from multiple options)
- **Level 3**: Multi-turn (following up appropriately after tool results)
- **Level 4**: Adversarial (not hallucinating tools when none fit)

### Phase 5: Community Verification

- Open source results for reproduction
- Crowdsource results from multiple API keys and regions
- Build statistical confidence through independent verification

## Reproduction

```bash
# Clone repository
git clone https://github.com/jw409/modelforecast
cd modelforecast

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Set API key
export OPENROUTER_API_KEY=your_key_here

# Run probes (results saved to ./results/)
uv run python -m modelforecast --level 0 --trials 10
```

## Raw Data

All individual trial results are saved in JSON format with:
- Full request/response hashes for verification
- OpenRouter request IDs for debugging
- Latency measurements
- Provenance information (Python version, SDK version, OS, contributor)

See `results/` directory for complete data.

---

*ModelForecast is maintained by [@jw409](https://github.com/jw409). Not affiliated with OpenRouter.*
