# ModelForecast Headlines

> Latest findings from our tool-calling capability benchmarks

## Current Headline

### 2025-12-02: Only 2 of 25 Free Models Can Actually Tool-Call

**Of 25 OpenRouter free models claiming tool support, only 2 achieve 100% reliability.**

We tested every free model on OpenRouter that advertises tool-calling capability. The results are sobering:

| Verdict | Count | % |
|---------|-------|---|
| Production Ready | 2 | 8% |
| Unreliable (20-60%) | 7 | 28% |
| Broken (0%) | 14 | 56% |
| Partial (~66%) | 2 | 8% |

**The winners:**
- `x-ai/grok-4.1-fast:free` - 10/10 (100%)
- `kwaipilot/kat-coder-pro:free` - 10/10 (100%)

**The 3-Trial Trap:** 8 models passed 3/3 quick tests but failed extended 10-trial testing. Small samples give false confidence.

**Completely broken:** All Qwen variants (6 models), all Google variants (3 models), Llama-4-Maverick, Microsoft MAI-DS-R1, and others returned 0% tool-call success despite marketing claims.

[Full analysis](PHASE3_RESULTS.md) | [Raw data](results/phase3_summary.csv)

---

## Archive

Headlines move here after new results are published.

<!-- Format:
### YYYY-MM-DD: Headline
Brief summary of key finding.
[Full article](articles/YYYY-MM-DD-slug.md)
-->
