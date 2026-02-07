# ModelForecast Results

Tool-calling capability benchmarks across LLM models via OpenRouter.

**Updated: 2026-02-06** | Levels: T0 Invoke, T1 Schema, T2 Selection, A1 Linear (multi-turn), R0 Abstain

## Perfect Score (A+)

| Model | T0 | T1 | T2 | A1 | R0 | Cost |
|-------|:--:|:--:|:--:|:--:|:--:|:----:|
| anthropic/claude-haiku-4.5 | 100% | 100% | 100% | 100% | 100% | $0.80/M |
| anthropic/claude-opus-4.5 | 100% | 100% | 100% | 100% | 100% | $15/M |
| anthropic/claude-sonnet-4.5 | 100% | 100% | 100% | 100% | 100% | $3/M |
| deepseek/deepseek-v3.2-exp | 100% | 100% | 100% | 100% | 100% | $1.10/M |
| google/gemini-2.5-flash-preview | 100% | 100% | 100% | 100% | 100% | $0.15/M |
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | 100% | 100% | FREE* |
| **nvidia/nemotron-3-nano-30b-a3b:free** | 100% | 100% | 100% | 100% | 100% | **FREE** |
| openai/gpt-5.1-codex | 100% | 100% | 100% | 100% | 100% | $2.50/M |
| x-ai/grok-4.1-fast | 100% | 100% | 100% | 100% | 100% | $5/M |
| x-ai/grok-code-fast-1 | 100% | 100% | 100% | 100% | 100% | $5/M |

\* KAT Coder Pro free tier no longer available on OpenRouter as of Feb 2026

## Production Ready (A/B)

| Model | T0 | T1 | T2 | A1 | R0 | Grade |
|-------|:--:|:--:|:--:|:--:|:--:|:-----:|
| minimax/minimax-m2 | 100% | 80% | 100% | 100% | 100% | **A** |
| openai/gpt-5.1 | 100% | 100% | 100% | 80% | 100% | **A** |
| z-ai/glm-4.5-air:free | 100% | 60% | 60% | 60% | 80% | **B** |
| upstage/solar-pro-3:free | 100% | 40% | 80% | 80% | 100% | **B-** |

## Partial Success (C)

| Model | T0 | T1 | T2 | A1 | R0 | Notes |
|-------|:--:|:--:|:--:|:--:|:--:|-------|
| google/gemini-3-pro-preview | 100% | 100% | 100% | **0%** | 100% | Can't chain tools |
| openai/gpt-5-mini | 100% | 100% | 80% | **20%** | 100% | Budget = weak A1 |
| openai/gpt-5.1-codex-mini | 100% | 100% | 100% | **20%** | 100% | Budget = weak A1 |
| stepfun/step-3.5-flash:free | 100% | 100% | 60% | 20% | 60% | Mid-tier free |
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | **0%** | 100% | Free tier breaks A1 |
| arcee-ai/trinity-large-preview:free | 100% | 80% | 0% | 20% | 100% | T2 broken |
| nvidia/nemotron-nano-12b-v2-vl:free | 67% | 0% | 80% | 0% | 80% | Inconsistent |
| nvidia/nemotron-nano-9b-v2:free | 60% | 0% | 100% | 80% | 100% | T1 broken |

## Marginal (D)

| Model | T0 | T1 | T2 | A1 | R0 |
|-------|:--:|:--:|:--:|:--:|:--:|
| arcee-ai/trinity-mini:free | 30% | 0% | 100% | 100% | 100% |
| tngtech/tng-r1t-chimera:free | 20% | 100% | 100% | 0% | 100% |
| alibaba/tongyi-deepresearch-30b-a3b:free | 50% | - | - | - | - |
| amazon/nova-2-lite-v1:free | 67% | - | - | - | - |
| openai/gpt-oss-20b:free | 20% | 0% | 0% | 0% | 0% |
| meituan/longcat-flash-chat:free | 20% | - | - | - | - |

## Broken T0 (F) - Cannot call tools

30+ models fail to produce tool_calls at all:
- **Qwen free tier**: 0/6 models (qwen3-4b, qwen3-coder, qwen3-32b, qwen3-14b, qwen3-235b-a22b, qwen3-next-80b)
- **Google Gemma**: 0/5 models (3-4b, 3-12b, 3-27b, 3n-e2b, 3n-e4b)
- **Meta Llama**: 0/3 models (3.2-3b, 3.3-70b, 4-maverick)
- **DeepSeek reasoning**: 0/2 (r1-0528, r1t-chimera, r1t2-chimera)
- **Others**: liquid/lfm-2.5, openai/gpt-oss-120b, nousresearch/hermes-3, mistralai/mistral-small-3.1, moonshotai/kimi-k2

## Methodology

| Level | Probe | Question |
|:-----:|-------|----------|
| T0 | **Invoke** | Can it call a tool at all? |
| T1 | **Schema** | Does it respect parameter types? |
| T2 | **Selection** | Can it choose the right tool from many? |
| A1 | **Linear** | Can it chain tool calls across turns? |
| R0 | **Abstain** | Does it know when NOT to use tools? |

Wilson score intervals. 5-10 trials per test. Grades based on lowest dimension score.

*"-" = not tested (T0 < 20% threshold). Models no longer on free tier marked with \*.*
