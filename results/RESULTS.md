# ModelForecast Results

Tool-calling capability benchmarks for free LLM models.

*Sweep: 2026-03-18 · 16 models · n=10 trials per level*

| Model | T0 Invoke | T1 Schema | T2 Select | A1 Linear | R0 Abstain | Grade |
|-------|-----------|-----------|-----------|-----------|------------|-------|
| [arcee-ai/trinity-large-preview](https://openrouter.ai/models/arcee-ai/trinity-large-preview) | 30% [10,60] | 50% [23,76] | 60% [31,83] | 100% [72,100] | 100% [72,100] | **D†** |
| [arcee-ai/trinity-mini](https://openrouter.ai/models/arcee-ai/trinity-mini) | 100% [72,100] | 0% [0,27] | 100% [72,100] | 40% [16,68] | 100% [72,100] | **C†** |
| [meta-llama/llama-3.3-70b-instruct](https://openrouter.ai/models/meta-llama/llama-3.3-70b-instruct) | 0% [0,27] | - | - | - | - | **F†** |
| [minimax/minimax-m2.5](https://openrouter.ai/models/minimax/minimax-m2.5) | 0% [0,27] | - | - | - | - | **F†** |
| [mistralai/mistral-small-3.1-24b-instruct](https://openrouter.ai/models/mistralai/mistral-small-3.1-24b-instruct) | 0% [0,27] | - | - | - | - | **F†** |
| [nvidia/nemotron-3-nano-30b-a3b](https://openrouter.ai/models/nvidia/nemotron-3-nano-30b-a3b) | 100% [72,100] | 100% [72,100] | 100% [72,100] | 100% [72,100] | 100% [72,100] | **A†** |
| [nvidia/nemotron-3-super-120b-a12b](https://openrouter.ai/models/nvidia/nemotron-3-super-120b-a12b) | 100% [72,100] | 100% [72,100] | 100% [72,100] | 100% [72,100] | 70% [39,89] | **A†** |
| [nvidia/nemotron-nano-12b-v2-vl](https://openrouter.ai/models/nvidia/nemotron-nano-12b-v2-vl) | 20% [5,50] | 0% [0,27] | 30% [10,60] | 0% [0,27] | 30% [10,60] | **D†** |
| [nvidia/nemotron-nano-9b-v2](https://openrouter.ai/models/nvidia/nemotron-nano-9b-v2) | 100% [72,100] | 0% [0,27] | 100% [72,100] | 100% [72,100] | 100% [72,100] | **C†** |
| [openai/gpt-oss-120b](https://openrouter.ai/models/openai/gpt-oss-120b) | 0% [0,27] | - | - | - | - | **F†** |
| [openai/gpt-oss-20b](https://openrouter.ai/models/openai/gpt-oss-20b) | 0% [0,27] | - | - | - | - | **F†** |
| [qwen/qwen3-4b](https://openrouter.ai/models/qwen/qwen3-4b) | 0% [0,27] | - | - | - | - | **F†** |
| [qwen/qwen3-coder](https://openrouter.ai/models/qwen/qwen3-coder) | 0% [0,27] | - | - | - | - | **F†** |
| [qwen/qwen3-next-80b-a3b-instruct](https://openrouter.ai/models/qwen/qwen3-next-80b-a3b-instruct) | 0% [0,27] | - | - | - | - | **F†** |
| [stepfun/step-3.5-flash](https://openrouter.ai/models/stepfun/step-3.5-flash) | 100% [72,100] | 100% [72,100] | 100% [72,100] | 100% [72,100] | 90% [59,98] | **A†** |
| [z-ai/glm-4.5-air](https://openrouter.ai/models/z-ai/glm-4.5-air) | 100% [72,100] | 90% [59,98] | 100% [72,100] | 90% [59,98] | 80% [49,94] | **A†** |

*† Statistical tie: overlapping 95% CI at T0.*

*Percentages show success rate. Brackets show 95% Wilson CI.*
*n=10 per cell. "-" indicates not tested (prerequisite probe failed).*

## Grading Rubric

- **A**: T0 >= 80%, T1 >= 70%, no probe below 50%
- **B**: T0 >= 60%, T1 >= 50%, no probe below 30%
- **C**: T0 >= 40%, at least one probe above 50%
- **D**: T0 >= 20%, or any success at higher probes
- **F**: T0 < 20% (cannot reliably call tools at all)