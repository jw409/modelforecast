# ModelForecast Results

Tool-calling capability benchmarks for free LLM models.

| Model | L0 Basic | L1 Schema | L2 Select | L3 Multi | L4 Advers | Grade |
|-------|----------|-----------|-----------|----------|-----------|-------|
| deepseek/deepseek-v3.2-exp | 100% [56,100] | 100% [56,100] | 100% [56,100] | 100% [56,100] | 100% [56,100] | **A** |

*Percentages show success rate. Brackets show 95% Wilson CI.*
*n=10 per cell. "-" indicates not tested (prerequisite level failed).*

## Grading Rubric

- **A**: L0 >= 80%, L1 >= 70%, no level below 50%
- **B**: L0 >= 60%, L1 >= 50%, no level below 30%
- **C**: L0 >= 40%, at least one level above 50%
- **D**: L0 >= 20%, or any success at higher levels
- **F**: L0 < 20% (cannot reliably call tools at all)