# ModelForecast Results

Tool-calling capability benchmarks for free LLM models.

| Model | T0 Invoke | T1 Schema | T2 Select | A1 Linear | R0 Abstain | Grade |
|-------|-----------|-----------|-----------|-----------|------------|-------|
| kwaipilot/kat-coder-pro | 100% [56,100] | 100% [56,100] | 100% [56,100] | 100% [56,100] | 100% [56,100] | **A** |

*Percentages show success rate. Brackets show 95% Wilson CI.*
*n=10 per cell. "-" indicates not tested (prerequisite probe failed).*

## Grading Rubric

- **A**: T0 >= 80%, T1 >= 70%, no probe below 50%
- **B**: T0 >= 60%, T1 >= 50%, no probe below 30%
- **C**: T0 >= 40%, at least one probe above 50%
- **D**: T0 >= 20%, or any success at higher probes
- **F**: T0 < 20% (cannot reliably call tools at all)