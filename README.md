# ModelForecast

Reproducible benchmark toolkit for evaluating LLM capabilities via OpenRouter. Pre-defined tests for tool-calling, embeddings, and adversarial code generation.

```bash
export OPENROUTER_API_KEY=your_key
uv run python -m modelforecast
```

## GPU Arena: CoreWars

**25 million battles in 96 seconds.** LLMs write assembly warriors, GPU (CUDA) accelerated execution.

| Rank | Model | Win Rate |
|:----:|-------|:--------:|
| 1 | Claude Sonnet 4.5 | 94.2% |
| 2 | KwaiPilot KAT Coder Pro | 66.2% |
| 3 | DeepSeek V3 | 52.5% |
| 4 | GPT-5.1 | 41.6% |
| 5 | Claude Opus 4.5 | 5.6% |

[Learn how CoreWars works →](docs/COREWARS.md)

### Performance

| Hardware | Throughput | Notes |
|----------|------------|-------|
| RTX 5090 (GPU) | 262,000 battles/sec | Batched, 300K concurrent |
| i9-14900K (CPU) | 1,700 battles/sec | pMARS, single-threaded |

**154x speedup** with GPU acceleration.

---

## Tool-Calling Benchmark (TRA Framework)

Can models reliably invoke, chain, and restrain tool usage?

| Dimension | Code | Question |
|-----------|------|----------|
| **Invoke** | T0 | Can it produce a tool call at all? |
| **Schema** | T1 | Does it respect parameter types? |
| **Selection** | T2 | Can it choose the right tool? |
| **Agency** | A1 | Can it chain tool calls across turns? |
| **Restraint** | R0 | Does it know when NOT to use tools? |

### Grade A (Production Ready)

| Model | T0 | T1 | T2 | A1 | R0 |
|-------|:--:|:--:|:--:|:--:|:--:|
| anthropic/claude-haiku-4.5 | 100% | 100% | 100% | 100% | 100% |
| anthropic/claude-opus-4.5 | 100% | 100% | 100% | 100% | 100% |
| anthropic/claude-sonnet-4.5 | 100% | 100% | 100% | 100% | 100% |
| deepseek/deepseek-v3.2-exp | 100% | 100% | 100% | 100% | 100% |
| google/gemini-2.5-flash-preview | 100% | 100% | 100% | 100% | 100% |
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | 100% | 100% |
| openai/gpt-5.1 | 100% | 100% | 100% | 100% | 100% |
| openai/gpt-5.1-codex | 100% | 100% | 100% | 100% | 100% |
| x-ai/grok-4.1-fast | 100% | 100% | 100% | 100% | 100% |
| x-ai/grok-code-fast-1 | 100% | 100% | 100% | 100% | 100% |

### Grade B (Agency Issues)

| Model | T0 | T1 | T2 | A1 | R0 | Notes |
|-------|:--:|:--:|:--:|:--:|:--:|-------|
| google/gemini-3-pro-preview | 100% | 100% | 100% | **0%** | 100% | Can't chain tools |
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | **0%** | 100% | Free tier throttled |

### Grade F (Broken Tool Calling)

30+ free-tier models cannot reliably call tools:

- **Qwen free tier**: qwen3-coder, qwen3-4b, qwen3-14b, qwen3-30b, qwen3-32b, qwen3-235b (all 0%)
- **Llama free tier**: llama-3.3-70b, llama-4-maverick, hermes-3-405b (all 0%)
- **DeepSeek free tier**: deepseek-chat-v3-0324, r1t-chimera variants (all 0%)
- **Gemma free tier**: gemma-3-4b, gemma-3-12b, gemma-3-27b (all 0%)

[Full methodology →](docs/METHODOLOGY.md)

---

## Embeddings Benchmark

Can embedding models distinguish relevant docs from keyword-matching distractors?

| Model | E0 (Invoke) | E1 (Retrieval) | Margin |
|-------|:-----------:|:--------------:|:------:|
| openai/text-embedding-3-small | PASS | PASS | 0.133 |
| google/gemini-embedding-001 | PASS | PASS | 0.135 |

**E1 test**: Query "async errors in Python" must rank Python docs above JavaScript docs (same keywords, wrong language). Both models pass with comfortable margins.

[Full embeddings methodology →](docs/EMBEDDINGS_BRIEFING.md)

---

## Installation

```bash
git clone https://github.com/jw409/modelforecast && cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
export OPENROUTER_API_KEY=your_key

# Tool-calling benchmark
uv run python -m modelforecast

# Embeddings benchmark
uv run python -m modelforecast.embedding_runner --model openai/text-embedding-3-small --level 1

# CoreWars tournament (GPU required)
cd games/corewars && make && ./gpu_mars_tournament

# CoreWars tournament (CPU only)
./pmars -r 1000 warriors/*.red
```

---

**Founders:** [@jw409](https://github.com/jw409) [@jw408](https://github.com/jw408)

MIT License
