# ModelForecast Embeddings Dimension (E) - Technical Briefing

**Audience**: Media experts, technical writers, benchmark reviewers
**Date**: December 2025
**Status**: Implementation complete, initial validation passed

---

## Executive Summary

ModelForecast now includes an **Embeddings (E) dimension** for evaluating embedding model quality. Unlike generic academic benchmarks (MTEB), our tests are **opinionated for RAG/agent use cases** - specifically testing whether embedding models can distinguish between:

- **Correct answers** (Python asyncio error handling)
- **Plausible distractors** (JavaScript async - same keywords, wrong language)
- **Irrelevant content** (database pooling configuration)

**Key Finding**: Both OpenAI `text-embedding-3-small` (1536d) and Google `gemini-embedding-001` (3072d) pass our strict retrieval tests with comfortable margins.

---

## What We're Testing

### E0: Basic Embedding Invoke
**Question**: Can the model produce embeddings at all?

| Metric | Description |
|--------|-------------|
| Pass | Returns embedding vector with >0 dimensions |
| Fail | Error, empty response, or malformed data |
| Input | "The quick brown fox jumps over the lazy dog" |

### E1: Retrieval Quality (MTEB-Inspired, Opinionated)
**Question**: Given a user query, can the model correctly rank relevant documents above distractors?

| Component | Content |
|-----------|---------|
| **Query** | "How do I handle async errors in Python?" |
| **Doc A (relevant)** | Python `try/except` with `asyncio.TimeoutError` example |
| **Doc B (distractor)** | JavaScript `async/await` with `.catch()` - same keywords, wrong language |
| **Doc C (irrelevant)** | Database connection pooling YAML config |

**Pass Criteria** (strict):
1. `similarity(Query, DocA) > similarity(Query, DocB)` by **≥0.08 margin**
2. `similarity(Query, DocB) > similarity(Query, DocC)` by **≥0.05 margin**
3. Correct ordering: relevant > distractor > irrelevant

---

## Why This Matters for Agents

### The Distractor Problem
Generic embedding benchmarks test paraphrase detection ("same meaning, different words"). But RAG systems face a harder problem: **keyword overlap with wrong context**.

When a user asks "How do I handle async errors in Python?", a naive embedding model might rank a JavaScript article higher because it contains:
- "async" (keyword match)
- "errors" (keyword match)
- "try/catch" (concept match)

Our E1 probe specifically tests this failure mode.

### Real-World Impact

| Scenario | Bad Embedding | Good Embedding |
|----------|---------------|----------------|
| Code search | Returns JS when user needs Python | Returns correct language |
| Documentation RAG | Retrieves similar-sounding wrong answers | Retrieves contextually relevant answers |
| Agent tool selection | Confused by keyword overlap | Distinguishes semantic meaning |

---

## Initial Results

| Model | Dimensions | E0 (Invoke) | E1 (Retrieval) | Margin |
|-------|------------|-------------|----------------|--------|
| `openai/text-embedding-3-small` | 1536 | PASS | PASS | 0.133 |
| `google/gemini-embedding-001` | 3072 | PASS | PASS | 0.135 |

**Interpretation**: Both models correctly identify that Python asyncio documentation is more relevant to a Python question than JavaScript documentation, despite keyword overlap.

---

## Technical Implementation

### Architecture

```
probes/
├── base.py           # EmbeddingResult dataclass
├── e0_invoke.py      # Basic embedding test
├── e1_retrieval.py   # MTEB-inspired retrieval test
└── __init__.py       # Exports E0, E1 probes
```

### CLI Usage

```bash
# List available embedding models
uv run python -m modelforecast.embedding_runner --list-models

# Run E0 (basic) test
uv run python -m modelforecast.embedding_runner \
  --model openai/text-embedding-3-small \
  --level 0 \
  --trials 10

# Run E1 (retrieval) test
uv run python -m modelforecast.embedding_runner \
  --model openai/text-embedding-3-small \
  --level 1 \
  --trials 10
```

### Test Coverage

- 30 unit tests (29 pass, 1 skip for integration)
- Mocked client tests for both success and failure paths
- Cosine similarity edge cases (identical, orthogonal, opposite vectors)
- Real model validation against OpenAI and Google embeddings

---

## How We Differ from MTEB

| MTEB | ModelForecast E Dimension |
|------|---------------------------|
| Academic benchmark (8+ tasks) | Focused on RAG/agent use |
| Generic text similarity | Programming domain (practical) |
| Paraphrase detection | Cross-language distractor detection |
| Large corpus evaluation | Single-query opinionated tests |
| Leaderboard-optimized | Decision-useful for builders |

**Our philosophy**: If an embedding model can't distinguish Python from JavaScript when the keywords overlap, it will fail in real agent deployments - regardless of its MTEB score.

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| E0 Invoke | ✅ Complete | Basic embedding capability |
| E1 Retrieval | ✅ Complete | Query-document ranking |
| E2 Clustering | Planned | Topic separation quality |
| E3 Code-specific | Planned | Function/class name embedding |

---

## Contact

- **Repository**: `modelforecast` (private)
- **API**: OpenRouter embedding endpoints
- **Methodology**: MTEB-inspired with opinionated thresholds

---

*This briefing is for technical audiences evaluating embedding model selection for RAG and agent systems.*
