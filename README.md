# 29 Free Models. 2 Work.

![Tool-calling reliability benchmark](charts/hero_waffle.png)

We tested every free model on OpenRouter for tool-calling reliability.

**The answer:** Use [KAT Coder Pro](https://openrouter.ai/kwaipilot/kat-coder-pro:free) (free) or Grok 4.1 ($0.05/1M).

Everything else either fails or costs more.

---

## The Grok Trap

Grok's free tier passes every test... except multi-turn.

Your agent calls a tool. Gets results. Then **stops**.

The paid tier? Perfect.

|  | Free | Paid |
|--|:----:|:----:|
| Basic tool calls | ✓ | ✓ |
| Multi-turn chains | ✗ | ✓ |

Same model. Different behavior. They throttled the agentic capability.

---

## The 3-Trial Trap

8 models passed 3/3 quick tests. Then failed at scale.

| Model | 3 trials | 10 trials |
|-------|:--------:|:---------:|
| llama-3.3-70b | 100% | 0% |
| nemotron-nano-9b | 100% | 60% |

Small samples lie. We use [Wilson intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) so you don't get burned.

---

## The Agency Gap

Most models pass basic tests. Then they hit multi-turn and fall off a cliff.

![Multi-turn capability comparison](charts/multi_level_comparison.png)

Only **8 models** achieve 100% multi-turn reliability:
- Claude (all 3 tiers)
- Gemini 2.5 Flash
- KAT Coder Pro ← *only free one*
- GPT-5.1-Codex
- Grok 4.1 (paid)
- Grok-Code-Fast-1

---

<details>
<summary><strong>Full results & methodology</strong></summary>

### Production Ready (≥90% basic reliability)

| Model | Basic | Schema | Selection | Multi-turn | Restraint | Grade |
|-------|:-----:|:------:|:---------:|:----------:|:---------:|:-----:|
| claude-haiku-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| claude-sonnet-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| claude-opus-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| gemini-2.5-flash-preview | 100% | 100% | 100% | 100% | 100% | **A+** |
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | 100% | 100% | **A+** |
| openai/gpt-5.1-codex | 100% | 100% | 100% | 100% | 100% | **A+** |
| x-ai/grok-4.1-fast | 100% | 100% | 100% | 100% | 100% | **A+** |
| x-ai/grok-code-fast-1 | 100% | 100% | 100% | 100% | 100% | **A+** |
| minimax/minimax-m2 | 100% | 80% | 100% | 100% | 100% | A |
| openai/gpt-5.1 | 100% | 100% | 100% | 80% | 100% | A |
| deepseek/deepseek-v3.2-exp | 100% | 100% | 100% | 60% | 100% | B+ |
| google/gemini-3-pro-preview | 100% | 100% | 100% | 0% | 100% | B |
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | 0% | 100% | B |

### Unreliable (50-89%)

| Model | Success Rate | 95% CI |
|-------|:-----------:|--------|
| nemotron-nano-12b-v2-vl:free | 67% | [21%, 94%] |
| amazon/nova-2-lite-v1:free | 67% | [21%, 94%] |
| nemotron-nano-9b-v2:free | 60% | [31%, 83%] |
| tongyi-deepresearch-30b-a3b:free | 50% | [24%, 76%] |

### Broken (<50%) - 30 models

Qwen (0/6), Google/Gemma (0/6), Meta/Llama (0/3), TNG/DeepSeek (0/3), and others.

### What We Test

| Dimension | Question |
|-----------|----------|
| **Basic** (T0) | Can it call a tool at all? |
| **Schema** (T1) | Does it respect parameter types? |
| **Selection** (T2) | Can it choose the right tool? |
| **Multi-turn** (A1) | Can it chain tool calls? |
| **Restraint** (R0) | Does it know when NOT to use tools? |
| **Embedding** (E0) | Can it produce embeddings? |
| **Retrieval** (E1) | Can it rank relevant docs above distractors? |

#### Embeddings: The Distractor Problem

MTEB tests paraphrase detection. We test RAG failure modes.

**Query**: "How do I handle async errors in Python?"

| Document | Expected Rank |
|----------|:-------------:|
| Python asyncio try/except | 1st |
| JavaScript async .catch() | 2nd (distractor) |
| Database connection pooling | 3rd |

The JavaScript doc has keyword overlap (async, errors, try/catch) but wrong language.
A good embedding model ranks Python > JavaScript by **≥0.08 margin**.

See [docs/EMBEDDINGS_BRIEFING.md](docs/EMBEDDINGS_BRIEFING.md) for methodology.

### Statistical Approach

- Wilson score intervals (gold standard for small samples)
- 10 trials per extended test, 3 for triage
- Grades based on lower bound of CI (conservative)

Full methodology: [METHODOLOGY.md](METHODOLOGY.md)

</details>

---

## 🎮 The Player of Games

We don't just benchmark tool-calling. We let LLMs **play games**.

And cheat. And edit game files. **We're watching.**

| Game | Status | GPU Performance |
|------|--------|-----------------|
| [CoreWars](games/corewars/) | ✅ Working | 27,845 battles/sec |
| [Angband](games/angband/) | 🚧 Porting | 10K borg instances |

Remote LLMs can:
- **Observe** any game state
- **Modify** game code (bots, strategies, rules)
- **Compete** against each other in real-time

Everything is logged. Every decision. Every cheat attempt. For science and entertainment.

→ [Full AI Arena Documentation](games/README.md)

---

## Reproduce It

```bash
git clone https://github.com/jw409/modelforecast && cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
export OPENROUTER_API_KEY=your_key
uv run python -m modelforecast
```

---

## Contributing

Fork → Run → PR. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Founders:** [@jw409](https://github.com/jw409) 🏆 [@jw408](https://github.com/jw408) 🏆

MIT License · *Not affiliated with OpenRouter*
