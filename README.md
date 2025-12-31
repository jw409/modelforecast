# ModelForecast

We built a GPU colosseum. LLMs wrote [Redcode](https://corewar.co.uk/icws94.txt). They competed. A **free model** tied the best paid model.

Welcome to ModelForecast: **a benchmark suite that tests whether LLMs can learn and adapt under adversarial pressure.**

Static benchmarks ask "can you do X?" once. We run 10 rounds with 10,000 battles each. Your code fights. You watch. You iterate. Or you die.

---

## The Colosseum

```mermaid
flowchart LR
    subgraph Turn["Each Turn (×10)"]
        A[🤖 LLM] -->|writes| B[📝 Redcode]
        B -->|battles| C[⚔️ GPU MARS]
        C -->|10K fights| D[📊 Results]
        D -->|feedback| A
    end

    subgraph Surprise["Turn 6-7"]
        E[🐭 Champion]
        E -->|boss fight| C
    end

    style A fill:#4a9eff
    style C fill:#ff6b6b
    style E fill:#ffd93d
```

**CoreWars**: Assembly gladiators fight for control of shared memory. Kill your opponent's processes or die trying.

Each model starts with a basic IMP (`MOV 0, 1`). They watch 10,000 battles. They write improved code. They repeat for 10 turns. At Turn 6, a surprise champion appears.

**The Results:**

| Rank | Model | Win Rate | Improvement | Cost |
|:----:|-------|:--------:|:-----------:|:----:|
| 🥇 | **KAT Coder Pro** | 36.0% | +19.0% | FREE |
| 🥇 | GPT-4o Mini | 36.0% | +19.2% | $0.15/1M |
| 🥉 | Gemini Flash | 33.0% | +10.6% | $0.075/1M |
| 4 | DeepSeek V3 | 27.8% | +11.3% | $0.14/1M |
| — | GLM 4.5 Air | — | — | FREE |
| — | Claude Haiku | — | — | $0.25/1M |

The free model tied the champion.

---

## The Drama

### 🐙 KAT Coder Pro: Zero-Dollar Champion

Turn 1: Basic IMP. 16.6% win rate.
Turn 10: Multi-process bomber with imp gates. 36.0% win rate.

**+19% improvement. Zero cost.** A model you've never heard of, available for free, matched the best paid model on the market.

### 🐭 The Boss Fight

Turn 6-7: Surprise champion. Everyone faces Mice.red—a self-replicating nightmare:

```asm
; Mice - Self-replicating warrior
    ptr     DAT #0
    start   MOV #12, count
    loop    MOV @ptr, <dest
            DJN loop, count
            SPL @dest
            ADD #653, ptr
            JMZ start, ptr
    count   DAT #0
    dest    DAT #833
```

Win rates dropped 5-15%. The real benchmark isn't the average case. It's whether you can handle the boss.

---

## Watch The Battles

### **[▶ OPEN ARENA](https://jw409.github.io/modelforecast/docs/corewars/)**

8192-cell memory grid. Auto-zoom. Dramatic intros. Watch LLMs fight in real time.

- **27,845 battles/sec** on GPU (RTX 5090)
- Full tournament playback with leaderboard
- Every battle recorded and replayable

---

## Why Games Beat Benchmarks

Static benchmarks ask: "Can you call a function?"

Game benchmarks ask:
- **Learning**: Did performance improve over time?
- **Adversarial pressure**: Can you handle surprise opponents?
- **Real stakes**: Wrong moves = immediate loss

One measures capability. The other measures adaptation.

---

## Tool-Calling Benchmark

Separate from tournaments, we test whether models can use tools reliably. Five levels: basic calls (L0), schema compliance (L1), tool selection (L2), multi-turn chaining (L3), and knowing when NOT to call tools (L4).

139 result files across 25+ providers. Wilson score intervals. 5-10 trials per test.

### Perfect Score (A+)

| Model | L0 Basic | L1 Schema | L2 Selection | L3 Multi-turn | L4 Restraint | Grade |
|-------|:--------:|:---------:|:------------:|:-------------:|:------------:|:-----:|
| anthropic/claude-haiku-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| anthropic/claude-sonnet-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| anthropic/claude-opus-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| google/gemini-2.5-flash-preview | 100% | 100% | 100% | 100% | 100% | **A+** |
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | 100% | 100% | **A+** |
| openai/gpt-5.1-codex | 100% | 100% | 100% | 100% | 100% | **A+** |
| x-ai/grok-4.1-fast | 100% | 100% | 100% | 100% | 100% | **A+** |
| x-ai/grok-code-fast-1 | 100% | 100% | 100% | 100% | 100% | **A+** |

### Production Ready (A/A-)

| Model | L0 | L1 | L2 | L3 | L4 | Grade |
|-------|:--:|:--:|:--:|:--:|:--:|:-----:|
| minimax/minimax-m2 | 100% | 80% | 100% | 100% | 100% | A |
| openai/gpt-5.1 | 100% | 100% | 100% | 80% | 100% | A |
| deepseek/deepseek-v3.2-exp | 100% | 100% | 100% | 60% | 100% | B+ |

### L3 Multi-Turn Failures

| Model | L0 | L1 | L2 | L3 | L4 | Notes |
|-------|:--:|:--:|:--:|:--:|:--:|-------|
| google/gemini-3-pro-preview | 100% | 100% | 100% | **0%** | 100% | Can't chain tools |
| openai/gpt-5-mini | 100% | 100% | 80% | **20%** | 100% | Budget = weak L3 |
| openai/gpt-5.1-codex-mini | 100% | 100% | 100% | **20%** | 100% | Budget = weak L3 |

### Unreliable L0

| Model | L0 Rate | 95% CI |
|-------|:-------:|--------|
| nemotron-nano-12b-v2-vl:free | 67% | [21%, 94%] |
| amazon/nova-2-lite-v1:free | 67% | [21%, 94%] |
| nemotron-nano-9b-v2:free | 60% | [31%, 83%] |
| alibaba/tongyi-deepresearch-30b-a3b:free | 50% | [23%, 76%] |

### Broken L0 (<50%)

30+ models can't reliably call tools at all:
- **Qwen free tier**: 0/6 models
- **Llama free tier**: 0/3 models
- **DeepSeek R1T variants**: 0/3 models
- **Others**: longcat-flash, gpt-oss-20b, kimi-k2

### Methodology

| Level | Dimension | Question |
|:-----:|-----------|----------|
| L0 | **Basic** | Can it call a tool at all? |
| L1 | **Schema** | Does it respect parameter types? |
| L2 | **Selection** | Can it choose the right tool? |
| L3 | **Multi-turn** | Can it chain tool calls? |
| L4 | **Restraint** | Does it know when NOT to use tools? |

Wilson score intervals. 5-10 trials per test. Grades based on lowest dimension score.

---

## Run It Yourself

Test your own models on your own hardware.

```bash
git clone https://github.com/jw409/modelforecast && cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
export OPENROUTER_API_KEY=your_key

# Tool-calling benchmark
uv run python -m modelforecast

# CoreWars tournament
uv run python games/corewars/model_benchmark.py
```

---

## What We Learned

**Price doesn't predict performance.** KAT Coder Pro (free) tied GPT-4o Mini ($0.15/1M tokens). Both hit 36% win rate. Both improved +19% over 10 rounds.

**Small samples lie.** 8 models passed 3/3 trials then failed at scale. Wilson score intervals or you're fooling yourself.

**Boss fights matter.** When Mice.red appeared at Turn 6, win rates dropped 5-15%. The benchmark isn't your average case—it's your worst.

---

**Founders:** [@jw409](https://github.com/jw409) [@jw408](https://github.com/jw408)

MIT License · *Not affiliated with OpenRouter*
