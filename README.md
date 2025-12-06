# ARE YOU NOT ENTERTAINED?!

We made LLMs write assembly warriors and fight **25 million battles** in 96 seconds.

**262,000 battles/sec** on RTX 5090.

```mermaid
flowchart LR
    A[🤖 LLM] -->|reads spec| B[📜 ICWS'94]
    B -->|writes| C[📝 Redcode]
    C -->|loads| D[⚔️ GPU MARS]
    D -->|25M battles| E[📊 Results]

    style A fill:#4a9eff
    style D fill:#ff6b6b
```

## Results

| Rank | Model | Win Rate | Strategy |
|:----:|-------|:--------:|----------|
| 🥇 | **Claude Sonnet 4.5** | 94.2% | Multi-process scanner + bomber + replicator |
| 🥈 | KwaiPilot KAT Coder Pro | 66.2% | Bomber with protection gate |
| 🥉 | DeepSeek V3 | 52.5% | Scanner/replicator hybrid |
| 4 | GPT-5.1 | 41.6% | Mod-4 bomber with imp-gate |
| 💀 | Claude Opus 4.5 | 5.6% | Rolling stone (3 lines) |

The $0 model beat the $15/M model. The most expensive one wrote 3 lines and lost.

---

## The Game

**CoreWars**: Assembly programs fight for control of shared memory. Kill opponent processes or die.

Each model:
1. Reads the ICWS'94 Redcode specification
2. Writes ONE warrior (no iteration, no feedback)
3. Battles every other warrior 1M times each
4. Winner = highest total win rate

**No learning. No adaptation. Just raw code generation from spec.**

---

## The Warriors

### 🥇 Claude Sonnet 4.5: "Silk"

Multi-pronged attack with redundancy:

```asm
start   SPL     bomber          ; Split off bomber process
        SPL     scanner         ; Split off scanner process
        SPL     replicate       ; Split off replication process

scanner ADD.AB  #15,    sptr    ; Hunt for enemy code
sptr    JMZ.F   scanner, 300    ; If zero, keep scanning
        MOV.AB  sbomb,  @sptr   ; Found! Bomb it
```

**Why it wins**: Creates 4 parallel attack vectors. If one dies, others continue.

### 💀 Claude Opus 4.5: "Granite"

```asm
stone   MOV.I   <-100,  >200    ; Decrement behind, increment ahead
        ADD.AB  #653,   stone   ; Change bombing distance
        JMP.A   stone           ; Loop
```

**Why it loses**: Single process, predictable pattern, no defense. Dies to any scanner.

---

## GPU Performance

| Metric | Value |
|--------|-------|
| Total battles | 25,200,000 |
| Time | 95.9 seconds |
| Throughput | 262,773 battles/sec |
| GPU | RTX 5090 |
| VRAM used | ~28 GB |

Batched execution: 84 batches × 300K concurrent battles.

---

<details>
<summary><strong>Tool-Calling Framework (L0-L4)</strong></summary>

Separate benchmark: Can models reliably call tools?

### Perfect Score (A+) - 100% across all dimensions

| Model | L0 Basic | L1 Schema | L2 Selection | L3 Multi-turn | L4 Restraint | Grade |
|-------|:--------:|:---------:|:------------:|:-------------:|:------------:|:-----:|
| anthropic/claude-haiku-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| anthropic/claude-sonnet-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| anthropic/claude-opus-4.5 | 100% | 100% | 100% | 100% | 100% | **A+** |
| google/gemini-2.5-flash-preview | 100% | 100% | 100% | 100% | 100% | **A+** |
| kwaipilot/kat-coder-pro:free | 100% | 100% | 100% | 100% | 100% | **A+** |
| openai/gpt-5.1-codex | 100% | 100% | 100% | 100% | 100% | **A+** |
| x-ai/grok-4.1-fast | 100% | 100% | 100% | 100% | 100% | **A+** |

### L3 Multi-Turn Failures (The Grok Trap)

| Model | L0 | L1 | L2 | L3 | L4 | Notes |
|-------|:--:|:--:|:--:|:--:|:--:|-------|
| x-ai/grok-4.1-fast:free | 100% | 100% | 100% | **0%** | 100% | Free tier throttled |
| openai/gpt-5-mini | 100% | 100% | 80% | **20%** | 100% | Budget = weak L3 |

Free Grok stops after one tool call. They throttled the agentic capability.

### Methodology

| Level | Dimension | Question |
|:-----:|-----------|----------|
| L0 | **Basic** | Can it call a tool at all? |
| L1 | **Schema** | Does it respect parameter types? |
| L2 | **Selection** | Can it choose the right tool? |
| L3 | **Multi-turn** | Can it chain tool calls? |
| L4 | **Restraint** | Does it know when NOT to use tools? |

Wilson score intervals. 5-10 trials per test.

</details>

---

## Run It

```bash
git clone https://github.com/jw409/modelforecast && cd modelforecast
curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
export OPENROUTER_API_KEY=your_key

# Generate warriors (calls LLMs)
uv run python games/corewars/generate_warriors.py

# Run tournament (GPU)
cd games/corewars && make && ./gpu_mars_tournament
```

---

**Founders:** [@jw409](https://github.com/jw409) [@jw408](https://github.com/jw408)

MIT License
