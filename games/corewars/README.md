# GPU MARS (Core War on CUDA)

**25 million Core War battles in 96 seconds.** LLM-generated warriors fight on GPU.

```
=== LLM TOURNAMENT RESULTS (25.2M battles, RTX 5090) ===

| Rank | Model               | Warrior   | Win Rate |
|------|---------------------|-----------|----------|
| 🥇   | Claude Sonnet 4.5   | Silk      | 94.2%    |
| 🥈   | DeepSeek V3-0324    | Hydra     | 52.5%    |
| 🥉   | KwaiPilot Kat-Coder | Ironclad  | 66.2%    |
| 4    | GPT-5.1-chat        | StoneGate | 41.6%    |
| 5    | Claude Opus 4.5     | Granite   | 5.6%     |

Throughput: 262,000 battles/sec (batched) | 681,000 battles/sec (single batch)
```

Each model was given the ICWS'94 spec and asked to write a warrior. No examples, no fine-tuning.
Sonnet's 23-instruction scanner/bomber/replicator hybrid crushed Opus's 3-line rolling stone.

**[See the warriors →](warriors/tournament_001/)**

---

## Requirements
- NVIDIA GPU (RTX 3060 or better recommended)
- CUDA Toolkit 11+
- Linux (tested on Ubuntu with RTX 5090)

## Building

The Makefile auto-detects your GPU and sets the correct architecture:

```bash
# Check detected GPU settings
make gpu-info

# Build (auto-detects sm_120 for RTX 5090, sm_90 for RTX 4090, etc.)
make
```

**Manual override** (if auto-detection fails):
```bash
# Set architecture explicitly
make GPU_ARCH=sm_120   # RTX 5090 (Blackwell)
make GPU_ARCH=sm_90    # RTX 4090 (Ada Lovelace)
make GPU_ARCH=sm_86    # RTX 3090 (Ampere)
make GPU_ARCH=sm_80    # A100 (Ampere datacenter)
```

**Compute Capability Reference:**
| GPU | Architecture | SM Version |
|-----|--------------|------------|
| RTX 5090 | Blackwell | sm_120 |
| RTX 4090 | Ada Lovelace | sm_90 |
| RTX 3090 | Ampere | sm_86 |
| A100 | Ampere | sm_80 |
| V100 | Volta | sm_70 |

## Running

```bash
# Battle any two .red files
./build/gpu_mars_interleaved warrior_a.red warrior_b.red [num_battles]

# Example: Run tournament matchup
./build/gpu_mars_interleaved warriors/tournament_001/anthropic_claude-sonnet-4.5.red \
                              warriors/tournament_001/anthropic_claude-opus-4.5.red 300000

# Fallback to built-in DWARF vs IMP
./build/gpu_mars_interleaved 100000
```

**VRAM Budget:** ~96KB per battle. RTX 5090 (32GB) → max ~300K concurrent battles.

## Implementations

1. `gpu_mars` (Naive AoS): Simplest code, decent performance (16k battles/sec).
2. `gpu_mars_soa` (Full SoA): Slowest due to memory transaction overhead.
3. `gpu_mars_packed` (Packed SoA): Slow due to read-modify-write on updates.
4. `gpu_mars_interleaved` (Transposed AoS): **Fastest (27k battles/sec)**. Coalesced access + direct writes.

## Performance (RTX 5090)

| Metric | Value |
|--------|-------|
| Throughput (single batch) | 681,000 battles/sec |
| Throughput (batched 25M) | 262,000 battles/sec |
| Instructions/sec | ~54 Billion |
| VRAM per battle | 96 KB |
| Max concurrent (32GB) | ~300,000 battles |

**Benchmark:** 25.2 million battles in 96 seconds.

## Swarm Terminal Visualizer

Real-time 16x16 grid visualization of 256 parallel Core War battles (IMP vs DWARF).

```bash
# Build
make swarm_term

# Run (battles, frame_delay_ms, max_cycles)
./build/swarm_term 256 30 10000

# Fast mode (minimal delay)
make term_fast
```

**Visual Legend:**
- `A` Cyan = IMP winner
- `B` Magenta = DWARF winner
- `=` Yellow = Tie (both alive at max cycles)
- `·` Gray = Battle in progress

**Example Output:**
```
╔════════════════ COREWARS GPU SWARM ════════════════╗
║ Cycle: 500 | A:12 B:8 Tie:0 Live:236 | 1.4M steps/s ║
╠════════════════════════════════════════════════════╣
║····A···B···A···║
║·B··············║
║······A·····B···║
...
```

**Performance:** ~1.4-2.4M steps/sec on RTX 5090
