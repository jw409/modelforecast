# AI Arena: GPU-Accelerated AI Competition

> **The next Netflix show about AI.**

## Concept

AI models compete by playing games. Not just playing - **programming the games**.

Remote LLMs (contestants) can:
- **Observe** any game state they request
- **Modify** game code (bots, strategies, rules)
- **Compete** against each other in real-time

Everything is logged for replay. Drama emerges naturally.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI ARENA ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│  CONTESTANTS (Remote LLMs via OpenRouter)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GPT-5  │  Claude-4  │  Gemini-3  │  Grok-5  │  etc...   │  │
│  │  Can see anything. Can modify code. We're watching.      │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  FERTILE FIELD (Local GPU - RTX 5090)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Embarrassingly parallel game execution                   │  │
│  │  • CoreWars: 100,000 battles/sec (MARS on GPU)           │  │
│  │  • Angband: 10,000 borg instances in parallel            │  │
│  │  • Full game rules, not approximations                   │  │
│  │  • Custom code from contestants runs in sandbox          │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  NARRATOR (Local Model)                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Watches all contestant actions                        │  │
│  │  • Generates commentary/narrative                        │  │
│  │  • Detects dramatic moments for highlights               │  │
│  │  • "Why did GPT-5 sacrifice its queen?"                  │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  PRODUCTION                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Full replay logs                                      │  │
│  │  • AI decision traces                                    │  │
│  │  • Multi-camera views (per-contestant)                   │  │
│  │  • Highlight reels                                       │  │
│  │  • Episode compilation                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Games

### CoreWars (Working)

Two programs battle in shared memory. Classic 1984 competition.

- **GPU Implementation**: CUDA kernel with interleaved memory layout
- **Performance**: 27,845 battles/sec on RTX 5090
- **Status**: ✅ Complete

```bash
cd games/corewars
make
./build/gpu_mars_interleaved 100000  # Run 100K battles
```

### Angband (In Progress)

Classic roguelike. AI controls the APWBorg autonomous player.

- **Borg Source**: 1.9MB of decision logic (borg1-9.c)
- **Config**: 1400+ parameters in borg.txt
- **GPU Port**: Converting borg_think() to CUDA kernel
- **Status**: 🚧 Porting to GPU

```
games/angband/
├── apwborg/          # Original borg source
├── configs/          # 8 AI-generated configurations
│   ├── meta.txt      # Opus: game-theoretic optimization
│   ├── evolution_*.txt  # Opus: phase-adaptive
│   ├── aggro.txt     # Sonnet: high risk
│   └── cheat.txt     # Gemini: immortal glass cannon
├── gpu/              # CUDA port (WIP)
└── harness/          # Python interface
```

## Memory Layout: Interleaved

All games use interleaved memory for optimal GPU coalescing:

```cuda
// Standard (bad for GPU):
state[instance].field[row]

// Interleaved (good for GPU):
field[row * num_instances + instance]
// Adjacent threads access adjacent memory addresses
```

See `common/interleaved.h` for helpers.

## Contestant API

Remote LLMs interact via HTTP:

```python
# Get game state
GET /arena/{game}/state?contestant_id=42
→ {"position": [10, 20], "hp": 150, "monsters": [...]}

# Submit action
POST /arena/{game}/action
{"contestant_id": 42, "action": "MOVE_N", "reasoning": "Avoiding the dragon"}

# Submit custom code (!)
POST /arena/{game}/code
{"contestant_id": 42, "code": "...", "language": "cuda"}
```

## What Makes This Different

| Traditional Benchmarks | AI Arena |
|------------------------|----------|
| Run games | Run **television** |
| Measure scores | Capture **drama** |
| Test capability | Reveal **personality** |
| Static rules | Contestants **modify code** |
| Single instance | 100,000 **parallel** |
| Report numbers | Generate **narratives** |

## Hardware Targets

| Platform | Use Case | Performance |
|----------|----------|-------------|
| RTX 5090 (local) | Primary execution | 27K+ battles/sec |
| Colab TPU v6e | Cloud comparison | TBD |
| WebGPU (future) | Browser streaming | TBD |

## Roadmap

- [x] CoreWars GPU implementation
- [x] Angband borg configs (8 variants)
- [ ] Angband borg GPU port
- [ ] Unified arena API
- [ ] Contestant HTTP interface
- [ ] Narrator integration
- [ ] Replay renderer
- [ ] First episode

## License

MIT - but if you make a Netflix show, credit us.
