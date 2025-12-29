# Games: GPU-Accelerated Simulation

GPU implementations of classic games for high-throughput AI evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU GAME EXECUTION                            │
├─────────────────────────────────────────────────────────────────┤
│  AI MODELS (Remote LLMs via OpenRouter)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GPT-5  │  Claude  │  Gemini  │  Grok  │  DeepSeek       │  │
│  │  Generate strategies, configs, or code                   │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  GPU SIMULATION (Local - RTX 5090)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Embarrassingly parallel game execution                   │  │
│  │  • CoreWars: 100,000 battles/sec (MARS on GPU)           │  │
│  │  • Angband: 10,000 borg instances in parallel            │  │
│  │  • Full game rules, not approximations                   │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  LOGGING                                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  • Full replay logs                                      │  │
│  │  • Decision traces                                       │  │
│  │  • Per-instance state snapshots                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Games

### CoreWars (Complete)

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
├── configs/          # AI-generated configurations
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

## Hardware Targets

| Platform | Use Case | Performance |
|----------|----------|-------------|
| RTX 5090 (local) | Primary execution | 27K+ battles/sec |
| Colab TPU v6e | Cloud comparison | TBD |
| WebGPU (future) | Browser execution | TBD |

## Status

- [x] CoreWars GPU implementation
- [x] Angband borg configs
- [ ] Angband borg GPU port

## License

MIT
