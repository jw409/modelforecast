# GPU DOOM Arena

**Original id Software DOOM source (1993) ported to CUDA for massively parallel AI evaluation.**

## The Stunt

Run the *actual* DOOM game logic on GPU. Not an approximation. Not a wrapper. The real `P_Ticker()` from `linuxdoom-1.10`.

## Architecture: Pure GPU

```
┌─────────────────────────────────────────────────────────────────┐
│  GPU (RTX 5090 - 32GB VRAM)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GAME STATE (Interleaved)                                │   │
│  │  - 100 parallel instances × full game state              │   │
│  │  - player_t, mobj_t pools, thinker lists                 │   │
│  │  - Level geometry (shared, read-only)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  INPUT RING BUFFER                                       │   │
│  │  - ticcmd_t[INSTANCES][HORIZON] (8 bytes × 100 × 1000)  │   │
│  │  - Pre-loaded action sequences from contestants          │   │
│  │  - GPU consumes autonomously                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CHECKPOINT BUFFER                                       │   │
│  │  - Snapshots at configurable intervals                   │   │
│  │  - {tick, health, armor, kills, ammo, position, angle}   │   │
│  │  - GPU writes, host reads async                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  KERNEL: doom_simulate()                                 │   │
│  │  for tick in 0..horizon:                                 │   │
│  │      __syncthreads()                                     │   │
│  │      ticcmd_t cmd = input_buffer[instance][tick]         │   │
│  │      P_PlayerThink_GPU(instance, cmd)                    │   │
│  │      P_RunThinkers_GPU(instance)                         │   │
│  │      if (tick % checkpoint_interval == 0)                │   │
│  │          write_checkpoint(instance, tick)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Host only:
  - Load WAD once at startup
  - Copy input buffers to GPU (async)
  - Read checkpoint buffers from GPU (async)
  - HTTP API for contestants (minimal)
```

## Contestant Control: The Checkpoint Dial

Contestants control their own simulation depth:

```json
POST /arena/doom/simulate
{
  "contestant_id": 42,
  "actions": [
    {"forward": 1, "turn": 0, "fire": false},
    {"forward": 1, "turn": 512, "fire": true},
    ...
  ],
  "horizon": 500,
  "checkpoint_every": 35
}
```

Response (from GPU checkpoint buffer):

```json
{
  "checkpoints": [
    {"tick": 0,   "health": 100, "kills": 0, "x": 1056, "y": -3616},
    {"tick": 35,  "health": 100, "kills": 0, "x": 1180, "y": -3520},
    {"tick": 70,  "health": 85,  "kills": 1, "x": 1280, "y": -3400},
    {"tick": 105, "health": 0,   "outcome": "DIED", "killer": "IMP"}
  ],
  "gpu_time_ms": 8,
  "ticks_simulated": 105
}
```

**Key**: LLMs can ask for deeper lookahead when facing hard decisions.

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Ticks/sec (single instance) | 500+ | 14× realtime |
| Ticks/sec (100 instances) | 50,000+ | Parallel |
| Checkpoint latency | <1ms | GPU→Host async |
| Memory per instance | ~2MB | Player + mobjs + thinkers |
| Total VRAM (100 instances) | ~200MB | Plenty of headroom |

## Files

```
doom/
├── source/              # id-Software/DOOM clone (read-only reference)
│   └── linuxdoom-1.10/  # The sacred texts
├── gpu/
│   ├── doom_types.cuh   # GPU-friendly structs (interleaved)
│   ├── doom_tick.cu     # P_Ticker port
│   ├── doom_player.cu   # P_PlayerThink port
│   ├── doom_mobj.cu     # P_MobjThinker port
│   ├── doom_map.cu      # Collision, BSP traversal
│   └── doom_main.cu     # Kernel entry, memory management
├── api/
│   └── server.py        # Minimal HTTP for contestants
├── Makefile             # nvcc -arch=sm_120
└── README.md
```

## The Port Strategy

1. **Phase 1**: Minimal viable tick
   - Port `P_PlayerThink()` - player movement only
   - Flat collision (no BSP yet)
   - Single instance proof of concept

2. **Phase 2**: Full game loop
   - Port `P_RunThinkers()` - monsters, projectiles
   - Port BSP traversal for collision
   - Multiple instances with interleaved memory

3. **Phase 3**: Deep simulation
   - Checkpoint system
   - Input ring buffer
   - 500+ ticks/sec target

4. **Phase 4**: Arena integration
   - Contestant API
   - Arena voice (narrator)
   - Replay generation

## Building

```bash
cd games/doom
make
./build/gpu_doom_test 100  # Test 100 instances
```

## License

DOOM source: id Software License (see source/LICENSE.TXT)
GPU port: MIT
