# Angband GPU Borg Port

**Status**: PLANNED
**Priority**: P0 (blocks Child 3)
**Created**: 2025-12-05

## Goal

Run 10,000 APWBorg instances in parallel on GPU. Feed metrics back to LLM contestants for config evolution.

## Current State

```
games/angband/
├── apwborg/           # 1.9MB C source (borg1-9.c)
├── configs/           # 8 AI-generated configs
├── gpu/               # WIP CUDA port
│   ├── borg_kernel.cu # Started
│   ├── borg_state.h   # State structures
│   └── build/         # Empty
└── harness/           # Python interface
```

## Architecture

### Phase 1: State Extraction (DONE-ish)
Extract borg decision state from C to GPU-friendly structs:
```c
typedef struct {
    // Position
    int depth, x, y;

    // Resources
    int hp, max_hp, sp, max_sp;
    int gold, food, light;

    // Inventory (simplified)
    int weapons[10], armor[10], potions[20];

    // Dungeon view (local)
    uint8_t map[21][79];  // What borg sees

    // Monsters in view
    Monster visible[50];

    // Config params (from borg.txt)
    BorgConfig config;
} BorgState;
```

### Phase 2: Decision Kernel
Port `borg_think()` to CUDA. The 600KB monster:
```cuda
__global__ void borg_think_kernel(
    BorgState* states,      // [N] instances
    Action* actions,        // [N] output actions
    int num_instances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    BorgState* s = &states[idx];

    // Priority cascade (from borg.txt)
    if (s->hp < s->config.flee_hp) {
        actions[idx] = find_escape_route(s);
        return;
    }

    if (has_valuable_loot(s) && s->depth < s->config.sell_depth) {
        actions[idx] = ACTION_RECALL;
        return;
    }

    // ... 50+ more decision branches
}
```

### Phase 3: World Simulation
Simplified Angband world on GPU:
```cuda
__global__ void simulate_turn_kernel(
    BorgState* states,
    Action* actions,
    DungeonLevel* levels,
    int num_instances
) {
    int idx = ...;

    Action a = actions[idx];
    BorgState* s = &states[idx];
    DungeonLevel* level = &levels[s->depth];

    switch (a.type) {
        case ACTION_MOVE:
            // Check collision, move, trigger traps
            break;
        case ACTION_ATTACK:
            // Combat resolution
            break;
        case ACTION_CAST:
            // Spell effects
            break;
        // ...
    }

    // Monster turns
    for (int m = 0; m < level->num_monsters; m++) {
        monster_act(&level->monsters[m], s);
    }
}
```

### Phase 4: Metrics Collection
After N turns, collect:
```c
typedef struct {
    int depth_reached;
    int gold_collected;
    int monsters_killed;
    int deaths;
    int turns_survived;
    int items_found[100];  // Histogram
} BorgMetrics;
```

## Implementation Plan

### Task 1: Compile Existing GPU Code
```bash
cd games/angband/gpu
make  # See if borg_kernel.cu compiles
```

### Task 2: Monster Database
Extract monster stats from Angband to GPU structs:
```cuda
// From monster.txt
__constant__ MonsterTemplate MONSTERS[500] = {
    {.name="Kobold", .hp=5, .ac=10, .damage="1d4", .speed=110},
    {.name="Orc", .hp=12, .ac=20, .damage="1d8", .speed=110},
    // ...
};
```

### Task 3: Simplified Combat
```cuda
__device__ int attack(BorgState* borg, Monster* mon) {
    int to_hit = borg->skill_thn + borg->weapon_bonus;
    int ac = mon->ac;

    if (rand() % 100 < (to_hit - ac)) {
        int damage = roll_dice(borg->weapon_dice);
        mon->hp -= damage;
        return damage;
    }
    return 0;
}
```

### Task 4: Config Loading
Parse borg.txt into BorgConfig struct:
```python
def load_borg_config(path: str) -> dict:
    """Parse borg.txt into config dict."""
    config = {}
    with open(path) as f:
        for line in f:
            if '=' in line:
                key, val = line.split('=', 1)
                config[key.strip()] = parse_value(val.strip())
    return config
```

### Task 5: Python Harness
```python
class AngbandGPU:
    def __init__(self, num_instances: int = 10000):
        self.num_instances = num_instances
        self.kernel = load_kernel("borg_kernel.cu")

    def run_batch(self, configs: list[dict], turns: int = 10000) -> list[BorgMetrics]:
        """Run N borg instances for T turns each."""
        states = self.init_states(configs)

        for t in range(turns):
            actions = self.kernel.borg_think(states)
            states = self.kernel.simulate_turn(states, actions)

            # Check for deaths, victories
            self.update_metrics(states)

        return self.collect_metrics()
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Instances | 10,000 parallel |
| Throughput | 1000 turns/sec total |
| Metrics | depth, gold, deaths, turns |
| Config params | 50+ tunable |

## Blockers

1. **borg_think() is 600KB of C** - Need to identify core decision paths
2. **Monster AI** - Simplified or faithful?
3. **Dungeon generation** - Pre-generate or procedural on GPU?

## References

- `games/angband/apwborg/borg.txt` - Config reference
- `games/angband/apwborg/borgread.txt` - Documentation
- `games/angband/gpu/borg_state.h` - Current state structs
- `games/common/interleaved.h` - Memory layout helpers

## Timeline

| Day | Task |
|-----|------|
| Fri | Compile existing, extract monster DB |
| Sat | Simplified combat kernel |
| Sun | Full decision tree port |
| Mon | Python harness + metrics |
| Tue | Integration with arena_session.py |
