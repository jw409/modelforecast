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

### Phase 2: Decision Kernel (DIRECT PORT - NO SIMPLIFICATION)

**Philosophy**: 600KB of C can be optimized OR IT CAN JUST BE PORTED TO GPU.
We choose: **JUST PORT IT**.

Each GPU thread = one complete borg brain. Parallelism is INSTANCES, not LOGIC.

```cuda
// Don't simplify. Don't optimize. JUST PORT.
// 600KB of borg logic, verbatim on GPU
__global__ void borg_think_kernel(
    BorgState* states,      // [N] instances (10,000 borgs)
    Action* actions,        // [N] output actions
    int num_instances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    BorgState* s = &states[idx];

    // ALL priority cascades from borg9-1.c
    // Every conditional, every heuristic, every edge case
    // Port line-by-line, preserve behavior exactly

    // This is NOT a simplification - it's the FULL borg brain
    // running 10,000 times in parallel

    actions[idx] = borg_think_full(s);  // The whole 600KB
}

// The complete borg decision tree - all 50+ branches
__device__ Action borg_think_full(BorgState* s) {
    // Phase 1: Emergency responses (flee, heal, teleport)
    // Phase 2: Combat decisions (attack, buff, position)
    // Phase 3: Exploration (pathfinding, door handling)
    // Phase 4: Item management (identify, enchant, sell)
    // Phase 5: Town behavior (shopping, restocking)
    // Phase 6: Long-term planning (stat goals, depth targets)

    // EVERY decision from the original C code
    // No shortcuts. No "simplified combat".
}
```

### Phase 3: World Simulation (FAITHFUL + PROCEDURAL)

**Monster AI**: FAITHFUL to original Angband. No simplification.
**Dungeon Generation**: PROCEDURAL ON GPU. Each instance gets unique levels.

```cuda
__global__ void simulate_turn_kernel(
    BorgState* states,
    Action* actions,
    DungeonLevel* levels,
    uint32_t* rng_states,      // Per-instance RNG for procedural gen
    int num_instances
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    Action a = actions[idx];
    BorgState* s = &states[idx];
    DungeonLevel* level = &levels[s->depth * num_instances + idx];  // Per-instance level

    // FULL action resolution - faithful to Angband
    switch (a.type) {
        case ACTION_MOVE:
            // Full movement: collision, trap triggers, door opening
            // Includes: terrain effects, confusion, fear
            break;
        case ACTION_ATTACK:
            // FAITHFUL combat: to-hit, AC, damage dice, criticals
            // Monster resistances, immunities, vulnerabilities
            break;
        case ACTION_CAST:
            // Full spell system: mana cost, failure rate, effects
            // Beam vs bolt vs ball, resistances
            break;
        case ACTION_USE_ITEM:
            // Wands, staves, scrolls, potions, rods
            // Recharge, identify, cursed items
            break;
        // ALL other actions from Angband
    }

    // FAITHFUL Monster AI - each monster acts intelligently
    for (int m = 0; m < level->num_monsters; m++) {
        Monster* mon = &level->monsters[m];
        // Full monster behavior: pursuit, flee at low HP, spellcasting
        // Pathfinding (A* or equivalent), LOS checks
        // Monster-specific AI flags from monster.txt
        monster_act_faithful(mon, s, level, &rng_states[idx]);
    }
}

// Procedural dungeon generation ON GPU
__device__ void generate_dungeon_level(
    DungeonLevel* level,
    int depth,
    uint32_t* rng
) {
    // Full Angband dungeon gen: rooms, corridors, vaults
    // Themed levels, special rooms, permanent walls
    // Stairs, traps, treasure distribution
    // Monster placement based on depth and OOD rolls
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

### Task 3: FAITHFUL Combat (Full Angband Rules)
```cuda
__device__ int attack(BorgState* borg, Monster* mon, uint32_t* rng) {
    // FULL Angband combat from melee.c and ranged.c

    // 1. To-hit calculation (skill + bonuses - penalties)
    int to_hit = borg->skill_thn;
    to_hit += borg->weapon_bonus;
    to_hit -= mon->ac;
    to_hit -= distance_penalty(borg, mon);  // Ranged only

    // 2. Hit determination with critical chance
    int roll = xorshift32(rng) % 100;
    bool hit = roll < to_hit;
    bool critical = hit && (roll < to_hit / 5);

    if (!hit) return 0;

    // 3. Damage calculation (FAITHFUL dice + slays + brands)
    int damage = roll_dice(borg->weapon_dice, rng);

    // Slay multipliers from object.txt
    if (mon->flags & RF_EVIL && borg->weapon_slays & SLAY_EVIL) damage = damage * 2;
    if (mon->flags & RF_UNDEAD && borg->weapon_slays & SLAY_UNDEAD) damage = damage * 3;
    if (mon->flags & RF_DEMON && borg->weapon_slays & SLAY_DEMON) damage = damage * 3;
    if (mon->flags & RF_DRAGON && borg->weapon_slays & SLAY_DRAGON) damage = damage * 3;
    // ... ALL slays from the original

    // Brand effects (fire, cold, acid, lightning, poison)
    damage = apply_brands(damage, borg, mon, rng);

    // Critical multiplier
    if (critical) damage = damage * (2 + xorshift32(rng) % 3);

    // Monster resistance/immunity
    damage = apply_resistances(damage, mon);

    mon->hp -= damage;
    return damage;
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

1. **borg_think() is 600KB of C** - ~~Need to identify core decision paths~~ **RESOLVED: JUST PORT IT**
2. **Monster AI** - ~~Simplified or faithful?~~ **RESOLVED: FAITHFUL**
3. **Dungeon generation** - ~~Pre-generate or procedural on GPU?~~ **RESOLVED: PROCEDURAL ON GPU**

## Design Decisions (LOCKED)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Port vs Optimize borg.c | **DIRECT PORT** | 600KB can be ported, parallelism is instances not logic |
| Monster AI | **FAITHFUL** | Full Angband monster behavior, no shortcuts |
| Dungeon Gen | **PROCEDURAL ON GPU** | Each instance gets unique procedural levels |
| Memory Layout | **Per-instance levels** | `levels[depth * N + idx]` gives each borg its own dungeon |

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
