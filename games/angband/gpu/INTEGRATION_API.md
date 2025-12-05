# Angband GPU Port - Phase 4 Integration API

This document defines the API contracts that `borg_kernel_v2.cu` expects from the three rule headers being created by parallel agents.

## 1. angband_combat.cuh

**Purpose**: Real Angband combat mechanics (hit/damage calculations)

### Required Functions

```cuda
// Calculate player's to-hit bonus
__device__ int calc_player_to_hit(int skill, int weapon_bonus);

// Execute a melee attack and return damage dealt
__device__ int melee_attack(
    int to_hit,           // Total to-hit bonus
    int deadliness,       // Damage multiplier
    int weapon_dd,        // Weapon dice count (e.g., 2 for 2d6)
    int weapon_ds,        // Weapon dice sides (e.g., 6 for 2d6)
    int player_level,     // Player level
    int monster_ac,       // Monster armor class
    curandState* rng      // RNG state for random rolls
);

// Monster attacks player - returns damage
__device__ int monster_attack(
    int monster_type,     // Monster race ID
    int player_ac,        // Player armor class
    int player_level,     // Player level (for resistance checks)
    curandState* rng      // RNG state
);

// Get monster's potential damage output (for danger calculation)
__device__ int get_monster_damage_potential(
    int monster_type,     // Monster race ID
    int player_ac         // Player AC (affects expected damage)
);
```

### Combat Formula References

- **To-hit**: Based on skill + weapon bonus vs monster AC
- **Damage**: `(weapon_dd)d(weapon_ds) * deadliness_multiplier`
- **Monster attacks**: Use monster blow structure (method, effect, damage dice)

### Data Structures Needed

Consider a compact monster race table with:
- Base AC
- Number of blows
- Blow damage dice (e.g., 1d6, 2d8)
- Blow effects (hurt, poison, etc.)

## 2. angband_monsters.cuh

**Purpose**: Real monster data and spawning logic

### Required Functions

```cuda
// Get appropriate monster type for dungeon depth
__device__ int get_monster_for_depth(
    int depth,            // Dungeon depth (1-127)
    curandState* rng      // RNG for selection
);

// Spawn monsters for a new level - returns monster count
__device__ int spawn_monsters_for_depth(
    int depth,                    // Dungeon depth
    int16_t* monster_x,           // Output: monster X positions (interleaved)
    int16_t* monster_y,           // Output: monster Y positions (interleaved)
    int16_t* monster_hp,          // Output: monster HP (interleaved)
    uint8_t* monster_type,        // Output: monster race IDs (interleaved)
    uint8_t* monster_awake,       // Output: awake flags (interleaved)
    uint8_t* dungeon_terrain,     // Input: dungeon terrain (for placement)
    int instance_id,              // Which instance
    int num_instances,            // Total instances (for interleaving)
    curandState* rng              // RNG state
);

// Get monster AC for combat
__device__ int get_monster_ac(int monster_type);

// Get XP value for killing this monster
__device__ int get_monster_xp(int monster_type, int depth);

// Wake monsters near player position
__device__ void wake_nearby_monsters(
    int16_t* monster_x,           // Monster positions (interleaved)
    int16_t* monster_y,
    uint8_t* monster_awake,       // Monster awake flags (interleaved)
    int monster_count,            // How many monsters to check
    int player_x,                 // Player position
    int player_y,
    int instance_id,
    int num_instances,
    curandState* rng
);
```

### Monster Race Table Format

Suggested compact format (can be a constant array):

```cuda
struct MonsterRace {
    uint8_t depth;          // Native depth
    uint8_t rarity;         // Spawn rarity (1-100)
    int16_t hp_dice;        // HP dice (e.g., 4d8 = dice=4, sides=8)
    int16_t hp_sides;
    uint8_t ac;             // Armor class
    uint8_t speed;          // Monster speed
    uint8_t num_blows;      // Number of attacks
    uint8_t blow_dd[4];     // Damage dice for each blow
    uint8_t blow_ds[4];     // Damage sides for each blow
    uint16_t xp_base;       // Base XP value
};

// Example: Lookup table with ~50-100 common monsters
__constant__ MonsterRace MONSTER_RACES[NUM_MONSTER_RACES];
```

### Spawning Logic

- Number of monsters scales with depth: `base_count + depth / 5`
- Select monsters appropriate for depth ±5 levels
- Place on walkable terrain, not on player

## 3. angband_dungeon.cuh

**Purpose**: Real dungeon generation

### Required Functions

```cuda
// Generate a complete dungeon level
__device__ void generate_level(
    uint8_t* dungeon_terrain,     // Output: terrain grid (interleaved)
    int depth,                    // Dungeon depth (affects room types)
    int* player_x,                // Output: player start X
    int* player_y,                // Output: player start Y
    int* stairs_x,                // Output: down stairs X
    int* stairs_y,                // Output: down stairs Y
    int instance_id,              // Which instance
    int num_instances,            // Total instances (for interleaving)
    curandState* rng              // RNG state
);

// Check if terrain is walkable
__device__ bool is_walkable(
    uint8_t* dungeon_terrain,     // Terrain grid (interleaved)
    int x, int y,                 // Position to check
    int instance_id,
    int num_instances
);

// Teleport player to random walkable location
__device__ void teleport_player(
    int* player_x,                // Input/Output: player X
    int* player_y,                // Input/Output: player Y
    uint8_t* dungeon_terrain,     // Terrain grid
    int depth,                    // Current depth (affects teleport range)
    int instance_id,
    int num_instances,
    curandState* rng
);
```

### Terrain Types

Suggested encoding (uint8_t per cell):

```cuda
#define TERRAIN_WALL        0
#define TERRAIN_FLOOR       1
#define TERRAIN_DOOR        2
#define TERRAIN_STAIRS_DOWN 3
#define TERRAIN_STAIRS_UP   4
#define TERRAIN_RUBBLE      5
#define TERRAIN_LAVA        6
#define TERRAIN_WATER       7
```

### Generation Algorithm

Simplified options:
1. **Rooms + corridors**: Generate rectangular rooms, connect with corridors
2. **Cellular automata**: Iterative cave generation (fast on GPU)
3. **Prefabs**: Store small level templates, stitch together

Recommended: Start with simple rooms + corridors for correctness, optimize later.

### Interleaved Grid Access

The dungeon grid is interleaved across instances:

```cuda
// Grid layout: grid[y * width + x][instance]
int cell_idx = y * DUNGEON_WIDTH + x;
uint8_t terrain = dungeon_terrain[cell_idx * num_instances + instance_id];
```

Helper macros in `interleaved.h`:
- `IGET(grid, cell_idx, instance_id, num_instances)` - read cell
- `ISET(grid, cell_idx, instance_id, num_instances, value)` - write cell

Or use `grid_get_interleaved()` / `grid_set_interleaved()` functions.

## Memory Layout

All arrays use **interleaved layout** for coalesced memory access:

```
Instance 0: [elem0] [elem1] [elem2] ...
Instance 1: [elem0] [elem1] [elem2] ...
Instance 2: [elem0] [elem1] [elem2] ...

Memory order: I0_E0, I1_E0, I2_E0, I0_E1, I1_E1, I2_E1, ...
```

When 32 threads (warp) access the same element index, they hit consecutive memory addresses → perfect coalescing.

## Testing Strategy

### Unit Tests (per header)

Each header should be testable independently:

1. **Combat**: Test hit calculation, damage rolls with fixed RNG
2. **Monsters**: Test depth-appropriate spawning, XP values
3. **Dungeon**: Test that generated levels have floors, stairs, no disconnected regions

### Integration Test

`borg_kernel_v2.cu` includes `--verify` mode:

```bash
./borg_sim --verify 12345
```

Runs 100 instances with fixed seed, prints first 10 results. Should be reproducible across runs.

Expected outcomes with REAL rules (unlike v1):
- **Death rate**: 30-70% depending on config (not 0% like fake version)
- **Average depth**: 3-15 depending on config (not stuck at 1)
- **Max depth**: 20-50+ for lucky runs
- **Win rate**: <1% (Morgoth at depth 100 is HARD)

## Implementation Order

Suggested parallel development:

1. **Start with stubs**: Create headers with stub implementations first
   - Combat: Always hit for 10 damage
   - Monsters: Always spawn white mice (easy monster)
   - Dungeon: All floors, no walls
2. **Test integration**: Verify `borg_kernel_v2.cu` compiles and runs
3. **Implement incrementally**:
   - Combat formulas
   - Monster selection
   - Room generation
4. **Verify correctness**: Use `--verify` mode to check results

## Performance Notes

- Keep monster race table in `__constant__` memory (64KB limit)
- Minimize divergence in warp (use lookup tables over conditionals)
- Dungeon generation is one-time cost per level (not per turn)
- Combat is hottest path - optimize hit/damage calculation

## Questions?

If the API needs adjustment, coordinate through the integration layer. The `borg_kernel_v2.cu` can be updated as needed.
