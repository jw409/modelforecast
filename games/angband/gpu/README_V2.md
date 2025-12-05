# Angband GPU Borg - Phase 4 Integration (V2)

## Overview

This is the **integration layer** for Phase 4 of the Angband GPU port. `borg_kernel_v2.cu` connects the borg AI with real Angband game rules through three modular headers:

- `angband_combat.cuh` - Combat formulas (hit/damage calculations)
- `angband_monsters.cuh` - Monster data and spawning
- `angband_dungeon.cuh` - Dungeon generation

## Architecture

```
┌─────────────────┐
│ borg_kernel_v2  │  Integration layer (decision + execution)
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼────┐  │
│Combat │ │Monster│ │Dungeon│  │
│Rules  │ │ Data  │ │  Gen  │  │
└───────┘ └───────┘ └───────┘  │
                                │
                          ┌─────▼──────┐
                          │ Borg State │
                          │(Interleaved)│
                          └────────────┘
```

## Current Status

### Working (Stub Implementations)
- ✅ **Compilation**: All headers compile without errors
- ✅ **RNG**: cuRAND initialized per instance
- ✅ **Dungeon generation**: Simple rooms + corridors
- ✅ **Monster spawning**: Depth-appropriate selection from 24 monster types
- ✅ **Combat**: Basic hit/damage with AC reduction
- ✅ **Depth progression**: Borgs descending to depth 15-25 on average
- ✅ **Verification mode**: Reproducible runs with fixed seed

### Needs Real Implementation
- ⚠️ **Combat formulas**: Using simplified hit/damage (not authentic Angband)
- ⚠️ **Monster AI**: Basic distance-based waking (no noise/stealth)
- ⚠️ **Dungeon variety**: Only rectangular rooms (no vaults, pits, caverns)
- ⚠️ **Death system**: Too easy - 100% survival rate (combat needs tuning)

## Build & Run

### Build
```bash
make                    # Build both v1 and v2
make clean && make      # Clean rebuild
```

### Run
```bash
make run_v2            # Run v2 with 10K instances, 1K turns
make verify_v2         # Verification mode (100 instances, fixed seed)
make benchmark_v2      # Performance benchmark
```

### Manual Invocation
```bash
# Normal run
./build/borg_sim_v2 10000 1000

# Verification mode (reproducible)
./build/borg_sim_v2 --verify 12345

# Custom verification seed
./build/borg_sim_v2 --verify 54321
```

## Results Comparison: V1 vs V2

### V1 (Fake Rules)
```
Config       Alive%    Dead%     Win% AvgDepth MaxDepth
Aggro        100.0%     0.0%     0.0%      1.0        1
Speed        100.0%     0.0%     0.0%      1.0        1
```
**Problem**: Stuck at depth 1, no real progression

### V2 (Stub Real Rules)
```
Config       Alive%    Dead%     Win% AvgDepth MaxDepth
Aggro        100.0%     0.0%     0.0%     12.1       20
Speed        100.0%     0.0%     0.0%     12.7       20
Tank         100.0%     0.0%     0.0%     13.1       25
Scummer      100.0%     0.0%     0.0%     15.3       22
```
**Improvement**: Real depth progression! Descending 10-15 levels on average.

### Expected with Full Real Rules
```
Config       Alive%    Dead%     Win% AvgDepth MaxDepth
Aggro         30.0%    70.0%     0.0%      8.5       35
Speed         45.0%    55.0%     0.0%     12.3       50
Tank          60.0%    40.0%     0.0%     15.7       45
```
**Target**: 30-70% death rate, depth 8-15 average, occasional deep runs to 40-50

## API Documentation

See `INTEGRATION_API.md` for complete function signatures and contracts.

### Quick Reference

#### Combat (angband_combat.cuh)
```cuda
__device__ int melee_attack(to_hit, deadliness, weapon_dd, weapon_ds,
                             player_level, monster_ac, rng);
__device__ int monster_attack(monster_type, player_ac, player_level, rng);
```

#### Monsters (angband_monsters.cuh)
```cuda
__device__ int get_monster_for_depth(depth, rng);
__device__ int spawn_monsters_for_depth(depth, monster_arrays..., rng);
__device__ int get_monster_ac(monster_type);
__device__ int get_monster_xp(monster_type, depth);
```

#### Dungeon (angband_dungeon.cuh)
```cuda
__device__ void generate_level(terrain, depth, player_pos, stairs_pos, ..., rng);
__device__ bool is_walkable(terrain, x, y, instance_id, num_instances);
__device__ void teleport_player(player_pos, terrain, depth, ..., rng);
```

## Memory Layout

All state uses **interleaved layout** for GPU coalescing:

```
Instance 0: [HP] [X] [Y] [Depth] ...
Instance 1: [HP] [X] [Y] [Depth] ...
Instance 2: [HP] [X] [Y] [Depth] ...

Memory order: HP0, HP1, HP2, X0, X1, X2, Y0, Y1, Y2, ...
```

When warp threads access element N, they hit consecutive addresses → perfect coalescing.

Use macros from `interleaved.h`:
- `IGET(array, row, instance_id, num_instances)` - read
- `ISET(array, row, instance_id, num_instances, value)` - write

## Verification Mode

Verification mode enables reproducibility testing:

```bash
./build/borg_sim_v2 --verify 12345
```

**Purpose**: Ensure results are deterministic across:
- Different runs (same machine)
- Different GPUs (if RNG is consistent)
- Code changes (regression testing)

**Output**: First 10 instances with full state:
```
ID  Alive  Depth  Level  HP
 0    YES     10      1  100
 1    YES      1      1  100
 2    YES      4      1  100
```

**Usage**: Run before/after changes to verify behavior unchanged.

## Performance

With stub implementations:
- **Throughput**: 4-5M instance-turns/sec
- **100 instances, 1K turns**: ~20-25ms
- **10K instances, 1K turns**: ~2-3 seconds

Expected with full real rules:
- **Throughput**: 2-3M instance-turns/sec (more complex computation)
- **10K instances, 1K turns**: ~3-5 seconds

## Next Steps

### Priority 1: Combat Realism
Replace stub combat with authentic Angband formulas:
1. BTH (base to-hit) calculation from player-attack.c
2. Deadliness formula with critical hits
3. Monster blow structure (method + effect + damage)
4. AC damage reduction from adjust_dam_armor()

**Impact**: Death rate should rise to 30-70%

### Priority 2: Monster Expansion
Expand monster table from 24 to 100+ races:
1. Extract from monster.txt (Angband data files)
2. Add blow structures (1d6+poison, 2d8+fire, etc.)
3. Implement rarity-based spawning
4. Add unique monsters (Sauron, Morgoth)

**Impact**: More varied gameplay, deeper progression

### Priority 3: Dungeon Variety
Add advanced generation features:
1. Vaults (special rooms with treasure/guardians)
2. Pits (themed monster groups)
3. Caverns (organic cave systems)
4. Labyrinth generation

**Impact**: Strategic depth, interesting level layouts

## Testing Strategy

### Unit Tests (per header)
Test each header independently:
```bash
# Combat: Fixed RNG, verify damage ranges
test_combat_formulas()

# Monsters: Verify depth-appropriate spawning
test_monster_selection()

# Dungeon: Verify connectivity, stair placement
test_dungeon_generation()
```

### Integration Tests
```bash
# Reproducibility
./build/borg_sim_v2 --verify 12345 > run1.txt
./build/borg_sim_v2 --verify 12345 > run2.txt
diff run1.txt run2.txt  # Should be identical

# Performance regression
time make benchmark_v2  # Compare against baseline
```

### Balance Tests
```bash
# Run 100K instances, analyze outcomes
./build/borg_sim_v2 100000 1000 > results.txt

# Check for reasonable distributions:
# - Death rate: 30-70%
# - Average depth: 8-15
# - Max depth: 30-50
# - Win rate: <0.1%
```

## Known Issues

1. **100% survival**: Stub combat too easy
   - **Fix**: Implement real combat formulas

2. **No level-up**: XP system tracks but doesn't increase power
   - **Fix**: Implement stat increases on level-up

3. **Monster clumping**: All spawn at random positions
   - **Fix**: Respect terrain, use group spawning

4. **Corridor connectivity**: Simple horizontal corridor may leave rooms isolated
   - **Fix**: Implement proper room graph + corridor carving

## File Structure

```
gpu/
├── borg_kernel.cu          # V1 (fake rules, baseline)
├── borg_kernel_v2.cu       # V2 (integration layer) ← YOU ARE HERE
├── borg_state.h            # Shared state structure
├── angband_combat.cuh      # Combat rules (stub)
├── angband_monsters.cuh    # Monster data (stub)
├── angband_dungeon.cuh     # Dungeon generation (stub)
├── INTEGRATION_API.md      # API contracts for headers
├── README_V2.md            # This file
└── Makefile                # Build system
```

## Contributing

When enhancing the headers:

1. **Read the API**: See `INTEGRATION_API.md` for contracts
2. **Test incrementally**: Compile after each change
3. **Verify results**: Use `--verify` mode before/after
4. **Check balance**: Run 100K instances, analyze death rate/depth

**Coordinate changes**: If API needs adjustment, update both the header AND `borg_kernel_v2.cu`.

## References

- Original Angband source: `/home/jw/dev/game1/external/angband-arena/angband/src/`
- Combat system: `player-attack.c`, `mon-attack.c`
- Monster data: `monster.h`, `list-mon-*.h`
- Generation: `generate.c`, `gen-cave.c`, `gen-room.c`

## License

Based on Angband 4.2.x (GPLv2) and APWBorg (MIT).
