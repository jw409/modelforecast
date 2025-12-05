# Agent Handoff: Implementing Real Angband Rules

## For Combat Agent

**Your task**: Replace stub combat in `angband_combat.cuh` with authentic Angband formulas.

### Reference Files
```
/home/jw/dev/game1/external/angband-arena/angband/src/player-attack.c
/home/jw/dev/game1/external/angband-arena/angband/src/mon-attack.c
/home/jw/dev/game1/external/angband-arena/angband/src/obj-util.c
```

### Functions to Implement

1. **`calc_player_to_hit()`**
   - Reference: `chance_of_melee_hit()` in player-attack.c
   - Formula: BTH (base to-hit) = 3 * level + weapon skill
   - Hit chance = BTH + to_hit_bonus - monster_ac

2. **`melee_attack()`**
   - Reference: `py_attack()` in player-attack.c
   - Steps:
     - Roll to hit (d20 or percentage check)
     - If hit, roll weapon damage: `(weapon_dd)d(weapon_ds)`
     - Apply deadliness multiplier
     - Check for critical hit (complex table in player-attack.c)
     - Return total damage

3. **`monster_attack()`**
   - Reference: `make_attack_normal()` in mon-attack.c
   - Steps:
     - Lookup monster blow structure (method + effect + dice)
     - Roll to hit vs player AC
     - If hit, roll damage dice
     - Apply AC reduction: `adjust_dam_armor()`
     - Return damage (may have side effects like poison)

### Expected Outcome
Death rate should rise from 0% to 30-70% once real combat is implemented.

### Testing
```bash
# Compile
make clean && make

# Verify
make verify_v2

# Check death rate
./build/borg_sim_v2 100000 1000 | grep "SUMMARY"
# Expected: 30-70% dead
```

---

## For Monster Agent

**Your task**: Expand monster table in `angband_monsters.cuh` from 24 to 100+ races.

### Reference Files
```
/home/jw/dev/game1/external/angband-arena/angband/lib/gamedata/monster.txt
/home/jw/dev/game1/external/angband-arena/angband/src/monster.h
/home/jw/dev/game1/external/angband-arena/angband/src/list-mon-race-flags.h
```

### Data to Extract

From `monster.txt`:
```
name:kobold
base:kobold
glyph:k:slate
pain:1
depth:1
rarity:1
speed:110
hp:4d4
ac:22
blow:HIT:HURT:1d5
flags:MALE | EVIL | OPEN_DOOR
desc:He is a small, dog-headed humanoid that is considered a pest by most other races.
```

### Monster Race Structure (Expand)
```cuda
struct MonsterRace {
    uint8_t depth;
    uint8_t rarity;
    uint8_t hp_dice, hp_sides;
    uint8_t ac;
    uint8_t speed;
    uint8_t num_blows;

    // Add blow structure:
    uint8_t blow_method[4];  // HIT, BITE, CLAW, etc.
    uint8_t blow_effect[4];  // HURT, POISON, CONFUSE, etc.
    uint8_t blow_dd[4];      // Damage dice
    uint8_t blow_ds[4];      // Damage sides

    uint16_t xp_base;
    uint16_t flags;          // RF_EVIL, RF_UNDEAD, etc.
};
```

### Priority Monsters

**Early game (depth 1-20)**:
- Kobolds, orcs, wolves, giant insects
- Trolls, ogres, wargs
- Young dragons

**Mid game (depth 21-50)**:
- Giants, vampires, liches
- Ancient dragons
- Demons

**Late game (depth 51-100)**:
- Greater demons, ancient wyrms
- Unique bosses: Sauron, Morgoth

### Testing
```bash
# Verify monster variety
./build/borg_sim_v2 10000 1000 | grep "AvgDepth"
# Should see deeper average depths with appropriate monsters
```

---

## For Dungeon Agent

**Your task**: Add vaults, pits, and caverns to `angband_dungeon.cuh`.

### Reference Files
```
/home/jw/dev/game1/external/angband-arena/angband/src/gen-cave.c
/home/jw/dev/game1/external/angband-arena/angband/src/gen-room.c
/home/jw/dev/game1/external/angband-arena/angband/src/gen-util.c
/home/jw/dev/game1/external/angband-arena/angband/lib/gamedata/vault.txt
```

### Current Implementation
- Simple rectangular rooms (3-5 per level)
- Basic corridor connecting rooms

### Features to Add

1. **Room Types**
   - `build_room_rectangle()` - Current implementation
   - `build_room_circular()` - Round rooms
   - `build_room_cross()` - Cross-shaped rooms
   - `build_room_large()` - Big open areas

2. **Vaults**
   - Small vaults (10x10) - Extra treasure/monsters
   - Medium vaults (20x20) - Themed (undead, dragons, etc.)
   - Large vaults (30x30) - Boss encounters

3. **Pits**
   - Monster pits - Dense spawn of single type
   - Nested pits - Concentric rings of monsters

4. **Caverns**
   - Cellular automata generation
   - Organic cave systems
   - Lakes (water/lava)

### Vault Format (from vault.txt)
```
name:Test vault
type:Lesser vault
rating:10
rows:5
columns:7
D:###+#+###
D:#.......#
D:#.......#
D:#.......#
D:#########
```

### Implementation Strategy

**Phase 1**: Improve corridor connectivity
```cuda
// Connect all rooms with proper corridors
for each room pair:
    carve_corridor(room1_center, room2_center)
```

**Phase 2**: Add room variety
```cuda
int room_type = curand(rng) % 5;
switch(room_type) {
    case 0: build_room_rectangle(...);
    case 1: build_room_circular(...);
    case 2: build_room_cross(...);
    ...
}
```

**Phase 3**: Add vaults (depth-dependent)
```cuda
if (depth > 20 && curand(rng) % 100 < 10) {  // 10% chance
    place_vault(vault_template, x, y);
}
```

### Testing
```bash
# Visual verification (future: render to image)
# For now, check that borgs reach deeper levels
./build/borg_sim_v2 100000 1000 | grep "MaxDepth"
# Should see occasional runs to depth 40-50
```

---

## Common Guidelines (All Agents)

### 1. Memory Layout
All arrays are **interleaved** for GPU coalescing:
```cuda
// Read
value = IGET(array, row, instance_id, num_instances);

// Write
ISET(array, row, instance_id, num_instances, value);
```

### 2. RNG Usage
Each instance has its own `curandState*`:
```cuda
int random_value = curand(rng) % max;
int dice_roll = curand(rng) % sides + 1;
```

### 3. Constant Memory
Use `__constant__` for lookup tables (64KB limit):
```cuda
__constant__ MonsterRace MONSTER_RACES[256];
__constant__ VaultTemplate VAULT_TEMPLATES[100];
```

### 4. Testing Workflow
```bash
# 1. Edit header file
vim angband_combat.cuh

# 2. Compile
make clean && make

# 3. Verify reproducibility
make verify_v2 > before.txt
# (make change)
make clean && make
make verify_v2 > after.txt
diff before.txt after.txt

# 4. Check balance (100K instances)
./build/borg_sim_v2 100000 1000 > results.txt

# 5. Analyze
grep "SUMMARY" results.txt
grep "AvgDepth" results.txt
```

### 5. Performance Considerations
- Minimize divergence (all threads take same path)
- Use lookup tables over complex conditionals
- Keep hot paths (combat) as simple as possible
- Complex generation (dungeons) happens once per level

### 6. Debugging
```cuda
// Device-side printf (use sparingly)
if (instance_id == 0 && threadIdx.x == 0) {
    printf("Debug: value=%d\n", value);
}
```

---

## Integration Points

### How Your Code is Called

**Combat functions** (called every combat turn):
```cuda
// From borg_execute_kernel(), action BORG_ACTION_ATTACK
int dmg = melee_attack(to_hit, deadliness, weapon_dd, weapon_ds,
                       player_level, monster_ac, rng);
int retaliation = monster_attack(monster_type, player_ac, player_level, rng);
```

**Monster functions** (called on level generation):
```cuda
// From init_dungeons_kernel() and BORG_ACTION_DESCEND
int count = spawn_monsters_for_depth(depth, monster_arrays..., rng);
```

**Dungeon functions** (called on level generation):
```cuda
// From init_dungeons_kernel() and BORG_ACTION_DESCEND
generate_level(terrain, depth, &px, &py, &sx, &sy, instance_id, num_instances, rng);
```

---

## API Contracts

See `INTEGRATION_API.md` for complete function signatures.

**Key principle**: Don't change function signatures without updating `borg_kernel_v2.cu`.

If you need to add new parameters:
1. Update function signature in header
2. Update call site in `borg_kernel_v2.cu`
3. Update `INTEGRATION_API.md` documentation

---

## Expected Results

### Current (Stub)
```
Config       Alive%    Dead%     Win% AvgDepth MaxDepth
Aggro        100.0%     0.0%     0.0%     12.1       20
```

### Target (Real)
```
Config       Alive%    Dead%     Win% AvgDepth MaxDepth
Aggro         30.0%    70.0%     0.0%      8.5       35
Speed         45.0%    55.0%     0.0%     12.3       50
Tank          60.0%    40.0%     0.0%     15.7       45
Scummer       55.0%    45.0%     0.5%     18.2       80
```

Key metrics:
- **Death rate**: 30-70% (varies by config)
- **Average depth**: 8-18
- **Max depth**: 30-80 (occasional deep runs)
- **Win rate**: <1% (Morgoth is at depth 100, very hard)

---

## Questions?

If you need clarification:
1. Read `INTEGRATION_API.md` for API details
2. Read `README_V2.md` for architecture overview
3. Check `PHASE4_COMPLETE.md` for what's already done
4. Look at stub implementation in your header for structure

**Communication**: If API needs to change, coordinate with integration layer (`borg_kernel_v2.cu`).

---

## File Locations

```
Project root: /home/jw/dev/modelforecast/games/angband/gpu/

Your files:
├── angband_combat.cuh      ← Combat agent
├── angband_monsters.cuh    ← Monster agent
├── angband_dungeon.cuh     ← Dungeon agent

Integration:
├── borg_kernel_v2.cu       ← Calls your functions

Reference (Angband source):
/home/jw/dev/game1/external/angband-arena/angband/src/
├── player-attack.c         ← Combat formulas
├── mon-attack.c            ← Monster attacks
├── monster.h               ← Monster structures
├── gen-cave.c              ← Cave generation
├── gen-room.c              ← Room building
└── ...

Data files:
/home/jw/dev/game1/external/angband-arena/angband/lib/gamedata/
├── monster.txt             ← Monster definitions
├── vault.txt               ← Vault templates
└── ...
```

---

## Success Criteria

Your implementation is ready when:
- ✅ Compiles without errors
- ✅ Verification mode produces reproducible results
- ✅ 100K instance run shows expected death rate/depth distribution
- ✅ Performance <2x slower than stub (acceptable overhead)

Good luck! 🎮
