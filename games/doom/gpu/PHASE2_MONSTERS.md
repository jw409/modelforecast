# DOOM GPU - Phase 2: Monsters and Combat

## Overview

Phase 2 adds monsters that can kill the player, and the player can kill monsters. This creates a functional combat loop for the GPU DOOM simulation.

## Architecture

### Files

1. **doom_monsters.cuh** - Monster type definitions and stats
2. **doom_combat.cuh** - Combat system (AI, attacks, damage)
3. **doom_main.cu** - Main kernel with monster integration

### Monster Types

Six monster types from original DOOM E1:

```cpp
MT_ZOMBIE      // Former Human - 20 HP, pistol (3-15 damage)
MT_SHOTGUY     // Shotgun Guy - 30 HP, shotgun (3-15 damage)
MT_IMP         // Imp - 60 HP, melee/fireball (3-24 damage)
MT_DEMON       // Pinky - 150 HP, melee only (4-40 damage)
MT_CACODEMON   // Cacodemon - 400 HP, fireball (5-40 damage)
MT_BARON       // Baron of Hell - 1000 HP, fireball (8-64 damage)
```

### Memory Layout

Interleaved arrays for coalesced GPU access:

```cuda
// Monster state (64 monsters per instance)
__device__ fixed_t* d_monster_x;        // [MAX_MONSTERS * num_instances]
__device__ fixed_t* d_monster_y;
__device__ int32_t* d_monster_health;
__device__ uint8_t* d_monster_type;
__device__ uint8_t* d_monster_alive;
__device__ int16_t* d_monster_target_idx;  // 0 = player, -1 = none
__device__ int16_t* d_monster_movecount;
__device__ int16_t* d_monster_reactiontime;

// Player additions
__device__ int16_t* d_player_kills;     // Track kill count
```

Access pattern: `monster_x[monster_id * num_instances + instance_id]`

## Combat System

### Monster AI (P_MonsterThink)

Simplified from original `A_Chase` function:

1. **Reaction Time** - Cooldown after attacks (8-16 ticks)
2. **Target Acquisition** - Always target player (if alive)
3. **Line of Sight** - Simplified distance check (Phase 2)
4. **Melee Attack** - If within melee range (64 units)
5. **Ranged Attack** - If within missile range (2048 units), 25% chance/tick
6. **Movement** - Move directly toward player (no pathfinding)

```cuda
__device__ void P_MonsterThink(int monster_id, int instance_id, int tick, int num_instances) {
    // Skip if dead or in cooldown
    // Check target (player) is alive
    // Check line of sight
    // Calculate distance
    // Try melee attack (if in range)
    // Try ranged attack (if in range, random chance)
    // Move toward player (simplified)
}
```

### Player Attack (P_PlayerAttack)

Hitscan weapon (pistol):

1. Check `BT_ATTACK` button in TicCmd
2. Find nearest monster in front of player (dot product test)
3. Apply damage 5-15 (random 1-3 * 5)

```cuda
if (cmd.buttons & BT_ATTACK) {
    P_PlayerAttack(instance_id, tick, num_instances, angle, x, y);
}
```

### Damage System

**Player Damage:**
- Armor absorbs 1/3 of damage
- Health drops to 0 → player dead (`d_player_alive = 0`)

**Monster Damage:**
- Health drops to 0 → monster dead (`d_monster_alive = 0`)
- Player gets kill credit (`d_player_kills++`)

## Simplifications (Phase 2)

These will be added in Phase 3:

1. **No BSP collision** - Monsters walk through walls
2. **No projectiles** - Ranged attacks are instant hitscan
3. **No states** - No animation/sprite states
4. **No sound** - No sound propagation/alerts
5. **Direct movement** - No pathfinding or door opening
6. **No multiplayer targeting** - Monsters only target player 0

## Integration with Main Kernel

```cuda
__global__ void doom_simulate(int num_instances, int num_ticks, int checkpoint_interval) {
    int instance_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Spawn monsters at level start
    if (tick == 0) {
        InitMonsters(instance_id, num_instances);
    }

    for (int tick = 0; tick < num_ticks; tick++) {
        // Player thinking (movement + shooting)
        P_PlayerThink_GPU(instance_id, tick, num_instances);

        // Monster AI (all monsters think each tick)
        for (int m = 0; m < MAX_MONSTERS; m++) {
            P_MonsterThink(m, instance_id, tick, num_instances);
        }

        // Checkpoints, sync, etc.
    }
}
```

## Monster Spawning (E1M1 Example)

E1M1 has ~50 monsters:

```cuda
void InitMonsters(int instance_id, int num_instances) {
    // Spawn zombies near start
    SpawnMonster(0, instance_id, 1000 << FRACBITS, -3800 << FRACBITS, MT_ZOMBIE);
    SpawnMonster(1, instance_id, 1100 << FRACBITS, -3700 << FRACBITS, MT_ZOMBIE);

    // Spawn imps in courtyard
    SpawnMonster(2, instance_id, 800 << FRACBITS, -2400 << FRACBITS, MT_IMP);

    // Spawn pinky in dark room
    SpawnMonster(3, instance_id, 1600 << FRACBITS, -3000 << FRACBITS, MT_DEMON);

    // ... etc for all 50 monsters
}
```

## Testing

Expected behavior:

1. **Players should die sometimes** - Monsters attack and deal damage
2. **Monsters should die when shot** - Player pistol kills zombies in 2-4 shots
3. **Kill count increases** - d_player_kills tracks successful kills
4. **Checkpoints show combat** - Health decreases, kills increase over time

## Performance

With 1024 instances and 64 monsters per instance:
- Total monsters: 65,536
- Each monster thinks every tick
- Expected: ~10-20x slower than Phase 1 (movement only)
- Still massively parallel: ~100,000+ ticks/sec

## Next Steps (Phase 3)

1. Add BSP collision detection
2. Add projectile system (fireballs, rockets)
3. Add proper state machine (animations)
4. Add level completion/exit triggers
5. Add sound propagation (alert nearby monsters)
6. Add proper pathfinding (A_NewChaseDir)

## References

Original DOOM source:
- `p_enemy.c` - Monster AI (`A_Chase`, attack functions)
- `p_map.c` - Line attacks, damage (`P_LineAttack`, `P_DamageMobj`)
- `p_mobj.c` - Object spawning (`P_SpawnMobj`)
