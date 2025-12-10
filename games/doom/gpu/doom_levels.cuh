/**
 * GPU DOOM - Level Completion System
 *
 * Handles level progression for E1M1 -> E1M2 -> E1M3
 * - Exit trigger detection
 * - Level state tracking
 * - Monster kill counting
 * - Win condition
 *
 * Original: id Software DOOM (1993)
 * GPU Port: MIT License
 */

#ifndef DOOM_LEVELS_CUH
#define DOOM_LEVELS_CUH

#include "doom_types.cuh"
#include "doom_combat.cuh"  // For P_AproxDistance, gpu_random, P_DamagePlayer

// =============================================================================
// Level Exit Definitions
// =============================================================================

struct LevelExit {
    fixed_t x;           // Exit switch/linedef X position
    fixed_t y;           // Exit switch/linedef Y position
    fixed_t radius;      // Trigger radius around exit
};

// E1M1-E1M3 exit locations (simplified - using approximate exit switch positions)
// These are from the original DOOM WAD files, converted to fixed point
// Coordinates are in map units (1 map unit = 1 FRACUNIT)

__constant__ LevelExit c_exits[3] = {
    // E1M1 exit: Green armor room, southeast corner
    { 1952 * FRACUNIT, -3424 * FRACUNIT, 128 * FRACUNIT },

    // E1M2 exit: Behind yellow door, northeast area
    { 2048 * FRACUNIT, -2304 * FRACUNIT, 128 * FRACUNIT },

    // E1M3 exit: Central courtyard, north side
    { 1920 * FRACUNIT, -4352 * FRACUNIT, 128 * FRACUNIT }
};

// Player start positions for each level
struct LevelStart {
    fixed_t x;
    fixed_t y;
    fixed_t z;
    angle_t angle;
};

__constant__ LevelStart c_starts[3] = {
    // E1M1 start: Southwest corner of starting room
    { 1056 * FRACUNIT, -3616 * FRACUNIT, 0, ANG90 },

    // E1M2 start: Northwest area
    { 1024 * FRACUNIT, -3008 * FRACUNIT, 0, ANG90 },

    // E1M3 start: Southwest starting room
    { 1120 * FRACUNIT, -4160 * FRACUNIT, 0, ANG90 }
};

// Monster counts per level (simplified - approximate totals)
__constant__ int32_t c_monster_counts[3] = {
    10,  // E1M1: ~10 monsters (simplified)
    15,  // E1M2: ~15 monsters (simplified)
    20   // E1M3: ~20 monsters (simplified)
};

// =============================================================================
// E1M1 Monster Spawn Data (simplified positions from actual WAD)
// =============================================================================

struct MonsterSpawn {
    fixed_t x;
    fixed_t y;
    uint8_t type;  // MonsterTypeID
};

// E1M1 has ~10 monsters in Hurt Me Plenty - these are approximate positions
__constant__ MonsterSpawn c_e1m1_monsters[10] = {
    // Starting area - 2 zombiemen
    { 1280 * FRACUNIT, -3328 * FRACUNIT, 0 },  // MT_ZOMBIE
    { 1536 * FRACUNIT, -3200 * FRACUNIT, 0 },  // MT_ZOMBIE

    // Zigzag corridor - 2 zombies, 1 imp
    { 1792 * FRACUNIT, -2816 * FRACUNIT, 0 },  // MT_ZOMBIE
    { 2048 * FRACUNIT, -2560 * FRACUNIT, 2 },  // MT_IMP
    { 1920 * FRACUNIT, -2432 * FRACUNIT, 0 },  // MT_ZOMBIE

    // Nukage room - 2 imps
    { 2304 * FRACUNIT, -2944 * FRACUNIT, 2 },  // MT_IMP
    { 2560 * FRACUNIT, -3072 * FRACUNIT, 2 },  // MT_IMP

    // Computer room - 1 shotgunner, 2 zombies
    { 1664 * FRACUNIT, -3520 * FRACUNIT, 1 },  // MT_SHOTGUY
    { 1472 * FRACUNIT, -3648 * FRACUNIT, 0 },  // MT_ZOMBIE
    { 1856 * FRACUNIT, -3712 * FRACUNIT, 0 },  // MT_ZOMBIE
};

// Level names for output
__constant__ const char* c_level_names[3] = {
    "E1M1",
    "E1M2",
    "E1M3"
};

// =============================================================================
// Forward Declarations
// =============================================================================

// This header assumes the following are already declared:
// - d_player_x, d_player_y, d_player_z, d_player_angle
// - d_player_momx, d_player_momy, d_player_momz, d_player_alive
// - d_current_level, d_level_complete, d_game_won
// - d_completion_tick, d_monsters_killed, d_monsters_total
// - FixedMul() function

// =============================================================================
// Level Completion Logic
// =============================================================================

/**
 * Check if player has reached the exit trigger
 * Returns true if player is within radius of current level's exit
 */
__device__ bool CheckExitTrigger(int instance_id, fixed_t player_x, fixed_t player_y, uint8_t level) {
    if (level >= 3) return false;  // No exit beyond E1M3

    LevelExit exit = c_exits[level];

    // Calculate distance from player to exit
    fixed_t dx = player_x - exit.x;
    fixed_t dy = player_y - exit.y;

    // Fast distance check using radius squared (avoids sqrt)
    fixed_t dist_sq = FixedMul(dx, dx) + FixedMul(dy, dy);
    fixed_t radius_sq = FixedMul(exit.radius, exit.radius);

    return dist_sq <= radius_sq;
}

/**
 * Transition to next level
 * Resets player position and level state
 */
__device__ void TransitionToNextLevel(int instance_id) {
    uint8_t current_level = d_current_level[instance_id];

    // Increment level (0->1, 1->2, 2 stays at 2)
    if (current_level < 2) {
        current_level++;
        d_current_level[instance_id] = current_level;
    }

    // Get new level start position
    LevelStart start = c_starts[current_level];

    // Reset player position
    d_player_x[instance_id] = start.x;
    d_player_y[instance_id] = start.y;
    d_player_z[instance_id] = start.z;
    d_player_angle[instance_id] = start.angle;

    // Reset momentum
    d_player_momx[instance_id] = 0;
    d_player_momy[instance_id] = 0;
    d_player_momz[instance_id] = 0;

    // Clear level complete flag for new level
    d_level_complete[instance_id] = 0;

    // Add this level's monster count to total
    d_monsters_total[instance_id] += c_monster_counts[current_level];

    // TODO: Spawn monsters for new level (when monster system is ready)
}

/**
 * Check for level completion and handle transitions
 * Called once per tick after player movement
 *
 * IMPORTANT: Requires BT_USE button to activate exit (like real DOOM)
 * Player must be at exit AND press USE to complete level
 */
__device__ void CheckLevelCompletion(int instance_id, int tick, uint8_t buttons) {
    // Skip if dead or already won
    if (!d_player_alive[instance_id] || d_game_won[instance_id]) {
        return;
    }

    // Must press USE button to activate exit (just like real DOOM)
    if (!(buttons & BT_USE)) {
        return;
    }

    uint8_t current_level = d_current_level[instance_id];

    // Check if player reached exit
    fixed_t player_x = d_player_x[instance_id];
    fixed_t player_y = d_player_y[instance_id];

    if (CheckExitTrigger(instance_id, player_x, player_y, current_level)) {
        // Mark level complete
        if (!d_level_complete[instance_id]) {
            d_level_complete[instance_id] = 1;

            // Check if this was final level (E1M3)
            if (current_level == 2) {
                // GAME WON!
                d_game_won[instance_id] = 1;
                d_completion_tick[instance_id] = tick;
            } else {
                // Transition to next level
                TransitionToNextLevel(instance_id);
            }
        }
    }
}

/**
 * Get kill percentage for current instance
 * Returns 0-100
 */
__device__ int32_t GetKillPercentage(int instance_id) {
    int32_t total = d_monsters_total[instance_id];
    if (total == 0) return 0;

    int32_t killed = d_monsters_killed[instance_id];
    return (killed * 100) / total;
}

/**
 * Spawn monsters for E1M1
 * Called during level initialization
 */
__device__ void SpawnE1M1Monsters(int instance_id, int num_instances) {
    for (int m = 0; m < 10; m++) {
        int idx = m * num_instances + instance_id;
        MonsterSpawn spawn = c_e1m1_monsters[m];

        d_monster_x[idx] = spawn.x;
        d_monster_y[idx] = spawn.y;
        d_monster_z[idx] = 0;
        d_monster_angle[idx] = ANG90;  // Facing north initially
        d_monster_type[idx] = spawn.type;
        d_monster_alive[idx] = 1;
        d_monster_target_idx[idx] = -1;  // No target yet
        d_monster_movedir[idx] = DI_NODIR;
        d_monster_movecount[idx] = 0;
        d_monster_reactiontime[idx] = 8;  // Ticks before monster reacts

        // Set health based on monster type
        if (spawn.type == 0) {  // MT_ZOMBIE
            d_monster_health[idx] = 20;
        } else if (spawn.type == 1) {  // MT_SHOTGUY
            d_monster_health[idx] = 30;
        } else if (spawn.type == 2) {  // MT_IMP
            d_monster_health[idx] = 60;
        } else {
            d_monster_health[idx] = 100;  // Default
        }
    }
    d_monster_count[instance_id] = 10;
}

/**
 * Initialize level state for instance
 * Called during instance setup
 */
__device__ void InitLevelState(int instance_id, int num_instances) {
    d_current_level[instance_id] = 0;      // Start at E1M1
    d_level_complete[instance_id] = 0;
    d_game_won[instance_id] = 0;
    d_completion_tick[instance_id] = 0;
    d_monsters_killed[instance_id] = 0;
    d_monsters_total[instance_id] = c_monster_counts[0];  // E1M1 monster count

    // Spawn E1M1 monsters
    SpawnE1M1Monsters(instance_id, num_instances);
}

/**
 * Record monster kill
 * Called when a monster dies (when monster system is ready)
 */
__device__ void RecordMonsterKill(int instance_id) {
    // Atomic increment to prevent race conditions if we add monster AI threads
    atomicAdd(&d_monsters_killed[instance_id], 1);
}

/**
 * Monster AI tick - called for each monster each tick
 * Handles: activation, chasing, attacking
 */
__device__ void P_MonsterThink_GPU(int instance_id, int tick, int num_instances) {
    int num_monsters = d_monster_count[instance_id];
    if (num_monsters == 0) return;

    // Player position
    fixed_t player_x = d_player_x[instance_id];
    fixed_t player_y = d_player_y[instance_id];
    uint8_t player_alive = d_player_alive[instance_id];

    for (int m = 0; m < num_monsters; m++) {
        int idx = m * num_instances + instance_id;

        // Skip dead monsters
        if (!d_monster_alive[idx]) continue;

        // Get monster state
        fixed_t mon_x = d_monster_x[idx];
        fixed_t mon_y = d_monster_y[idx];
        uint8_t mon_type = d_monster_type[idx];
        int16_t reaction = d_monster_reactiontime[idx];

        // Calculate distance to player
        fixed_t dx = player_x - mon_x;
        fixed_t dy = player_y - mon_y;
        fixed_t dist = P_AproxDistance(dx, dy);

        // Reaction time countdown (monster wakes up)
        if (reaction > 0) {
            // Wake up if player is close (within ~512 map units)
            if (dist < 512 * FRACUNIT) {
                d_monster_reactiontime[idx] = reaction - 1;
            }
            continue;  // Still sleeping
        }

        // Monster is awake - chase and attack player
        if (!player_alive) continue;  // Player dead, nothing to do

        // Check if in melee range (64 units)
        if (dist < 64 * FRACUNIT) {
            // Melee attack!
            int damage;
            if (mon_type == 0 || mon_type == 1) {  // Zombie/Shotguy - punch
                damage = 3 + (gpu_random(instance_id, tick, m) % 8);
            } else if (mon_type == 2) {  // Imp scratch
                damage = 3 + (gpu_random(instance_id, tick, m) % 21);
            } else {
                damage = 5 + (gpu_random(instance_id, tick, m) % 15);
            }
            P_DamagePlayer(instance_id, damage);
        }
        // Check if in missile range and can attack (every 35 ticks)
        else if (dist < 2048 * FRACUNIT && (tick % 35) == (m % 35)) {
            // Ranged attack (simplified - hitscan for now)
            int damage;
            if (mon_type == 0) {  // Zombie pistol
                damage = 3 + (gpu_random(instance_id, tick, m) % 12);
            } else if (mon_type == 1) {  // Shotgun
                damage = 3 + (gpu_random(instance_id, tick, m) % 12);
            } else if (mon_type == 2) {  // Imp fireball
                damage = 3 + (gpu_random(instance_id, tick, m) % 21);
            } else {
                damage = 5;
            }
            P_DamagePlayer(instance_id, damage);
        }

        // Move toward player (simplified - no pathfinding)
        if (dist > 64 * FRACUNIT) {
            // Calculate direction
            fixed_t speed = 4 * FRACUNIT;  // Movement speed

            // Normalize direction
            int64_t norm = (int64_t)dist;
            if (norm > 0) {
                fixed_t move_x = (int32_t)(((int64_t)dx * speed) / norm);
                fixed_t move_y = (int32_t)(((int64_t)dy * speed) / norm);

                d_monster_x[idx] = mon_x + move_x;
                d_monster_y[idx] = mon_y + move_y;
            }
        }
    }
}

// =============================================================================
// Level Stats Output (Host-side)
// =============================================================================

/**
 * Get level name string (host function)
 */
const char* GetLevelName(uint8_t level) {
    if (level >= 3) return "???";
    const char* names[] = { "E1M1", "E1M2", "E1M3" };
    return names[level];
}

/**
 * Get completion status string (host function)
 */
const char* GetCompletionStatus(uint8_t game_won, uint8_t alive) {
    if (game_won) return "COMPLETED";
    if (!alive) return "DIED";
    return "IN PROGRESS";
}

#endif // DOOM_LEVELS_CUH
