/*
 * ANGBAND MONSTERS - GPU Implementation
 *
 * Real monster data, spawning logic, and AI
 * Based on Angband monster system
 *
 * STUB IMPLEMENTATION - Replace with real monster data
 */

#ifndef ANGBAND_MONSTERS_CUH
#define ANGBAND_MONSTERS_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../../common/interleaved.h"
#include "borg_state.h"

// ============================================================================
// MONSTER RACE DATA
// ============================================================================

// Simplified monster race structure
struct MonsterRace {
    uint8_t depth;          // Native depth (1-127)
    uint8_t rarity;         // Spawn rarity (1-100, lower = rarer)
    uint8_t hp_dice;        // HP dice count
    uint8_t hp_sides;       // HP dice sides
    uint8_t ac;             // Armor class
    uint8_t speed;          // Monster speed (110 = normal)
    uint16_t xp_base;       // Base XP value
};

// STUB: Minimal monster table (expand to ~100 monsters)
// Format: {depth, rarity, hp_dice, hp_sides, ac, speed, xp}
__constant__ MonsterRace MONSTER_RACES[] = {
    // Depth 1-10
    {1, 1, 1, 4, 5, 110, 10},      // 0: White worm mass
    {1, 1, 1, 6, 8, 110, 15},      // 1: Rock lizard
    {2, 1, 2, 4, 10, 110, 25},     // 2: Giant white mouse
    {3, 2, 2, 6, 12, 110, 40},     // 3: Large kobold
    {4, 2, 3, 6, 15, 110, 60},     // 4: Orc
    {5, 3, 4, 6, 18, 110, 80},     // 5: Hill orc
    {6, 3, 5, 6, 20, 110, 100},    // 6: Troll
    {8, 4, 6, 8, 25, 110, 150},    // 7: Uruk
    {10, 4, 8, 8, 30, 110, 200},   // 8: Cave troll

    // Depth 11-30
    {12, 5, 10, 8, 35, 120, 300},  // 9: Hill giant
    {15, 5, 12, 8, 40, 120, 400},  // 10: Stone giant
    {18, 6, 15, 8, 45, 120, 600},  // 11: Young dragon
    {20, 6, 18, 10, 50, 120, 800}, // 12: Vampire
    {25, 7, 20, 10, 60, 130, 1200}, // 13: Ancient dragon
    {30, 8, 25, 12, 70, 130, 1800}, // 14: Balrog

    // Depth 31-50
    {35, 8, 30, 12, 80, 130, 2500}, // 15: Archlich
    {40, 9, 35, 15, 90, 140, 3500}, // 16: Greater balrog
    {45, 9, 40, 15, 100, 140, 5000}, // 17: Ancient wyrm
    {50, 10, 50, 20, 120, 150, 7500}, // 18: Dreadlord

    // Depth 51-100 (endgame)
    {60, 12, 60, 20, 140, 150, 10000}, // 19: Archlich of chaos
    {70, 15, 80, 25, 160, 160, 15000}, // 20: Ancient pit fiend
    {80, 20, 100, 30, 180, 170, 25000}, // 21: Greater titan
    {90, 25, 150, 40, 200, 180, 40000}, // 22: Chaos hound
    {100, 50, 300, 100, 250, 200, 100000}, // 23: Morgoth (boss)
};

#define NUM_MONSTER_RACES 24

// ============================================================================
// MONSTER SELECTION
// ============================================================================

// Get appropriate monster type for dungeon depth
// Based on get_mon_num() in mon-make.c
__device__ int get_monster_for_depth(
    int depth,            // Dungeon depth (1-127)
    curandState* rng      // RNG for selection
) {
    // STUB: Simple depth-based selection
    // REAL: Should implement probability tables and rarity checks

    // Select monsters within depth ±5 levels
    int min_depth = max(1, depth - 5);
    int max_depth = depth + 5;

    // Count valid monsters
    int valid_count = 0;
    int valid_indices[NUM_MONSTER_RACES];

    for (int i = 0; i < NUM_MONSTER_RACES; i++) {
        if (MONSTER_RACES[i].depth >= min_depth && MONSTER_RACES[i].depth <= max_depth) {
            valid_indices[valid_count++] = i;
        }
    }

    // Random selection from valid monsters
    if (valid_count > 0) {
        int pick = curand(rng) % valid_count;
        return valid_indices[pick];
    }

    // Fallback: return safest monster
    return 0;
}

// ============================================================================
// MONSTER SPAWNING
// ============================================================================

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
) {
    // STUB: Simple spawning
    // REAL: Should respect dungeon layout, group spawning, etc.

    // Number of monsters scales with depth
    int base_count = 5;
    int depth_bonus = depth / 5;
    int count = min(base_count + depth_bonus, (int)MAX_MONSTERS);

    for (int i = 0; i < count; i++) {
        // Select monster type
        int mtype = get_monster_for_depth(depth, rng);
        const MonsterRace& race = MONSTER_RACES[mtype];

        // Random position (STUB: should check terrain)
        int mx = curand(rng) % DUNGEON_WIDTH;
        int my = curand(rng) % DUNGEON_HEIGHT;

        // Roll HP
        int hp = 0;
        for (int d = 0; d < race.hp_dice; d++) {
            hp += curand(rng) % race.hp_sides + 1;
        }

        // Store in interleaved arrays
        ISET(monster_x, i, instance_id, num_instances, mx);
        ISET(monster_y, i, instance_id, num_instances, my);
        ISET(monster_hp, i, instance_id, num_instances, hp);
        ISET(monster_type, i, instance_id, num_instances, mtype);
        ISET(monster_awake, i, instance_id, num_instances, 0);  // Start asleep
    }

    return count;
}

// ============================================================================
// MONSTER STATS
// ============================================================================

// Get monster AC for combat
__device__ int get_monster_ac(int monster_type) {
    if (monster_type >= 0 && monster_type < NUM_MONSTER_RACES) {
        return MONSTER_RACES[monster_type].ac;
    }
    return 10;  // Default AC
}

// Get XP value for killing this monster
__device__ int get_monster_xp(int monster_type, int depth) {
    if (monster_type >= 0 && monster_type < NUM_MONSTER_RACES) {
        // XP scales with depth (out-of-depth monsters give bonus XP)
        int base_xp = MONSTER_RACES[monster_type].xp_base;
        int monster_depth = MONSTER_RACES[monster_type].depth;

        // Bonus for out-of-depth monsters
        if (monster_depth < depth - 5) {
            base_xp = base_xp * 3 / 2;  // 50% bonus
        }

        return base_xp;
    }
    return 10;  // Default XP
}

// ============================================================================
// MONSTER AI
// ============================================================================

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
) {
    // STUB: Simple distance-based waking
    // REAL: Should consider noise, stealth, monster sleep depth

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        int mx = IGET(monster_x, i, instance_id, num_instances);
        int my = IGET(monster_y, i, instance_id, num_instances);

        // Distance to player
        int dx = abs(mx - player_x);
        int dy = abs(my - player_y);
        int dist = max(dx, dy);

        // Wake if close (distance < 5)
        if (dist < 5) {
            // 50% chance to wake per turn when close
            if (curand(rng) % 100 < 50) {
                ISET(monster_awake, i, instance_id, num_instances, 1);
            }
        }
    }
}

#endif // ANGBAND_MONSTERS_CUH
