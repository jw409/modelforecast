/*
 * ANGBAND MONSTERS - GPU Implementation
 * PHASE 3: REAL LETHALITY (Breeding + Stealth)
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

// Monster blow (attack) data
struct MonsterBlow {
    uint8_t dd;  // Damage dice count
    uint8_t ds;  // Damage dice sides
};

struct MonsterRace {
    uint8_t depth;
    uint8_t rarity;
    uint8_t hp_dice;
    uint8_t hp_sides;
    uint8_t ac;
    uint8_t speed;
    uint16_t xp_base;
    uint8_t sleep;      // Base sleep value
    uint8_t flags;      // 1 = Breeds
    uint8_t num_blows;  // Number of attacks per round
    MonsterBlow blows[4];  // Up to 4 attacks
};

// Monster Flags
#define MFLAG_BREEDS 1

__constant__ MonsterRace MONSTER_RACES[] = {
    // ID 0: White Worm Mass - BREEDS!
    {1, 1, 2, 2, 2, 110, 2, 10, MFLAG_BREEDS}, 
    
    // ID 1: Grid Bug
    {1, 1, 1, 4, 10, 110, 3, 20, 0},
    
    // ID 2: Kobold
    {2, 1, 2, 6, 12, 110, 7, 30, 0},
    
    // ID 3: Orc
    {4, 2, 3, 6, 15, 110, 15, 10, 0},
    
    // ... (truncated for brevity, concept is key)
};

#define NUM_MONSTER_RACES 4

// ============================================================================
// MONSTER BEHAVIOR: BREEDING
// ============================================================================

// Try to breed a monster
// Returns true if spawned
__device__ bool try_breed_monster(
    int parent_idx,
    int mtype,
    int x, int y,
    int16_t* monster_x,
    int16_t* monster_y,
    int16_t* monster_hp,
    uint8_t* monster_type,
    uint8_t* monster_awake,
    uint8_t* monster_count_ptr,
    uint8_t* dungeon_terrain,
    int instance_id,
    int num_instances,
    curandState* rng
) {
    // Cap max monsters
    if (*monster_count_ptr >= MAX_MONSTERS) return false;

    // 1 in 10 chance per turn to breed if space available
    if ((curand(rng) % 100) > 10) return false;

    // Find empty spot
    int dx = (curand(rng) % 3) - 1;
    int dy = (curand(rng) % 3) - 1;
    int nx = x + dx;
    int ny = y + dy;
    
    // Check boundaries
    if (nx < 0 || nx >= DUNGEON_WIDTH || ny < 0 || ny >= DUNGEON_HEIGHT) return false;

    // STUB: Check collision with other monsters (expensive loop) or terrain
    // Assuming valid for now for speed in this kernel
    
    // Spawn
    int new_idx = *monster_count_ptr;
    ISET(monster_x, new_idx, instance_id, num_instances, nx);
    ISET(monster_y, new_idx, instance_id, num_instances, ny);
    ISET(monster_hp, new_idx, instance_id, num_instances, 5); // Fixed HP for clones
    ISET(monster_type, new_idx, instance_id, num_instances, mtype);
    ISET(monster_awake, new_idx, instance_id, num_instances, 1);
    
    (*monster_count_ptr)++;
    return true;
}

// ============================================================================
// MONSTER SELECTION
// ============================================================================

__device__ int get_monster_for_depth(int depth, curandState* rng) {
    // Brutal: 20% chance of Worm Mass (ID 0) regardless of depth to annoy user
    if ((curand(rng) % 100) < 20) return 0;
    
    return (curand(rng) % (NUM_MONSTER_RACES - 1)) + 1;
}

__device__ int spawn_monsters_for_depth(
    int depth,
    int16_t* monster_x,
    int16_t* monster_y,
    int16_t* monster_hp,
    uint8_t* monster_type,
    uint8_t* monster_awake,
    uint8_t* dungeon_terrain,
    int instance_id,
    int num_instances,
    curandState* rng
) {
    int count = 5 + depth;
    if (count > 20) count = 20;

    for (int i = 0; i < count; i++) {
        int mtype = get_monster_for_depth(depth, rng);
        
        ISET(monster_x, i, instance_id, num_instances, (curand(rng) % 50) + 10);
        ISET(monster_y, i, instance_id, num_instances, (curand(rng) % 20) + 5);
        ISET(monster_hp, i, instance_id, num_instances, 10); // Stub
        ISET(monster_type, i, instance_id, num_instances, mtype);
        ISET(monster_awake, i, instance_id, num_instances, 0); // Asleep by default
    }
    return count;
}

__device__ int get_monster_ac(int monster_type) {
    return 5 + monster_type * 2;
}

__device__ int get_monster_xp(int monster_type, int depth) {
    return 10 + monster_type * 10;
}

// ============================================================================
// MONSTER AI: WAKING (Stealth & Noise)
// ============================================================================

__device__ void wake_nearby_monsters(
    int16_t* monster_x,
    int16_t* monster_y,
    uint8_t* monster_awake,
    int monster_count,
    int player_x,
    int player_y,
    int instance_id,
    int num_instances,
    curandState* rng
) {
    // REAL: Wake chance based on distance and stealth
    // We don't have 'stealth' in BorgStateInterleaved explicitly, 
    // but we can approximate it from class/level logic or assume base.
    // Let's assume base stealth for now (e.g., 3).
    int player_stealth = 3; 

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        // Skip if already awake
        if (IGET(monster_awake, i, instance_id, num_instances)) continue;

        int mx = IGET(monster_x, i, instance_id, num_instances);
        int my = IGET(monster_y, i, instance_id, num_instances);

        // Distance to player
        int dx = abs(mx - player_x);
        int dy = abs(my - player_y);
        int dist = max(dx, dy);
        if (dist == 0) dist = 1;

        // Wake chance formula approximation
        // Chance = (100 / dist) - (stealth * 10)
        // At dist 1: 100 - 30 = 70% chance
        // At dist 5: 20 - 30 = 0% chance (safe)
        // At dist 10: 10 - 30 = 0% chance
        
        int wake_chance = (100 / dist) - (player_stealth * 10);
        
        if (wake_chance > 0) {
             if ((curand(rng) % 100) < wake_chance) {
                ISET(monster_awake, i, instance_id, num_instances, 1);
            }
        }
    }
}

#endif // ANGBAND_MONSTERS_CUH
