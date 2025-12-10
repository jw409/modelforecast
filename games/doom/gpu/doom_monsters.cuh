/**
 * GPU DOOM - Monster AI and Combat
 *
 * Phase 2: Monster behavior, attacks, and damage system
 *
 * Original: id Software DOOM (1993)
 * GPU Port: MIT License
 */

#ifndef DOOM_MONSTERS_CUH
#define DOOM_MONSTERS_CUH

#include "doom_types.cuh"

// =============================================================================
// Monster Type Stats
// =============================================================================

struct MonsterStats {
    int health;
    int melee_damage_min;
    int melee_damage_max;
    int ranged_damage_min;
    int ranged_damage_max;
    int melee_range;       // In FRACUNIT
    int missile_range;     // In FRACUNIT
    int speed;             // Movement speed (FRACUNIT per tick)
};

// Use MobjType from doom_types.cuh for monster types
// Mapping: MT_POSSESSED=1 (Zombieman), MT_SHOTGUY=2, MT_IMP=3, MT_DEMON=4, etc.

// Number of monster types for stats array (matches first N MobjTypes that are monsters)
#define MONSTER_STATS_COUNT 6

// Monster stats lookup table (constant device memory)
// Indexed by MobjType - 1 (MT_POSSESSED=1 -> index 0, etc.)
__constant__ MonsterStats c_monster_stats[MONSTER_STATS_COUNT] = {
    // MT_ZOMBIE: health, melee_min, melee_max, ranged_min, ranged_max, melee_range, missile_range, speed
    {20, 0, 0, 3, 15, 0, 2048 * FRACUNIT, 8 * FRACUNIT},
    // MT_SHOTGUY
    {30, 0, 0, 3, 15, 0, 2048 * FRACUNIT, 8 * FRACUNIT},
    // MT_IMP
    {60, 3, 24, 3, 24, 64 * FRACUNIT, 2048 * FRACUNIT, 8 * FRACUNIT},
    // MT_DEMON (Pinky)
    {150, 4, 40, 0, 0, 64 * FRACUNIT, 0, 10 * FRACUNIT},
    // MT_CACODEMON
    {400, 0, 0, 5, 40, 0, 2048 * FRACUNIT, 8 * FRACUNIT},
    // MT_BARON
    {1000, 10, 80, 8, 64, 64 * FRACUNIT, 2048 * FRACUNIT, 8 * FRACUNIT}
};

// Direction constants defined in doom_types.cuh

// Combat constants
#define MELEERANGE    (64 * FRACUNIT)
#define MISSILERANGE  (2048 * FRACUNIT)

#endif // DOOM_MONSTERS_CUH
