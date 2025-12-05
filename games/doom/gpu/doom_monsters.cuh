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

// Monster type constants
enum MonsterTypeID : uint8_t {
    MT_ZOMBIE = 0,     // Former Human - 20 HP, hitscan pistol
    MT_SHOTGUY = 1,    // Shotgun guy - 30 HP, hitscan shotgun
    MT_IMP = 2,        // Imp - 60 HP, fireball (treated as hitscan for Phase 2)
    MT_DEMON = 3,      // Pinky - 150 HP, melee only
    MT_CACODEMON = 4,  // Cacodemon - 400 HP, fireball
    MT_BARON = 5,      // Baron of Hell - 1000 HP, fireball
    MT_TYPE_COUNT = 6
};

// Monster stats lookup table (constant device memory)
__constant__ MonsterStats c_monster_stats[MT_TYPE_COUNT] = {
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

// Direction constants (8-way movement)
#define DI_EAST       0
#define DI_NORTHEAST  1
#define DI_NORTH      2
#define DI_NORTHWEST  3
#define DI_WEST       4
#define DI_SOUTHWEST  5
#define DI_SOUTH      6
#define DI_SOUTHEAST  7
#define DI_NODIR      8

// Combat constants
#define MELEERANGE    (64 * FRACUNIT)
#define MISSILERANGE  (2048 * FRACUNIT)

#endif // DOOM_MONSTERS_CUH
