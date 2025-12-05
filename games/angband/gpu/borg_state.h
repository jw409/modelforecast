/*
 * ANGBAND BORG STATE - GPU Interleaved Layout
 *
 * Represents borg game state for parallel GPU execution.
 * Based on APWBorg (borg1.h) simplified for CUDA.
 */

#ifndef BORG_STATE_H
#define BORG_STATE_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "../../common/interleaved.h"

// ============================================================================
// CONSTANTS (from Angband)
// ============================================================================

#define DUNGEON_WIDTH       198
#define DUNGEON_HEIGHT      66
#define DUNGEON_SIZE        (DUNGEON_WIDTH * DUNGEON_HEIGHT)
#define MAX_DEPTH           127
#define MAX_MONSTERS        100     // Per instance, visible
#define MAX_ITEMS           100     // Per instance, visible
#define MAX_LEVEL           50

// Borg config flags (packed into uint16_t)
#define CFG_WORSHIPS_DAMAGE  (1 << 0)
#define CFG_WORSHIPS_SPEED   (1 << 1)
#define CFG_WORSHIPS_HP      (1 << 2)
#define CFG_WORSHIPS_MANA    (1 << 3)
#define CFG_WORSHIPS_AC      (1 << 4)
#define CFG_WORSHIPS_GOLD    (1 << 5)
#define CFG_PLAYS_RISKY      (1 << 6)
#define CFG_KILLS_UNIQUES    (1 << 7)
#define CFG_USES_SWAPS       (1 << 8)
#define CFG_CHEAT_DEATH      (1 << 9)

// Borg actions (from borg decision tree)
#define BORG_ACTION_NONE        0
#define BORG_ACTION_REST        1
#define BORG_ACTION_HEAL        2
#define BORG_ACTION_FLEE        3
#define BORG_ACTION_ATTACK      4
#define BORG_ACTION_EXPLORE     5
#define BORG_ACTION_DESCEND     6
#define BORG_ACTION_ASCEND      7
#define BORG_ACTION_PICKUP      8
#define BORG_ACTION_EQUIP       9
#define BORG_ACTION_USE_ITEM    10
#define BORG_ACTION_CAST        11
#define BORG_ACTION_SHOP        12

// Death causes
#define DEATH_NONE              0
#define DEATH_MONSTER           1
#define DEATH_TRAP              2
#define DEATH_STARVATION        3
#define DEATH_POISON            4
#define DEATH_UNIQUE            5

// ============================================================================
// BORG STATE (Interleaved)
// ============================================================================

typedef struct {
    // --- Player Position & Stats ---
    // All arrays: [num_instances]
    int16_t* x;                 // Current X position
    int16_t* y;                 // Current Y position
    int16_t* depth;             // Dungeon depth (1-127)
    int16_t* level;             // Character level (1-50)

    int16_t* hp;                // Current HP
    int16_t* max_hp;            // Max HP
    int16_t* mana;              // Current mana
    int16_t* max_mana;          // Max mana

    int16_t* speed;             // Speed (+0 to +30 typically)
    int16_t* ac;                // Armor class
    int16_t* damage;            // Average melee damage

    uint32_t* gold;             // Gold collected
    uint32_t* turns;            // Turns elapsed
    uint32_t* exp;              // Experience points

    // --- Borg Configuration ---
    uint16_t* config;           // Packed config flags
    uint8_t* no_deeper;         // Max depth limit

    // --- Dungeon State ---
    // Grid: [DUNGEON_SIZE * num_instances]
    uint8_t* dungeon_terrain;   // Terrain type per cell
    uint8_t* dungeon_known;     // Has borg seen this cell?
    uint8_t* dungeon_danger;    // Calculated danger per cell

    // --- Monsters (Interleaved) ---
    // [MAX_MONSTERS * num_instances]
    int16_t* monster_x;
    int16_t* monster_y;
    int16_t* monster_hp;
    uint8_t* monster_type;      // Monster race ID
    uint8_t* monster_awake;     // Is monster awake?
    uint8_t* monster_count;     // [num_instances] - how many monsters visible

    // --- Inventory (Simplified) ---
    uint8_t* has_healing;       // [num_instances] - has healing potions
    uint8_t* has_recall;        // Has recall scrolls
    uint8_t* has_teleport;      // Has teleport items
    uint8_t* has_detection;     // Has detection items

    // --- Results ---
    uint8_t* alive;             // [num_instances] - still alive?
    uint8_t* death_cause;       // What killed the borg
    uint16_t* final_depth;      // Deepest level reached
    uint32_t* final_turns;      // Total turns survived
    uint8_t* winner;            // Beat Morgoth?

} BorgStateInterleaved;

// ============================================================================
// MEMORY MANAGEMENT (Host)
// ============================================================================

// Allocate borg state for N instances
static inline BorgStateInterleaved* borg_state_alloc(uint32_t num_instances) {
    BorgStateInterleaved* s = (BorgStateInterleaved*)malloc(sizeof(BorgStateInterleaved));

    // Player stats
    cudaMalloc(&s->x, num_instances * sizeof(int16_t));
    cudaMalloc(&s->y, num_instances * sizeof(int16_t));
    cudaMalloc(&s->depth, num_instances * sizeof(int16_t));
    cudaMalloc(&s->level, num_instances * sizeof(int16_t));
    cudaMalloc(&s->hp, num_instances * sizeof(int16_t));
    cudaMalloc(&s->max_hp, num_instances * sizeof(int16_t));
    cudaMalloc(&s->mana, num_instances * sizeof(int16_t));
    cudaMalloc(&s->max_mana, num_instances * sizeof(int16_t));
    cudaMalloc(&s->speed, num_instances * sizeof(int16_t));
    cudaMalloc(&s->ac, num_instances * sizeof(int16_t));
    cudaMalloc(&s->damage, num_instances * sizeof(int16_t));
    cudaMalloc(&s->gold, num_instances * sizeof(uint32_t));
    cudaMalloc(&s->turns, num_instances * sizeof(uint32_t));
    cudaMalloc(&s->exp, num_instances * sizeof(uint32_t));

    // Config
    cudaMalloc(&s->config, num_instances * sizeof(uint16_t));
    cudaMalloc(&s->no_deeper, num_instances * sizeof(uint8_t));

    // Dungeon (interleaved)
    size_t dungeon_size = (size_t)DUNGEON_SIZE * num_instances;
    cudaMalloc(&s->dungeon_terrain, dungeon_size);
    cudaMalloc(&s->dungeon_known, dungeon_size);
    cudaMalloc(&s->dungeon_danger, dungeon_size);

    // Monsters (interleaved)
    size_t monster_size = (size_t)MAX_MONSTERS * num_instances;
    cudaMalloc(&s->monster_x, monster_size * sizeof(int16_t));
    cudaMalloc(&s->monster_y, monster_size * sizeof(int16_t));
    cudaMalloc(&s->monster_hp, monster_size * sizeof(int16_t));
    cudaMalloc(&s->monster_type, monster_size);
    cudaMalloc(&s->monster_awake, monster_size);
    cudaMalloc(&s->monster_count, num_instances);

    // Inventory
    cudaMalloc(&s->has_healing, num_instances);
    cudaMalloc(&s->has_recall, num_instances);
    cudaMalloc(&s->has_teleport, num_instances);
    cudaMalloc(&s->has_detection, num_instances);

    // Results
    cudaMalloc(&s->alive, num_instances);
    cudaMalloc(&s->death_cause, num_instances);
    cudaMalloc(&s->final_depth, num_instances * sizeof(uint16_t));
    cudaMalloc(&s->final_turns, num_instances * sizeof(uint32_t));
    cudaMalloc(&s->winner, num_instances);

    return s;
}

// Free borg state
static inline void borg_state_free(BorgStateInterleaved* s) {
    cudaFree(s->x); cudaFree(s->y); cudaFree(s->depth); cudaFree(s->level);
    cudaFree(s->hp); cudaFree(s->max_hp); cudaFree(s->mana); cudaFree(s->max_mana);
    cudaFree(s->speed); cudaFree(s->ac); cudaFree(s->damage);
    cudaFree(s->gold); cudaFree(s->turns); cudaFree(s->exp);
    cudaFree(s->config); cudaFree(s->no_deeper);
    cudaFree(s->dungeon_terrain); cudaFree(s->dungeon_known); cudaFree(s->dungeon_danger);
    cudaFree(s->monster_x); cudaFree(s->monster_y); cudaFree(s->monster_hp);
    cudaFree(s->monster_type); cudaFree(s->monster_awake); cudaFree(s->monster_count);
    cudaFree(s->has_healing); cudaFree(s->has_recall);
    cudaFree(s->has_teleport); cudaFree(s->has_detection);
    cudaFree(s->alive); cudaFree(s->death_cause);
    cudaFree(s->final_depth); cudaFree(s->final_turns); cudaFree(s->winner);
    free(s);
}

#endif // BORG_STATE_H
