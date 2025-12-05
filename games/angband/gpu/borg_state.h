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

// Borg actions (comprehensive combat + exploration)
// === BASIC ===
#define BORG_ACTION_NONE        0
#define BORG_ACTION_REST        1   // Recover HP/mana
#define BORG_ACTION_EXPLORE     2   // Move/explore
#define BORG_ACTION_DESCEND     3   // Go down stairs
#define BORG_ACTION_ASCEND      4   // Go up stairs
#define BORG_ACTION_PICKUP      5   // Pick up item
#define BORG_ACTION_DROP        6   // Drop item

// === MELEE COMBAT ===
#define BORG_ACTION_MELEE       10  // Standard melee attack
#define BORG_ACTION_CHARGE      11  // Rush + attack (risky)

// === RANGED COMBAT ===
#define BORG_ACTION_SHOOT       20  // Bow/crossbow attack
#define BORG_ACTION_THROW       21  // Throw item (rock, potion, oil)
#define BORG_ACTION_THROW_OIL   22  // Burn area - critical vs breeders!

// === MAGIC (Spells) ===
#define BORG_ACTION_CAST_BOLT   30  // Single target damage (Magic Missile, Lightning)
#define BORG_ACTION_CAST_BALL   31  // Area damage (Fireball, Frost Ball)
#define BORG_ACTION_CAST_BEAM   32  // Line damage (Light Beam)
#define BORG_ACTION_CAST_HEAL   33  // Heal self
#define BORG_ACTION_CAST_BUFF   34  // Haste, Resist, Bless
#define BORG_ACTION_CAST_ESCAPE 35  // Teleport, Phase Door

// === ITEMS (Consumables) ===
#define BORG_ACTION_QUAFF       40  // Drink potion
#define BORG_ACTION_READ        41  // Read scroll
#define BORG_ACTION_ZAP         42  // Use wand/staff
#define BORG_ACTION_ACTIVATE    43  // Activate artifact/rod

// === DEFENSIVE ===
#define BORG_ACTION_FLEE        50  // Tactical retreat
#define BORG_ACTION_PHASE       51  // Phase door (short teleport)
#define BORG_ACTION_TELEPORT    52  // Long teleport
#define BORG_ACTION_RECALL      53  // Word of Recall (town/dungeon)

// === EQUIPMENT ===
#define BORG_ACTION_EQUIP       60  // Equip item
#define BORG_ACTION_UNEQUIP     61  // Remove item
#define BORG_ACTION_SWAP        62  // Swap weapon sets

// === SPECIAL ===
#define BORG_ACTION_SHOP        70  // Town: buy/sell
#define BORG_ACTION_ENCHANT     71  // Town: enchant item
#define BORG_ACTION_IDENTIFY    72  // ID unknown item

// Energy costs (Angband: 100 energy = 1 normal action)
#define ENERGY_MOVE         100
#define ENERGY_ATTACK       100
#define ENERGY_REST         100
#define ENERGY_THROW        100
#define ENERGY_PICKUP       50   // Half action
#define ENERGY_QUAFF        100

// Death causes
#define DEATH_NONE              0
#define DEATH_MONSTER           1
#define DEATH_TRAP              2
#define DEATH_STARVATION        3
#define DEATH_POISON            4
#define DEATH_UNIQUE            5

// Attack effects (from Angband monster.txt)
#define EFFECT_HURT             0   // Plain damage
#define EFFECT_POISON           1   // Ongoing poison damage
#define EFFECT_FIRE             2   // Fire damage
#define EFFECT_COLD             3   // Cold damage
#define EFFECT_ACID             4   // Acid damage + equipment damage
#define EFFECT_ELEC             5   // Electric damage
#define EFFECT_PARALYZE         6   // Can't act! Fatal if surrounded
#define EFFECT_CONFUSE          7   // Random movement
#define EFFECT_BLIND            8   // Can't see monsters
#define EFFECT_TERRIFY          9   // Forced flee, can't attack
#define EFFECT_LOSE_STR         10  // Stat drain (permanent!)
#define EFFECT_LOSE_DEX         11
#define EFFECT_LOSE_CON         12
#define EFFECT_LOSE_INT         13
#define EFFECT_LOSE_WIS         14
#define EFFECT_LOSE_ALL         15  // Morgoth special - ALL stats
#define EFFECT_EXP_DRAIN        16  // Experience drain (level loss!)
#define EFFECT_DRAIN_CHARGES    17  // Empties wands/staves
#define EFFECT_DISENCHANT       18  // Removes item +hit/+dam/+AC
#define EFFECT_BLACK_BREATH     19  // Nazgul - prevents HP regen
#define EFFECT_SHATTER          20  // Destroys equipment
#define EFFECT_HALLU            21  // Hallucination
#define EFFECT_EAT_GOLD         22  // Steals gold
#define EFFECT_EAT_ITEM         23  // Steals item
#define EFFECT_EAT_FOOD         24  // Steals food
#define EFFECT_EAT_LIGHT        25  // Drains light source

// Timed status effects (player debuffs)
#define STATUS_POISONED         (1 << 0)
#define STATUS_PARALYZED        (1 << 1)
#define STATUS_CONFUSED         (1 << 2)
#define STATUS_BLIND            (1 << 3)
#define STATUS_AFRAID           (1 << 4)
#define STATUS_HALLUCINATING    (1 << 5)
#define STATUS_BLACK_BREATH     (1 << 6)  // No HP regen

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

    // --- Inventory (Item Counts) ---
    uint8_t* potions_healing;   // [num_instances] - Cure Light/Serious/Critical
    uint8_t* potions_restore;   // Restore mana/stats
    uint8_t* scrolls_recall;    // Word of Recall
    uint8_t* scrolls_teleport;  // Teleport/Phase Door
    uint8_t* scrolls_detection; // Detect Monsters/Traps
    uint8_t* flasks_oil;        // OIL - burn area, critical vs breeders!
    uint8_t* wands_charges;     // Generic wand charges (Magic Missile, etc)

    // --- Equipment Slots ---
    uint8_t* weapon_dd;         // Weapon damage dice
    uint8_t* weapon_ds;         // Weapon damage sides
    int8_t* weapon_to_hit;      // Weapon to-hit bonus
    int8_t* weapon_to_dam;      // Weapon to-damage bonus
    uint8_t* armor_ac;          // Armor base AC

    // --- Energy System (Speed-based turns) ---
    int16_t* energy;            // Current energy (100 = 1 action)

    // --- Legacy compatibility ---
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

    // Inventory (new item system)
    cudaMalloc(&s->potions_healing, num_instances);
    cudaMalloc(&s->potions_restore, num_instances);
    cudaMalloc(&s->scrolls_recall, num_instances);
    cudaMalloc(&s->scrolls_teleport, num_instances);
    cudaMalloc(&s->scrolls_detection, num_instances);
    cudaMalloc(&s->flasks_oil, num_instances);
    cudaMalloc(&s->wands_charges, num_instances);

    // Equipment
    cudaMalloc(&s->weapon_dd, num_instances);
    cudaMalloc(&s->weapon_ds, num_instances);
    cudaMalloc(&s->weapon_to_hit, num_instances);
    cudaMalloc(&s->weapon_to_dam, num_instances);
    cudaMalloc(&s->armor_ac, num_instances);

    // Energy system
    cudaMalloc(&s->energy, num_instances * sizeof(int16_t));

    // Legacy compatibility
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
    // Inventory
    cudaFree(s->potions_healing); cudaFree(s->potions_restore);
    cudaFree(s->scrolls_recall); cudaFree(s->scrolls_teleport); cudaFree(s->scrolls_detection);
    cudaFree(s->flasks_oil); cudaFree(s->wands_charges);
    // Equipment
    cudaFree(s->weapon_dd); cudaFree(s->weapon_ds);
    cudaFree(s->weapon_to_hit); cudaFree(s->weapon_to_dam); cudaFree(s->armor_ac);
    // Energy
    cudaFree(s->energy);
    // Legacy
    cudaFree(s->has_healing); cudaFree(s->has_recall);
    cudaFree(s->has_teleport); cudaFree(s->has_detection);
    cudaFree(s->alive); cudaFree(s->death_cause);
    cudaFree(s->final_depth); cudaFree(s->final_turns); cudaFree(s->winner);
    free(s);
}

#endif // BORG_STATE_H
