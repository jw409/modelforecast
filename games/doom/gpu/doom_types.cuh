/**
 * GPU DOOM Types
 *
 * Port of id Software DOOM (1993) data structures to CUDA.
 * Pointers converted to indices. Interleaved layout for coalescing.
 *
 * Original: linuxdoom-1.10 (id Software License)
 * GPU Port: MIT License
 */

#ifndef DOOM_TYPES_CUH
#define DOOM_TYPES_CUH

#include <cstdint>

// =============================================================================
// Configuration
// =============================================================================

#define MAX_INSTANCES     200000  // Target: 150k parallel sessions
#define MAX_MOBJS         1024    // Support slaughter maps
#define MAX_THINKERS      1024    // Active thinkers per instance
#define INPUT_HORIZON     1024    // Max ticks of pre-buffered input
#define CHECKPOINT_SLOTS  10      // Reduced slots to save VRAM at high instance counts

// DOOM constants (from original source)
#define MAXPLAYERS        4
#define NUMWEAPONS        9
#define NUMAMMO           4
#define NUMPOWERS         6
#define NUMCARDS          6

// Fixed point (16.16)
typedef int32_t fixed_t;
#define FRACBITS          16
#define FRACUNIT          (1 << FRACBITS)

// Angle (binary angle measurement)
typedef uint32_t angle_t;
#define ANG45             0x20000000u
#define ANG90             0x40000000u
#define ANG180            0x80000000u
#define ANG270            0xc0000000u

// =============================================================================
// Input: ticcmd_t (8 bytes) - Player input per tick
// =============================================================================

struct __align__(8) TicCmd {
    int8_t   forwardmove;    // -127 to 127
    int8_t   sidemove;       // -127 to 127
    int16_t  angleturn;      // Angle delta
    int16_t  consistency;    // Checksum (unused in GPU)
    uint8_t  chatchar;       // Chat (unused)
    uint8_t  buttons;        // BT_ATTACK, BT_USE, BT_CHANGE
};

// Button flags
#define BT_ATTACK         1
#define BT_USE            2
#define BT_CHANGE         4     // Weapon change
#define BT_WEAPONMASK     (8+16+32)
#define BT_WEAPONSHIFT    3

// =============================================================================
// Map Object: mobj_t → GPUMobj (GPU-friendly)
// =============================================================================

// Object types (subset - add more as needed)
enum MobjType : uint16_t {
    MT_PLAYER = 0,
    MT_POSSESSED,      // Zombieman
    MT_SHOTGUY,        // Shotgun guy
    MT_IMP,
    MT_DEMON,          // Pinky
    MT_SPECTRE,
    MT_CACODEMON,
    MT_BRUISER,        // Baron
    MT_SKULL,          // Lost soul
    MT_SPIDER,         // Spider mastermind
    MT_CYBORG,         // Cyberdemon
    MT_BARREL,
    MT_TROOPSHOT,      // Imp fireball
    MT_BRUISERSHOT,    // Baron fireball
    MT_ROCKET,
    MT_PLASMA,
    MT_BFG,
    MT_PUFF,           // Bullet puff
    MT_BLOOD,
    MT_CLIP,           // Ammo
    MT_MISC0,          // Health bonus
    // ... add more as needed
    MT_COUNT
};

// Object flags (from mobjflag_t)
#define MF_SPECIAL      0x0001
#define MF_SOLID        0x0002
#define MF_SHOOTABLE    0x0004
#define MF_NOSECTOR     0x0008
#define MF_NOBLOCKMAP   0x0010
#define MF_AMBUSH       0x0020
#define MF_JUSTHIT      0x0040
#define MF_JUSTATTACKED 0x0080
#define MF_NOGRAVITY    0x0200
#define MF_DROPOFF      0x0400
#define MF_PICKUP       0x0800
#define MF_NOCLIP       0x1000
#define MF_FLOAT        0x4000
#define MF_MISSILE      0x10000
#define MF_CORPSE       0x100000
#define MF_COUNTKILL    0x400000
#define MF_COUNTITEM    0x800000

// Thinker function types
enum ThinkerFunc : uint8_t {
    TF_NONE = 0,
    TF_MOBJ,           // P_MobjThinker
    TF_PLAYER,         // P_PlayerThink (special case)
    TF_DOOR,           // T_VerticalDoor
    TF_FLOOR,          // T_MoveFloor
    TF_CEILING,        // T_MoveCeiling
    TF_PLAT,           // T_PlatRaise
    TF_LIGHT,          // T_LightFlash
};

// GPU-friendly mobj (no pointers)
struct __align__(16) GPUMobj {
    // Position (fixed point)
    fixed_t x, y, z;

    // Movement
    fixed_t momx, momy, momz;

    // Collision
    fixed_t radius, height;
    fixed_t floorz, ceilingz;

    // Orientation
    angle_t angle;

    // State
    MobjType type;
    uint16_t state_id;       // Index into state table
    int16_t  tics;           // State tic counter
    int32_t  health;
    uint32_t flags;

    // AI
    uint8_t  movedir;        // 0-7 direction
    uint8_t  movecount;      // Steps until direction change
    int16_t  reactiontime;
    int16_t  threshold;

    // Links (indices, not pointers)
    int16_t  target_idx;     // -1 = none
    int16_t  tracer_idx;     // -1 = none
    int16_t  player_idx;     // -1 = not a player, 0-3 = player number

    // Thinker
    ThinkerFunc thinker_func;
    uint8_t  active;         // 0 = dead/removed

    uint8_t  _pad[2];
};

static_assert(sizeof(GPUMobj) == 80, "GPUMobj size check");

// =============================================================================
// Player: player_t → GPUPlayer
// =============================================================================

enum PlayerState : uint8_t {
    PST_LIVE = 0,
    PST_DEAD,
    PST_REBORN
};

enum WeaponType : uint8_t {
    WP_FIST = 0,
    WP_PISTOL,
    WP_SHOTGUN,
    WP_CHAINGUN,
    WP_MISSILE,
    WP_PLASMA,
    WP_BFG,
    WP_CHAINSAW,
    WP_SUPERSHOTGUN,
    WP_NOCHANGE = 255
};

enum AmmoType : uint8_t {
    AM_CLIP = 0,    // Pistol / chaingun
    AM_SHELL,       // Shotgun
    AM_CELL,        // Plasma / BFG
    AM_MISL         // Rocket launcher
};

struct __align__(64) GPUPlayer {
    // Core stats
    int32_t  health;
    int32_t  armorpoints;
    uint8_t  armortype;      // 0-2
    PlayerState playerstate;

    // Weapons
    WeaponType readyweapon;
    WeaponType pendingweapon;
    uint16_t weaponowned;    // Bitfield

    // Ammo
    int16_t  ammo[NUMAMMO];
    int16_t  maxammo[NUMAMMO];

    // Powers (tic counters)
    int16_t  powers[NUMPOWERS];

    // Keys
    uint8_t  cards;          // Bitfield for 6 keys
    uint8_t  backpack;

    // Combat state
    uint8_t  attackdown;
    uint8_t  usedown;
    int16_t  refire;

    // View
    fixed_t  viewz;
    fixed_t  viewheight;
    fixed_t  deltaviewheight;
    fixed_t  bob;

    // Stats
    int16_t  killcount;
    int16_t  itemcount;
    int16_t  secretcount;

    // Screen effects
    int16_t  damagecount;
    int16_t  bonuscount;
    int16_t  extralight;

    // Link to mobj
    int16_t  mo_idx;         // Index into mobj pool
    int16_t  attacker_idx;   // Who damaged us last

    // Cheats
    uint8_t  cheats;
    uint8_t  _pad[1];
};

static_assert(sizeof(GPUPlayer) == 128, "GPUPlayer size check");

// =============================================================================
// Checkpoint: Snapshot of game state for AI evaluation
// =============================================================================

struct __align__(32) Checkpoint {
    int32_t  tick;

    // Player state
    int16_t  health;
    int16_t  armor;
    int16_t  ammo[NUMAMMO];

    // Position
    fixed_t  x, y, z;
    angle_t  angle;

    // Progress
    int16_t  kills;
    int16_t  items;
    int16_t  secrets;

    // Status
    uint8_t  alive;
    uint8_t  weapon;

    // Nearby threats
    uint8_t  monsters_visible;
    uint8_t  projectiles_nearby;

    uint8_t  _pad[4];
};

static_assert(sizeof(Checkpoint) == 64, "Checkpoint should be 64 bytes");

// =============================================================================
// Instance State: Complete game state for one parallel instance
// =============================================================================

struct GPUInstance {
    // Players (usually just 1 for AI arena)
    GPUPlayer players[MAXPLAYERS];
    uint8_t   playeringame[MAXPLAYERS];

    // All map objects
    GPUMobj   mobjs[MAX_MOBJS];
    int16_t   num_mobjs;
    int16_t   player_mobj_idx[MAXPLAYERS];  // Quick lookup

    // Level state
    int32_t   leveltime;
    uint8_t   paused;
    uint8_t   gamestate;    // GS_LEVEL, GS_INTERMISSION, etc.

    // Input ring buffer position
    int32_t   input_head;

    // Outcome
    uint8_t   player_died;
    uint8_t   level_complete;
};

// =============================================================================
// Interleaved Memory Layout (for coalesced access)
// =============================================================================

// Instead of: GPUInstance instances[N]
// We use:     InterleaveField<T> field[N] where adjacent threads access adjacent memory

// Example for mobjs:
// Standard:   instances[instance_id].mobjs[mobj_id].x
// Interleaved: mobj_x[mobj_id * NUM_INSTANCES + instance_id]

// This header defines the structs. The actual interleaved arrays are in doom_main.cu

// =============================================================================
// Input Buffer Layout
// =============================================================================

// Pre-loaded actions from contestants
// Layout: input_buffer[tick * MAX_INSTANCES + instance_id]
// Each instance can have different horizon lengths

struct InputBuffer {
    TicCmd* commands;           // GPU memory
    int32_t* horizon_lengths;   // Per-instance: how many ticks loaded
    int32_t max_horizon;        // INPUT_HORIZON
};

// =============================================================================
// Checkpoint Buffer Layout
// =============================================================================

// Output checkpoints written by GPU
// Layout: checkpoints[slot * MAX_INSTANCES + instance_id]

struct CheckpointBuffer {
    Checkpoint* slots;          // GPU memory
    int32_t* write_heads;       // Per-instance: next write slot
    int32_t checkpoint_interval; // Write every N ticks
};

// =============================================================================
// Kernel Configuration
// =============================================================================

struct SimConfig {
    int num_instances;
    int ticks_to_simulate;
    int checkpoint_interval;

    // Level data (shared, read-only)
    // TODO: Add BSP, sectors, linedefs when we port collision
};

#endif // DOOM_TYPES_CUH
