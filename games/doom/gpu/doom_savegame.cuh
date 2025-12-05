/**
 * GPU DOOM Savegame Format
 *
 * Compatible with Chocolate Doom / original DOOM save format.
 * GPU writes these, browser loads them to resume play.
 *
 * Original format from p_saveg.c (id Software)
 */

#ifndef DOOM_SAVEGAME_CUH
#define DOOM_SAVEGAME_CUH

#include "doom_types.cuh"
#include <cstdint>

// =============================================================================
// Savegame Header (matches original DOOM format)
// =============================================================================

#define SAVEGAME_VERSION 109  // Chocolate Doom compatible
#define SAVE_DESCRIPTION_LEN 32

#pragma pack(push, 1)

struct SaveHeader {
    char description[SAVE_DESCRIPTION_LEN];  // "AI Arena - Tick 12345"
    uint8_t version;                          // SAVEGAME_VERSION
    uint8_t skill;                            // 0-4 (I'm too young to die -> Nightmare)
    uint8_t episode;                          // 1-4
    uint8_t map;                              // 1-9
    uint8_t playeringame[MAXPLAYERS];         // Which players are active
    int32_t leveltime;                        // Tics played on this level
};

// =============================================================================
// Full Game State Snapshot
// =============================================================================

// Maximum sizes for variable-length sections
#define MAX_SECTORS 1024
#define MAX_LINES 4096
#define MAX_SIDES 8192
#define MAX_THINKERS 256

// Sector state (for doors, lifts, lights)
struct SectorState {
    int16_t floorheight;
    int16_t ceilingheight;
    int16_t floorpic;
    int16_t ceilingpic;
    int16_t lightlevel;
    int16_t special;
    int16_t tag;
};

// Line state (for switches, triggers)
struct LineState {
    int16_t flags;
    int16_t special;
    int16_t tag;
};

// Side state (for animated textures)
struct SideState {
    int16_t textureoffset;
    int16_t rowoffset;
    int16_t toptexture;
    int16_t bottomtexture;
    int16_t midtexture;
};

// Thinker state (doors, platforms, lights, ceilings)
enum ThinkerType : uint8_t {
    TT_NONE = 0,
    TT_CEILING,      // T_MoveCeiling
    TT_DOOR,         // T_VerticalDoor
    TT_FLOOR,        // T_MoveFloor
    TT_PLAT,         // T_PlatRaise
    TT_FLASH,        // T_LightFlash
    TT_STROBE,       // T_StrobeFlash
    TT_GLOW,         // T_Glow
    TT_MOBJ          // P_MobjThinker (monsters, projectiles)
};

// Generic thinker save structure
struct ThinkerState {
    ThinkerType type;
    uint8_t _pad[3];

    // Union of all thinker types (simplified)
    union {
        struct {
            int32_t sector_idx;
            fixed_t speed;
            fixed_t low;
            fixed_t high;
            int32_t direction;
            int32_t tag;
            int32_t type_specific;
        } mover;

        struct {
            fixed_t x, y, z;
            fixed_t momx, momy, momz;
            angle_t angle;
            int32_t type;
            int32_t state_idx;
            int32_t tics;
            int32_t health;
            uint32_t flags;
            int32_t movedir;
            int32_t movecount;
            int32_t target_idx;  // -1 = none, else mobj index
            int32_t reactiontime;
            int32_t threshold;
            int32_t tracer_idx;
        } mobj;
    };
};

// Player weapon sprite state
struct PspriteState {
    int32_t state_idx;  // Index into states[] or -1
    int32_t tics;
    fixed_t sx, sy;
};

// Full player state (for savegame)
struct PlayerSaveState {
    // From player_t
    int32_t health;
    int32_t armorpoints;
    int32_t armortype;
    int32_t playerstate;  // PST_LIVE, PST_DEAD, PST_REBORN

    // Weapons
    int32_t readyweapon;
    int32_t pendingweapon;
    uint32_t weaponowned;  // Bitfield

    // Ammo
    int32_t ammo[NUMAMMO];
    int32_t maxammo[NUMAMMO];

    // Powers (tic counters)
    int32_t powers[NUMPOWERS];

    // Keys
    uint32_t cards;  // Bitfield

    // Combat state
    int32_t attackdown;
    int32_t usedown;
    int32_t refire;

    // View
    fixed_t viewz;
    fixed_t viewheight;
    fixed_t deltaviewheight;
    fixed_t bob;

    // Stats
    int32_t killcount;
    int32_t itemcount;
    int32_t secretcount;

    // Screen effects
    int32_t damagecount;
    int32_t bonuscount;
    int32_t extralight;
    int32_t fixedcolormap;

    // Weapon sprites
    PspriteState psprites[2];  // NUMPSPRITES

    // Cheats (we add these!)
    uint32_t cheats;  // CF_GODMODE, CF_NOCLIP, etc.
};

// =============================================================================
// Complete Savegame Structure
// =============================================================================

struct FullSavegame {
    SaveHeader header;

    // Players
    PlayerSaveState players[MAXPLAYERS];

    // World state
    int32_t num_sectors;
    int32_t num_lines;
    int32_t num_sides;
    SectorState sectors[MAX_SECTORS];
    LineState lines[MAX_LINES];
    SideState sides[MAX_SIDES];

    // Active thinkers (doors, platforms, monsters, projectiles)
    int32_t num_thinkers;
    ThinkerState thinkers[MAX_THINKERS];

    // RNG state (for deterministic replay)
    int32_t rng_index;

    // Terminator
    uint8_t terminator;  // 0x1d
};

#pragma pack(pop)

// =============================================================================
// Cheat Flags (browser menu activates these)
// =============================================================================

#define CF_GODMODE       0x0001  // IDDQD
#define CF_NOCLIP        0x0002  // IDCLIP
#define CF_IDKFA         0x0004  // All keys, weapons, ammo
#define CF_IDFA          0x0008  // All weapons, ammo (no keys)
#define CF_UNLIMITED_AMMO 0x0010 // Never runs out
#define CF_ALLMAP        0x0020  // IDDT (show full map)
#define CF_NOTARGET      0x0040  // Monsters ignore you

// =============================================================================
// GPU → Savegame Serialization
// =============================================================================

// These run on GPU, write directly to savegame buffer
__device__ void SerializePlayer_GPU(
    int instance_id,
    int player_idx,
    PlayerSaveState* out
);

__device__ void SerializeMobj_GPU(
    int instance_id,
    int mobj_idx,
    ThinkerState* out
);

// Host function: Extract full savegame from GPU state
void ExtractSavegame(
    int instance_id,
    int tick,
    FullSavegame* out
);

// =============================================================================
// Savegame → File (for browser download)
// =============================================================================

// Write savegame to buffer in Chocolate Doom compatible format
size_t WriteSavegameBuffer(
    const FullSavegame* save,
    uint8_t* buffer,
    size_t buffer_size
);

// Generate downloadable .sav file
bool WriteSavegameFile(
    const FullSavegame* save,
    const char* filename
);

#endif // DOOM_SAVEGAME_CUH
