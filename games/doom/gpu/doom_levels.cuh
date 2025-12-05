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
 */
__device__ void CheckLevelCompletion(int instance_id, int tick) {
    // Skip if dead or already won
    if (!d_player_alive[instance_id] || d_game_won[instance_id]) {
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
 * Initialize level state for instance
 * Called during instance setup
 */
__device__ void InitLevelState(int instance_id) {
    d_current_level[instance_id] = 0;      // Start at E1M1
    d_level_complete[instance_id] = 0;
    d_game_won[instance_id] = 0;
    d_completion_tick[instance_id] = 0;
    d_monsters_killed[instance_id] = 0;
    d_monsters_total[instance_id] = c_monster_counts[0];  // E1M1 monster count
}

/**
 * Record monster kill
 * Called when a monster dies (when monster system is ready)
 */
__device__ void RecordMonsterKill(int instance_id) {
    // Atomic increment to prevent race conditions if we add monster AI threads
    atomicAdd(&d_monsters_killed[instance_id], 1);
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
