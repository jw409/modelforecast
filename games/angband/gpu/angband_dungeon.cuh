/*
 * ANGBAND DUNGEON - GPU Implementation
 *
 * Real dungeon generation with rooms and corridors
 * Based on Angband's generate.c
 *
 * STUB IMPLEMENTATION - Replace with real generation
 */

#ifndef ANGBAND_DUNGEON_CUH
#define ANGBAND_DUNGEON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../../common/interleaved.h"
#include "borg_state.h"

// ============================================================================
// TERRAIN TYPES
// ============================================================================

#define TERRAIN_WALL        0
#define TERRAIN_FLOOR       1
#define TERRAIN_DOOR        2
#define TERRAIN_STAIRS_DOWN 3
#define TERRAIN_STAIRS_UP   4
#define TERRAIN_RUBBLE      5
#define TERRAIN_LAVA        6
#define TERRAIN_WATER       7

// ============================================================================
// DUNGEON GENERATION
// ============================================================================

// Generate a complete dungeon level
// Based on cave_gen() in generate.c
__device__ void generate_level(
    uint8_t* dungeon_terrain,     // Output: terrain grid (interleaved)
    int depth,                    // Dungeon depth (affects room types)
    int* player_x,                // Output: player start X
    int* player_y,                // Output: player start Y
    int* stairs_x,                // Output: down stairs X
    int* stairs_y,                // Output: down stairs Y
    int instance_id,              // Which instance
    int num_instances,            // Total instances (for interleaving)
    curandState* rng              // RNG state
) {
    // STUB: Simple rectangular room generation
    // REAL: Should implement rooms + corridors, vault generation, etc.

    // Fill with walls
    for (int y = 0; y < DUNGEON_HEIGHT; y++) {
        for (int x = 0; x < DUNGEON_WIDTH; x++) {
            int cell_idx = y * DUNGEON_WIDTH + x;
            ISET(dungeon_terrain, cell_idx, instance_id, num_instances, TERRAIN_WALL);
        }
    }

    // Generate 3-5 rooms
    int num_rooms = 3 + curand(rng) % 3;

    for (int r = 0; r < num_rooms; r++) {
        // Random room size and position
        int room_w = 5 + curand(rng) % 10;  // 5-14 wide
        int room_h = 3 + curand(rng) % 6;   // 3-8 tall
        int room_x = 5 + curand(rng) % (DUNGEON_WIDTH - room_w - 10);
        int room_y = 2 + curand(rng) % (DUNGEON_HEIGHT - room_h - 4);

        // Carve out room
        for (int y = room_y; y < room_y + room_h; y++) {
            for (int x = room_x; x < room_x + room_w; x++) {
                int cell_idx = y * DUNGEON_WIDTH + x;
                ISET(dungeon_terrain, cell_idx, instance_id, num_instances, TERRAIN_FLOOR);
            }
        }

        // First room: place player
        if (r == 0) {
            *player_x = room_x + room_w / 2;
            *player_y = room_y + room_h / 2;
        }

        // Last room: place stairs
        if (r == num_rooms - 1) {
            *stairs_x = room_x + room_w / 2;
            *stairs_y = room_y + room_h / 2;
            int stairs_idx = (*stairs_y) * DUNGEON_WIDTH + (*stairs_x);
            ISET(dungeon_terrain, stairs_idx, instance_id, num_instances, TERRAIN_STAIRS_DOWN);
        }
    }

    // STUB: Simple corridor connecting rooms
    // REAL: Should implement proper corridor carving between all rooms
    // For now, just ensure some connectivity by carving horizontal corridors
    for (int y = DUNGEON_HEIGHT / 2; y < DUNGEON_HEIGHT / 2 + 1; y++) {
        for (int x = 10; x < DUNGEON_WIDTH - 10; x++) {
            int cell_idx = y * DUNGEON_WIDTH + x;
            uint8_t terrain = IGET(dungeon_terrain, cell_idx, instance_id, num_instances);
            if (terrain == TERRAIN_WALL) {
                ISET(dungeon_terrain, cell_idx, instance_id, num_instances, TERRAIN_FLOOR);
            }
        }
    }
}

// ============================================================================
// TERRAIN QUERIES
// ============================================================================

// Check if terrain is walkable
__device__ bool is_walkable(
    uint8_t* dungeon_terrain,     // Terrain grid (interleaved)
    int x, int y,                 // Position to check
    int instance_id,
    int num_instances
) {
    // Bounds check
    if (x < 0 || x >= DUNGEON_WIDTH || y < 0 || y >= DUNGEON_HEIGHT) {
        return false;
    }

    int cell_idx = y * DUNGEON_WIDTH + x;
    uint8_t terrain = IGET(dungeon_terrain, cell_idx, instance_id, num_instances);

    // Walkable terrain types
    switch (terrain) {
        case TERRAIN_FLOOR:
        case TERRAIN_DOOR:
        case TERRAIN_STAIRS_DOWN:
        case TERRAIN_STAIRS_UP:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// TELEPORTATION
// ============================================================================

// Teleport player to random walkable location
// Based on teleport_player() in player-util.c
__device__ void teleport_player(
    int* player_x,                // Input/Output: player X
    int* player_y,                // Input/Output: player Y
    uint8_t* dungeon_terrain,     // Terrain grid
    int depth,                    // Current depth (affects teleport range)
    int instance_id,
    int num_instances,
    curandState* rng
) {
    // STUB: Simple random teleport to any walkable cell
    // REAL: Should implement distance-based teleport, line-of-sight checks

    // Try up to 100 times to find a walkable cell
    for (int attempt = 0; attempt < 100; attempt++) {
        int new_x = curand(rng) % DUNGEON_WIDTH;
        int new_y = curand(rng) % DUNGEON_HEIGHT;

        if (is_walkable(dungeon_terrain, new_x, new_y, instance_id, num_instances)) {
            *player_x = new_x;
            *player_y = new_y;
            return;
        }
    }

    // Fallback: don't move (couldn't find walkable cell)
}

// ============================================================================
// ADVANCED GENERATION (Future)
// ============================================================================

// Generate a vault (special room with treasure/monsters)
__device__ void generate_vault(
    uint8_t* dungeon_terrain,
    int vault_x, int vault_y,
    int vault_type,
    int instance_id,
    int num_instances,
    curandState* rng
) {
    // STUB: Not implemented yet
    // REAL: Should place prefab vault templates
}

// Connect two rooms with corridor
__device__ void carve_corridor(
    uint8_t* dungeon_terrain,
    int x1, int y1,
    int x2, int y2,
    int instance_id,
    int num_instances
) {
    // STUB: Simple L-shaped corridor
    // REAL: Should implement varied corridor styles

    // Horizontal first
    int start_x = min(x1, x2);
    int end_x = max(x1, x2);
    for (int x = start_x; x <= end_x; x++) {
        int cell_idx = y1 * DUNGEON_WIDTH + x;
        ISET(dungeon_terrain, cell_idx, instance_id, num_instances, TERRAIN_FLOOR);
    }

    // Then vertical
    int start_y = min(y1, y2);
    int end_y = max(y1, y2);
    for (int y = start_y; y <= end_y; y++) {
        int cell_idx = y * DUNGEON_WIDTH + x2;
        ISET(dungeon_terrain, cell_idx, instance_id, num_instances, TERRAIN_FLOOR);
    }
}

#endif // ANGBAND_DUNGEON_CUH
