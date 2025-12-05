#ifndef ANGBAND_DUNGEON_CUH
#define ANGBAND_DUNGEON_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Dungeon dimensions from constants.txt
#define DUNGEON_WIDTH 198
#define DUNGEON_HEIGHT 66
#define DUNGEON_SIZE (DUNGEON_WIDTH * DUNGEON_HEIGHT)

// Terrain types (simplified from Angband's feature system)
#define TERRAIN_WALL 0
#define TERRAIN_FLOOR 1
#define TERRAIN_DOOR 2
#define TERRAIN_STAIRS_UP 3
#define TERRAIN_STAIRS_DOWN 4
#define TERRAIN_RUBBLE 5

// Room generation constants
#define MAX_ROOMS 10
#define MIN_ROOMS 5
#define MIN_ROOM_WIDTH 4
#define MAX_ROOM_WIDTH 11
#define MIN_ROOM_HEIGHT 4
#define MAX_ROOM_HEIGHT 11

// Room structure for generation
struct Room {
    int16_t x1, y1;  // top-left corner
    int16_t x2, y2;  // bottom-right corner
};

/**
 * Get terrain at position (with bounds checking)
 */
__device__ inline uint8_t get_terrain(uint8_t* terrain, int x, int y) {
    if (x < 0 || x >= DUNGEON_WIDTH || y < 0 || y >= DUNGEON_HEIGHT) {
        return TERRAIN_WALL;
    }
    return terrain[y * DUNGEON_WIDTH + x];
}

/**
 * Set terrain at position (with bounds checking)
 */
__device__ inline void set_terrain(uint8_t* terrain, int x, int y, uint8_t type) {
    if (x >= 0 && x < DUNGEON_WIDTH && y >= 0 && y < DUNGEON_HEIGHT) {
        terrain[y * DUNGEON_WIDTH + x] = type;
    }
}

/**
 * Check if two rooms overlap (with 1-tile padding)
 */
__device__ bool rooms_overlap(const Room& r1, const Room& r2) {
    // Add 1-tile padding around each room to ensure spacing
    return !(r1.x2 + 1 < r2.x1 || r1.x1 - 1 > r2.x2 ||
             r1.y2 + 1 < r2.y1 || r1.y1 - 1 > r2.y2);
}

/**
 * Check if a room fits within dungeon bounds
 */
__device__ bool room_in_bounds(const Room& room) {
    return room.x1 > 0 && room.x2 < DUNGEON_WIDTH - 1 &&
           room.y1 > 0 && room.y2 < DUNGEON_HEIGHT - 1;
}

/**
 * Carve a rectangular room into terrain
 * Creates floor tiles and walls around the perimeter
 */
__device__ void carve_room(uint8_t* terrain, int x1, int y1, int x2, int y2) {
    // Carve the floor
    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            set_terrain(terrain, x, y, TERRAIN_FLOOR);
        }
    }

    // Optionally add doors at random points on room perimeter
    // (simplified - just leave as floor for now, corridors will connect)
}

/**
 * Carve an L-shaped corridor between two points
 * Randomly chooses horizontal-then-vertical or vertical-then-horizontal
 */
__device__ void carve_corridor(uint8_t* terrain, int x1, int y1, int x2, int y2, curandState* rng) {
    // Choose direction: 0 = horizontal first, 1 = vertical first
    bool horizontal_first = (curand(rng) % 2) == 0;

    if (horizontal_first) {
        // Horizontal segment
        int x_start = (x1 < x2) ? x1 : x2;
        int x_end = (x1 < x2) ? x2 : x1;
        for (int x = x_start; x <= x_end; x++) {
            set_terrain(terrain, x, y1, TERRAIN_FLOOR);
        }

        // Vertical segment
        int y_start = (y1 < y2) ? y1 : y2;
        int y_end = (y1 < y2) ? y2 : y1;
        for (int y = y_start; y <= y_end; y++) {
            set_terrain(terrain, x2, y, TERRAIN_FLOOR);
        }
    } else {
        // Vertical segment
        int y_start = (y1 < y2) ? y1 : y2;
        int y_end = (y1 < y2) ? y2 : y1;
        for (int y = y_start; y <= y_end; y++) {
            set_terrain(terrain, x1, y, TERRAIN_FLOOR);
        }

        // Horizontal segment
        int x_start = (x1 < x2) ? x1 : x2;
        int x_end = (x1 < x2) ? x2 : x1;
        for (int x = x_start; x <= x_end; x++) {
            set_terrain(terrain, x, y2, TERRAIN_FLOOR);
        }
    }
}

/**
 * Place stairs at a specific location
 */
__device__ void place_stairs(uint8_t* terrain, int x, int y, uint8_t type) {
    set_terrain(terrain, x, y, type);
}

/**
 * Find a random floor tile in a room
 */
__device__ void get_room_floor(const Room& room, int* x, int* y, curandState* rng) {
    // Pick a random floor tile within the room (not on the edges)
    int room_width = room.x2 - room.x1 + 1;
    int room_height = room.y2 - room.y1 + 1;

    *x = room.x1 + (curand(rng) % room_width);
    *y = room.y1 + (curand(rng) % room_height);
}

/**
 * Generate a dungeon level
 *
 * @param terrain Output array [DUNGEON_SIZE] for this instance
 * @param depth Dungeon level (1-100)
 * @param player_x Output starting position X
 * @param player_y Output starting position Y
 * @param stair_down_x Output position of down stairs X
 * @param stair_down_y Output position of down stairs Y
 * @param rng Random number generator state
 */
__device__ void generate_level(
    uint8_t* terrain,
    int depth,
    int16_t* player_x, int16_t* player_y,
    int16_t* stair_down_x, int16_t* stair_down_y,
    curandState* rng
) {
    // Step 1: Fill entire level with walls
    for (int i = 0; i < DUNGEON_SIZE; i++) {
        terrain[i] = TERRAIN_WALL;
    }

    // Step 2: Generate random rooms
    Room rooms[MAX_ROOMS];
    int num_rooms = MIN_ROOMS + (curand(rng) % (MAX_ROOMS - MIN_ROOMS + 1));
    int rooms_placed = 0;

    // Try to place each room (with maximum attempts)
    const int MAX_ATTEMPTS = 1000;
    for (int attempt = 0; attempt < MAX_ATTEMPTS && rooms_placed < num_rooms; attempt++) {
        Room new_room;

        // Generate random room dimensions
        int width = MIN_ROOM_WIDTH + (curand(rng) % (MAX_ROOM_WIDTH - MIN_ROOM_WIDTH + 1));
        int height = MIN_ROOM_HEIGHT + (curand(rng) % (MAX_ROOM_HEIGHT - MIN_ROOM_HEIGHT + 1));

        // Generate random position
        new_room.x1 = 2 + (curand(rng) % (DUNGEON_WIDTH - width - 4));
        new_room.y1 = 2 + (curand(rng) % (DUNGEON_HEIGHT - height - 4));
        new_room.x2 = new_room.x1 + width - 1;
        new_room.y2 = new_room.y1 + height - 1;

        // Check if room is valid
        if (!room_in_bounds(new_room)) {
            continue;
        }

        // Check for overlap with existing rooms
        bool overlaps = false;
        for (int i = 0; i < rooms_placed; i++) {
            if (rooms_overlap(new_room, rooms[i])) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            rooms[rooms_placed] = new_room;
            carve_room(terrain, new_room.x1, new_room.y1, new_room.x2, new_room.y2);
            rooms_placed++;
        }
    }

    // Step 3: Connect rooms with corridors
    // Connect each room to the next room in sequence
    for (int i = 0; i < rooms_placed - 1; i++) {
        // Get center points of both rooms
        int x1 = (rooms[i].x1 + rooms[i].x2) / 2;
        int y1 = (rooms[i].y1 + rooms[i].y2) / 2;
        int x2 = (rooms[i + 1].x1 + rooms[i + 1].x2) / 2;
        int y2 = (rooms[i + 1].y1 + rooms[i + 1].y2) / 2;

        carve_corridor(terrain, x1, y1, x2, y2, rng);
    }

    // Step 4: Place stairs up in first room
    if (rooms_placed > 0) {
        int stair_up_x, stair_up_y;
        get_room_floor(rooms[0], &stair_up_x, &stair_up_y, rng);
        place_stairs(terrain, stair_up_x, stair_up_y, TERRAIN_STAIRS_UP);

        // Place player near stairs up
        *player_x = stair_up_x;
        *player_y = stair_up_y;
    }

    // Step 5: Place stairs down in last room
    if (rooms_placed > 1) {
        int stair_dn_x, stair_dn_y;
        get_room_floor(rooms[rooms_placed - 1], &stair_dn_x, &stair_dn_y, rng);
        place_stairs(terrain, stair_dn_x, stair_dn_y, TERRAIN_STAIRS_DOWN);

        *stair_down_x = stair_dn_x;
        *stair_down_y = stair_dn_y;
    }

    // Note: Monster and item placement will be handled in a separate phase
    // This focuses on dungeon structure generation
}

/**
 * Count the number of specific terrain types in the dungeon
 * Useful for validation and statistics
 */
__device__ int count_terrain_type(uint8_t* terrain, uint8_t type) {
    int count = 0;
    for (int i = 0; i < DUNGEON_SIZE; i++) {
        if (terrain[i] == type) {
            count++;
        }
    }
    return count;
}

/**
 * Check if dungeon generation was successful
 * Returns true if the dungeon has floor tiles and stairs
 */
__device__ bool validate_dungeon(uint8_t* terrain) {
    int floor_count = count_terrain_type(terrain, TERRAIN_FLOOR);
    int stairs_up = count_terrain_type(terrain, TERRAIN_STAIRS_UP);
    int stairs_down = count_terrain_type(terrain, TERRAIN_STAIRS_DOWN);

    // Must have some floor space and at least one set of stairs
    return floor_count > 100 && stairs_up > 0 && stairs_down > 0;
}

#endif // ANGBAND_DUNGEON_CUH
