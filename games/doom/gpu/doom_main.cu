/**
 * GPU DOOM - Main Kernel
 *
 * Phase 2: Monsters and Combat
 * - Monsters chase and attack player
 * - Player can shoot monsters
 * - Death and damage system
 *
 * Phase 3: Level Completion
 * - Exit triggers for E1M1, E1M2, E1M3
 * - Level progression and transitions
 * - Win condition tracking
 *
 * Original: id Software DOOM (1993)
 * GPU Port: MIT License
 */

#include "doom_types.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// =============================================================================
// Interleaved Memory Arrays (Global Device Memory)
// =============================================================================

// Player state (interleaved: player_health[instance_id])
__device__ int32_t* d_player_health;
__device__ int32_t* d_player_armor;
__device__ fixed_t* d_player_x;
__device__ fixed_t* d_player_y;
__device__ fixed_t* d_player_z;
__device__ angle_t* d_player_angle;
__device__ fixed_t* d_player_momx;
__device__ fixed_t* d_player_momy;
__device__ fixed_t* d_player_momz;
__device__ uint8_t* d_player_alive;
__device__ int16_t* d_player_kills;

// Monster state (interleaved: monster_x[monster_id * num_instances + instance_id])
#define MAX_MONSTERS 1024  // Per instance (support heavy maps)
__device__ fixed_t* d_monster_x;
__device__ fixed_t* d_monster_y;
__device__ fixed_t* d_monster_z;
__device__ angle_t* d_monster_angle;
__device__ int32_t* d_monster_health;
__device__ uint8_t* d_monster_type;
__device__ uint8_t* d_monster_alive;
__device__ int16_t* d_monster_target_idx;  // -1 = no target, 0 = player
__device__ uint8_t* d_monster_movedir;
__device__ int16_t* d_monster_movecount;
__device__ int16_t* d_monster_reactiontime;
__device__ int16_t* d_monster_count;

// Input buffer (interleaved: input[tick * num_instances + instance_id])
__device__ TicCmd* d_input_buffer;

// Checkpoint buffer (interleaved: checkpoint[slot * num_instances + instance_id])
__device__ Checkpoint* d_checkpoints;
__device__ int32_t* d_checkpoint_count;

// Level state (defined in doom_levels.cuh, allocated here)
__device__ uint8_t* d_current_level;
__device__ uint8_t* d_level_complete;
__device__ uint8_t* d_game_won;
__device__ int32_t* d_completion_tick;
__device__ int32_t* d_monsters_killed;
__device__ int32_t* d_monsters_total;

// Configuration
__constant__ int c_num_instances;
__constant__ int c_checkpoint_interval;

// =============================================================================
// Monster Types and Stats
// =============================================================================

// Note: MonsterType (MT_*) is defined in doom_types.cuh

struct MonsterInfo {
    int health;
    int melee_damage_min;
    int melee_damage_max;
    int ranged_damage_min;
    int ranged_damage_max;
    int melee_range;       // In FRACUNIT
    int missile_range;     // In FRACUNIT
    int speed;             // Movement speed
};

__constant__ MonsterInfo c_monster_info[32] = {
    // MT_PLAYER (0)
    {100, 0, 0, 0, 0, 0, 0, 0},
    // MT_POSSESSED (1) - Zombieman
    {20, 0, 0, 3, 15, 0, 2048 * FRACUNIT, 8},
    // MT_SHOTGUY (2)
    {30, 0, 0, 3, 15, 0, 2048 * FRACUNIT, 8},
    // MT_IMP (3)
    {60, 3, 24, 3, 24, 64 * FRACUNIT, 2048 * FRACUNIT, 8},
    // MT_DEMON (4) - Pinky
    {150, 4, 40, 0, 0, 64 * FRACUNIT, 0, 10},
    // MT_SPECTRE (5)
    {150, 4, 40, 0, 0, 64 * FRACUNIT, 0, 10},
    // MT_CACODEMON (6)
    {400, 0, 0, 5, 40, 0, 2048 * FRACUNIT, 8},
    // MT_BRUISER (7) - Baron
    {1000, 10, 80, 8, 64, 64 * FRACUNIT, 2048 * FRACUNIT, 8},
    // Fill rest with zeros
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}
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

// =============================================================================
// Fixed Point Math (from DOOM source)
// =============================================================================

#define FIXEDMUL_DEFINED
__device__ __forceinline__ fixed_t FixedMul(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a * b) >> FRACBITS);
}

__device__ __forceinline__ fixed_t FixedDiv(fixed_t a, fixed_t b) {
    if ((abs(a) >> 14) >= abs(b)) {
        return (a ^ b) < 0 ? INT32_MIN : INT32_MAX;
    }
    return (fixed_t)(((int64_t)a << FRACBITS) / b);
}

// =============================================================================
// Level Completion System (include after device variable declarations)
// =============================================================================

#include "doom_levels.cuh"

// =============================================================================
// Angle/Trig Tables (simplified - full version loads from WAD)
// =============================================================================

// Fine angles: 8192 entries for full circle
#define FINEANGLES      8192
#define FINEMASK        (FINEANGLES - 1)
#define ANGLETOFINESHIFT 19

// Precomputed sine table (first quadrant, others derived)
__constant__ fixed_t c_finesine[FINEANGLES / 4 + 1];

__device__ fixed_t finesine(int idx) {
    idx &= FINEMASK;
    if (idx < FINEANGLES / 4) return c_finesine[idx];
    if (idx < FINEANGLES / 2) return c_finesine[FINEANGLES / 2 - idx];
    if (idx < 3 * FINEANGLES / 4) return -c_finesine[idx - FINEANGLES / 2];
    return -c_finesine[FINEANGLES - idx];
}

__device__ fixed_t finecosine(int idx) {
    return finesine(idx + FINEANGLES / 4);
}

// =============================================================================
// Random Number Generator (simplified from P_Random)
// =============================================================================

// Simple LCG for GPU (each instance uses different seed)
__device__ uint32_t gpu_random(int instance_id, int tick) {
    uint32_t seed = instance_id * 1103515245 + tick * 12345;
    return (seed / 65536) % 256;
}

// =============================================================================
// Combat Helper Functions
// =============================================================================

// Approximate distance (Manhattan distance scaled by 0.7 for diagonal)
__device__ fixed_t P_AproxDistance(fixed_t dx, fixed_t dy) {
    dx = abs(dx);
    dy = abs(dy);
    if (dx < dy)
        return dx + dy - (dx >> 1);
    return dx + dy - (dy >> 1);
}

// Check if monster can see player (simplified - no BSP check for Phase 2)
__device__ bool P_CheckSight_Simple(
    fixed_t mon_x, fixed_t mon_y,
    fixed_t target_x, fixed_t target_y)
{
    // Phase 2: Always visible if in range
    // Phase 3 will add BSP line-of-sight checks
    fixed_t dx = target_x - mon_x;
    fixed_t dy = target_y - mon_y;
    fixed_t dist = P_AproxDistance(dx, dy);
    return dist < MISSILERANGE * 2;
}

// Damage player
__device__ void P_DamagePlayer(int instance_id, int damage) {
    int health = d_player_health[instance_id];
    int armor = d_player_armor[instance_id];

    // Armor absorbs some damage
    if (armor > 0) {
        int absorbed = damage / 3;
        armor -= absorbed;
        damage -= absorbed;
        if (armor < 0) {
            damage += armor;  // Overflow damage
            armor = 0;
        }
        d_player_armor[instance_id] = armor;
    }

    health -= damage;
    if (health <= 0) {
        health = 0;
        d_player_alive[instance_id] = 0;
    }
    d_player_health[instance_id] = health;
}

// Damage monster
__device__ void P_DamageMonster(int monster_idx, int instance_id, int num_instances, int damage) {
    int idx = monster_idx * num_instances + instance_id;
    int health = d_monster_health[idx];
    health -= damage;

    if (health <= 0) {
        health = 0;
        d_monster_alive[idx] = 0;
        // Player gets kill credit
        d_player_kills[instance_id]++;
    }

    d_monster_health[idx] = health;
}

// =============================================================================
// P_PlayerThink - GPU Port (Simplified)
// =============================================================================

// Movement constants (from original)
#define MAXMOVE         (30 * FRACUNIT)
#define FRICTION        0xe800  // ~0.90625 in fixed point
#define STOPSPEED       0x1000  // Below this, stop completely

__device__ void P_PlayerThink_GPU(int instance_id, int tick, int num_instances) {
    // Get input for this tick
    int input_idx = tick * num_instances + instance_id;
    TicCmd cmd = d_input_buffer[input_idx];

    // Skip if dead
    if (!d_player_alive[instance_id]) return;

    // Current state
    fixed_t x = d_player_x[instance_id];
    fixed_t y = d_player_y[instance_id];
    angle_t angle = d_player_angle[instance_id];
    fixed_t momx = d_player_momx[instance_id];
    fixed_t momy = d_player_momy[instance_id];

    // Apply turning
    angle += (cmd.angleturn << 16);

    // Calculate movement vector
    int fineangle = angle >> ANGLETOFINESHIFT;

    // Forward/backward movement
    if (cmd.forwardmove) {
        fixed_t thrust = cmd.forwardmove * 2048;  // Scale factor
        momx += FixedMul(thrust, finecosine(fineangle));
        momy += FixedMul(thrust, finesine(fineangle));
    }

    // Strafe movement
    if (cmd.sidemove) {
        fixed_t thrust = cmd.sidemove * 2048;
        fineangle = (angle - ANG90) >> ANGLETOFINESHIFT;
        momx += FixedMul(thrust, finecosine(fineangle));
        momy += FixedMul(thrust, finesine(fineangle));
    }

    // Clamp momentum
    if (momx > MAXMOVE) momx = MAXMOVE;
    if (momx < -MAXMOVE) momx = -MAXMOVE;
    if (momy > MAXMOVE) momy = MAXMOVE;
    if (momy < -MAXMOVE) momy = -MAXMOVE;

    // Apply momentum to position
    // NOTE: Real DOOM does P_TryMove with collision detection here
    // Phase 1: Just move freely
    x += momx;
    y += momy;

    // Apply friction
    momx = FixedMul(momx, FRICTION);
    momy = FixedMul(momy, FRICTION);

    // Stop if very slow
    if (abs(momx) < STOPSPEED && abs(momy) < STOPSPEED) {
        momx = 0;
        momy = 0;
    }

    // Handle weapon fire (BT_ATTACK button)
    if (cmd.buttons & BT_ATTACK) {
        // Pistol: Hitscan attack straight ahead
        // Find nearest monster in front of player
        int nearest_monster = -1;
        fixed_t nearest_dist = INT32_MAX;

        int num_monsters = d_monster_count[instance_id];
        for (int m = 0; m < num_monsters; m++) {
            int midx = m * num_instances + instance_id;
            if (!d_monster_alive[midx]) continue;

            fixed_t mx = d_monster_x[midx];
            fixed_t my = d_monster_y[midx];

            // Check if roughly in front of player (within 45 degrees)
            fixed_t dx = mx - x;
            fixed_t dy = my - y;

            // Convert to angle
            // Simplified angle check: use dot product
            int fineangle_player = angle >> ANGLETOFINESHIFT;
            fixed_t forward_x = finecosine(fineangle_player);
            fixed_t forward_y = finesine(fineangle_player);

            // Dot product (normalized roughly)
            int64_t dot = ((int64_t)dx * forward_x + (int64_t)dy * forward_y) >> FRACBITS;

            if (dot > 0) {  // In front
                fixed_t dist = P_AproxDistance(dx, dy);
                if (dist < nearest_dist && dist < MISSILERANGE) {
                    nearest_dist = dist;
                    nearest_monster = m;
                }
            }
        }

        // Hit the nearest monster
        if (nearest_monster >= 0) {
            // Pistol damage: 5-15 (random 1-3 * 5)
            int damage = ((gpu_random(instance_id, tick) % 3) + 1) * 5;
            P_DamageMonster(nearest_monster, instance_id, num_instances, damage);
        }
    }

    // Write back state
    d_player_x[instance_id] = x;
    d_player_y[instance_id] = y;
    d_player_angle[instance_id] = angle;
    d_player_momx[instance_id] = momx;
    d_player_momy[instance_id] = momy;
}

// =============================================================================
// Write Checkpoint
// =============================================================================

__device__ void WriteCheckpoint_GPU(int instance_id, int tick, int num_instances) {
    int slot = d_checkpoint_count[instance_id];
    if (slot >= CHECKPOINT_SLOTS) return;  // Buffer full

    int idx = slot * num_instances + instance_id;

    Checkpoint cp;
    cp.tick = tick;
    cp.health = d_player_health[instance_id];
    cp.armor = d_player_armor[instance_id];
    cp.x = d_player_x[instance_id];
    cp.y = d_player_y[instance_id];
    cp.z = d_player_z[instance_id];
    cp.angle = d_player_angle[instance_id];
    cp.alive = d_player_alive[instance_id];
    cp.kills = 0;  // TODO: track kills
    cp.items = 0;
    cp.secrets = 0;
    cp.weapon = 0;
    cp.monsters_visible = 0;
    cp.projectiles_nearby = 0;

    d_checkpoints[idx] = cp;
    d_checkpoint_count[instance_id] = slot + 1;
}

// =============================================================================
// Main Simulation Kernel
// =============================================================================

__global__ void doom_simulate(int num_instances, int num_ticks, int checkpoint_interval) {
    int instance_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance_id >= num_instances) return;

    // Initialize level state on first tick
    if (instance_id < num_instances) {
        InitLevelState(instance_id);
    }
    __syncthreads();

    for (int tick = 0; tick < num_ticks; tick++) {
        // Skip if game already won
        if (d_game_won[instance_id]) {
            __syncthreads();
            continue;
        }

        // Player thinking (movement, actions)
        P_PlayerThink_GPU(instance_id, tick, num_instances);

        // Check for level completion (exit trigger)
        CheckLevelCompletion(instance_id, tick);

        // TODO Phase 3: P_RunThinkers_GPU (monsters, projectiles)
        // TODO Phase 3: P_UpdateSpecials_GPU (doors, platforms)

        // Write checkpoint if interval hit
        if (checkpoint_interval > 0 && (tick % checkpoint_interval) == 0) {
            WriteCheckpoint_GPU(instance_id, tick, num_instances);
        }

        // Sync all instances before next tick (important for multiplayer logic)
        __syncthreads();
    }

    // Final checkpoint
    if (checkpoint_interval > 0) {
        WriteCheckpoint_GPU(instance_id, num_ticks, num_instances);
    }
}

// =============================================================================
// Host Code: Memory Management
// =============================================================================

struct DoomArena {
    // Host pointers to device memory
    int32_t* player_health;
    int32_t* player_armor;
    fixed_t* player_x;
    fixed_t* player_y;
    fixed_t* player_z;
    angle_t* player_angle;
    fixed_t* player_momx;
    fixed_t* player_momy;
    fixed_t* player_momz;
    uint8_t* player_alive;
    int16_t* player_kills;

    // Monster state pointers
    fixed_t* monster_x;
    fixed_t* monster_y;
    fixed_t* monster_z;
    angle_t* monster_angle;
    int32_t* monster_health;
    uint8_t* monster_type;
    uint8_t* monster_alive;
    int16_t* monster_target_idx;
    uint8_t* monster_movedir;
    int16_t* monster_movecount;
    int16_t* monster_reactiontime;
    int16_t* monster_count;

    TicCmd* input_buffer;
    Checkpoint* checkpoints;
    int32_t* checkpoint_count;

    // Level state
    uint8_t* current_level;
    uint8_t* level_complete;
    uint8_t* game_won;
    int32_t* completion_tick;
    int32_t* monsters_killed;
    int32_t* monsters_total;

    int num_instances;
    int max_ticks;
};

void InitArena(DoomArena* arena, int num_instances, int max_ticks) {
    arena->num_instances = num_instances;
    arena->max_ticks = max_ticks;

    size_t n = num_instances;
    size_t input_size = n * max_ticks;
    size_t checkpoint_size = n * CHECKPOINT_SLOTS;

    // Allocate player state
    cudaMalloc(&arena->player_health, n * sizeof(int32_t));
    cudaMalloc(&arena->player_armor, n * sizeof(int32_t));
    cudaMalloc(&arena->player_x, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_y, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_z, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_angle, n * sizeof(angle_t));
    cudaMalloc(&arena->player_momx, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_momy, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_momz, n * sizeof(fixed_t));
    cudaMalloc(&arena->player_alive, n * sizeof(uint8_t));
    cudaMalloc(&arena->player_kills, n * sizeof(int16_t));

    // Allocate monster state
    size_t monster_size = n * MAX_MONSTERS;
    cudaMalloc(&arena->monster_x, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_y, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_z, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_angle, monster_size * sizeof(angle_t));
    cudaMalloc(&arena->monster_health, monster_size * sizeof(int32_t));
    cudaMalloc(&arena->monster_type, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_alive, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_target_idx, monster_size * sizeof(int16_t));
    cudaMalloc(&arena->monster_movedir, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_movecount, monster_size * sizeof(int16_t));
    cudaMalloc(&arena->monster_reactiontime, monster_size * sizeof(int16_t));
    cudaMalloc(&arena->monster_count, n * sizeof(int16_t));

    // Copy monster pointers

    // Allocate buffers
    cudaMalloc(&arena->input_buffer, input_size * sizeof(TicCmd));
    cudaMalloc(&arena->checkpoints, checkpoint_size * sizeof(Checkpoint));
    cudaMalloc(&arena->checkpoint_count, n * sizeof(int32_t));

    // Allocate level state
    cudaMalloc(&arena->current_level, n * sizeof(uint8_t));
    cudaMalloc(&arena->level_complete, n * sizeof(uint8_t));
    cudaMalloc(&arena->game_won, n * sizeof(uint8_t));
    cudaMalloc(&arena->completion_tick, n * sizeof(int32_t));
    cudaMalloc(&arena->monsters_killed, n * sizeof(int32_t));
    cudaMalloc(&arena->monsters_total, n * sizeof(int32_t));

    // Copy device pointers to device global vars
    cudaMemcpyToSymbol(d_player_health, &arena->player_health, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_player_armor, &arena->player_armor, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_player_x, &arena->player_x, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_y, &arena->player_y, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_z, &arena->player_z, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_angle, &arena->player_angle, sizeof(angle_t*));
    cudaMemcpyToSymbol(d_player_momx, &arena->player_momx, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_momy, &arena->player_momy, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_momz, &arena->player_momz, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_player_alive, &arena->player_alive, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_player_kills, &arena->player_kills, sizeof(int16_t*));

    // Copy monster pointers
    cudaMemcpyToSymbol(d_monster_x, &arena->monster_x, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_y, &arena->monster_y, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_z, &arena->monster_z, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_angle, &arena->monster_angle, sizeof(angle_t*));
    cudaMemcpyToSymbol(d_monster_health, &arena->monster_health, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_monster_type, &arena->monster_type, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_alive, &arena->monster_alive, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_target_idx, &arena->monster_target_idx, sizeof(int16_t*));
    cudaMemcpyToSymbol(d_monster_movedir, &arena->monster_movedir, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_movecount, &arena->monster_movecount, sizeof(int16_t*));
    cudaMemcpyToSymbol(d_monster_reactiontime, &arena->monster_reactiontime, sizeof(int16_t*));
    cudaMemcpyToSymbol(d_monster_count, &arena->monster_count, sizeof(int16_t*));

    cudaMemcpyToSymbol(d_input_buffer, &arena->input_buffer, sizeof(TicCmd*));
    cudaMemcpyToSymbol(d_checkpoints, &arena->checkpoints, sizeof(Checkpoint*));
    cudaMemcpyToSymbol(d_checkpoint_count, &arena->checkpoint_count, sizeof(int32_t*));

    // Copy level state pointers
    cudaMemcpyToSymbol(d_current_level, &arena->current_level, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_level_complete, &arena->level_complete, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_game_won, &arena->game_won, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_completion_tick, &arena->completion_tick, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_monsters_killed, &arena->monsters_killed, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_monsters_total, &arena->monsters_total, sizeof(int32_t*));

    printf("Arena initialized: %d instances, %d max ticks\n", num_instances, max_ticks);
    printf("Memory allocated:\n");
    printf("  Player state: %.2f MB\n", (n * 10 * sizeof(fixed_t)) / (1024.0 * 1024.0));
    printf("  Input buffer: %.2f MB\n", (input_size * sizeof(TicCmd)) / (1024.0 * 1024.0));
    printf("  Checkpoints:  %.2f MB\n", (checkpoint_size * sizeof(Checkpoint)) / (1024.0 * 1024.0));
}

void FreeArena(DoomArena* arena) {
    cudaFree(arena->player_health);
    cudaFree(arena->player_armor);
    cudaFree(arena->player_x);
    cudaFree(arena->player_y);
    cudaFree(arena->player_z);
    cudaFree(arena->player_angle);
    cudaFree(arena->player_momx);
    cudaFree(arena->player_momy);
    cudaFree(arena->player_momz);
    cudaFree(arena->player_alive);
    cudaFree(arena->player_kills);

    cudaFree(arena->monster_x);
    cudaFree(arena->monster_y);
    cudaFree(arena->monster_z);
    cudaFree(arena->monster_angle);
    cudaFree(arena->monster_health);
    cudaFree(arena->monster_type);
    cudaFree(arena->monster_alive);
    cudaFree(arena->monster_target_idx);
    cudaFree(arena->monster_movedir);
    cudaFree(arena->monster_movecount);
    cudaFree(arena->monster_reactiontime);
    cudaFree(arena->monster_count);

    cudaFree(arena->input_buffer);
    cudaFree(arena->checkpoints);
    cudaFree(arena->checkpoint_count);
    cudaFree(arena->current_level);
    cudaFree(arena->level_complete);
    cudaFree(arena->game_won);
    cudaFree(arena->completion_tick);
    cudaFree(arena->monsters_killed);
    cudaFree(arena->monsters_total);
}

// =============================================================================
// Initialize Sine Table
// =============================================================================

void InitSineTable() {
    fixed_t* h_sine = new fixed_t[FINEANGLES / 4 + 1];
    for (int i = 0; i <= FINEANGLES / 4; i++) {
        double angle = (double)i * M_PI / (FINEANGLES / 2);
        h_sine[i] = (fixed_t)(sin(angle) * FRACUNIT);
    }
    cudaMemcpyToSymbol(c_finesine, h_sine, (FINEANGLES / 4 + 1) * sizeof(fixed_t));
    delete[] h_sine;
}

// =============================================================================
// Initialize Player State
// =============================================================================

void InitPlayers(DoomArena* arena, fixed_t start_x, fixed_t start_y) {
    int n = arena->num_instances;

    // Host arrays
    int32_t* h_health = new int32_t[n];
    int32_t* h_armor = new int32_t[n];
    fixed_t* h_x = new fixed_t[n];
    fixed_t* h_y = new fixed_t[n];
    fixed_t* h_z = new fixed_t[n];
    angle_t* h_angle = new angle_t[n];
    fixed_t* h_momx = new fixed_t[n];
    fixed_t* h_momy = new fixed_t[n];
    fixed_t* h_momz = new fixed_t[n];
    uint8_t* h_alive = new uint8_t[n];
    int16_t* h_monster_count = new int16_t[n];

    for (int i = 0; i < n; i++) {
        h_health[i] = 100;
        h_armor[i] = 0;
        h_x[i] = start_x;
        h_y[i] = start_y;
        h_z[i] = 0;
        h_angle[i] = ANG90;  // Facing north
        h_momx[i] = 0;
        h_momy[i] = 0;
        h_momz[i] = 0;
        h_alive[i] = 1;
        h_monster_count[i] = 100;
    }

    // Copy to device
    cudaMemcpy(arena->player_health, h_health, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_armor, h_armor, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_x, h_x, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_y, h_y, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_z, h_z, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_angle, h_angle, n * sizeof(angle_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_momx, h_momx, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_momy, h_momy, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_momz, h_momz, n * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->player_alive, h_alive, n * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(arena->monster_count, h_monster_count, n * sizeof(int16_t), cudaMemcpyHostToDevice);

    // Clear checkpoint counters and kill counters
    cudaMemset(arena->checkpoint_count, 0, n * sizeof(int32_t));
    cudaMemset(arena->player_kills, 0, n * sizeof(int16_t));

    delete[] h_health;
    delete[] h_armor;
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;
    delete[] h_angle;
    delete[] h_momx;
    delete[] h_momy;
    delete[] h_momz;
    delete[] h_alive;
    delete[] h_monster_count;
}

// =============================================================================
// Generate Test Input (Simple: walk forward, turn occasionally)
// =============================================================================

void GenerateTestInput(DoomArena* arena, int num_ticks) {
    int n = arena->num_instances;
    size_t size = n * num_ticks;
    TicCmd* h_input = new TicCmd[size];

    for (int tick = 0; tick < num_ticks; tick++) {
        for (int i = 0; i < n; i++) {
            int idx = tick * n + i;
            TicCmd& cmd = h_input[idx];

            // Each instance walks forward with slight variation
            cmd.forwardmove = 50;  // Walk forward
            cmd.sidemove = 0;

            // Turn based on instance id to create different paths
            if ((tick + i) % 70 == 0) {
                cmd.angleturn = (i % 2 == 0) ? 512 : -512;  // Turn left/right
            } else {
                cmd.angleturn = 0;
            }

            cmd.buttons = 0;
            cmd.consistency = 0;
            cmd.chatchar = 0;
        }
    }

    cudaMemcpy(arena->input_buffer, h_input, size * sizeof(TicCmd), cudaMemcpyHostToDevice);
    delete[] h_input;
}

// =============================================================================
// Read Checkpoints Back
// =============================================================================

void ReadCheckpoints(DoomArena* arena, int instance_id) {
    int32_t count;
    cudaMemcpy(&count, &arena->checkpoint_count[instance_id], sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (count == 0) {
        printf("Instance %d: No checkpoints\n", instance_id);
        return;
    }

    Checkpoint* h_cp = new Checkpoint[count];
    for (int i = 0; i < count; i++) {
        int idx = i * arena->num_instances + instance_id;
        cudaMemcpy(&h_cp[i], &arena->checkpoints[idx], sizeof(Checkpoint), cudaMemcpyDeviceToHost);
    }

    printf("Instance %d: %d checkpoints\n", instance_id, count);
    for (int i = 0; i < count && i < 5; i++) {  // Print first 5
        Checkpoint& cp = h_cp[i];
        printf("  [tick %4d] pos=(%.1f, %.1f) health=%d alive=%d\n",
               cp.tick,
               cp.x / (float)FRACUNIT,
               cp.y / (float)FRACUNIT,
               cp.health,
               cp.alive);
    }
    if (count > 5) {
        printf("  ... (%d more)\n", count - 5);
    }

    delete[] h_cp;
}

// =============================================================================
// Read Level Completion Results
// =============================================================================

void PrintLevelResults(DoomArena* arena, int instance_id) {
    // Read level completion state
    uint8_t current_level, game_won;
    int32_t completion_tick, monsters_killed, monsters_total;
    uint8_t alive;
    int32_t health;

    cudaMemcpy(&current_level, &arena->current_level[instance_id], sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&game_won, &arena->game_won[instance_id], sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&completion_tick, &arena->completion_tick[instance_id], sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&monsters_killed, &arena->monsters_killed[instance_id], sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&monsters_total, &arena->monsters_total[instance_id], sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&alive, &arena->player_alive[instance_id], sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&health, &arena->player_health[instance_id], sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Format output
    const char* level_name = GetLevelName(current_level);
    const char* status = GetCompletionStatus(game_won, alive);

    printf("Instance %d: Level %s, %s", instance_id, level_name, status);

    if (game_won) {
        printf(" in %d ticks (%.1f sec)\n", completion_tick, completion_tick / 35.0f);
    } else {
        printf("\n");
    }

    // Monster stats
    int kill_pct = (monsters_total > 0) ? (monsters_killed * 100) / monsters_total : 0;
    printf("  Monsters killed: %d/%d (%d%%)\n", monsters_killed, monsters_total, kill_pct);
    printf("  Health remaining: %d\n", health);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int num_instances = 150000;
    int num_ticks = 500;
    int checkpoint_interval = 35;  // Every second of game time

    if (argc > 1) num_instances = atoi(argv[1]);
    if (argc > 2) num_ticks = atoi(argv[2]);
    if (argc > 3) checkpoint_interval = atoi(argv[3]);

    printf("=== GPU DOOM Test ===\n");
    printf("Instances: %d\n", num_instances);
    printf("Ticks: %d (%.1f seconds game time)\n", num_ticks, num_ticks / 35.0f);
    printf("Checkpoint interval: %d ticks\n\n", checkpoint_interval);

    // Initialize
    InitSineTable();

    DoomArena arena;
    InitArena(&arena, num_instances, num_ticks);

    // E1M1 player start position (converted to fixed point)
    fixed_t start_x = 1056 << FRACBITS;
    fixed_t start_y = -3616 << FRACBITS;
    InitPlayers(&arena, start_x, start_y);

    // Generate test input
    GenerateTestInput(&arena, num_ticks);

    // Run simulation
    int threads_per_block = 256;
    int num_blocks = (num_instances + threads_per_block - 1) / threads_per_block;

    printf("Launching kernel: %d blocks × %d threads\n\n", num_blocks, threads_per_block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    doom_simulate<<<num_blocks, threads_per_block>>>(num_instances, num_ticks, checkpoint_interval);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("=== Results ===\n");
    printf("Kernel time: %.2f ms\n", ms);
    printf("Total ticks: %d\n", num_instances * num_ticks);
    printf("Throughput: %.0f ticks/sec\n", (num_instances * num_ticks) / (ms / 1000.0f));
    printf("Per-instance: %.0f ticks/sec (%.1f× realtime)\n",
           num_ticks / (ms / 1000.0f),
           (num_ticks / (ms / 1000.0f)) / 35.0f);
    printf("\n");

    // Show level completion results
    printf("=== Level Completion ===\n");
    PrintLevelResults(&arena, 0);
    if (num_instances > 1) {
        printf("\n");
        PrintLevelResults(&arena, num_instances / 2);
    }
    if (num_instances > 2) {
        printf("\n");
        PrintLevelResults(&arena, num_instances - 1);
    }
    printf("\n");

    // Show sample checkpoints
    printf("=== Checkpoints ===\n");
    ReadCheckpoints(&arena, 0);
    if (num_instances > 1) {
        ReadCheckpoints(&arena, num_instances / 2);
    }

    // Cleanup
    FreeArena(&arena);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
