/**
 * GPU DOOM Verification Binary
 *
 * Single-instance mode for CPU vs GPU verification.
 * Reads TicCmd from stdin (binary format), outputs state JSONL to stdout.
 *
 * Usage:
 *   ./doom_verify < input.ticcmd > output.jsonl
 *   ./doom_verify --self-test
 *
 * Original: id Software DOOM (1993)
 * GPU Port: MIT License
 */

#include "doom_types.cuh"
#include "doom_data.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// =============================================================================
// Fixed Point Math (from DOOM source)
// =============================================================================

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
// Trig Functions (using canonical c_finesine from doom_data.cuh)
// =============================================================================

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
// Random Number Generation (using canonical c_rndtable)
// =============================================================================

__device__ int d_prndindex = 0;

__device__ int P_Random(void) {
    d_prndindex = (d_prndindex + 1) & 0xFF;
    return c_rndtable[d_prndindex];
}

// =============================================================================
// Player State (single instance)
// =============================================================================

__device__ fixed_t d_player_x;
__device__ fixed_t d_player_y;
__device__ fixed_t d_player_z;
__device__ angle_t d_player_angle;
__device__ fixed_t d_player_momx;
__device__ fixed_t d_player_momy;
__device__ fixed_t d_player_momz;
__device__ int32_t d_player_health;
__device__ int32_t d_player_armor;
__device__ int16_t d_player_kills;
__device__ uint8_t d_player_alive;

// =============================================================================
// P_PlayerThink - Single Instance
// =============================================================================

// Movement constants (from original)
#define MAXMOVE         (30 * FRACUNIT)
#define FRICTION        0xe800  // ~0.90625 in fixed point
#define STOPSPEED       0x1000  // Below this, stop completely

__global__ void P_PlayerThink_Kernel(TicCmd cmd) {
    // Skip if dead
    if (!d_player_alive) return;

    // Current state
    fixed_t x = d_player_x;
    fixed_t y = d_player_y;
    angle_t angle = d_player_angle;
    fixed_t momx = d_player_momx;
    fixed_t momy = d_player_momy;

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

    // Write back state
    d_player_x = x;
    d_player_y = y;
    d_player_angle = angle;
    d_player_momx = momx;
    d_player_momy = momy;
}

// =============================================================================
// Host Functions
// =============================================================================

void InitPlayer(fixed_t start_x, fixed_t start_y) {
    // Initial state (E1M1 player start)
    fixed_t h_x = start_x;
    fixed_t h_y = start_y;
    fixed_t h_z = 0;
    angle_t h_angle = ANG90;  // Facing north
    fixed_t h_momx = 0;
    fixed_t h_momy = 0;
    fixed_t h_momz = 0;
    int32_t h_health = 100;
    int32_t h_armor = 0;
    int16_t h_kills = 0;
    uint8_t h_alive = 1;

    // Copy to device
    cudaMemcpyToSymbol(d_player_x, &h_x, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_y, &h_y, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_z, &h_z, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_angle, &h_angle, sizeof(angle_t));
    cudaMemcpyToSymbol(d_player_momx, &h_momx, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_momy, &h_momy, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_momz, &h_momz, sizeof(fixed_t));
    cudaMemcpyToSymbol(d_player_health, &h_health, sizeof(int32_t));
    cudaMemcpyToSymbol(d_player_armor, &h_armor, sizeof(int32_t));
    cudaMemcpyToSymbol(d_player_kills, &h_kills, sizeof(int16_t));
    cudaMemcpyToSymbol(d_player_alive, &h_alive, sizeof(uint8_t));

    // Initialize random index to 0 (match original DOOM)
    int h_prndindex = 0;
    cudaMemcpyToSymbol(d_prndindex, &h_prndindex, sizeof(int));
}

void OutputState(int tick) {
    // Copy state from GPU
    fixed_t x, y, z;
    angle_t angle;
    int32_t health, armor;
    int16_t kills;
    uint8_t alive;

    cudaMemcpyFromSymbol(&x, d_player_x, sizeof(fixed_t));
    cudaMemcpyFromSymbol(&y, d_player_y, sizeof(fixed_t));
    cudaMemcpyFromSymbol(&z, d_player_z, sizeof(fixed_t));
    cudaMemcpyFromSymbol(&angle, d_player_angle, sizeof(angle_t));
    cudaMemcpyFromSymbol(&health, d_player_health, sizeof(int32_t));
    cudaMemcpyFromSymbol(&armor, d_player_armor, sizeof(int32_t));
    cudaMemcpyFromSymbol(&kills, d_player_kills, sizeof(int16_t));
    cudaMemcpyFromSymbol(&alive, d_player_alive, sizeof(uint8_t));

    // Output JSONL (matches CPU reference format)
    printf("{\"tick\":%d,\"x\":%d,\"y\":%d,\"z\":%d,\"angle\":%u,"
           "\"health\":%d,\"armor\":%d,\"kills\":%d,\"alive\":%d}\n",
           tick, x, y, z, angle, health, armor, kills, alive);
}

// =============================================================================
// Input Handling
// =============================================================================

bool ReadTicCmd(TicCmd* cmd) {
    // Binary format: int8 forwardmove, int8 sidemove, int16 angleturn,
    //                int16 consistency, uint8 chatchar, uint8 buttons
    size_t read = fread(cmd, sizeof(TicCmd), 1, stdin);
    return read == 1;
}

// =============================================================================
// Self-Test Mode
// =============================================================================

void SelfTest() {
    fprintf(stderr, "=== GPU DOOM Verification - Self Test ===\n\n");

    // Initialize
    fixed_t start_x = 1056 << FRACBITS;  // E1M1 start
    fixed_t start_y = -3616 << FRACBITS;
    InitPlayer(start_x, start_y);

    fprintf(stderr, "Testing basic movement (walk forward 10 ticks)...\n");

    // Test: Walk forward
    TicCmd cmd;
    memset(&cmd, 0, sizeof(TicCmd));
    cmd.forwardmove = 50;  // Walk forward

    for (int tick = 0; tick < 10; tick++) {
        P_PlayerThink_Kernel<<<1, 1>>>(cmd);
        cudaDeviceSynchronize();

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at tick %d: %s\n", tick, cudaGetErrorString(err));
            exit(1);
        }
    }

    // Read final position
    fixed_t final_x, final_y;
    cudaMemcpyFromSymbol(&final_x, d_player_x, sizeof(fixed_t));
    cudaMemcpyFromSymbol(&final_y, d_player_y, sizeof(fixed_t));

    // Player should have moved north (positive Y in DOOM coordinates)
    if (final_y > start_y) {
        fprintf(stderr, "✓ Movement test PASSED\n");
        fprintf(stderr, "  Start: (%d, %d)\n", start_x >> FRACBITS, start_y >> FRACBITS);
        fprintf(stderr, "  End:   (%d, %d)\n", final_x >> FRACBITS, final_y >> FRACBITS);
    } else {
        fprintf(stderr, "✗ Movement test FAILED\n");
        fprintf(stderr, "  Expected Y to increase, but start_y=%d, final_y=%d\n",
                start_y >> FRACBITS, final_y >> FRACBITS);
        exit(1);
    }

    fprintf(stderr, "\n=== Self Test PASSED ===\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Check for self-test flag
    if (argc > 1 && strcmp(argv[1], "--self-test") == 0) {
        SelfTest();
        return 0;
    }

    // Normal verification mode: read TicCmd from stdin
    fprintf(stderr, "GPU DOOM Verification Mode\n");
    fprintf(stderr, "Reading TicCmd from stdin (binary format)...\n");

    // Initialize player at E1M1 start position
    fixed_t start_x = 1056 << FRACBITS;
    fixed_t start_y = -3616 << FRACBITS;
    InitPlayer(start_x, start_y);

    // Output initial state (tick 0)
    OutputState(0);

    // Process input ticks
    TicCmd cmd;
    int tick = 1;

    while (ReadTicCmd(&cmd)) {
        // Run one tick
        P_PlayerThink_Kernel<<<1, 1>>>(cmd);
        cudaDeviceSynchronize();

        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error at tick %d: %s\n", tick, cudaGetErrorString(err));
            return 1;
        }

        // Output state
        OutputState(tick);

        tick++;
    }

    fprintf(stderr, "Processed %d ticks\n", tick - 1);
    return 0;
}
