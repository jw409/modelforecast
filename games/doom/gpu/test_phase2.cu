/**
 * GPU DOOM Phase 2 Test
 *
 * Simple test program for monster combat system
 * Spawns a few monsters, player shoots and gets attacked
 */

#include "doom_types.cuh"
#include "doom_monsters.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// =============================================================================
// Device Memory Pointers
// =============================================================================

// Player state
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

// Monster state
#define MAX_MONSTERS 16  // Keep it small for testing
__device__ fixed_t* d_monster_x;
__device__ fixed_t* d_monster_y;
__device__ fixed_t* d_monster_z;
__device__ angle_t* d_monster_angle;
__device__ int32_t* d_monster_health;
__device__ uint8_t* d_monster_type;
__device__ uint8_t* d_monster_alive;
__device__ int16_t* d_monster_target_idx;
__device__ uint8_t* d_monster_movedir;
__device__ int16_t* d_monster_movecount;
__device__ int16_t* d_monster_reactiontime;

// Input
__device__ TicCmd* d_input_buffer;

// =============================================================================
// Fixed Point Math
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
// Trig (simplified - just use real trig for test)
// =============================================================================

__device__ fixed_t finesine(int idx) {
    double angle = (double)idx * M_PI / 4096.0;
    return (fixed_t)(sin(angle) * FRACUNIT);
}

__device__ fixed_t finecosine(int idx) {
    double angle = (double)idx * M_PI / 4096.0;
    return (fixed_t)(cos(angle) * FRACUNIT);
}

// =============================================================================
// Include combat system
// =============================================================================

#include "doom_combat.cuh"

// =============================================================================
// Spawn Monsters
// =============================================================================

__device__ void SpawnMonster(int monster_id, int instance_id, int num_instances,
                             fixed_t x, fixed_t y, MonsterTypeID type) {
    int idx = monster_id * num_instances + instance_id;

    d_monster_x[idx] = x;
    d_monster_y[idx] = y;
    d_monster_z[idx] = 0;
    d_monster_angle[idx] = 0;
    d_monster_type[idx] = type;
    d_monster_health[idx] = c_monster_stats[type].health;
    d_monster_alive[idx] = 1;
    d_monster_target_idx[idx] = 0;  // Target player
    d_monster_movedir[idx] = DI_NODIR;
    d_monster_movecount[idx] = 5;
    d_monster_reactiontime[idx] = 0;
}

// =============================================================================
// Simple Test Kernel
// =============================================================================

__global__ void test_combat(int num_instances, int num_ticks) {
    int instance_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance_id >= num_instances) return;

    // Spawn some test monsters
    if (instance_id == 0) {
        // Spawn 4 monsters in a line in front of player
        SpawnMonster(0, instance_id, num_instances, 1200 << FRACBITS, -3600 << FRACBITS, MT_ZOMBIE);
        SpawnMonster(1, instance_id, num_instances, 1300 << FRACBITS, -3500 << FRACBITS, MT_IMP);
        SpawnMonster(2, instance_id, num_instances, 1400 << FRACBITS, -3400 << FRACBITS, MT_DEMON);
        SpawnMonster(3, instance_id, num_instances, 1500 << FRACBITS, -3300 << FRACBITS, MT_ZOMBIE);
    }

    __syncthreads();

    // Simulate
    for (int tick = 0; tick < num_ticks; tick++) {
        // Player position (just standing still for test)
        fixed_t px = d_player_x[instance_id];
        fixed_t py = d_player_y[instance_id];
        angle_t pangle = d_player_angle[instance_id];

        // Get input
        TicCmd cmd = d_input_buffer[tick * num_instances + instance_id];

        // Player shoots
        if (cmd.buttons & BT_ATTACK) {
            P_PlayerAttack(instance_id, tick, num_instances, pangle, px, py);
        }

        // Monsters think
        for (int m = 0; m < MAX_MONSTERS; m++) {
            P_MonsterThink(m, instance_id, tick, num_instances);
        }

        __syncthreads();
    }
}

// =============================================================================
// Host Code
// =============================================================================

int main() {
    const int num_instances = 16;
    const int num_ticks = 100;

    printf("=== GPU DOOM Phase 2 Test ===\n");
    printf("Instances: %d\n", num_instances);
    printf("Ticks: %d\n\n", num_ticks);

    // Allocate device memory
    int32_t *h_player_health, *h_player_armor, *d_ph, *d_pa;
    fixed_t *h_player_x, *h_player_y, *h_player_z, *d_px, *d_py, *d_pz;
    angle_t *h_player_angle, *d_pangle;
    fixed_t *h_momx, *h_momy, *h_momz, *d_pmx, *d_pmy, *d_pmz;
    uint8_t *h_player_alive, *d_palive;
    int16_t *h_player_kills, *d_pkills;

    cudaMalloc(&d_ph, num_instances * sizeof(int32_t));
    cudaMalloc(&d_pa, num_instances * sizeof(int32_t));
    cudaMalloc(&d_px, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_py, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_pz, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_pangle, num_instances * sizeof(angle_t));
    cudaMalloc(&d_pmx, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_pmy, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_pmz, num_instances * sizeof(fixed_t));
    cudaMalloc(&d_palive, num_instances * sizeof(uint8_t));
    cudaMalloc(&d_pkills, num_instances * sizeof(int16_t));

    // Initialize player state
    h_player_health = new int32_t[num_instances];
    h_player_armor = new int32_t[num_instances];
    h_player_x = new fixed_t[num_instances];
    h_player_y = new fixed_t[num_instances];
    h_player_z = new fixed_t[num_instances];
    h_player_angle = new angle_t[num_instances];
    h_momx = new fixed_t[num_instances];
    h_momy = new fixed_t[num_instances];
    h_momz = new fixed_t[num_instances];
    h_player_alive = new uint8_t[num_instances];
    h_player_kills = new int16_t[num_instances];

    for (int i = 0; i < num_instances; i++) {
        h_player_health[i] = 100;
        h_player_armor[i] = 0;
        h_player_x[i] = 1056 << FRACBITS;  // E1M1 start
        h_player_y[i] = -3616 << FRACBITS;
        h_player_z[i] = 0;
        h_player_angle[i] = ANG90;
        h_momx[i] = 0;
        h_momy[i] = 0;
        h_momz[i] = 0;
        h_player_alive[i] = 1;
        h_player_kills[i] = 0;
    }

    cudaMemcpy(d_ph, h_player_health, num_instances * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pa, h_player_armor, num_instances * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_px, h_player_x, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_player_y, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, h_player_z, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pangle, h_player_angle, num_instances * sizeof(angle_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pmx, h_momx, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pmy, h_momy, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pmz, h_momz, num_instances * sizeof(fixed_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_palive, h_player_alive, num_instances * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pkills, h_player_kills, num_instances * sizeof(int16_t), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_player_health, &d_ph, sizeof(void*));
    cudaMemcpyToSymbol(d_player_armor, &d_pa, sizeof(void*));
    cudaMemcpyToSymbol(d_player_x, &d_px, sizeof(void*));
    cudaMemcpyToSymbol(d_player_y, &d_py, sizeof(void*));
    cudaMemcpyToSymbol(d_player_z, &d_pz, sizeof(void*));
    cudaMemcpyToSymbol(d_player_angle, &d_pangle, sizeof(void*));
    cudaMemcpyToSymbol(d_player_momx, &d_pmx, sizeof(void*));
    cudaMemcpyToSymbol(d_player_momy, &d_pmy, sizeof(void*));
    cudaMemcpyToSymbol(d_player_momz, &d_pmz, sizeof(void*));
    cudaMemcpyToSymbol(d_player_alive, &d_palive, sizeof(void*));
    cudaMemcpyToSymbol(d_player_kills, &d_pkills, sizeof(void*));

    // Allocate monster memory
    size_t monster_size = MAX_MONSTERS * num_instances;
    fixed_t *d_mx, *d_my, *d_mz;
    angle_t *d_mangle;
    int32_t *d_mhealth;
    uint8_t *d_mtype, *d_malive, *d_mdir;
    int16_t *d_mtarget, *d_mmovecount, *d_mreaction;

    cudaMalloc(&d_mx, monster_size * sizeof(fixed_t));
    cudaMalloc(&d_my, monster_size * sizeof(fixed_t));
    cudaMalloc(&d_mz, monster_size * sizeof(fixed_t));
    cudaMalloc(&d_mangle, monster_size * sizeof(angle_t));
    cudaMalloc(&d_mhealth, monster_size * sizeof(int32_t));
    cudaMalloc(&d_mtype, monster_size * sizeof(uint8_t));
    cudaMalloc(&d_malive, monster_size * sizeof(uint8_t));
    cudaMalloc(&d_mtarget, monster_size * sizeof(int16_t));
    cudaMalloc(&d_mdir, monster_size * sizeof(uint8_t));
    cudaMalloc(&d_mmovecount, monster_size * sizeof(int16_t));
    cudaMalloc(&d_mreaction, monster_size * sizeof(int16_t));

    cudaMemset(d_malive, 0, monster_size * sizeof(uint8_t));  // All dead initially

    cudaMemcpyToSymbol(d_monster_x, &d_mx, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_y, &d_my, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_z, &d_mz, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_angle, &d_mangle, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_health, &d_mhealth, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_type, &d_mtype, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_alive, &d_malive, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_target_idx, &d_mtarget, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_movedir, &d_mdir, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_movecount, &d_mmovecount, sizeof(void*));
    cudaMemcpyToSymbol(d_monster_reactiontime, &d_mreaction, sizeof(void*));

    // Create input (player shoots every 10 ticks)
    TicCmd *h_input = new TicCmd[num_ticks * num_instances];
    for (int t = 0; t < num_ticks; t++) {
        for (int i = 0; i < num_instances; i++) {
            TicCmd& cmd = h_input[t * num_instances + i];
            cmd.forwardmove = 0;
            cmd.sidemove = 0;
            cmd.angleturn = 0;
            cmd.buttons = (t % 10 == 0) ? BT_ATTACK : 0;  // Shoot every 10 ticks
        }
    }

    TicCmd* d_input;
    cudaMalloc(&d_input, num_ticks * num_instances * sizeof(TicCmd));
    cudaMemcpy(d_input, h_input, num_ticks * num_instances * sizeof(TicCmd), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_input_buffer, &d_input, sizeof(void*));

    // Run kernel
    int threads_per_block = 16;
    int num_blocks = (num_instances + threads_per_block - 1) / threads_per_block;

    printf("Launching kernel...\n");
    test_combat<<<num_blocks, threads_per_block>>>(num_instances, num_ticks);
    cudaDeviceSynchronize();

    // Read results
    cudaMemcpy(h_player_health, d_ph, num_instances * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_player_alive, d_palive, num_instances * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_player_kills, d_pkills, num_instances * sizeof(int16_t), cudaMemcpyDeviceToHost);

    printf("\n=== Results ===\n");
    for (int i = 0; i < num_instances && i < 5; i++) {
        printf("Instance %d: Health=%d Alive=%d Kills=%d\n",
               i, h_player_health[i], h_player_alive[i], h_player_kills[i]);
    }

    // Cleanup
    delete[] h_player_health;
    delete[] h_player_armor;
    delete[] h_player_x;
    delete[] h_player_y;
    delete[] h_player_z;
    delete[] h_player_angle;
    delete[] h_momx;
    delete[] h_momy;
    delete[] h_momz;
    delete[] h_player_alive;
    delete[] h_player_kills;
    delete[] h_input;

    cudaFree(d_ph);
    cudaFree(d_pa);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);
    cudaFree(d_pangle);
    cudaFree(d_pmx);
    cudaFree(d_pmy);
    cudaFree(d_pmz);
    cudaFree(d_palive);
    cudaFree(d_pkills);
    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_mz);
    cudaFree(d_mangle);
    cudaFree(d_mhealth);
    cudaFree(d_mtype);
    cudaFree(d_malive);
    cudaFree(d_mtarget);
    cudaFree(d_mdir);
    cudaFree(d_mmovecount);
    cudaFree(d_mreaction);
    cudaFree(d_input);

    printf("\nPhase 2 test complete!\n");
    return 0;
}
