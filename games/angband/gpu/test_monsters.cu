#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "angband_monsters.cuh"

// Simple kernel to test monster data access
__global__ void test_monster_data() {
    int tid = threadIdx.x;

    if (tid < 10) {
        // Test accessing first 10 monsters
        MonsterRace* monster = &MONSTER_RACES[tid];
        printf("Monster %d: HP=%d, AC=%d, Depth=%d, Speed=%d, Blows=%d\n",
               tid, monster->hp, monster->ac, monster->depth,
               monster->speed, monster->num_blows);
    }

    // Test Morgoth (last monster)
    if (tid == 0) {
        int morgoth_idx = NUM_MONSTER_RACES - 1;
        MonsterRace* morgoth = &MONSTER_RACES[morgoth_idx];
        printf("\nMorgoth: HP=%d, AC=%d, Depth=%d, Speed=%d, Blows=%d, Flags=0x%02X\n",
               morgoth->hp, morgoth->ac, morgoth->depth,
               morgoth->speed, morgoth->num_blows, morgoth->flags);

        // Show Morgoth's attacks
        printf("Morgoth's attacks:\n");
        for (int i = 0; i < morgoth->num_blows; i++) {
            MonsterBlow* blow = &morgoth->blows[i];
            printf("  Blow %d: Effect=%d, Damage=%dd%d\n",
                   i, blow->effect, blow->dd, blow->ds);
        }
    }
}

// Test monster spawning and attack
__global__ void test_monster_combat(int depth, int player_ac) {
    int tid = threadIdx.x;

    if (tid == 0) {
        // Initialize RNG
        curandState rng;
        curand_init(12345, 0, 0, &rng);

        printf("\n=== Testing Monster Combat at Depth %d ===\n", depth);

        // Spawn 5 monsters at this depth
        for (int i = 0; i < 5; i++) {
            int monster_idx = get_monster_for_depth(depth, &rng);
            MonsterRace* monster = &MONSTER_RACES[monster_idx];

            printf("Spawned monster %d: HP=%d, AC=%d, Speed=%d\n",
                   monster_idx, monster->hp, monster->ac, monster->speed);

            // Test attack
            int damage = monster_attack(monster_idx, player_ac, &rng);
            printf("  Attack deals %d damage to player (AC=%d)\n", damage, player_ac);
        }
    }
}

int main() {
    printf("Angband Monster System Test\n");
    printf("============================\n");
    printf("Total monsters: %d\n\n", MAX_MONSTER_RACES);

    // Test basic data access
    printf("Testing monster data access...\n");
    test_monster_data<<<1, 32>>>();
    cudaDeviceSynchronize();

    // Test combat at various depths
    printf("\nTesting monster spawning and combat...\n");
    test_monster_combat<<<1, 1>>>(5, 20);   // Depth 5, Player AC 20
    cudaDeviceSynchronize();

    test_monster_combat<<<1, 1>>>(50, 100); // Depth 50, Player AC 100
    cudaDeviceSynchronize();

    test_monster_combat<<<1, 1>>>(100, 150); // Depth 100 (Morgoth!), Player AC 150
    cudaDeviceSynchronize();

    printf("\nAll tests completed!\n");

    return 0;
}
