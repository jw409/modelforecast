/*
 * ANGBAND BORG KERNEL - GPU Implementation
 *
 * Simplified borg_think() for CUDA execution.
 * Based on APWBorg decision logic from borg9.c.
 *
 * Memory Layout: Interleaved (best coalescing)
 */

#include "borg_state.h"
#include "../../common/interleaved.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

// ============================================================================
// DEVICE HELPERS
// ============================================================================

// Calculate danger at position (simplified from borg6.c)
__device__ int calculate_danger(
    BorgStateInterleaved* s,
    int id, int x, int y,
    int num_instances
) {
    int danger = 0;
    int monster_count = s->monster_count[id];

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        int mx = IGET(s->monster_x, i, id, num_instances);
        int my = IGET(s->monster_y, i, id, num_instances);
        int mhp = IGET(s->monster_hp, i, id, num_instances);
        int awake = IGET(s->monster_awake, i, id, num_instances);

        // Distance to monster
        int dx = abs(mx - x);
        int dy = abs(my - y);
        int dist = max(dx, dy);

        if (dist == 0) dist = 1;

        // Danger inversely proportional to distance
        int threat = (mhp * 10) / dist;
        if (awake) threat *= 2;

        danger += threat;
    }

    return danger;
}

// Find nearest monster
__device__ int find_nearest_monster(
    BorgStateInterleaved* s,
    int id, int x, int y,
    int num_instances,
    int* out_mx, int* out_my
) {
    int monster_count = s->monster_count[id];
    int min_dist = 9999;
    int best_idx = -1;

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        int mx = IGET(s->monster_x, i, id, num_instances);
        int my = IGET(s->monster_y, i, id, num_instances);

        int dx = abs(mx - x);
        int dy = abs(my - y);
        int dist = max(dx, dy);

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
            *out_mx = mx;
            *out_my = my;
        }
    }

    return min_dist;
}

// Check if config flag is set
__device__ bool config_has(uint16_t config, uint16_t flag) {
    return (config & flag) != 0;
}

// ============================================================================
// BORG DECISION KERNEL
// ============================================================================

__global__ void borg_think_kernel(
    BorgStateInterleaved state,
    uint8_t* actions,           // Output: action per instance
    uint32_t num_instances
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_instances) return;

    // Skip dead borgs
    if (!state.alive[id]) {
        actions[id] = BORG_ACTION_NONE;
        return;
    }

    // Load state into registers
    int x = state.x[id];
    int y = state.y[id];
    int hp = state.hp[id];
    int max_hp = state.max_hp[id];
    int depth = state.depth[id];
    int speed = state.speed[id];
    uint16_t config = state.config[id];
    int no_deeper = state.no_deeper[id];

    bool has_healing = state.has_healing[id];
    bool has_recall = state.has_recall[id];
    bool has_teleport = state.has_teleport[id];
    bool plays_risky = config_has(config, CFG_PLAYS_RISKY);

    // Calculate danger
    int danger = calculate_danger(&state, id, x, y, num_instances);

    // Find nearest monster
    int mx, my;
    int monster_dist = find_nearest_monster(&state, id, x, y, num_instances, &mx, &my);

    // =========================================================================
    // BORG DECISION TREE (simplified from borg_think in borg9.c)
    // =========================================================================

    uint8_t action = BORG_ACTION_EXPLORE;  // Default

    // 1. CRITICAL: Very low HP - try to survive
    float hp_pct = (float)hp / (float)max_hp;
    if (hp_pct < 0.2f) {
        if (has_healing) {
            action = BORG_ACTION_HEAL;
        } else if (has_teleport && danger > 50) {
            action = BORG_ACTION_FLEE;
        } else if (has_recall) {
            action = BORG_ACTION_ASCEND;  // Recall to town
        } else if (!plays_risky) {
            action = BORG_ACTION_FLEE;
        }
    }
    // 2. LOW HP: Consider healing/fleeing
    else if (hp_pct < 0.5f && danger > 100) {
        if (has_healing) {
            action = BORG_ACTION_HEAL;
        } else if (!plays_risky) {
            action = BORG_ACTION_FLEE;
        }
    }
    // 3. COMBAT: Monster in range
    else if (monster_dist <= 1) {
        action = BORG_ACTION_ATTACK;
    }
    // 4. APPROACH: Monster nearby
    else if (monster_dist <= 5 && danger < 200) {
        action = BORG_ACTION_ATTACK;  // Will approach
    }
    // 5. REST: Low HP but safe
    else if (hp_pct < 0.8f && danger < 20) {
        action = BORG_ACTION_REST;
    }
    // 6. DESCEND: At stairs and ready
    else if (depth < no_deeper && hp_pct > 0.7f && danger < 50) {
        // Check if speed worship requires more speed
        bool speed_ok = !config_has(config, CFG_WORSHIPS_SPEED) || speed >= 10;
        if (speed_ok) {
            action = BORG_ACTION_DESCEND;
        }
    }
    // 7. EXPLORE: Default
    else {
        action = BORG_ACTION_EXPLORE;
    }

    actions[id] = action;
}

// ============================================================================
// GAME STEP KERNEL (Execute actions, update state)
// ============================================================================

__global__ void borg_execute_kernel(
    BorgStateInterleaved state,
    uint8_t* actions,
    uint32_t num_instances
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_instances) return;

    if (!state.alive[id]) return;

    uint8_t action = actions[id];
    int x = state.x[id];
    int y = state.y[id];
    int hp = state.hp[id];
    int max_hp = state.max_hp[id];

    // Execute action (simplified)
    switch (action) {
        case BORG_ACTION_REST:
            // Recover 1% HP
            hp = min(hp + max(1, max_hp / 100), max_hp);
            break;

        case BORG_ACTION_HEAL:
            // Use healing potion: restore 25% HP
            hp = min(hp + max_hp / 4, max_hp);
            state.has_healing[id] = 0;  // Used up (simplified)
            break;

        case BORG_ACTION_FLEE:
            // Teleport away (randomize position)
            // In full implementation: proper teleport logic
            x = (x + 50) % DUNGEON_WIDTH;
            y = (y + 30) % DUNGEON_HEIGHT;
            break;

        case BORG_ACTION_ATTACK:
            // Combat: simplified damage exchange
            // Monster deals damage back
            hp -= 5;  // Simplified monster damage
            // Clear monster if killed (simplified)
            if (state.monster_count[id] > 0) {
                int idx = 0;  // Attack nearest
                int mhp = IGET(state.monster_hp, idx, id, num_instances);
                mhp -= state.damage[id];
                if (mhp <= 0) {
                    state.monster_count[id]--;
                    state.exp[id] += 100;  // XP gain
                } else {
                    ISET(state.monster_hp, idx, id, num_instances, mhp);
                }
            }
            break;

        case BORG_ACTION_DESCEND:
            // Go deeper
            state.depth[id]++;
            // Generate new dungeon level (simplified: random monsters)
            state.monster_count[id] = min(state.depth[id] / 5 + 1, (int)MAX_MONSTERS);
            for (int i = 0; i < state.monster_count[id]; i++) {
                ISET(state.monster_x, i, id, num_instances, (x + 10 + i * 5) % DUNGEON_WIDTH);
                ISET(state.monster_y, i, id, num_instances, (y + 10 + i * 3) % DUNGEON_HEIGHT);
                ISET(state.monster_hp, i, id, num_instances, 10 + state.depth[id] * 2);
                ISET(state.monster_awake, i, id, num_instances, 0);
            }
            break;

        case BORG_ACTION_EXPLORE:
            // Move randomly
            x = (x + 1) % DUNGEON_WIDTH;
            // Chance to wake nearby monsters
            break;

        default:
            break;
    }

    // Check for death
    if (hp <= 0) {
        state.alive[id] = 0;
        state.death_cause[id] = DEATH_MONSTER;
        state.final_depth[id] = state.depth[id];
        state.final_turns[id] = state.turns[id];
    }

    // Update state
    state.x[id] = x;
    state.y[id] = y;
    state.hp[id] = hp;
    state.turns[id]++;

    // Check for win (depth 100 = Morgoth)
    if (state.depth[id] >= 100 && state.monster_count[id] == 0) {
        state.winner[id] = 1;
        state.alive[id] = 0;  // End game
    }
}

// ============================================================================
// HOST DRIVER
// ============================================================================

#define CHECK(call) \
do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

void run_borg_simulation(uint32_t num_instances, uint32_t max_turns) {
    printf("Running Angband Borg GPU simulation\n");
    printf("Instances: %d, Max turns: %d\n", num_instances, max_turns);

    // Allocate state
    BorgStateInterleaved* state = borg_state_alloc(num_instances);

    // Allocate actions
    uint8_t* d_actions;
    CHECK(cudaMalloc(&d_actions, num_instances));

    // Initialize state (simplified)
    std::vector<int16_t> init_x(num_instances, DUNGEON_WIDTH / 2);
    std::vector<int16_t> init_y(num_instances, DUNGEON_HEIGHT / 2);
    std::vector<int16_t> init_hp(num_instances, 100);
    std::vector<int16_t> init_depth(num_instances, 1);
    std::vector<uint8_t> init_alive(num_instances, 1);
    std::vector<uint16_t> init_config(num_instances);
    std::vector<uint8_t> init_no_deeper(num_instances, 127);

    // Assign different configs to batches
    for (int i = 0; i < num_instances; i++) {
        int config_type = i % 8;
        switch (config_type) {
            case 0: init_config[i] = CFG_WORSHIPS_SPEED | CFG_WORSHIPS_DAMAGE | CFG_PLAYS_RISKY; break;  // Aggro
            case 1: init_config[i] = CFG_WORSHIPS_SPEED | CFG_PLAYS_RISKY; break;  // Speed
            case 2: init_config[i] = CFG_WORSHIPS_HP | CFG_WORSHIPS_AC; init_no_deeper[i] = 30; break;  // Tank
            case 3: init_config[i] = CFG_WORSHIPS_SPEED | CFG_WORSHIPS_HP | CFG_KILLS_UNIQUES; break;  // Scummer
            case 4: init_config[i] = CFG_WORSHIPS_SPEED | CFG_WORSHIPS_HP | CFG_PLAYS_RISKY; break;  // Meta
            case 5: init_config[i] = CFG_WORSHIPS_HP | CFG_WORSHIPS_AC | CFG_WORSHIPS_GOLD; break;  // Economy
            case 6: init_config[i] = CFG_WORSHIPS_SPEED | CFG_WORSHIPS_DAMAGE | CFG_PLAYS_RISKY | CFG_CHEAT_DEATH; break;  // Cheat
            default: init_config[i] = 0; break;  // Default
        }
    }

    CHECK(cudaMemcpy(state->x, init_x.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->y, init_y.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->hp, init_hp.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->max_hp, init_hp.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->depth, init_depth.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->alive, init_alive.data(), num_instances, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->config, init_config.data(), num_instances * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->no_deeper, init_no_deeper.data(), num_instances, cudaMemcpyHostToDevice));

    // Initialize other fields
    CHECK(cudaMemset(state->speed, 10, num_instances * sizeof(int16_t)));  // Base speed
    CHECK(cudaMemset(state->damage, 20, num_instances * sizeof(int16_t)));  // Base damage
    CHECK(cudaMemset(state->monster_count, 1, num_instances));  // Start with 1 monster
    CHECK(cudaMemset(state->has_healing, 1, num_instances));  // Start with healing
    CHECK(cudaMemset(state->turns, 0, num_instances * sizeof(uint32_t)));
    CHECK(cudaMemset(state->winner, 0, num_instances));

    int threads = 256;
    int blocks = (num_instances + threads - 1) / threads;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t turn = 0; turn < max_turns; turn++) {
        // Think
        borg_think_kernel<<<blocks, threads>>>(*state, d_actions, num_instances);

        // Execute
        borg_execute_kernel<<<blocks, threads>>>(*state, d_actions, num_instances);
    }

    CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    printf("Simulation time: %.4f s\n", diff.count());
    printf("Throughput: %.2f instance-turns/sec\n", (double)num_instances * max_turns / diff.count());

    // Collect results
    std::vector<uint8_t> h_alive(num_instances);
    std::vector<uint16_t> h_final_depth(num_instances);
    std::vector<uint8_t> h_winner(num_instances);

    CHECK(cudaMemcpy(h_alive.data(), state->alive, num_instances, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_final_depth.data(), state->final_depth, num_instances * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_winner.data(), state->winner, num_instances, cudaMemcpyDeviceToHost));

    // Count results by config type
    int alive_count[8] = {0};
    int winner_count[8] = {0};
    int total_depth[8] = {0};
    int count[8] = {0};

    for (int i = 0; i < num_instances; i++) {
        int config_type = i % 8;
        count[config_type]++;
        if (h_alive[i]) alive_count[config_type]++;
        if (h_winner[i]) winner_count[config_type]++;
        total_depth[config_type] += h_final_depth[i];
    }

    const char* config_names[] = {"Aggro", "Speed", "Tank", "Scummer", "Meta", "Economy", "Cheat", "Default"};

    printf("\n=== RESULTS BY CONFIG ===\n");
    printf("%-10s %8s %8s %8s %8s\n", "Config", "Alive%", "Win%", "AvgDepth", "Count");
    for (int i = 0; i < 8; i++) {
        if (count[i] > 0) {
            printf("%-10s %7.1f%% %7.1f%% %8.1f %8d\n",
                config_names[i],
                100.0 * alive_count[i] / count[i],
                100.0 * winner_count[i] / count[i],
                (double)total_depth[i] / count[i],
                count[i]);
        }
    }

    // Cleanup
    cudaFree(d_actions);
    borg_state_free(state);
}

int main(int argc, char** argv) {
    uint32_t num_instances = 10000;
    uint32_t max_turns = 1000;

    if (argc > 1) num_instances = atoi(argv[1]);
    if (argc > 2) max_turns = atoi(argv[2]);

    run_borg_simulation(num_instances, max_turns);

    return 0;
}
