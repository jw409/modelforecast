/*
 * ANGBAND BORG KERNEL V2 - GPU Implementation with Real Game Rules
 *
 * Integration layer for Phase 4: Uses real Angband combat, monsters, and dungeons
 * Based on APWBorg decision logic with authentic game mechanics.
 *
 * Memory Layout: Interleaved (best coalescing)
 */

#include "borg_state.h"
#include "../../common/interleaved.h"

// Real Angband rules (being created by other agents)
#include "angband_combat.cuh"      // Real hit/damage formulas
#include "angband_monsters.cuh"    // Real monster data and AI
#include "angband_dungeon.cuh"     // Real dungeon generation

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <chrono>
#include <vector>

// ============================================================================
// DEVICE HELPERS
// ============================================================================

// Calculate danger at position using REAL monster stats
__device__ int calculate_danger(
    BorgStateInterleaved* s,
    int id, int x, int y,
    int num_instances
) {
    int danger = 0;
    int monster_count = s->monster_count[id];
    int player_ac = s->ac[id];

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        int mx = IGET(s->monster_x, i, id, num_instances);
        int my = IGET(s->monster_y, i, id, num_instances);
        int mhp = IGET(s->monster_hp, i, id, num_instances);
        int mtype = IGET(s->monster_type, i, id, num_instances);
        int awake = IGET(s->monster_awake, i, id, num_instances);

        // Distance to monster
        int dx = abs(mx - x);
        int dy = abs(my - y);
        int dist = max(dx, dy);

        if (dist == 0) dist = 1;

        // Get REAL monster potential damage from monster race
        int monster_damage_potential = get_monster_damage_potential(mtype, player_ac);

        // Calculate threat based on real damage and distance
        int threat = (monster_damage_potential * (mhp / 10)) / dist;
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
    int* out_mx, int* out_my,
    int* out_idx
) {
    int monster_count = s->monster_count[id];
    int min_dist = 9999;
    int best_idx = -1;

    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        int mx = IGET(s->monster_x, i, id, num_instances);
        int my = IGET(s->monster_y, i, id, num_instances);
        int mhp = IGET(s->monster_hp, i, id, num_instances);

        // Skip dead monsters
        if (mhp <= 0) continue;

        int dx = abs(mx - x);
        int dy = abs(my - y);
        int dist = max(dx, dy);

        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
            *out_mx = mx;
            *out_my = my;
            *out_idx = i;
        }
    }

    return min_dist;
}

// Check if config flag is set
__device__ bool config_has(uint16_t config, uint16_t flag) {
    return (config & flag) != 0;
}

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

// Initialize RNG state for each instance
__global__ void init_rng_kernel(
    curandState* states,
    uint32_t seed,
    uint32_t num_instances
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_instances) return;

    // Each instance gets unique seed
    curand_init(seed, id, 0, &states[id]);
}

// Generate initial dungeons for all instances
__global__ void init_dungeons_kernel(
    BorgStateInterleaved state,
    curandState* rng_states,
    uint32_t num_instances
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_instances) return;

    curandState* rng = &rng_states[id];
    int depth = state.depth[id];

    // Generate level using REAL Angband dungeon generation
    int px, py;  // Player start position
    int sx, sy;  // Stairs position

    generate_level(
        state.dungeon_terrain,  // Terrain output
        depth,                  // Dungeon depth
        &px, &py,              // Player position output
        &sx, &sy,              // Stairs position output
        id,                    // Instance ID
        num_instances,         // Total instances (for interleaving)
        rng                    // RNG state
    );

    // Set player position
    state.x[id] = px;
    state.y[id] = py;

    // Spawn monsters using REAL monster selection for depth
    int monster_count = spawn_monsters_for_depth(
        depth,
        state.monster_x,
        state.monster_y,
        state.monster_hp,
        state.monster_type,
        state.monster_awake,
        state.dungeon_terrain,
        id,
        num_instances,
        rng
    );

    state.monster_count[id] = monster_count;
}

// ============================================================================
// BORG DECISION KERNEL
// ============================================================================

__global__ void borg_think_kernel(
    BorgStateInterleaved state,
    uint8_t* actions,           // Output: action per instance
    curandState* rng_states,
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

    // Calculate danger using REAL monster stats
    int danger = calculate_danger(&state, id, x, y, num_instances);

    // Find nearest monster
    int mx, my, nearest_idx;
    int monster_dist = find_nearest_monster(&state, id, x, y, num_instances, &mx, &my, &nearest_idx);

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
// GAME STEP KERNEL (Execute actions with REAL game rules)
// ============================================================================

__global__ void borg_execute_kernel(
    BorgStateInterleaved state,
    uint8_t* actions,
    curandState* rng_states,
    uint32_t num_instances
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_instances) return;

    if (!state.alive[id]) return;

    curandState* rng = &rng_states[id];
    uint8_t action = actions[id];
    int x = state.x[id];
    int y = state.y[id];
    int hp = state.hp[id];
    int max_hp = state.max_hp[id];
    int depth = state.depth[id];
    int player_level = state.level[id];

    // Get player combat stats
    int player_skill = state.level[id] * 3;  // Base to-hit skill
    int player_ac = state.ac[id];
    int player_deadliness = state.damage[id];  // Deadliness bonus

    // Execute action
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
            // Teleport using REAL teleport logic
            teleport_player(
                &x, &y,
                state.dungeon_terrain,
                depth,
                id,
                num_instances,
                rng
            );
            break;

        case BORG_ACTION_ATTACK:
            {
                // Find nearest monster to attack
                int mx, my, target_idx;
                int dist = find_nearest_monster(&state, id, x, y, num_instances, &mx, &my, &target_idx);

                if (target_idx >= 0 && dist <= 1) {
                    // REAL COMBAT: Player attacks monster
                    int monster_type = IGET(state.monster_type, target_idx, id, num_instances);
                    int monster_hp = IGET(state.monster_hp, target_idx, id, num_instances);
                    int monster_ac = get_monster_ac(monster_type);

                    // Weapon stats (simplified - could be from inventory later)
                    int weapon_dd = 2;  // 2d6 weapon
                    int weapon_ds = 6;
                    int weapon_to_hit = 5;

                    // Calculate player to-hit
                    int to_hit = calc_player_to_hit(player_skill, weapon_to_hit);

                    // REAL MELEE ATTACK
                    int damage = melee_attack(
                        to_hit,
                        player_deadliness,
                        weapon_dd,
                        weapon_ds,
                        player_level,
                        monster_ac,
                        rng
                    );

                    monster_hp -= damage;

                    // Monster dies
                    if (monster_hp <= 0) {
                        // Award XP based on real monster
                        int xp_value = get_monster_xp(monster_type, depth);
                        state.exp[id] += xp_value;

                        // Remove monster (set HP to 0, will be cleaned up)
                        ISET(state.monster_hp, target_idx, id, num_instances, 0);

                        // Check for level up (simplified)
                        if (state.exp[id] > player_level * 100) {
                            state.level[id]++;
                            // Increase stats on level up
                            state.max_hp[id] += 10;
                            state.damage[id] += 2;
                        }
                    } else {
                        ISET(state.monster_hp, target_idx, id, num_instances, monster_hp);

                        // MONSTER RETALIATES using REAL monster attack
                        int retaliation_damage = monster_attack(
                            monster_type,
                            player_ac,
                            player_level,
                            rng
                        );

                        hp -= retaliation_damage;
                    }
                }
            }
            break;

        case BORG_ACTION_DESCEND:
            {
                // Go deeper - generate NEW level with REAL generation
                int new_depth = depth + 1;
                state.depth[id] = new_depth;

                // Generate level using REAL Angband dungeon generation
                int px, py, sx, sy;

                generate_level(
                    state.dungeon_terrain,
                    new_depth,
                    &px, &py,
                    &sx, &sy,
                    id,
                    num_instances,
                    rng
                );

                // Set new player position
                x = px;
                y = py;

                // Spawn monsters using REAL monster selection for depth
                int monster_count = spawn_monsters_for_depth(
                    new_depth,
                    state.monster_x,
                    state.monster_y,
                    state.monster_hp,
                    state.monster_type,
                    state.monster_awake,
                    state.dungeon_terrain,
                    id,
                    num_instances,
                    rng
                );

                state.monster_count[id] = monster_count;
            }
            break;

        case BORG_ACTION_EXPLORE:
            {
                // Move randomly (simplified pathfinding)
                int dx = curand(rng) % 3 - 1;  // -1, 0, or 1
                int dy = curand(rng) % 3 - 1;

                int new_x = x + dx;
                int new_y = y + dy;

                // Check if walkable using REAL terrain
                if (is_walkable(state.dungeon_terrain, new_x, new_y, id, num_instances)) {
                    x = new_x;
                    y = new_y;

                    // Chance to wake nearby monsters
                    wake_nearby_monsters(
                        state.monster_x,
                        state.monster_y,
                        state.monster_awake,
                        state.monster_count[id],
                        x, y,
                        id,
                        num_instances,
                        rng
                    );
                }
            }
            break;

        default:
            break;
    }

    // === MONSTER AI LOOP (Process monsters) ===
    int monster_count = state.monster_count[id];
    for (int i = 0; i < monster_count && i < MAX_MONSTERS; i++) {
        // Skip dead monsters
        if (IGET(state.monster_hp, i, id, num_instances) <= 0) continue;

        int mtype = IGET(state.monster_type, i, id, num_instances);
        const MonsterRace& race = MONSTER_RACES[mtype]; // From angband_monsters.cuh

        // 1. BREEDING LOGIC
        if (race.flags & MFLAG_BREEDS) {
            int mx = IGET(state.monster_x, i, id, num_instances);
            int my = IGET(state.monster_y, i, id, num_instances);
            
            try_breed_monster(
                i, mtype, mx, my,
                state.monster_x, state.monster_y, state.monster_hp,
                state.monster_type, state.monster_awake,
                &state.monster_count[id],
                state.dungeon_terrain,
                id, num_instances, rng
            );
        }
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

    // Check for win (depth 100 = Morgoth defeated)
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

void run_borg_simulation(uint32_t num_instances, uint32_t max_turns, bool verify_mode = false, uint32_t verify_seed = 12345) {
    printf("Running Angband Borg GPU simulation (V2 - Real Rules)\n");
    printf("Instances: %d, Max turns: %d\n", num_instances, max_turns);
    if (verify_mode) {
        printf("VERIFICATION MODE: seed=%u\n", verify_seed);
    }

    // Allocate state
    BorgStateInterleaved* state = borg_state_alloc(num_instances);

    // Allocate actions
    uint8_t* d_actions;
    CHECK(cudaMalloc(&d_actions, num_instances));

    // Allocate RNG states
    curandState* d_rng_states;
    CHECK(cudaMalloc(&d_rng_states, num_instances * sizeof(curandState)));

    // Initialize RNG
    int threads = 256;
    int blocks = (num_instances + threads - 1) / threads;

    uint32_t seed = verify_mode ? verify_seed : (uint32_t)time(NULL);
    init_rng_kernel<<<blocks, threads>>>(d_rng_states, seed, num_instances);
    CHECK(cudaDeviceSynchronize());

    // Initialize state (simplified)
    std::vector<int16_t> init_depth(num_instances, 1);
    std::vector<int16_t> init_level(num_instances, 1);
    std::vector<int16_t> init_hp(num_instances, 20); // Realistic Level 1 HP
    std::vector<int16_t> init_speed(num_instances, 10);
    std::vector<int16_t> init_ac(num_instances, 20);
    std::vector<int16_t> init_damage(num_instances, 15);
    std::vector<uint8_t> init_alive(num_instances, 1);
    std::vector<uint16_t> init_config(num_instances);
    std::vector<uint8_t> init_no_deeper(num_instances, 127);

    // Assign different configs to batches
    for (uint32_t i = 0; i < num_instances; i++) {
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

    CHECK(cudaMemcpy(state->depth, init_depth.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->level, init_level.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->hp, init_hp.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->max_hp, init_hp.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->speed, init_speed.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->ac, init_ac.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->damage, init_damage.data(), num_instances * sizeof(int16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->alive, init_alive.data(), num_instances, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->config, init_config.data(), num_instances * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(state->no_deeper, init_no_deeper.data(), num_instances, cudaMemcpyHostToDevice));

    // Initialize other fields
    CHECK(cudaMemset(state->has_healing, 1, num_instances));  // Start with healing
    CHECK(cudaMemset(state->has_recall, 1, num_instances));   // Start with recall
    CHECK(cudaMemset(state->has_teleport, 1, num_instances)); // Start with teleport
    CHECK(cudaMemset(state->turns, 0, num_instances * sizeof(uint32_t)));
    CHECK(cudaMemset(state->exp, 0, num_instances * sizeof(uint32_t)));
    CHECK(cudaMemset(state->winner, 0, num_instances));
    CHECK(cudaMemset(state->final_depth, 0, num_instances * sizeof(uint16_t)));
    CHECK(cudaMemset(state->final_turns, 0, num_instances * sizeof(uint32_t)));
    CHECK(cudaMemset(state->monster_count, 0, num_instances));

    // Generate initial dungeons
    printf("Generating initial dungeons...\n");
    init_dungeons_kernel<<<blocks, threads>>>(*state, d_rng_states, num_instances);
    CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Starting simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t turn = 0; turn < max_turns; turn++) {
        // Think
        borg_think_kernel<<<blocks, threads>>>(*state, d_actions, d_rng_states, num_instances);

        // Execute
        borg_execute_kernel<<<blocks, threads>>>(*state, d_actions, d_rng_states, num_instances);

        // Progress indicator every 100 turns
        if ((turn + 1) % 100 == 0) {
            CHECK(cudaDeviceSynchronize());
            printf("Turn %d/%d completed\r", turn + 1, max_turns);
            fflush(stdout);
        }
    }

    CHECK(cudaDeviceSynchronize());
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    printf("Simulation time: %.4f s\n", diff.count());
    printf("Throughput: %.2f instance-turns/sec\n", (double)num_instances * max_turns / diff.count());

    // Collect results
    std::vector<uint8_t> h_alive(num_instances);
    std::vector<int16_t> h_depth(num_instances);  // Current depth (for alive borgs)
    std::vector<uint16_t> h_final_depth(num_instances);  // Final depth (for dead borgs)
    std::vector<uint8_t> h_winner(num_instances);
    std::vector<int16_t> h_hp(num_instances);
    std::vector<int16_t> h_level(num_instances);

    CHECK(cudaMemcpy(h_alive.data(), state->alive, num_instances, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_depth.data(), state->depth, num_instances * sizeof(int16_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_final_depth.data(), state->final_depth, num_instances * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_winner.data(), state->winner, num_instances, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_hp.data(), state->hp, num_instances * sizeof(int16_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_level.data(), state->level, num_instances * sizeof(int16_t), cudaMemcpyDeviceToHost));

    // Count results by config type
    int alive_count[8] = {0};
    int winner_count[8] = {0};
    int dead_count[8] = {0};
    int total_depth[8] = {0};
    int max_depth[8] = {0};
    int total_level[8] = {0};
    int count[8] = {0};

    for (uint32_t i = 0; i < num_instances; i++) {
        int config_type = i % 8;
        count[config_type]++;
        total_level[config_type] += h_level[i];

        if (h_alive[i]) {
            alive_count[config_type]++;
            total_depth[config_type] += h_depth[i];  // Use current depth for alive
            if (h_depth[i] > max_depth[config_type]) max_depth[config_type] = h_depth[i];
        } else {
            dead_count[config_type]++;
            total_depth[config_type] += h_final_depth[i];  // Use final depth for dead
            if (h_final_depth[i] > max_depth[config_type]) max_depth[config_type] = h_final_depth[i];
        }
        if (h_winner[i]) winner_count[config_type]++;
    }

    const char* config_names[] = {"Aggro", "Speed", "Tank", "Scummer", "Meta", "Economy", "Cheat", "Default"};

    printf("\n=== RESULTS BY CONFIG ===\n");
    printf("% -10s %8s %8s %8s %8s %8s %8s %8s\n",
           "Config", "Alive%", "Dead%", "Win%", "AvgDepth", "MaxDepth", "AvgLevel", "Count");
    for (int i = 0; i < 8; i++) {
        if (count[i] > 0) {
            printf("% -10s %7.1f%% %7.1f%% %7.1f%% %8.1f %8d %8.1f %8d\n",
                config_names[i],
                100.0 * alive_count[i] / count[i],
                100.0 * dead_count[i] / count[i],
                100.0 * winner_count[i] / count[i],
                (double)total_depth[i] / count[i],
                max_depth[i],
                (double)total_level[i] / count[i],
                count[i]);
        }
    }

    // Summary stats
    int total_alive = 0, total_dead = 0, total_winners = 0;
    for (int i = 0; i < 8; i++) {
        total_alive += alive_count[i];
        total_dead += dead_count[i];
        total_winners += winner_count[i];
    }

    printf("\nSUMMARY: %d alive (%.1f%%), %d dead (%.1f%%), %d winners (%.3f%%)\n",
           total_alive, 100.0 * total_alive / num_instances,
           total_dead, 100.0 * total_dead / num_instances,
           total_winners, 100.0 * total_winners / num_instances);

    // Verification mode: print first 10 instances for reproducibility check
    if (verify_mode && num_instances >= 10) {
        printf("\n=== VERIFICATION DATA (first 10 instances) ===\n");
        printf("ID  Alive  Depth  Level  HP\n");
        for (int i = 0; i < 10; i++) {
            printf("%2d  %5s  %5d  %5d  %3d\n",
                   i,
                   h_alive[i] ? "YES" : "NO",
                   h_alive[i] ? h_depth[i] : h_final_depth[i],
                   h_level[i],
                   h_hp[i]);
        }
    }

    // Cleanup
    cudaFree(d_actions);
    cudaFree(d_rng_states);
    borg_state_free(state);
}

int main(int argc, char** argv) {
    uint32_t num_instances = 10000;
    uint32_t max_turns = 1000;
    bool verify_mode = false;
    uint32_t verify_seed = 12345;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify") == 0) {
            verify_mode = true;
            num_instances = 100;  // Smaller for verification
            if (i + 1 < argc) {
                verify_seed = atoi(argv[i + 1]);
                i++;
            }
        } else if (i == 1) {
            num_instances = atoi(argv[i]);
        } else if (i == 2) {
            max_turns = atoi(argv[i]);
        }
    }

    run_borg_simulation(num_instances, max_turns, verify_mode, verify_seed);

    return 0;
}