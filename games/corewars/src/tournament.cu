#ifndef TOURNAMENT_CU
#define TOURNAMENT_CU

#include "mars.h"
#include "warrior_loader.cu"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

struct MatchResult {
    int warrior_a_wins;
    int warrior_b_wins;
    int ties;
    float avg_cycles;
};

struct TournamentResult {
    int warrior_id;
    int total_wins;
    int total_losses;
    int total_ties;
    float win_rate;
    int points; // 3 for win, 1 for tie, 0 for loss
};

// Forward declare the battle kernel (implemented in mars_kernel.cu or similar)
extern __global__ void mars_battle_kernel(
    BattleState* battles,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    const uint16_t* warrior_lengths_a,
    const uint16_t* warrior_lengths_b,
    const uint32_t* start_positions_a,
    const uint32_t* start_positions_b,
    uint32_t num_battles
);

// CUDA error checking
void checkCudaTournament(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK_T(val) checkCudaTournament((val), #val, __FILE__, __LINE__)

// Run a single match: warrior_a vs warrior_b for num_rounds battles
__host__ void run_match(
    const Warrior* warrior_a,
    const Warrior* warrior_b,
    int num_rounds,
    MatchResult* result
) {
    printf("  Match: %s vs %s (%d rounds)\n", warrior_a->name, warrior_b->name, num_rounds);

    // Allocate device memory for warriors
    Instruction *d_warrior_a, *d_warrior_b;
    CUDA_CHECK_T(cudaMalloc(&d_warrior_a, warrior_a->length * sizeof(Instruction)));
    CUDA_CHECK_T(cudaMalloc(&d_warrior_b, warrior_b->length * sizeof(Instruction)));
    CUDA_CHECK_T(cudaMemcpy(d_warrior_a, warrior_a->code, warrior_a->length * sizeof(Instruction), cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_warrior_b, warrior_b->code, warrior_b->length * sizeof(Instruction), cudaMemcpyHostToDevice));

    // Prepare battle configurations with randomized starting positions
    uint16_t *d_len_a, *d_len_b;
    uint32_t *d_pos_a, *d_pos_b;
    CUDA_CHECK_T(cudaMalloc(&d_len_a, num_rounds * sizeof(uint16_t)));
    CUDA_CHECK_T(cudaMalloc(&d_len_b, num_rounds * sizeof(uint16_t)));
    CUDA_CHECK_T(cudaMalloc(&d_pos_a, num_rounds * sizeof(uint32_t)));
    CUDA_CHECK_T(cudaMalloc(&d_pos_b, num_rounds * sizeof(uint32_t)));

    uint16_t* h_len_a = (uint16_t*)malloc(num_rounds * sizeof(uint16_t));
    uint16_t* h_len_b = (uint16_t*)malloc(num_rounds * sizeof(uint16_t));
    uint32_t* h_pos_a = (uint32_t*)malloc(num_rounds * sizeof(uint32_t));
    uint32_t* h_pos_b = (uint32_t*)malloc(num_rounds * sizeof(uint32_t));

    for (int i = 0; i < num_rounds; i++) {
        h_len_a[i] = warrior_a->length;
        h_len_b[i] = warrior_b->length;
        h_pos_a[i] = rand() % (CORE_SIZE - 100);
        h_pos_b[i] = (h_pos_a[i] + 4000) % CORE_SIZE; // Start far apart
    }

    CUDA_CHECK_T(cudaMemcpy(d_len_a, h_len_a, num_rounds * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_len_b, h_len_b, num_rounds * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_pos_a, h_pos_a, num_rounds * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK_T(cudaMemcpy(d_pos_b, h_pos_b, num_rounds * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate battle states
    BattleState* d_battles;
    CUDA_CHECK_T(cudaMalloc(&d_battles, num_rounds * sizeof(BattleState)));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rounds + threadsPerBlock - 1) / threadsPerBlock;

    mars_battle_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_battles,
        d_warrior_a, d_warrior_b,
        d_len_a, d_len_b,
        d_pos_a, d_pos_b,
        num_rounds
    );

    CUDA_CHECK_T(cudaDeviceSynchronize());
    CUDA_CHECK_T(cudaGetLastError());

    // Retrieve results
    BattleState* h_battles = (BattleState*)malloc(num_rounds * sizeof(BattleState));
    CUDA_CHECK_T(cudaMemcpy(h_battles, d_battles, num_rounds * sizeof(BattleState), cudaMemcpyDeviceToHost));

    // Tally results
    result->warrior_a_wins = 0;
    result->warrior_b_wins = 0;
    result->ties = 0;
    long total_cycles = 0;

    for (int i = 0; i < num_rounds; i++) {
        if (h_battles[i].winner == 1) result->warrior_a_wins++;
        else if (h_battles[i].winner == 2) result->warrior_b_wins++;
        else result->ties++;
        total_cycles += h_battles[i].cycles;
    }

    result->avg_cycles = (float)total_cycles / num_rounds;

    printf("    Result: %s:%d %s:%d Ties:%d (avg cycles: %.0f)\n",
           warrior_a->name, result->warrior_a_wins,
           warrior_b->name, result->warrior_b_wins,
           result->ties, result->avg_cycles);

    // Cleanup
    free(h_len_a);
    free(h_len_b);
    free(h_pos_a);
    free(h_pos_b);
    free(h_battles);
    cudaFree(d_warrior_a);
    cudaFree(d_warrior_b);
    cudaFree(d_len_a);
    cudaFree(d_len_b);
    cudaFree(d_pos_a);
    cudaFree(d_pos_b);
    cudaFree(d_battles);
}

// Run full round-robin tournament
__host__ void run_tournament(
    Warrior* warriors,
    int num_warriors,
    int rounds_per_match,
    TournamentResult* results
) {
    printf("\n=== TOURNAMENT: %d warriors, %d rounds per match ===\n\n", num_warriors, rounds_per_match);

    // Initialize results
    for (int i = 0; i < num_warriors; i++) {
        results[i].warrior_id = i;
        results[i].total_wins = 0;
        results[i].total_losses = 0;
        results[i].total_ties = 0;
        results[i].points = 0;
    }

    // Round-robin: every warrior vs every other warrior
    for (int i = 0; i < num_warriors; i++) {
        for (int j = i + 1; j < num_warriors; j++) {
            MatchResult match;
            run_match(&warriors[i], &warriors[j], rounds_per_match, &match);

            // Update results for warrior i
            results[i].total_wins += match.warrior_a_wins;
            results[i].total_losses += match.warrior_b_wins;
            results[i].total_ties += match.ties;
            results[i].points += match.warrior_a_wins * 3 + match.ties * 1;

            // Update results for warrior j
            results[j].total_wins += match.warrior_b_wins;
            results[j].total_losses += match.warrior_a_wins;
            results[j].total_ties += match.ties;
            results[j].points += match.warrior_b_wins * 3 + match.ties * 1;
        }
    }

    // Calculate win rates
    for (int i = 0; i < num_warriors; i++) {
        int total_battles = results[i].total_wins + results[i].total_losses + results[i].total_ties;
        results[i].win_rate = total_battles > 0 ? (float)results[i].total_wins / total_battles : 0.0f;
    }

    // Print final standings (sorted by points)
    printf("\n=== FINAL STANDINGS ===\n");
    printf("%-20s | Wins | Losses | Ties | Points | Win Rate\n", "Warrior");
    printf("----------------------------------------------------------------------\n");

    // Simple bubble sort by points
    for (int i = 0; i < num_warriors - 1; i++) {
        for (int j = 0; j < num_warriors - i - 1; j++) {
            if (results[j].points < results[j + 1].points) {
                TournamentResult temp = results[j];
                results[j] = results[j + 1];
                results[j + 1] = temp;
            }
        }
    }

    for (int i = 0; i < num_warriors; i++) {
        int id = results[i].warrior_id;
        printf("%-20s | %4d | %6d | %4d | %6d | %5.1f%%\n",
               warriors[id].name,
               results[i].total_wins,
               results[i].total_losses,
               results[i].total_ties,
               results[i].points,
               results[i].win_rate * 100.0f);
    }
    printf("\n");
}

// Convenience function: Load warriors from directory and run tournament
__host__ void run_tournament_from_directory(const char* warrior_dir, int rounds_per_match) {
    // This is a simplified version - in practice you'd scan the directory
    // For now, manually specify warrior files
    const char* warrior_files[] = {
        "warriors/champions/dwarf.red",
        "warriors/champions/imp_gate.red",
        "warriors/champions/mice.red"
    };
    int num_warriors = 3;

    Warrior* warriors = (Warrior*)malloc(num_warriors * sizeof(Warrior));

    for (int i = 0; i < num_warriors; i++) {
        if (!load_warrior_from_file(warrior_files[i], &warriors[i])) {
            fprintf(stderr, "Failed to load warrior from %s\n", warrior_files[i]);
            free(warriors);
            return;
        }
    }

    TournamentResult* results = (TournamentResult*)malloc(num_warriors * sizeof(TournamentResult));
    run_tournament(warriors, num_warriors, rounds_per_match, results);

    free(warriors);
    free(results);
}

#endif // TOURNAMENT_CU
