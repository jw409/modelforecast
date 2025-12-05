#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <vector>
#include "warrior_loader.cu"
#include "tournament.cu"

// Forward declare kernel (from mars_kernel.cu or mars_aos_optimized.cu)
__global__ void mars_battle_kernel(
    BattleState* battles,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    const uint16_t* warrior_lengths_a,
    const uint16_t* warrior_lengths_b,
    const uint32_t* start_positions_a,
    const uint32_t* start_positions_b,
    uint32_t num_battles
);

// Load all .red files from a directory
int load_warriors_from_directory(const char* dir_path, std::vector<Warrior>& warriors) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Failed to open directory: %s\n", dir_path);
        return 0;
    }

    struct dirent* entry;
    int count = 0;

    while ((entry = readdir(dir)) != nullptr) {
        // Check for .red extension
        size_t len = strlen(entry->d_name);
        if (len < 5 || strcmp(entry->d_name + len - 4, ".red") != 0) {
            continue;
        }

        // Build full path
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);

        // Load warrior
        Warrior w;
        if (load_warrior_from_file(full_path, &w)) {
            warriors.push_back(w);
            count++;
        } else {
            fprintf(stderr, "Warning: Failed to load %s\n", full_path);
        }
    }

    closedir(dir);
    return count;
}

int main(int argc, char** argv) {
    int rounds_per_match = 1000;

    if (argc < 2) {
        printf("Usage: %s <warrior_directory> [rounds_per_match]\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s warriors/gemini-2.5-flash 1000\n", argv[0]);
        printf("  %s warriors/champions 5000\n\n", argv[0]);
        return 1;
    }

    if (argc > 2) {
        rounds_per_match = atoi(argv[2]);
    }

    const char* warrior_dir = argv[1];

    printf("CoreWars GPU Tournament\n");
    printf("=======================\n");
    printf("Loading warriors from: %s\n", warrior_dir);
    printf("Rounds per match: %d\n\n", rounds_per_match);

    // Load all warriors from directory
    std::vector<Warrior> warriors;
    int num_loaded = load_warriors_from_directory(warrior_dir, warriors);

    if (num_loaded < 2) {
        fprintf(stderr, "Error: Need at least 2 warriors for tournament (loaded %d)\n", num_loaded);
        return 1;
    }

    printf("Loaded %d warriors:\n", num_loaded);
    for (size_t i = 0; i < warriors.size(); i++) {
        printf("  [%zu] %s by %s (%d instructions)\n",
               i, warriors[i].name, warriors[i].author, warriors[i].length);
    }
    printf("\n");

    // Allocate results
    TournamentResult* results = (TournamentResult*)malloc(warriors.size() * sizeof(TournamentResult));

    // Run tournament
    run_tournament(warriors.data(), warriors.size(), rounds_per_match, results);

    // Print final summary
    printf("\n=== CHAMPION ===\n");
    printf("%s by %s\n", warriors[results[0].warrior_id].name,
                         warriors[results[0].warrior_id].author);
    printf("Win Rate: %.1f%% (%d wins, %d losses, %d ties)\n",
           results[0].win_rate * 100.0f,
           results[0].total_wins,
           results[0].total_losses,
           results[0].total_ties);

    free(results);

    return 0;
}
