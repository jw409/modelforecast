/**
 * GPU DOOM Export
 *
 * Exports interesting moments as JSON scenarios for browser playback.
 * Also generates DOOM .lmp demo files for native playback.
 */

#ifndef DOOM_EXPORT_CUH
#define DOOM_EXPORT_CUH

#include "doom_types.cuh"
#include "doom_interest.cuh"
#include <cstdint>

// =============================================================================
// Scenario Export
// =============================================================================

struct ScenarioExport {
    char scenario_id[64];       // e.g., "near-death-e1m1-42"
    char type[32];              // e.g., "near_death"
    char description[256];      // Human-readable description
    char level[8];              // e.g., "E1M1"
    int32_t tick;               // Tick when event occurred
    int32_t interest_score;     // 0-100
    int32_t instance_id;        // Which GPU instance

    // Player state at moment
    struct {
        int32_t x, y, z;        // Position (world coords, not fixed)
        int32_t angle;          // Degrees (0-360)
        int32_t health;
        int32_t armor;
    } player;

    // Input history (CHECKPOINT_INTERVAL ticks before event)
    int32_t num_inputs;
    TicCmd* inputs;             // Allocated separately

    // Optional: monster snapshot (for replay accuracy)
    int32_t num_monsters;
    // MonsterSnapshot* monsters; // Could add if needed
};

// =============================================================================
// Export Functions
// =============================================================================

/**
 * Export all interesting scenarios to JSON files.
 *
 * @param output_dir Directory to write scenarios/ subfolder
 * @param num_instances Number of GPU instances
 * @param h_interest Interest events copied from GPU
 * @param h_inputs Input buffer copied from GPU (or NULL for header-only)
 * @param num_ticks Total ticks simulated
 * @param checkpoint_interval Interval for input capture window
 * @return Number of scenarios exported
 */
int ExportScenarios(
    const char* output_dir,
    int num_instances,
    InterestState* h_interest,
    TicCmd* h_inputs,
    int num_ticks,
    int checkpoint_interval
);

/**
 * Write scenario index (scenarios/index.json) for browser.
 */
void WriteScenarioIndex(
    const char* output_dir,
    ScenarioExport* scenarios,
    int num_scenarios
);

/**
 * Write single scenario JSON file.
 */
bool WriteScenarioJSON(
    const char* filepath,
    const ScenarioExport* scenario
);

/**
 * Convert scenario to DOOM demo format (.lmp).
 * Returns demo size in bytes, or 0 on failure.
 */
size_t ScenarioToDemo(
    const ScenarioExport* scenario,
    uint8_t* demo_buffer,
    size_t buffer_size
);

#endif // DOOM_EXPORT_CUH
