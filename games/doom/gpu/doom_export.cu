/**
 * GPU DOOM Export - Implementation
 *
 * Exports interesting moments as JSON for browser playback.
 */

#include "doom_export.cuh"
#include "doom_interest.cuh"
#include "doom_types.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

// =============================================================================
// Helper functions for interest type names and descriptions
// =============================================================================

static const char* GetInterestTypeName(InterestType type) {
    switch (type) {
        case IT_NEAR_DEATH:      return "near_death";
        case IT_MULTI_KILL:      return "multi_kill";
        case IT_DEATH:           return "death";
        case IT_LEVEL_COMPLETE:  return "level_complete";
        case IT_SPEEDRUN:        return "speedrun";
        case IT_LOW_HEALTH_KILL: return "low_health_kill";
        default:                 return "unknown";
    }
}

static const char* GetInterestDescription(InterestType type, int health, int kills, int min_health) {
    static char buf[256];

    switch (type) {
        case IT_NEAR_DEATH:
            snprintf(buf, sizeof(buf),
                "Survived with only %d HP after dropping to %d HP",
                health, min_health);
            break;
        case IT_MULTI_KILL:
            snprintf(buf, sizeof(buf),
                "%d kills in rapid succession",
                kills);
            break;
        case IT_DEATH:
            snprintf(buf, sizeof(buf),
                "Player died - can you survive?");
            break;
        case IT_LOW_HEALTH_KILL:
            snprintf(buf, sizeof(buf),
                "Got a kill at only %d HP",
                health);
            break;
        case IT_LEVEL_COMPLETE:
            snprintf(buf, sizeof(buf),
                "Completed the level with %d HP",
                health);
            break;
        case IT_SPEEDRUN:
            snprintf(buf, sizeof(buf),
                "Fastest clear recorded");
            break;
        default:
            snprintf(buf, sizeof(buf), "Unknown event");
    }

    return buf;
}

static const char* GetLevelNameFromNum(int level) {
    static const char* names[] = {
        "E1M1", "E1M2", "E1M3", "E1M4", "E1M5",
        "E1M6", "E1M7", "E1M8", "E1M9"
    };
    if (level >= 1 && level <= 9) return names[level - 1];
    return "Unknown";
}

// =============================================================================
// Helper: Create directory recursively
// =============================================================================

static void MakeDir(const char* path) {
    char tmp[512];
    char* p = nullptr;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    if (tmp[len - 1] == '/') tmp[len - 1] = 0;

    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

// =============================================================================
// Export Scenarios
// =============================================================================

int ExportScenarios(
    const char* output_dir,
    int num_instances,
    InterestState* h_interest,
    TicCmd* h_inputs,
    int num_ticks,
    int checkpoint_interval
) {
    // Create output directories
    char scenarios_dir[512];
    snprintf(scenarios_dir, sizeof(scenarios_dir), "%s/scenarios", output_dir);
    MakeDir(scenarios_dir);

    // Collect all interesting scenarios
    ScenarioExport* scenarios = (ScenarioExport*)malloc(
        num_instances * MAX_INTEREST_EVENTS * sizeof(ScenarioExport)
    );
    int num_scenarios = 0;

    printf("\nScanning for interesting moments...\n");

    for (int i = 0; i < num_instances; i++) {
        InterestState& state = h_interest[i];

        for (int e = 0; e < state.num_events; e++) {
            InterestEvent& event = state.events[e];

            if (event.type == IT_NONE) continue;

            // Only export high-quality events (score >= 60)
            if (event.interest_score < 60) continue;

            ScenarioExport& sc = scenarios[num_scenarios];
            memset(&sc, 0, sizeof(sc));

            // Generate unique ID
            snprintf(sc.scenario_id, sizeof(sc.scenario_id),
                "%s-%s-%d",
                GetInterestTypeName(event.type),
                GetLevelNameFromNum(event.level),
                i
            );

            // Copy type
            strncpy(sc.type, GetInterestTypeName(event.type), sizeof(sc.type) - 1);

            // Generate description
            strncpy(sc.description,
                GetInterestDescription(
                    event.type,
                    event.player_health,
                    event.kills_in_window,
                    event.min_health
                ),
                sizeof(sc.description) - 1
            );

            // Level
            strncpy(sc.level, GetLevelNameFromNum(event.level), sizeof(sc.level) - 1);

            // Event details
            sc.tick = event.tick;
            sc.interest_score = event.interest_score;
            sc.instance_id = i;

            // Player state (would need to be captured from checkpoints)
            sc.player.health = event.player_health;
            sc.player.armor = 0;  // TODO: capture from checkpoint

            // Input history
            if (h_inputs != nullptr) {
                // Capture inputs from (tick - checkpoint_interval) to tick
                int start_tick = event.tick - checkpoint_interval;
                if (start_tick < 0) start_tick = 0;
                int input_count = event.tick - start_tick;

                sc.inputs = (TicCmd*)malloc(input_count * sizeof(TicCmd));
                sc.num_inputs = input_count;

                for (int t = 0; t < input_count; t++) {
                    int src_tick = start_tick + t;
                    int src_idx = src_tick * num_instances + i;
                    sc.inputs[t] = h_inputs[src_idx];
                }
            }

            num_scenarios++;

            // Limit total exports
            if (num_scenarios >= 100) {
                printf("  Reached export limit (100 scenarios)\n");
                goto done_scanning;
            }
        }
    }

done_scanning:
    printf("  Found %d interesting scenarios (score >= 60)\n", num_scenarios);

    // Sort by interest score (descending)
    for (int i = 0; i < num_scenarios - 1; i++) {
        for (int j = i + 1; j < num_scenarios; j++) {
            if (scenarios[j].interest_score > scenarios[i].interest_score) {
                ScenarioExport tmp = scenarios[i];
                scenarios[i] = scenarios[j];
                scenarios[j] = tmp;
            }
        }
    }

    // Write individual scenario files
    for (int i = 0; i < num_scenarios; i++) {
        char filepath[512];
        snprintf(filepath, sizeof(filepath),
            "%s/%s.json", scenarios_dir, scenarios[i].scenario_id);

        if (WriteScenarioJSON(filepath, &scenarios[i])) {
            printf("  [%2d] %s (score: %d)\n",
                i + 1, scenarios[i].scenario_id, scenarios[i].interest_score);
        }
    }

    // Write index file
    WriteScenarioIndex(output_dir, scenarios, num_scenarios);

    // Cleanup
    for (int i = 0; i < num_scenarios; i++) {
        if (scenarios[i].inputs) free(scenarios[i].inputs);
    }
    free(scenarios);

    return num_scenarios;
}

// =============================================================================
// Write Scenario Index
// =============================================================================

void WriteScenarioIndex(
    const char* output_dir,
    ScenarioExport* scenarios,
    int num_scenarios
) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/scenarios/index.json", output_dir);

    FILE* f = fopen(filepath, "w");
    if (!f) {
        printf("ERROR: Could not write %s: %s\n", filepath, strerror(errno));
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"version\": 1,\n");
    fprintf(f, "  \"generated\": \"GPU DOOM Arena\",\n");
    fprintf(f, "  \"count\": %d,\n", num_scenarios);
    fprintf(f, "  \"scenarios\": [\n");

    for (int i = 0; i < num_scenarios; i++) {
        ScenarioExport& sc = scenarios[i];

        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": \"%s\",\n", sc.scenario_id);
        fprintf(f, "      \"type\": \"%s\",\n", sc.type);
        fprintf(f, "      \"description\": \"%s\",\n", sc.description);
        fprintf(f, "      \"level\": \"%s\",\n", sc.level);
        fprintf(f, "      \"score\": %d,\n", sc.interest_score);
        fprintf(f, "      \"health\": %d\n", sc.player.health);
        fprintf(f, "    }%s\n", i < num_scenarios - 1 ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("\nWrote scenario index: %s\n", filepath);
}

// =============================================================================
// Write Single Scenario JSON
// =============================================================================

bool WriteScenarioJSON(const char* filepath, const ScenarioExport* scenario) {
    FILE* f = fopen(filepath, "w");
    if (!f) return false;

    fprintf(f, "{\n");
    fprintf(f, "  \"scenario_id\": \"%s\",\n", scenario->scenario_id);
    fprintf(f, "  \"type\": \"%s\",\n", scenario->type);
    fprintf(f, "  \"description\": \"%s\",\n", scenario->description);
    fprintf(f, "  \"level\": \"%s\",\n", scenario->level);
    fprintf(f, "  \"tick\": %d,\n", scenario->tick);
    fprintf(f, "  \"interest_score\": %d,\n", scenario->interest_score);
    fprintf(f, "  \"instance_id\": %d,\n", scenario->instance_id);

    // Player state
    fprintf(f, "  \"player\": {\n");
    fprintf(f, "    \"x\": %d,\n", scenario->player.x);
    fprintf(f, "    \"y\": %d,\n", scenario->player.y);
    fprintf(f, "    \"z\": %d,\n", scenario->player.z);
    fprintf(f, "    \"angle\": %d,\n", scenario->player.angle);
    fprintf(f, "    \"health\": %d,\n", scenario->player.health);
    fprintf(f, "    \"armor\": %d\n", scenario->player.armor);
    fprintf(f, "  },\n");

    // Input history
    fprintf(f, "  \"input_history\": [\n");
    for (int t = 0; t < scenario->num_inputs; t++) {
        const TicCmd& cmd = scenario->inputs[t];
        fprintf(f, "    {\"tick\": %d, \"forward\": %d, \"side\": %d, \"angle\": %d, \"buttons\": %d}%s\n",
            t,
            cmd.forwardmove,
            cmd.sidemove,
            cmd.angleturn,
            cmd.buttons,
            t < scenario->num_inputs - 1 ? "," : ""
        );
    }
    fprintf(f, "  ]\n");

    fprintf(f, "}\n");

    fclose(f);
    return true;
}

// =============================================================================
// Convert to DOOM Demo (.lmp)
// =============================================================================

size_t ScenarioToDemo(
    const ScenarioExport* scenario,
    uint8_t* demo_buffer,
    size_t buffer_size
) {
    // DOOM demo format:
    // Byte 0: Version (109 for DOOM 1.9)
    // Byte 1: Skill (0-4)
    // Byte 2: Episode (1-4)
    // Byte 3: Map (1-9)
    // Bytes 4+: TicCmds (4 bytes each)
    // End: 0x80 terminator

    size_t demo_size = 4 + (scenario->num_inputs * 4) + 1;
    if (demo_size > buffer_size) return 0;

    uint8_t* p = demo_buffer;

    // Header
    *p++ = 109;  // Version (DOOM 1.9)
    *p++ = 3;    // Skill (Ultra-Violence)

    // Parse level string (e.g., "E1M1" -> episode=1, map=1)
    int episode = 1, map = 1;
    if (strlen(scenario->level) >= 4) {
        episode = scenario->level[1] - '0';
        map = scenario->level[3] - '0';
    }
    *p++ = episode;
    *p++ = map;

    // TicCmds (4 bytes each in demo format)
    for (int t = 0; t < scenario->num_inputs; t++) {
        const TicCmd& cmd = scenario->inputs[t];
        *p++ = (uint8_t)cmd.forwardmove;
        *p++ = (uint8_t)cmd.sidemove;
        *p++ = (uint8_t)(cmd.angleturn >> 8);  // High byte of turn
        *p++ = cmd.buttons;
    }

    // Terminator
    *p++ = 0x80;

    return p - demo_buffer;
}
