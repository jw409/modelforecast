/*
 * AI ARENA - Unified Game Interface
 *
 * Common header for GPU-accelerated game simulations.
 * Supports CoreWars, Angband, and future games.
 *
 * Memory Layout: Interleaved (best coalescing on NVIDIA GPUs)
 * Pattern: array[row * num_instances + instance_id]
 */

#ifndef ARENA_H
#define ARENA_H

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// CONSTANTS
// ============================================================================

#define MAX_INSTANCES       100000   // Max parallel game instances
#define MAX_CONTESTANTS     64       // Max remote LLMs competing
#define MAX_ACTION_LOG      1000000  // Actions logged per episode
#define ACTION_TIMEOUT_MS   5000     // Remote LLM think time limit

// Game type identifiers
#define GAME_COREWARS       0
#define GAME_ANGBAND        1
#define GAME_NETHACK        2   // Future
#define GAME_CUSTOM         255

// ============================================================================
// INTERLEAVED INDEXING
// ============================================================================

#define IDX(row, col, width) ((size_t)(row) * (width) + (col))

// Accessor macros for interleaved arrays
#define GET_INTERLEAVED(arr, row, instance, num_instances) \
    (arr)[IDX(row, instance, num_instances)]

#define SET_INTERLEAVED(arr, row, instance, num_instances, val) \
    (arr)[IDX(row, instance, num_instances)] = (val)

// ============================================================================
// CONTESTANT (Remote LLM)
// ============================================================================

typedef struct {
    uint32_t id;                    // Contestant ID
    char name[64];                  // Model name (e.g., "gpt-5", "claude-4")
    char api_endpoint[256];         // OpenRouter or direct endpoint
    uint32_t instances_owned;       // How many game instances this contestant controls
    uint32_t* instance_ids;         // Which instances

    // Stats
    uint32_t wins;
    uint32_t losses;
    uint32_t ties;
    float elo_rating;

    // Code modifications (if allowed)
    char* custom_code;              // CUDA/C code submitted
    size_t code_size;
    uint8_t code_validated;         // Passed sandbox check
} Contestant;

// ============================================================================
// ACTION TYPES (Common across games)
// ============================================================================

typedef enum {
    // Universal
    ACTION_NONE = 0,
    ACTION_WAIT,

    // Movement
    ACTION_MOVE_N,
    ACTION_MOVE_S,
    ACTION_MOVE_E,
    ACTION_MOVE_W,
    ACTION_MOVE_NE,
    ACTION_MOVE_NW,
    ACTION_MOVE_SE,
    ACTION_MOVE_SW,

    // Combat
    ACTION_ATTACK,
    ACTION_DEFEND,
    ACTION_FLEE,

    // Items (Angband)
    ACTION_USE_ITEM,
    ACTION_PICKUP,
    ACTION_DROP,

    // Special
    ACTION_CAST_SPELL,
    ACTION_REST,
    ACTION_DESCEND,
    ACTION_ASCEND,

    // CoreWars specific (mapped to instructions)
    ACTION_CW_MOV = 100,
    ACTION_CW_ADD,
    ACTION_CW_SUB,
    ACTION_CW_JMP,
    ACTION_CW_SPL,
    ACTION_CW_DAT,

    // Meta actions (for LLM contestants)
    ACTION_SUBMIT_CODE = 200,       // Submit custom borg/warrior code
    ACTION_MODIFY_CONFIG,           // Change config parameters
    ACTION_REQUEST_STATE,           // Ask for detailed state dump

    ACTION_MAX
} ActionType;

// ============================================================================
// ACTION LOG ENTRY (For Netflix show replay)
// ============================================================================

typedef struct {
    uint32_t turn;                  // Game turn number
    uint32_t instance_id;           // Which game instance
    uint32_t contestant_id;         // Which LLM
    uint8_t game_type;              // GAME_COREWARS, GAME_ANGBAND, etc.
    uint8_t action;                 // ActionType
    int16_t param1;                 // Action-specific parameter
    int16_t param2;
    uint32_t thinking_time_ms;      // How long LLM took to decide
    uint8_t result;                 // 0=success, 1=fail, 2=death
    char reasoning[256];            // LLM's explanation (truncated)
} ActionLogEntry;

// ============================================================================
// GAME STATE (Abstract - each game implements)
// ============================================================================

typedef struct {
    uint8_t game_type;
    uint32_t num_instances;
    void* game_state;               // Game-specific state pointer
    size_t state_size_per_instance;

    // Interleaved result arrays
    uint8_t* winners;               // [num_instances]
    uint32_t* turns_played;         // [num_instances]
    uint8_t* death_causes;          // [num_instances]

    // Action log
    ActionLogEntry* action_log;
    uint32_t log_count;
    uint32_t log_capacity;
} ArenaState;

// ============================================================================
// ARENA API (Host-side)
// ============================================================================

// Initialize arena with specified game type and instance count
ArenaState* arena_init(uint8_t game_type, uint32_t num_instances);

// Register a contestant (remote LLM)
int arena_register_contestant(ArenaState* arena, Contestant* contestant);

// Submit action for a contestant's instances
int arena_submit_action(ArenaState* arena, uint32_t contestant_id,
                        uint8_t action, int16_t param1, int16_t param2);

// Submit custom code (validated in sandbox before compilation)
int arena_submit_code(ArenaState* arena, uint32_t contestant_id,
                      const char* code, size_t code_size);

// Execute one turn for all instances (launches GPU kernel)
int arena_step(ArenaState* arena);

// Get state for contestant observation (serialized for API)
char* arena_get_state_json(ArenaState* arena, uint32_t contestant_id);

// Get action log for replay
ActionLogEntry* arena_get_log(ArenaState* arena, uint32_t* count);

// Cleanup
void arena_destroy(ArenaState* arena);

// ============================================================================
// NARRATOR API (Local observer model)
// ============================================================================

typedef struct {
    char* commentary;               // Generated text
    float drama_score;              // 0-1, how dramatic was this turn
    uint8_t highlight;              // Flag for highlight reel
    char highlight_reason[128];     // Why this is a highlight
} NarratorOutput;

// Generate commentary for current state
NarratorOutput* narrator_describe(ArenaState* arena, uint32_t turn);

// ============================================================================
// CUDA KERNEL INTERFACE
// ============================================================================

// Generic step kernel - dispatches to game-specific implementation
__global__ void arena_step_kernel(
    void* game_state,
    uint8_t game_type,
    uint8_t* actions,
    uint8_t* results,
    uint32_t num_instances
);

#endif // ARENA_H
