/**
 * GPU DOOM Interest Detection
 *
 * Detects "interesting moments" for browser playback:
 * - Near-death escapes (health dropped low, recovered)
 * - Multi-kills (3+ kills in short window)
 * - Deaths (for "can you survive?" challenges)
 * - Speed achievements
 *
 * Outputs JSON scenarios for browser consumption.
 */

#ifndef DOOM_INTEREST_CUH
#define DOOM_INTEREST_CUH

#include "doom_types.cuh"
#include <cstdint>

// =============================================================================
// Interest Types
// =============================================================================

enum InterestType : uint8_t {
    IT_NONE = 0,
    IT_NEAR_DEATH,      // Health dropped below 20, now above 50
    IT_MULTI_KILL,      // 3+ kills in 175 ticks (5 seconds)
    IT_DEATH,           // Player died (can you survive?)
    IT_LEVEL_COMPLETE,  // Finished the level
    IT_SPEEDRUN,        // Fastest clear of section
    IT_LOW_HEALTH_KILL, // Got a kill while at low health
};

// =============================================================================
// Interest Event (GPU-side tracking)
// =============================================================================

#define MAX_INTEREST_EVENTS 16  // Per instance

struct __align__(16) InterestEvent {
    InterestType type;
    uint8_t      interest_score;    // 0-100, higher = more interesting
    uint8_t      level;             // E1M1 = 1, etc.
    uint8_t      _pad;
    int32_t      tick;              // When it happened
    int32_t      player_health;     // Health at moment
    int16_t      kills_in_window;   // For multi-kill
    int16_t      min_health;        // Lowest health reached (for near-death)
};

// =============================================================================
// Interest Tracking State (per instance)
// =============================================================================

struct __align__(32) InterestState {
    // Rolling window for kills (last 175 ticks = 5 seconds)
    int16_t  kills_window[8];       // 8 slots, each covers ~22 ticks
    int16_t  window_idx;
    int16_t  total_kills_window;

    // Health tracking for near-death
    int16_t  min_health_recent;     // Lowest in last 350 ticks (10 sec)
    int16_t  min_health_tick;       // When we hit minimum

    // Recorded events
    InterestEvent events[MAX_INTEREST_EVENTS];
    int16_t  num_events;
    int16_t  _pad;
};

// =============================================================================
// Device Global State
// =============================================================================

// Interest state array (allocated in doom_main.cu)
// Defined in doom_main.cu, declared here for reference
// __device__ InterestState* d_interest_state;

// =============================================================================
// GPU Functions (inline to avoid link issues)
// =============================================================================

/**
 * Record an interesting event.
 * Returns true if event was recorded (space available).
 */
__device__ inline bool RecordInterestEvent(
    InterestState* state,
    InterestType type,
    int tick,
    int health,
    int score,
    int level,
    int kills_in_window,
    int min_health
) {
    if (state->num_events >= MAX_INTEREST_EVENTS) {
        return false;  // Buffer full
    }

    // Don't record duplicate events too close together
    if (state->num_events > 0) {
        InterestEvent& last = state->events[state->num_events - 1];
        if (last.type == type && (tick - last.tick) < 70) {  // 2 seconds
            return false;  // Too close to previous
        }
    }

    InterestEvent& event = state->events[state->num_events];
    event.type = type;
    event.interest_score = (uint8_t)score;
    event.level = (uint8_t)level;
    event.tick = tick;
    event.player_health = health;
    event.kills_in_window = kills_in_window;
    event.min_health = min_health;

    state->num_events++;
    return true;
}

/**
 * Initialize interest tracking for an instance.
 * Called at start of simulation.
 */
__device__ inline void InitInterestTracking_Impl(InterestState* state) {
    // Clear kills window
    for (int i = 0; i < 8; i++) {
        state->kills_window[i] = 0;
    }
    state->window_idx = 0;
    state->total_kills_window = 0;

    // Clear health tracking
    state->min_health_recent = 100;
    state->min_health_tick = 0;

    // Clear events
    state->num_events = 0;
}

/**
 * Update interest tracking after each tick.
 * Detects interesting moments based on state changes.
 */
__device__ inline void UpdateInterestTracking_Impl(
    InterestState* state,
    int tick,
    int prev_health,
    int curr_health,
    int kills_this_tick,
    int level
) {
    // Update kills window (each slot covers ~22 ticks for 175 tick window)
    int slot_idx = (tick / 22) % 8;
    if (slot_idx != state->window_idx) {
        // Moving to new slot, clear old count
        state->total_kills_window -= state->kills_window[slot_idx];
        state->kills_window[slot_idx] = 0;
        state->window_idx = slot_idx;
    }
    state->kills_window[slot_idx] += kills_this_tick;
    state->total_kills_window += kills_this_tick;

    // Track minimum health
    if (curr_health < state->min_health_recent && curr_health > 0) {
        state->min_health_recent = curr_health;
        state->min_health_tick = tick;
    }

    // Reset min health tracker every 350 ticks (10 seconds)
    if (tick > 0 && (tick % 350) == 0) {
        state->min_health_recent = curr_health;
    }

    // ===================
    // Detect Interest Events
    // ===================

    // 1. Near-death escape: was below 20 HP, now above 50 HP
    if (state->min_health_recent <= 20 && curr_health >= 50) {
        int score = 50 + (20 - state->min_health_recent) * 2;  // Lower min = more interesting
        if (score > 100) score = 100;

        RecordInterestEvent(
            state,
            IT_NEAR_DEATH, tick, curr_health, score, level,
            0, state->min_health_recent
        );

        // Reset min health tracker
        state->min_health_recent = curr_health;
    }

    // 2. Multi-kill: 3+ kills in window
    if (kills_this_tick > 0 && state->total_kills_window >= 3) {
        int score = 40 + state->total_kills_window * 15;  // More kills = more interesting
        if (score > 100) score = 100;

        RecordInterestEvent(
            state,
            IT_MULTI_KILL, tick, curr_health, score, level,
            state->total_kills_window, 0
        );

        // Don't double-count, reset window
        for (int i = 0; i < 8; i++) state->kills_window[i] = 0;
        state->total_kills_window = 0;
    }

    // 3. Death
    if (prev_health > 0 && curr_health <= 0) {
        RecordInterestEvent(
            state,
            IT_DEATH, tick, 0, 80, level,
            0, 0
        );
    }

    // 4. Low-health kill: got a kill while below 25 HP
    if (kills_this_tick > 0 && curr_health > 0 && curr_health <= 25) {
        int score = 60 + (25 - curr_health) * 2;
        if (score > 100) score = 100;

        RecordInterestEvent(
            state,
            IT_LOW_HEALTH_KILL, tick, curr_health, score, level,
            kills_this_tick, 0
        );
    }
}

// =============================================================================
// Host Functions
// =============================================================================
// Note: Host functions are implemented directly in doom_main.cu (allocation)
// and doom_export.cu (JSON export). No declarations needed here since
// we inline the allocation in InitArena and use static functions in export.

#endif // DOOM_INTEREST_CUH
