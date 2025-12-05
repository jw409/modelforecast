/**
 * GPU DOOM - Combat System
 *
 * Monster AI, player attacks, and damage handling
 *
 * Original: id Software DOOM (1993)
 * GPU Port: MIT License
 */

#ifndef DOOM_COMBAT_CUH
#define DOOM_COMBAT_CUH

#include "doom_types.cuh"
#include "doom_monsters.cuh"

// External device memory pointers (defined in doom_main.cu)
extern __device__ fixed_t* d_monster_x;
extern __device__ fixed_t* d_monster_y;
extern __device__ fixed_t* d_monster_z;
extern __device__ angle_t* d_monster_angle;
extern __device__ int32_t* d_monster_health;
extern __device__ uint8_t* d_monster_type;
extern __device__ uint8_t* d_monster_alive;
extern __device__ int16_t* d_monster_target_idx;
extern __device__ uint8_t* d_monster_movedir;
extern __device__ int16_t* d_monster_movecount;
extern __device__ int16_t* d_monster_reactiontime;

extern __device__ int32_t* d_player_health;
extern __device__ int32_t* d_player_armor;
extern __device__ fixed_t* d_player_x;
extern __device__ fixed_t* d_player_y;
extern __device__ fixed_t* d_player_z;
extern __device__ angle_t* d_player_angle;
extern __device__ uint8_t* d_player_alive;
extern __device__ int16_t* d_player_kills;

// =============================================================================
// Random Number Generator (simple LCG for GPU)
// =============================================================================

__device__ inline uint32_t gpu_random(int instance_id, int tick, int seed = 0) {
    uint32_t s = instance_id * 1103515245 + tick * 12345 + seed * 67890;
    return (s / 65536) % 256;
}

// =============================================================================
// Combat Helper Functions
// =============================================================================

// Approximate distance (Manhattan distance scaled for diagonal)
__device__ inline fixed_t P_AproxDistance(fixed_t dx, fixed_t dy) {
    dx = abs(dx);
    dy = abs(dy);
    if (dx < dy)
        return dx + dy - (dx >> 1);
    return dx + dy - (dy >> 1);
}

// Simplified line of sight check (Phase 2: no BSP)
__device__ inline bool P_CheckSight_Simple(
    fixed_t mon_x, fixed_t mon_y,
    fixed_t target_x, fixed_t target_y)
{
    fixed_t dx = target_x - mon_x;
    fixed_t dy = target_y - mon_y;
    fixed_t dist = P_AproxDistance(dx, dy);
    return dist < MISSILERANGE * 2;
}

// Damage player
__device__ inline void P_DamagePlayer(int instance_id, int damage) {
    int health = d_player_health[instance_id];
    int armor = d_player_armor[instance_id];

    // Armor absorbs 1/3 of damage
    if (armor > 0) {
        int absorbed = damage / 3;
        armor -= absorbed;
        damage -= absorbed;
        if (armor < 0) {
            damage += armor;  // Overflow damage
            armor = 0;
        }
        d_player_armor[instance_id] = armor;
    }

    health -= damage;
    if (health <= 0) {
        health = 0;
        d_player_alive[instance_id] = 0;
    }
    d_player_health[instance_id] = health;
}

// Damage monster
__device__ inline void P_DamageMonster(int monster_idx, int instance_id, int num_instances, int damage) {
    int idx = monster_idx * num_instances + instance_id;
    int health = d_monster_health[idx];
    health -= damage;

    if (health <= 0) {
        health = 0;
        d_monster_alive[idx] = 0;
        // Player gets kill credit
        atomicAdd(&d_player_kills[instance_id], 1);
    }

    d_monster_health[idx] = health;
}

// =============================================================================
// Monster AI - Simplified A_Chase
// =============================================================================

__device__ void P_MonsterThink(int monster_id, int instance_id, int tick, int num_instances) {
    int idx = monster_id * num_instances + instance_id;

    // Skip if dead
    if (!d_monster_alive[idx]) return;

    // Get monster state
    fixed_t mx = d_monster_x[idx];
    fixed_t my = d_monster_y[idx];
    uint8_t mtype = d_monster_type[idx];
    int16_t target = d_monster_target_idx[idx];
    int16_t movecount = d_monster_movecount[idx];
    int16_t reactiontime = d_monster_reactiontime[idx];

    const MonsterStats& stats = c_monster_stats[mtype];

    // Decrement reaction time (attack cooldown)
    if (reactiontime > 0) {
        d_monster_reactiontime[idx] = reactiontime - 1;
        return;
    }

    // Get player position
    fixed_t px = d_player_x[instance_id];
    fixed_t py = d_player_y[instance_id];
    uint8_t player_alive = d_player_alive[instance_id];

    // If no target or player dead, go dormant
    if (target < 0 || !player_alive) {
        d_monster_target_idx[idx] = -1;
        return;
    }

    // Check line of sight
    if (!P_CheckSight_Simple(mx, my, px, py)) {
        d_monster_target_idx[idx] = -1;  // Lost sight
        return;
    }

    // Calculate distance to player
    fixed_t dx = px - mx;
    fixed_t dy = py - my;
    fixed_t dist = P_AproxDistance(dx, dy);

    // Check for melee attack
    if (stats.melee_range > 0 && dist < stats.melee_range) {
        int damage_range = stats.melee_damage_max - stats.melee_damage_min + 1;
        int damage = stats.melee_damage_min + (gpu_random(instance_id, tick, monster_id) % damage_range);
        P_DamagePlayer(instance_id, damage);
        d_monster_reactiontime[idx] = 8;  // Cooldown
        return;
    }

    // Check for ranged attack
    if (stats.missile_range > 0 && dist < stats.missile_range && dist > stats.melee_range) {
        // 25% chance to attack each tick when in range
        if (gpu_random(instance_id, tick, monster_id) % 4 == 0) {
            int damage_range = stats.ranged_damage_max - stats.ranged_damage_min + 1;
            int damage = stats.ranged_damage_min +
                         (gpu_random(instance_id, tick, monster_id * 2) % damage_range);
            P_DamagePlayer(instance_id, damage);
            d_monster_reactiontime[idx] = 16;  // Longer cooldown for ranged
            return;
        }
    }

    // Move toward player (simplified - no collision detection)
    if (--movecount <= 0) {
        fixed_t move_dist = stats.speed;

        // Normalize direction and move
        if (dist > 0) {
            mx += FixedMul(dx, FixedDiv(move_dist, dist));
            my += FixedMul(dy, FixedDiv(move_dist, dist));
        }

        // Write back position
        d_monster_x[idx] = mx;
        d_monster_y[idx] = my;

        // Reset move counter (random 5-10 ticks)
        d_monster_movecount[idx] = 5 + (gpu_random(instance_id, tick, monster_id * 3) % 6);
    } else {
        d_monster_movecount[idx] = movecount;
    }
}

// =============================================================================
// Player Attack - Hitscan weapon
// =============================================================================

__device__ void P_PlayerAttack(int instance_id, int tick, int num_instances, angle_t player_angle,
                               fixed_t player_x, fixed_t player_y) {
    // Find nearest monster in front of player (within aiming cone)
    int nearest_monster = -1;
    fixed_t nearest_dist = INT32_MAX;

    int fineangle_player = player_angle >> ANGLETOFINESHIFT;
    fixed_t forward_x = finecosine(fineangle_player);
    fixed_t forward_y = finesine(fineangle_player);

    for (int m = 0; m < MAX_MONSTERS; m++) {
        int midx = m * num_instances + instance_id;
        if (!d_monster_alive[midx]) continue;

        fixed_t mx = d_monster_x[midx];
        fixed_t my = d_monster_y[midx];

        fixed_t dx = mx - player_x;
        fixed_t dy = my - player_y;

        // Dot product to check if in front
        int64_t dot = ((int64_t)dx * forward_x + (int64_t)dy * forward_y) >> FRACBITS;

        if (dot > 0) {  // In front of player
            fixed_t dist = P_AproxDistance(dx, dy);
            if (dist < nearest_dist && dist < MISSILERANGE) {
                nearest_dist = dist;
                nearest_monster = m;
            }
        }
    }

    // Hit the nearest monster
    if (nearest_monster >= 0) {
        // Pistol damage: 5-15 (random 1-3 * 5)
        int damage = ((gpu_random(instance_id, tick) % 3) + 1) * 5;
        P_DamageMonster(nearest_monster, instance_id, num_instances, damage);
    }
}

#endif // DOOM_COMBAT_CUH
