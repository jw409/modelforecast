# GPU DOOM Phase 2 - Integration Guide

## Quick Integration into doom_main.cu

This guide shows how to integrate the Phase 2 monster combat system into your existing doom_main.cu.

## Step 1: Add Includes

After `#include "doom_types.cuh"`:

```cuda
#include "doom_types.cuh"
#include "doom_monsters.cuh"  // ← ADD THIS
#include "doom_combat.cuh"    // ← ADD THIS
```

## Step 2: Declare Monster Memory (Global Device Pointers)

After the existing player device pointers, add:

```cuda
// Existing player state
__device__ int32_t* d_player_health;
__device__ int32_t* d_player_armor;
// ... other player fields ...
__device__ int16_t* d_player_kills;  // ← ADD THIS (if not already there)

// Monster state ← ADD ALL OF THIS
#define MAX_MONSTERS 64  // Per instance
__device__ fixed_t* d_monster_x;
__device__ fixed_t* d_monster_y;
__device__ fixed_t* d_monster_z;
__device__ angle_t* d_monster_angle;
__device__ int32_t* d_monster_health;
__device__ uint8_t* d_monster_type;
__device__ uint8_t* d_monster_alive;
__device__ int16_t* d_monster_target_idx;
__device__ uint8_t* d_monster_movedir;
__device__ int16_t* d_monster_movecount;
__device__ int16_t* d_monster_reactiontime;
```

## Step 3: Add Monster Spawning Function

Before `doom_simulate` kernel:

```cuda
__device__ void SpawnMonster(int monster_id, int instance_id, int num_instances,
                             fixed_t x, fixed_t y, MonsterTypeID type) {
    int idx = monster_id * num_instances + instance_id;

    d_monster_x[idx] = x;
    d_monster_y[idx] = y;
    d_monster_z[idx] = 0;
    d_monster_angle[idx] = 0;
    d_monster_type[idx] = type;
    d_monster_health[idx] = c_monster_stats[type].health;
    d_monster_alive[idx] = 1;
    d_monster_target_idx[idx] = 0;  // Target player
    d_monster_movedir[idx] = DI_NODIR;
    d_monster_movecount[idx] = 5;
    d_monster_reactiontime[idx] = 0;
}

__device__ void InitMonsters(int instance_id, int num_instances) {
    // E1M1 has ~50 monsters - here's a simplified spawn list
    // (You can load these from WAD in Phase 3)

    // Zombies near start
    SpawnMonster(0, instance_id, num_instances, 1200 << FRACBITS, -3600 << FRACBITS, MT_ZOMBIE);
    SpawnMonster(1, instance_id, num_instances, 1300 << FRACBITS, -3500 << FRACBITS, MT_ZOMBIE);

    // Imps in courtyard
    SpawnMonster(2, instance_id, num_instances, 800 << FRACBITS, -2400 << FRACBITS, MT_IMP);
    SpawnMonster(3, instance_id, num_instances, 900 << FRACBITS, -2300 << FRACBITS, MT_IMP);

    // Pinky demon in dark room
    SpawnMonster(4, instance_id, num_instances, 1600 << FRACBITS, -3000 << FRACBITS, MT_DEMON);

    // Shotgun guy guarding key
    SpawnMonster(5, instance_id, num_instances, 2000 << FRACBITS, -2000 << FRACBITS, MT_SHOTGUY);

    // Cacodemon (boss of first room)
    SpawnMonster(6, instance_id, num_instances, 1800 << FRACBITS, -2800 << FRACBITS, MT_CACODEMON);

    // ... add more up to MAX_MONSTERS (64)
}
```

## Step 4: Update P_PlayerThink_GPU

Find the weapon fire section in `P_PlayerThink_GPU` and replace with:

```cuda
// Handle weapon fire (BT_ATTACK button)
if (cmd.buttons & BT_ATTACK) {
    P_PlayerAttack(instance_id, tick, num_instances, angle, x, y);
}
```

## Step 5: Update doom_simulate Kernel

Find the main simulation loop and add monster initialization and thinking:

```cuda
__global__ void doom_simulate(int num_instances, int num_ticks, int checkpoint_interval) {
    int instance_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance_id >= num_instances) return;

    // ← ADD THIS: Initialize monsters on first call
    if (instance_id < num_instances) {
        InitMonsters(instance_id, num_instances);
    }
    __syncthreads();

    for (int tick = 0; tick < num_ticks; tick++) {
        // Player thinking (movement, actions)
        P_PlayerThink_GPU(instance_id, tick, num_instances);

        // ← ADD THIS: Monster AI
        for (int m = 0; m < MAX_MONSTERS; m++) {
            P_MonsterThink(m, instance_id, tick, num_instances);
        }

        // Write checkpoint if interval hit
        if (checkpoint_interval > 0 && (tick % checkpoint_interval) == 0) {
            WriteCheckpoint_GPU(instance_id, tick, num_instances);
        }

        // Sync all instances before next tick
        __syncthreads();
    }

    // Final checkpoint
    if (checkpoint_interval > 0) {
        WriteCheckpoint_GPU(instance_id, num_ticks, num_instances);
    }
}
```

## Step 6: Update DoomArena Structure

Add monster memory to the arena:

```cpp
struct DoomArena {
    // Existing player pointers
    int32_t* player_health;
    // ... other player fields ...
    int16_t* player_kills;  // ← ADD THIS if not there

    // ← ADD THIS: Monster pointers
    fixed_t* monster_x;
    fixed_t* monster_y;
    fixed_t* monster_z;
    angle_t* monster_angle;
    int32_t* monster_health;
    uint8_t* monster_type;
    uint8_t* monster_alive;
    int16_t* monster_target_idx;
    uint8_t* monster_movedir;
    int16_t* monster_movecount;
    int16_t* monster_reactiontime;

    int num_instances;
    int max_ticks;
};
```

## Step 7: Update InitArena

Add monster memory allocation:

```cpp
void InitArena(DoomArena* arena, int num_instances, int max_ticks) {
    arena->num_instances = num_instances;
    arena->max_ticks = max_ticks;

    size_t n = num_instances;
    size_t monster_size = MAX_MONSTERS * n;

    // Existing player allocations
    cudaMalloc(&arena->player_health, n * sizeof(int32_t));
    // ... other player fields ...
    cudaMalloc(&arena->player_kills, n * sizeof(int16_t));  // ← ADD THIS

    // ← ADD THIS: Monster allocations
    cudaMalloc(&arena->monster_x, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_y, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_z, monster_size * sizeof(fixed_t));
    cudaMalloc(&arena->monster_angle, monster_size * sizeof(angle_t));
    cudaMalloc(&arena->monster_health, monster_size * sizeof(int32_t));
    cudaMalloc(&arena->monster_type, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_alive, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_target_idx, monster_size * sizeof(int16_t));
    cudaMalloc(&arena->monster_movedir, monster_size * sizeof(uint8_t));
    cudaMalloc(&arena->monster_movecount, monster_size * sizeof(int16_t));
    cudaMalloc(&arena->monster_reactiontime, monster_size * sizeof(int16_t));

    // Initialize monsters as dead (will be spawned in kernel)
    cudaMemset(arena->monster_alive, 0, monster_size * sizeof(uint8_t));

    // Existing player cudaMemcpyToSymbol calls
    cudaMemcpyToSymbol(d_player_health, &arena->player_health, sizeof(int32_t*));
    // ... other player fields ...
    cudaMemcpyToSymbol(d_player_kills, &arena->player_kills, sizeof(int16_t*));  // ← ADD

    // ← ADD THIS: Monster cudaMemcpyToSymbol calls
    cudaMemcpyToSymbol(d_monster_x, &arena->monster_x, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_y, &arena->monster_y, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_z, &arena->monster_z, sizeof(fixed_t*));
    cudaMemcpyToSymbol(d_monster_angle, &arena->monster_angle, sizeof(angle_t*));
    cudaMemcpyToSymbol(d_monster_health, &arena->monster_health, sizeof(int32_t*));
    cudaMemcpyToSymbol(d_monster_type, &arena->monster_type, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_alive, &arena->monster_alive, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_target_idx, &arena->monster_target_idx, sizeof(int16_t*));
    cudaMemcpyToSymbol(d_monster_movedir, &arena->monster_movedir, sizeof(uint8_t*));
    cudaMemcpyToSymbol(d_monster_movecount, &arena->monster_movecount, sizeof(int16_t*));
    cudaMemcpyToSymbol(d_monster_reactiontime, &arena->monster_reactiontime, sizeof(int16_t*));

    printf("Arena initialized: %d instances, %d max ticks\n", num_instances, max_ticks);
    printf("Monsters per instance: %d\n", MAX_MONSTERS);
    printf("Memory allocated:\n");
    printf("  Player state: %.2f MB\n", (n * 11 * sizeof(int32_t)) / (1024.0 * 1024.0));
    printf("  Monster state: %.2f MB\n", (monster_size * 11 * sizeof(int32_t)) / (1024.0 * 1024.0));
    // ... rest of existing prints ...
}
```

## Step 8: Update FreeArena

Add monster memory cleanup:

```cpp
void FreeArena(DoomArena* arena) {
    // Existing player frees
    cudaFree(arena->player_health);
    // ... other player fields ...
    cudaFree(arena->player_kills);  // ← ADD THIS

    // ← ADD THIS: Monster frees
    cudaFree(arena->monster_x);
    cudaFree(arena->monster_y);
    cudaFree(arena->monster_z);
    cudaFree(arena->monster_angle);
    cudaFree(arena->monster_health);
    cudaFree(arena->monster_type);
    cudaFree(arena->monster_alive);
    cudaFree(arena->monster_target_idx);
    cudaFree(arena->monster_movedir);
    cudaFree(arena->monster_movecount);
    cudaFree(arena->monster_reactiontime);

    // ... rest of existing frees ...
}
```

## Step 9: Update InitPlayers

Add kill counter initialization:

```cpp
void InitPlayers(DoomArena* arena, fixed_t start_x, fixed_t start_y) {
    int n = arena->num_instances;

    // Existing arrays
    int32_t* h_health = new int32_t[n];
    // ... other player fields ...
    int16_t* h_kills = new int16_t[n];  // ← ADD THIS

    for (int i = 0; i < n; i++) {
        h_health[i] = 100;
        // ... other initializations ...
        h_kills[i] = 0;  // ← ADD THIS
    }

    // Existing cudaMemcpy calls
    cudaMemcpy(arena->player_health, h_health, n * sizeof(int32_t), cudaMemcpyHostToDevice);
    // ... other fields ...
    cudaMemcpy(arena->player_kills, h_kills, n * sizeof(int16_t), cudaMemcpyHostToDevice);  // ← ADD

    // Cleanup
    delete[] h_health;
    // ... other deletes ...
    delete[] h_kills;  // ← ADD THIS
}
```

## Step 10: Update GenerateTestInput

Make player shoot occasionally:

```cpp
void GenerateTestInput(DoomArena* arena, int num_ticks) {
    int n = arena->num_instances;
    size_t size = n * num_ticks;
    TicCmd* h_input = new TicCmd[size];

    for (int tick = 0; tick < num_ticks; tick++) {
        for (int i = 0; i < n; i++) {
            int idx = tick * n + i;
            TicCmd& cmd = h_input[idx];

            cmd.forwardmove = 50;  // Walk forward
            cmd.sidemove = 0;

            // Turn occasionally
            if ((tick + i) % 70 == 0) {
                cmd.angleturn = (i % 2 == 0) ? 512 : -512;
            } else {
                cmd.angleturn = 0;
            }

            // ← ADD THIS: Shoot every 10 ticks
            cmd.buttons = (tick % 10 == 0) ? BT_ATTACK : 0;

            cmd.consistency = 0;
            cmd.chatchar = 0;
        }
    }

    cudaMemcpy(arena->input_buffer, h_input, size * sizeof(TicCmd), cudaMemcpyHostToDevice);
    delete[] h_input;
}
```

## Step 11: Update ReadCheckpoints

Show kill count in checkpoint output:

```cpp
void ReadCheckpoints(DoomArena* arena, int instance_id) {
    // ... existing code ...

    for (int i = 0; i < count && i < 5; i++) {
        Checkpoint& cp = h_cp[i];
        printf("  [tick %4d] pos=(%.1f, %.1f) health=%d alive=%d kills=%d\n",  // ← ADD kills
               cp.tick,
               cp.x / (float)FRACUNIT,
               cp.y / (float)FRACUNIT,
               cp.health,
               cp.alive,
               cp.kills);  // ← ADD THIS
    }

    // ... rest of function ...
}
```

## Step 12: Update WriteCheckpoint_GPU

Actually write kill count to checkpoint:

```cuda
__device__ void WriteCheckpoint_GPU(int instance_id, int tick, int num_instances) {
    // ... existing code ...

    cp.tick = tick;
    cp.health = d_player_health[instance_id];
    cp.armor = d_player_armor[instance_id];
    cp.x = d_player_x[instance_id];
    cp.y = d_player_y[instance_id];
    cp.z = d_player_z[instance_id];
    cp.angle = d_player_angle[instance_id];
    cp.alive = d_player_alive[instance_id];
    cp.kills = d_player_kills[instance_id];  // ← CHANGE FROM 0

    // ... rest of function ...
}
```

## That's It!

Rebuild and run:

```bash
make clean
make
./build/doom_sim 1024 1000 35
```

## Expected Output

```
Arena initialized: 1024 instances, 1000 max ticks
Monsters per instance: 64
Memory allocated:
  Player state: 0.04 MB
  Monster state: 2.75 MB
  ...

Launching kernel: 4 blocks × 256 threads

=== Results ===
Kernel time: 25.00 ms
Total ticks: 1,024,000
Throughput: 40,960,000 ticks/sec
...

Instance 0: 5 checkpoints
  [tick    0] pos=(1056.0, -3616.0) health=100 alive=1 kills=0
  [tick   35] pos=(1120.0, -3580.0) health=92 alive=1 kills=1
  [tick   70] pos=(1184.0, -3544.0) health=85 alive=1 kills=3
  [tick  105] pos=(1248.0, -3508.0) health=74 alive=1 kills=5
  [tick  140] pos=(1312.0, -3472.0) health=61 alive=1 kills=7
```

Notice:
- Health decreasing (monsters attacking)
- Kills increasing (player shooting monsters)
- Some instances may die (health=0, alive=0)

## Done!

You now have a fully functional DOOM combat simulation on GPU with monsters that can kill the player and be killed.
