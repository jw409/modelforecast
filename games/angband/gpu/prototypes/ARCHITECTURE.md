# 🧠 Ultrathink: Angband GPU Architecture

**Hypothesis**: Angband instances, typically CPU-bound and state-heavy, can be refactored into an "Embarrassingly Parallel" architecture for GPU execution to achieve massive throughput (10k+ instances) for reinforcement learning or genetic algorithm optimization.

**Constraint**: The current C codebase relies on pointer indirection (`struct square **`, linked lists for objects). This is incompatible with SIMT (Single Instruction, Multiple Threads) execution.

**Solution**: A Data-Oriented Design (DOD) transformation mapping Game State to Tensors.

---

## 1. Parallelism Strategy: "Mega-Batch" Simulation

Instead of one thread per instance (which stalls on branching), we use **Kernel-Level Parallelism** where each kernel performs *one specific sub-system update* for *all* instances simultaneously.

### The Phase Loop
The main loop is inverted. Instead of `for agent in agents: agent.update()`, we do:
```python
while True:
    PhysicsKernel<<<blocks, threads>>>(All_Grids, All_Players)
    LOSKernel<<<blocks, threads>>>(All_Grids, All_Players, Visibility_Masks)
    BorgThinkKernel<<<blocks, threads>>>(Visibility_Masks, State_History)
    MonsterMoveKernel<<<blocks, threads>>>(All_Monsters, All_Grids)
```

*   **Warp coherency**: All instances execute "Monster Moving" code at the same time, minimizing instruction cache misses and divergence.
*   **Masking**: Instances that don't need an update (e.g., player dead, waiting for input) are masked out via predicate registers.

## 2. Data Partitioning: Structure of Arrays (SoA)

We must abolish `struct`. GPU memory coalescing requires contiguous access to similar data.

### The Transformation
**CPU (Current AoS):**
```c
struct square { uint8 feat; struct object *obj; };
square *grid[1000][66][198]; // [Instance][Y][X]
// Access: grid[i][y][x].feat -> random memory jump
```

**GPU (Target SoA):**
```python
# Global Tensor: [Batch, Channels, Height, Width]
Global_Features = torch.zeros((10000, 16, 66, 198), dtype=uint8, device='cuda')
# Channel 0: Terrain ID
# Channel 1: Lighting Level
# Channel 2: Resident Monster Index
# Channel 3: Top Object Index
```

*   **Benefit**: Reading "Terrain ID" for 32 neighbors (warp) is a single coalesced 128-byte memory transaction.
*   **Partitioning**: Split data by *frequency of access*.
    *   **Hot Data** (Position, HP, Energy): Keep in registers/L1.
    *   **Cold Data** (Inventory Item #20 description): Keep in global memory or demand-paged.

## 3. Kernel Optimization: The "Micro-Band"

We break Angband's monolithic `dungeon.c` into micro-kernels.

*   **LOS (Line of Sight)**:
    *   *Current*: Raycasting with `cave_info` bitflags.
    *   *GPU*: Parallel Ray-Marching or Breadth-First Search (flood fill) on the `Global_Features` tensor.
    *   *Optimization*: Use shared memory to cache the 10x10 local grid around the player for the entire block.

*   **RNG (Random Number Generation)**:
    *   *Current*: Global `rand_int()`.
    *   *GPU*: `curand` state per instance.
    *   *Optimization*: Pre-generate buffers of random numbers (Consumption Kernel) to avoid RNG logic inside physics kernels.

## 4. Memory Management: The "ECS" Approach

Handling dynamic lists (Monsters, Inventory) is the hardest part of GPU porting. We use a fixed-capacity **Entity Component System**.

### The Monster Tensor
Instead of `m_list` linked list:
```python
# [Batch, Max_Monsters_Per_Level, Attributes]
Monster_Pool = torch.zeros((10000, 64, 12), device='cuda')
# Attributes: [Active_Flag, Race_ID, X, Y, HP, Energy, Target_Y, Target_X, ...]
```

*   **Compaction**: When a monster dies, we don't free memory. We flip `Active_Flag = 0`.
*   **Periodic Compact**: Every 100 turns, a `scan` primitive (prefix sum) compacts the active monsters to the left to maintain warp efficiency.

### Inventory (Sparse vs Dense)
*   **Dense**: `Inventory[Batch, 23, Attr]` (Wasteful, most slots empty).
*   **Sparse**: Hash map in shared memory (Complex).
*   **Hybrid**: First 5 slots (Weapon, Armor, Shield) are dense registers. Backpack is a slower lookup in global memory.

## 5. Performance Metrics & Observability

To tune this ultrathink architecture, we track:

*   **SIMD Efficiency**: % of active threads in a warp. If 1 instance triggers "Earthquake" logic and 31 don't, we lose 97% performance.
    *   *Fix*: Sorting instances by state (e.g., "Level Generation Phase" batch vs "Gameplay Phase" batch).
*   **Memory Throughput**: DRAM vs L2 throughput.
*   **Steps Per Second (SPS)**:
    *   CPU Swarm: ~100k SPS (across all cores).
    *   GPU Target: ~100M SPS (assuming 10k instances x 10kHz kernel).

---

## Implementation Prototype

See `talent-os/projects/angband/gpu_prototype.py` for a PyTorch proof-of-concept demonstrating the tensor mapping.
