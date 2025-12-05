# Angband GPU Status Report

**Date:** 2025-12-05
**Device:** NVIDIA GeForce RTX 5090
**Status:** Working Prototype (V2 Kernel - Phase 3 Lethality Verified)

## Benchmark Results (Phase 3 + Realistic Stats)

| Metric | Value |
|:---|:---|
| **Throughput** | **79,405,343** instance-turns/sec |
| **Simulation Time** | 0.1259s (10,000 instances x 1,000 turns) |
| **Survival Rate** | **99.3%** (74 confirmed deaths) |

*Verification:* By lowering starting HP to 20 (realistic Level 1), we confirmed that monsters (including breeding worms) CAN kill the player. The 0.7% death rate over 1000 turns is consistent with cautious early-game play in open terrain. Survival rates will drop significantly once complex dungeon terrain (corridors) forces combat engagements.

## Architecture

The implementation uses a Data-Oriented Design (DOD) with Structure-of-Arrays (SoA) layout to maximize memory coalescing on the GPU.

- **Interleaved Memory:** `State[Instance][Property]` transformed to `State[Property][Instance]`.
- **Warp Coherency:** All instances execute the same phase (Think -> Move -> Combat) simultaneously.
- **Kernel Split:** Logic is split into `borg_think_kernel` and `borg_execute_kernel` to manage register pressure.

## Phase 3 Features (Implemented)

1.  **Real Combat Math:**
    *   To-hit calculations (60% base + bonus).
    *   Damage rolls (dice + sides).
    *   Critical hits (10% chance for x2 damage).
    *   AC reduction (percentage based).
2.  **Monster Breeding:**
    *   `try_breed_monster` kernel function.
    *   Worm Masses (ID 0) breed 10% chance/turn.
3.  **Stealth Mechanics:**
    *   `wake_nearby_monsters` uses distance + stealth to determine wake chance.

## Artifacts

- `prototypes/gpu_prototype.py`: Python/PyTorch proof-of-concept (2.5M SPS).
- `prototypes/ARCHITECTURE.md`: Detailed architectural analysis ("Ultrathink").
- `borg_kernel_v2.cu`: High-performance CUDA implementation (79M SPS with complex logic).

## Next Steps

1.  **Dungeon Generation:** Implement `generate_level` kernel (currently simplified).
2.  **Python Bindings:** Expose the CUDA kernel via PyBind11 or CFFI for training RL agents.
