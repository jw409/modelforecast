# Angband GPU Status Report

**Date:** 2025-12-05
**Device:** NVIDIA GeForce RTX 5090
**Status:** Working Prototype (V2 Kernel)

## Benchmark Results

| Metric | Value |
|:---|:---|
| **Throughput** | **141,302,248** instance-turns/sec |
| **Simulation Time** | 0.0708s (10,000 instances x 1,000 turns) |
| **Survival Rate** | 100% (Depth ~12, Level 1) |

## Architecture

The implementation uses a Data-Oriented Design (DOD) with Structure-of-Arrays (SoA) layout to maximize memory coalescing on the GPU.

- **Interleaved Memory:** `State[Instance][Property]` transformed to `State[Property][Instance]`.
- **Warp Coherency:** All instances execute the same phase (Think -> Move -> Combat) simultaneously.
- **Kernel Split:** Logic is split into `borg_think_kernel` and `borg_execute_kernel` to manage register pressure.

## Artifacts

- `prototypes/gpu_prototype.py`: Python/PyTorch proof-of-concept (2.5M SPS).
- `prototypes/ARCHITECTURE.md`: Detailed architectural analysis ("Ultrathink").
- `borg_kernel_v2.cu`: High-performance CUDA implementation (141M SPS).

## Next Steps

1.  **Real Combat Math:** Port full `calc_hit` and `calc_damage` from C to CUDA (headers exist but need verification).
2.  **Dungeon Generation:** Implement `generate_level` kernel (currently simplified).
3.  **Python Bindings:** Expose the CUDA kernel via PyBind11 or CFFI for training RL agents.
