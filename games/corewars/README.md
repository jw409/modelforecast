# GPU MARS (Core War on CUDA)

This project implements a GPU-accelerated Core War simulator (MARS) using CUDA.
It demonstrates massive parallelism by running hundreds of thousands of battles simultaneously on a single GPU.

## Requirements
- NVIDIA GPU (RTX 3060 or better recommended)
- CUDA Toolkit 11+
- Linux (tested on Ubuntu with RTX 5090)

## Building

```bash
make
```

## Running

Runs 3 variants of the kernel for benchmarking.

```bash
# Run the optimized version (Interleaved Layout)
./build/gpu_mars_interleaved <num_battles>

# Example (requires ~19GB VRAM)
./build/gpu_mars_interleaved 300000
```

## Implementations

1. `gpu_mars` (Naive AoS): Simplest code, decent performance (16k battles/sec).
2. `gpu_mars_soa` (Full SoA): Slowest due to memory transaction overhead.
3. `gpu_mars_packed` (Packed SoA): Slow due to read-modify-write on updates.
4. `gpu_mars_interleaved` (Transposed AoS): **Fastest (27k battles/sec)**. Coalesced access + direct writes.

## Performance (RTX 5090)
- Throughput: ~28,000 battles/sec (80,000 cycles each)
- IPS: ~4.5 Billion Instructions/sec
- VRAM Usage: ~64KB per battle. 300k battles = ~19GB.
