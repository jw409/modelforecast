# GPU MARS Benchmark Results

Target: NVIDIA RTX 5090 (21,760 Cores)
Task: Core War Simulation (Imp vs Dwarf)
Battles: 300,000 simultaneous battles
Max Cycles: 80,000 per battle
Total Instructions Executed: ~48 Billion

## Implementations Tested

1. **Naive AoS (Array of Structures)**
   - Layout: `BattleState battles[NUM_BATTLES]`
   - Inner: `Instruction core[CORE_SIZE]`
   - Access: `battles[id].core[pc]`
   - Pros: Simple logic, direct 16-bit writes.
   - Cons: Uncoalesced memory access (stride ~96KB).

2. **Full SoA (Structure of Arrays)**
   - Layout: `uint8_t opcodes[TOTAL]`, `int16_t a_fields[TOTAL]`...
   - Access: `opcodes[pc * NUM + id]`
   - Pros: Perfectly coalesced.
   - Cons: 6 separate memory transactions to read one instruction. High latency.

3. **Packed SoA**
   - Layout: `uint64_t core[TOTAL]` (Manual bit-packing)
   - Access: `core[pc * NUM + id]`
   - Pros: Coalesced, single fetch.
   - Cons: Partial writes (B-field update) require Read-Modify-Write (atomic overhead or just extra instructions).

4. **Interleaved (Transposed AoS) - WINNER**
   - Layout: `Instruction core[TOTAL]` (Structs)
   - Logical: `core[pc][battle_id]`
   - Physical: `core[pc * NUM + battle_id]`
   - Pros: Coalesced memory access. Direct field access (16-bit writes). Single instruction fetch.

## Results

| Implementation | Time (300k battles) | Throughput (battles/sec) | Speedup |
|----------------|---------------------|--------------------------|---------|
| Naive AoS      | ~18.2 s (est)       | ~16,400                  | 1.0x    |
| Full SoA       | 78.4 s (est)        | ~3,800                   | 0.23x   |
| Packed SoA     | 68.4 s              | ~4,384                   | 0.26x   |
| **Interleaved**| **10.77 s**         | **27,845**               | **1.70x**|

## Conclusion

The **Interleaved** layout is superior for "embarrassingly parallel" simulations where state per thread is large (64KB). It balances memory coalescence with efficient instruction execution.

Achieved **~4.45 Billion Instructions Per Second** (effective) on RTX 5090.
Utilization is latency-bound due to dependent memory accesses in the simulation loop.
Scaling to 1 Million battles (requires 64GB VRAM) would likely improve throughput further by hiding latency.
Current 5090 (32GB VRAM) limits us to ~400,000 battles.
