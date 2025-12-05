#include "mars.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// AoS Optimized:
// - Uses BattleState layout (good for cache locality of instructions)
// - Uses Register Queues (reduces memory traffic)

__device__ int16_t resolve_addr_aos(BattleState* state, uint16_t pc, int16_t val, uint8_t mode) {
    if (mode == IMMEDIATE) return 0;
    
    int eff_addr;
    if (mode == DIRECT) {
        eff_addr = (int)pc + (int)val;
        eff_addr %= CORE_SIZE;
        if (eff_addr < 0) eff_addr += CORE_SIZE;
        return (uint16_t)eff_addr;
    }
    if (mode == INDIRECT_B) {
        int target = (int)pc + (int)val;
        target %= CORE_SIZE;
        if (target < 0) target += CORE_SIZE;
        
        // AoS Read
        int16_t ptr = state->core[target].b_field;
        
        eff_addr = target + (int)ptr;
        eff_addr %= CORE_SIZE;
        if (eff_addr < 0) eff_addr += CORE_SIZE;
        return (uint16_t)eff_addr;
    }
    
    // Fallback
    eff_addr = (int)pc + (int)val;
    eff_addr %= CORE_SIZE;
    if (eff_addr < 0) eff_addr += CORE_SIZE;
    return (uint16_t)eff_addr;
}

__global__ void mars_aos_optimized_kernel(
    BattleState* battles,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    uint16_t len_a, uint16_t len_b,
    const uint32_t* start_pos_a,
    const uint32_t* start_pos_b,
    uint32_t num_battles
) {
    int battle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (battle_id >= num_battles) return;

    BattleState* state = &battles[battle_id];

    // --- INITIALIZATION ---
    // Note: AoS init is sequential per thread. Good cache locality!
    Instruction dat = {DAT, F, IMMEDIATE, IMMEDIATE, 0, 0};
    for (int i = 0; i < CORE_SIZE; i++) {
        state->core[i] = dat;
    }

    uint32_t pa = start_pos_a[battle_id];
    for (int i = 0; i < len_a; i++) {
        state->core[(pa + i)%CORE_SIZE] = warriors_a[i];
    }
    uint32_t pb = start_pos_b[battle_id];
    for (int i = 0; i < len_b; i++) {
        state->core[(pb + i)%CORE_SIZE] = warriors_b[i];
    }

    // --- REGISTER PROMOTION ---
    uint16_t qa_head = 0;
    uint16_t qa_tail = 0;
    uint16_t qa_count = 0;
    
    uint16_t qb_head = 0;
    uint16_t qb_tail = 0;
    uint16_t qb_count = 0;

    // Use Global Memory for Queue Data, but pointers are local
    // Access: state->queue_a.pcs[...]
    
    if (qa_count < MAX_PROCESSES) {
        state->queue_a.pcs[qa_tail] = pa;
        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
        qa_count++;
    }
    if (qb_count < MAX_PROCESSES) {
        state->queue_b.pcs[qb_tail] = pb;
        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
        qb_count++;
    }

    // --- SIMULATION ---
    uint8_t winner = 0;
    int cycles = 0;
    
    while (cycles < MAX_CYCLES) {
        // Warrior A
        if (qa_count > 0) {
            uint16_t pc = state->queue_a.pcs[qa_head];
            qa_head = (qa_head + 1) % MAX_PROCESSES;
            qa_count--;

            Instruction instr = state->core[pc];
            
            uint16_t next_pc = (pc + 1) % CORE_SIZE;
            bool advance_pc = true;

            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    int16_t addr_b = resolve_addr_aos(state, pc, instr.b_field, instr.mode_b);
                    state->core[addr_b] = state->core[addr_a];
                    break;
                }
                case ADD: {
                     int16_t addr_b = resolve_addr_aos(state, pc, instr.b_field, instr.mode_b);
                     if (instr.mode_a == IMMEDIATE) {
                         state->core[addr_b].b_field += instr.a_field;
                     } else {
                         int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                         state->core[addr_b].b_field += state->core[addr_a].b_field;
                     }
                     break;
                }
                case JMP: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    next_pc = addr_a;
                    advance_pc = false;
                    if (qa_count < MAX_PROCESSES) {
                        state->queue_a.pcs[qa_tail] = next_pc;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    break;
                }
                case SPL: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    if (qa_count < MAX_PROCESSES) {
                        state->queue_a.pcs[qa_tail] = next_pc;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    if (qa_count < MAX_PROCESSES) {
                        state->queue_a.pcs[qa_tail] = addr_a;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    advance_pc = false;
                    break;
                }
                default:
                    break;
            }
            
            if (advance_pc) {
                if (qa_count < MAX_PROCESSES) {
                    state->queue_a.pcs[qa_tail] = next_pc;
                    qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                    qa_count++;
                }
            }
        } else {
            winner = 2;
            break;
        }

        // Warrior B
        if (qb_count > 0) {
            uint16_t pc = state->queue_b.pcs[qb_head];
            qb_head = (qb_head + 1) % MAX_PROCESSES;
            qb_count--;

            Instruction instr = state->core[pc];
            
            uint16_t next_pc = (pc + 1) % CORE_SIZE;
            bool advance_pc = true;

            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    int16_t addr_b = resolve_addr_aos(state, pc, instr.b_field, instr.mode_b);
                    state->core[addr_b] = state->core[addr_a];
                    break;
                }
                case ADD: {
                     int16_t addr_b = resolve_addr_aos(state, pc, instr.b_field, instr.mode_b);
                     if (instr.mode_a == IMMEDIATE) {
                         state->core[addr_b].b_field += instr.a_field;
                     } else {
                         int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                         state->core[addr_b].b_field += state->core[addr_a].b_field;
                     }
                     break;
                }
                case JMP: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    next_pc = addr_a;
                    advance_pc = false;
                    if (qb_count < MAX_PROCESSES) {
                        state->queue_b.pcs[qb_tail] = next_pc;
                        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                        qb_count++;
                    }
                    break;
                }
                case SPL: {
                    int16_t addr_a = resolve_addr_aos(state, pc, instr.a_field, instr.mode_a);
                    if (qb_count < MAX_PROCESSES) {
                        state->queue_b.pcs[qb_tail] = next_pc;
                        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                        qb_count++;
                    }
                    if (qb_count < MAX_PROCESSES) {
                        state->queue_b.pcs[qb_tail] = addr_a;
                        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                        qb_count++;
                    }
                    advance_pc = false;
                    break;
                }
                default:
                    break;
            }
            
            if (advance_pc) {
                if (qb_count < MAX_PROCESSES) {
                    state->queue_b.pcs[qb_tail] = next_pc;
                    qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                    qb_count++;
                }
            }
        } else {
            winner = 1;
            break;
        }

        cycles++;
    }

    state->winner = winner;
    state->cycles = cycles;
}

#define CHECK(call) \
do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    uint32_t num_battles = 100000;
    if (argc > 1) num_battles = atoi(argv[1]);
    
    printf("Running AoS Optimized GPU MARS with %d battles...\n", num_battles); 
    
    size_t battle_size = sizeof(BattleState);
    printf("BattleState size: %zu bytes. Total: %.2f MB\n", battle_size, (double)battle_size * num_battles / 1024/1024);
    
    BattleState* d_battles;
    CHECK(cudaMalloc(&d_battles, num_battles * battle_size));
    
    // Warriors
    std::vector<Instruction> imp;
    imp.push_back({MOV, I, DIRECT, DIRECT, 0, 1});

    std::vector<Instruction> dwarf;
    dwarf.push_back({ADD, AB, IMMEDIATE, DIRECT, 4, 3});
    dwarf.push_back({MOV, I, DIRECT, INDIRECT_B, 2, 2});
    dwarf.push_back({JMP, I, DIRECT, DIRECT, -2, 0});
    dwarf.push_back({DAT, F, IMMEDIATE, IMMEDIATE, 0, 0});
    
    Instruction *d_imp, *d_dwarf;
    CHECK(cudaMalloc(&d_imp, imp.size() * sizeof(Instruction)));
    CHECK(cudaMalloc(&d_dwarf, dwarf.size() * sizeof(Instruction)));
    CHECK(cudaMemcpy(d_imp, imp.data(), imp.size() * sizeof(Instruction), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dwarf, dwarf.data(), dwarf.size() * sizeof(Instruction), cudaMemcpyHostToDevice));
    
    uint32_t *h_pos_a = new uint32_t[num_battles];
    uint32_t *h_pos_b = new uint32_t[num_battles];
    for(int i=0; i<num_battles; i++) {
        h_pos_a[i] = rand() % (CORE_SIZE - 100);
        h_pos_b[i] = (h_pos_a[i] + 4000) % CORE_SIZE;
    }
    uint32_t *d_pos_a, *d_pos_b;
    CHECK(cudaMalloc(&d_pos_a, num_battles * 4));
    CHECK(cudaMalloc(&d_pos_b, num_battles * 4));
    CHECK(cudaMemcpy(d_pos_a, h_pos_a, num_battles * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_pos_b, h_pos_b, num_battles * 4, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_battles + threads - 1) / threads;
    
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    mars_aos_optimized_kernel<<<blocks, threads>>>(
        d_battles,
        d_imp, d_dwarf,
        imp.size(), dwarf.size(),
        d_pos_a, d_pos_b,
        num_battles
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK(cudaGetLastError());
    
    std::chrono::duration<double> diff = end - start;
    printf("AoS Optimized Kernel time: %.4f s\n", diff.count());
    printf("Throughput: %.2f battles/sec\n", num_battles / diff.count());
    
    return 0;
}
