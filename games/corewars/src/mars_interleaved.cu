#include "mars.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// Interleaved Storage: Array of Structs, Transposed
// Layout: core[pc * num_battles + battle_id]
// Access: core[idx].opcode, core[idx].b_field etc.
// This gives coalesced access (stride 8 bytes per thread) AND allows direct field access.
struct BattleStorageInterleaved {
    Instruction* core; // [CORE_SIZE * NUM_BATTLES]

    // Queues (SoA layout for pointers)
    uint16_t* qa_head; 
    uint16_t* qa_tail; 
    uint16_t* qa_count; 
    uint16_t* qa_pcs; // [MAX_PROCESSES * NUM_BATTLES]

    uint16_t* qb_head;
    uint16_t* qb_tail;
    uint16_t* qb_count;
    uint16_t* qb_pcs;

    uint8_t* winners;
    uint32_t* cycles;
};

#define IDX(row, col, width) ((size_t)(row) * (width) + (col))

__device__ void set_instruction_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles, Instruction instr) {
    size_t idx = IDX(pc, battle_id, num_battles);
    s->core[idx] = instr;
}

__device__ Instruction get_instruction_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles) {
    size_t idx = IDX(pc, battle_id, num_battles);
    return s->core[idx];
}

__device__ void set_b_field_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles, int16_t val) {
    size_t idx = IDX(pc, battle_id, num_battles);
    s->core[idx].b_field = val; // Direct 16-bit write, coalesced!
}

__device__ int16_t get_b_field_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles) {
    size_t idx = IDX(pc, battle_id, num_battles);
    return s->core[idx].b_field; // Direct 16-bit read, coalesced!
}

__device__ void push_q_interleaved(
    uint16_t* head, uint16_t* tail, uint16_t* count, uint16_t* pcs,
    int battle_id, int num_battles, uint16_t pc
) {
    if (*count < MAX_PROCESSES) {
        int t = *tail;
        size_t idx = IDX(t, battle_id, num_battles);
        pcs[idx] = pc;
        *tail = (t + 1) % MAX_PROCESSES;
        (*count)++;
    }
}

__device__ uint16_t pop_q_interleaved(
    uint16_t* head, uint16_t* tail, uint16_t* count, uint16_t* pcs,
    int battle_id, int num_battles
) {
    int h = *head;
    size_t idx = IDX(h, battle_id, num_battles);
    uint16_t pc = pcs[idx];
    *head = (h + 1) % MAX_PROCESSES;
    (*count)--;
    return pc;
}

__device__ int16_t resolve_addr_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int16_t val, uint8_t mode, int num_battles) {
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
        
        int16_t ptr = get_b_field_interleaved(s, battle_id, target, num_battles);
        eff_addr = target + (int)ptr;
        eff_addr %= CORE_SIZE;
        if (eff_addr < 0) eff_addr += CORE_SIZE;
        return (uint16_t)eff_addr;
    }
    
    eff_addr = (int)pc + (int)val;
    eff_addr %= CORE_SIZE;
    if (eff_addr < 0) eff_addr += CORE_SIZE;
    return (uint16_t)eff_addr;
}

__device__ void exec_instr_interleaved(BattleStorageInterleaved* s, int battle_id, int num_battles, 
                              uint16_t* q_head, uint16_t* q_tail, uint16_t* q_count, uint16_t* q_pcs) {
    
    uint16_t pc = pop_q_interleaved(q_head, q_tail, q_count, q_pcs, battle_id, num_battles);
    Instruction instr = get_instruction_interleaved(s, battle_id, pc, num_battles);
    
    uint16_t next_pc = (pc + 1) % CORE_SIZE;
    bool advance_pc = true;

    switch (instr.opcode) {
        case DAT:
            advance_pc = false;
            break;
        case MOV: {
            int16_t addr_a = resolve_addr_interleaved(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            int16_t addr_b = resolve_addr_interleaved(s, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
            Instruction src = get_instruction_interleaved(s, battle_id, addr_a, num_battles);
            set_instruction_interleaved(s, battle_id, addr_b, num_battles, src);
            break;
        }
        case ADD: {
             int16_t addr_b = resolve_addr_interleaved(s, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
             if (instr.mode_a == IMMEDIATE) {
                 int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
                 set_b_field_interleaved(s, battle_id, addr_b, num_battles, val + instr.a_field);
             } else {
                 int16_t addr_a = resolve_addr_interleaved(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                 int16_t val_a = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
                 int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
                 set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b + val_a);
             }
             break;
        }
        case JMP: {
            int16_t addr_a = resolve_addr_interleaved(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            next_pc = addr_a;
            advance_pc = false;
            push_q_interleaved(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
            return;
        }
        case SPL: {
            int16_t addr_a = resolve_addr_interleaved(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            push_q_interleaved(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
            push_q_interleaved(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, addr_a);
            return;
        }
        default:
            break;
    }
    
    if (advance_pc) {
        push_q_interleaved(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
    }
}

__global__ void mars_interleaved_kernel(
    BattleStorageInterleaved storage,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    uint16_t len_a, uint16_t len_b,
    const uint32_t* start_pos_a,
    const uint32_t* start_pos_b,
    uint32_t num_battles
) {
    int battle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (battle_id >= num_battles) return;

    // --- INITIALIZATION ---
    Instruction dat = {DAT, F, IMMEDIATE, IMMEDIATE, 0, 0};
    for (int i = 0; i < CORE_SIZE; i++) {
        set_instruction_interleaved(&storage, battle_id, i, num_battles, dat);
    }

    uint32_t pa = start_pos_a[battle_id];
    for (int i = 0; i < len_a; i++) {
        set_instruction_interleaved(&storage, battle_id, (pa + i)%CORE_SIZE, num_battles, warriors_a[i]);
    }
    uint32_t pb = start_pos_b[battle_id];
    for (int i = 0; i < len_b; i++) {
        set_instruction_interleaved(&storage, battle_id, (pb + i)%CORE_SIZE, num_battles, warriors_b[i]);
    }

    // --- REGISTER PROMOTION ---
    // Move queue state to registers to avoid global memory traffic
    uint16_t qa_head = 0;
    uint16_t qa_tail = 0;
    uint16_t qa_count = 0;
    
    uint16_t qb_head = 0;
    uint16_t qb_tail = 0;
    uint16_t qb_count = 0;

    // Push Initial Processes
    // We still use global memory for the process queues (pcs) themselves, 
    // but we save the pointer math overhead.
    
    // Push A
    if (qa_count < MAX_PROCESSES) {
        size_t idx = IDX(qa_tail, battle_id, num_battles);
        storage.qa_pcs[idx] = pa;
        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
        qa_count++;
    }

    // Push B
    if (qb_count < MAX_PROCESSES) {
        size_t idx = IDX(qb_tail, battle_id, num_battles);
        storage.qb_pcs[idx] = pb;
        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
        qb_count++;
    }

    // --- SIMULATION LOOP ---
    uint8_t winner = 0;
    int cycles = 0;
    
    while (cycles < MAX_CYCLES) {
        // --- Warrior A ---
        if (qa_count > 0) {
            // Pop
            size_t pop_idx = IDX(qa_head, battle_id, num_battles);
            uint16_t pc = storage.qa_pcs[pop_idx];
            qa_head = (qa_head + 1) % MAX_PROCESSES;
            qa_count--;

            // Fetch
            Instruction instr = get_instruction_interleaved(&storage, battle_id, pc, num_battles);
            
            uint16_t next_pc = (pc + 1) % CORE_SIZE;
            bool advance_pc = true;

            // Execute (Inlined for registers)
            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction src = get_instruction_interleaved(&storage, battle_id, addr_a, num_battles);
                    set_instruction_interleaved(&storage, battle_id, addr_b, num_battles, src);
                    break;
                }
                case ADD: {
                     int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                     if (instr.mode_a == IMMEDIATE) {
                         int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                         set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val + instr.a_field);
                     } else {
                         int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                         int16_t val_a = get_b_field_interleaved(&storage, battle_id, addr_a, num_battles);
                         int16_t val_b = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                         set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val_b + val_a);
                     }
                     break;
                }
                case JMP: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    next_pc = addr_a;
                    advance_pc = false;
                    // Push (JMP doesn't split, just continues)
                    if (qa_count < MAX_PROCESSES) {
                        size_t idx = IDX(qa_tail, battle_id, num_battles);
                        storage.qa_pcs[idx] = next_pc;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    break; // Break out of switch, but we handled push
                }
                case SPL: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    // Push next_pc
                    if (qa_count < MAX_PROCESSES) {
                        size_t idx = IDX(qa_tail, battle_id, num_battles);
                        storage.qa_pcs[idx] = next_pc;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    // Push target
                    if (qa_count < MAX_PROCESSES) {
                        size_t idx = IDX(qa_tail, battle_id, num_battles);
                        storage.qa_pcs[idx] = addr_a;
                        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                        qa_count++;
                    }
                    advance_pc = false; // Handled
                    break;
                }
                default:
                    break;
            }
            
            if (advance_pc) {
                if (qa_count < MAX_PROCESSES) {
                    size_t idx = IDX(qa_tail, battle_id, num_battles);
                    storage.qa_pcs[idx] = next_pc;
                    qa_tail = (qa_tail + 1) % MAX_PROCESSES;
                    qa_count++;
                }
            }

        } else {
            winner = 2; // B wins
            break;
        }
        
        // --- Warrior B ---
        if (qb_count > 0) {
            // Pop
            size_t pop_idx = IDX(qb_head, battle_id, num_battles);
            uint16_t pc = storage.qb_pcs[pop_idx];
            qb_head = (qb_head + 1) % MAX_PROCESSES;
            qb_count--;

            // Fetch
            Instruction instr = get_instruction_interleaved(&storage, battle_id, pc, num_battles);
            
            uint16_t next_pc = (pc + 1) % CORE_SIZE;
            bool advance_pc = true;

            // Execute (Inlined for registers)
            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction src = get_instruction_interleaved(&storage, battle_id, addr_a, num_battles);
                    set_instruction_interleaved(&storage, battle_id, addr_b, num_battles, src);
                    break;
                }
                case ADD: {
                     int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                     if (instr.mode_a == IMMEDIATE) {
                         int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                         set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val + instr.a_field);
                     } else {
                         int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                         int16_t val_a = get_b_field_interleaved(&storage, battle_id, addr_a, num_battles);
                         int16_t val_b = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                         set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val_b + val_a);
                     }
                     break;
                }
                case JMP: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    next_pc = addr_a;
                    advance_pc = false;
                    // Push
                    if (qb_count < MAX_PROCESSES) {
                        size_t idx = IDX(qb_tail, battle_id, num_battles);
                        storage.qb_pcs[idx] = next_pc;
                        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                        qb_count++;
                    }
                    break;
                }
                case SPL: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    // Push next_pc
                    if (qb_count < MAX_PROCESSES) {
                        size_t idx = IDX(qb_tail, battle_id, num_battles);
                        storage.qb_pcs[idx] = next_pc;
                        qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                        qb_count++;
                    }
                    // Push target
                    if (qb_count < MAX_PROCESSES) {
                        size_t idx = IDX(qb_tail, battle_id, num_battles);
                        storage.qb_pcs[idx] = addr_a;
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
                    size_t idx = IDX(qb_tail, battle_id, num_battles);
                    storage.qb_pcs[idx] = next_pc;
                    qb_tail = (qb_tail + 1) % MAX_PROCESSES;
                    qb_count++;
                }
            }

        } else {
            winner = 1; // A wins
            break;
        }
        
        cycles++;
    }

    // Write back results
    storage.winners[battle_id] = winner;
    storage.cycles[battle_id] = cycles;
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
    
    printf("Running Interleaved GPU MARS with %d battles...\n", num_battles);
    
    BattleStorageInterleaved d_store;
    size_t core_elems = (size_t)CORE_SIZE * num_battles;
    size_t q_elems = (size_t)MAX_PROCESSES * num_battles;
    
    printf("Allocating core memory: %zu elements (%.2f MB)\n", core_elems, (double)core_elems * 8 / 1024/1024);
    
    CHECK(cudaMalloc(&d_store.core, core_elems * sizeof(Instruction)));
    
    CHECK(cudaMalloc(&d_store.qa_head, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qa_tail, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qa_count, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qa_pcs, q_elems * 2));

    CHECK(cudaMalloc(&d_store.qb_head, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qb_tail, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qb_count, num_battles * 2));
    CHECK(cudaMalloc(&d_store.qb_pcs, q_elems * 2));
    
    CHECK(cudaMalloc(&d_store.winners, num_battles));
    CHECK(cudaMalloc(&d_store.cycles, num_battles * 4));
    
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
    
    mars_interleaved_kernel<<<blocks, threads>>>(
        d_store,
        d_imp, d_dwarf,
        imp.size(), dwarf.size(),
        d_pos_a, d_pos_b,
        num_battles
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK(cudaGetLastError());
    
    std::chrono::duration<double> diff = end - start;
    printf("Interleaved Kernel time: %.4f s\n", diff.count());
    printf("Throughput: %.2f battles/sec\n", num_battles / diff.count());
    
    return 0;
}
