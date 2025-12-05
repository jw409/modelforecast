#include "mars.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

// Packed Storage: SoA of uint64_t
// Layout: core[pc * num_battles + battle_id]
// This ensures coalesced access for every instruction fetch
struct BattleStoragePacked {
    uint64_t* core; // [CORE_SIZE * NUM_BATTLES]

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

__device__ uint64_t pack_instr(Instruction instr) {
    // Format: [Op:8][Mod:8][Ma:8][Mb:8][A:16][B:16]
    // Note: Opcode is 5 bits, Mod 3 bits etc. But we use 8 for simplicity and alignment
    uint64_t raw = 0;
    raw |= ((uint64_t)instr.opcode) << 56;
    raw |= ((uint64_t)instr.modifier) << 48;
    raw |= ((uint64_t)instr.mode_a) << 40;
    raw |= ((uint64_t)instr.mode_b) << 32;
    raw |= ((uint64_t)(uint16_t)instr.a_field) << 16; // Cast to uint16 to preserve bits
    raw |= ((uint64_t)(uint16_t)instr.b_field);
    return raw;
}

__device__ Instruction unpack_instr(uint64_t raw) {
    Instruction instr;
    instr.opcode = (raw >> 56) & 0xFF;
    instr.modifier = (raw >> 48) & 0xFF;
    instr.mode_a = (raw >> 40) & 0xFF;
    instr.mode_b = (raw >> 32) & 0xFF;
    instr.a_field = (int16_t)((raw >> 16) & 0xFFFF);
    instr.b_field = (int16_t)(raw & 0xFFFF);
    return instr;
}

__device__ void set_instruction_packed(BattleStoragePacked* s, int battle_id, int pc, int num_battles, Instruction instr) {
    size_t idx = IDX(pc, battle_id, num_battles);
    s->core[idx] = pack_instr(instr);
}

__device__ Instruction get_instruction_packed(BattleStoragePacked* s, int battle_id, int pc, int num_battles) {
    size_t idx = IDX(pc, battle_id, num_battles);
    return unpack_instr(s->core[idx]);
}

__device__ void set_b_field_packed(BattleStoragePacked* s, int battle_id, int pc, int num_battles, int16_t val) {
    size_t idx = IDX(pc, battle_id, num_battles);
    uint64_t raw = s->core[idx];
    // B field is bottom 16 bits
    raw &= ~0xFFFFULL;
    raw |= ((uint64_t)(uint16_t)val);
    s->core[idx] = raw;
}

__device__ int16_t get_b_field_packed(BattleStoragePacked* s, int battle_id, int pc, int num_battles) {
    size_t idx = IDX(pc, battle_id, num_battles);
    uint64_t raw = s->core[idx];
    return (int16_t)(raw & 0xFFFF);
}

__device__ void push_q_packed(
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

__device__ uint16_t pop_q_packed(
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

__device__ int16_t resolve_addr_packed(BattleStoragePacked* s, int battle_id, int pc, int16_t val, uint8_t mode, int num_battles) {
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
        
        int16_t ptr = get_b_field_packed(s, battle_id, target, num_battles);
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

__device__ void exec_instr_packed(BattleStoragePacked* s, int battle_id, int num_battles, 
                              uint16_t* q_head, uint16_t* q_tail, uint16_t* q_count, uint16_t* q_pcs) {
    
    uint16_t pc = pop_q_packed(q_head, q_tail, q_count, q_pcs, battle_id, num_battles);
    Instruction instr = get_instruction_packed(s, battle_id, pc, num_battles);
    
    uint16_t next_pc = (pc + 1) % CORE_SIZE;
    bool advance_pc = true;

    switch (instr.opcode) {
        case DAT:
            advance_pc = false;
            break;
        case MOV: {
            int16_t addr_a = resolve_addr_packed(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            int16_t addr_b = resolve_addr_packed(s, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
            Instruction src = get_instruction_packed(s, battle_id, addr_a, num_battles);
            set_instruction_packed(s, battle_id, addr_b, num_battles, src);
            break;
        }
        case ADD: {
             int16_t addr_b = resolve_addr_packed(s, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
             if (instr.mode_a == IMMEDIATE) {
                 int16_t val = get_b_field_packed(s, battle_id, addr_b, num_battles);
                 set_b_field_packed(s, battle_id, addr_b, num_battles, val + instr.a_field);
             } else {
                 int16_t addr_a = resolve_addr_packed(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                 int16_t val_a = get_b_field_packed(s, battle_id, addr_a, num_battles);
                 int16_t val_b = get_b_field_packed(s, battle_id, addr_b, num_battles);
                 set_b_field_packed(s, battle_id, addr_b, num_battles, val_b + val_a);
             }
             break;
        }
        case JMP: {
            int16_t addr_a = resolve_addr_packed(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            next_pc = addr_a;
            advance_pc = false;
            push_q_packed(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
            return;
        }
        case SPL: {
            int16_t addr_a = resolve_addr_packed(s, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
            push_q_packed(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
            push_q_packed(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, addr_a);
            return;
        }
        default:
            break;
    }
    
    if (advance_pc) {
        push_q_packed(q_head, q_tail, q_count, q_pcs, battle_id, num_battles, next_pc);
    }
}

__global__ void mars_packed_kernel(
    BattleStoragePacked storage,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    uint16_t len_a, uint16_t len_b,
    const uint32_t* start_pos_a,
    const uint32_t* start_pos_b,
    uint32_t num_battles
) {
    int battle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (battle_id >= num_battles) return;

    // Init Core
    Instruction dat = {DAT, F, IMMEDIATE, IMMEDIATE, 0, 0};
    for (int i = 0; i < CORE_SIZE; i++) {
        set_instruction_packed(&storage, battle_id, i, num_battles, dat);
    }

    // Load Warriors
    uint32_t pa = start_pos_a[battle_id];
    for (int i = 0; i < len_a; i++) {
        set_instruction_packed(&storage, battle_id, (pa + i)%CORE_SIZE, num_battles, warriors_a[i]);
    }
    uint32_t pb = start_pos_b[battle_id];
    for (int i = 0; i < len_b; i++) {
        set_instruction_packed(&storage, battle_id, (pb + i)%CORE_SIZE, num_battles, warriors_b[i]);
    }

    // Init Queues
    storage.qa_head[battle_id] = 0;
    storage.qa_tail[battle_id] = 0;
    storage.qa_count[battle_id] = 0;
    push_q_packed(&storage.qa_head[battle_id], &storage.qa_tail[battle_id], &storage.qa_count[battle_id], storage.qa_pcs, battle_id, num_battles, pa);

    storage.qb_head[battle_id] = 0;
    storage.qb_tail[battle_id] = 0;
    storage.qb_count[battle_id] = 0;
    push_q_packed(&storage.qb_head[battle_id], &storage.qb_tail[battle_id], &storage.qb_count[battle_id], storage.qb_pcs, battle_id, num_battles, pb);

    // Run
    storage.winners[battle_id] = 0;
    int cycles = 0;
    while (cycles < MAX_CYCLES) {
        if (storage.qa_count[battle_id] > 0) {
            exec_instr_packed(&storage, battle_id, num_battles, 
                &storage.qa_head[battle_id], &storage.qa_tail[battle_id], &storage.qa_count[battle_id], storage.qa_pcs);
        } else {
            storage.winners[battle_id] = 2;
            break;
        }
        
        if (storage.qb_count[battle_id] > 0) {
            exec_instr_packed(&storage, battle_id, num_battles, 
                &storage.qb_head[battle_id], &storage.qb_tail[battle_id], &storage.qb_count[battle_id], storage.qb_pcs);
        } else {
            storage.winners[battle_id] = 1;
            break;
        }
        cycles++;
    }
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
    
    printf("Running Packed (Coalesced) GPU MARS with %d battles...\n", num_battles);
    
    BattleStoragePacked d_store;
    size_t core_elems = (size_t)CORE_SIZE * num_battles;
    size_t q_elems = (size_t)MAX_PROCESSES * num_battles;
    
    printf("Allocating core memory: %zu elements (%.2f MB)\n", core_elems, (double)core_elems * 8 / 1024/1024);
    
    CHECK(cudaMalloc(&d_store.core, core_elems * sizeof(uint64_t)));
    
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
    
    mars_packed_kernel<<<blocks, threads>>>(
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
    printf("Packed Kernel time: %.4f s\n", diff.count());
    printf("Throughput: %.2f battles/sec\n", num_battles / diff.count());
    
    return 0;
}
