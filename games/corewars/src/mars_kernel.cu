#include "mars.h"
#include <cuda_runtime.h>

__device__ void init_queue(ProcessQueue* q, uint16_t start_pc) {
    q->head = 0;
    q->tail = 0;
    q->count = 0;
    // Push initial process
    q->pcs[0] = start_pc;
    q->tail = 1;
    q->count = 1;
}

__device__ void push(ProcessQueue* q, uint16_t pc) {
    if (q->count < MAX_PROCESSES) {
        q->pcs[q->tail] = pc;
        q->tail = (q->tail + 1) % MAX_PROCESSES;
        q->count++;
    }
}

__device__ uint16_t pop(ProcessQueue* q) {
    uint16_t pc = q->pcs[q->head];
    q->head = (q->head + 1) % MAX_PROCESSES;
    q->count--;
    return pc;
}

__device__ int16_t resolve_address(Instruction* core, uint16_t pc, int16_t val, uint8_t mode) {
    // Simplified addressing modes for prototype
    // Only Direct ($) and Immediate (#) and Indirect B (@) implemented for now
    if (mode == IMMEDIATE) return 0; // Value is the operand itself, handled by caller usually
    
    int eff_addr;
    if (mode == DIRECT) {
        eff_addr = (int)pc + (int)val;
        eff_addr %= CORE_SIZE;
        if (eff_addr < 0) eff_addr += CORE_SIZE;
        return (uint16_t)eff_addr;
    }
    if (mode == INDIRECT_B) {
        // @ mode: pointer is in B-field of target
        int target = (int)pc + (int)val;
        target %= CORE_SIZE;
        if (target < 0) target += CORE_SIZE;
        
        int16_t ptr = core[target].b_field;
        eff_addr = target + (int)ptr;
        eff_addr %= CORE_SIZE;
        if (eff_addr < 0) eff_addr += CORE_SIZE;
        return (uint16_t)eff_addr;
    }
    
    // Fallback to Direct
    eff_addr = (int)pc + (int)val;
    eff_addr %= CORE_SIZE;
    if (eff_addr < 0) eff_addr += CORE_SIZE;
    return (uint16_t)eff_addr;
}

__device__ void execute_instruction(BattleState* state, ProcessQueue* q) {
    uint16_t pc = pop(q);
    Instruction instr = state->core[pc];

    // Decode
    uint16_t next_pc = (pc + 1) % CORE_SIZE;
    bool advance_pc = true;

    // Simple simplified execution logic
    switch (instr.opcode) {
        case DAT:
            // Kill process
            advance_pc = false; 
            break;
            
        case MOV:
            {
                // MOV A, B (copy instruction at A to B)
                // Assume Direct/Direct for simplicity prototype
                uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE; 
                uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                state->core[addr_b] = state->core[addr_a];
            }
            break;
            
        case ADD:
            {
                // ADD #A, B or ADD A, B
                if (instr.mode_a == IMMEDIATE) {
                    uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                    state->core[addr_b].b_field += instr.a_field;
                } else {
                    // Simplified
                    uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE;
                    uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                    state->core[addr_b].b_field += state->core[addr_a].b_field; // Simplified
                }
            }
            break;
            
        case SUB:
             {
                if (instr.mode_a == IMMEDIATE) {
                    uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                    state->core[addr_b].b_field -= instr.a_field;
                }
             }
            break;

        case JMP:
            {
                // JMP A
                uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE;
                next_pc = addr_a;
                advance_pc = false; // We jumped, so don't inc
                push(q, next_pc);
                return; 
            }
            break;
            
        case JMZ:
            {
                // JMZ A, B (Jump to A if B is zero)
                uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                if (state->core[addr_b].b_field == 0) {
                     uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE;
                     next_pc = addr_a;
                     advance_pc = false;
                     push(q, next_pc);
                     return;
                }
            }
            break;
            
        case SPL:
            {
                // SPL A (Split to A)
                uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE;
                push(q, next_pc); // Continue execution here
                push(q, addr_a);  // And spawn new process at A
                return;
            }
            break;
            
        case DJN:
            {
                 // DJN A, B (Dec B, Jump to A if B != 0)
                 uint16_t addr_b = (pc + instr.b_field) % CORE_SIZE;
                 state->core[addr_b].b_field--;
                 if (state->core[addr_b].b_field != 0) {
                     uint16_t addr_a = (pc + instr.a_field) % CORE_SIZE;
                     next_pc = addr_a;
                     advance_pc = false;
                     push(q, next_pc);
                     return;
                 }
            }
            break;

        default:
            // Treat as NOP or skip
            break;
    }

    if (advance_pc) {
        push(q, next_pc);
    }
}

__global__ void mars_battle_kernel(
    BattleState* battles,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    const uint16_t* warrior_lengths_a,
    const uint16_t* warrior_lengths_b,
    const uint32_t* start_positions_a,
    const uint32_t* start_positions_b,
    uint32_t num_battles
) {
    int battle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (battle_id >= num_battles) return;

    BattleState* state = &battles[battle_id];

    // Initialize core to DAT 0,0
    // Note: This loop is expensive (8000 iters). 
    // In optimized version, we might use memset or parallel init if one thread != one battle
    // But for "one thread per battle", we just do it.
    for (int i = 0; i < CORE_SIZE; i++) {
        state->core[i].opcode = DAT;
        state->core[i].modifier = F;
        state->core[i].mode_a = IMMEDIATE;
        state->core[i].mode_b = IMMEDIATE;
        state->core[i].a_field = 0;
        state->core[i].b_field = 0;
    }

    // Load Warrior A
    uint16_t len_a = warrior_lengths_a[battle_id];
    uint32_t pos_a = start_positions_a[battle_id];
    for (int i = 0; i < len_a; i++) {
        state->core[(pos_a + i) % CORE_SIZE] = warriors_a[i]; // Simplified: assuming 1 warrior template for now
    }

    // Load Warrior B
    uint16_t len_b = warrior_lengths_b[battle_id];
    uint32_t pos_b = start_positions_b[battle_id];
    for (int i = 0; i < len_b; i++) {
        state->core[(pos_b + i) % CORE_SIZE] = warriors_b[i];
    }

    // Init Queues
    init_queue(&state->queue_a, pos_a);
    init_queue(&state->queue_b, pos_b);

    // Main Loop
    state->cycles = 0;
    state->winner = 0; // Tie

    while (state->cycles < MAX_CYCLES) {
        // Warrior A turn
        if (state->queue_a.count > 0) {
            execute_instruction(state, &state->queue_a);
        } else {
            state->winner = 2; // B wins
            break;
        }

        // Warrior B turn
        if (state->queue_b.count > 0) {
            execute_instruction(state, &state->queue_b);
        } else {
            state->winner = 1; // A wins
            break;
        }

        state->cycles++;
    }
}
