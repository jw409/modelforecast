#include "mars.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <cctype>
#include <algorithm>

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

__device__ void set_a_field_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles, int16_t val) {
    size_t idx = IDX(pc, battle_id, num_battles);
    s->core[idx].a_field = val;
}

__device__ int16_t get_a_field_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int num_battles) {
    size_t idx = IDX(pc, battle_id, num_battles);
    return s->core[idx].a_field;
}

// ============================================================================
// MODIFIER-AWARE INSTRUCTION EXECUTION (ICWS'94 Compliant)
// ============================================================================
// Modifiers define WHICH fields are operated on:
//   .A  - A-field of source → A-field of dest
//   .B  - B-field of source → B-field of dest
//   .AB - A-field of source → B-field of dest
//   .BA - B-field of source → A-field of dest
//   .F  - Both fields parallel (A→A, B→B)
//   .X  - Cross fields (A→B, B→A)
//   .I  - Entire instruction (opcode, modifier, modes, both fields)
// ============================================================================

__device__ void exec_mov_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier
) {
    switch (modifier) {
        case A: {
            // MOV.A: Copy A-field of source to A-field of dest
            int16_t val = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val);
            break;
        }
        case B: {
            // MOV.B: Copy B-field of source to B-field of dest
            int16_t val = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val);
            break;
        }
        case AB: {
            // MOV.AB: Copy A-field of source to B-field of dest
            int16_t val = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val);
            break;
        }
        case BA: {
            // MOV.BA: Copy B-field of source to A-field of dest
            int16_t val = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val);
            break;
        }
        case F: {
            // MOV.F: Copy both fields (A→A, B→B)
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_a);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b);
            break;
        }
        case X: {
            // MOV.X: Copy cross fields (A→B, B→A)
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_a);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_b);
            break;
        }
        case I:
        default: {
            // MOV.I: Copy entire instruction
            Instruction src = get_instruction_interleaved(s, battle_id, addr_a, num_battles);
            set_instruction_interleaved(s, battle_id, addr_b, num_battles, src);
            break;
        }
    }
}

__device__ void exec_add_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    // For IMMEDIATE mode, we use a_field directly instead of reading from addr_a
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;  // For .B and .BA with immediate, use a_field as source
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    switch (modifier) {
        case A: {
            // ADD.A: Add A-field of source to A-field of dest
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val + src_a) % CORE_SIZE);
            break;
        }
        case B: {
            // ADD.B: Add B-field of source to B-field of dest
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val + src_b) % CORE_SIZE);
            break;
        }
        case AB: {
            // ADD.AB: Add A-field of source to B-field of dest
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val + src_a) % CORE_SIZE);
            break;
        }
        case BA: {
            // ADD.BA: Add B-field of source to A-field of dest
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val + src_b) % CORE_SIZE);
            break;
        }
        case F:
        case I: {
            // ADD.F/ADD.I: Add both fields (A+=A, B+=B)
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a + src_a) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b + src_b) % CORE_SIZE);
            break;
        }
        case X: {
            // ADD.X: Add cross fields (A+=B, B+=A)
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a + src_b) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b + src_a) % CORE_SIZE);
            break;
        }
        default: {
            // Default to AB for backwards compatibility with old code
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val + src_a) % CORE_SIZE);
            break;
        }
    }
}

// SUB: Subtract source from destination
__device__ void exec_sub_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    switch (modifier) {
        case A: {
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val - src_a + CORE_SIZE) % CORE_SIZE);
            break;
        }
        case B: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val - src_b + CORE_SIZE) % CORE_SIZE);
            break;
        }
        case AB: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val - src_a + CORE_SIZE) % CORE_SIZE);
            break;
        }
        case BA: {
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val - src_b + CORE_SIZE) % CORE_SIZE);
            break;
        }
        case F:
        case I: {
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a - src_a + CORE_SIZE) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b - src_b + CORE_SIZE) % CORE_SIZE);
            break;
        }
        case X: {
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a - src_b + CORE_SIZE) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b - src_a + CORE_SIZE) % CORE_SIZE);
            break;
        }
        default: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val - src_a + CORE_SIZE) % CORE_SIZE);
            break;
        }
    }
}

// MUL: Multiply
__device__ void exec_mul_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    switch (modifier) {
        case A: {
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val * src_a) % CORE_SIZE);
            break;
        }
        case B: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val * src_b) % CORE_SIZE);
            break;
        }
        case AB: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val * src_a) % CORE_SIZE);
            break;
        }
        case BA: {
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val * src_b) % CORE_SIZE);
            break;
        }
        case F:
        case I: {
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a * src_a) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b * src_b) % CORE_SIZE);
            break;
        }
        case X: {
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, (val_a * src_b) % CORE_SIZE);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val_b * src_a) % CORE_SIZE);
            break;
        }
        default: {
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, (val * src_a) % CORE_SIZE);
            break;
        }
    }
}

// DIV: Divide (returns true if division by zero - process dies)
__device__ bool exec_div_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    switch (modifier) {
        case A: {
            if (src_a == 0) return true;
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val / src_a);
            break;
        }
        case B: {
            if (src_b == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val / src_b);
            break;
        }
        case AB: {
            if (src_a == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val / src_a);
            break;
        }
        case BA: {
            if (src_b == 0) return true;
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val / src_b);
            break;
        }
        case F:
        case I: {
            if (src_a == 0 || src_b == 0) return true;
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_a / src_a);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b / src_b);
            break;
        }
        case X: {
            if (src_a == 0 || src_b == 0) return true;
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_a / src_b);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b / src_a);
            break;
        }
        default: {
            if (src_a == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val / src_a);
            break;
        }
    }
    return false;
}

// MOD: Modulo (returns true if division by zero - process dies)
__device__ bool exec_mod_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    switch (modifier) {
        case A: {
            if (src_a == 0) return true;
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val % src_a);
            break;
        }
        case B: {
            if (src_b == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val % src_b);
            break;
        }
        case AB: {
            if (src_a == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val % src_a);
            break;
        }
        case BA: {
            if (src_b == 0) return true;
            int16_t val = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val % src_b);
            break;
        }
        case F:
        case I: {
            if (src_a == 0 || src_b == 0) return true;
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_a % src_a);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b % src_b);
            break;
        }
        case X: {
            if (src_a == 0 || src_b == 0) return true;
            int16_t val_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
            int16_t val_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_a_field_interleaved(s, battle_id, addr_b, num_battles, val_a % src_b);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val_b % src_a);
            break;
        }
        default: {
            if (src_a == 0) return true;
            int16_t val = get_b_field_interleaved(s, battle_id, addr_b, num_battles);
            set_b_field_interleaved(s, battle_id, addr_b, num_battles, val % src_a);
            break;
        }
    }
    return false;
}

// SLT: Skip if Less Than (returns true if should skip)
__device__ bool exec_slt_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier, uint8_t mode_a, int16_t a_field
) {
    int16_t src_a, src_b;
    if (mode_a == IMMEDIATE) {
        src_a = a_field;
        src_b = a_field;
    } else {
        src_a = get_a_field_interleaved(s, battle_id, addr_a, num_battles);
        src_b = get_b_field_interleaved(s, battle_id, addr_a, num_battles);
    }

    int16_t dst_a = get_a_field_interleaved(s, battle_id, addr_b, num_battles);
    int16_t dst_b = get_b_field_interleaved(s, battle_id, addr_b, num_battles);

    switch (modifier) {
        case A: return src_a < dst_a;
        case B: return src_b < dst_b;
        case AB: return src_a < dst_b;
        case BA: return src_b < dst_a;
        case F:
        case I: return (src_a < dst_a) && (src_b < dst_b);
        case X: return (src_a < dst_b) && (src_b < dst_a);
        default: return src_a < dst_b;
    }
}

// CMP/SEQ: Skip if Equal (returns true if should skip)
__device__ bool exec_seq_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier
) {
    Instruction src = get_instruction_interleaved(s, battle_id, addr_a, num_battles);
    Instruction dst = get_instruction_interleaved(s, battle_id, addr_b, num_battles);

    switch (modifier) {
        case A: return src.a_field == dst.a_field;
        case B: return src.b_field == dst.b_field;
        case AB: return src.a_field == dst.b_field;
        case BA: return src.b_field == dst.a_field;
        case F: return (src.a_field == dst.a_field) && (src.b_field == dst.b_field);
        case X: return (src.a_field == dst.b_field) && (src.b_field == dst.a_field);
        case I: return (src.opcode == dst.opcode) && (src.modifier == dst.modifier) &&
                       (src.mode_a == dst.mode_a) && (src.mode_b == dst.mode_b) &&
                       (src.a_field == dst.a_field) && (src.b_field == dst.b_field);
        default: return src.a_field == dst.b_field;
    }
}

// SNE: Skip if Not Equal (returns true if should skip)
__device__ bool exec_sne_interleaved(
    BattleStorageInterleaved* s, int battle_id, int num_battles,
    int16_t addr_a, int16_t addr_b, uint8_t modifier
) {
    return !exec_seq_interleaved(s, battle_id, num_battles, addr_a, addr_b, modifier);
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

// ============================================================================
// ADDRESSING MODE RESOLUTION (ICWS'94 Compliant)
// ============================================================================
// Helper to normalize addresses to [0, CORE_SIZE)
__device__ int16_t normalize_addr(int addr) {
    addr %= CORE_SIZE;
    if (addr < 0) addr += CORE_SIZE;
    return (int16_t)addr;
}

// Full ICWS'94 addressing mode resolution with pre-decrement/post-increment
__device__ int16_t resolve_addr_interleaved(BattleStorageInterleaved* s, int battle_id, int pc, int16_t val, uint8_t mode, int num_battles) {
    // IMMEDIATE (#): operand is the value itself, effective address is 0 (current instruction)
    if (mode == IMMEDIATE) return pc;  // Point to current instruction for field access

    // DIRECT ($): pc + value
    if (mode == DIRECT) {
        return normalize_addr((int)pc + (int)val);
    }

    // Calculate intermediate address for indirect modes
    int16_t interm = normalize_addr((int)pc + (int)val);

    // INDIRECT_A (*): dereference via A-field
    if (mode == INDIRECT_A) {
        int16_t ptr = get_a_field_interleaved(s, battle_id, interm, num_battles);
        return normalize_addr((int)interm + (int)ptr);
    }

    // INDIRECT_B (@): dereference via B-field
    if (mode == INDIRECT_B) {
        int16_t ptr = get_b_field_interleaved(s, battle_id, interm, num_battles);
        return normalize_addr((int)interm + (int)ptr);
    }

    // PREDEC_A ({): decrement A-field first, then dereference
    if (mode == PREDEC_A) {
        int16_t ptr = get_a_field_interleaved(s, battle_id, interm, num_battles);
        ptr = normalize_addr(ptr - 1);
        set_a_field_interleaved(s, battle_id, interm, num_battles, ptr);
        return normalize_addr((int)interm + (int)ptr);
    }

    // PREDEC_B (<): decrement B-field first, then dereference
    if (mode == PREDEC_B) {
        int16_t ptr = get_b_field_interleaved(s, battle_id, interm, num_battles);
        ptr = normalize_addr(ptr - 1);
        set_b_field_interleaved(s, battle_id, interm, num_battles, ptr);
        return normalize_addr((int)interm + (int)ptr);
    }

    // POSTINC_A (}): dereference via A-field, then increment
    if (mode == POSTINC_A) {
        int16_t ptr = get_a_field_interleaved(s, battle_id, interm, num_battles);
        int16_t result = normalize_addr((int)interm + (int)ptr);
        set_a_field_interleaved(s, battle_id, interm, num_battles, normalize_addr(ptr + 1));
        return result;
    }

    // POSTINC_B (>): dereference via B-field, then increment
    if (mode == POSTINC_B) {
        int16_t ptr = get_b_field_interleaved(s, battle_id, interm, num_battles);
        int16_t result = normalize_addr((int)interm + (int)ptr);
        set_b_field_interleaved(s, battle_id, interm, num_battles, normalize_addr(ptr + 1));
        return result;
    }

    // Fallback to DIRECT
    return normalize_addr((int)pc + (int)val);
}

// NOTE: exec_instr_interleaved was removed - all execution is now inlined in kernel
// with full ICWS'94 opcode and modifier support

__global__ void mars_interleaved_kernel(
    BattleStorageInterleaved storage,
    const Instruction* warriors_a,
    const Instruction* warriors_b,
    uint16_t len_a, uint16_t len_b,
    uint16_t entry_a, uint16_t entry_b,  // Entry point offsets (ORG)
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
    
    // Push A (at entry point offset)
    if (qa_count < MAX_PROCESSES) {
        size_t idx = IDX(qa_tail, battle_id, num_battles);
        storage.qa_pcs[idx] = (pa + entry_a) % CORE_SIZE;
        qa_tail = (qa_tail + 1) % MAX_PROCESSES;
        qa_count++;
    }

    // Push B (at entry point offset)
    if (qb_count < MAX_PROCESSES) {
        size_t idx = IDX(qb_tail, battle_id, num_battles);
        storage.qb_pcs[idx] = (pb + entry_b) % CORE_SIZE;
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

            // Execute (Inlined for registers) - MODIFIER-AWARE (ICWS'94 Compliant)
            // Debug first few cycles AND cycles around key events
            if (battle_id == 0 && (cycles < 10 || (cycles >= 3998 && cycles <= 4005) || (cycles >= 11994 && cycles <= 11997))) {
                printf("C%d A: pc=%d op=%d mod=%d a=%d b=%d modeA=%d modeB=%d\n",
                       cycles, pc, instr.opcode, instr.modifier, instr.a_field, instr.b_field, instr.mode_a, instr.mode_b);
            }

            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    if (battle_id == 0) printf("C%d A: HIT DAT at pc=%d\n", cycles, pc);
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    // Handle IMMEDIATE mode for source: use current instruction as source
                    if (instr.mode_a == IMMEDIATE) {
                        addr_a = pc;  // Source is the current instruction itself
                    }
                    exec_mov_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier);
                    if (battle_id == 0 && (cycles < 10 || (cycles >= 3998 && cycles <= 4005) || (cycles >= 11994 && cycles <= 11997))) {
                        printf("  MOV: addr_a=%d addr_b=%d\n", addr_a, addr_b);
                    }
                    break;
                }
                case ADD: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_add_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    if (battle_id == 0 && (cycles >= 11994 && cycles <= 11997)) {
                        printf("  ADD: addr_b=%d\n", addr_b);
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
                case SUB: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_sub_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    break;
                }
                case MUL: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_mul_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    break;
                }
                case DIV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_div_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        advance_pc = false; // Process dies on divide by zero
                    }
                    break;
                }
                case MOD: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_mod_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        advance_pc = false; // Process dies on divide by zero
                    }
                    break;
                }
                case JMZ: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction target = get_instruction_interleaved(&storage, battle_id, addr_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: jump = (target.a_field == 0); break;
                        case B: case AB: jump = (target.b_field == 0); break;
                        case F: case X: case I: jump = (target.a_field == 0 && target.b_field == 0); break;
                        default: jump = (target.b_field == 0); break;
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case JMN: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction target = get_instruction_interleaved(&storage, battle_id, addr_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: jump = (target.a_field != 0); break;
                        case B: case AB: jump = (target.b_field != 0); break;
                        case F: case X: case I: jump = (target.a_field != 0 || target.b_field != 0); break;
                        default: jump = (target.b_field != 0); break;
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case DJN: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: {
                            int16_t val = get_a_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_a_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                        case B: case AB: {
                            int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                        case F: case X: case I: {
                            int16_t val_a = get_a_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            int16_t val_b = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val_a = (val_a - 1 + CORE_SIZE) % CORE_SIZE;
                            val_b = (val_b - 1 + CORE_SIZE) % CORE_SIZE;
                            set_a_field_interleaved(&storage, battle_id, addr_b, num_battles, val_a);
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val_b);
                            jump = (val_a != 0 || val_b != 0);
                            break;
                        }
                        default: {
                            int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case SLT: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_slt_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case CMP:
                case SEQ: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_seq_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case SNE: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_sne_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case NOP:
                    // Do nothing
                    break;
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

            // Debug first few cycles AND cycles around 11995 of battle 0
            if (battle_id == 0 && (cycles < 10 || (cycles >= 11994 && cycles <= 11997))) {
                printf("C%d B: pc=%d op=%d mod=%d a=%d b=%d modeA=%d modeB=%d\n",
                       cycles, pc, instr.opcode, instr.modifier, instr.a_field, instr.b_field, instr.mode_a, instr.mode_b);
            }

            // Execute (Inlined for registers) - MODIFIER-AWARE (ICWS'94 Compliant)
            switch (instr.opcode) {
                case DAT:
                    advance_pc = false;
                    if (battle_id == 0) printf("C%d B: HIT DAT at pc=%d\n", cycles, pc);
                    break;
                case MOV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    // Handle IMMEDIATE mode for source: use current instruction as source
                    if (instr.mode_a == IMMEDIATE) {
                        addr_a = pc;  // Source is the current instruction itself
                    }
                    exec_mov_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier);
                    if (battle_id == 0 && cycles < 10) {
                        printf("  MOV: addr_a=%d addr_b=%d\n", addr_a, addr_b);
                    }
                    break;
                }
                case ADD: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_add_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    if (battle_id == 0 && cycles < 10) {
                        printf("  ADD: addr_b=%d\n", addr_b);
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
                    if (battle_id == 0 && cycles < 10) {
                        printf("  JMP: next_pc=%d\n", next_pc);
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
                case SUB: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_sub_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    break;
                }
                case MUL: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    exec_mul_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field);
                    break;
                }
                case DIV: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_div_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        advance_pc = false; // Process dies on divide by zero
                    }
                    break;
                }
                case MOD: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_mod_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        advance_pc = false; // Process dies on divide by zero
                    }
                    break;
                }
                case JMZ: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction target = get_instruction_interleaved(&storage, battle_id, addr_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: jump = (target.a_field == 0); break;
                        case B: case AB: jump = (target.b_field == 0); break;
                        case F: case X: case I: jump = (target.a_field == 0 && target.b_field == 0); break;
                        default: jump = (target.b_field == 0); break;
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case JMN: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    Instruction target = get_instruction_interleaved(&storage, battle_id, addr_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: jump = (target.a_field != 0); break;
                        case B: case AB: jump = (target.b_field != 0); break;
                        case F: case X: case I: jump = (target.a_field != 0 || target.b_field != 0); break;
                        default: jump = (target.b_field != 0); break;
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case DJN: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    bool jump = false;
                    switch (instr.modifier) {
                        case A: case BA: {
                            int16_t val = get_a_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_a_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                        case B: case AB: {
                            int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                        case F: case X: case I: {
                            int16_t val_a = get_a_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            int16_t val_b = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val_a = (val_a - 1 + CORE_SIZE) % CORE_SIZE;
                            val_b = (val_b - 1 + CORE_SIZE) % CORE_SIZE;
                            set_a_field_interleaved(&storage, battle_id, addr_b, num_battles, val_a);
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val_b);
                            jump = (val_a != 0 || val_b != 0);
                            break;
                        }
                        default: {
                            int16_t val = get_b_field_interleaved(&storage, battle_id, addr_b, num_battles);
                            val = (val - 1 + CORE_SIZE) % CORE_SIZE;
                            set_b_field_interleaved(&storage, battle_id, addr_b, num_battles, val);
                            jump = (val != 0);
                            break;
                        }
                    }
                    if (jump) {
                        next_pc = addr_a;
                    }
                    break;
                }
                case SLT: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_slt_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier, instr.mode_a, instr.a_field)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case CMP:
                case SEQ: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_seq_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case SNE: {
                    int16_t addr_a = resolve_addr_interleaved(&storage, battle_id, pc, instr.a_field, instr.mode_a, num_battles);
                    int16_t addr_b = resolve_addr_interleaved(&storage, battle_id, pc, instr.b_field, instr.mode_b, num_battles);
                    if (exec_sne_interleaved(&storage, battle_id, num_battles, addr_a, addr_b, instr.modifier)) {
                        next_pc = (next_pc + 1) % CORE_SIZE;  // Skip one more
                    }
                    break;
                }
                case NOP:
                    // Do nothing
                    break;
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

    // Debug output for first battle
    if (battle_id == 0) {
        printf("Battle 0: winner=%d cycles=%d qa_count=%d qb_count=%d\n",
               winner, cycles, qa_count, qb_count);
        // Check cell 4479 which should be bombed at cycle 11995
        Instruction cell4479 = get_instruction_interleaved(&storage, 0, 4479, num_battles);
        printf("Cell 4479: op=%d mod=%d a=%d b=%d\n",
               cell4479.opcode, cell4479.modifier, cell4479.a_field, cell4479.b_field);
    }
}

// ============================================================================
// REDCODE LOADER - Parses pMARS -A output to load .red files
// ============================================================================

struct Warrior {
    std::vector<Instruction> code;
    uint16_t entry;  // Entry point (from ORG)
    std::string name;
};

// Parse addressing mode from pMARS output
uint8_t parse_mode(char c) {
    switch (c) {
        case '#': return IMMEDIATE;
        case '$': return DIRECT;
        case '*': return INDIRECT_A;
        case '@': return INDIRECT_B;
        case '{': return PREDEC_A;
        case '<': return PREDEC_B;
        case '}': return POSTINC_A;
        case '>': return POSTINC_B;
        default:  return DIRECT;  // Default to direct
    }
}

// Parse opcode from string
uint8_t parse_opcode(const std::string& op) {
    static std::map<std::string, uint8_t> opcodes = {
        {"DAT", DAT}, {"MOV", MOV}, {"ADD", ADD}, {"SUB", SUB},
        {"MUL", MUL}, {"DIV", DIV}, {"MOD", MOD}, {"JMP", JMP},
        {"JMZ", JMZ}, {"JMN", JMN}, {"DJN", DJN}, {"SPL", SPL},
        {"SLT", SLT}, {"CMP", CMP}, {"SEQ", SEQ}, {"SNE", SNE},
        {"LDP", LDP}, {"STP", STP}, {"NOP", NOP}
    };
    auto it = opcodes.find(op);
    return it != opcodes.end() ? it->second : DAT;
}

// Parse modifier from string
uint8_t parse_modifier(const std::string& mod) {
    static std::map<std::string, uint8_t> modifiers = {
        {"A", A}, {"B", B}, {"AB", AB}, {"BA", BA},
        {"F", F}, {"X", X}, {"I", I}
    };
    auto it = modifiers.find(mod);
    return it != modifiers.end() ? it->second : F;
}

// Load warrior from .red file using pMARS to assemble
Warrior load_warrior(const char* filename) {
    Warrior w;
    w.entry = 0;
    w.name = filename;

    // Run pMARS -A to get assembled output
    // Try multiple pMARS locations (relative paths for portability)
    char cmd[512];
    const char* pmars_paths[] = {
        "pmars",
        "../pmars",
        "../../pmars",
        "/usr/local/bin/pmars",
        "/usr/bin/pmars",
        nullptr
    };

    // Find working pMARS path
    const char* working_pmars = nullptr;
    for (int i = 0; pmars_paths[i] != nullptr; i++) {
        snprintf(cmd, sizeof(cmd), "%s --version 2>&1", pmars_paths[i]);
        FILE* test_pipe = popen(cmd, "r");
        if (test_pipe) {
            char test[64];
            if (fgets(test, sizeof(test), test_pipe)) {
                // Check if output contains "pMARS" or version info (not "not found")
                if (strstr(test, "pMARS") || strstr(test, "0.9") || strstr(test, "usage")) {
                    working_pmars = pmars_paths[i];
                    pclose(test_pipe);
                    break;
                }
            }
            pclose(test_pipe);
        }
    }

    if (!working_pmars) {
        fprintf(stderr, "ERROR: Could not find pMARS. Install it or ensure it's in PATH.\n");
        return w;
    }

    // Run pMARS -A with found path
    snprintf(cmd, sizeof(cmd), "%s '%s' -A 2>&1", working_pmars, filename);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        fprintf(stderr, "ERROR: Could not run pMARS on %s\n", filename);
        return w;
    }

    char line[256];
    bool in_code = false;

    while (fgets(line, sizeof(line), pipe)) {
        std::string s(line);

        // Skip warnings
        if (s.find("Warning") != std::string::npos) continue;
        if (s.find("Number of warnings") != std::string::npos) continue;
        if (s.find("Error") != std::string::npos) {
            fprintf(stderr, "pMARS error for %s: %s", filename, line);
            continue;
        }

        // Parse ORG line
        if (s.find("ORG") != std::string::npos) {
            size_t pos = s.find("ORG");
            int org_val = 0;
            if (sscanf(s.c_str() + pos + 3, "%d", &org_val) == 1) {
                w.entry = (uint16_t)org_val;
            }
            in_code = true;
            continue;
        }

        // Parse END line - we're done
        if (s.find("END") != std::string::npos) {
            break;
        }

        // Parse instruction line
        // Format: "       MOV.I  <  -100, >   200"
        if (!in_code) continue;

        // Find opcode.modifier
        size_t dot_pos = s.find('.');
        if (dot_pos == std::string::npos) continue;

        // Extract opcode (3 chars before dot)
        size_t op_start = dot_pos;
        while (op_start > 0 && !isspace(s[op_start-1])) op_start--;
        std::string opcode_str = s.substr(op_start, dot_pos - op_start);

        // Extract modifier (1-2 chars after dot)
        size_t mod_end = dot_pos + 1;
        while (mod_end < s.length() && isalpha(s[mod_end])) mod_end++;
        std::string mod_str = s.substr(dot_pos + 1, mod_end - dot_pos - 1);

        // Find mode_a and a_field
        size_t a_start = mod_end;
        while (a_start < s.length() && isspace(s[a_start])) a_start++;

        char mode_a_char = '$';  // default
        if (a_start < s.length() && !isdigit(s[a_start]) && s[a_start] != '-') {
            mode_a_char = s[a_start];
            a_start++;
        }
        while (a_start < s.length() && isspace(s[a_start])) a_start++;

        int a_val = 0;
        sscanf(s.c_str() + a_start, "%d", &a_val);

        // Find comma and mode_b, b_field
        size_t comma = s.find(',', a_start);
        if (comma == std::string::npos) continue;

        size_t b_start = comma + 1;
        while (b_start < s.length() && isspace(s[b_start])) b_start++;

        char mode_b_char = '$';  // default
        if (b_start < s.length() && !isdigit(s[b_start]) && s[b_start] != '-') {
            mode_b_char = s[b_start];
            b_start++;
        }
        while (b_start < s.length() && isspace(s[b_start])) b_start++;

        int b_val = 0;
        sscanf(s.c_str() + b_start, "%d", &b_val);

        // Build instruction
        Instruction instr;
        instr.opcode = parse_opcode(opcode_str);
        instr.modifier = parse_modifier(mod_str);
        instr.mode_a = parse_mode(mode_a_char);
        instr.mode_b = parse_mode(mode_b_char);
        instr.a_field = (int16_t)a_val;
        instr.b_field = (int16_t)b_val;

        w.code.push_back(instr);
    }

    pclose(pipe);

    printf("Loaded %s: %zu instructions, entry=%d\n", filename, w.code.size(), w.entry);
    return w;
}

void print_usage(const char* prog) {
    printf("Usage: %s <warrior_a.red> <warrior_b.red> [num_battles] [options]\n", prog);
    printf("Options:\n");
    printf("  --fixed-pos POS  Fix warrior B at position POS (default: random)\n");
    printf("  --seed SEED      Random seed (default: 42)\n");
    printf("  --help           Show this help\n");
    printf("\nExample:\n");
    printf("  %s dwarf.red imp.red 100000\n", prog);
    printf("  %s warriors/tournament_001/*.red 100000\n", prog);
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
    int32_t fixed_pos_b = -1;  // -1 = random, >=0 = fixed position for warrior B
    uint32_t seed = 42;
    const char* warrior_a_file = nullptr;
    const char* warrior_b_file = nullptr;

    // Parse arguments: warrior_a.red warrior_b.red [num_battles] [options]
    std::vector<const char*> positional;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fixed-pos") == 0 && i+1 < argc) {
            fixed_pos_b = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) {
            seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            positional.push_back(argv[i]);
        }
    }

    // Extract positional arguments
    if (positional.size() >= 2) {
        // Check if first two are .red files or numbers
        bool first_is_file = strstr(positional[0], ".red") != nullptr;
        if (first_is_file) {
            warrior_a_file = positional[0];
            warrior_b_file = positional[1];
            if (positional.size() >= 3) {
                num_battles = atoi(positional[2]);
            }
        } else {
            // Old style: just num_battles (use hardcoded warriors)
            num_battles = atoi(positional[0]);
        }
    } else if (positional.size() == 1) {
        // Single arg - could be num_battles or a file
        if (strstr(positional[0], ".red") != nullptr) {
            fprintf(stderr, "ERROR: Need two warrior files\n");
            print_usage(argv[0]);
            return 1;
        }
        num_battles = atoi(positional[0]);
    }

    // Load warriors from files or use hardcoded defaults
    Warrior warrior_a, warrior_b;

    if (warrior_a_file && warrior_b_file) {
        printf("Loading warriors from files...\n");
        warrior_a = load_warrior(warrior_a_file);
        warrior_b = load_warrior(warrior_b_file);

        if (warrior_a.code.empty() || warrior_b.code.empty()) {
            fprintf(stderr, "ERROR: Failed to load warriors\n");
            return 1;
        }
    } else {
        // Fallback to hardcoded DWARF vs IMP for backwards compatibility
        printf("No warrior files specified, using hardcoded DWARF vs IMP\n");
        warrior_a.name = "DWARF";
        warrior_a.entry = 1;
        warrior_a.code.push_back({DAT, F, IMMEDIATE, IMMEDIATE, 0, 0});
        warrior_a.code.push_back({ADD, AB, IMMEDIATE, DIRECT, 4, -1});
        warrior_a.code.push_back({MOV, AB, IMMEDIATE, INDIRECT_B, 0, -2});
        warrior_a.code.push_back({JMP, A, DIRECT, DIRECT, -2, 0});

        warrior_b.name = "IMP";
        warrior_b.entry = 0;
        warrior_b.code.push_back({MOV, I, DIRECT, DIRECT, 0, 1});
    }

    printf("\n=== GPU MARS Tournament ===\n");
    printf("Warrior A: %s (%zu instructions, entry=%d)\n",
           warrior_a.name.c_str(), warrior_a.code.size(), warrior_a.entry);
    printf("Warrior B: %s (%zu instructions, entry=%d)\n",
           warrior_b.name.c_str(), warrior_b.code.size(), warrior_b.entry);
    printf("Battles: %u", num_battles);
    if (fixed_pos_b >= 0) printf(" (fixed pos B=%d)", fixed_pos_b);
    printf(" seed=%u\n\n", seed);
    
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

    // Copy warriors to GPU
    Instruction *d_warrior_a, *d_warrior_b;
    CHECK(cudaMalloc(&d_warrior_a, warrior_a.code.size() * sizeof(Instruction)));
    CHECK(cudaMalloc(&d_warrior_b, warrior_b.code.size() * sizeof(Instruction)));
    CHECK(cudaMemcpy(d_warrior_a, warrior_a.code.data(), warrior_a.code.size() * sizeof(Instruction), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_warrior_b, warrior_b.code.data(), warrior_b.code.size() * sizeof(Instruction), cudaMemcpyHostToDevice));

    // Random positions with ICWS'94 minimum separation rules
    // Minimum distance is typically core_size/10 or warrior_length, whichever is larger
    const int MIN_DISTANCE = CORE_SIZE / 10;  // 800 for 8000 core

    uint32_t *h_pos_a = new uint32_t[num_battles];
    uint32_t *h_pos_b = new uint32_t[num_battles];
    srand(seed);

    for(uint32_t i=0; i<num_battles; i++) {
        if (fixed_pos_b >= 0) {
            // Deterministic mode: warrior A at 0, warrior B at fixed position
            h_pos_a[i] = 0;
            h_pos_b[i] = fixed_pos_b;
        } else {
            // Random mode: Place warrior A randomly
            h_pos_a[i] = rand() % CORE_SIZE;

            // Place warrior B at random distance (at least MIN_DISTANCE away)
            // This matches pMARS behavior where separation varies randomly
            int separation = MIN_DISTANCE + (rand() % (CORE_SIZE - 2 * MIN_DISTANCE));
            h_pos_b[i] = (h_pos_a[i] + separation) % CORE_SIZE;
        }
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

    // Launch GPU kernel
    mars_interleaved_kernel<<<blocks, threads>>>(
        d_store,
        d_warrior_a, d_warrior_b,
        warrior_a.code.size(), warrior_b.code.size(),
        warrior_a.entry, warrior_b.entry,
        d_pos_a, d_pos_b,
        num_battles
    );
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    CHECK(cudaGetLastError());

    std::chrono::duration<double> diff = end - start;
    printf("Interleaved Kernel time: %.4f s\n", diff.count());
    printf("Throughput: %.2f battles/sec\n", num_battles / diff.count());

    // Collect results for cross-validation
    uint8_t* h_winners = new uint8_t[num_battles];
    CHECK(cudaMemcpy(h_winners, d_store.winners, num_battles, cudaMemcpyDeviceToHost));

    uint32_t a_wins = 0, b_wins = 0, ties = 0;
    for (uint32_t i = 0; i < num_battles; i++) {
        if (h_winners[i] == 1) a_wins++;
        else if (h_winners[i] == 2) b_wins++;
        else ties++;
    }

    printf("\n=== Battle Results ===\n");
    printf("Warrior A (%s): %u wins (%.1f%%)\n", warrior_a.name.c_str(), a_wins, 100.0 * a_wins / num_battles);
    printf("Warrior B (%s): %u wins (%.1f%%)\n", warrior_b.name.c_str(), b_wins, 100.0 * b_wins / num_battles);
    printf("Ties: %u (%.1f%%)\n", ties, 100.0 * ties / num_battles);
    printf("Results: %u %u %u\n", a_wins, b_wins, ties);  // pMARS-compatible format: A B Tie

    delete[] h_winners;
    delete[] h_pos_a;
    delete[] h_pos_b;

    // Cleanup GPU memory
    cudaFree(d_store.core);
    cudaFree(d_store.qa_head); cudaFree(d_store.qa_tail);
    cudaFree(d_store.qa_count); cudaFree(d_store.qa_pcs);
    cudaFree(d_store.qb_head); cudaFree(d_store.qb_tail);
    cudaFree(d_store.qb_count); cudaFree(d_store.qb_pcs);
    cudaFree(d_store.winners); cudaFree(d_store.cycles);
    cudaFree(d_warrior_a); cudaFree(d_warrior_b);
    cudaFree(d_pos_a); cudaFree(d_pos_b);

    return 0;
}
