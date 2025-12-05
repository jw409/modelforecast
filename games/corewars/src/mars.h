#ifndef MARS_H
#define MARS_H

#include <stdint.h>

#define CORE_SIZE 8000
#define MAX_CYCLES 80000
#define MAX_PROCESSES 8000

// OpCodes (ICWS '94 Standard)
enum OpCode {
    DAT = 0, MOV, ADD, SUB, MUL, DIV, MOD, JMP, JMZ, JMN, DJN, SPL, SLT, CMP, SEQ, SNE, LDP, STP, NOP
};

// Modifiers
enum Modifier {
    A = 0, B, AB, BA, F, X, I
};

// Addressing Modes
enum Mode {
    IMMEDIATE = 0,  // #
    DIRECT,         // $
    INDIRECT_A,     // *
    INDIRECT_B,     // @
    PREDEC_A,       // {
    PREDEC_B,       // <
    POSTINC_A,      // }
    POSTINC_B       // >
};

struct Instruction {
    uint8_t opcode;     // 5 bits sufficient
    uint8_t modifier;   // 3 bits
    uint8_t mode_a;     // 3 bits
    uint8_t mode_b;     // 3 bits
    int16_t a_field;
    int16_t b_field;
};

struct ProcessQueue {
    uint16_t head;
    uint16_t tail;
    uint16_t count;
    uint16_t pcs[MAX_PROCESSES];
};

struct BattleState {
    Instruction core[CORE_SIZE];
    ProcessQueue queue_a;
    ProcessQueue queue_b;
    uint32_t cycles;
    uint8_t winner; // 0=Tie, 1=A, 2=B
};

// Helper for initialization
struct WarriorData {
    Instruction* code;
    uint16_t length;
    uint32_t start_pos;
};

#endif // MARS_H
