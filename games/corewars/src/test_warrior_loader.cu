#include <stdio.h>
#include <stdlib.h>
#include "warrior_loader.cu"

void print_instruction(const Instruction* instr) {
    const char* opcodes[] = {"DAT", "MOV", "ADD", "SUB", "MUL", "DIV", "MOD",
                             "JMP", "JMZ", "JMN", "DJN", "SPL", "SLT", "CMP",
                             "SEQ", "SNE", "LDP", "STP", "NOP"};
    const char* modifiers[] = {"A", "B", "AB", "BA", "F", "X", "I"};
    const char* modes[] = {"#", "$", "*", "@", "{", "<", "}", ">"};

    printf("  %s.%-2s %s%-4d, %s%-4d\n",
           opcodes[instr->opcode],
           modifiers[instr->modifier],
           modes[instr->mode_a],
           instr->a_field,
           modes[instr->mode_b],
           instr->b_field);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <warrior.red>\n", argv[0]);
        return 1;
    }

    Warrior warrior;
    if (!load_warrior_from_file(argv[1], &warrior)) {
        fprintf(stderr, "Failed to load warrior from %s\n", argv[1]);
        return 1;
    }

    printf("\n=== WARRIOR LOADED ===\n");
    printf("Name: %s\n", warrior.name);
    printf("Author: %s\n", warrior.author);
    printf("Length: %d instructions\n\n", warrior.length);

    printf("Code:\n");
    for (int i = 0; i < warrior.length; i++) {
        printf("[%2d] ", i);
        print_instruction(&warrior.code[i]);
    }
    printf("\n");

    return 0;
}
