#ifndef WARRIOR_LOADER_CU
#define WARRIOR_LOADER_CU

#include "mars.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_WARRIOR_LENGTH 100
#define MAX_LINE_LENGTH 256

struct Warrior {
    char name[64];
    char author[64];
    Instruction code[MAX_WARRIOR_LENGTH];
    int length;
};

// Helper: Skip whitespace
static const char* skip_whitespace(const char* s) {
    while (*s && isspace(*s)) s++;
    return s;
}

// Helper: Parse addressing mode
static bool parse_mode(char c, uint8_t* mode) {
    switch(c) {
        case '#': *mode = IMMEDIATE; return true;
        case '$': *mode = DIRECT; return true;
        case '*': *mode = INDIRECT_A; return true;
        case '@': *mode = INDIRECT_B; return true;
        case '{': *mode = PREDEC_A; return true;
        case '<': *mode = PREDEC_B; return true;
        case '}': *mode = POSTINC_A; return true;
        case '>': *mode = POSTINC_B; return true;
        default: return false;
    }
}

// Helper: Parse opcode
static bool parse_opcode(const char* token, uint8_t* opcode) {
    if (strncasecmp(token, "DAT", 3) == 0) { *opcode = DAT; return true; }
    if (strncasecmp(token, "MOV", 3) == 0) { *opcode = MOV; return true; }
    if (strncasecmp(token, "ADD", 3) == 0) { *opcode = ADD; return true; }
    if (strncasecmp(token, "SUB", 3) == 0) { *opcode = SUB; return true; }
    if (strncasecmp(token, "MUL", 3) == 0) { *opcode = MUL; return true; }
    if (strncasecmp(token, "DIV", 3) == 0) { *opcode = DIV; return true; }
    if (strncasecmp(token, "MOD", 3) == 0) { *opcode = MOD; return true; }
    if (strncasecmp(token, "JMP", 3) == 0) { *opcode = JMP; return true; }
    if (strncasecmp(token, "JMZ", 3) == 0) { *opcode = JMZ; return true; }
    if (strncasecmp(token, "JMN", 3) == 0) { *opcode = JMN; return true; }
    if (strncasecmp(token, "DJN", 3) == 0) { *opcode = DJN; return true; }
    if (strncasecmp(token, "SPL", 3) == 0) { *opcode = SPL; return true; }
    if (strncasecmp(token, "SLT", 3) == 0) { *opcode = SLT; return true; }
    if (strncasecmp(token, "CMP", 3) == 0) { *opcode = CMP; return true; }
    if (strncasecmp(token, "SEQ", 3) == 0) { *opcode = SEQ; return true; }
    if (strncasecmp(token, "SNE", 3) == 0) { *opcode = SNE; return true; }
    if (strncasecmp(token, "NOP", 3) == 0) { *opcode = NOP; return true; }
    return false;
}

// Helper: Parse modifier
static bool parse_modifier(const char* token, uint8_t* modifier) {
    // Look for .X suffix after opcode
    const char* dot = strchr(token, '.');
    if (!dot) {
        *modifier = I; // Default modifier
        return true;
    }

    dot++; // Skip the dot
    if (*dot == 'A' || *dot == 'a') { *modifier = A; return true; }
    if (*dot == 'B' || *dot == 'b') { *modifier = B; return true; }
    if (*dot == 'F' || *dot == 'f') { *modifier = F; return true; }
    if (*dot == 'X' || *dot == 'x') { *modifier = X; return true; }
    if (*dot == 'I' || *dot == 'i') { *modifier = I; return true; }
    if (strncasecmp(dot, "AB", 2) == 0) { *modifier = AB; return true; }
    if (strncasecmp(dot, "BA", 2) == 0) { *modifier = BA; return true; }

    return false;
}

// Helper: Parse field (e.g., "#5", "@-3", "bomb")
static bool parse_field(const char* field_str, uint8_t* mode, int16_t* value,
                       const char** labels, const int16_t* label_addrs, int num_labels, int current_pc) {
    field_str = skip_whitespace(field_str);

    // Parse addressing mode
    if (parse_mode(*field_str, mode)) {
        field_str++; // Skip mode character
    } else {
        *mode = DIRECT; // Default to direct addressing
    }

    // Check for label reference
    if (isalpha(*field_str)) {
        // Label reference
        char label_name[64];
        int i = 0;
        while (isalnum(*field_str) || *field_str == '_') {
            label_name[i++] = *field_str++;
            if (i >= 63) break;
        }
        label_name[i] = '\0';

        // Find label
        for (int j = 0; j < num_labels; j++) {
            if (strcmp(labels[j], label_name) == 0) {
                // Calculate relative offset
                *value = label_addrs[j] - current_pc;
                return true;
            }
        }

        fprintf(stderr, "Unknown label: %s\n", label_name);
        return false;
    }

    // Parse numeric value
    *value = (int16_t)atoi(field_str);
    return true;
}

// Parse Redcode source into Warrior structure
__host__ bool parse_warrior(const char* source, Warrior* out) {
    memset(out, 0, sizeof(Warrior));
    strcpy(out->name, "Unnamed");
    strcpy(out->author, "Unknown");

    // First pass: Extract metadata and labels
    const char* labels[MAX_WARRIOR_LENGTH];
    int16_t label_addrs[MAX_WARRIOR_LENGTH];
    int num_labels = 0;

    char line_buf[MAX_LINE_LENGTH];
    const char* ptr = source;
    int pc = 0;

    // First pass: Find labels
    while (*ptr) {
        // Copy line
        int i = 0;
        while (*ptr && *ptr != '\n' && i < MAX_LINE_LENGTH - 1) {
            line_buf[i++] = *ptr++;
        }
        line_buf[i] = '\0';
        if (*ptr == '\n') ptr++;

        // Parse line
        const char* line = skip_whitespace(line_buf);

        // Skip comments and empty lines
        if (*line == ';' || *line == '\0') {
            // Check for metadata
            if (*line == ';') {
                line++;
                line = skip_whitespace(line);
                if (strncasecmp(line, "name ", 5) == 0) {
                    strncpy(out->name, skip_whitespace(line + 5), 63);
                    out->name[63] = '\0';
                } else if (strncasecmp(line, "author ", 7) == 0) {
                    strncpy(out->author, skip_whitespace(line + 7), 63);
                    out->author[63] = '\0';
                }
            }
            continue;
        }

        // Check for label (either "label:" or "label OPCODE")
        char token[64];
        i = 0;
        const char* peek = line;

        // Scan first token
        while (*peek && !isspace(*peek) && i < 63) {
            token[i++] = *peek++;
        }
        token[i] = '\0';

        bool is_label = false;

        // Check if it's a label (ends with ':' or followed by valid opcode)
        if (i > 0 && token[i-1] == ':') {
            // Explicit label with colon
            token[i-1] = '\0';
            is_label = true;
            line = peek; // Advance past label
        } else {
            // Check if next token is a valid opcode (label without colon)
            const char* next_token_start = skip_whitespace(peek);
            char next_token[64];
            int j = 0;
            while (*next_token_start && !isspace(*next_token_start) && j < 63) {
                next_token[j++] = *next_token_start++;
            }
            next_token[j] = '\0';

            uint8_t dummy_opcode;
            if (parse_opcode(next_token, &dummy_opcode)) {
                // First token is a label, second is opcode
                is_label = true;
                line = skip_whitespace(peek); // Advance to opcode
            }
        }

        if (is_label) {
            labels[num_labels] = strdup(token);
            label_addrs[num_labels] = pc;
            num_labels++;
        }

        // Check if there's an instruction on this line
        if (*line && *line != '\0' && *line != ';') {
            pc++;
        }
    }

    // Second pass: Parse instructions
    ptr = source;
    pc = 0;

    while (*ptr && pc < MAX_WARRIOR_LENGTH) {
        // Copy line
        int i = 0;
        while (*ptr && *ptr != '\n' && i < MAX_LINE_LENGTH - 1) {
            line_buf[i++] = *ptr++;
        }
        line_buf[i] = '\0';
        if (*ptr == '\n') ptr++;

        const char* line = skip_whitespace(line_buf);

        // Skip comments, metadata, and empty lines
        if (*line == ';' || *line == '\0') continue;

        // Handle labels (same logic as first pass)
        char token[64];
        i = 0;
        const char* peek = line;

        // Scan first token
        while (*peek && !isspace(*peek) && i < 63) {
            token[i++] = *peek++;
        }
        token[i] = '\0';

        bool is_label = false;

        // Check if it's a label
        if (i > 0 && token[i-1] == ':') {
            // Explicit label with colon
            is_label = true;
            line = skip_whitespace(peek); // Advance past label
            if (*line == '\0' || *line == ';') continue; // Label only line
        } else {
            // Check if next token is a valid opcode (label without colon)
            const char* next_token_start = skip_whitespace(peek);
            char next_token[64];
            int j = 0;
            while (*next_token_start && !isspace(*next_token_start) && j < 63) {
                next_token[j++] = *next_token_start++;
            }
            next_token[j] = '\0';

            uint8_t dummy_opcode;
            if (parse_opcode(next_token, &dummy_opcode)) {
                // First token is a label, second is opcode
                is_label = true;
                line = skip_whitespace(peek); // Advance to opcode
            }
        }

        // Parse opcode
        i = 0;
        while (*line && !isspace(*line) && i < 63) {
            token[i++] = *line++;
        }
        token[i] = '\0';

        uint8_t opcode, modifier;
        if (!parse_opcode(token, &opcode)) {
            fprintf(stderr, "Unknown opcode: %s\n", token);
            return false;
        }

        if (!parse_modifier(token, &modifier)) {
            fprintf(stderr, "Invalid modifier: %s\n", token);
            return false;
        }

        // Parse A-field
        line = skip_whitespace(line);
        char a_field_str[64] = "";
        i = 0;
        while (*line && *line != ',' && *line != '\n' && i < 63) {
            a_field_str[i++] = *line++;
        }
        a_field_str[i] = '\0';

        uint8_t mode_a = DIRECT;
        int16_t a_value = 0;

        if (strlen(a_field_str) > 0) {
            if (!parse_field(a_field_str, &mode_a, &a_value, labels, label_addrs, num_labels, pc)) {
                return false;
            }
        }

        // Parse B-field (optional)
        uint8_t mode_b = DIRECT;
        int16_t b_value = 0;

        if (*line == ',') {
            line++;
            line = skip_whitespace(line);

            char b_field_str[64] = "";
            i = 0;
            while (*line && *line != '\n' && i < 63) {
                b_field_str[i++] = *line++;
            }
            b_field_str[i] = '\0';

            if (strlen(b_field_str) > 0) {
                if (!parse_field(b_field_str, &mode_b, &b_value, labels, label_addrs, num_labels, pc)) {
                    return false;
                }
            }
        }

        // Store instruction
        out->code[pc].opcode = opcode;
        out->code[pc].modifier = modifier;
        out->code[pc].mode_a = mode_a;
        out->code[pc].mode_b = mode_b;
        out->code[pc].a_field = a_value;
        out->code[pc].b_field = b_value;

        pc++;
    }

    out->length = pc;

    // Cleanup labels
    for (int i = 0; i < num_labels; i++) {
        free((void*)labels[i]);
    }

    return out->length > 0;
}

// Load warrior into MARS core at specified position
__device__ void load_warrior(Instruction* core, const Warrior* w, int start_pos) {
    for (int i = 0; i < w->length; i++) {
        int addr = (start_pos + i) % CORE_SIZE;
        core[addr] = w->code[i];
    }
}

// Host helper: Load warrior from file
__host__ bool load_warrior_from_file(const char* filename, Warrior* out) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return false;
    }

    // Read entire file
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, f);
    source[size] = '\0';
    fclose(f);

    bool result = parse_warrior(source, out);
    free(source);

    if (result) {
        printf("Loaded warrior: %s by %s (%d instructions)\n",
               out->name, out->author, out->length);
    }

    return result;
}

#endif // WARRIOR_LOADER_CU
