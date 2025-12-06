# Redcode Warrior Generation Prompt

You are competing in Core War, a programming game where two programs ("warriors") battle in shared memory. Write a warrior in Redcode (ICWS'94 standard).

## Core War Rules

- **Memory**: 8000 cells (circular, wraps around)
- **Execution**: Warriors take turns executing one instruction
- **Victory**: Kill opponent by making them execute DAT (illegal instruction)
- **Tie**: Both alive after 80,000 cycles

## Redcode Syntax

```
LABEL   OPCODE.MODIFIER  A-MODE A-VALUE, B-MODE B-VALUE
```

### Opcodes (15 total)
| Opcode | Description |
|--------|-------------|
| DAT | Data (kills process if executed) |
| MOV | Copy A to B |
| ADD | Add A to B |
| SUB | Subtract A from B |
| MUL | Multiply B by A |
| DIV | Divide B by A (process dies if A=0) |
| MOD | B modulo A (process dies if A=0) |
| JMP | Jump to A |
| JMZ | Jump to A if B is zero |
| JMN | Jump to A if B is non-zero |
| DJN | Decrement B, jump to A if B non-zero |
| SEQ/CMP | Skip next if A equals B |
| SNE | Skip next if A not equal B |
| SLT | Skip next if A less than B |
| SPL | Split: spawn new process at A |
| NOP | No operation |

### Addressing Modes
| Mode | Symbol | Meaning |
|------|--------|---------|
| Immediate | # | The value itself |
| Direct | $ (default) | Relative address |
| Indirect | @ | Address points to address |
| Predecrement | < | Decrement B-field, then indirect |

### Modifiers
`.A` `.B` `.AB` `.BA` `.F` `.X` `.I` (control which fields are affected)

## Example Warriors

### IMP (simplest replicator)
```redcode
;redcode
;name Imp
;author A. K. Dewdney
        MOV.I   0, 1
        END
```
Strategy: Copies itself forward forever. Hard to kill but can't kill.

### DWARF (simple bomber)
```redcode
;redcode
;name Dwarf
;author A. K. Dewdney
target  DAT.F   #0,     #0
start   ADD.AB  #4,     target
        MOV.AB  #0,     @target
        JMP.A   start
        END start
```
Strategy: Drops DAT bombs every 4 cells. Eventually hits enemy.

### Simple Scanner
```redcode
;redcode
;name Scanner
;author Example
scan    ADD.AB  #10,    ptr
ptr     JMZ.F   scan,   100
        MOV.I   bomb,   @ptr
        JMP.A   scan
bomb    DAT.F   #0,     #0
        END scan
```
Strategy: Scans memory for non-zero cells (likely enemy), then bombs them.

## Your Task

Write an original warrior that can defeat both IMP and DWARF. Be creative with your strategy.

**Output format:**
```redcode
;redcode
;name [Your warrior name]
;author [Model name]
;strategy [Brief description]

[Your code here]

        END [start label]
```

**Constraints:**
- Maximum 100 instructions
- Must be valid ICWS'94 syntax
- Include comments explaining your strategy
