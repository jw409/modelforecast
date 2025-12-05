# CoreWars Warrior Programming Challenge

You are an expert CoreWars warrior programmer. Your task is to write a Redcode program that will compete in the Memory Array Redcode Simulator (MARS).

## Arena Environment

- **Core Size**: 8000 memory addresses (0-7999, wraps around)
- **Max Cycles**: 80,000 execution cycles
- **Max Processes**: 8,000 per warrior
- **Initial Distance**: Warriors start ~4000 addresses apart

## Victory Conditions

1. **Win**: Eliminate all enemy processes (enemy executes DAT instruction)
2. **Loss**: All your processes eliminated
3. **Tie**: Time limit reached with both warriors still running

## Redcode Instruction Set (ICWS '94)

### Data/Movement Instructions

- `DAT a, b` - Data/bomb. Kills any process that executes it. Use for memory clearing.
- `MOV a, b` - Copy instruction from A to B. Core of most strategies.
- `NOP a, b` - No operation. Delays/synchronization.

### Arithmetic Instructions

- `ADD a, b` - Add A to B field of instruction at B
- `SUB a, b` - Subtract A from B
- `MUL a, b` - Multiply
- `DIV a, b` - Divide (division by zero kills process)
- `MOD a, b` - Modulo

### Control Flow Instructions

- `JMP a` - Jump to address A (unconditional)
- `JMZ a, b` - Jump to A if B-field is zero
- `JMN a, b` - Jump to A if B-field is non-zero
- `DJN a, b` - Decrement B, jump to A if non-zero (loop counter)

### Process Management

- `SPL a` - Split execution. Creates new process starting at A. Critical for multi-threading.

### Comparison Instructions

- `CMP/SEQ a, b` - Skip next if A equals B
- `SNE a, b` - Skip next if A not equal to B
- `SLT a, b` - Skip next if A less than B

## Addressing Modes

- `#n` - Immediate: Use value n directly
- `$n` or `n` - Direct: Address n relative to current position
- `@n` - B-field Indirect: Use B-field of instruction at n as address
- `*n` - A-field Indirect: Use A-field of instruction at n as address
- `<n` - Pre-decrement B-indirect: Decrement B-field at n, then use as address
- `>n` - Post-increment B-indirect: Use B-field at n as address, then increment
- `{n` - Pre-decrement A-indirect
- `}n` - Post-increment A-indirect

## Instruction Modifiers

Modifiers specify which fields to operate on (defaults shown):

- `.A` - Use A-fields only
- `.B` - Use B-fields only
- `.AB` - A-field to B-field
- `.BA` - B-field to A-field
- `.F` - Both fields (Full)
- `.X` - Exchange fields
- `.I` - Entire instruction

Default modifiers by opcode:
- DAT, NOP: `.F`
- MOV, SEQ, SNE, CMP: `.I`
- ADD, SUB, MUL, DIV, MOD: `.AB`
- JMP, JMZ, JMN, DJN, SPL, SLT: `.B`

## Classic Strategies

### 1. Imp (Simplest)
```redcode
;name Imp
;author A. K. Dewdney
MOV 0, 1    ; Copy self forward
```
**Pros**: Spreads through memory, hard to kill completely
**Cons**: Weak offense, easily defeated by active scanners

### 2. Dwarf (Bomber)
```redcode
;name Dwarf
;author A. K. Dewdney
ADD #4, 3      ; Increment bomb target
MOV 2, @2      ; Drop bomb at target
JMP -2         ; Loop
DAT #0, #0     ; The bomb
```
**Pros**: Blankets memory with DAT bombs, kills static code
**Cons**: Predictable pattern, vulnerable to imps

### 3. Gemini (Replicator)
```redcode
;name Gemini
;author Unknown
ptr  DAT    #0
     SPL    @ptr
     MOV    <ptr, @ptr
     ADD    #9, ptr
     JMZ    -2, @ptr
     JMP    -4
```
**Pros**: Creates multiple copies, resilient
**Cons**: Resource intensive, slow

### 4. Scanner/Vampire
```redcode
;name Scanner
scan SEQ #0, @scan_pos    ; Look for non-zero
     JMP attack            ; Found something!
     ADD #5, scan_pos      ; Increment search
     JMP scan              ; Keep scanning
scan_pos DAT #10, #0
attack   SPL 0             ; Split to attack
         MOV bomb, >scan_pos
         JMP -1
bomb     DAT #0, #0
```
**Pros**: Actively hunts enemies, efficient
**Cons**: More complex, code size overhead

### 5. Silk (Core-clear)
```redcode
;name Silk
SPL 1          ; Create parallel processes
SPL 1
SPL 1
clear MOV <src, <dst
      JMP clear
src   DAT #0, #-1
dst   DAT #0, #0
```
**Pros**: Fast memory clearing from both directions
**Cons**: Vulnerable during startup

## Strategy Guidelines

### Offensive Tactics
1. **Bombing**: Systematically overwrite memory with DAT
2. **Scanning**: Search for enemy code before attacking
3. **Core-clearing**: Quick sequential memory overwrites
4. **Imping**: Flood memory with self-replicating code

### Defensive Tactics
1. **Replication**: Create multiple copies of yourself
2. **Spacing**: Spread processes far apart
3. **Decoys**: Include dummy code that looks important
4. **Evasion**: Keep moving through memory

### Advanced Techniques
1. **Vampire**: Convert enemy processes to work for you (modify their code)
2. **Stone**: Combine imp with bomber
3. **Paper**: Fast replicator with imp components
4. **Scissors**: Scanner that adapts to enemy

### Anti-Strategies
- **Anti-Imp**: Use scanners with tight bombing patterns
- **Anti-Bomber**: Rapid replication before being hit
- **Anti-Scanner**: Minimize code footprint, use decoys

## Your Task

Write a Redcode warrior that:
1. Has a clear strategy (bomber/scanner/imp/replicator/hybrid)
2. Uses at least 3-5 different instruction types
3. Includes metadata (;name and ;author comments)
4. Is between 5-20 instructions (optimal size)
5. Can compete against the classic strategies above

## Output Format

Provide ONLY valid Redcode. Start with metadata comments, then instructions:

```redcode
;name YourWarriorName
;author YourName
; Strategy description here

; Your code here
MOV ...
ADD ...
```

## Evaluation Criteria

Your warrior will battle against:
- Imp (baseline)
- Dwarf (bomber)
- Scanner (active hunter)
- Gemini (replicator)
- Other AI-generated warriors

Success measured by:
- Win rate in round-robin tournament
- Survival time
- Process efficiency
- Code elegance

## Constraints

- Maximum 100 instructions
- No external dependencies
- Standard ICWS '94 only
- Must be complete, runnable code

**Now write your warrior. Be creative, be strategic, be deadly.**
