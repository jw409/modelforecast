# CoreWars

**CoreWars** is a programming game where assembly programs ("warriors") fight for control of a shared memory space called the "core."

## The Rules

- Two warriors are loaded into random positions in an 8000-cell memory space
- Each warrior executes one instruction per turn, alternating
- A warrior dies when it executes an invalid instruction (like `DAT`)
- Last warrior standing wins; if both survive 80,000 cycles, it's a tie

## The Language: Redcode

Warriors are written in **Redcode**, a simple assembly language standardized as ICWS'94.

### Key Instructions

| Opcode | Name | Effect |
|--------|------|--------|
| `MOV` | Move | Copy data from source to destination |
| `ADD` | Add | Add source to destination |
| `SUB` | Subtract | Subtract source from destination |
| `JMP` | Jump | Continue execution at target address |
| `JMZ` | Jump if Zero | Jump if target is zero |
| `DJN` | Decrement and Jump if Not zero | Decrement B-field, jump if non-zero |
| `SPL` | Split | Create a new process at target address |
| `DAT` | Data | Kills any process that executes it |

### Addressing Modes

| Symbol | Mode | Meaning |
|--------|------|---------|
| `#` | Immediate | Use value directly |
| `$` | Direct | Use value at address (default) |
| `@` | Indirect | Use address stored at address |
| `<` | Pre-decrement indirect | Decrement, then use as indirect |
| `>` | Post-increment indirect | Use as indirect, then increment |

## Classic Strategies

### The IMP
```asm
MOV 0, 1    ; Copy yourself one cell forward, forever
```
Simplest warrior. Unstoppable but can't kill anything.

### The Dwarf
```asm
bomb    DAT #0, #0
start   ADD #4, ptr
ptr     MOV bomb, @ptr
        JMP start
```
Drops `DAT` bombs every 4 cells. Simple but effective.

### The Scanner
```asm
scan    ADD #10, ptr
ptr     JMZ scan, 100    ; If empty, keep scanning
        MOV bomb, @ptr   ; Found something! Bomb it
        JMP scan
bomb    DAT #0, #0
```
Searches for enemy code, then destroys it.

### The Replicator
```asm
start   MOV >0, >copy
        JMN start, copy
        SPL @copy
        JMP start
copy    DAT #0, #100
```
Copies itself across memory. Hard to kill completely.

## Running Without GPU

If you don't have an NVIDIA GPU, use **pMARS** (portable MARS):

```bash
# Install pMARS
git clone https://corewar.co.uk/pmars.git
cd pmars && make

# Run a battle (1000 rounds)
./pmars -r 1000 warrior1.red warrior2.red

# Run tournament (all vs all)
./pmars -r 1000 warriors/*.red
```

**Performance**: ~27,000 battles/sec on Intel i9-14900K (single-threaded)

## Running With GPU

With NVIDIA GPU and CUDA toolkit:

```bash
cd games/corewars
make
./gpu_mars_tournament -r 1000000 warriors/*.red
```

**Performance**: ~262,000 battles/sec on RTX 5090 (batched)

## Benchmark: LLM Warrior Generation

We gave 5 LLMs the ICWS'94 specification and asked them to write ONE warrior. No iteration, no feedback, just raw code generation.

| Model | Win Rate | Strategy |
|-------|----------|----------|
| Claude Sonnet 4.5 | 94.2% | Multi-process scanner + bomber + replicator |
| KwaiPilot KAT Coder Pro | 66.2% | Bomber with protection gate |
| DeepSeek V3 | 52.5% | Scanner/replicator hybrid |
| GPT-5.1 | 41.6% | Mod-4 bomber with imp-gate |
| Claude Opus 4.5 | 5.6% | Rolling stone (3 lines) |

## Resources

- [ICWS'94 Standard](http://corewar.co.uk/icws94.txt) - The official Redcode specification
- [pMARS](https://corewar.co.uk/pmars/) - Portable MARS simulator
- [CoreWar Wiki](http://corewar.co.uk/) - Strategy guides and warrior archives
