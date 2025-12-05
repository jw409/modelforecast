# AI Warrior Generation System for CoreWars GPU

## Overview

This system enables LLMs to generate CoreWars warriors that compete in GPU-accelerated tournaments. Warriors are written in Redcode and battle in the MARS (Memory Array Redcode Simulator) at 37,000+ battles/second.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Warrior Generation                      │
│                                                              │
│  LLM (OpenRouter) ──> Redcode (.red) ──> GPU Tournament     │
│                            │                   │             │
│                            │                   │             │
│                            v                   v             │
│                    warrior_loader.cu    tournament.cu        │
│                            │                   │             │
│                            └───────┬───────────┘             │
│                                    │                         │
│                                    v                         │
│                            MARS GPU Kernel                   │
│                           (37K battles/sec)                  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Warrior Loader (`src/warrior_loader.cu`)

Parses Redcode text files into GPU-compatible instruction format.

**Features:**
- Full ICWS '94 standard support
- Label resolution (both `label:` and `label OPCODE` syntax)
- Addressing mode parsing (`#`, `$`, `@`, `*`, `<`, `>`, `{`, `}`)
- Instruction modifiers (`.A`, `.B`, `.AB`, `.BA`, `.F`, `.X`, `.I`)
- Metadata extraction (`;name` and `;author` comments)

**API:**
```cuda
struct Warrior {
    char name[64];
    char author[64];
    Instruction code[MAX_WARRIOR_LENGTH];  // Max 100 instructions
    int length;
};

// Parse Redcode source text
__host__ bool parse_warrior(const char* source, Warrior* out);

// Load warrior from .red file
__host__ bool load_warrior_from_file(const char* filename, Warrior* out);

// Load warrior into MARS core
__device__ void load_warrior(Instruction* core, const Warrior* w, int start_pos);
```

**Usage:**
```bash
# Test warrior loader
./build/test_warrior_loader warriors/champions/dwarf.red
```

### 2. Tournament System (`src/tournament.cu`)

Round-robin tournament with GPU acceleration.

**Features:**
- Round-robin: Every warrior vs every other warrior
- Configurable rounds per match (default: 1000)
- Randomized starting positions each round
- Win/Loss/Tie tracking
- Points system (3 pts/win, 1 pt/tie, 0 pts/loss)
- Win rate calculation
- Sorted standings table

**API:**
```cuda
struct MatchResult {
    int warrior_a_wins;
    int warrior_b_wins;
    int ties;
    float avg_cycles;
};

struct TournamentResult {
    int warrior_id;
    int total_wins;
    int total_losses;
    int total_ties;
    float win_rate;
    int points;
};

// Run single match
__host__ void run_match(
    const Warrior* warrior_a,
    const Warrior* warrior_b,
    int num_rounds,
    MatchResult* result
);

// Run full tournament
__host__ void run_tournament(
    Warrior* warriors,
    int num_warriors,
    int rounds_per_match,
    TournamentResult* results
);
```

### 3. AI Prompt Template (`ai/warrior_prompt.md`)

Comprehensive prompt for LLM warrior generation.

**Includes:**
- Complete instruction set reference (ICWS '94)
- Addressing modes with examples
- Instruction modifiers
- Classic strategies (Imp, Dwarf, Scanner, Gemini, etc.)
- Strategy guidelines (offensive/defensive/advanced tactics)
- Anti-strategies for each approach
- Output format specification
- Evaluation criteria

**Key Strategies Covered:**
1. **Imp** - Self-replicating spreader
2. **Dwarf** - Memory bomber
3. **Gemini** - Multi-copy replicator
4. **Scanner/Vampire** - Active hunter
5. **Silk** - Core-clear from both directions

### 4. AI Integration (`ai/generate_warrior.py`)

Python script to generate warriors using OpenRouter API.

**Features:**
- OpenRouter integration (any LLM)
- Strategy hints for targeted generation
- Batch generation with diversity
- Automatic Redcode extraction from markdown
- File saving with sanitized names

**Usage:**
```bash
# Generate single warrior
python ai/generate_warrior.py \
  --model google/gemini-2.5-flash-lite-preview-09-2025 \
  --strategy "Create an aggressive bomber"

# Generate batch with diverse strategies
python ai/generate_warrior.py --batch --count 5 \
  --model deepseek/deepseek-chat-v3-0324

# Custom output directory
python ai/generate_warrior.py --batch --count 10 \
  --model x-ai/grok-4-fast \
  --output warriors/grok-batch-1
```

**API:**
```python
from ai.generate_warrior import WarriorGenerator

gen = WarriorGenerator()

# Generate single warrior
code = gen.generate_warrior(
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    strategy_hint="Focus on replication and spreading",
    temperature=0.8
)

# Save to file
path = gen.save_warrior(code, output_dir=Path("warriors/my_warriors"))
```

## Workflow

### Basic Workflow: Generate → Test → Tournament

```bash
# 1. Generate warriors
python ai/generate_warrior.py --batch --count 5 --model gemini-2.5-flash

# 2. Test individual warrior
./build/test_warrior_loader warriors/gemini-2.5-flash/warrior_1.red

# 3. Run tournament (requires integration with MARS kernel)
# (Future: ./build/tournament warriors/gemini-2.5-flash/*.red)
```

### Advanced Workflow: Model Comparison

```bash
# Generate warriors from multiple models
python ai/generate_warrior.py --batch --count 5 --model gemini-2.5-flash
python ai/generate_warrior.py --batch --count 5 --model deepseek/deepseek-chat-v3
python ai/generate_warrior.py --batch --count 5 --model x-ai/grok-4-fast

# Run cross-model tournament
# (Future: Mix warriors from different models and compare)
```

## Redcode Format

### Metadata
```redcode
;name WarriorName
;author AuthorName
; Strategy description
```

### Instructions
```redcode
label    OPCODE.MOD mode_a_value, mode_b_value
```

**Examples:**
```redcode
;name Imp
;author A. K. Dewdney
MOV.I $0, $1        ; Copy self forward

;name Dwarf
;author A. K. Dewdney
bomb    DAT.F #0, #0     ; The bomb
        ADD.AB #4, bomb   ; Increment target
        MOV.I bomb, @bomb ; Drop bomb
        JMP.B $-2         ; Loop
```

### Addressing Modes
- `#n` - Immediate (literal value)
- `$n` - Direct (relative address)
- `@n` - B-indirect (use B-field as pointer)
- `*n` - A-indirect (use A-field as pointer)
- `<n` - Pre-decrement B-indirect
- `>n` - Post-increment B-indirect
- `{n` - Pre-decrement A-indirect
- `}n` - Post-increment A-indirect

### Instruction Modifiers
- `.A` - A-field only
- `.B` - B-field only
- `.AB` - A to B
- `.BA` - B to A
- `.F` - Full (both fields)
- `.X` - Exchange
- `.I` - Entire instruction

## Performance

- **Parser**: Handles warriors up to 100 instructions
- **Tournament**: GPU-accelerated, 37,000+ battles/second
- **Generation**: ~2-5 seconds per warrior (LLM-dependent)
- **Batch Generation**: Parallel API calls possible

## File Structure

```
corewars/
├── src/
│   ├── warrior_loader.cu          # Redcode parser
│   ├── tournament.cu               # Tournament system
│   ├── test_warrior_loader.cu      # Parser test program
│   ├── mars_kernel.cu              # MARS execution kernel
│   └── mars.h                      # Core data structures
├── ai/
│   ├── warrior_prompt.md           # LLM generation prompt
│   └── generate_warrior.py         # Generation script
├── warriors/
│   ├── champions/                  # Classic warriors
│   │   ├── dwarf.red
│   │   ├── imp_gate.red
│   │   └── mice.red
│   ├── gemini-2.5-flash/          # AI-generated (example)
│   ├── deepseek/                   # AI-generated (example)
│   └── grok-free/                  # AI-generated (example)
└── build/
    ├── test_warrior_loader         # Parser test
    └── gpu_mars_aos_optimized      # MARS simulator
```

## Integration with Existing System

The AI warrior system integrates with the existing GPU MARS simulator:

1. **Warrior Loading**: Replace hardcoded warriors in `mars_main.cu` with loaded warriors
2. **Tournament Mode**: Use `tournament.cu` instead of single match mode
3. **Results Tracking**: Export tournament results to JSON/CSV for analysis

### Example Integration

```cuda
// In main program
#include "warrior_loader.cu"
#include "tournament.cu"

int main() {
    // Load AI-generated warriors
    Warrior warriors[5];
    load_warrior_from_file("warriors/gemini/warrior_1.red", &warriors[0]);
    load_warrior_from_file("warriors/gemini/warrior_2.red", &warriors[1]);
    load_warrior_from_file("warriors/deepseek/warrior_1.red", &warriors[2]);
    load_warrior_from_file("warriors/grok/warrior_1.red", &warriors[3]);
    load_warrior_from_file("warriors/champions/dwarf.red", &warriors[4]);

    // Run tournament
    TournamentResult results[5];
    run_tournament(warriors, 5, 1000, results);

    return 0;
}
```

## Environment Variables

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Recommended Models

### Fast & Free
- `google/gemini-2.5-flash-lite-preview-09-2025:free`
- `x-ai/grok-4-fast:free`
- `qwen/qwen3-32b:free`

### Creative
- `deepseek/deepseek-chat-v3-0324`
- `nvidia/llama-3.3-nemotron-super-49b-v1.5`

### Premium
- `x-ai/grok-4`
- `deepseek/deepseek-v3.1-terminus`

## Future Enhancements

1. **Evolutionary Training**: Use genetic algorithms to evolve warriors
2. **Strategy Classification**: Automatically classify warrior strategies
3. **Benchmark Suite**: Standard opponents for evaluation
4. **Live Visualization**: Web-based tournament viewer
5. **Multi-round Evolution**: Winners generate improved versions
6. **Strategy Diversity Scoring**: Reward novel approaches
7. **Hybrid Generation**: LLM + evolutionary search

## Testing

```bash
# Build all components
make

# Test warrior parser
./build/test_warrior_loader warriors/champions/dwarf.red

# Generate test warrior
python ai/generate_warrior.py --model gemini-2.5-flash --strategy "bomber"

# Run single battle (existing)
./build/gpu_mars_aos_optimized

# Future: Run tournament
# ./build/tournament warriors/*.red
```

## References

- ICWS '94 Standard: http://corewar.co.uk/icws94.txt
- CoreWar Documentation: http://corewar.co.uk
- Redcode Tutorial: http://corewar.co.uk/tutorial.htm
- Strategy Guide: http://corewar.co.uk/strategy.htm
