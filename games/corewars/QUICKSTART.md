# CoreWars AI Warrior System - Quick Start Guide

## Prerequisites

1. CUDA-capable GPU (tested on RTX 5090)
2. NVCC compiler
3. Python 3.8+
4. OpenRouter API key

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Installation

```bash
cd /home/jw/dev/modelforecast/games/corewars

# Build all components
make

# Verify build
ls -la build/
```

## 5-Minute Tutorial

### Step 1: Test Existing Warriors

```bash
# Test the warrior parser
./build/test_warrior_loader warriors/champions/dwarf.red
./build/test_warrior_loader warriors/champions/imp_gate.red
```

**Output:**
```
=== WARRIOR LOADED ===
Name: Unnamed
Author: Unknown
Length: 4 instructions

Code:
[ 0]   DAT.I  #0   , $0
[ 1]   ADD.I  #4   , $-1
[ 2]   MOV.I  $-2  , @-2
[ 3]   JMP.I  $-2  , $0
```

### Step 2: Generate Your First AI Warrior

```bash
# Generate a single warrior
python ai/generate_warrior.py \
  --model google/gemini-2.5-flash-lite-preview-09-2025 \
  --strategy "Create an aggressive bomber that clears memory systematically"
```

**Output:**
```
Generating warrior with google/gemini-2.5-flash-lite-preview-09-2025...
Strategy hint: Create an aggressive bomber that clears memory systematically
Saved warrior to: warriors/gemini-2.5-flash-lite-preview-09-2025/warrior.red
✓ Warrior saved to: warriors/gemini-2.5-flash-lite-preview-09-2025/warrior.red
```

### Step 3: Test Your AI Warrior

```bash
./build/test_warrior_loader warriors/gemini-2.5-flash-lite-preview-09-2025/warrior.red
```

### Step 4: Generate a Batch of Warriors

```bash
# Generate 5 diverse warriors
python ai/generate_warrior.py --batch --count 5 \
  --model google/gemini-2.5-flash-lite-preview-09-2025
```

**Output:**
```
=== Generating warrior 1/5 ===
Generating warrior with google/gemini-2.5-flash-lite-preview-09-2025...
Strategy hint: Create an aggressive bomber that clears memory quickly
Saved warrior to: warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_1.red
✓ Generated warrior 1

=== Generating warrior 2/5 ===
...

=== Generated 5 warriors ===
  - warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_1.red
  - warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_2.red
  - warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_3.red
  - warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_4.red
  - warriors/gemini-2.5-flash-lite-preview-09-2025/warrior_5.red
```

### Step 5: Run Existing GPU Battles (Current System)

```bash
# Run single battle between hardcoded warriors
./build/gpu_mars_aos_optimized 10000
```

**Output:**
```
Launching kernel: 40 blocks, 256 threads
Kernel execution time: 0.2701 s
Throughput: 37020.36 battles/sec
Battle 0 Winner: 2 (Cycles: 80000)
```

## Common Workflows

### Workflow 1: Model Comparison

Compare different LLMs by generating warriors from each:

```bash
# Generate from multiple models
python ai/generate_warrior.py --batch --count 5 --model google/gemini-2.5-flash-lite-preview-09-2025
python ai/generate_warrior.py --batch --count 5 --model deepseek/deepseek-chat-v3-0324
python ai/generate_warrior.py --batch --count 5 --model x-ai/grok-4-fast

# Inspect results
ls -la warriors/*/
```

### Workflow 2: Strategy Exploration

Generate warriors with different strategic hints:

```bash
# Bomber strategy
python ai/generate_warrior.py \
  --strategy "Create a bomber that systematically clears memory with DAT bombs"

# Scanner strategy
python ai/generate_warrior.py \
  --strategy "Create a scanner that searches for enemies before attacking"

# Replicator strategy
python ai/generate_warrior.py \
  --strategy "Create a self-replicating warrior that spreads copies across memory"

# Hybrid strategy
python ai/generate_warrior.py \
  --strategy "Create a hybrid combining imp spreading with aggressive bombing"
```

### Workflow 3: Iterative Improvement

```bash
# Generate initial batch
python ai/generate_warrior.py --batch --count 10 --model gemini-2.5-flash

# Test each warrior (manual inspection)
for f in warriors/gemini-2.5-flash/*.red; do
    echo "=== $f ==="
    ./build/test_warrior_loader "$f"
    echo ""
done

# Identify best strategy, generate more variations
python ai/generate_warrior.py \
  --strategy "Improve on this strategy: [paste best warrior]"
```

## Advanced Usage

### Custom Temperature for Creativity

```python
from ai.generate_warrior import WarriorGenerator
from pathlib import Path

gen = WarriorGenerator()

# Low temperature (conservative, reliable)
code = gen.generate_warrior(
    model="google/gemini-2.5-flash-lite-preview-09-2025",
    temperature=0.3
)

# High temperature (creative, experimental)
code = gen.generate_warrior(
    model="deepseek/deepseek-chat-v3-0324",
    temperature=1.5
)

gen.save_warrior(code, Path("warriors/experimental"))
```

### Batch Generation with Custom Strategies

```python
from ai.generate_warrior import generate_warrior_batch
from pathlib import Path

strategies = [
    "Create an imp that spreads while avoiding detection",
    "Create a vampire that converts enemy processes",
    "Create a stone warrior combining imp and bomber",
    "Create a paper replicator with decoy copies",
    "Create a scanner with adaptive bombing patterns"
]

warriors = generate_warrior_batch(
    num_warriors=5,
    model="x-ai/grok-4-fast",
    output_dir=Path("warriors/custom-batch"),
    strategies=strategies
)

print(f"Generated {len(warriors)} warriors")
```

## Troubleshooting

### Parser Errors

**Problem:** "Unknown opcode: XXX"
```bash
./build/test_warrior_loader my_warrior.red
# Output: Unknown opcode: XXX
```

**Solution:** Check that opcode is valid ICWS '94:
- Valid: DAT, MOV, ADD, SUB, MUL, DIV, MOD, JMP, JMZ, JMN, DJN, SPL, SLT, CMP, SEQ, SNE, NOP
- Case-insensitive

**Problem:** "Unknown label: YYY"

**Solution:** Ensure labels are defined before use:
```redcode
; WRONG
JMP bomb      ; bomb not defined yet
bomb DAT #0

; CORRECT
bomb DAT #0
JMP bomb      ; bomb already defined
```

### Generation Errors

**Problem:** API error 401/403

**Solution:** Check OPENROUTER_API_KEY is set:
```bash
echo $OPENROUTER_API_KEY
export OPENROUTER_API_KEY="your_key_here"
```

**Problem:** Generated code is not valid Redcode

**Solution:** Try different model or adjust prompt:
```bash
# More reliable models
--model google/gemini-2.5-flash-lite-preview-09-2025
--model deepseek/deepseek-chat-v3-0324

# Lower temperature for more conservative output
# (Edit generate_warrior.py or use API directly)
```

### Build Errors

**Problem:** nvcc: command not found

**Solution:** Install CUDA toolkit and add to PATH

**Problem:** Architecture mismatch (sm_120)

**Solution:** Edit Makefile to match your GPU:
```makefile
# RTX 5090
NVCC_FLAGS = -O3 -arch=sm_90

# RTX 4090
NVCC_FLAGS = -O3 -arch=sm_89

# RTX 3090
NVCC_FLAGS = -O3 -arch=sm_86
```

## File Organization

Recommended directory structure:

```
corewars/
├── warriors/
│   ├── champions/              # Classic reference warriors
│   ├── gemini-2.5-flash/       # Gemini-generated
│   ├── deepseek/               # DeepSeek-generated
│   ├── grok-free/              # Grok-generated
│   ├── experiments/            # Your experiments
│   └── tournament-winners/     # Best performers
```

## Next Steps

1. **Build Tournament System**: Integrate `tournament.cu` with MARS kernel
2. **Automate Evaluation**: Script to generate → test → rank warriors
3. **Visualize Results**: Export tournament data to JSON/CSV
4. **Evolutionary Improvement**: Use tournament results to guide generation
5. **Strategy Analysis**: Classify and categorize generated strategies

## Resources

- **Full Documentation**: `AI_WARRIOR_SYSTEM.md`
- **Prompt Template**: `ai/warrior_prompt.md`
- **ICWS Standard**: http://corewar.co.uk/icws94.txt
- **Strategy Guide**: http://corewar.co.uk/strategy.htm

## Example Session

Complete workflow from zero to tournament-ready:

```bash
# 1. Setup
export OPENROUTER_API_KEY="sk-or-v1-..."
cd /home/jw/dev/modelforecast/games/corewars
make

# 2. Generate warriors from 3 different models
python ai/generate_warrior.py --batch --count 5 --model gemini-2.5-flash
python ai/generate_warrior.py --batch --count 5 --model deepseek/deepseek-chat-v3
python ai/generate_warrior.py --batch --count 5 --model x-ai/grok-4-fast

# 3. Test loading
./build/test_warrior_loader warriors/gemini-2.5-flash/warrior_1.red

# 4. Run GPU battles (existing system)
./build/gpu_mars_aos_optimized 50000

# 5. Future: Run tournament with AI warriors
# ./build/tournament warriors/gemini-2.5-flash/ 1000
```

## Performance Expectations

- **Warrior Generation**: 2-5 seconds per warrior (LLM-dependent)
- **Parsing**: <1ms per warrior
- **GPU Battles**: 37,000+ battles/second
- **Tournament (5 warriors, 1000 rounds/match)**: ~1-2 seconds

## Tips for Best Results

1. **Be Specific in Strategy Hints**: "bomber" vs "Create a bomber that uses exponential spacing"
2. **Vary Temperature**: 0.5-0.8 for reliable code, 0.9-1.5 for creative experiments
3. **Test Incrementally**: Parse → Inspect → Battle
4. **Compare Models**: Different LLMs excel at different strategies
5. **Iterate on Winners**: Use successful warriors as examples for next generation
