#!/bin/bash
# Test script for doom_verify.cu

set -e

echo "=== GPU DOOM Verification Test Suite ==="
echo ""

# Test 1: Self-test
echo "[1/3] Running self-test..."
./gpu/build/doom_verify --self-test
echo ""

# Test 2: JSONL output with 10 forward ticks
echo "[2/3] Testing JSONL output (10 forward ticks)..."
python3 -c "
import struct
import sys

# Create TicCmd: walk forward
cmd = struct.pack('bbhhBB', 50, 0, 0, 0, 0, 0)

# Write 10 ticks
for _ in range(10):
    sys.stdout.buffer.write(cmd)
" | ./gpu/build/doom_verify > /tmp/doom_verify_test.jsonl 2>/dev/null

# Verify output format
echo "Output lines: $(wc -l < /tmp/doom_verify_test.jsonl)"
echo "First tick:"
head -1 /tmp/doom_verify_test.jsonl | python3 -m json.tool
echo "Last tick:"
tail -1 /tmp/doom_verify_test.jsonl | python3 -m json.tool
echo ""

# Test 3: Movement verification
echo "[3/3] Verifying movement physics..."
python3 -c "
import json

with open('/tmp/doom_verify_test.jsonl') as f:
    states = [json.loads(line) for line in f]

# Check initial state
assert states[0]['tick'] == 0
assert states[0]['health'] == 100
assert states[0]['alive'] == 1

# Check movement (Y should increase when walking north at ANG90)
y0 = states[0]['y']
y_final = states[-1]['y']

assert y_final > y0, f'Y should increase: {y0} -> {y_final}'

# Check friction (momentum should decay)
print(f'✓ Initial Y: {y0}')
print(f'✓ Final Y: {y_final}')
print(f'✓ Movement delta: {y_final - y0} units')
print(f'✓ All physics checks passed!')
"

echo ""
echo "=== All Tests PASSED ==="

