#!/bin/bash
# Run 25 million battles in batches (VRAM limited to ~300K per batch)

BATCHES=84
BATTLES_PER_BATCH=300000
TOTAL=$((BATCHES * BATTLES_PER_BATCH))

echo "=== 25 MILLION BATTLES (${BATCHES} batches x ${BATTLES_PER_BATCH}) ==="
echo "GPU: RTX 5090 (32GB VRAM)"
echo ""

total_a=0
total_b=0
total_t=0

start_time=$(date +%s.%N)

for i in $(seq 1 $BATCHES); do
  result=$(./build/gpu_mars_interleaved \
    warriors/tournament_001/anthropic_claude-sonnet-4.5.red \
    warriors/tournament_001/anthropic_claude-opus-4.5.red \
    $BATTLES_PER_BATCH --seed $i 2>&1 | grep "Results:")

  a=$(echo $result | awk '{print $2}')
  b=$(echo $result | awk '{print $3}')
  t=$(echo $result | awk '{print $4}')

  total_a=$((total_a + a))
  total_b=$((total_b + b))
  total_t=$((total_t + t))

  if [ $((i % 10)) -eq 0 ]; then
    echo "Batch $i/$BATCHES: Sonnet=$total_a Opus=$total_b Ties=$total_t"
  fi
done

end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
throughput=$(echo "scale=0; $TOTAL / $elapsed" | bc)

echo ""
echo "=== FINAL RESULTS (${TOTAL} battles) ==="
echo "Sonnet (Silk):   $total_a wins ($(echo "scale=1; $total_a * 100 / $TOTAL" | bc)%)"
echo "Opus (Granite):  $total_b wins ($(echo "scale=1; $total_b * 100 / $TOTAL" | bc)%)"
echo "Ties:            $total_t ($(echo "scale=2; $total_t * 100 / $TOTAL" | bc)%)"
echo ""
echo "Total time:      ${elapsed}s"
echo "Throughput:      ${throughput} battles/sec"
