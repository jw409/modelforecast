#!/usr/bin/env python3
"""Generate results table from JSON probe files."""

import json
from pathlib import Path
from collections import defaultdict

results = defaultdict(dict)
for f in Path('results').glob('**/*__level_*.json'):
    if 'embedding' in f.name:
        continue
    parts = f.stem.rsplit('__level_', 1)
    if len(parts) != 2:
        continue
    model, level = parts
    level = int(level)
    try:
        data = json.loads(f.read_text())
        probes = data.get('probes', {})
        if isinstance(probes, dict) and 'trials' in probes:
            trials_list = probes['trials']
            n = len(trials_list)
            if n == 0:
                continue
            if level == 0:
                successes = sum(1 for t in trials_list if t.get('tool_called'))
            elif level == 1:
                successes = sum(1 for t in trials_list if t.get('schema_valid'))
            elif level == 2:
                successes = sum(1 for t in trials_list if t.get('tool_called') and t.get('schema_valid'))
            elif level == 3:
                successes = sum(1 for t in trials_list if t.get('tool_called'))
            elif level == 4:
                successes = sum(1 for t in trials_list if not t.get('tool_called'))
            else:
                continue
            rate = int(100 * successes / n) if n > 0 else 0
            results[model][level] = (rate, n)
    except Exception:
        pass

# Sort by T0 score descending
sorted_models = sorted(results.keys(), key=lambda m: (-results[m].get(0, (0,0))[0], m))

print('| Model | T0 | T1 | T2 | A1 | R0 | Grade |')
print('|-------|:--:|:--:|:--:|:--:|:--:|:-----:|')
for model in sorted_models:
    r = results[model]
    def fmt(lvl):
        if lvl not in r:
            return '-'
        return f'{r[lvl][0]}%'
    t0 = fmt(0)
    t1 = fmt(1)
    t2 = fmt(2)
    a1 = fmt(3)
    r0 = fmt(4)
    # Grade
    t0_val = r.get(0, (0,0))[0]
    t1_val = r.get(1, (100,0))[0]
    t2_val = r.get(2, (100,0))[0]
    a1_val = r.get(3, (100,0))[0]
    r0_val = r.get(4, (100,0))[0]

    grade = 'F'
    if t0_val >= 80 and t1_val >= 70 and min(t0_val, t1_val, t2_val, a1_val, r0_val) >= 50:
        grade = 'A'
    elif t0_val >= 60 and t1_val >= 50 and min(t0_val, t1_val, t2_val) >= 30:
        grade = 'B'
    elif t0_val >= 40:
        grade = 'C'
    elif t0_val >= 20:
        grade = 'D'

    name = model.replace('_', '/').replace('/free',':free')
    print(f'| {name} | {t0} | {t1} | {t2} | {a1} | {r0} | {grade} |')
