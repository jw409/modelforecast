# Borg Config-Based Behavior Implementation

## Summary

Implemented config-dependent AI decision-making for GPU Angband Borg simulation based on APWBorg (borg9.c, borg6.c, borg8.c) analysis.

## Changes Made

### 1. Config-Aware Danger Calculation (`calculate_danger`)

**Location**: `borg_kernel_v2.cu:28-77`

**Behavior**:
- **Risky borgs** (`CFG_PLAYS_RISKY`): Perceive 30% less danger (multiplier: 0.7)
  - From APWBorg `borg6.c:2092`: "Risky borgs are more likely to stay in a fight"

- **Tank borgs** (`CFG_WORSHIPS_HP` + `CFG_WORSHIPS_AC`): Perceive 30% more danger (multiplier: 1.3)
  - Conservative danger assessment leads to defensive play

### 2. Config-Dependent Thresholds

**Location**: `borg_kernel_v2.cu:238-277`

#### HP Thresholds by Config

| Config | Critical HP | Low HP | Rest HP | Descend HP |
|--------|------------|--------|---------|------------|
| **Default** | 20% | 50% | 80% | 70% |
| **Tank** (HP+AC) | 30% | 60% | 90% | 85% |
| **Aggro** (Risky+Damage) | 15% | 35% | 60% | 50% |

**Reasoning**:
- **Tank**: Panic earlier, heal more, rest more, require higher HP before descending
- **Aggro**: Fight longer at low HP, heal less, rest less, descend with riskier HP

#### Danger Thresholds by Config

| Config | Flee Threshold | Fight Threshold | Descend Threshold |
|--------|---------------|-----------------|-------------------|
| **Default** | 300 | 200 | 50 |
| **Risky** | 450 (+50%) | 300 | 100 |
| **Tank** | 200 (-33%) | 150 | 30 |

**Reasoning**:
- **Risky**: Higher danger tolerance across all actions
- **Tank**: Lower danger tolerance, more cautious decisions

### 3. Config-Specific Decision Logic

**Location**: `borg_kernel_v2.cu:286-393`

#### Critical HP Response (hp_pct < critical_threshold)
- **All configs**: Try healing first
- **Risky**: Fight desperately at low HP if monster adjacent
- **Conservative**: Recall to town if available

#### Combat Approach (monster_dist <= 5)
- **Aggro** (`WORSHIPS_DAMAGE` or `PLAYS_RISKY`): Charge in if danger < fight_threshold
- **Speed** (`WORSHIPS_SPEED`): Hit-and-run tactics (requires teleport + speed > 10)
- **Tank** (`WORSHIPS_HP`/`AC`): Only approach if very safe (danger < descend_threshold)
- **Scummer** (`KILLS_UNIQUES`): Hunt uniques if danger acceptable

#### Descending Behavior (depth < no_deeper)

**Speed Config**:
```cuda
should_descend = (speed >= 10 + depth / 10);
```
- Requires progressively more speed at deeper depths
- From APWBorg: Speed worshippers won't descend without adequate speed

**Tank Config**:
```cuda
should_descend = (hp_pct > 0.9f && danger < danger_descend_threshold / 2);
```
- Requires 90%+ HP and very low danger
- Most conservative descending

**Aggro Config**:
```cuda
should_descend = (hp_pct > 0.5f);
```
- Low bar: only 50% HP required
- From APWBorg `borg8.c:4570`: "Risky borgs are in a hurry"

**Scummer Config**:
```cuda
should_descend = (state.monster_count[id] < 3 && hp_pct > 0.75f);
```
- Clears level more thoroughly before descending
- Ensures most monsters killed

**Economy Config**:
```cuda
should_descend = (state.gold[id] > depth * 100 && hp_pct > 0.75f);
```
- Requires adequate gold collection before descending
- Gold-per-depth threshold

## Expected Performance Differences

### Aggro (CFG_WORSHIPS_DAMAGE | CFG_PLAYS_RISKY)
- **Higher avg depth**: 15-20+ (aggressive descending)
- **Higher death rate**: 50-70% (risky combat, low HP tolerance)
- **Faster gameplay**: Fewer rest/heal actions

### Speed (CFG_WORSHIPS_SPEED)
- **Moderate depth**: 12-15 (blocked by speed requirements)
- **Lower death rate**: 30-40% (hit-and-run tactics)
- **Tactical play**: Requires teleport for combat

### Tank (CFG_WORSHIPS_HP | CFG_WORSHIPS_AC)
- **Lower avg depth**: 8-12 (conservative descending)
- **Lowest death rate**: 10-20% (defensive play)
- **Slowest gameplay**: Frequent resting/healing

### Scummer (CFG_KILLS_UNIQUES)
- **Moderate depth**: 10-14 (level clearing)
- **Moderate death rate**: 30-40%
- **Thorough exploration**: Kills most monsters per level

### Economy (CFG_WORSHIPS_GOLD)
- **Moderate depth**: 10-13 (gold collection focus)
- **Low death rate**: 20-30% (careful play)
- **Resource accumulation**: Higher gold totals

## APWBorg References

### Key Behavioral Patterns Found

1. **Danger Perception** (`borg6.c:2092`):
   ```c
   if (borg_cfg[BORG_PLAYS_RISKY]) risky_boost = 3;
   ```
   - Risky borgs tolerate higher danger levels

2. **Healing Threshold** (`borg6.c:2886`):
   ```c
   if (borg_cfg[BORG_PLAYS_RISKY]) chance += 5;
   ```
   - Risky borgs less likely to heal in combat

3. **Descending Logic** (`borg8.c:4570`):
   ```c
   if (!borg_cfg[BORG_PLAYS_RISKY])  /* Risky borg in a hurry */
   ```
   - Risky borgs skip preparedness checks

4. **HP Worship** (`borg4.c:4533`):
   ```c
   if (borg_cfg[BORG_WORSHIPS_HP])
   ```
   - HP worshippers value constitution and max HP more

5. **Damage Worship** (`borg4.c:4242, 4263, 4286, 4297`):
   ```c
   if (borg_cfg[BORG_WORSHIPS_DAMAGE])
       value += (damage multiplier);
   ```
   - Damage worshippers prioritize offensive stats

6. **Speed Worship** (`borg4.c:4429`):
   ```c
   if (borg_cfg[BORG_WORSHIPS_SPEED])
       value += (speed bonus scaling);
   ```
   - Speed worshippers require high speed before descending

## Testing Recommendations

Run simulation with:
```bash
./borg_kernel_v2 10000 1000
```

Expected differentiation:
- **Aggro**: Depth 15-20, Death 60%
- **Speed**: Depth 12-15, Death 35%
- **Tank**: Depth 8-12, Death 15%
- **Scummer**: Depth 10-14, Death 35%
- **Economy**: Depth 10-13, Death 25%

## Future Enhancements

1. **Unique Tracking**: Scummer should track which uniques killed
2. **Gold Economy**: Economy config should visit shops, value gold items
3. **Speed Items**: Speed config should prioritize speed-boosting equipment
4. **Monster Fear**: Tank config should flee from specific dangerous monsters
5. **Combat Metrics**: Track combat effectiveness per config type

## Code Locations

- Danger calculation: `borg_kernel_v2.cu:28-77`
- Threshold calculation: `borg_kernel_v2.cu:238-277`
- Decision tree: `borg_kernel_v2.cu:286-393`
- Config assignment: `borg_kernel_v2.cu:533-545`

## Validation

Compare against APWBorg behavior patterns:
- [ ] Risky borgs descend faster
- [ ] Tank borgs survive longer but reach lower depths
- [ ] Speed borgs use hit-and-run tactics
- [ ] Aggro borgs die more but reach deeper
- [ ] Configs show measurably different performance profiles
