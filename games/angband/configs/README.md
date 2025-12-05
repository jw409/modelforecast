# APWBorg Alternative Configurations

8 AI-generated configurations exploring different strategic approaches.

## Configuration Matrix

| Config | Speed | Damage | HP | AC | Gold | Risky | Uniques | Depth | Model |
|--------|-------|--------|----|----|------|-------|---------|-------|-------|
| AggroBorg | ✓ | ✓ | - | - | - | ✓ | - | 127 | Sonnet 4.5 |
| SpeedBorg | ✓ | - | - | - | - | ✓ | - | 127 | Sonnet 4.5 |
| TankBorg | - | - | ✓ | ✓ | - | - | - | 30 | Sonnet 4.5 |
| ScummerBorg | ✓ | ✓ | ✓ | ✓ | - | - | ✓ | 98 | Sonnet 4.5 |
| MetaBorg | ✓ | - | ✓ | - | - | ✓ | - | 127 | Opus 4.5 |
| EvolutionBorg | *varies by phase* | | | | | | | | Opus 4.5 |
| EconomyBorg | ✓ | ✓ | - | - | ✓ | - | ✓ | 45 | Gemini 3.0 |
| CheatBorg | ✓ | ✓ | - | - | - | ✓ | ✓ | 127 | Gemini 3.0 |

## Strategic Archetypes

### 1. AggroBorg (Sonnet)
**Philosophy**: Kill before killed. Dive fast, hit hard.
- Speed + Damage worship
- Max risk tolerance
- No unique waiting

### 2. SpeedBorg (Sonnet)
**Philosophy**: Action economy is everything.
- Speed worship ONLY
- Multiplicative advantage (3x actions at +30 speed)
- Glass cannon approach

### 3. TankBorg (Sonnet)
**Philosophy**: Dead borgs make zero progress.
- HP + AC maximization
- Conservative depth cap (30)
- Expected 80%+ survival rate

### 4. ScummerBorg (Sonnet)
**Philosophy**: Artifacts compound. Hunt uniques.
- Balanced survivability + offense
- Unique hunting enabled
- Cheat death for persistence

### 5. MetaBorg (Opus)
**Philosophy**: Game-theoretic optimization.
- Speed + HP (multiplicative + buffer)
- "Wins per hour" > "win rate per run"
- Risky play accepted for faster cycles

### 6. EvolutionBorg (Opus)
**Philosophy**: Adapt strategy to game phase.
- Phase 1 (Early): Defensive, gold farming
- Phase 2 (Mid): Speed, unique hunting
- Phase 3 (Late): Offensive, all-in
- Transition triggers defined

### 7. EconomyBorg (Gemini)
**Philosophy**: Dungeon = resource extraction.
- Gold worship enabled
- Moderate depth cap (45)
- "Mobs per hour" optimization

### 8. CheatBorg (Gemini)
**Philosophy**: Immortality changes everything.
- Max aggression with cheat_death
- Zero defensive investment
- Brute-force RNG via infinite trials

## Benchmark Plan

Run each config N times, measure:
1. **Depth reached** (max, avg)
2. **Survival rate** (lives/deaths)
3. **Time to depth 50** (turns)
4. **Uniques killed**
5. **Wins (if any)**

## Files

- `aggro.txt` - AggroBorg
- `speed.txt` - SpeedBorg
- `tank.txt` - TankBorg
- `scummer.txt` - ScummerBorg
- `meta.txt` - MetaBorg
- `evolution_*.txt` - EvolutionBorg phases
- `economy.txt` - EconomyBorg
- `cheat.txt` - CheatBorg
