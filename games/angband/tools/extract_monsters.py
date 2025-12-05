#!/usr/bin/env python3
"""
Extract monster data from Angband's monster.txt file for GPU implementation.
Parses monster definitions and outputs C arrays for CUDA.
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MonsterBlow:
    """Represents a single monster attack."""
    method: str
    effect: str
    damage_dice: int = 0
    damage_sides: int = 0

    @classmethod
    def from_line(cls, line: str) -> Optional['MonsterBlow']:
        """Parse a blow line like 'blow:HIT:HURT:2d6' or 'blow:CLAW'"""
        parts = line.split(':')
        if len(parts) < 2:
            return None

        method = parts[1]
        effect = parts[2] if len(parts) > 2 else ""

        # Parse damage like "2d6" or "20d10"
        dd, ds = 0, 0
        if len(parts) > 3:
            damage_str = parts[3]
            match = re.match(r'(\d+)d(\d+)', damage_str)
            if match:
                dd = int(match.group(1))
                ds = int(match.group(2))

        return cls(method=method, effect=effect, damage_dice=dd, damage_sides=ds)


@dataclass
class MonsterRace:
    """Represents a monster race."""
    name: str
    speed: int = 110
    hit_points: int = 1
    armor_class: int = 1
    depth: int = 0
    rarity: int = 1
    experience: int = 0
    blows: List[MonsterBlow] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def get_flag_bits(self) -> int:
        """Convert flags to bitfield (simplified for GPU)."""
        bits = 0
        if 'UNIQUE' in self.flags:
            bits |= 0x01
        if 'EVIL' in self.flags:
            bits |= 0x02
        if 'UNDEAD' in self.flags:
            bits |= 0x04
        if 'DRAGON' in self.flags:
            bits |= 0x08
        if 'DEMON' in self.flags:
            bits |= 0x10
        if 'ANIMAL' in self.flags:
            bits |= 0x20
        if 'SMART' in self.flags:
            bits |= 0x40
        if 'REGENERATE' in self.flags:
            bits |= 0x80
        return bits


def parse_monster_txt(filepath: str) -> List[MonsterRace]:
    """Parse monster.txt and extract monster races."""
    monsters = []
    current_monster = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # New monster entry
            if line.startswith('name:'):
                # Save previous monster if exists
                if current_monster and current_monster.name != '<player>':
                    monsters.append(current_monster)

                # Start new monster
                monster_name = line[5:].strip()
                current_monster = MonsterRace(name=monster_name)

            elif current_monster:
                # Parse attributes
                if line.startswith('speed:'):
                    current_monster.speed = int(line[6:].strip())

                elif line.startswith('hit-points:'):
                    current_monster.hit_points = int(line[11:].strip())

                elif line.startswith('armor-class:'):
                    current_monster.armor_class = int(line[12:].strip())

                elif line.startswith('depth:'):
                    current_monster.depth = int(line[6:].strip())

                elif line.startswith('rarity:'):
                    current_monster.rarity = int(line[7:].strip())

                elif line.startswith('experience:'):
                    current_monster.experience = int(line[11:].strip())

                elif line.startswith('blow:'):
                    blow = MonsterBlow.from_line(line)
                    if blow and len(current_monster.blows) < 4:  # Max 4 blows
                        current_monster.blows.append(blow)

                elif line.startswith('flags:'):
                    # Parse flags separated by |
                    flag_str = line[6:].strip()
                    flags = [f.strip() for f in flag_str.split('|')]
                    current_monster.flags.extend(flags)

    # Don't forget the last monster
    if current_monster and current_monster.name != '<player>':
        monsters.append(current_monster)

    return monsters


def generate_effect_enum(monsters: List[MonsterRace]) -> dict:
    """Generate effect enum mapping from all monster attacks."""
    effects = set()
    for monster in monsters:
        for blow in monster.blows:
            if blow.effect:
                effects.add(blow.effect)

    # Sort for consistent output
    effects = sorted(effects)

    # Create mapping
    effect_map = {}
    for idx, effect in enumerate(effects, 1):
        effect_map[effect] = idx

    return effect_map


def generate_cuda_header(monsters: List[MonsterRace], effect_map: dict,
                         output_path: str):
    """Generate CUDA header with monster data."""

    # Limit to reasonable number for GPU memory
    max_monsters = min(len(monsters), 700)
    monsters = monsters[:max_monsters]

    with open(output_path, 'w') as f:
        f.write("""#ifndef ANGBAND_MONSTERS_CUH
#define ANGBAND_MONSTERS_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Maximum monsters and blows
#define MAX_MONSTER_RACES {}
#define MAX_BLOWS_PER_MONSTER 4

// Attack effects
""".format(max_monsters))

        # Write effect enum
        for effect, value in sorted(effect_map.items(), key=lambda x: x[1]):
            f.write(f"#define BLOW_EFFECT_{effect} {value}\n")

        f.write("""
// Monster flags
#define MFLAG_UNIQUE     0x01
#define MFLAG_EVIL       0x02
#define MFLAG_UNDEAD     0x04
#define MFLAG_DRAGON     0x08
#define MFLAG_DEMON      0x10
#define MFLAG_ANIMAL     0x20
#define MFLAG_SMART      0x40
#define MFLAG_REGENERATE 0x80

// Monster blow (attack)
struct MonsterBlow {
    uint8_t effect;    // BLOW_EFFECT_*
    uint8_t dd;        // damage dice
    uint8_t ds;        // damage sides
    uint8_t pad;       // padding for alignment
};

// Monster race definition
struct MonsterRace {
    int16_t speed;      // 110 = normal, higher = faster
    int16_t hp;         // average hit points (will overflow for Morgoth, but scaled in practice)
    int16_t ac;         // armor class
    int16_t depth;      // native depth (1-100)
    int16_t rarity;     // inverse spawn probability
    int32_t exp;        // experience value (needs 32-bit for high-level monsters)
    uint8_t num_blows;  // number of attacks
    uint8_t flags;      // packed flags (UNIQUE, EVIL, etc)
    MonsterBlow blows[MAX_BLOWS_PER_MONSTER];
};

// Monster data in global memory (too large for constant memory)
__device__ MonsterRace MONSTER_RACES[MAX_MONSTER_RACES] = {
""")

        # Write monster data
        for i, monster in enumerate(monsters):
            f.write(f"    // {i}: {monster.name}\n")
            f.write("    {\n")
            f.write(f"        {monster.speed}, // speed\n")
            f.write(f"        {monster.hit_points}, // hp\n")
            f.write(f"        {monster.armor_class}, // ac\n")
            f.write(f"        {monster.depth}, // depth\n")
            f.write(f"        {monster.rarity}, // rarity\n")
            f.write(f"        {monster.experience}, // exp\n")
            f.write(f"        {len(monster.blows)}, // num_blows\n")
            f.write(f"        0x{monster.get_flag_bits():02X}, // flags\n")
            f.write("        { // blows\n")

            # Write blows
            for j in range(4):
                if j < len(monster.blows):
                    blow = monster.blows[j]
                    effect_val = effect_map.get(blow.effect, 0)
                    f.write(f"            {{{effect_val}, {blow.damage_dice}, "
                           f"{blow.damage_sides}, 0}}")
                else:
                    f.write("            {0, 0, 0, 0}")

                if j < 3:
                    f.write(",")
                f.write("\n")

            f.write("        }\n")
            f.write("    }")
            if i < len(monsters) - 1:
                f.write(",")
            f.write("\n")

        f.write("""};

__device__ int NUM_MONSTER_RACES = MAX_MONSTER_RACES;

// Get appropriate monster for given depth
// Uses inverse rarity as probability weight
__device__ inline int get_monster_for_depth(int depth, curandState* rng) {
    // Simple algorithm: pick random monster at or below depth
    // In full implementation, would use rarity-weighted selection

    int candidates[MAX_MONSTER_RACES];
    int count = 0;

    // Find all monsters at or below this depth
    for (int i = 0; i < NUM_MONSTER_RACES; i++) {
        if (MONSTER_RACES[i].depth <= depth && MONSTER_RACES[i].depth > 0) {
            candidates[count++] = i;
        }
    }

    if (count == 0) {
        // No valid monsters, return weakest non-town monster
        for (int i = 0; i < NUM_MONSTER_RACES; i++) {
            if (MONSTER_RACES[i].depth > 0) {
                return i;
            }
        }
        return 0;
    }

    // Pick random candidate
    float r = curand_uniform(rng);
    int idx = (int)(r * count);
    if (idx >= count) idx = count - 1;

    return candidates[idx];
}

// Monster attack calculation
// Returns damage dealt to player
__device__ inline int monster_attack(int race_idx, int player_ac, curandState* rng) {
    if (race_idx < 0 || race_idx >= NUM_MONSTER_RACES) {
        return 0;
    }

    MonsterRace* monster = &MONSTER_RACES[race_idx];
    int total_damage = 0;

    // Process each blow
    for (int i = 0; i < monster->num_blows; i++) {
        MonsterBlow* blow = &monster->blows[i];
        if (blow->effect == 0) continue;

        // Hit roll: monster hits if random(100) > player_ac * 2/3
        float hit_roll = curand_uniform(rng) * 100.0f;
        float defense = player_ac * (2.0f / 3.0f);

        if (hit_roll > defense) {
            // Hit! Roll damage
            int damage = 0;
            for (int d = 0; d < blow->dd; d++) {
                float r = curand_uniform(rng);
                damage += (int)(r * blow->ds) + 1;
            }

            total_damage += damage;
        }
    }

    return total_damage;
}

// Get monster name index (for display)
// In full implementation, would have string table
__device__ inline const char* get_monster_name(int race_idx) {
    // Would need to add name strings to constant memory
    // For now, just return generic name
    return "monster";
}

#endif // ANGBAND_MONSTERS_CUH
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: extract_monsters.py <monster.txt> [output.cuh]")
        print("Example: extract_monsters.py /tmp/angband-check/lib/gamedata/monster.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "angband_monsters.cuh"

    print(f"Parsing {input_file}...")
    monsters = parse_monster_txt(input_file)
    print(f"Found {len(monsters)} monsters")

    print("Generating effect enum...")
    effect_map = generate_effect_enum(monsters)
    print(f"Found {len(effect_map)} unique attack effects")

    print(f"Generating CUDA header: {output_file}")
    generate_cuda_header(monsters, effect_map, output_file)

    print("\nStatistics:")
    print(f"  Total monsters: {len(monsters)}")
    print(f"  Monsters in output: {min(len(monsters), 700)}")
    print(f"  Unique effects: {len(effect_map)}")
    print(f"  Deepest monster: depth {max(m.depth for m in monsters)}")
    print(f"  Strongest monster: {max(monsters, key=lambda m: m.hit_points).name} "
          f"({max(m.hit_points for m in monsters)} HP)")

    # Size estimate (updated for int32_t exp field)
    # MonsterRace: 5*int16_t(10) + int32_t(4) + 2*uint8_t(2) + 4*MonsterBlow(16) = 32 bytes
    struct_size = 32
    total_size = struct_size * min(len(monsters), 700)
    print(f"\nMemory usage: ~{total_size:,} bytes ({total_size / 1024:.1f} KB)")

    print(f"\nDone! Header written to {output_file}")


if __name__ == "__main__":
    main()
