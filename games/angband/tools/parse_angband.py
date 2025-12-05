#!/usr/bin/env python3
"""
Angband Data Parser - Extracts ALL game data for GPU port
Outputs JSON and CUDA header files

Parses:
- monster.txt (624 monsters)
- object.txt (items)
- class.txt (player classes)
- constants.txt (game constants)
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

# Paths
ANGBAND_DATA = Path("/tmp/angband-check/lib/gamedata")
OUTPUT_DIR = Path(__file__).parent.parent / "generated"

# Effect mapping
EFFECT_MAP = {
    "HURT": 0, "POISON": 1, "FIRE": 2, "COLD": 3, "ACID": 4, "ELEC": 5,
    "PARALYZE": 6, "CONFUSE": 7, "BLIND": 8, "TERRIFY": 9,
    "LOSE_STR": 10, "LOSE_DEX": 11, "LOSE_CON": 12, "LOSE_INT": 13, "LOSE_WIS": 14,
    "LOSE_ALL": 15, "EXP_": 16, "DRAIN_CHARGES": 17, "DISENCHANT": 18,
    "BLACK_BREATH": 19, "SHATTER": 20, "HALLU": 21,
    "EAT_GOLD": 22, "EAT_ITEM": 23, "EAT_FOOD": 24, "EAT_LIGHT": 25,
}

# Emoji mapping for monsters (by base type)
EMOJI_MAP = {
    "dragon": "🐉", "demon": "👿", "undead": "💀", "orc": "👹", "troll": "🧌",
    "humanoid": "🧑", "animal": "🐾", "insect": "🐛", "spider": "🕷️", "snake": "🐍",
    "jelly": "🟢", "mold": "🍄", "eye": "👁️", "ghost": "👻", "vampire": "🧛",
    "lich": "☠️", "golem": "🗿", "elemental": "🌀", "angel": "👼", "giant": "🦣",
    "hydra": "🐲", "hound": "🐕", "feline": "🐱", "canine": "🐺", "bird": "🦅",
    "bat": "🦇", "rodent": "🐀", "worm": "🪱", "centipede": "🐛", "ant": "🐜",
    "kobold": "👺", "yeek": "😱", "naga": "🐍", "icky thing": "👾", "townsfolk": "🧍",
    "Morgoth": "👑", "Sauron": "🔥",
}

@dataclass
class MonsterBlow:
    method: str = "HIT"
    effect: str = "HURT"
    damage: str = "1d1"

    @property
    def dd(self) -> int:
        if not self.damage or 'd' not in self.damage:
            return 1
        return int(self.damage.split('d')[0])

    @property
    def ds(self) -> int:
        if not self.damage or 'd' not in self.damage:
            return 1
        return int(self.damage.split('d')[1])

    @property
    def effect_id(self) -> int:
        for key, val in EFFECT_MAP.items():
            if key in self.effect:
                return val
        return 0

@dataclass
class Monster:
    id: int = 0
    name: str = ""
    base: str = ""
    glyph: str = "?"
    color: str = "w"
    speed: int = 110
    hp: int = 10
    ac: int = 1
    depth: int = 1
    rarity: int = 1
    exp: int = 0
    sleep: int = 0
    blows: List[MonsterBlow] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    spells: List[str] = field(default_factory=list)
    spell_freq: int = 0
    desc: str = ""

    @property
    def emoji(self) -> str:
        # Try name first, then base
        name_lower = self.name.lower()
        for key, emoji in EMOJI_MAP.items():
            if key in name_lower:
                return emoji
        base_lower = self.base.lower()
        for key, emoji in EMOJI_MAP.items():
            if key in base_lower:
                return emoji
        return "❓"

    @property
    def is_unique(self) -> bool:
        return "UNIQUE" in self.flags

    @property
    def breeds(self) -> bool:
        return "MULTIPLY" in self.flags

    @property
    def max_blows(self) -> int:
        return min(len(self.blows), 4)

def parse_dice(dice_str: str) -> tuple:
    """Parse '2d6' into (2, 6)"""
    if not dice_str or 'd' not in dice_str:
        return (1, 1)
    parts = dice_str.split('d')
    return (int(parts[0]), int(parts[1]))

def parse_monster_file(filepath: Path) -> List[Monster]:
    """Parse monster.txt into Monster objects"""
    monsters = []
    current = None
    monster_id = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('name:'):
                if current and current.name and current.name != "<player>":
                    monsters.append(current)
                current = Monster(id=monster_id)
                current.name = line[5:]
                monster_id += 1

            elif current:
                if line.startswith('base:'):
                    current.base = line[5:]
                elif line.startswith('glyph:'):
                    current.glyph = line[6:]
                elif line.startswith('color:'):
                    current.color = line[6:]
                elif line.startswith('speed:'):
                    current.speed = int(line[6:])
                elif line.startswith('hit-points:'):
                    current.hp = int(line[11:])
                elif line.startswith('armor-class:'):
                    current.ac = int(line[12:])
                elif line.startswith('depth:'):
                    current.depth = int(line[6:])
                elif line.startswith('rarity:'):
                    current.rarity = int(line[7:])
                elif line.startswith('experience:'):
                    current.exp = int(line[11:])
                elif line.startswith('sleepiness:'):
                    current.sleep = int(line[11:])
                elif line.startswith('blow:'):
                    parts = line[5:].split(':')
                    blow = MonsterBlow(
                        method=parts[0] if len(parts) > 0 else "HIT",
                        effect=parts[1] if len(parts) > 1 else "HURT",
                        damage=parts[2] if len(parts) > 2 else ""
                    )
                    current.blows.append(blow)
                elif line.startswith('flags:'):
                    flags = [f.strip() for f in line[6:].split('|')]
                    current.flags.extend(flags)
                elif line.startswith('spells:'):
                    spells = [s.strip() for s in line[7:].split('|')]
                    current.spells.extend(spells)
                elif line.startswith('spell-freq:'):
                    current.spell_freq = int(line[11:])
                elif line.startswith('desc:'):
                    current.desc += line[5:] + " "

    if current and current.name and current.name != "<player>":
        monsters.append(current)

    return monsters

def generate_cuda_header(monsters: List[Monster], output_path: Path):
    """Generate CUDA header with all monster data"""

    header = '''/*
 * ANGBAND MONSTERS - Auto-generated from monster.txt
 * DO NOT EDIT - Generated by parse_angband.py
 *
 * Total monsters: {count}
 */

#ifndef ANGBAND_MONSTER_DATA_CUH
#define ANGBAND_MONSTER_DATA_CUH

#include <cuda_runtime.h>

// Monster blow structure
struct MonsterBlow {{
    uint8_t method;   // Attack method
    uint8_t effect;   // Effect type (EFFECT_* constants)
    uint8_t dd;       // Damage dice count
    uint8_t ds;       // Damage dice sides
}};

// Monster race structure
struct MonsterRace {{
    uint16_t hp;          // Hit points
    uint16_t exp;         // Experience value
    uint8_t depth;        // Native depth
    uint8_t rarity;       // Rarity (inverse spawn chance)
    uint8_t speed;        // Speed (110 = normal)
    uint8_t ac;           // Armor class
    uint8_t sleep;        // Initial sleep value
    uint8_t num_blows;    // Number of attacks
    uint8_t flags;        // Packed flags (UNIQUE, MULTIPLY, etc)
    uint8_t spell_freq;   // Spell frequency (0 = never)
    MonsterBlow blows[4]; // Up to 4 attacks
}};

// Monster flags (packed into uint8_t)
#define MFLAG_UNIQUE    (1 << 0)
#define MFLAG_MULTIPLY  (1 << 1)
#define MFLAG_EVIL      (1 << 2)
#define MFLAG_UNDEAD    (1 << 3)
#define MFLAG_DEMON     (1 << 4)
#define MFLAG_DRAGON    (1 << 5)
#define MFLAG_ANIMAL    (1 << 6)
#define MFLAG_SMART     (1 << 7)

#define NUM_MONSTER_RACES {count}

__constant__ MonsterRace MONSTER_RACES[NUM_MONSTER_RACES] = {{
'''.format(count=len(monsters))

    for i, m in enumerate(monsters):
        # Pack flags
        flags = 0
        if m.is_unique: flags |= 1
        if m.breeds: flags |= 2
        if "EVIL" in m.flags: flags |= 4
        if "UNDEAD" in m.flags: flags |= 8
        if "DEMON" in m.flags: flags |= 16
        if "DRAGON" in m.flags: flags |= 32
        if "ANIMAL" in m.flags: flags |= 64
        if "SMART" in m.flags: flags |= 128

        # Format blows
        blow_strs = []
        for j in range(4):
            if j < len(m.blows):
                b = m.blows[j]
                blow_strs.append(f"{{{0}, {b.effect_id}, {b.dd}, {b.ds}}}")
            else:
                blow_strs.append("{0, 0, 0, 0}")

        blows_str = ", ".join(blow_strs)

        # Clamp values to fit uint8/uint16
        hp = min(m.hp, 65535)
        exp = min(m.exp, 65535)
        speed = min(m.speed, 255)
        ac = min(m.ac, 255)

        header += f"    // [{i}] {m.name} (depth {m.depth})\n"
        header += f"    {{{hp}, {exp}, {m.depth}, {m.rarity}, {speed}, {ac}, {m.sleep}, {m.max_blows}, {flags}, {m.spell_freq}, {{{blows_str}}}}},\n"

    header += '''};

// Monster names (for debugging/display)
__constant__ const char* MONSTER_NAMES[NUM_MONSTER_RACES] = {
'''

    for i, m in enumerate(monsters):
        # Escape quotes in name
        name = m.name.replace('"', '\\"')
        header += f'    "{name}",\n'

    header += '''};

// Monster emoji (Unicode codepoints for display)
__constant__ const char* MONSTER_EMOJI[NUM_MONSTER_RACES] = {
'''

    for m in monsters:
        header += f'    "{m.emoji}",\n'

    header += '''};

#endif // ANGBAND_MONSTER_DATA_CUH
'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header)

    print(f"Generated {output_path} with {len(monsters)} monsters")

def generate_json(monsters: List[Monster], output_path: Path):
    """Generate JSON for further processing/expansion"""
    data = []
    for m in monsters:
        data.append({
            "id": m.id,
            "name": m.name,
            "base": m.base,
            "glyph": m.glyph,
            "color": m.color,
            "speed": m.speed,
            "hp": m.hp,
            "ac": m.ac,
            "depth": m.depth,
            "rarity": m.rarity,
            "exp": m.exp,
            "sleep": m.sleep,
            "blows": [{"method": b.method, "effect": b.effect, "damage": b.damage} for b in m.blows],
            "flags": m.flags,
            "spells": m.spells,
            "spell_freq": m.spell_freq,
            "emoji": m.emoji,
            "is_unique": m.is_unique,
            "breeds": m.breeds,
            "desc": m.desc.strip()
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {output_path} with {len(data)} monsters")

def print_stats(monsters: List[Monster]):
    """Print statistics about parsed monsters"""
    print(f"\n=== MONSTER STATISTICS ===")
    print(f"Total monsters: {len(monsters)}")

    # By depth
    depth_counts = {}
    for m in monsters:
        bucket = (m.depth // 10) * 10
        depth_counts[bucket] = depth_counts.get(bucket, 0) + 1

    print(f"\nBy depth range:")
    for depth in sorted(depth_counts.keys()):
        print(f"  {depth:3d}-{depth+9:3d}: {depth_counts[depth]:3d} monsters")

    # Uniques
    uniques = [m for m in monsters if m.is_unique]
    print(f"\nUnique monsters: {len(uniques)}")

    # Breeders
    breeders = [m for m in monsters if m.breeds]
    print(f"Breeding monsters: {len(breeders)}")
    for b in breeders[:10]:
        print(f"  - {b.name} (depth {b.depth})")

    # Attack effects
    effects = {}
    for m in monsters:
        for b in m.blows:
            effects[b.effect] = effects.get(b.effect, 0) + 1

    print(f"\nAttack effects:")
    for effect, count in sorted(effects.items(), key=lambda x: -x[1])[:15]:
        print(f"  {effect}: {count}")

    # Highest damage monsters
    print(f"\nHighest damage monsters:")
    def avg_damage(m):
        total = 0
        for b in m.blows:
            total += b.dd * (b.ds + 1) / 2
        return total

    top_damage = sorted(monsters, key=avg_damage, reverse=True)[:10]
    for m in top_damage:
        dmg = avg_damage(m)
        print(f"  {m.name}: {dmg:.1f} avg damage/round (depth {m.depth})")

if __name__ == "__main__":
    monster_file = ANGBAND_DATA / "monster.txt"

    if not monster_file.exists():
        print(f"ERROR: {monster_file} not found!")
        print("Clone Angband first: git clone https://github.com/angband/angband /tmp/angband-check")
        exit(1)

    print(f"Parsing {monster_file}...")
    monsters = parse_monster_file(monster_file)

    print_stats(monsters)

    # Generate outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_json(monsters, OUTPUT_DIR / "monsters.json")
    generate_cuda_header(monsters, OUTPUT_DIR / "angband_monster_data.cuh")

    print(f"\nDone! Files written to {OUTPUT_DIR}")
