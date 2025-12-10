#!/usr/bin/env python3
"""
Extract Static Data from Original DOOM Source Code

Parses linuxdoom-1.10 source files and generates GPU headers with identical data:
- rndtable[256] from m_random.c - Random number lookup table
- finesine[10240] from tables.c - Sine lookup table (also used for cosine)
- states[] from info.c - State machine definitions (~1000 states)
- mobjinfo[] from info.c - Monster/item definitions (138 entries)

This data forms the "shared truth" between CPU and GPU implementations.

Usage:
    python extract_doom_data.py [--verify] [--output doom_data.cuh]
"""

import os
import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Paths
SCRIPT_DIR = Path(__file__).parent
GPU_DIR = SCRIPT_DIR.parent
DOOM_SOURCE = GPU_DIR / "source" / "linuxdoom-1.10"


@dataclass
class State:
    """DOOM state machine state (from state_t in info.h)."""
    sprite: int      # Sprite number
    frame: int       # Frame number (bit 15 = full bright)
    tics: int        # Duration in tics (-1 = infinite)
    action: str      # Action function name (or "NULL")
    nextstate: int   # Next state index
    misc1: int = 0   # Extra parameter
    misc2: int = 0   # Extra parameter


@dataclass
class MobjInfo:
    """DOOM map object info (from mobjinfo_t in info.h)."""
    doomednum: int       # Editor thing number
    spawnstate: int      # Initial state
    spawnhealth: int     # Starting health
    seestate: int        # State when seeing player
    seesound: int        # Sound when seeing player
    reactiontime: int    # Tics before reacting
    attacksound: int     # Attack sound
    painstate: int       # Pain animation state
    painchance: int      # Chance to flinch (0-255)
    painsound: int       # Pain sound
    meleestate: int      # Melee attack state
    missilestate: int    # Ranged attack state
    deathstate: int      # Death animation state
    xdeathstate: int     # Gib death state
    deathsound: int      # Death sound
    speed: int           # Movement speed
    radius: int          # Collision radius (fixed point)
    height: int          # Collision height (fixed point)
    mass: int            # Mass for physics
    damage: int          # Contact damage
    activesound: int     # Ambient sound
    flags: int           # MF_* flags
    raisestate: int      # Archvile resurrect state


def extract_rndtable(source_dir: Path) -> List[int]:
    """Extract rndtable[256] from m_random.c."""
    path = source_dir / "m_random.c"
    content = path.read_text()

    # Find the rndtable array
    match = re.search(r'rndtable\[256\]\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find rndtable in m_random.c")

    # Extract all numbers
    numbers_str = match.group(1)
    numbers = [int(x.strip()) for x in re.findall(r'\d+', numbers_str)]

    if len(numbers) != 256:
        raise ValueError(f"Expected 256 entries in rndtable, got {len(numbers)}")

    return numbers


def extract_finesine(source_dir: Path) -> List[int]:
    """Extract finesine[10240] from tables.c."""
    path = source_dir / "tables.c"
    content = path.read_text()

    # Find the finesine array (it's after finetangent)
    # finesine has 10240 entries = 8192 sine + 2048 for cosine overlap
    match = re.search(r'finesine\[(?:\d+|FINEANGLES\s*\+\s*\d+)\]\s*=\s*\{([^}]+)\}', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find finesine in tables.c")

    # Extract all numbers (including negatives)
    numbers_str = match.group(1)
    numbers = [int(x.strip()) for x in re.findall(r'-?\d+', numbers_str)]

    print(f"  Extracted {len(numbers)} finesine entries")
    return numbers


def parse_state_enums(source_dir: Path) -> Dict[str, int]:
    """Parse statenum_t enum from info.h to get state name -> index mapping."""
    path = source_dir / "info.h"
    content = path.read_text()

    # Find statenum_t enum
    match = re.search(r'typedef\s+enum\s*\{([^}]+)\}\s*statenum_t', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find statenum_t enum in info.h")

    enum_body = match.group(1)
    states = {}
    index = 0

    for line in enum_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        # Match state names (e.g., "S_NULL," or "S_PLAY,")
        m = re.match(r'(S_\w+)\s*,?', line)
        if m:
            states[m.group(1)] = index
            index += 1

    print(f"  Parsed {len(states)} state enum values")
    return states


def parse_sprite_enums(source_dir: Path) -> Dict[str, int]:
    """Parse spritenum_t enum from info.h."""
    path = source_dir / "info.h"
    content = path.read_text()

    match = re.search(r'typedef\s+enum\s*\{([^}]+)\}\s*spritenum_t', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find spritenum_t enum in info.h")

    enum_body = match.group(1)
    sprites = {}
    index = 0

    for line in enum_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        m = re.match(r'(SPR_\w+)\s*,?', line)
        if m:
            sprites[m.group(1)] = index
            index += 1

    print(f"  Parsed {len(sprites)} sprite enum values")
    return sprites


def parse_mobjtype_enums(source_dir: Path) -> Dict[str, int]:
    """Parse mobjtype_t enum from info.h."""
    path = source_dir / "info.h"
    content = path.read_text()

    match = re.search(r'typedef\s+enum\s*\{([^}]+)\}\s*mobjtype_t', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find mobjtype_t enum in info.h")

    enum_body = match.group(1)
    types = {}
    index = 0

    for line in enum_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        m = re.match(r'(MT_\w+)\s*,?', line)
        if m:
            types[m.group(1)] = index
            index += 1

    print(f"  Parsed {len(types)} mobjtype enum values")
    return types


def parse_sound_enums(source_dir: Path) -> Dict[str, int]:
    """Parse sfxenum_t from sounds.h."""
    path = source_dir / "sounds.h"
    content = path.read_text()

    match = re.search(r'typedef\s+enum\s*\{([^}]+)\}\s*sfxenum_t', content, re.DOTALL)
    if not match:
        # Try alternative format
        match = re.search(r'enum\s+sfxenum_t\s*\{([^}]+)\}', content, re.DOTALL)

    if not match:
        print("  WARNING: Could not find sfxenum_t, using fallback")
        return {"sfx_None": 0}

    enum_body = match.group(1)
    sounds = {}
    index = 0

    for line in enum_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue

        m = re.match(r'(sfx_\w+)\s*,?', line)
        if m:
            sounds[m.group(1)] = index
            index += 1

    print(f"  Parsed {len(sounds)} sound enum values")
    return sounds


def extract_states(source_dir: Path, state_enums: Dict[str, int],
                   sprite_enums: Dict[str, int]) -> List[State]:
    """Extract states[] array from info.c."""
    path = source_dir / "info.c"
    content = path.read_text()

    # Find states array
    match = re.search(r'state_t\s+states\[NUMSTATES\]\s*=\s*\{(.+?)^\};', content,
                      re.DOTALL | re.MULTILINE)
    if not match:
        raise ValueError("Could not find states array in info.c")

    states_body = match.group(1)
    states = []

    # Parse each state entry: {SPR_xxx, frame, tics, {action}, nextstate, misc1, misc2}
    # The action is wrapped in braces and can be NULL or a function pointer
    state_pattern = re.compile(
        r'\{(SPR_\w+),\s*(\d+),\s*(-?\d+),\s*\{(\w+)\},\s*(S_\w+),\s*(\d+),\s*(\d+)\}',
        re.MULTILINE
    )

    for m in state_pattern.finditer(states_body):
        sprite_name = m.group(1)
        frame = int(m.group(2))
        tics = int(m.group(3))
        action = m.group(4)
        nextstate_name = m.group(5)
        misc1 = int(m.group(6))
        misc2 = int(m.group(7))

        sprite = sprite_enums.get(sprite_name, 0)
        nextstate = state_enums.get(nextstate_name, 0)

        states.append(State(
            sprite=sprite,
            frame=frame,
            tics=tics,
            action=action,
            nextstate=nextstate,
            misc1=misc1,
            misc2=misc2
        ))

    print(f"  Extracted {len(states)} states")
    return states


def extract_mobjinfo(source_dir: Path, state_enums: Dict[str, int],
                     sound_enums: Dict[str, int]) -> List[MobjInfo]:
    """Extract mobjinfo[] array from info.c."""
    path = source_dir / "info.c"
    content = path.read_text()

    # Find mobjinfo array
    match = re.search(r'mobjinfo_t\s+mobjinfo\[NUMMOBJTYPES\]\s*=\s*\{(.+?)^\};',
                      content, re.DOTALL | re.MULTILINE)
    if not match:
        raise ValueError("Could not find mobjinfo array in info.c")

    mobjinfo_body = match.group(1)
    mobjinfos = []

    # Each mobjinfo entry is a block { ... }
    # We need to parse 23 fields per entry
    entry_pattern = re.compile(r'\{[^{}]*//\s*MT_\w+[^{}]+\}', re.DOTALL)

    for entry_match in entry_pattern.finditer(mobjinfo_body):
        entry = entry_match.group(0)

        # Extract the MT_xxx type name from comment
        type_match = re.search(r'//\s*(MT_\w+)', entry)
        if type_match:
            type_name = type_match.group(1)

        # Extract all values - they're on separate lines with comments
        values = []
        for line in entry.split('\n'):
            # Skip comment-only lines and the type comment
            if '//' in line:
                line = line.split('//')[0]
            line = line.strip().rstrip(',')
            if not line or line in '{}':
                continue

            # Handle expressions like 16*FRACUNIT
            if '*' in line and 'FRACUNIT' in line:
                # Evaluate: N*FRACUNIT = N << 16
                num_match = re.match(r'(-?\d+)\s*\*\s*FRACUNIT', line)
                if num_match:
                    values.append(int(num_match.group(1)) << 16)
                    continue

            # Handle flag combinations
            if 'MF_' in line:
                # Parse flags - we'll compute the value
                flags = parse_flags(line)
                values.append(flags)
                continue

            # Handle state references (S_xxx)
            state_match = re.match(r'(S_\w+)', line)
            if state_match:
                values.append(state_enums.get(state_match.group(1), 0))
                continue

            # Handle sound references (sfx_xxx)
            sound_match = re.match(r'(sfx_\w+)', line)
            if sound_match:
                values.append(sound_enums.get(sound_match.group(1), 0))
                continue

            # Handle plain numbers
            num_match = re.match(r'(-?\d+)', line)
            if num_match:
                values.append(int(num_match.group(1)))
                continue

        if len(values) >= 23:
            mobjinfos.append(MobjInfo(
                doomednum=values[0],
                spawnstate=values[1],
                spawnhealth=values[2],
                seestate=values[3],
                seesound=values[4],
                reactiontime=values[5],
                attacksound=values[6],
                painstate=values[7],
                painchance=values[8],
                painsound=values[9],
                meleestate=values[10],
                missilestate=values[11],
                deathstate=values[12],
                xdeathstate=values[13],
                deathsound=values[14],
                speed=values[15],
                radius=values[16],
                height=values[17],
                mass=values[18],
                damage=values[19],
                activesound=values[20],
                flags=values[21],
                raisestate=values[22]
            ))

    print(f"  Extracted {len(mobjinfos)} mobjinfo entries")
    return mobjinfos


def parse_flags(flags_str: str) -> int:
    """Parse MF_xxx flag combinations into integer value."""
    # Flag values from p_mobj.h
    MF_FLAGS = {
        'MF_SPECIAL': 0x0001,
        'MF_SOLID': 0x0002,
        'MF_SHOOTABLE': 0x0004,
        'MF_NOSECTOR': 0x0008,
        'MF_NOBLOCKMAP': 0x0010,
        'MF_AMBUSH': 0x0020,
        'MF_JUSTHIT': 0x0040,
        'MF_JUSTATTACKED': 0x0080,
        'MF_SPAWNCEILING': 0x0100,
        'MF_NOGRAVITY': 0x0200,
        'MF_DROPOFF': 0x0400,
        'MF_PICKUP': 0x0800,
        'MF_NOCLIP': 0x1000,
        'MF_SLIDE': 0x2000,
        'MF_FLOAT': 0x4000,
        'MF_TELEPORT': 0x8000,
        'MF_MISSILE': 0x10000,
        'MF_DROPPED': 0x20000,
        'MF_SHADOW': 0x40000,
        'MF_NOBLOOD': 0x80000,
        'MF_CORPSE': 0x100000,
        'MF_INFLOAT': 0x200000,
        'MF_COUNTKILL': 0x400000,
        'MF_COUNTITEM': 0x800000,
        'MF_SKULLFLY': 0x1000000,
        'MF_NOTDMATCH': 0x2000000,
        'MF_TRANSLATION': 0xc000000,
        'MF_TRANSSHIFT': 26,
    }

    result = 0
    for flag_name, flag_value in MF_FLAGS.items():
        if flag_name in flags_str:
            result |= flag_value

    return result


def generate_header(rndtable: List[int], finesine: List[int],
                    states: List[State], mobjinfos: List[MobjInfo],
                    output_path: Path) -> None:
    """Generate doom_data.cuh with all extracted data."""

    lines = [
        "/**",
        " * GPU DOOM Data Tables",
        " *",
        " * AUTO-GENERATED from original linuxdoom-1.10 source",
        " * DO NOT EDIT - regenerate with extract_doom_data.py",
        " *",
        " * Source: id Software DOOM (1993)",
        " * License: DOOM Source Code License",
        " */",
        "",
        "#ifndef DOOM_DATA_CUH",
        "#define DOOM_DATA_CUH",
        "",
        "#include <cstdint>",
        "",
        "// Fixed point (16.16)",
        "typedef int32_t fixed_t;",
        "#define FRACBITS 16",
        "#define FRACUNIT (1 << FRACBITS)",
        "",
        "// =============================================================================",
        "// Random Number Table (from m_random.c)",
        "// =============================================================================",
        "",
        f"#define RNDTABLE_SIZE 256",
        "",
        "__constant__ uint8_t c_rndtable[RNDTABLE_SIZE] = {",
    ]

    # Output rndtable in rows of 16
    for i in range(0, 256, 16):
        row = rndtable[i:i+16]
        lines.append("    " + ", ".join(f"{x:3d}" for x in row) + ",")
    lines.append("};")
    lines.append("")

    # Finesine table
    lines.extend([
        "// =============================================================================",
        "// Sine Lookup Table (from tables.c)",
        "// =============================================================================",
        "",
        f"#define FINEANGLES 8192",
        f"#define FINEMASK (FINEANGLES - 1)",
        f"#define FINESINE_SIZE {len(finesine)}",
        "",
        "__constant__ fixed_t c_finesine[FINESINE_SIZE] = {",
    ])

    # Output finesine in rows of 8
    for i in range(0, len(finesine), 8):
        row = finesine[i:i+8]
        lines.append("    " + ", ".join(f"{x:12d}" for x in row) + ",")
    lines.append("};")
    lines.append("")

    # States table
    lines.extend([
        "// =============================================================================",
        "// State Machine States (from info.c)",
        "// =============================================================================",
        "",
        "struct GPUState {",
        "    int16_t sprite;      // Sprite number",
        "    int16_t frame;       // Frame (bit 15 = full bright)",
        "    int16_t tics;        // Duration (-1 = infinite)",
        "    int16_t nextstate;   // Next state index",
        "    int16_t misc1;",
        "    int16_t misc2;",
        "    uint8_t action_id;   // Action function ID (0 = none)",
        "    uint8_t _pad;",
        "};",
        "",
        f"#define NUMSTATES {len(states)}",
        "",
        "__constant__ GPUState c_states[NUMSTATES] = {",
    ])

    # Map action names to IDs
    action_map = {"NULL": 0}
    action_id = 1
    for s in states:
        if s.action not in action_map:
            action_map[s.action] = action_id
            action_id += 1

    for i, s in enumerate(states):
        action = action_map.get(s.action, 0)
        lines.append(f"    {{{s.sprite:3d}, {s.frame:5d}, {s.tics:3d}, {s.nextstate:4d}, "
                     f"{s.misc1}, {s.misc2}, {action:3d}, 0}},  // {i}")

    lines.append("};")
    lines.append("")

    # Mobjinfo table
    lines.extend([
        "// =============================================================================",
        "// Map Object Info (from info.c)",
        "// =============================================================================",
        "",
        "struct GPUMobjInfo {",
        "    int32_t doomednum;",
        "    int16_t spawnstate;",
        "    int16_t spawnhealth;",
        "    int16_t seestate;",
        "    int16_t seesound;",
        "    int16_t reactiontime;",
        "    int16_t attacksound;",
        "    int16_t painstate;",
        "    int16_t painchance;",
        "    int16_t painsound;",
        "    int16_t meleestate;",
        "    int16_t missilestate;",
        "    int16_t deathstate;",
        "    int16_t xdeathstate;",
        "    int16_t deathsound;",
        "    int32_t speed;",
        "    fixed_t radius;",
        "    fixed_t height;",
        "    int32_t mass;",
        "    int32_t damage;",
        "    int16_t activesound;",
        "    uint32_t flags;",
        "    int16_t raisestate;",
        "    int16_t _pad;",
        "};",
        "",
        f"#define NUMMOBJTYPES {len(mobjinfos)}",
        "",
        "__constant__ GPUMobjInfo c_mobjinfo[NUMMOBJTYPES] = {",
    ])

    for i, m in enumerate(mobjinfos):
        lines.append(f"    {{{m.doomednum}, {m.spawnstate}, {m.spawnhealth}, "
                     f"{m.seestate}, {m.seesound}, {m.reactiontime}, {m.attacksound}, "
                     f"{m.painstate}, {m.painchance}, {m.painsound}, "
                     f"{m.meleestate}, {m.missilestate}, {m.deathstate}, {m.xdeathstate}, "
                     f"{m.deathsound}, {m.speed}, {m.radius}, {m.height}, "
                     f"{m.mass}, {m.damage}, {m.activesound}, 0x{m.flags:08x}u, "
                     f"{m.raisestate}, 0}},  // {i}")

    lines.append("};")
    lines.append("")

    # Action function enum
    lines.extend([
        "// =============================================================================",
        "// Action Function IDs",
        "// =============================================================================",
        "",
        "enum ActionID : uint8_t {",
    ])

    for action, aid in sorted(action_map.items(), key=lambda x: x[1]):
        lines.append(f"    ACTION_{action.upper()} = {aid},")

    lines.extend([
        "    ACTION_COUNT",
        "};",
        "",
        "#endif // DOOM_DATA_CUH",
        "",
    ])

    output_path.write_text("\n".join(lines))
    print(f"\nGenerated: {output_path}")


def verify_extraction(rndtable: List[int], finesine: List[int],
                      states: List[State], mobjinfos: List[MobjInfo]) -> bool:
    """Verify extracted data against known values."""
    print("\n=== Verification ===")

    errors = 0

    # Verify rndtable
    if len(rndtable) != 256:
        print(f"ERROR: rndtable has {len(rndtable)} entries, expected 256")
        errors += 1
    elif rndtable[0] != 0 or rndtable[1] != 8 or rndtable[2] != 109:
        print(f"ERROR: rndtable starts with {rndtable[:3]}, expected [0, 8, 109]")
        errors += 1
    else:
        print(f"OK: rndtable[256] - first 3: {rndtable[:3]}")

    # Verify finesine - should have 10240 entries
    # Actually DOOM uses 5*FINEANGLES + (FINEANGLES/4) = 5*8192 + 2048 = 43008? No.
    # Looking at tables.c: finesine[10240] = 8192 + 2048 for cosine
    if len(finesine) < 8192:
        print(f"ERROR: finesine has {len(finesine)} entries, expected >= 8192")
        errors += 1
    else:
        print(f"OK: finesine[{len(finesine)}] - first entry: {finesine[0]}")

    # Verify states
    if len(states) < 100:
        print(f"ERROR: Only {len(states)} states extracted, expected ~1000")
        errors += 1
    else:
        print(f"OK: states[{len(states)}]")

    # Verify mobjinfo
    if len(mobjinfos) < 100:
        print(f"ERROR: Only {len(mobjinfos)} mobjinfo extracted, expected 138+")
        errors += 1
    else:
        # Check MT_PLAYER (index 0)
        player = mobjinfos[0]
        if player.spawnhealth != 100:
            print(f"ERROR: MT_PLAYER spawnhealth is {player.spawnhealth}, expected 100")
            errors += 1
        elif player.radius != 16 << 16:  # 16*FRACUNIT
            print(f"ERROR: MT_PLAYER radius is {player.radius}, expected {16 << 16}")
            errors += 1
        else:
            print(f"OK: mobjinfo[{len(mobjinfos)}] - MT_PLAYER health=100, radius=16*FRACUNIT")

    if errors == 0:
        print("\nAll verifications passed!")
        return True
    else:
        print(f"\n{errors} verification(s) failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract DOOM data tables")
    parser.add_argument("--verify", action="store_true",
                        help="Verify extracted data against known values")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: ../doom_data.cuh)")
    parser.add_argument("--source", type=str, default=None,
                        help="DOOM source directory (default: ../source/linuxdoom-1.10)")

    args = parser.parse_args()

    source_dir = Path(args.source) if args.source else DOOM_SOURCE
    output_path = Path(args.output) if args.output else GPU_DIR / "doom_data.cuh"

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        print("Download linuxdoom-1.10 and extract to source/linuxdoom-1.10/")
        sys.exit(1)

    print(f"=== DOOM Data Extractor ===")
    print(f"Source: {source_dir}")
    print(f"Output: {output_path}")
    print()

    # Extract all data
    print("Extracting rndtable from m_random.c...")
    rndtable = extract_rndtable(source_dir)

    print("Extracting finesine from tables.c...")
    finesine = extract_finesine(source_dir)

    print("Parsing enums from info.h...")
    state_enums = parse_state_enums(source_dir)
    sprite_enums = parse_sprite_enums(source_dir)
    sound_enums = parse_sound_enums(source_dir)

    print("Extracting states from info.c...")
    states = extract_states(source_dir, state_enums, sprite_enums)

    print("Extracting mobjinfo from info.c...")
    mobjinfo_types = parse_mobjtype_enums(source_dir)
    mobjinfos = extract_mobjinfo(source_dir, state_enums, sound_enums)

    if args.verify:
        if not verify_extraction(rndtable, finesine, states, mobjinfos):
            sys.exit(1)

    # Generate header
    generate_header(rndtable, finesine, states, mobjinfos, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
