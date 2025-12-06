#!/usr/bin/env python3
"""
Convert GPU DOOM scenario JSON to DOOM demo (.lmp) format.

DOOM demo format:
- Byte 0: Version (109 for DOOM 1.9)
- Byte 1: Skill (0-4, where 0=I'm too young to die, 4=Nightmare)
- Byte 2: Episode (1-4)
- Byte 3: Map (1-9)
- Bytes 4+: TicCmds (4 bytes each: forwardmove, sidemove, angleturn_hi, buttons)
- End: 0x80 terminator

Usage:
    python convert_to_demo.py scenario.json output.lmp
    python convert_to_demo.py --all scenarios/ demos/
"""

import json
import struct
import argparse
import os
from pathlib import Path


def parse_level(level_str: str) -> tuple[int, int]:
    """Parse level string like 'E1M1' to (episode, map)."""
    level_str = level_str.upper()
    if len(level_str) >= 4 and level_str[0] == 'E' and level_str[2] == 'M':
        try:
            episode = int(level_str[1])
            map_num = int(level_str[3])
            return (episode, map_num)
        except ValueError:
            pass
    return (1, 1)  # Default to E1M1


def scenario_to_demo(scenario: dict, skill: int = 3) -> bytes:
    """
    Convert a scenario JSON to DOOM demo format.

    Args:
        scenario: Parsed JSON scenario data
        skill: Skill level 0-4 (default 3 = Ultra-Violence)

    Returns:
        bytes: Demo file content
    """
    output = bytearray()

    # Header
    output.append(109)  # Version (DOOM 1.9)
    output.append(skill)  # Skill level

    # Parse level
    episode, map_num = parse_level(scenario.get('level', 'E1M1'))
    output.append(episode)
    output.append(map_num)

    # TicCmds
    input_history = scenario.get('input_history', [])
    for cmd in input_history:
        forwardmove = cmd.get('forward', 0)
        sidemove = cmd.get('side', 0)
        angleturn = cmd.get('angle', 0)
        buttons = cmd.get('buttons', 0)

        # Convert to bytes (signed for forwardmove/sidemove)
        output.append(forwardmove & 0xFF)
        output.append(sidemove & 0xFF)
        output.append((angleturn >> 8) & 0xFF)  # High byte of angle turn
        output.append(buttons & 0xFF)

    # Terminator
    output.append(0x80)

    return bytes(output)


def convert_file(input_path: str, output_path: str, skill: int = 3) -> bool:
    """Convert a single JSON file to .lmp demo."""
    try:
        with open(input_path, 'r') as f:
            scenario = json.load(f)

        demo_data = scenario_to_demo(scenario, skill)

        with open(output_path, 'wb') as f:
            f.write(demo_data)

        print(f"  {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({len(demo_data)} bytes)")
        return True

    except Exception as e:
        print(f"  ERROR: {input_path}: {e}")
        return False


def convert_all(input_dir: str, output_dir: str, skill: int = 3) -> int:
    """Convert all JSON files in a directory to .lmp demos."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converted = 0
    json_files = list(input_path.glob('*.json'))

    # Skip index.json
    json_files = [f for f in json_files if f.name != 'index.json']

    print(f"Converting {len(json_files)} scenarios...")

    for json_file in sorted(json_files):
        lmp_name = json_file.stem + '.lmp'
        lmp_path = output_path / lmp_name

        if convert_file(str(json_file), str(lmp_path), skill):
            converted += 1

    return converted


def main():
    parser = argparse.ArgumentParser(
        description='Convert GPU DOOM scenario JSON to DOOM demo (.lmp) format'
    )
    parser.add_argument('input', help='Input JSON file or directory (with --all)')
    parser.add_argument('output', help='Output .lmp file or directory (with --all)')
    parser.add_argument('--all', action='store_true',
                       help='Convert all JSON files in input directory')
    parser.add_argument('--skill', type=int, default=3, choices=[0, 1, 2, 3, 4],
                       help='Skill level (0=Easy, 3=UV, 4=Nightmare)')

    args = parser.parse_args()

    if args.all:
        converted = convert_all(args.input, args.output, args.skill)
        print(f"\nConverted {converted} scenarios to .lmp demos")
    else:
        if convert_file(args.input, args.output, args.skill):
            print("Conversion complete!")
        else:
            exit(1)


if __name__ == '__main__':
    main()
