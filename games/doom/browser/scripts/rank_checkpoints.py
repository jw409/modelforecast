#!/usr/bin/env python3
"""
Checkpoint Ranker for DOOM Arena

Analyzes GPU simulation checkpoints and ranks them by "interestingness"
to find the top 10 moments worth publishing to GitHub Pages.

Ranking criteria:
1. Near-death escapes (health drops to <20, then recovers)
2. Multi-kills (3+ enemies in short span)
3. Boss encounters (Cyberdemon, Spider Mastermind)
4. Speed records (fastest clear of section)
5. Creative solutions (unusual paths, infighting triggers)
6. Dramatic deaths (spectacular failures)
"""

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Checkpoint structure (matches GPU doom_types.cuh Checkpoint)
CHECKPOINT_FORMAT = '<i hh hhhh ii I hhhBBBBBBBB'  # 64 bytes
CHECKPOINT_SIZE = 64

# Monster type IDs (from doom_types.cuh)
BOSS_TYPES = {
    10: 'Spider Mastermind',
    11: 'Cyberdemon',
}


@dataclass
class Checkpoint:
    tick: int
    health: int
    armor: int
    ammo: tuple  # (clip, shell, cell, misl)
    x: int
    y: int
    z: int
    angle: int
    kills: int
    items: int
    secrets: int
    alive: bool
    weapon: int
    monsters_visible: int
    projectiles_nearby: int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Checkpoint':
        """Parse checkpoint from binary data."""
        fields = struct.unpack(CHECKPOINT_FORMAT, data[:CHECKPOINT_SIZE])
        return cls(
            tick=fields[0],
            health=fields[1],
            armor=fields[2],
            ammo=(fields[3], fields[4], fields[5], fields[6]),
            x=fields[7],
            y=fields[8],
            z=fields[9],
            angle=fields[10],
            kills=fields[11],
            items=fields[12],
            secrets=fields[13],
            alive=bool(fields[14]),
            weapon=fields[15],
            monsters_visible=fields[16],
            projectiles_nearby=fields[17],
        )


@dataclass
class RankedMoment:
    """A ranked interesting moment from the simulation."""
    checkpoint_file: str
    tick: int
    score: float
    reason: str
    health: int
    kills: int
    description: str


def score_near_death_escape(checkpoints: List[Checkpoint], idx: int) -> tuple:
    """
    Score near-death escapes: health drops to <20, then recovers.
    Returns (score, description) or (0, None) if not applicable.
    """
    if idx < 2 or idx >= len(checkpoints) - 5:
        return 0, None

    current = checkpoints[idx]
    if current.health >= 20 or not current.alive:
        return 0, None

    # Look ahead: did we recover?
    recovered = False
    peak_health = current.health
    for future in checkpoints[idx + 1:idx + 10]:
        if future.health > peak_health:
            peak_health = future.health
        if future.health >= 50:
            recovered = True
            break

    if recovered:
        # Score based on how close to death and how dramatic recovery
        danger_score = (20 - current.health) * 5  # Max 100 at 0 health
        recovery_score = (peak_health - current.health) * 0.5
        total = danger_score + recovery_score

        return total, f"Dropped to {current.health} HP, recovered to {peak_health}"

    return 0, None


def score_multi_kill(checkpoints: List[Checkpoint], idx: int) -> tuple:
    """
    Score multi-kills: 3+ enemies killed in 5 ticks.
    """
    if idx < 1 or idx >= len(checkpoints) - 5:
        return 0, None

    current = checkpoints[idx]
    past = checkpoints[idx - 1]

    # Look at kill count increase over window
    kills_start = past.kills
    kills_end = current.kills

    for future in checkpoints[idx + 1:min(idx + 6, len(checkpoints))]:
        if future.kills > kills_end:
            kills_end = future.kills

    kills_in_window = kills_end - kills_start
    if kills_in_window >= 3:
        score = kills_in_window * 30  # 90+ for triple kill
        return score, f"{kills_in_window}-kill streak"

    return 0, None


def score_boss_encounter(checkpoints: List[Checkpoint], idx: int) -> tuple:
    """
    Score boss encounters based on monster visibility and health tension.
    Note: Full implementation needs mobj tracking from GPU.
    """
    current = checkpoints[idx]

    # Heuristic: High monster visibility + taking damage = boss fight
    if current.monsters_visible >= 3 and current.projectiles_nearby >= 2:
        if current.health < 60:
            return 70, f"Intense combat: {current.monsters_visible} enemies visible"

    return 0, None


def score_dramatic_death(checkpoints: List[Checkpoint], idx: int) -> tuple:
    """
    Score dramatic deaths: Was alive, lots happening, then dead.
    """
    if idx < 1 or idx >= len(checkpoints):
        return 0, None

    past = checkpoints[idx - 1]
    current = checkpoints[idx]

    if past.alive and not current.alive:
        drama_score = 0
        reasons = []

        # More drama if surrounded
        if past.monsters_visible >= 3:
            drama_score += past.monsters_visible * 10
            reasons.append(f"{past.monsters_visible} enemies")

        # More drama if incoming projectiles
        if past.projectiles_nearby >= 2:
            drama_score += past.projectiles_nearby * 15
            reasons.append(f"{past.projectiles_nearby} projectiles")

        # More drama if had lots of kills
        if past.kills >= 20:
            drama_score += 30
            reasons.append(f"{past.kills} kills achieved")

        if drama_score > 0:
            return drama_score, f"Death: {', '.join(reasons)}"

    return 0, None


def analyze_checkpoints(checkpoint_dir: Path, instance_id: int = 0) -> List[RankedMoment]:
    """
    Analyze all checkpoints for an instance and return ranked moments.
    """
    moments: List[RankedMoment] = []

    # Load checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob(f'instance_{instance_id:04d}_*.bin'))
    if not checkpoint_files:
        # Try alternate naming
        checkpoint_files = sorted(checkpoint_dir.glob('*.bin'))

    checkpoints: List[Checkpoint] = []
    file_map: dict = {}  # tick -> filename

    for cp_file in checkpoint_files:
        try:
            data = cp_file.read_bytes()
            cp = Checkpoint.from_bytes(data)
            checkpoints.append(cp)
            file_map[cp.tick] = cp_file.name
        except Exception as e:
            print(f"Warning: Failed to parse {cp_file}: {e}", file=sys.stderr)

    if not checkpoints:
        print(f"No valid checkpoints found in {checkpoint_dir}", file=sys.stderr)
        return []

    # Sort by tick
    checkpoints.sort(key=lambda c: c.tick)

    # Score each checkpoint with each heuristic
    scorers = [
        ('near_death', score_near_death_escape),
        ('multi_kill', score_multi_kill),
        ('boss', score_boss_encounter),
        ('death', score_dramatic_death),
    ]

    for idx, cp in enumerate(checkpoints):
        for name, scorer in scorers:
            score, desc = scorer(checkpoints, idx)
            if score > 0 and desc:
                moments.append(RankedMoment(
                    checkpoint_file=file_map.get(cp.tick, f'tick_{cp.tick}.bin'),
                    tick=cp.tick,
                    score=score,
                    reason=name,
                    health=cp.health,
                    kills=cp.kills,
                    description=desc,
                ))

    # Sort by score descending
    moments.sort(key=lambda m: -m.score)

    return moments


def main():
    parser = argparse.ArgumentParser(
        description='Rank DOOM checkpoints by interestingness'
    )
    parser.add_argument('checkpoint_dir', type=Path,
                        help='Directory containing checkpoint .bin files')
    parser.add_argument('--top', type=int, default=10,
                        help='Number of top moments to output (default: 10)')
    parser.add_argument('--instance', type=int, default=0,
                        help='Instance ID to analyze (default: 0)')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')
    parser.add_argument('--output', type=Path,
                        help='Output file (default: stdout)')

    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(f"Error: Directory not found: {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    moments = analyze_checkpoints(args.checkpoint_dir, args.instance)

    if not moments:
        print("No interesting moments found.", file=sys.stderr)
        sys.exit(0)

    # Take top N
    top_moments = moments[:args.top]

    # Format output
    if args.json:
        output = json.dumps([
            {
                'rank': i + 1,
                'checkpoint': m.checkpoint_file,
                'tick': m.tick,
                'score': m.score,
                'reason': m.reason,
                'health': m.health,
                'kills': m.kills,
                'description': m.description,
            }
            for i, m in enumerate(top_moments)
        ], indent=2)
    else:
        lines = ["DOOM Arena - Top Interesting Moments", "=" * 40, ""]
        for i, m in enumerate(top_moments):
            lines.append(f"#{i+1}: {m.description}")
            lines.append(f"    File: {m.checkpoint_file}")
            lines.append(f"    Tick: {m.tick} | Health: {m.health} | Kills: {m.kills}")
            lines.append(f"    Score: {m.score:.1f} ({m.reason})")
            lines.append("")
        output = '\n'.join(lines)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Wrote {len(top_moments)} moments to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
