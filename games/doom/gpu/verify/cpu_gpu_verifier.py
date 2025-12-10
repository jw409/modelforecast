#!/usr/bin/env python3
"""
CPU/GPU DOOM Verification Harness

Compares Original DOOM (CPU reference) vs GPU DOOM using identical inputs.

Architecture:
- CPU Reference: doom_reference binary (original DOOM with instrumentation)
- GPU Implementation: doom_verify binary (single-instance GPU DOOM)
- Both read TicCmd from stdin (binary), output JSONL state to stdout
- Verification: Exact match required for all fields

Reference: /home/jw/dev/game1/contexts/modelforecast/private/docs/plans/DOOM-cpu-vs-gpu.md

Usage:
    python cpu_gpu_verifier.py --ticks 100
    python cpu_gpu_verifier.py --ticks 1000 --verbose
    python cpu_gpu_verifier.py --ticks 10000 --save-states states.json
"""

import os
import sys
import json
import struct
import argparse
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from pathlib import Path

# Paths
GPU_DOOM_DIR = Path(__file__).parent.parent
SOURCE_DIR = GPU_DOOM_DIR.parent / "source" / "linuxdoom-1.10"

# ViZDoom available for optional future use, but NOT primary verification
try:
    from vizdoom import DoomGame, GameVariable, Mode, ScreenResolution, ScreenFormat
    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False


@dataclass
class GameState:
    """Comparable game state snapshot.

    All fields are integers as they come from the binaries.
    Position/angle are fixed-point/angle_t integers, NOT floats.
    """
    tick: int
    # Player position (fixed_t = int32, 16.16 fixed point)
    x: int
    y: int
    z: int
    # Player angle (angle_t = uint32)
    angle: int
    # Player stats (int)
    health: int
    armor: int
    kills: int
    # Player alive (bool -> int 0/1 in JSON)
    alive: int


@dataclass
class TicCmd:
    """Player input command for one tick."""
    forwardmove: int  # -127 to 127
    sidemove: int     # -127 to 127
    angleturn: int    # Angle delta
    buttons: int      # BT_ATTACK=1, BT_USE=2, etc.


def generate_test_inputs(num_ticks: int, seed: int = 42) -> List[TicCmd]:
    """Generate deterministic test inputs for verification."""
    import random
    rng = random.Random(seed)

    inputs = []
    for tick in range(num_ticks):
        cmd = TicCmd(
            forwardmove=50,  # Walk forward
            sidemove=0,
            angleturn=512 if tick % 70 == 0 else 0,  # Turn occasionally
            buttons=0
        )

        # Add some shooting
        if tick % 35 == 0:
            cmd.buttons = 1  # BT_ATTACK

        inputs.append(cmd)

    return inputs


class OriginalDoomRunner:
    """Original DOOM CPU reference using doom_reference binary."""

    def __init__(self, binary_path: Optional[str] = None):
        self.binary = binary_path or str(SOURCE_DIR / "doom_reference")
        self.states: List[GameState] = []

    def run_with_inputs(self, inputs: List[TicCmd]) -> List[GameState]:
        """
        Run Original DOOM with pre-defined inputs and return state snapshots.

        The binary reads TicCmd from stdin as binary and outputs
        state snapshots as JSONL to stdout.
        """
        # Check binary exists
        if not os.path.exists(self.binary):
            print(f"ERROR: CPU reference binary not found: {self.binary}")
            print(f"Build with: cd {SOURCE_DIR} && make doom_reference")
            return []

        # Serialize inputs to binary format (TicCmd struct)
        input_data = self._serialize_inputs(inputs)

        # Run the CPU binary
        try:
            result = subprocess.run(
                [self.binary],
                input=input_data,
                capture_output=True,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            print("ERROR: CPU reference timed out")
            return []
        except Exception as e:
            print(f"ERROR: CPU reference failed: {e}")
            return []

        if result.returncode != 0:
            print(f"ERROR: CPU reference failed with code {result.returncode}")
            stderr = result.stderr.decode()
            if stderr:
                print(f"stderr: {stderr}")
            return []

        # Parse JSONL output
        self.states = self._parse_output(result.stdout)
        return self.states

    def _serialize_inputs(self, inputs: List[TicCmd]) -> bytes:
        """Pack inputs into binary format matching TicCmd struct.

        struct ticcmd_t {
            int8_t forwardmove;    // 1 byte
            int8_t sidemove;       // 1 byte
            int16_t angleturn;     // 2 bytes (little-endian)
            int16_t consistency;   // 2 bytes (little-endian)
            uint8_t chatchar;      // 1 byte
            uint8_t buttons;       // 1 byte
        }; // Total: 8 bytes
        """
        data = bytearray()
        for cmd in inputs:
            data.extend(struct.pack(
                "<bbhHBB",
                cmd.forwardmove,
                cmd.sidemove,
                cmd.angleturn,
                0,  # consistency
                0,  # chatchar
                cmd.buttons
            ))
        return bytes(data)

    def _parse_output(self, data: bytes) -> List[GameState]:
        """Parse JSONL state output from CPU binary.

        Expected format per line:
        {"tick":0,"x":1072693248,"y":-3811639296,"z":0,"angle":2147483648,
         "health":100,"armor":0,"kills":0,"alive":1}
        """
        states = []
        try:
            for line in data.decode().strip().split('\n'):
                if not line or not line.startswith('{'):
                    continue
                obj = json.loads(line)
                states.append(GameState(
                    tick=obj['tick'],
                    x=obj['x'],
                    y=obj['y'],
                    z=obj['z'],
                    angle=obj['angle'],
                    health=obj['health'],
                    armor=obj['armor'],
                    kills=obj['kills'],
                    alive=obj['alive']
                ))
        except Exception as e:
            print(f"ERROR: Failed to parse CPU output: {e}")
            return []

        return states


###############################################################################
# VIZDOOM RUNNER - DISABLED FOR PRIMARY VERIFICATION
#
# ViZDoom is based on ZDoom which has modified physics vs original DOOM.
# It's useful for AI training patterns but NOT for correctness verification.
#
# Status: Kept for reference, may be useful for Phase 5 (Arnold-style interface)
# Primary verification: OriginalDoomRunner (above)
###############################################################################

class ViZDoomRunner:
    """CPU reference implementation using ViZDoom.

    WARNING: NOT USED FOR PRIMARY VERIFICATION.
    ViZDoom uses ZDoom physics which differs from original DOOM.
    See plan document for details.
    """

    def __init__(self, scenario: str = "health_gathering"):
        if not VIZDOOM_AVAILABLE:
            raise RuntimeError("ViZDoom not installed")

        self.game = DoomGame()

        # Find scenario and game files
        scenario_path = RESOURCES_DIR / "scenarios" / f"{scenario}.wad"
        game_path = RESOURCES_DIR / "freedoom2.wad"

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario not found: {scenario_path}")
        if not game_path.exists():
            raise FileNotFoundError(f"Game WAD not found: {game_path}")

        self.game.set_doom_scenario_path(str(scenario_path))
        self.game.set_doom_game_path(str(game_path))

        # Minimal config for deterministic comparison
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_window_visible(False)
        self.game.set_mode(Mode.PLAYER)

        # Add game variables we want to compare
        self.game.add_available_game_variable(GameVariable.HEALTH)
        self.game.add_available_game_variable(GameVariable.ARMOR)
        self.game.add_available_game_variable(GameVariable.FRAGCOUNT)
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)
        self.game.add_available_game_variable(GameVariable.ANGLE)

        # Add buttons
        from vizdoom import Button
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.MOVE_BACKWARD)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        self.game.add_available_button(Button.USE)

        self.states: List[GameState] = []

    def start(self, map_id: int = 1):
        """Initialize game on specified map."""
        self.game.set_doom_map(f"map{map_id:02d}")
        self.game.init()
        self.game.new_episode()
        self.states = []

    def ticcmd_to_action(self, cmd: TicCmd) -> List[float]:
        """Convert TicCmd to ViZDoom action format."""
        # [FORWARD, BACKWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT, ATTACK, USE]
        action = [0.0] * 8

        if cmd.forwardmove > 0:
            action[0] = cmd.forwardmove / 127.0
        elif cmd.forwardmove < 0:
            action[1] = -cmd.forwardmove / 127.0

        if cmd.sidemove > 0:
            action[2] = cmd.sidemove / 127.0
        elif cmd.sidemove < 0:
            action[3] = -cmd.sidemove / 127.0

        if cmd.angleturn > 0:
            action[4] = cmd.angleturn / 512.0
        elif cmd.angleturn < 0:
            action[5] = -cmd.angleturn / 512.0

        if cmd.buttons & 1:  # BT_ATTACK
            action[6] = 1.0
        if cmd.buttons & 2:  # BT_USE
            action[7] = 1.0

        return action

    def step(self, tick: int, cmd: TicCmd) -> GameState:
        """Execute one tick and return state snapshot."""
        action = self.ticcmd_to_action(cmd)
        self.game.make_action(action)

        # Get game variables
        state = self.game.get_state()
        if state is None:
            # Player dead or episode ended
            return GameState(
                tick=tick,
                x=0, y=0, z=0, angle=0,
                health=0, armor=0, kills=0, alive=False
            )

        vars = state.game_variables

        game_state = GameState(
            tick=tick,
            x=vars[3],  # POSITION_X
            y=vars[4],  # POSITION_Y
            z=vars[5],  # POSITION_Z
            angle=vars[6],  # ANGLE
            health=int(vars[0]),  # HEALTH
            armor=int(vars[1]),  # ARMOR
            kills=int(vars[2]),  # FRAGCOUNT
            alive=not self.game.is_player_dead()
        )

        self.states.append(game_state)
        return game_state

    def close(self):
        """Clean up."""
        self.game.close()


class GPUDoomRunner:
    """GPU DOOM implementation runner using doom_verify binary."""

    def __init__(self, binary_path: Optional[str] = None):
        self.binary = binary_path or str(GPU_DOOM_DIR / "build" / "doom_verify")
        self.states: List[GameState] = []

    def run_with_inputs(self, inputs: List[TicCmd]) -> List[GameState]:
        """
        Run GPU DOOM with pre-defined inputs and return state snapshots.

        The binary reads TicCmd from stdin as binary and outputs
        state snapshots as JSONL to stdout (same format as CPU reference).
        """
        # Check binary exists
        if not os.path.exists(self.binary):
            print(f"ERROR: GPU binary not found: {self.binary}")
            print(f"Build with: cd {GPU_DOOM_DIR} && make doom_verify")
            return []

        # Serialize inputs to binary format
        input_data = self._serialize_inputs(inputs)

        # Run the GPU binary
        try:
            result = subprocess.run(
                [self.binary],
                input=input_data,
                capture_output=True,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            print("ERROR: GPU binary timed out")
            return []
        except Exception as e:
            print(f"ERROR: GPU binary failed: {e}")
            return []

        if result.returncode != 0:
            print(f"ERROR: GPU binary failed with code {result.returncode}")
            stderr = result.stderr.decode()
            if stderr:
                print(f"stderr: {stderr}")
            return []

        # Parse JSONL output
        self.states = self._parse_output(result.stdout)
        return self.states

    def _serialize_inputs(self, inputs: List[TicCmd]) -> bytes:
        """Pack inputs into binary format matching TicCmd struct.

        struct ticcmd_t {
            int8_t forwardmove;    // 1 byte
            int8_t sidemove;       // 1 byte
            int16_t angleturn;     // 2 bytes (little-endian)
            int16_t consistency;   // 2 bytes (little-endian)
            uint8_t chatchar;      // 1 byte
            uint8_t buttons;       // 1 byte
        }; // Total: 8 bytes
        """
        data = bytearray()
        for cmd in inputs:
            data.extend(struct.pack(
                "<bbhHBB",
                cmd.forwardmove,
                cmd.sidemove,
                cmd.angleturn,
                0,  # consistency
                0,  # chatchar
                cmd.buttons
            ))
        return bytes(data)

    def _parse_output(self, data: bytes) -> List[GameState]:
        """Parse JSONL state output from GPU binary.

        Expected format per line:
        {"tick":0,"x":1072693248,"y":-3811639296,"z":0,"angle":2147483648,
         "health":100,"armor":0,"kills":0,"alive":1}
        """
        states = []
        try:
            for line in data.decode().strip().split('\n'):
                if not line or not line.startswith('{'):
                    continue
                obj = json.loads(line)
                states.append(GameState(
                    tick=obj['tick'],
                    x=obj['x'],
                    y=obj['y'],
                    z=obj['z'],
                    angle=obj['angle'],
                    health=obj['health'],
                    armor=obj['armor'],
                    kills=obj['kills'],
                    alive=obj['alive']
                ))
        except Exception as e:
            print(f"ERROR: Failed to parse GPU output: {e}")
            return []

        return states


def compare_states(cpu: GameState, gpu: GameState) -> Tuple[bool, List[str]]:
    """
    Compare two game states and return (match, differences).

    ALL fields must match exactly (integer comparison).
    Position/angle are fixed-point/angle_t integers, NOT floats.
    """
    diffs = []

    # Position comparison (exact match required - these are int32 fixed_t)
    if cpu.x != gpu.x:
        diffs.append(f"x: CPU={cpu.x} (0x{cpu.x:08x}) GPU={gpu.x} (0x{gpu.x:08x})")
    if cpu.y != gpu.y:
        diffs.append(f"y: CPU={cpu.y} (0x{cpu.y:08x}) GPU={gpu.y} (0x{gpu.y:08x})")
    if cpu.z != gpu.z:
        diffs.append(f"z: CPU={cpu.z} (0x{cpu.z:08x}) GPU={gpu.z} (0x{gpu.z:08x})")

    # Angle (exact match required - this is uint32 angle_t)
    if cpu.angle != gpu.angle:
        diffs.append(f"angle: CPU={cpu.angle} (0x{cpu.angle:08x}) GPU={gpu.angle} (0x{gpu.angle:08x})")

    # Player stats (exact match required)
    if cpu.health != gpu.health:
        diffs.append(f"health: CPU={cpu.health} GPU={gpu.health}")
    if cpu.armor != gpu.armor:
        diffs.append(f"armor: CPU={cpu.armor} GPU={gpu.armor}")
    if cpu.kills != gpu.kills:
        diffs.append(f"kills: CPU={cpu.kills} GPU={gpu.kills}")
    if cpu.alive != gpu.alive:
        diffs.append(f"alive: CPU={cpu.alive} GPU={gpu.alive}")

    return len(diffs) == 0, diffs


def run_verification(
    num_ticks: int = 100,
    verbose: bool = False,
    cpu_binary: Optional[str] = None,
    gpu_binary: Optional[str] = None
) -> bool:
    """
    Run full CPU/GPU verification using Original DOOM vs GPU DOOM.

    Returns True if all states match exactly.
    """
    print("=== Original DOOM (CPU) vs GPU DOOM Verification ===")
    print(f"Ticks: {num_ticks}")
    print()

    # Generate test inputs
    inputs = generate_test_inputs(num_ticks)
    print(f"Generated {len(inputs)} input commands")

    # Run CPU (Original DOOM)
    print("\nRunning Original DOOM (CPU reference)...")
    cpu_runner = OriginalDoomRunner(cpu_binary)
    cpu_runner.run_with_inputs(inputs)

    if not cpu_runner.states:
        print("\nERROR: No CPU states collected. Verification cannot proceed.")
        print(f"Make sure doom_reference is built: cd {SOURCE_DIR} && make doom_reference")
        return False

    print(f"  Collected {len(cpu_runner.states)} states")

    # Run GPU
    print("\nRunning GPU DOOM...")
    gpu_runner = GPUDoomRunner(gpu_binary)
    gpu_runner.run_with_inputs(inputs)

    if not gpu_runner.states:
        print("\nERROR: No GPU states collected. Verification cannot proceed.")
        print(f"Make sure doom_verify is built: cd {GPU_DOOM_DIR} && make doom_verify")
        return False

    print(f"  Collected {len(gpu_runner.states)} states")

    # Compare
    print("\n=== State Comparison (Exact Match Required) ===")

    min_ticks = min(len(cpu_runner.states), len(gpu_runner.states))
    if min_ticks == 0:
        print("ERROR: No states to compare")
        return False

    if len(cpu_runner.states) != len(gpu_runner.states):
        print(f"WARNING: State count mismatch - CPU: {len(cpu_runner.states)}, GPU: {len(gpu_runner.states)}")
        print(f"Comparing first {min_ticks} ticks")

    matches = 0
    mismatches = 0
    first_mismatch_tick = None

    for i in range(min_ticks):
        cpu_state = cpu_runner.states[i]
        gpu_state = gpu_runner.states[i]

        is_match, diffs = compare_states(cpu_state, gpu_state)

        if is_match:
            matches += 1
            if verbose:
                print(f"  Tick {i}: MATCH")
        else:
            mismatches += 1
            if first_mismatch_tick is None:
                first_mismatch_tick = i

            print(f"\n  Tick {i}: MISMATCH")
            print(f"    CPU: {cpu_state}")
            print(f"    GPU: {gpu_state}")
            print(f"    Differences:")
            for diff in diffs:
                print(f"      - {diff}")

    # Summary
    print()
    print("=== Summary ===")
    print(f"Ticks compared: {min_ticks}")
    print(f"Matches: {matches} ({100*matches/min_ticks:.1f}%)")
    print(f"Mismatches: {mismatches} ({100*mismatches/min_ticks:.1f}%)")

    if mismatches > 0:
        print(f"\nFirst divergence at tick: {first_mismatch_tick}")
        print("\nDEBUGGING HINTS:")
        print("- Check random number sync (rndtable[] and prndindex)")
        print("- Check state table indices (states[] array)")
        print("- Check fixed-point math operations")
        print("- Check TicCmd serialization/deserialization")

    if mismatches == 0:
        print("\n✓ VERIFICATION PASSED: CPU and GPU produce IDENTICAL results")
        return True
    else:
        print("\n✗ VERIFICATION FAILED: CPU and GPU differ")
        return False


def save_states_for_analysis(
    states: List[GameState],
    output_path: str = "states.json"
):
    """Save states to JSON for offline analysis."""
    with open(output_path, 'w') as f:
        json.dump([asdict(s) for s in states], f, indent=2)
    print(f"Saved {len(states)} states to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU/GPU DOOM Verification - Compare Original DOOM vs GPU DOOM"
    )
    parser.add_argument("--ticks", type=int, default=100,
                        help="Number of ticks to simulate (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all tick comparisons (even matches)")
    parser.add_argument("--cpu-binary", type=str, default=None,
                        help="Path to doom_reference binary (default: source/linuxdoom-1.10/doom_reference)")
    parser.add_argument("--gpu-binary", type=str, default=None,
                        help="Path to doom_verify binary (default: gpu/build/doom_verify)")
    parser.add_argument("--save-cpu-states", type=str, default=None,
                        help="Save CPU states to JSON file")
    parser.add_argument("--save-gpu-states", type=str, default=None,
                        help="Save GPU states to JSON file")

    args = parser.parse_args()

    success = run_verification(
        num_ticks=args.ticks,
        verbose=args.verbose,
        cpu_binary=args.cpu_binary,
        gpu_binary=args.gpu_binary
    )

    # Optionally save states for offline analysis
    if args.save_cpu_states and success:
        cpu_runner = OriginalDoomRunner(args.cpu_binary)
        inputs = generate_test_inputs(args.ticks)
        cpu_runner.run_with_inputs(inputs)
        save_states_for_analysis(cpu_runner.states, args.save_cpu_states)

    if args.save_gpu_states and success:
        gpu_runner = GPUDoomRunner(args.gpu_binary)
        inputs = generate_test_inputs(args.ticks)
        gpu_runner.run_with_inputs(inputs)
        save_states_for_analysis(gpu_runner.states, args.save_gpu_states)

    sys.exit(0 if success else 1)
