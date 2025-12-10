#!/usr/bin/env python3
"""
DOOM C to GPU Transformer

Systematically transforms original DOOM C code to GPU-parallel code.
Supports multiple backends: CUDA (native), WGSL (WebGPU), and reference C.

Key transformations:
1. Pointer dereference: actor->field → d_mobj_field[idx]
2. Global state: players[0] → d_player_*[instance_id]
3. Function signatures: Add instance_id, num_instances
4. GPU annotations: __device__ (CUDA), @compute (WGSL)
5. Memory layout: Interleaved for coalesced access

Usage:
    python doom_cuda_transformer.py ../source/linuxdoom-1.10/p_enemy.c > p_enemy_gpu.cuh
    python doom_cuda_transformer.py --backend=wgsl ../source/linuxdoom-1.10/p_enemy.c > p_enemy.wgsl
    python doom_cuda_transformer.py --backend=c ../source/linuxdoom-1.10/p_enemy.c > p_enemy_parallel.c

Backends:
    cuda  - NVIDIA CUDA (__device__, __global__)
    wgsl  - WebGPU Shading Language (@compute, @group)
    c     - Portable C with SIMD hints (reference implementation)
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from enum import Enum


class Backend(Enum):
    CUDA = "cuda"
    WGSL = "wgsl"
    C = "c"


# Backend-specific annotations
BACKEND_CONFIG = {
    Backend.CUDA: {
        'device_fn': '__device__',
        'kernel_fn': '__global__',
        'constant': '__constant__',
        'include_header': '#include "doom_types.cuh"',
        'file_ext': '.cuh',
        'skip_hud': True,  # No HUD in CUDA batch mode
    },
    Backend.WGSL: {
        'device_fn': 'fn',
        'kernel_fn': '@compute @workgroup_size(256)',
        'constant': 'const',
        'include_header': '// WebGPU compute shader for DOOM',
        'file_ext': '.wgsl',
        'skip_hud': False,  # Keep HUD for browser
    },
    Backend.C: {
        'device_fn': 'static inline',
        'kernel_fn': '',
        'constant': 'static const',
        'include_header': '#include "doom_types.h"\n#include <stdint.h>',
        'file_ext': '_parallel.c',
        'skip_hud': False,  # Keep HUD for reference impl
    }
}

# mobj_t fields that need transformation (from p_mobj.h)
MOBJ_FIELDS = {
    # Position/movement
    'x': 'fixed_t',
    'y': 'fixed_t',
    'z': 'fixed_t',
    'momx': 'fixed_t',
    'momy': 'fixed_t',
    'momz': 'fixed_t',
    'angle': 'angle_t',
    'floorz': 'fixed_t',
    'ceilingz': 'fixed_t',
    'radius': 'fixed_t',
    'height': 'fixed_t',

    # State
    'health': 'int32_t',
    'flags': 'uint32_t',
    'type': 'uint16_t',  # mobjtype_t
    'tics': 'int16_t',
    'frame': 'int16_t',
    'sprite': 'int16_t',

    # AI
    'movedir': 'uint8_t',
    'movecount': 'int16_t',
    'reactiontime': 'int16_t',
    'threshold': 'int16_t',
    'lastlook': 'int8_t',

    # Pointers → indices
    'target': 'int16_t',  # mobj index, -1 for null
    'tracer': 'int16_t',
    'state': 'int16_t',   # state index
    'subsector': 'int16_t',  # subsector index
    'player': 'int8_t',  # player index, -1 for non-player mobj

    # Spawn data (from mapthing_t)
    'spawnpoint_x': 'int16_t',
    'spawnpoint_y': 'int16_t',
    'spawnpoint_angle': 'int16_t',
    'spawnpoint_type': 'int16_t',
    'spawnpoint_options': 'int16_t',
}

# mobjinfo_t fields (from info.h) - accessed via actor->info->field
MOBJINFO_FIELDS = {
    'doomednum': 'int32_t',
    'spawnstate': 'int32_t',
    'spawnhealth': 'int32_t',
    'seestate': 'int32_t',
    'seesound': 'int32_t',
    'reactiontime': 'int32_t',
    'attacksound': 'int32_t',
    'painstate': 'int32_t',
    'painchance': 'int32_t',
    'painsound': 'int32_t',
    'meleestate': 'int32_t',
    'missilestate': 'int32_t',
    'deathstate': 'int32_t',
    'xdeathstate': 'int32_t',
    'deathsound': 'int32_t',
    'speed': 'int32_t',
    'radius': 'int32_t',
    'height': 'int32_t',
    'mass': 'int32_t',
    'damage': 'int32_t',
    'activesound': 'int32_t',
    'flags': 'int32_t',
    'raisestate': 'int32_t',
}

# sector_t fields (from r_defs.h) - accessed via subsector->sector->field
SECTOR_FIELDS = {
    'floorheight': 'fixed_t',
    'ceilingheight': 'fixed_t',
    'floorpic': 'int16_t',
    'ceilingpic': 'int16_t',
    'lightlevel': 'int16_t',
    'special': 'int16_t',
    'tag': 'int16_t',
    'soundtraversed': 'int32_t',
    'soundtarget': 'int16_t',  # mobj index
    'validcount': 'int32_t',
    'thinglist': 'int16_t',    # first mobj index in sector
    'linecount': 'int32_t',
}

# player_t fields (from d_player.h)
PLAYER_FIELDS = {
    'health': 'int32_t',
    'armorpoints': 'int32_t',
    'armortype': 'int8_t',
    'powers': 'int16_t[6]',
    'cards': 'uint8_t',  # bitfield
    'backpack': 'uint8_t',
    'frags': 'int16_t[4]',
    'readyweapon': 'uint8_t',
    'pendingweapon': 'uint8_t',
    'weaponowned': 'uint8_t[9]',
    'ammo': 'int16_t[4]',
    'maxammo': 'int16_t[4]',
    'attackdown': 'uint8_t',
    'usedown': 'uint8_t',
    'cheats': 'uint32_t',
    'refire': 'int16_t',
    'killcount': 'int16_t',
    'itemcount': 'int16_t',
    'secretcount': 'int16_t',
    'damagecount': 'int16_t',
    'bonuscount': 'int16_t',
    'extralight': 'int8_t',
    'fixedcolormap': 'int8_t',
    'playerstate': 'uint8_t',
    # View/movement
    'viewz': 'fixed_t',
    'viewheight': 'fixed_t',
    'deltaviewheight': 'fixed_t',
    'bob': 'fixed_t',
    # Input
    'cmd': 'ticcmd_t',  # struct - needs special handling
    # Link to mobj
    'mo': 'int16_t',  # mobj index
    'attacker': 'int16_t',  # mobj index of who damaged us
}


@dataclass
class TransformContext:
    """Tracks state during transformation"""
    current_function: str = ""
    local_vars: Dict[str, str] = None  # var_name -> type
    mobj_params: Set[str] = None  # param names that are mobj_t*
    backend: Backend = Backend.CUDA

    def __post_init__(self):
        self.local_vars = {}
        self.mobj_params = set()

    @property
    def config(self):
        return BACKEND_CONFIG[self.backend]


def transform_function_signature(line: str, ctx: TransformContext) -> str:
    """Transform function signature to GPU version"""
    # Match: void A_Chase (mobj_t* actor)
    match = re.match(r'^(\w+)\s+(\w+)\s*\((.*)\)', line)
    if not match:
        return line

    ret_type, func_name, params = match.groups()
    ctx.current_function = func_name
    ctx.mobj_params.clear()

    # Parse parameters
    new_params = []
    for param in params.split(','):
        param = param.strip()
        if not param:
            continue

        # Check for mobj_t* parameters
        if 'mobj_t*' in param or 'mobj_t *' in param:
            # Extract param name
            name = param.split()[-1].replace('*', '')
            ctx.mobj_params.add(name)
            # Convert to index
            new_params.append(f'int {name}_idx')
        else:
            new_params.append(param)

    # Add instance parameters
    new_params.append('int instance_id')
    new_params.append('int num_instances')

    device_fn = ctx.config['device_fn']
    return f'{device_fn} {ret_type} {func_name}_GPU({", ".join(new_params)})'


def transform_field_access(line: str, ctx: TransformContext) -> str:
    """Transform actor->field to d_mobj_field[idx]"""
    result = line

    for mobj_param in ctx.mobj_params:
        idx_expr = f'{mobj_param}_idx * num_instances + instance_id'

        # Handle actor->player->field FIRST (before player becomes index)
        # actor->player->killcount → d_player_killcount[d_mobj_player[idx]]
        player_chain_pattern = rf'{mobj_param}->player->(\w+)'
        def replace_player_chain(m):
            field = m.group(1)
            player_idx = f'd_mobj_player[{idx_expr}]'
            if field in PLAYER_FIELDS:
                return f'd_player_{field}[{player_idx}]'
            return f'/* TODO: player->{field} */ 0'
        result = re.sub(player_chain_pattern, replace_player_chain, result)

        # Handle actor->target->info->field (triple chain)
        triple_info_pattern = rf'{mobj_param}->target->info->(\w+)'
        def replace_triple_info(m):
            field = m.group(1)
            target_idx = f'd_mobj_target[{idx_expr}]'
            target_type = f'd_mobj_type[{target_idx} * num_instances + instance_id]'
            if field in MOBJINFO_FIELDS:
                return f'c_mobjinfo[{target_type}].{field}'
            return f'/* TODO: {mobj_param}->target->info->{field} */ 0'
        result = re.sub(triple_info_pattern, replace_triple_info, result)

        # Handle actor->info->field (double chain)
        info_pattern = rf'{mobj_param}->info->(\w+)'
        def replace_info(m):
            field = m.group(1)
            mobj_type = f'd_mobj_type[{idx_expr}]'
            if field in MOBJINFO_FIELDS:
                return f'c_mobjinfo[{mobj_type}].{field}'
            return f'/* TODO: {mobj_param}->info->{field} */ 0'
        result = re.sub(info_pattern, replace_info, result)

        # Handle actor->subsector->sector->field (triple chain for BSP)
        subsec_sector_pattern = rf'{mobj_param}->subsector->sector->(\w+)'
        def replace_subsec_sector(m):
            field = m.group(1)
            subsec_idx = f'd_mobj_subsector[{idx_expr}]'
            sector_idx = f'd_subsector_sector[{subsec_idx}]'
            if field in SECTOR_FIELDS:
                return f'd_sector_{field}[{sector_idx}]'
            return f'/* TODO: {mobj_param}->subsector->sector->{field} */ 0'
        result = re.sub(subsec_sector_pattern, replace_subsec_sector, result)

        # Handle actor->subsector->sector (returns sector index)
        subsec_sector_bare = rf'{mobj_param}->subsector->sector\b'
        def replace_subsec_sector_bare(m):
            subsec_idx = f'd_mobj_subsector[{idx_expr}]'
            return f'd_subsector_sector[{subsec_idx}]'
        result = re.sub(subsec_sector_bare, replace_subsec_sector_bare, result)

        # Handle actor->target->field (chained mobj access)
        chained_pattern = rf'{mobj_param}->target->(\w+)'
        def replace_chained(m):
            field = m.group(1)
            target_idx = f'd_mobj_target[{idx_expr}]'
            if field in MOBJ_FIELDS:
                return f'd_mobj_{field}[{target_idx} * num_instances + instance_id]'
            return f'/* TODO: {mobj_param}->target->{field} */ 0'
        result = re.sub(chained_pattern, replace_chained, result)

        # Handle actor->tracer->field (chained mobj access)
        tracer_pattern = rf'{mobj_param}->tracer->(\w+)'
        def replace_tracer(m):
            field = m.group(1)
            tracer_idx = f'd_mobj_tracer[{idx_expr}]'
            if field in MOBJ_FIELDS:
                return f'd_mobj_{field}[{tracer_idx} * num_instances + instance_id]'
            return f'/* TODO: {mobj_param}->tracer->{field} */ 0'
        result = re.sub(tracer_pattern, replace_tracer, result)

        # Handle actor->player->field (player struct access through mobj)
        # actor->player is an index, so actor->player->killcount becomes:
        # d_player_killcount[d_mobj_player[idx]]
        player_field_pattern = rf'd_mobj_player\[{mobj_param}_idx \* num_instances \+ instance_id\]->(\w+)'
        def replace_player_field(m):
            field = m.group(1)
            player_idx = f'd_mobj_player[{idx_expr}]'
            if field in PLAYER_FIELDS:
                return f'd_player_{field}[{player_idx}]'
            return f'/* TODO: player->{field} */ 0'
        result = re.sub(player_field_pattern, replace_player_field, result)

        # Handle actor->spawnpoint (struct, return reference to spawnpoint fields)
        spawnpoint_pattern = rf'{mobj_param}->spawnpoint'
        # Just note that spawnpoint is accessed - individual fields handled above
        result = re.sub(rf'{mobj_param}->spawnpoint\.(\w+)',
                       lambda m: f'd_mobj_spawnpoint_{m.group(1)}[{idx_expr}]', result)

        # Handle actor->thinker (GPU handles thinkers differently)
        thinker_pattern = rf'{mobj_param}->thinker\.\w+'
        result = re.sub(thinker_pattern, f'/* GPU_THINKER */', result)

        # Pattern: actor->field (simple single access)
        # IMPORTANT: Use negative lookbehind to avoid matching when mobj_param
        # is part of a chain like player->mo->field (where mo would incorrectly match)
        pattern = rf'(?<!->){mobj_param}->(\w+)'
        def replace_field(m):
            field = m.group(1)
            if field in MOBJ_FIELDS:
                return f'd_mobj_{field}[{idx_expr}]'
            elif field == 'info':
                # info pointer → lookup via type (bare, for further chain)
                return f'c_mobjinfo[d_mobj_type[{idx_expr}]]'
            elif field == 'spawnpoint':
                # spawnpoint struct - return a reference marker
                return f'/* SPAWNPOINT:{mobj_param}_idx */'
            elif field == 'thinker':
                # thinker handled differently in GPU
                return f'/* GPU_THINKER */'
            else:
                # Unknown field - leave a TODO
                return f'/* TODO: {mobj_param}->{field} */ 0'
        result = re.sub(pattern, replace_field, result)

    return result


def transform_player_access(line: str, ctx: TransformContext) -> str:
    """Transform players[0].field and player->field to d_player_field[idx]"""
    result = line

    # Pattern: players[0].field or players[playernum].field
    pattern = r'players\[(\w+)\]\.(\w+)'
    def replace_player(m):
        _player_idx, field = m.groups()
        if field in PLAYER_FIELDS:
            return f'd_player_{field}[instance_id]'
        return f'/* TODO: players[].{field} */ 0'
    result = re.sub(pattern, replace_player, result)

    # Handle player->mo->field (player's mobj) BEFORE player->field
    # In DOOM, player->mo points back to the player's mobj.
    # For GPU: player is from mo->player which is an index, so player->mo
    # means the mobj at that player's mo index.
    # But commonly player = mo->player means player->mo == mo (circular reference).
    # So player->mo->state in the context of P_XYMovement(mobj_t* mo)
    # where player = mo->player means we can use mo_idx directly.
    # For simplicity: use instance_id (single player per instance)
    player_mo_pattern = r'player->mo->(\w+)'
    def replace_player_mo_field(m):
        field = m.group(1)
        # In single-player GPU, player->mo is the player's mobj which is at instance_id
        # Use d_player_mo to get the mobj index for this player
        mo_idx = 'd_player_mo[instance_id]'
        if field in MOBJ_FIELDS:
            return f'd_mobj_{field}[{mo_idx} * num_instances + instance_id]'
        return f'/* TODO: player->mo->{field} */ 0'
    result = re.sub(player_mo_pattern, replace_player_mo_field, result)

    # Also handle the bare player->mo (without chained field)
    result = re.sub(r'player->mo\b(?!->)', 'd_player_mo[instance_id]', result)

    # Handle player->attacker->field (attacker is an mobj index)
    player_attacker_pattern = r'player->attacker->(\w+)'
    def replace_player_attacker(m):
        field = m.group(1)
        attacker_idx = 'd_player_attacker[instance_id]'
        if field in MOBJ_FIELDS:
            return f'd_mobj_{field}[{attacker_idx} * num_instances + instance_id]'
        return f'/* TODO: player->attacker->{field} */ 0'
    result = re.sub(player_attacker_pattern, replace_player_attacker, result)

    # Handle bare player->field where player is a local variable
    # This happens when code does: player = mo->player; player->viewheight = x;
    # We transform to: d_player_viewheight[d_mobj_player[mo_idx...]]
    # But we need to know which mobj_param the player came from
    # For now, use instance_id (assuming single player per instance)
    player_field_pattern = r'player->(\w+)'
    def replace_player_field(m):
        field = m.group(1)
        if field in PLAYER_FIELDS:
            # Use instance_id since GPU instances are 1 player per instance
            return f'd_player_{field}[instance_id]'
        elif field == 'message':
            if ctx.config['skip_hud']:
                # Skip HUD messages in batch mode (CUDA)
                return f'/* GPU_NO_HUD */'
            else:
                # Keep HUD for browser/reference
                return f'd_player_message[instance_id]'
        return f'/* TODO: player->{field} */ 0'
    result = re.sub(player_field_pattern, replace_player_field, result)

    return result


def transform_null_checks(line: str, ctx: TransformContext) -> str:
    """Transform pointer null checks to index checks"""
    result = line

    for mobj_param in ctx.mobj_params:
        # !actor->target → d_mobj_target[idx] < 0
        pattern = rf'!\s*{mobj_param}->target\b'
        idx_expr = f'{mobj_param}_idx * num_instances + instance_id'
        result = re.sub(pattern, f'd_mobj_target[{idx_expr}] < 0', result)

        # actor->target → d_mobj_target[idx] >= 0 (in boolean context)
        # This is trickier - need context

    return result


def transform_function_call(line: str, ctx: TransformContext) -> str:
    """Transform function calls to GPU versions"""
    # Functions that need GPU transformation
    gpu_functions = {
        'P_SetMobjState': 'P_SetMobjState_GPU',
        'P_NewChaseDir': 'P_NewChaseDir_GPU',
        'P_LookForPlayers': 'P_LookForPlayers_GPU',
        'P_CheckMeleeRange': 'P_CheckMeleeRange_GPU',
        'P_CheckMissileRange': 'P_CheckMissileRange_GPU',
        'P_Random': 'gpu_random',
        'P_Move': 'P_Move_GPU',
        'P_TryWalk': 'P_TryWalk_GPU',
        'A_Chase': 'A_Chase_GPU',
        'A_Look': 'A_Look_GPU',
        'A_Fire': 'A_Fire_GPU',
        'P_SpawnMobj': 'P_SpawnMobj_GPU',
        'P_SpawnMissile': 'P_SpawnMissile_GPU',
        'P_DamageMobj': 'P_DamageMobj_GPU',
    }

    # Functions that should be commented out (no GPU equivalent)
    skip_functions = [
        'S_StartSound',
        'S_StopSound',
        'P_StartSound',
        'I_Error',
        'printf',
        'fprintf',
    ]

    result = line

    # Comment out sound/IO functions
    for skip in skip_functions:
        if f'{skip}(' in result or f'{skip} (' in result:
            result = f'    // SKIP-GPU: {result.strip()}'
            return result

    # Transform function names
    for orig, gpu in gpu_functions.items():
        if f'{orig}(' in result or f'{orig} (' in result:
            result = result.replace(orig, gpu)

    # Transform mobj pointer arguments to indices
    for mobj_param in ctx.mobj_params:
        # Pattern: function(actor) or function(actor, ...) or function(..., actor)
        # Replace standalone 'actor' with 'actor_idx' in function calls
        # This is tricky - we need to not replace in other contexts
        # Match: (actor) or (actor, or , actor) or , actor,
        result = re.sub(rf'\(\s*{mobj_param}\s*\)', f'({mobj_param}_idx, instance_id, num_instances)', result)
        result = re.sub(rf'\(\s*{mobj_param}\s*,', f'({mobj_param}_idx, ', result)
        result = re.sub(rf',\s*{mobj_param}\s*\)', f', {mobj_param}_idx)', result)
        result = re.sub(rf',\s*{mobj_param}\s*,', f', {mobj_param}_idx,', result)

        # Assignment: globalvar = actor; → globalvar_idx = actor_idx;
        result = re.sub(rf'=\s*{mobj_param}\s*;', f'= {mobj_param}_idx;', result)

    return result


def transform_line(line: str, ctx: TransformContext) -> str:
    """Apply all transformations to a line"""
    # Skip comments and preprocessor
    stripped = line.strip()
    if stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('/*'):
        return line

    result = line

    # Function signatures: Any function with mobj_t* parameter
    # Match: void A_Chase (mobj_t* actor)
    # Match: boolean P_Move (mobj_t* actor)
    # Match: void P_NoiseAlert (mobj_t* target, mobj_t* emmiter)
    # Match: int P_CheckMeleeRange (mobj_t* actor)
    if re.match(r'^(void|boolean|int|fixed_t|mobj_t\*?)\s+\w+\s*\(', stripped):
        if 'mobj_t' in stripped:
            result = transform_function_signature(stripped, ctx) + '\n'
            return result

    # Skip C includes (keep CUDA includes)
    if stripped.startswith('#include') and not stripped.endswith('.cuh"'):
        return f'// C-INCLUDE: {stripped}\n'

    # Field access (mobj_t* parameter access like actor->field)
    # This also handles actor->player->field chains
    result = transform_field_access(result, ctx)

    # Player access - for standalone player->field (local player variable)
    # Must run AFTER field access to avoid matching player inside mo->player
    result = transform_player_access(result, ctx)

    # Null checks
    result = transform_null_checks(result, ctx)

    # Function calls
    result = transform_function_call(result, ctx)

    return result


def generate_device_arrays():
    """Generate __device__ array declarations"""
    lines = [
        "// =============================================================================",
        "// GPU Device Memory Arrays (generated from mobj_t)",
        "// =============================================================================",
        "",
    ]

    for field, dtype in MOBJ_FIELDS.items():
        lines.append(f"__device__ {dtype}* d_mobj_{field};")

    lines.append("")
    lines.append("// Player state")
    for field, dtype in PLAYER_FIELDS.items():
        if '[' in dtype:
            # Array field - need special handling
            base, size = dtype.rstrip(']').split('[')
            lines.append(f"__device__ {base}* d_player_{field};  // [{size}] per instance")
        else:
            lines.append(f"__device__ {dtype}* d_player_{field};")

    return '\n'.join(lines)


def transform_file(filepath: str, backend: Backend = Backend.CUDA) -> str:
    """Transform an entire C file to GPU code"""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: {filepath} not found", file=sys.stderr)
        return ""

    ctx = TransformContext(backend=backend)
    config = ctx.config
    lines = []

    # Header
    lines.append(f"// GPU-transformed from {path.name}")
    lines.append(f"// Generated by doom_cuda_transformer.py (backend: {backend.value})")
    lines.append("// Original: id Software DOOM (1993)")
    lines.append("")
    lines.append(config['include_header'])
    lines.append("")

    with open(filepath, 'r') as f:
        file_lines = f.readlines()

    i = 0
    while i < len(file_lines):
        line = file_lines[i]

        # Check for multi-line function signature (return type on separate line)
        # Pattern: void\nFuncName\n( params )
        stripped = line.strip()
        if stripped in ('void', 'boolean', 'int', 'fixed_t', 'mobj_t*'):
            # Look ahead for function name and params
            if i + 1 < len(file_lines):
                next_line = file_lines[i + 1].strip()
                # Check if next line is a function name (starts with P_ or A_ or letter)
                if re.match(r'^[A-Za-z_]\w*$', next_line):
                    # Collect the full signature
                    sig_lines = [stripped, next_line]
                    j = i + 2
                    paren_count = 0
                    while j < len(file_lines):
                        sig_part = file_lines[j].strip()
                        sig_lines.append(sig_part)
                        paren_count += sig_part.count('(') - sig_part.count(')')
                        if ')' in sig_part and paren_count <= 0:
                            break
                        j += 1

                    full_sig = ' '.join(sig_lines)
                    # Remove extra whitespace
                    full_sig = re.sub(r'\s+', ' ', full_sig)

                    if 'mobj_t' in full_sig:
                        transformed = transform_function_signature(full_sig, ctx)
                        lines.append(transformed)
                        i = j + 1
                        continue

        transformed = transform_line(line, ctx)
        lines.append(transformed.rstrip())
        i += 1

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Transform DOOM C to GPU code')
    parser.add_argument('source', nargs='?', help='Source C file to transform')
    parser.add_argument('--backend', choices=['cuda', 'wgsl', 'c'], default='cuda',
                       help='Target backend (default: cuda)')
    parser.add_argument('--arrays', action='store_true',
                       help='Generate device array declarations')
    args = parser.parse_args()

    backend = Backend(args.backend)

    if args.arrays:
        print(generate_device_arrays())
    elif args.source:
        result = transform_file(args.source, backend)
        print(result)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
