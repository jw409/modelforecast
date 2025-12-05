"""
Borg Configuration Tool

Provides configure_borg() for LLM model interface.
Reads and writes borg.txt configuration files.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional
import shutil
from datetime import datetime


# Default borg.txt location
DEFAULT_BORG_PATH = Path.home() / ".angband" / "Angband" / "borg.txt"

# Valid configuration keys and their types
BORG_CONFIG_SCHEMA = {
    # Worship variables (bool)
    "borg_worships_damage": bool,
    "borg_worships_speed": bool,
    "borg_worships_hp": bool,
    "borg_worships_mana": bool,
    "borg_worships_ac": bool,
    "borg_worships_gold": bool,

    # Risk tolerance (bool)
    "borg_plays_risky": bool,
    "borg_kills_uniques": bool,

    # Swap items (bool)
    "borg_uses_swaps": bool,

    # Strange options (bool)
    "borg_allow_strange_opts": bool,

    # Cheat death (bool)
    "borg_cheat_death": bool,

    # Respawn settings
    "borg_respawn_race": int,  # -1 = random, 0-10 = specific race
    "borg_respawn_class": int,  # -1 = random, 0-8 = specific class
    "borg_respawn_winners": bool,

    # Dumps
    "borg_dump_level": int,

    # Delay
    "borg_delay_factor": int,

    # Verbose
    "borg_verbose": bool,

    # Dynamic calculations
    "borg_uses_dynamic_calcs": bool,

    # Depth limits
    "borg_stop_dlevel": int,
    "borg_stop_clevel": int,
    "borg_no_deeper": int,
}


@dataclass
class BorgConfig:
    """Represents current borg configuration state."""
    values: dict = field(default_factory=dict)
    path: Path = DEFAULT_BORG_PATH
    raw_content: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values


def read_borg_config(path: Optional[Path] = None) -> BorgConfig:
    """
    Read and parse borg.txt configuration file.

    Returns:
        BorgConfig with parsed values
    """
    path = Path(path) if path else DEFAULT_BORG_PATH

    if not path.exists():
        raise FileNotFoundError(f"borg.txt not found at {path}")

    content = path.read_text()
    values = {}

    # Parse key = value lines
    for line in content.split('\n'):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue

        # Match key = value pattern
        match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
        if match:
            key, value = match.groups()
            key = key.lower()
            value = value.strip()

            # Convert to appropriate type
            if key in BORG_CONFIG_SCHEMA:
                expected_type = BORG_CONFIG_SCHEMA[key]
                if expected_type == bool:
                    values[key] = value.upper() == 'TRUE'
                elif expected_type == int:
                    try:
                        values[key] = int(value)
                    except ValueError:
                        values[key] = value
                else:
                    values[key] = value
            else:
                # Unknown key, store as string
                values[key] = value

    return BorgConfig(values=values, path=path, raw_content=content)


@dataclass
class ConfigResult:
    """Result of configure_borg() call."""
    success: bool
    message: str
    diff: dict  # Changed values: {key: (old, new)}
    errors: list  # Any validation errors
    backup_path: Optional[Path] = None


def configure_borg(
    params: dict,
    path: Optional[Path] = None,
    backup: bool = True,
    validate: bool = True
) -> ConfigResult:
    """
    Configure the borg by modifying borg.txt.

    This is the primary LLM tool interface for changing borg behavior.

    Args:
        params: Dict of {config_key: new_value} to set
        path: Path to borg.txt (default: ~/.angband/Angband/borg.txt)
        backup: Create backup before modifying (default: True)
        validate: Validate keys against schema (default: True)

    Returns:
        ConfigResult with success status, diff, and any errors

    Example:
        >>> result = configure_borg({
        ...     "borg_worships_speed": True,
        ...     "borg_plays_risky": True,
        ...     "borg_no_deeper": 50
        ... })
        >>> print(result.success)
        True
        >>> print(result.diff)
        {'borg_worships_speed': (False, True), 'borg_plays_risky': (False, True)}
    """
    path = Path(path) if path else DEFAULT_BORG_PATH
    errors = []
    diff = {}
    backup_path = None

    # Validate params
    if validate:
        for key in params:
            if key.lower() not in BORG_CONFIG_SCHEMA:
                errors.append(f"Unknown config key: {key}")
            else:
                expected_type = BORG_CONFIG_SCHEMA[key.lower()]
                if not isinstance(params[key], expected_type):
                    # Try to coerce
                    try:
                        if expected_type == bool:
                            if isinstance(params[key], str):
                                params[key] = params[key].upper() == 'TRUE'
                            else:
                                params[key] = bool(params[key])
                        elif expected_type == int:
                            params[key] = int(params[key])
                    except (ValueError, TypeError):
                        errors.append(f"Invalid type for {key}: expected {expected_type.__name__}")

    if errors and validate:
        return ConfigResult(
            success=False,
            message=f"Validation failed: {'; '.join(errors)}",
            diff={},
            errors=errors
        )

    # Read current config
    try:
        current = read_borg_config(path)
    except FileNotFoundError as e:
        return ConfigResult(
            success=False,
            message=str(e),
            diff={},
            errors=[str(e)]
        )

    # Create backup
    if backup:
        backup_path = path.parent / f"borg.txt.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy(path, backup_path)

    # Build new content
    content = current.raw_content

    for key, new_value in params.items():
        key_lower = key.lower()
        old_value = current.get(key_lower)

        # Convert value to borg.txt format
        if isinstance(new_value, bool):
            value_str = "TRUE" if new_value else "FALSE"
        else:
            value_str = str(new_value)

        # Find and replace the line
        # Pattern: key = value (with any whitespace)
        pattern = re.compile(rf'^({key_lower})\s*=\s*.+$', re.MULTILINE | re.IGNORECASE)

        if pattern.search(content):
            content = pattern.sub(f'{key_lower} = {value_str}', content)
            if old_value != new_value:
                diff[key_lower] = (old_value, new_value)
        else:
            # Key doesn't exist, append it
            content += f"\n{key_lower} = {value_str}"
            diff[key_lower] = (None, new_value)

    # Write new content
    path.write_text(content)

    return ConfigResult(
        success=True,
        message=f"Updated {len(diff)} config values",
        diff=diff,
        errors=[],
        backup_path=backup_path
    )


def get_default_config() -> dict:
    """Return default borg configuration values."""
    return {
        "borg_worships_damage": False,
        "borg_worships_speed": False,
        "borg_worships_hp": False,
        "borg_worships_mana": False,
        "borg_worships_ac": False,
        "borg_worships_gold": False,
        "borg_plays_risky": False,
        "borg_kills_uniques": False,
        "borg_uses_swaps": True,
        "borg_allow_strange_opts": False,
        "borg_cheat_death": False,
        "borg_respawn_race": -1,
        "borg_respawn_class": -1,
        "borg_respawn_winners": False,
    }


if __name__ == "__main__":
    # Quick test
    print("Reading borg config...")
    try:
        config = read_borg_config()
        print(f"Found {len(config.values)} config values")
        print(f"  borg_cheat_death: {config.get('borg_cheat_death')}")
        print(f"  borg_plays_risky: {config.get('borg_plays_risky')}")
        print(f"  borg_worships_speed: {config.get('borg_worships_speed')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
