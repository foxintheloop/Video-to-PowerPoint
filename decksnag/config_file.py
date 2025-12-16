"""Configuration file support for DeckSnag.

Supports loading and saving configuration from TOML files with a
configuration hierarchy:
  1. Defaults (in code)
  2. User config file (~/.decksnag/config.toml or %APPDATA%/DeckSnag/config.toml)
  3. Environment variables (DECKSNAG_*)
  4. CLI arguments (highest priority)
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

logger = logging.getLogger("decksnag")


def get_user_config_dir() -> Path:
    """Get the user configuration directory for DeckSnag.

    Returns:
        Path to the configuration directory:
        - Windows: %APPDATA%/DeckSnag
        - macOS: ~/Library/Application Support/DeckSnag
        - Linux: ~/.config/decksnag
    """
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "DeckSnag"
    elif os.name == "posix":
        # Check for macOS
        if Path("/Library").exists():
            return Path.home() / "Library" / "Application Support" / "DeckSnag"
        else:
            # Linux/Unix
            xdg_config = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
            return Path(xdg_config) / "decksnag"
    else:
        return Path.home() / ".decksnag"


def get_user_data_dir() -> Path:
    """Get the user data directory for DeckSnag (for models, cache, etc.).

    Returns:
        Path to the data directory:
        - Windows: %LOCALAPPDATA%/DeckSnag
        - macOS: ~/Library/Application Support/DeckSnag
        - Linux: ~/.local/share/decksnag
    """
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "DeckSnag"
    elif os.name == "posix":
        # Check for macOS
        if Path("/Library").exists():
            return Path.home() / "Library" / "Application Support" / "DeckSnag"
        else:
            # Linux/Unix
            xdg_data = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
            return Path(xdg_data) / "decksnag"
    else:
        return Path.home() / ".decksnag"


def get_default_config_path() -> Path:
    """Get the default path for the user configuration file.

    Returns:
        Path to config.toml in the user config directory.
    """
    return get_user_config_dir() / "config.toml"


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists.

    Returns:
        Path to the configuration directory.
    """
    config_dir = get_user_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def ensure_data_dir() -> Path:
    """Ensure the data directory exists.

    Returns:
        Path to the data directory.
    """
    data_dir = get_user_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_config_file(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from a TOML file.

    Args:
        path: Path to the config file. If None, uses the default location.

    Returns:
        Dictionary of configuration values (empty dict if file doesn't exist).
    """
    if path is None:
        path = get_default_config_path()

    if not path.exists():
        logger.debug(f"Config file not found: {path}")
        return {}

    try:
        with open(path, "rb") as f:
            config = tomllib.load(f)
        logger.info(f"Loaded config from: {path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load config file {path}: {e}")
        return {}


def save_config_file(config: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save configuration to a TOML file.

    Args:
        config: Dictionary of configuration values.
        path: Path to the config file. If None, uses the default location.

    Returns:
        Path where the config was saved.
    """
    if path is None:
        ensure_config_dir()
        path = get_default_config_path()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# DeckSnag Configuration File", "# Generated automatically", ""]

    for key, value in config.items():
        if value is None:
            continue
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, (list, tuple)):
            # Format as TOML array
            formatted = ", ".join(str(v) for v in value)
            lines.append(f"{key} = [{formatted}]")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Saved config to: {path}")
    return path


def load_env_config() -> Dict[str, Any]:
    """Load configuration from environment variables.

    Supported environment variables:
    - DECKSNAG_OUTPUT_PATH
    - DECKSNAG_OUTPUT_FORMAT
    - DECKSNAG_INTERVAL
    - DECKSNAG_THRESHOLD
    - DECKSNAG_METHOD
    - DECKSNAG_STOP_HOTKEY
    - DECKSNAG_MONITOR
    - DECKSNAG_REGION (format: x1,y1,x2,y2)
    - DECKSNAG_VERBOSE (true/false)

    Returns:
        Dictionary of configuration values from environment.
    """
    config: Dict[str, Any] = {}

    env_mapping = {
        "DECKSNAG_OUTPUT_PATH": ("output_path", str),
        "DECKSNAG_OUTPUT_FORMAT": ("output_format", str),
        "DECKSNAG_INTERVAL": ("interval", float),
        "DECKSNAG_THRESHOLD": ("threshold", float),
        "DECKSNAG_METHOD": ("method", str),
        "DECKSNAG_STOP_HOTKEY": ("stop_hotkey", str),
        "DECKSNAG_MONITOR": ("monitor", int),
        "DECKSNAG_VERBOSE": ("verbose", lambda x: x.lower() in ("true", "1", "yes")),
    }

    for env_var, (config_key, converter) in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                config[config_key] = converter(value)
                logger.debug(f"Loaded {config_key} from {env_var}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid value for {env_var}: {value} ({e})")

    # Handle region separately (comma-separated format)
    region_str = os.environ.get("DECKSNAG_REGION")
    if region_str:
        try:
            parts = [int(p.strip()) for p in region_str.split(",")]
            if len(parts) == 4:
                config["region"] = tuple(parts)
            else:
                logger.warning(f"Invalid DECKSNAG_REGION format: {region_str}")
        except ValueError as e:
            logger.warning(f"Invalid DECKSNAG_REGION value: {region_str} ({e})")

    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later dictionaries take precedence over earlier ones.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    result: Dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if value is not None:
                result[key] = value
    return result


def config_dict_to_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a config dictionary to Config-compatible kwargs.

    Handles type conversions and path creation.

    Args:
        config_dict: Raw configuration dictionary.

    Returns:
        Dictionary suitable for passing to Config().
    """
    result = config_dict.copy()

    # Convert output_path to Path if present
    if "output_path" in result and isinstance(result["output_path"], str):
        result["output_path"] = Path(result["output_path"])

    # Convert region from list to tuple if needed
    if "region" in result and isinstance(result["region"], list):
        result["region"] = tuple(result["region"])

    return result


def create_default_config_file() -> Path:
    """Create a default configuration file with comments.

    Returns:
        Path to the created config file.
    """
    content = '''# DeckSnag Configuration File
#
# This file contains default settings for DeckSnag.
# Uncomment and modify lines to customize behavior.
# CLI arguments will override these settings.

# Output Settings
# output_path = "./presentation"
# output_format = "pptx"  # Options: pptx, pdf, images, all

# Capture Settings
# interval = 5.0  # Seconds between captures (0.5-60)
# threshold = 0.005  # Change detection threshold (0-1)
# method = "mse"  # Comparison method: mse, ssim, clip

# Controls
# stop_hotkey = "end"  # Key to stop capture

# Display Settings
# monitor = 0  # Monitor index (0 = primary)
# region = [100, 100, 800, 600]  # Capture region [x1, y1, x2, y2]

# Debug
# verbose = false
'''

    ensure_config_dir()
    path = get_default_config_path()

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Created default config file: {path}")
    return path
