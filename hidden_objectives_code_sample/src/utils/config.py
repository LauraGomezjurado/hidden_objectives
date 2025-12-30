"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Configuration to override base values
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_config_dir() -> Path:
    """Get the configs directory."""
    return get_project_root() / "configs"


def get_data_dir() -> Path:
    """Get the data directory, creating if needed."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_output_dir() -> Path:
    """Get the outputs directory, creating if needed."""
    output_dir = get_project_root() / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir

