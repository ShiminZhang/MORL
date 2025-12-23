"""
Utility functions for path management.
"""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / "models"


def get_experiments_dir() -> Path:
    """Get the experiments directory."""
    return get_project_root() / "Experiments"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path

