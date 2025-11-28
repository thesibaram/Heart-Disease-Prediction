"""Utility functions for the Heart Disease Prediction project."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_directory(path: Path | str) -> Path:
    """Ensure that a directory exists and return its Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Path | str) -> None:
    """Save a dictionary as a JSON file with pretty formatting."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with filepath.open('w', encoding='utf-8') as f:
        json.dump(_serialize_for_json(data), f, indent=2)
    print(f"JSON saved to {filepath}")


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects that are not JSON serializable by default."""
    if isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and other libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def get_timestamp() -> str:
    """Return current timestamp formatted for filenames."""
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')
