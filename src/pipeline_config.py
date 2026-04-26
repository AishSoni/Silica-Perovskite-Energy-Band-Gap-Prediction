"""
Configuration helpers for the corrected perovskite pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml
from dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path("experiments/query_config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the canonical YAML configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return config


def config_checksum(config: Dict[str, Any]) -> str:
    """Return a stable checksum for a loaded config dictionary."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_checksum(path: str | Path) -> str:
    """Return a SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_api_key() -> str:
    """Load the Materials Project API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("MP_API_KEY") or os.getenv("MAPI_KEY")
    if not api_key:
        raise ValueError("Materials Project API key not found. Set MP_API_KEY in your environment or .env file.")
    return api_key


def path_from_config(config: Dict[str, Any], key: str) -> Path:
    """Resolve an output path from the config's paths section."""
    try:
        return Path(config["paths"][key])
    except KeyError as exc:
        raise KeyError(f"Missing paths.{key} in query config") from exc


def range_tuple(section: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Convert a config range section with min/max keys to an API tuple."""
    return (section.get("min"), section.get("max"))


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a path and return the path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def as_list(value: Any) -> list:
    """Normalize a scalar or iterable config value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def unique_preserve_order(values: Iterable[Any]) -> list:
    """Return unique values while preserving first occurrence order."""
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique

