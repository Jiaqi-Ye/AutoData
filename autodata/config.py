"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_SECTIONS = {
    "project",
    "dataset",
    "models",
    "generation",
    "planning",
    "mixture",
    "training",
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and lightly validate a YAML config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def get_run_mode(config: Dict[str, Any]) -> str:
    return str(config.get("project", {}).get("run_mode", "smoke"))


def get_target_domains(config: Dict[str, Any]) -> list[str]:
    domains = config.get("dataset", {}).get("target_domains", [])
    if not domains:
        raise ValueError("dataset.target_domains must contain at least one domain")
    return [str(domain) for domain in domains]

