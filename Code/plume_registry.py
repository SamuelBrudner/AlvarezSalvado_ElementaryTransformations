"""Registry helpers for plume intensity ranges."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - fallback if PyYAML is unavailable
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal YAML support
    import types

    def _minimal_load(path: Path) -> Dict[str, Dict[str, float]]:
        data: Dict[str, Dict[str, float]] = {}
        current: str | None = None
        for raw_line in path.read_text().splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith("  "):
                current = stripped.rstrip(":")
                data[current] = {}
            else:
                key, value = stripped.split(":", 1)
                data[current][key.strip()] = float(value)
        return data

    def _minimal_dump(obj: Dict[str, Dict[str, float]], fh) -> None:
        for key, val in obj.items():
            fh.write(f"{key}:\n")
            fh.write(f"  min: {val['min']}\n")
            fh.write(f"  max: {val['max']}\n")

    yaml = types.SimpleNamespace(
        safe_load=lambda fh: _minimal_load(Path(fh.name)),
        safe_dump=lambda obj, fh: _minimal_dump(obj, fh),
    )


def load_registry(yaml_path: Path = Path("configs/plume_registry.yaml")) -> Dict[str, Dict[str, float]]:
    """Return the plume intensity registry as a dictionary."""
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                return {
                    k: {"min": float(v.get("min", 0.0)), "max": float(v.get("max", 0.0))}
                    for k, v in loaded.items()
                }
    return {}


def update_plume_registry(
    path: str,
    min_val: float,
    max_val: float,
    yaml_path: Path = Path("configs/plume_registry.yaml"),
) -> None:
    """Update or insert intensity range for ``path`` in ``yaml_path``."""
    registry: Dict[str, Dict[str, Any]] = {}
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                registry = loaded
    entry = registry.get(path)
    if entry:
        min_val = min(float(entry.get("min", min_val)), min_val)
        max_val = max(float(entry.get("max", max_val)), max_val)
    registry[path] = {"min": float(min_val), "max": float(max_val)}
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(registry, fh)


