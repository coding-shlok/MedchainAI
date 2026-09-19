"""YAML experiment config loading with dataclass validation."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def dataclass_from_dict(cls, data: Dict[str, Any]):
    """Build a (possibly nested) dataclass from a plain dict, ignoring unknown
    keys rather than crashing -- configs evolve faster than code during a thesis
    build, and a hard failure on a stray key is more friction than it's worth."""
    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for name, value in data.items():
        if name not in fields:
            continue
        ftype = fields[name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = dataclass_from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
