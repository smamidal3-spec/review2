"""Registry of networks. Each module exposes `load(cfg) -> Graph` and `files_present(cfg)`.

Loaders read only the edge files listed in SPEC_MULTIDATASET.md section 3. They never
open attribute files (ethnicity, gender, demographics, alcohol, delinquency).
"""
from __future__ import annotations

from importlib import import_module

NAMES = ["noordin", "caviar", "drugnet", "knecht"]


def module(name: str):
    if name not in NAMES:
        raise ValueError(f"unknown dataset '{name}'")
    return import_module(f"{__name__}.{name}")


def load(name: str, cfg: dict):
    return module(name).load(cfg)


def files_present(name: str, cfg: dict) -> bool:
    try:
        return module(name).files_present(cfg)
    except ModuleNotFoundError:
        return False
