"""Load and validate config.yaml.

The config is split in two on purpose. `load_public_config()` returns everything
except the answer key and is the only loader learn.py, explain.py and rollout.py may
use. `load_answer_key()` returns only the true reward and attention parameters.

The public config is resolved for one dataset: `cfg["dataset"]` names it, `cfg["ds"]`
is its block from `datasets:`, and `data_dir` points at its files.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config.yaml"
_SECRET_KEY = "reward_true"
THETA_NAMES = ["peer_approval", "material_gain", "status_gain", "sanction_risk"]


def _read(path=None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    for key in ("seed", "outputs_dir", "agents", "choice", "actions", "run", "quick", "datasets"):
        if key not in cfg:
            raise ValueError(f"config.yaml is missing '{key}'")
    lo, hi = cfg["agents"]["supervision_range"]
    if not hi > lo:
        raise ValueError("supervision_range must have positive width (identifiability, SPEC 5.3)")
    acts = cfg["actions"]
    n = len(acts["names"])
    for k in ("level", "severity", "material"):
        if len(acts[k]) != n:
            raise ValueError(f"actions.{k} must have {n} entries")


def dataset_names(path=None) -> list[str]:
    return list(_read(path)["datasets"])


def _resolve(cfg: dict, dataset: str | None) -> dict:
    name = dataset or cfg.get("dataset", "noordin")
    if name not in cfg["datasets"]:
        raise ValueError(f"unknown dataset '{name}'; choose from {list(cfg['datasets'])}")
    cfg["dataset"] = name
    cfg["ds"] = copy.deepcopy(cfg["datasets"][name])
    cfg["data_dir"] = str((ROOT / cfg["ds"]["data_dir"]).resolve())
    cfg["outputs_root"] = str((ROOT / cfg["outputs_dir"]).resolve())
    cfg["outputs_dir"] = str(Path(cfg["outputs_root"]) / name)
    return cfg


def load_public_config(path=None, quick: bool = False, dataset: str | None = None) -> dict:
    """Everything except the answer key. With quick=True, `quick` overrides `run`."""
    cfg = _read(path)
    cfg.pop(_SECRET_KEY, None)
    if quick:
        cfg = apply_quick(cfg)
    return _resolve(cfg, dataset)


def load_answer_key(path=None) -> dict:
    """Only the true reward weights and attention parameters (same for every dataset)."""
    key = _read(path)[_SECRET_KEY]
    att = key["attention"]
    theta = [float(key["theta"][k]) for k in THETA_NAMES]
    return {
        "theta": theta,
        "gamma_status": float(att["gamma_status"]),
        # gamma_kin is the old name, still accepted so older configs load
        "gamma_tie": float(att["gamma_tie"] if "gamma_tie" in att else att["gamma_kin"]),
    }


def apply_quick(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["run"].update(cfg["quick"])
    return cfg
