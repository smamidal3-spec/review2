"""Shared helper for the edge-list datasets (Caviar, Drugnet, Knecht)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sort_ids(ids) -> list:
    """Ascending original ID: numeric when every ID is a number."""
    ids = list(ids)
    try:
        return sorted(ids, key=lambda x: float(x))
    except (TypeError, ValueError):
        return sorted(ids, key=str)


def read_edge_list(path: Path, layer: str, ids: list | None = None):
    """One layer of a `source,target,weight,layer` file.

    Returns (ids, directed, weight): `ids` in ascending order (or the ones passed in, so a
    later layer maps onto the main network's agents), `directed[i, j]` true for each listed
    link i -> j, and `weight` symmetric with the weights of both directions added together.
    """
    df = pd.read_csv(path, dtype={"source": str, "target": str})
    d = df[df["layer"] == layer]
    if d.empty:
        raise ValueError(f"{Path(path).name}: no rows for layer '{layer}'")
    if d.duplicated(["source", "target"]).any():
        raise ValueError(f"{Path(path).name}: duplicate links in layer '{layer}'")
    if (d["source"] == d["target"]).any():
        raise ValueError(f"{Path(path).name}: self-loops in layer '{layer}'")
    if ids is None:
        ids = _sort_ids(set(d["source"]) | set(d["target"]))
    index = {x: k for k, x in enumerate(ids)}
    missing = (set(d["source"]) | set(d["target"])) - set(index)
    if missing:
        raise ValueError(f"layer '{layer}' has agents not in the main network: {sorted(missing)[:5]}")
    n = len(ids)
    directed = np.zeros((n, n), bool)
    w = np.zeros((n, n))
    i = d["source"].map(index).to_numpy()
    j = d["target"].map(index).to_numpy()
    directed[i, j] = True
    np.add.at(w, (i, j), d["weight"].to_numpy(float))
    return ids, directed, w + w.T


def layers(path: Path) -> list[str]:
    return list(dict.fromkeys(pd.read_csv(path, usecols=["layer"])["layer"]))
