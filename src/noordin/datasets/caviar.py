"""D2: Caviar, a drug-trafficking network observed through police phone intercepts.

Main network: the `full` layer (110 people, 205 pairs). A pair's weight is the calls in
both directions added together; the tie feature is log1p(weight) scaled by its maximum.
Later network: the `phase11` layer mapped onto the same 110 agents.
Agents are ordered by ascending original ID (A001..A110).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..load_graph import Graph, components, degree_status, make_labels
from .common import layers, read_edge_list

EDGES = "edges.csv"


def files_present(cfg: dict) -> bool:
    return (Path(cfg["data_dir"]) / EDGES).exists()


def load(cfg: dict) -> Graph:
    ds = cfg["ds"]
    path = Path(cfg["data_dir"]) / EDGES
    ids, directed, calls = read_edge_list(path, ds.get("main_layer", "full"))
    adj = calls > 0
    np.fill_diagonal(adj, False)
    strength = np.log1p(calls)
    tie = np.where(adj, strength / strength[adj].max(), 0.0)
    labels = make_labels(len(ids))

    _, later_dir, later_calls = read_edge_list(path, ds.get("later_layer", "phase11"), ids)
    later = later_calls > 0

    deg = adj.sum(1)
    hub = int(np.argsort(-deg, kind="stable")[0])
    extra = {
        "calls_total": int(calls[np.triu(adj, 1)].sum()),
        "hub_call_share": round(float(calls[hub].sum() / calls[np.triu(adj, 1)].sum()), 3),
    }
    for layer in layers(path):
        if not layer.startswith("phase"):
            continue
        _, d_dir, d_w = read_edge_list(path, layer, ids)
        a = d_w > 0
        d_deg = a.sum(1)
        top = int(np.argsort(-d_deg, kind="stable")[0])
        extra[f"pairs_{layer}"] = int(np.triu(a, 1).sum())
        extra[f"top_agent_{layer}"] = f"{labels[top]} (degree {int(d_deg[top])})"
    later_sizes = [len(c) for c in components(later) if len(c) > 1]
    extra["later_group_sizes"] = ", ".join(map(str, later_sizes))
    return Graph(
        name="caviar",
        labels=labels,
        adj=adj,
        tie=tie,
        tie_label=ds.get("tie_label", "Call volume"),
        directed=directed,
        later=later,
        later_label=ds.get("later_label", "Phase 11"),
        status=degree_status(adj),
        extra=extra,
        _original_ids=[str(x) for x in ids],
        _layers={"calls": calls, "later_directed": later_dir},
    )
