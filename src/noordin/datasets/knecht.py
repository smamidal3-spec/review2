"""D4: Knecht, best-friend nominations in one school class (klas12b), four waves.

Main network: wave 1 (26 students, A01..A26 by ascending ID). Later network: wave 4 on
the same 26. The tie feature is 1 for a mutual friendship, otherwise 0.

Knecht is a class of children. Only the friendship edge list is read; no demographic,
alcohol or delinquency file is ever opened, and nothing per student leaves the pipeline
except simulated behaviour. The edge list has already dropped the missing (9) and
left (10) codes, so an unlisted link is no tie; missing answers cannot be told apart
from "not a friend".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..load_graph import Graph, degree_status, make_labels
from .common import layers, read_edge_list

EDGES = "edges.csv"


def files_present(cfg: dict) -> bool:
    return (Path(cfg["data_dir"]) / EDGES).exists()


def load(cfg: dict) -> Graph:
    ds = cfg["ds"]
    path = Path(cfg["data_dir"]) / EDGES
    ids, directed, w = read_edge_list(path, ds.get("main_layer", "wave1"))
    adj = w > 0
    np.fill_diagonal(adj, False)
    mutual = directed & directed.T
    labels = make_labels(len(ids))
    _, later_dir, later_w = read_edge_list(path, ds.get("later_layer", "wave4"), ids)
    later = later_w > 0

    extra = {}
    for layer in layers(path):
        _, _, lw = read_edge_list(path, layer, ids)
        a = lw > 0
        deg = a.sum(1)
        top = int(np.argsort(-deg, kind="stable")[0])
        extra[f"pairs_{layer}"] = int(np.triu(a, 1).sum())
        extra[f"top_agent_{layer}"] = f"{labels[top]} (degree {int(deg[top])})"
    up = np.triu(adj, 1)
    extra["wave1_pairs_still_in_wave4"] = round(float((later[up]).mean()), 3)
    return Graph(
        name="knecht",
        labels=labels,
        adj=adj,
        tie=mutual.astype(float),
        tie_label=ds.get("tie_label", "Mutual friendship"),
        directed=directed,
        later=later,
        later_label=ds.get("later_label", "Wave 4"),
        status=degree_status(adj),
        extra=extra,
        _original_ids=[str(x) for x in ids],
        _layers={"later_directed": later_dir},
    )
