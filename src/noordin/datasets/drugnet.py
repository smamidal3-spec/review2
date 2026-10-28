"""D3: Drugnet, "A says they know B" among people who use drugs (Hartford).

The edge list holds only the 212 people with at least one tie. The 81 isolates on the
dataset card are not in the file and their IDs (which run to 298 with gaps) cannot be
reconstructed, so the model runs on the 212 connected agents (A001..A212 by ascending
ID). Isolates neither influence anyone nor are influenced, so they add nothing to
influence or explanation. The tie feature is 1 for a mutual pair, otherwise 0. There is
no later network. Only the edge list is read; the attribute file is never opened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..load_graph import Graph, degree_status, make_labels
from .common import read_edge_list

EDGES = "edges.csv"


def files_present(cfg: dict) -> bool:
    return (Path(cfg["data_dir"]) / EDGES).exists()


def load(cfg: dict) -> Graph:
    ds = cfg["ds"]
    ids, directed, w = read_edge_list(Path(cfg["data_dir"]) / EDGES, ds.get("main_layer", "acquaintance"))
    adj = w > 0
    np.fill_diagonal(adj, False)
    mutual = directed & directed.T
    return Graph(
        name="drugnet",
        labels=make_labels(len(ids)),
        adj=adj,
        tie=mutual.astype(float),
        tie_label=ds.get("tie_label", "Mutual tie"),
        directed=directed,
        later=None,
        later_label=None,
        status=degree_status(adj),
        extra={"isolates_not_in_file": int(ds.get("isolates_not_in_file", 81))},
        _original_ids=[str(x) for x in ids],
    )
