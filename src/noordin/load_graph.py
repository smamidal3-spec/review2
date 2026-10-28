"""Step 1: datasets to graph.

The loaders in `noordin.datasets` are the only code that sees original dataset IDs.
Everything downstream uses agent labels (A01..A28, A001..A110, ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class Graph:
    name: str                       # "noordin", "caviar", ...
    labels: list[str]
    adj: np.ndarray                 # (N, N) bool, undirected main network, no diagonal
    tie: np.ndarray                 # (N, N) float in [0, 1], symmetric, zero where adj is False
    tie_label: str                  # "Kinship", "Call volume", ...
    directed: np.ndarray            # (N, N) bool, original direction of the main network
    later: np.ndarray | None        # (N, N) bool, real later network for the "real network" intervention
    later_label: str | None         # "Post-attack", "Phase 11", "Wave 4"
    status: np.ndarray              # (N,) degree / max degree on adj
    extra: dict = field(default_factory=dict)            # dataset-specific stats for graph_stats.csv
    _original_ids: list[str] = field(repr=False, default_factory=list)
    _layers: dict = field(repr=False, default_factory=dict)   # raw matrices kept for tests only

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def degree(self) -> np.ndarray:
        return self.adj.sum(1)

    def label_index(self, label: str) -> int:
        return self.labels.index(label)

    def ranked(self) -> np.ndarray:
        """Agent indices by degree, highest first (stable, so ties keep agent order)."""
        return np.argsort(-self.degree, kind="stable")


def make_labels(n: int) -> list[str]:
    width = max(2, len(str(n)))
    return [f"A{i + 1:0{width}d}" for i in range(n)]


def degree_status(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(1)
    return deg / deg.max()


def load_graph(cfg: dict) -> Graph:
    from .datasets import load
    return load(cfg["dataset"], cfg)


def edge_list(adj: np.ndarray) -> list[tuple[int, int]]:
    i, j = np.where(np.triu(adj, 1))
    return list(zip(i.tolist(), j.tolist()))


def components(adj: np.ndarray) -> list[list[int]]:
    """Connected groups, largest first, singletons included."""
    g = nx.from_numpy_array(adj.astype(int))
    comps = [sorted(c) for c in nx.connected_components(g)]
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def _sizes_text(sizes: list[int], limit: int = 8) -> str:
    text = ", ".join(map(str, sizes[:limit]))
    return text + (f" and {len(sizes) - limit} more" if len(sizes) > limit else "")


def graph_stats(g: Graph) -> pd.DataFrame:
    """Generic statistics first, then the dataset's own (`extra`, scalar values only)."""
    deg = g.degree
    ties = int(np.triu(g.adj, 1).sum())
    comps = components(g.adj)
    sizes = [len(c) for c in comps if len(c) > 1]
    order = g.ranked()
    mutual = int(np.triu(g.directed & g.directed.T, 1).sum())
    rows = [
        ("nodes", g.n),
        ("ties", ties),
        ("mutual_pairs", mutual),
        ("group_sizes", _sizes_text(sizes)),
        ("groups", len(sizes)),
        ("isolates", int((deg == 0).sum())),
        ("max_degree", int(deg[order[0]])),
        ("highest_degree_agent", f"{g.labels[order[0]]} (degree {deg[order[0]]})"),
        ("second_highest_degree_agent", f"{g.labels[order[1]]} (degree {deg[order[1]]})"),
        ("average_degree", round(float(deg.mean()), 3)),
        ("hub_share", round(float(deg[order[0]] / ties), 3) if ties else 0.0),
        ("tie_feature", g.tie_label),
        ("tie_feature_nonzero", int(np.triu(g.tie > 0, 1).sum())),
        ("tie_feature_mean", round(float(g.tie[np.triu(g.adj, 1)].mean()), 3) if ties else 0.0),
    ]
    if g.later is not None:
        rows += [("later_network", g.later_label),
                 ("later_ties", int(np.triu(g.later, 1).sum())),
                 ("later_isolated_agents", int((g.later.sum(1) == 0).sum()))]
    rows += [(k, v) for k, v in g.extra.items() if not k.startswith("_") and np.isscalar(v)]
    return pd.DataFrame(rows, columns=["stat", "value"])


def tie_identification_note(graph) -> str:
    """Why gamma_tie is or is not well identified on this network, from the data."""
    up = np.triu(graph.adj, 1)
    w = graph.tie[up]
    n, on = int(up.sum()), int((w > 0).sum())
    if np.isin(w, [0.0, 1.0]).all():
        share = on / n if n else 0.0
        weak = share > 0.8 or share < 0.2
        return (f"gamma_tie ({graph.tie_label.lower()}) is {'weakly' if weak else 'reasonably'} identified: "
                f"{on} of the {n} ties have the feature, so the data "
                f"{'rarely' if weak else 'often'} contrasts a tie with the feature against one without it.")
    return (f"gamma_tie ({graph.tie_label.lower()}) is a continuous weight with mean {w.mean():.2f} and spread "
            f"{w.std():.2f} across the {n} ties, which gives the learner variation to identify it.")


def stat_value(stats: pd.DataFrame, key: str, default=None):
    m = stats.loc[stats["stat"] == key, "value"]
    return m.iloc[0] if len(m) else default


def write_outputs(g: Graph, cfg: dict) -> None:
    out = Path(cfg["outputs_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": g.labels, "original_id": g._original_ids}).to_csv(out / "agent_map.csv", index=False)
    later = g.later if g.later is not None else np.zeros_like(g.adj)
    edges = [
        (g.labels[i], g.labels[j], round(float(g.tie[i, j]), 4), int(later[i, j]),
         int(g.directed[i, j] and g.directed[j, i]))
        for i, j in edge_list(g.adj)
    ]
    pd.DataFrame(edges, columns=["agent_i", "agent_j", "tie", "later", "mutual"]).to_csv(out / "graph_edges.csv", index=False)
    graph_stats(g).to_csv(out / "graph_stats.csv", index=False)
