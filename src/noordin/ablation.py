"""Step 2: ablation study and sensitivity check (SPEC 7).

Each tie is removed in turn and the same seeds are re-run, so differences are paired
by seed (common random numbers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load_graph import Graph, edge_list
from .simulator import STREAM_ABLATION, STREAM_SENSITIVITY, Scenario, run_episode, true_policy


def final_bad_shares(graph: Graph, cfg: dict, policy, stream: int, seeds: int, scenario=None) -> np.ndarray:
    w = int(cfg["run"]["final_window"])
    return np.array([run_episode(graph, cfg, policy, stream, s, scenario).final_bad_share(w) for s in range(seeds)])


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    m = float(np.mean(x))
    half = 1.96 * float(np.std(x, ddof=1)) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, m - half, m + half


def without_tie(adj: np.ndarray, i: int, j: int) -> np.ndarray:
    a = adj.copy()
    a[i, j] = a[j, i] = False
    return a


def ablate_tie(graph: Graph, cfg: dict, i: int, j: int, seeds: int | None = None, baseline=None) -> dict:
    """One tie, paired against the baseline. Used by the app's live picker."""
    seeds = int(seeds or cfg["run"]["ablation_seeds"])
    pol = true_policy(cfg)
    base = final_bad_shares(graph, cfg, pol, STREAM_ABLATION, seeds) if baseline is None else baseline
    abl = final_bad_shares(graph, cfg, pol, STREAM_ABLATION, seeds, Scenario("ablate", adj=without_tie(graph.adj, i, j)))
    m, lo, hi = mean_ci(abl - base)
    return {"agent_i": graph.labels[i], "agent_j": graph.labels[j], "tie": round(float(graph.tie[i, j]), 4),
            "baseline": float(base.mean()), "ablated": float(abl.mean()),
            "mean_delta": m, "ci_low": lo, "ci_high": hi}


def ties_to_ablate(graph: Graph, cfg: dict) -> list[tuple[int, int]]:
    """All ties, or, above `run.ablation_max_ties`, those with the highest edge betweenness on
    the undirected network. A structure-only choice: never the answer key or learner output."""
    ties = edge_list(graph.adj)
    cap = int(cfg["run"].get("ablation_max_ties", 60))
    if len(ties) <= cap:
        return ties
    import networkx as nx
    bt = nx.edge_betweenness_centrality(nx.from_numpy_array(graph.adj.astype(int)), normalized=True)
    score = {tuple(sorted(e)): v for e, v in bt.items()}
    ranked = sorted(ties, key=lambda e: (-score[e], e))[:cap]
    return sorted(ranked)


def run_ablation(graph: Graph, cfg: dict) -> pd.DataFrame:
    seeds = int(cfg["run"]["ablation_seeds"])
    base = final_bad_shares(graph, cfg, true_policy(cfg), STREAM_ABLATION, seeds)
    rows = [ablate_tie(graph, cfg, i, j, seeds, base) for i, j in ties_to_ablate(graph, cfg)]
    df = pd.DataFrame(rows).drop(columns=["baseline", "ablated"])
    df["baseline_bad_share"] = float(base.mean())
    return df.reindex(df["mean_delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def run_sensitivity(graph: Graph, cfg: dict, seeds: int = 20) -> pd.DataFrame:
    rows = []
    for name, scale in (("true theta", None), ("sanction weight doubled", [1, 1, 1, 2])):
        x = final_bad_shares(graph, cfg, true_policy(cfg, scale), STREAM_SENSITIVITY, seeds)
        m, lo, hi = mean_ci(x)
        rows.append({"setting": name, "final_bad_share": m, "ci_low": lo, "ci_high": hi, "seeds": seeds})
    return pd.DataFrame(rows)


def run(graph: Graph, cfg: dict, write: bool = True) -> dict:
    abl, sens = run_ablation(graph, cfg), run_sensitivity(graph, cfg)
    if write:
        out = Path(cfg["outputs_dir"])
        abl.to_csv(out / "ablation.csv", index=False)
        sens.to_csv(out / "sensitivity.csv", index=False)
    return {"ablation": abl, "sensitivity": sens}
