"""Step 6: interventions and the pre vs post-attack comparison (SPEC 9, step 6).

Every scenario runs the same seeds as the baseline, so each difference is paired.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ablation import mean_ci
from .load_graph import Graph
from .simulator import STREAM_INTERVENTION, Scenario, run_episode, true_policy


def scenarios(graph: Graph, cfg: dict) -> list[Scenario]:
    """Hub = highest-degree agent, mentor = second-highest (A11 and A07 on Noordin).
    The real-network scenario uses the dataset's later network and is skipped without one."""
    hub, mentor = (int(x) for x in graph.ranked()[:2])
    isolated = graph.adj.copy()
    isolated[hub, :] = isolated[:, hub] = False
    out = [
        Scenario(f"Hub isolation ({graph.labels[hub]})", adj=isolated),
        Scenario(f"Mentor injection ({graph.labels[mentor]})", mentor=mentor),
        Scenario(f"Supervision +{cfg['interventions']['supervision_increase']:g}",
                 supervision_add=float(cfg["interventions"]["supervision_increase"])),
    ]
    if graph.later is not None:
        out.append(Scenario(f"Real disruption ({graph.later_label.lower()} graph)", adj=graph.later.copy()))
    return out


def run(graph: Graph, cfg: dict, write: bool = True) -> dict:
    seeds = int(cfg["run"]["intervention_seeds"])
    w = int(cfg["run"]["final_window"])
    pol = true_policy(cfg)
    base_eps = [run_episode(graph, cfg, pol, STREAM_INTERVENTION, s) for s in range(seeds)]
    base = np.array([e.final_bad_share(w) for e in base_eps])
    rows, curves = [], [(f"Baseline ({cfg['ds'].get('main_label', 'main')} graph)", base_eps)]
    for sc in scenarios(graph, cfg):
        eps = [run_episode(graph, cfg, pol, STREAM_INTERVENTION, s, sc) for s in range(seeds)]
        d = np.array([e.final_bad_share(w) for e in eps]) - base
        m, lo, hi = mean_ci(d)
        rows.append({"scenario": sc.name, "baseline_bad_share": float(base.mean()),
                     "scenario_bad_share": float(base.mean() + m), "delta": m, "ci_low": lo, "ci_high": hi,
                     "seeds": seeds})
        curves.append((sc.name, eps))
    table = pd.DataFrame(rows)
    curve_df = pd.concat([
        pd.DataFrame({"scenario": name, "step": np.arange(eps[0].bad_share.size),
                      "mean": np.mean([e.bad_share for e in eps], 0),
                      "sd": np.std([e.bad_share for e in eps], 0, ddof=1) if len(eps) > 1 else 0.0,
                      "n": len(eps)})
        for name, eps in curves], ignore_index=True)
    half = 1.96 * curve_df["sd"] / np.sqrt(curve_df["n"])
    curve_df["low"], curve_df["high"] = curve_df["mean"] - half, curve_df["mean"] + half
    curve_df = curve_df.drop(columns=["sd", "n"])
    if write:
        out = Path(cfg["outputs_dir"])
        table.to_csv(out / "interventions.csv", index=False)
        curve_df.to_csv(out / "intervention_curves.csv", index=False)
    return {"table": table, "curves": curve_df}


def adjacency_from_edges(graph: Graph, edges, added=()) -> np.ndarray:
    """Symmetric adjacency from (label, label) pairs.

    `edges` must be pre-attack ties. `added` are new ties the user drew between any two
    agents; they are never kinship ties.
    """
    adj = np.zeros_like(graph.adj)
    for a, b in edges:
        i, j = graph.label_index(a), graph.label_index(b)
        if not graph.adj[i, j]:
            raise ValueError(f"{a}-{b} is not a pre-attack tie")
        adj[i, j] = adj[j, i] = True
    for a, b in added:
        i, j = graph.label_index(a), graph.label_index(b)
        if i == j:
            raise ValueError(f"cannot tie {a} to itself")
        adj[i, j] = adj[j, i] = True
    return adj


def simulate_edited(graph: Graph, cfg: dict, edges, removed=(), added=(), seeds: int | None = None) -> dict:
    """Paired comparison of a user-edited network against the intact pre-attack network.

    Each of the `seeds` runs draws one set of agent traits and one sequence of dice rolls,
    and plays it on both networks, so the only difference between a pair is the ties.
    Removed agents lose every tie; both networks are then scored over the agents that
    remain, so taking someone out does not lower the bad share just by not counting them.
    """
    seeds = int(seeds or cfg["run"]["intervention_seeds"])
    w = int(cfg["run"]["final_window"])
    gone = tuple(sorted(graph.label_index(r) for r in removed))
    edges = [e for e in edges if e[0] not in removed and e[1] not in removed]
    added = [e for e in added if e[0] not in removed and e[1] not in removed]
    pol = true_policy(cfg)
    base_sc = Scenario("Intact", exclude=gone)
    new_adj = adjacency_from_edges(graph, edges, added)
    sc = Scenario("Your network", adj=new_adj, exclude=gone)
    base = [run_episode(graph, cfg, pol, STREAM_INTERVENTION, s, base_sc) for s in range(seeds)]
    mine = [run_episode(graph, cfg, pol, STREAM_INTERVENTION, s, sc) for s in range(seeds)]
    b = np.array([e.final_bad_share(w) for e in base])
    y = np.array([e.final_bad_share(w) for e in mine])
    m, lo, hi = mean_ci(y - b)
    base_mean = float(b.mean())
    rows = []
    for name, eps in (("Intact pre-attack network", base), ("Your network", mine)):
        arr = np.array([e.bad_share for e in eps])
        half = 1.96 * arr.std(0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        rows.append(pd.DataFrame({"scenario": name, "step": np.arange(arr.shape[1]), "mean": arr.mean(0),
                                  "low": arr.mean(0) - half, "high": arr.mean(0) + half}))
    groups = group_breakdown(graph, cfg, new_adj, gone, base, mine)
    return {"baseline": base_mean, "edited": base_mean + float(m), "delta": float(m), "ci_low": float(lo),
            "ci_high": float(hi),
            # change relative to the intact network: -0.05 means 5% less bad behaviour
            "relative": float(m) / base_mean if base_mean else 0.0,
            "relative_low": float(lo) / base_mean if base_mean else 0.0,
            "relative_high": float(hi) / base_mean if base_mean else 0.0,
            "runs": {"intact": b.round(4).tolist(), "yours": y.round(4).tolist()},
            "seeds": seeds, "ties": int(len(edges) + len(added)), "added": int(len(added)),
            "removed": [graph.labels[i] for i in gone], "counted": int(graph.n - len(gone)),
            "groups": groups, "explanation": explain_groups(groups, sig=lo > 0 or hi < 0),
            "curves": pd.concat(rows, ignore_index=True)}


GROUPS = [   # key, short label, description, phrase used in sentences ("the 20 agents ...")
    ("isolated", "Left with no ties", "had ties before your edits and have none now", "left with no ties"),
    ("lost", "Lost some ties", "still have ties, but fewer or different ones", "who lost some ties"),
    ("gained", "Gained ties", "got at least one new tie from you", "who gained ties"),
    ("unchanged", "Ties unchanged", "have exactly the same neighbours as before", "whose ties did not change"),
]


def group_breakdown(graph: Graph, cfg: dict, new_adj: np.ndarray, gone: tuple, base, mine) -> list[dict]:
    """Bad share over the final window for each kind of agent, intact vs edited, paired by run.
    Explains a small overall change: groups often move in opposite directions."""
    w = int(cfg["run"]["final_window"])
    keep = np.ones(graph.n, bool)
    keep[list(gone)] = False
    before, after = graph.adj, new_adj
    had = before.sum(1) > 0
    kind = np.full(graph.n, "unchanged", dtype=object)
    changed = (before != after).any(1)
    kind[changed] = "lost"
    kind[(after & ~before).any(1)] = "gained"
    kind[had & (after.sum(1) == 0)] = "isolated"
    agent_bad = lambda eps: np.array([np.isin(e.actions[-w:], cfg["actions"]["bad"]).mean(0) for e in eps])  # (runs, N)
    b, y = agent_bad(base), agent_bad(mine)
    out = []
    for key, label, desc, phrase in GROUPS:
        idx = np.where((kind == key) & keep)[0]
        if not len(idx):
            continue
        d = y[:, idx].mean(1) - b[:, idx].mean(1)
        m, lo, hi = mean_ci(d)
        out.append({"group": key, "label": label, "description": desc, "phrase": phrase, "agents": int(len(idx)),
                    "intact": float(b[:, idx].mean()), "yours": float(y[:, idx].mean()),
                    "change": float(m), "ci_low": float(lo), "ci_high": float(hi),
                    "share_of_counted": float(len(idx) / keep.sum())})
    return out


def explain_groups(groups: list[dict], sig: bool) -> str:
    """One plain sentence on how the groups add up to the overall result."""
    moved = [g for g in groups if g["ci_high"] < 0 or g["ci_low"] > 0]
    if not moved:
        return "No group of agents changed by more than chance."
    down = [g for g in moved if g["change"] < 0]
    up = [g for g in moved if g["change"] > 0]
    part = lambda g: (f"the {g['agents']} agents {g['phrase']} went from {g['intact']:.0%} to {g['yours']:.0%} "
                      f"behaving badly ({g['change'] * 100:+.0f} points)")
    if down and up:
        text = "; ".join(part(g) for g in down + up)
        return ("Your edits helped some agents and hurt others. " + text[0].upper() + text[1:]
                + ". " + ("Overall these largely cancel out." if not sig else "The overall result is what is left over."))
    return "The change comes from " + "; ".join(part(g) for g in moved) + "."

