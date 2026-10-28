"""Step 4: criminal behaviour adoption (SPEC 9, step 4).

Rolls the simulator forward under different policies with the same seeds and compares
the cascades of bad behaviour. Policies arrive as callables; this module never builds
the true one itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import Actions
from .learn import bc_features_dense
from .load_graph import Graph
from .simulator import STREAM_ROLLOUT, UtilityPolicy, run_episode

ROLLOUT_SEEDS = 20


class BCPolicy:
    """Behavioural-cloning classifier as a simulator policy."""

    def __init__(self, clf, cfg: dict):
        self.clf, self.acts = clf, Actions(cfg)
        self.cols = np.asarray(clf.classes_, dtype=int)

    def __call__(self, ctx) -> np.ndarray:
        p = np.zeros((len(ctx["sup"]), self.acts.n))
        p[:, self.cols] = self.clf.predict_proba(bc_features_dense(ctx, self.acts))
        return p


def recovered_policy(cfg: dict, rec: dict) -> UtilityPolicy:
    g = rec["gamma_hat"]
    return UtilityPolicy(Actions(cfg), cfg["choice"]["beta"], rec["theta_hat"], g["gamma_status"], g["gamma_tie"])


def cascades(graph: Graph, cfg: dict, policies: dict, seeds: int = ROLLOUT_SEEDS):
    """Bad share per step for each policy, plus first escalation step per agent."""
    curves, escal = [], []
    for name, pol in policies.items():
        for s in range(seeds):
            ep = run_episode(graph, cfg, pol, STREAM_ROLLOUT, s)
            curves.append(pd.DataFrame({"policy": name, "seed": s, "step": np.arange(len(ep.bad_share)),
                                        "bad_share": ep.bad_share}))
            escal.append(pd.DataFrame({"policy": name, "seed": s, "agent": graph.labels,
                                       "first_escalation": ep.first_escalation}))
    return pd.concat(curves, ignore_index=True), pd.concat(escal, ignore_index=True)


def summarise_curves(curves: pd.DataFrame) -> pd.DataFrame:
    g = curves.groupby(["policy", "step"])["bad_share"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    half = 1.96 * out["std"].fillna(0) / np.sqrt(out["count"])
    out["low"], out["high"] = out["mean"] - half, out["mean"] + half
    return out.drop(columns=["std", "count"])


def cascade_gap(summary: pd.DataFrame, reference: str = "true") -> pd.DataFrame:
    ref = summary[summary["policy"] == reference].set_index("step")["mean"]
    rows = []
    for name, grp in summary.groupby("policy", sort=False):
        if name == reference:
            continue
        rows.append({"policy": name, "mean_abs_gap": float((grp.set_index("step")["mean"] - ref).abs().mean())})
    return pd.DataFrame(rows)


def time_to_escalation(escal: pd.DataFrame) -> pd.DataFrame:
    """Mean first step of action 3 per agent and policy, over the seeds where it happened."""
    g = escal.groupby(["policy", "agent"])["first_escalation"]
    out = g.agg(mean_step="mean", escalated_share=lambda x: x.notna().mean()).reset_index()
    return out


def run(graph: Graph, cfg: dict, policies: dict, write: bool = True, seeds: int = ROLLOUT_SEEDS) -> dict:
    curves, escal = cascades(graph, cfg, policies, seeds)
    summary = summarise_curves(curves)
    gap = cascade_gap(summary)
    tte = time_to_escalation(escal)
    if write:
        out = Path(cfg["outputs_dir"])
        summary.to_csv(out / "cascade_curves.csv", index=False)
        gap.to_csv(out / "cascade_gap.csv", index=False)
        tte.to_csv(out / "time_to_escalation.csv", index=False)
    return {"curves": summary, "gap": gap, "time_to_escalation": tte}
