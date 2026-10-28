"""Step 5: explainable AI from recovered parameters only (SPEC 9, step 5).

A decision is explained in two layers: which concepts pushed the agent toward the
chosen action (beta * theta_hat_k * phi_k), and which neighbour's example mattered most
in the peer-approval concept (alpha_hat_ij * similarity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from .features import (FEATURE_NAMES, Actions, padded_attention, padded_features, padded_similarity,
                       softmax_rows, static_features)
from .learn import Decisions, bc_features, BC_FEATURES


def recovered_alpha(data: Decisions, rec: dict) -> np.ndarray:
    g = rec["gamma_hat"]
    return padded_attention(data.nbr_status, data.nbr_tie, data.mask, g["gamma_status"], g["gamma_tie"])


def neighbour_contributions(data: Decisions, cfg: dict, rec: dict, actions=None) -> np.ndarray:
    """alpha_hat_ij * (1 - |level[a_i] - level[last_j]|) for the chosen action, (D, K); 0 where masked."""
    acts = Actions(cfg)
    a = data.action if actions is None else np.asarray(actions)
    sim = 1.0 - np.abs(acts.level[a][:, None] - acts.level[data.nbr_last])
    return recovered_alpha(data, rec) * sim * data.mask


def ranked_neighbours(contrib: np.ndarray, mask: np.ndarray, labels: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k neighbour labels per decision by contribution, "" where fewer neighbours."""
    score = np.where(mask, contrib, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    out = np.take_along_axis(labels, order, 1).astype(object)
    out[~np.take_along_axis(mask, order, 1)] = ""
    return out


def _by_chunk(data: Decisions, fn) -> np.ndarray:
    parts = [fn(c) for c in data.chunks()]
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=object)


def predicted_influencers(data: Decisions, cfg: dict, rec: dict, k: int = 3) -> np.ndarray:
    return _by_chunk(data, lambda c: ranked_neighbours(neighbour_contributions(c, cfg, rec), c.mask, c.nbr_label, k))


def most_similar_baseline(data: Decisions, cfg: dict, k: int = 3) -> np.ndarray:
    """Attention-free baseline: most similar neighbour, ties broken by status."""
    acts = Actions(cfg)

    def one(c):
        sim = 1.0 - np.abs(acts.level[c.action][:, None] - acts.level[c.nbr_last])
        return ranked_neighbours(sim + 1e-6 * c.nbr_status, c.mask, c.nbr_label, k)
    return _by_chunk(data, one)


def explain_decision(data: Decisions, row: int, cfg: dict, rec: dict) -> dict:
    """Concept contributions for every action and the neighbour table for one decision."""
    acts = Actions(cfg)
    one = data.subset([row])
    alpha = recovered_alpha(one, rec)
    phi = padded_features(acts, alpha, padded_similarity(acts, one.nbr_last, one.mask),
                          static_features(acts, one.sup, one.mean_dev))[0]                 # (A, 4)
    beta, th = float(cfg["choice"]["beta"]), np.asarray(rec["theta_hat"])
    contrib = beta * phi * th[None, :]
    chosen = int(one.action[0])
    concepts = pd.DataFrame(contrib, columns=FEATURE_NAMES)
    concepts.insert(0, "action", acts.names)
    concepts["utility"] = contrib.sum(1)
    concepts["probability"] = softmax_rows(contrib.sum(1)[None, :])[0]
    concepts["chosen"] = np.arange(acts.n) == chosen
    m = one.mask[0]
    sim = 1.0 - np.abs(acts.level[chosen] - acts.level[one.nbr_last[0]])
    nb = pd.DataFrame({
        "neighbour": one.nbr_label[0][m],
        "last_action": [acts.names[x] for x in one.nbr_last[0][m]],
        "status": one.nbr_status[0][m],
        "tie": one.nbr_tie[0][m],
        "attention": alpha[0][m],
        "similarity": sim[m],
        "contribution": (alpha[0] * sim)[m],
    }).sort_values("contribution", ascending=False).reset_index(drop=True)
    return {"meta": one.meta.iloc[0].to_dict(), "concepts": concepts, "neighbours": nb,
            "chosen": acts.names[chosen],
            "predicted_influencer": nb["neighbour"].iloc[0] if len(nb) else ""}


def tie_attribution(data: Decisions, cfg: dict, rec: dict) -> pd.DataFrame:
    """Importance of tie (i, j): mean share of j in i's peer approval plus mean share of i in j's."""
    parts = []
    for c in data.chunks():
        contrib = neighbour_contributions(c, cfg, rec)
        tot = contrib.sum(1, keepdims=True)
        share = np.divide(contrib, tot, out=np.zeros_like(contrib), where=tot > 0)
        m = c.mask
        agents = np.repeat(c.meta["agent"].to_numpy()[:, None], m.shape[1], 1)
        parts.append(pd.DataFrame({"agent": agents[m], "neighbour": c.nbr_label[m], "share": share[m]}))
    df = pd.concat(parts, ignore_index=True)
    directed = df.groupby(["agent", "neighbour"])["share"].mean()
    rows = {}
    for (i, j), v in directed.items():
        key = tuple(sorted((i, j)))
        rows[key] = rows.get(key, 0.0) + v
    return pd.DataFrame([(i, j, v) for (i, j), v in rows.items()], columns=["agent_i", "agent_j", "attributed_importance"])


def surrogate_tree(data: Decisions, cfg: dict, rec: dict, depth: int = 3) -> dict:
    """Stretch: a depth-3 decision tree mimicking the recovered policy's most likely action."""
    acts = Actions(cfg)
    target = []
    for c in data.chunks():
        phi = padded_features(acts, recovered_alpha(c, rec), padded_similarity(acts, c.nbr_last, c.mask),
                              static_features(acts, c.sup, c.mean_dev))
        target.append((phi @ np.asarray(rec["theta_hat"])).argmax(1))
    target = np.concatenate(target)
    x = bc_features(data, acts)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=int(cfg["seed"])).fit(x, target)
    rules = export_text(tree, feature_names=BC_FEATURES, class_names=[acts.names[c] for c in tree.classes_])
    return {"rules": rules, "fidelity": float((tree.predict(x) == target).mean())}
