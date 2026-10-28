"""Concept features and attention (SPEC 5.2, 5.3).

Shared by the simulator and the learner. Contains no true parameter values: every
weight is passed in by the caller.

Two layouts are supported:
  * dense:  (N, N) adjacency matrices, used by the simulator for one step of all agents;
  * padded: one row per decision with neighbours padded to K, used by the learner.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["peer_approval", "material_gain", "status_gain", "sanction_risk"]


class Actions:
    """Action tables from the public config."""

    def __init__(self, cfg: dict):
        a = cfg["actions"]
        self.names = list(a["names"])
        self.level = np.asarray(a["level"], dtype=float)
        self.severity = np.asarray(a["severity"], dtype=float)
        self.material = np.asarray(a["material"], dtype=float)
        self.bad = np.asarray(a["bad"], dtype=int)
        self.n = len(self.names)


# ---------------------------------------------------------------- dense layout

def attention(adj: np.ndarray, tie: np.ndarray, status: np.ndarray, gamma_status: float, gamma_tie: float) -> np.ndarray:
    """alpha[i, j] = softmax over j in N(i) of gamma_status*status_j + gamma_tie*w_ij (w = the tie feature)."""
    score = gamma_status * status[None, :] + gamma_tie * tie.astype(float)
    score = np.where(adj, score, -np.inf)
    has = adj.any(1, keepdims=True)
    score = score - np.where(has, np.max(np.where(adj, score, -1e300), 1, keepdims=True), 0.0)
    e = np.where(adj, np.exp(score), 0.0)
    z = e.sum(1, keepdims=True)
    return np.divide(e, z, out=np.zeros_like(e), where=z > 0)


def similarity(acts: Actions, last: np.ndarray) -> np.ndarray:
    """sim[a, j] = 1 - |level[a] - level[last_j]|, shape (A, N)."""
    return 1.0 - np.abs(acts.level[:, None] - acts.level[last][None, :])


def dense_features(acts: Actions, alpha, adj, last, dev, sup) -> np.ndarray:
    """phi with shape (N, A, 4) for every agent and candidate action."""
    sim = similarity(acts, last)                                   # (A, N)
    pa = alpha @ sim.T                                             # (N, A)
    deg = adj.sum(1)
    nbr_dev = np.where(deg > 0, (adj * dev[None, :]).sum(1) / np.maximum(deg, 1), 0.0)
    n = len(sup)
    mat = np.broadcast_to(acts.material[None, :], (n, acts.n))
    sg = nbr_dev[:, None] * acts.severity[None, :]
    sr = sup[:, None] * acts.severity[None, :]
    return np.stack([pa, mat, sg, sr], -1)


def softmax_rows(u: np.ndarray) -> np.ndarray:
    u = u - u.max(-1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- padded layout

def padded_attention(nbr_status, nbr_tie, mask, gamma_status, gamma_tie):
    """Same attention as `attention`, one row per decision. Shapes (D, K)."""
    score = gamma_status * nbr_status + gamma_tie * nbr_tie
    score = np.where(mask, score, -np.inf)
    has = mask.any(1, keepdims=True)
    score = score - np.where(has, np.max(np.where(mask, score, -1e300), 1, keepdims=True), 0.0)
    e = np.where(mask, np.exp(score), 0.0)
    z = e.sum(1, keepdims=True)
    return np.divide(e, z, out=np.zeros_like(e), where=z > 0)


def padded_similarity(acts: Actions, nbr_last, mask):
    """sim[d, a, k], zero where masked. nbr_last holds action indices (any value where masked)."""
    lv = acts.level[np.clip(nbr_last, 0, acts.n - 1)]
    sim = 1.0 - np.abs(acts.level[None, :, None] - lv[:, None, :])
    return sim * mask[:, None, :]


def static_features(acts: Actions, sup, mean_dev):
    """The three features that do not depend on attention: (D, A, 3).
    `mean_dev` is each decision's mean neighbour deviance (0 without neighbours)."""
    d = len(sup)
    mat = np.broadcast_to(acts.material[None, :], (d, acts.n))
    sg = mean_dev[:, None] * acts.severity[None, :]
    sr = sup[:, None] * acts.severity[None, :]
    return np.stack([mat, sg, sr], -1)


def padded_features(acts: Actions, alpha, sim, static):
    """Full phi (D, A, 4) from padded attention, similarity and static features."""
    pa = np.einsum("dk,dak->da", alpha, sim)
    return np.concatenate([pa[..., None], static], -1)
