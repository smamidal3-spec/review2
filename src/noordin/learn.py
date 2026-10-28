"""Step 3: imitation learning from recorded behaviour only (SPEC 8).

Inputs are the public tables decisions.csv and neighbours.csv plus the public config.
This module never sees the true parameters.

Main learner: maximum-likelihood fit of (theta_1..4, gamma_status, gamma_tie) under the
same softmax-over-linear-reward choice model the agents use. Because agents choose
myopically, recovering their reward is exactly MLE of a multinomial-logit (Luce) model:
inverse reinforcement learning in its simplest exact form.

Baseline: behavioural cloning with a multinomial logistic regression on attention-free
aggregate features.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import (Actions, padded_attention, padded_features, padded_similarity, softmax_rows,
                       static_features)

PARAM_NAMES = ["theta_peer_approval", "theta_material_gain", "theta_status_gain", "theta_sanction_risk",
               "gamma_status", "gamma_tie"]
STARTS = ([0, 0, 0, 0, 1, 0], [0.5, 0.5, 0.5, -0.5, 3, 1])
BC_FEATURES = ["supervision", "deviance", "n_neighbours", "mean_nbr_level", "max_nbr_level",
               "mean_nbr_deviance", "mean_nbr_status"]


# ---------------------------------------------------------------- data

CHUNK = 20000          # decisions per padded block; bounds memory on high-degree networks


@dataclass
class Decisions:
    """Decisions with neighbours padded to K = maximum degree. Row order follows decisions.csv.

    An agent's neighbours, their status and their tie feature are the same in every decision
    (the network is fixed), so they are stored once per agent (`a_*`, shape (agents, K)).
    Only what changes per decision is stored per row: the neighbours' last actions and their
    mean deviance. The padded (D, K) views are built on access; use `chunks()` for large data.
    """
    meta: pd.DataFrame          # episode, step, agent, supervision, deviance, n_neighbours, action
    action: np.ndarray          # (D,)
    sup: np.ndarray             # (D,)
    agent: np.ndarray           # (D,) row into the per-agent tables
    nbr_last: np.ndarray        # (D, K) int8 neighbour's previous action
    mean_dev: np.ndarray        # (D,) mean neighbour deviance (0 without neighbours)
    a_mask: np.ndarray          # (agents, K) bool
    a_label: np.ndarray         # (agents, K) neighbour label, "" where masked
    a_status: np.ndarray        # (agents, K)
    a_tie: np.ndarray           # (agents, K) tie feature w_ij

    def __len__(self):
        return len(self.action)

    @property
    def mask(self):
        return self.a_mask[self.agent]

    @property
    def nbr_label(self):
        return self.a_label[self.agent]

    @property
    def nbr_status(self):
        return self.a_status[self.agent]

    @property
    def nbr_tie(self):
        return self.a_tie[self.agent]

    @property
    def n_nbr(self):
        return self.a_mask.sum(1)[self.agent]

    def subset(self, rows) -> "Decisions":
        rows = np.asarray(rows)
        return Decisions(self.meta.iloc[rows].reset_index(drop=True), self.action[rows], self.sup[rows],
                         self.agent[rows], self.nbr_last[rows], self.mean_dev[rows],
                         self.a_mask, self.a_label, self.a_status, self.a_tie)

    def episodes(self, eps) -> "Decisions":
        return self.subset(np.where(self.meta["episode"].isin(list(eps)).to_numpy())[0])

    def chunks(self, size: int = CHUNK):
        for start in range(0, len(self), size):
            yield self.subset(np.arange(start, min(start + size, len(self))))


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the masked slots of each row, 0 for rows with no slots."""
    n = mask.sum(1)
    return np.where(n > 0, (values * mask).sum(1) / np.maximum(n, 1), 0.0)


def prepare(decisions: pd.DataFrame, neighbours: pd.DataFrame, k_max: int | None = None) -> Decisions:
    dec = decisions.reset_index(drop=True)
    key = ["episode", "step", "agent"]
    idx = dec[key].assign(row=np.arange(len(dec)))
    nb = neighbours.merge(idx, on=key, how="inner", validate="many_to_one")
    nb = nb.sort_values("row", kind="stable").reset_index(drop=True)
    nb["slot"] = nb.groupby("row").cumcount()
    k = int(k_max or max(1, int(dec["n_neighbours"].max())))
    d = len(dec)
    agents, agent = np.unique(dec["agent"].to_numpy(), return_inverse=True)
    n_agents = len(agents)

    # per-agent neighbour tables, taken from each agent's first decision
    first = pd.Series(np.arange(d)).groupby(agent).first().to_numpy()
    nb_agent = agent[nb["row"].to_numpy()]
    r, s = nb["row"].to_numpy(), nb["slot"].to_numpy()
    a_mask = np.zeros((n_agents, k), bool)
    a_label = np.full((n_agents, k), "", dtype=object)
    a_status, a_tie = np.zeros((n_agents, k)), np.zeros((n_agents, k))
    lead = np.isin(r, first)
    a_mask[nb_agent[lead], s[lead]] = True
    a_label[nb_agent[lead], s[lead]] = nb["neighbour"].to_numpy()[lead]
    a_status[nb_agent[lead], s[lead]] = nb["nbr_status"].to_numpy()[lead]
    a_tie[nb_agent[lead], s[lead]] = nb["tie"].to_numpy()[lead]
    # the network is fixed, so every decision must list the same neighbours in the same order
    same = ((a_label[nb_agent, s] == nb["neighbour"].to_numpy())
            & (a_status[nb_agent, s] == nb["nbr_status"].to_numpy())
            & (a_tie[nb_agent, s] == nb["tie"].to_numpy()))
    counts = np.bincount(r, minlength=d)
    if not same.all() or not (counts == a_mask.sum(1)[agent]).all():
        raise ValueError("an agent's neighbours change between decisions; the learner assumes a fixed network")

    nbr_last = np.zeros((d, k), np.int8)
    nbr_last[r, s] = nb["nbr_last_action"].to_numpy()
    # mean neighbour deviance, computed on padded blocks exactly as the padded formula would
    mean_dev = np.zeros(d)
    dev_all = nb["nbr_deviance"].to_numpy()
    for r0 in range(0, d, CHUNK):
        r1 = min(r0 + CHUNK, d)
        lo, hi = np.searchsorted(r, [r0, r1])
        block = np.zeros((r1 - r0, k))
        block[r[lo:hi] - r0, s[lo:hi]] = dev_all[lo:hi]
        mean_dev[r0:r1] = masked_mean(block, a_mask[agent[r0:r1]])
    return Decisions(dec, dec["action"].to_numpy(int), dec["supervision"].to_numpy(float), agent,
                     nbr_last, mean_dev, a_mask, a_label, a_status, a_tie)


def split_by_episode(data: Decisions, test_fraction: float):
    eps = np.sort(data.meta["episode"].unique())
    n_test = max(1, int(round(len(eps) * test_fraction))) if len(eps) > 1 else 0
    train_eps, test_eps = eps[: len(eps) - n_test], eps[len(eps) - n_test:]
    return data.episodes(train_eps), data.episodes(test_eps), train_eps, test_eps


# ---------------------------------------------------------------- reward and attention recovery

class ChoiceModel:
    """Negative log-likelihood and its analytic gradient for the six parameters.

    Works on a flat list of (decision, neighbour) entries rather than the padded
    tensors, so the cost scales with the number of real ties, not with max degree.
    Features are held feature-major as (D, A) arrays to keep every step elementwise.
    """

    def __init__(self, data: Decisions, acts: Actions, beta: float):
        self.beta = float(beta)
        self.d = len(data)
        self.n_act = acts.n
        self.row, slot = np.nonzero(data.mask)                                  # entries, row-major
        a = data.agent[self.row]
        self.status = data.a_status[a, slot]
        self.tie = data.a_tie[a, slot]
        last = data.nbr_last[self.row, slot]
        # sim[a, e] = 1 - |level[a] - level[last_e]|
        self.sim = 1.0 - np.abs(acts.level[:, None] - acts.level[last][None, :])
        static = static_features(acts, data.sup, data.mean_dev)                 # (D, A, 3)
        self.static = [np.ascontiguousarray(static[..., k]) for k in range(3)]
        self.onehot = np.zeros((self.d, acts.n))
        self.onehot[np.arange(self.d), data.action] = 1.0

    def _rowsum(self, w):
        return np.bincount(self.row, weights=w, minlength=self.d)

    def _per_action(self, w):
        """sum over a decision's neighbours of w_e * sim[a, e], shape (D, A)."""
        return np.column_stack([self._rowsum(w * s) for s in self.sim])

    def _resid(self, f, prob):
        """sum over decisions of f[chosen] - E_prob[f]."""
        return float(((self.onehot - prob) * f).sum())

    def nll_grad(self, p):
        th, gs, gk = p[:4], p[4], p[5]
        e = np.exp(np.clip(gs * self.status + gk * self.tie, -500, 500))
        alpha = e / self._rowsum(e)[self.row]
        feats = [self._per_action(alpha)] + self.static                         # 4 x (D, A)
        u = self.beta * sum(t * f for t, f in zip(th, feats))
        u -= u.max(1, keepdims=True)
        eu = np.exp(u)
        z = eu.sum(1)
        prob = eu / z[:, None]
        nll = float(np.log(z).sum() - (self.onehot * u).sum())
        grad = [-self.beta * self._resid(f, prob) for f in feats]
        for x in (self.status, self.tie):                                       # d alpha / d gamma
            dalpha = alpha * (x - self._rowsum(alpha * x)[self.row])
            grad.append(-self.beta * th[0] * self._resid(self._per_action(dalpha), prob))
        return nll, np.asarray(grad)


def fit_reward(data: Decisions, cfg: dict) -> dict:
    model = ChoiceModel(data, Actions(cfg), cfg["choice"]["beta"])
    best = None
    for x0 in STARTS:
        res = minimize(model.nll_grad, np.asarray(x0, float), jac=True, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    x = best.x
    return {
        "theta_hat": x[:4].tolist(),
        "gamma_hat": {"gamma_status": float(x[4]), "gamma_tie": float(x[5])},
        "nll": float(best.fun),
        "n_decisions": int(len(data)),
        "converged": bool(best.success),
    }


def data_efficiency(train: Decisions, train_eps, cfg: dict, full_fit: dict | None = None) -> pd.DataFrame:
    """Refit on the first k training episodes. k is capped at the number available."""
    rows = []
    for k in sorted({min(int(k), len(train_eps)) for k in cfg["run"]["data_efficiency_episodes"]}):
        used = train_eps[:k]
        fit = full_fit if (full_fit and k == len(train_eps)) else fit_reward(train.episodes(used), cfg)
        rows.append({"episodes": len(used), "decisions": fit["n_decisions"],
                     **dict(zip(PARAM_NAMES[:4], fit["theta_hat"])),
                     "gamma_status_hat": fit["gamma_hat"]["gamma_status"],
                     "gamma_tie_hat": fit["gamma_hat"]["gamma_tie"]})
    return pd.DataFrame(rows)


def recovered_policy_probs(data: Decisions, cfg: dict, rec: dict) -> np.ndarray:
    """P(a | decision) under recovered parameters, shape (D, A)."""
    acts = Actions(cfg)
    out = []
    for c in data.chunks():
        alpha = padded_attention(c.nbr_status, c.nbr_tie, c.mask,
                                 rec["gamma_hat"]["gamma_status"], rec["gamma_hat"]["gamma_tie"])
        phi = padded_features(acts, alpha, padded_similarity(acts, c.nbr_last, c.mask),
                              static_features(acts, c.sup, c.mean_dev))
        out.append(softmax_rows(cfg["choice"]["beta"] * phi @ np.asarray(rec["theta_hat"])))
    return np.concatenate(out) if out else np.zeros((0, acts.n))


# ---------------------------------------------------------------- behavioural cloning

def bc_features(data: Decisions, acts: Actions) -> np.ndarray:
    n = data.n_nbr
    safe = np.maximum(n, 1)
    mean_status = ((data.a_status * data.a_mask).sum(1) / np.maximum(data.a_mask.sum(1), 1))[data.agent]
    lv_sum, lv_max = np.zeros(len(data)), np.zeros(len(data))
    pos = 0
    for c in data.chunks():
        lv = acts.level[c.nbr_last] * c.mask
        lv_sum[pos:pos + len(c)], lv_max[pos:pos + len(c)] = lv.sum(1), lv.max(1)
        pos += len(c)
    return np.column_stack([
        data.sup, data.meta["deviance"].to_numpy(float), n,
        lv_sum / safe, lv_max, data.mean_dev, mean_status,
    ])


def bc_features_dense(ctx: dict, acts: Actions) -> np.ndarray:
    """Same features from a simulator state, for rolling the BC policy forward."""
    adj = ctx["adj"].astype(float)
    n = adj.sum(1)
    safe = np.maximum(n, 1)
    lv = acts.level[ctx["last"]]
    return np.column_stack([
        ctx["sup"], ctx["dev"], n, adj @ lv / safe, (adj * lv[None, :]).max(1),
        adj @ ctx["dev"] / safe, adj @ ctx["status"] / safe,
    ])


def fit_bc(train: Decisions, cfg: dict):
    acts = Actions(cfg)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(bc_features(train, acts), train.action)
    return clf


def evaluate_bc(clf, test: Decisions, train: Decisions, cfg: dict) -> dict:
    acts = Actions(cfg)
    pred = clf.predict(bc_features(test, acts))
    labels = list(range(acts.n))
    bad_true, bad_pred = np.isin(test.action, acts.bad), np.isin(pred, acts.bad)
    majority = np.bincount(train.action, minlength=acts.n).argmax()
    return {
        "accuracy": float((pred == test.action).mean()),
        "majority_class": int(majority),
        "majority_accuracy": float((test.action == majority).mean()),
        "confusion": confusion_matrix(test.action, pred, labels=labels).tolist(),
        "good_bad_confusion": confusion_matrix(bad_true, bad_pred, labels=[False, True]).tolist(),
        "good_bad_accuracy": float((bad_true == bad_pred).mean()),
    }


def evaluate_recovered(test: Decisions, cfg: dict, rec: dict) -> dict:
    acts = Actions(cfg)
    prob = recovered_policy_probs(test, cfg, rec)
    pred = prob.argmax(1)
    bad_true, bad_pred = np.isin(test.action, acts.bad), np.isin(pred, acts.bad)
    return {
        "accuracy": float((pred == test.action).mean()),
        "mean_log_likelihood": float(np.log(prob[np.arange(len(test)), test.action]).mean()),
        "confusion": confusion_matrix(test.action, pred, labels=list(range(acts.n))).tolist(),
        "good_bad_confusion": confusion_matrix(bad_true, bad_pred, labels=[False, True]).tolist(),
        "good_bad_accuracy": float((bad_true == bad_pred).mean()),
    }


# ---------------------------------------------------------------- driver

def load_public_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(cfg["outputs_dir"])
    return pd.read_csv(out / "decisions.csv"), pd.read_csv(out / "neighbours.csv", keep_default_na=False)


def run(cfg: dict, decisions: pd.DataFrame | None = None, neighbours: pd.DataFrame | None = None,
        write: bool = True, efficiency: bool = True) -> dict:
    if decisions is None:
        decisions, neighbours = load_public_tables(cfg)
    data = prepare(decisions, neighbours)
    train, test, train_eps, test_eps = split_by_episode(data, cfg["run"]["test_fraction"])
    rec = fit_reward(train, cfg)
    clf = fit_bc(train, cfg)
    result = {
        "recovered": rec,
        "bc": evaluate_bc(clf, test, train, cfg),
        "recovered_eval": evaluate_recovered(test, cfg, rec),
        "train_episodes": train_eps.tolist(),
        "test_episodes": test_eps.tolist(),
        "bc_model": clf,
        "train": train,
        "test": test,
    }
    if efficiency:
        result["data_efficiency"] = data_efficiency(train, train_eps, cfg, rec)
    if write:
        out = Path(cfg["outputs_dir"])
        with open(out / "recovered.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        with open(out / "learn_metrics.json", "w", encoding="utf-8") as f:
            json.dump({k: result[k] for k in ("bc", "recovered_eval", "train_episodes", "test_episodes")}, f, indent=2)
    return result
