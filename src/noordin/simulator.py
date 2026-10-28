"""Step 2: agents, hidden reward, choice and logging (SPEC 5).

The simulator is one of the few modules allowed to read the answer key. Policies are
plain callables, so the same dynamics can be driven by the true reward, a recovered
reward or a behavioural-cloning classifier (rollout.py passes its own policy in).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_answer_key
from .features import Actions, attention, dense_features, softmax_rows
from .load_graph import Graph, components

# Independent RNG streams so experiments never share draws by accident.
STREAM_TRAIN, STREAM_ABLATION, STREAM_INTERVENTION, STREAM_ROLLOUT, STREAM_SENSITIVITY = range(5)


# ---------------------------------------------------------------- policies

class UtilityPolicy:
    """Softmax over beta * theta . phi(i, a) with attention parameters (gamma_status, gamma_tie)."""

    def __init__(self, acts: Actions, beta: float, theta, gamma_status: float, gamma_tie: float):
        self.acts = acts
        self.beta = float(beta)
        self.theta = np.asarray(theta, dtype=float)
        self.gamma_status = float(gamma_status)
        self.gamma_tie = float(gamma_tie)
        self._alpha_cache: dict[bytes, np.ndarray] = {}

    def alpha(self, ctx) -> np.ndarray:
        key = np.packbits(ctx["adj"]).tobytes()      # content key: ids are reused after GC
        if key not in self._alpha_cache:
            self._alpha_cache[key] = attention(ctx["adj"], ctx["tie"], ctx["status"], self.gamma_status, self.gamma_tie)
        return self._alpha_cache[key]

    def features(self, ctx) -> np.ndarray:
        return dense_features(self.acts, self.alpha(ctx), ctx["adj"], ctx["last"], ctx["dev"], ctx["sup"])

    def __call__(self, ctx) -> np.ndarray:
        return softmax_rows(self.beta * self.features(ctx) @ self.theta)


def true_policy(cfg: dict, theta_scale=None, theta=None) -> UtilityPolicy:
    """Policy under the answer key. `theta` overrides the true weights (used by the app's sliders)."""
    key = load_answer_key()
    th = np.asarray(key["theta"] if theta is None else theta, dtype=float)
    if theta_scale is not None:
        th = th * np.asarray(theta_scale)
    return UtilityPolicy(Actions(cfg), cfg["choice"]["beta"], th, key["gamma_status"], key["gamma_tie"])


# ---------------------------------------------------------------- scenarios and episodes

@dataclass
class Scenario:
    """What differs from the baseline. Defaults reproduce the baseline exactly."""
    name: str = "baseline"
    adj: np.ndarray | None = None       # episode graph; None = pre-attack graph
    supervision_add: float = 0.0
    mentor: int | None = None           # agent index forced to prosocial with deviance 0
    exclude: tuple = ()                 # agent indices left out of the bad-share measurement


@dataclass
class Episode:
    actions: np.ndarray                 # (T, N)
    bad_share: np.ndarray               # (T,)
    first_escalation: np.ndarray        # (N,) step of first action 3, NaN if never
    logs: dict = field(default_factory=dict)

    def final_bad_share(self, window: int) -> float:
        return float(self.bad_share[-window:].mean())


def episode_rngs(cfg: dict, stream: int, seed: int):
    ss = np.random.SeedSequence([int(cfg["seed"]), int(stream), int(seed)])
    init, steps = ss.spawn(2)
    return np.random.default_rng(init), np.random.default_rng(steps)


def initial_state(graph: Graph, cfg: dict, rng: np.random.Generator):
    """SPEC 5.1. Seeded agents are chosen per connected group of the pre-attack graph,
    so a scenario that changes the network never changes who starts out deviant."""
    ag = cfg["agents"]
    n = graph.n
    sup = rng.uniform(*ag["supervision_range"], n)
    self_control = rng.uniform(*ag["self_control_range"], n)
    dev = rng.uniform(*ag["init_deviance_range"], n)
    for comp in components(graph.adj):
        if len(comp) >= 2:
            dev[comp[rng.integers(len(comp))]] = ag["seed_deviance"]
    return sup, self_control, dev


def run_episode(graph: Graph, cfg: dict, policy, stream: int, seed: int,
                scenario: Scenario | None = None, log: bool = False, episode_id: int = 0) -> Episode:
    sc = scenario or Scenario()
    acts = Actions(cfg)
    T = int(cfg["run"]["steps"])
    kappa = float(cfg["agents"]["kappa"])
    adj = graph.adj if sc.adj is None else sc.adj
    tie = graph.tie * adj                         # added ties carry no tie feature
    init_rng, step_rng = episode_rngs(cfg, stream, seed)

    sup, self_control, dev = initial_state(graph, cfg, init_rng)
    sup = np.clip(sup + sc.supervision_add, 0.0, 1.0)
    if sc.mentor is not None:
        dev[sc.mentor] = 0.0
    last = np.where(dev > 0.5, 3, 1)
    if sc.mentor is not None:
        last[sc.mentor] = 0

    n = graph.n
    actions = np.zeros((T, n), dtype=int)
    first = np.full(n, np.nan)
    logger = _Logger(graph, adj, tie, policy, acts, episode_id) if log else None
    ctx = {"adj": adj, "tie": tie, "status": graph.status}

    for t in range(T):
        ctx.update(last=last, dev=dev, sup=sup)
        p = policy(ctx)
        u = step_rng.random(n)                       # exactly one draw per agent per step
        a = np.minimum((u[:, None] > np.cumsum(p, 1)).sum(1), acts.n - 1)
        if sc.mentor is not None:
            a[sc.mentor] = 0
        if logger:
            logger.record(t, a, last, dev, sup, ctx)
        dev = np.clip(dev + kappa * (1 - 0.5 * self_control) * (acts.severity[a] - dev), 0.0, 1.0)
        if sc.mentor is not None:
            dev[sc.mentor] = 0.0
        last = a
        actions[t] = a
        newly = np.isnan(first) & (a == 3)
        first[newly] = t

    keep = np.ones(n, bool)
    keep[list(sc.exclude)] = False
    bad = np.isin(actions[:, keep], acts.bad).mean(1)
    return Episode(actions, bad, first, logger.frames() if logger else {})


class _Logger:
    """Builds the four SPEC 5.5 tables. Only the private two need the true parameters."""

    def __init__(self, graph, adj, tie, policy, acts, episode_id):
        self.labels = np.array(graph.labels)
        self.status = graph.status
        self.adj, self.tie, self.acts, self.ep = adj, tie, acts, episode_id
        self.policy = policy
        self.private = isinstance(policy, UtilityPolicy)
        self.ii, self.jj = np.where(adj)
        self.deg = adj.sum(1)
        self.cols = {k: {} for k in ("decisions", "neighbours", "reward_log", "influence_log")}

    def _add(self, table, **cols):
        for k, v in cols.items():
            self.cols[table].setdefault(k, []).append(np.asarray(v))

    def record(self, t, a, last, dev, sup, ctx):
        n = len(a)
        idx = np.arange(n)
        ii, jj = self.ii, self.jj
        self._add("decisions", step=np.full(n, t), agent=idx, supervision=sup.copy(), deviance=dev.copy(),
                  n_neighbours=self.deg, action=a.copy())
        self._add("neighbours", step=np.full(len(ii), t), agent=ii, neighbour=jj, nbr_last_action=last[jj],
                  nbr_status=self.status[jj], tie=self.tie[ii, jj], nbr_deviance=dev[jj])
        if not self.private:
            return
        phi = self.policy.features(ctx)[idx, a]                  # (N, 4) for the chosen action
        self._add("reward_log", step=np.full(n, t), agent=idx, phi_peer_approval=phi[:, 0],
                  phi_material_gain=phi[:, 1], phi_status_gain=phi[:, 2], phi_sanction_risk=phi[:, 3],
                  reward=phi @ self.policy.theta)
        alpha = self.policy.alpha(ctx)
        contrib = alpha * (1 - np.abs(self.acts.level[a][:, None] - self.acts.level[last][None, :]))
        peer = np.where(self.adj, contrib, -1.0).argmax(1)
        has = self.deg > 0
        self._add("influence_log", step=np.full(n, t), agent=idx, peer_idx=np.where(has, peer, -1),
                  true_alpha_of_peer=np.where(has, alpha[idx, peer], np.nan))

    def frames(self):
        out = {}
        labels_or_blank = np.append(self.labels, "")          # index -1 maps to ""
        for name, cols in self.cols.items():
            if not cols:
                continue
            data = {k: np.concatenate(v) for k, v in cols.items()}
            df = pd.DataFrame({"episode": self.ep, **data})
            for c in ("agent", "neighbour"):
                if c in df:
                    df[c] = self.labels[df[c].to_numpy()]
            if "peer_idx" in df:
                df.insert(df.columns.get_loc("peer_idx"), "peer_id", labels_or_blank[df.pop("peer_idx").to_numpy()])
            out[name] = df
        return out


# ---------------------------------------------------------------- dataset

def simulate_dataset(graph: Graph, cfg: dict, n_episodes: int | None = None,
                     policy: UtilityPolicy | None = None) -> dict[str, pd.DataFrame]:
    """Training data: n_episodes on the pre-attack graph under the true reward (or `policy`)."""
    n_episodes = int(n_episodes or cfg["run"]["train_episodes"])
    policy = policy or true_policy(cfg)
    parts: dict[str, list] = {}
    for e in range(n_episodes):
        ep = run_episode(graph, cfg, policy, STREAM_TRAIN, e, log=True, episode_id=e)
        for k, df in ep.logs.items():
            parts.setdefault(k, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def write_dataset(data: dict[str, pd.DataFrame], cfg: dict) -> None:
    out = Path(cfg["outputs_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
