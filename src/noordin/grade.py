"""All comparisons against ground truth. One of the few modules allowed the answer key."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import THETA_NAMES, load_answer_key
from .learn import PARAM_NAMES


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def grade_recovery(rec: dict) -> dict:
    key = load_answer_key()
    th, th_hat = np.asarray(key["theta"]), np.asarray(rec["theta_hat"])
    return {
        "theta_true": th.tolist(),
        "theta_hat": th_hat.tolist(),
        "cosine": cosine(th_hat, th),
        "max_abs_error": float(np.abs(th_hat - th).max()),
        "signs_correct": bool((np.sign(th_hat) == np.sign(th)).all()),
        "gamma_status_true": key["gamma_status"],
        "gamma_tie_true": key["gamma_tie"],
        "gamma_status_hat": rec["gamma_hat"]["gamma_status"],
        "gamma_tie_hat": rec["gamma_hat"]["gamma_tie"],
    }


def recovery_table(rec: dict) -> pd.DataFrame:
    g = grade_recovery(rec)
    rows = [(n, t, h) for n, t, h in zip(THETA_NAMES, g["theta_true"], g["theta_hat"])]
    rows += [("gamma_status", g["gamma_status_true"], g["gamma_status_hat"]),
             ("gamma_tie", g["gamma_tie_true"], g["gamma_tie_hat"])]
    df = pd.DataFrame(rows, columns=["parameter", "true", "recovered"])
    df["error"] = df["recovered"] - df["true"]
    return df


def grade_data_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    th = np.asarray(load_answer_key()["theta"])
    df = df.copy()
    df.insert(2, "cosine", [cosine(r, th) for r in df[PARAM_NAMES[:4]].to_numpy(float)])
    return df


# ---------------------------------------------------------------- explanations

def true_influencers(meta: pd.DataFrame, influence: pd.DataFrame) -> np.ndarray:
    """peer_id from the influence log, aligned to the rows of `meta`."""
    key = ["episode", "step", "agent"]
    m = meta[key].merge(influence[key + ["peer_id"]], on=key, how="left", validate="one_to_one")
    return m["peer_id"].fillna("").to_numpy(object)


def match_rates(pred_topk: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> dict:
    p, t = pred_topk[keep], truth[keep]
    return {"top1": float((p[:, 0] == t).mean()), "top3": float((p == t[:, None]).any(1).mean()), "n": int(keep.sum())}


def grade_explanations(test, predicted: np.ndarray, similar: np.ndarray, influence: pd.DataFrame) -> dict:
    """Top-1/Top-3 on held-out decisions of agents with at least two neighbours."""
    truth = true_influencers(test.meta, influence)
    deg = test.mask.sum(1)
    keep = deg >= 2
    rand1 = float(np.mean(1.0 / deg[keep]))
    rand3 = float(np.mean(np.minimum(3, deg[keep]) / deg[keep]))
    return {
        "recovered": match_rates(predicted, truth, keep),
        "most_similar": match_rates(similar, truth, keep),
        "random": {"top1": rand1, "top3": rand3, "n": int(keep.sum())},
    }


def ablation_agreement(attribution: pd.DataFrame, ablation: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    a = attribution.copy()
    a[["agent_i", "agent_j"]] = np.sort(a[["agent_i", "agent_j"]].to_numpy(), 1)
    b = ablation.copy()
    b[["agent_i", "agent_j"]] = np.sort(b[["agent_i", "agent_j"]].to_numpy(), 1)
    m = b.merge(a, on=["agent_i", "agent_j"], how="left").fillna({"attributed_importance": 0.0})
    rho, p = spearmanr(m["attributed_importance"], m["mean_delta"].abs())
    return {"spearman": float(rho), "p_value": float(p), "n_ties": int(len(m)), "table": m}
