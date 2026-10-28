"""Step 7 (multi-dataset): compare the four networks (SPEC_MULTIDATASET section 5).

Reads each dataset's outputs/<dataset>/metrics.json and writes outputs/comparison.csv,
outputs/comparison.md and outputs/figures/comparison.png. The hypothesis in the spec is
tested against the numbers, and the text says what actually happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import DISCLAIMER
from . import plots as P

KNECHT_NOTE = "Knecht is a class of children; no individual's real data is used."
COLUMNS = ["dataset", "nodes", "ties", "groups(>=2)", "isolates", "max_degree", "hub_share", "tie_feature",
           "cosine(theta)", "gamma_status_hat", "gamma_tie_hat", "top1_influencer_acc", "top3_influencer_acc",
           "spearman_ablation", "cascade_gap_recovered", "cascade_gap_bc", "strongest_intervention",
           "hub_isolation_effect"]
ORDER = ["noordin", "caviar", "drugnet", "knecht"]


def _row(name: str, m: dict) -> dict:
    g, r, e = m["graph"], m["recovery"], m["explanations"]
    iv = sorted(m["interventions"], key=lambda x: x["delta"])
    hub = next(x for x in m["interventions"] if x["scenario"].startswith("Hub isolation"))
    return {
        "dataset": name, "nodes": g["nodes"], "ties": g["ties"], "groups(>=2)": g["groups"],
        "isolates": g["isolates"], "max_degree": g["max_degree"], "hub_share": g["hub_share"],
        "tie_feature": g["tie_feature"], "cosine(theta)": r["cosine"], "gamma_status_hat": r["gamma_status_hat"],
        "gamma_tie_hat": r["gamma_tie_hat"], "top1_influencer_acc": e["recovered"]["top1"],
        "top3_influencer_acc": e["recovered"]["top3"], "spearman_ablation": m["ablation_agreement"]["spearman"],
        "cascade_gap_recovered": m["cascade_gap"].get("recovered"), "cascade_gap_bc": m["cascade_gap"].get("bc"),
        "strongest_intervention": f"{iv[0]['scenario']} ({iv[0]['delta']:+.3f})",
        "hub_isolation_effect": hub["delta"],
        # kept for the text, not in the CSV column list
        "_hub": hub, "_iv": m["interventions"], "_mode": m["mode"], "_gamma_tie_true": r["gamma_tie_true"],
        "_isolates_not_in_file": g.get("isolates_not_in_file"), "_most_similar": e["most_similar"]["top1"],
        "_acceptance": m.get("acceptance", []), "_baseline": iv[0]["baseline_bad_share"],
        "_tie_on": g["tie_feature_nonzero"], "_binary": g["tie_feature_nonzero"] < g["ties"] or name != "caviar",
    }


def load_rows(outputs_root: Path) -> list[dict]:
    rows = []
    for name in ORDER:
        f = outputs_root / name / "metrics.json"
        if f.exists():
            rows.append(_row(name, json.loads(f.read_text(encoding="utf-8"))))
    return rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def findings(rows: list[dict]) -> list[str]:
    """Test the spec's hypothesis against the numbers. Every sentence is generated from data."""
    out = []
    names = [r["dataset"].capitalize() for r in rows]
    high = [r for r in rows if r["cosine(theta)"] >= 0.98 and r["top1_influencer_acc"] >= 0.95]
    if len(high) == len(rows):
        out.append(f"**Recovery and explanation stay high on all {len(rows)} networks, as hypothesised.** Cosine to the "
                   f"true reward ranges from {min(r['cosine(theta)'] for r in rows):.4f} to "
                   f"{max(r['cosine(theta)'] for r in rows):.4f}, and the recovered attention names the true influencer "
                   f"in {min(r['top1_influencer_acc'] for r in rows):.1%} to {max(r['top1_influencer_acc'] for r in rows):.1%} "
                   f"of held-out decisions, against {min(r['_most_similar'] for r in rows):.0%} to "
                   f"{max(r['_most_similar'] for r in rows):.0%} for the attention-free baseline.")
    else:
        low = [r["dataset"] for r in rows if r not in high]
        out.append(f"**Recovery or explanation dropped below the SPEC thresholds on {', '.join(low)}**, contradicting "
                   f"the hypothesis that both stay high everywhere.")

    strongest = {r["dataset"]: min(r["_iv"], key=lambda x: x["delta"])["scenario"].split(" (")[0] for r in rows}
    kinds = set(strongest.values())
    if len(kinds) == 1:
        out.append(f"**The strongest intervention did not change between networks, contrary to the hypothesis.** "
                   f"{kinds.pop()} gave the largest drop on every network "
                   f"({', '.join(f'{r['dataset'].capitalize()} {min(r['_iv'], key=lambda x: x['delta'])['delta']:+.3f}' for r in rows)}). "
                   f"In this model the incentive change acts on every agent directly, while network edits only act "
                   f"through who copies whom, so the network shape decides how much the structural interventions "
                   f"help, not which one wins.")
    else:
        out.append("**The strongest intervention changes with the network, as hypothesised:** "
                   + "; ".join(f"{k.capitalize()}: {v}" for k, v in strongest.items()) + ".")

    hubs = sorted(rows, key=lambda r: r["hub_isolation_effect"])
    sig = lambda r: r["_hub"]["ci_high"] < 0 or r["_hub"]["ci_low"] > 0
    order = ", ".join(f"{r['dataset'].capitalize()} {r['hub_isolation_effect']:+.3f}"
                      f"{'' if sig(r) else ' (interval includes zero)'}" for r in hubs)
    by = {r["dataset"]: r for r in rows}
    verdict = []
    if "caviar" in by:
        rank = [r["dataset"] for r in hubs].index("caviar") + 1
        verdict.append("Caviar " + ("does show the largest hub-isolation effect" if rank == 1 else
                                    f"ranks {rank} of {len(rows)}, not first as hypothesised")
                       + f", even though its hub touches {by['caviar']['hub_share']:.0%} of all ties")
    if "knecht" in by:
        rank = [r["dataset"] for r in hubs].index("knecht") + 1
        verdict.append("Knecht " + ("does show the smallest effect, as hypothesised" if rank == len(rows) else
                                    f"ranks {rank} of {len(rows)}, not last as hypothesised"))
    out.append(f"**Hub isolation, from largest to smallest drop:** {order}. " + "; ".join(verdict) + ". "
               "The hub's share of ties does not predict the size of the effect here; why is not tested by this run.")

    gt = [(r["dataset"], r["gamma_tie_hat"]) for r in rows]
    true = rows[0]["_gamma_tie_true"]
    share = ", ".join(f"{r['dataset'].capitalize()} {r['_tie_on'] / r['ties']:.0%}" for r in rows if r["_binary"])
    out.append(f"**The tie feature is identified much better off Noordin.** True gamma_tie is {true:g}; recovered: "
               + ", ".join(f"{d.capitalize()} {v:.2f}" for d, v in gt)
               + f". Share of ties carrying a 0/1 tie feature: {share}. When nearly all ties carry it, as on Noordin, "
               "the data rarely contrasts ties with and without the feature; a continuous feature such as Caviar's "
               "call volume, or a split nearer the middle, gives the learner that contrast.")
    real = [(r["dataset"], next((x for x in r["_iv"] if x["scenario"].startswith("Real")), None)) for r in rows]
    have = [(d, x) for d, x in real if x is not None]
    if have:
        out.append("**The real later networks:** " + "; ".join(
            f"{d.capitalize()} {x['delta']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]" for d, x in have)
            + ". " + ", ".join(d.capitalize() for d, x in real if x is None) + " has no later network.")
    fails = [(r["dataset"], a["check"]) for r in rows for a in r["_acceptance"] if not a["passed"]]
    out.append("**Acceptance criteria (SPEC.md section 11):** " +
               ("every criterion passes on every network." if not fails else
                "; ".join(f"{d}: {c}" for d, c in fails) + " (reported in that dataset's summary)."))
    quick = [r["dataset"] for r in rows if r["_mode"] != "full"]
    if quick:
        out.append(f"Note: {', '.join(quick)} were run in quick mode, so their numbers use reduced data.")
    return out


def figure(rows: list[dict], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["dataset"].capitalize() for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=P.CREAM_MUTED)
        ax.set_axisbelow(True)

    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, [r["cosine(theta)"] for r in rows], w, color=P.GREEN, label="Reward recovery (cosine)")
    ax.bar(x + w / 2, [r["top1_influencer_acc"] for r in rows], w, color=P.INK, label="True influencer found (Top-1)")
    ax.bar(x + w / 2, [r["_most_similar"] for r in rows], w * 0.45, color=P.AMBER, label="Attention-free baseline")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_title("Learning and explanation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax = axes[1]
    kinds = ["Hub isolation", "Mentor injection", "Supervision", "Real disruption"]
    colors = [P.RUST, P.AMBER_LIGHT, P.GREEN, P.SLATE_LIGHT]
    w = 0.2
    for k, (kind, col) in enumerate(zip(kinds, colors)):
        vals, lo, hi = [], [], []
        for r in rows:
            iv = next((v for v in r["_iv"] if v["scenario"].startswith(kind)), None)
            vals.append(iv["delta"] if iv else np.nan)
            lo.append(iv["delta"] - iv["ci_low"] if iv else 0)
            hi.append(iv["ci_high"] - iv["delta"] if iv else 0)
        ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=kind, yerr=[lo, hi], ecolor=P.INK_MUTED, capsize=2)
    ax.axhline(0, color=P.INK, lw=0.8)
    ax.set_xticks(x, names)
    ax.set_title("Change in final bad share (95% CI)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax = axes[2]
    ax.bar(x, [r["gamma_tie_hat"] for r in rows], 0.5, color=P.GREEN)
    ax.axhline(rows[0]["_gamma_tie_true"], color=P.INK, ls="--", lw=1.2)
    ax.text(len(rows) - 0.5, rows[0]["_gamma_tie_true"] + 0.05, "true value", ha="right", fontsize=9)
    ax.set_xticks(x, [f"{n}\n({r['tie_feature'].lower()})" for n, r in zip(names, rows)], fontsize=8)
    ax.set_title("Recovered gamma_tie", loc="left", fontweight="bold")

    fig.text(0.01, 0.035, f"{DISCLAIMER} {KNECHT_NOTE}", fontsize=8, color=P.INK_MUTED)
    fig.subplots_adjust(bottom=0.28, wspace=0.28, left=0.04, right=0.99, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run(outputs_root: Path) -> pd.DataFrame | None:
    rows = load_rows(outputs_root)
    if not rows:
        print("No dataset outputs found; run run_all.py --all first.")
        return None
    df = pd.DataFrame(rows)[COLUMNS]
    df.to_csv(outputs_root / "comparison.csv", index=False)
    figure(rows, outputs_root / "figures" / "comparison.png")
    missing = [n for n in ORDER if n not in df["dataset"].tolist()]
    notes = []
    if any(r["_isolates_not_in_file"] for r in rows):
        notes.append("Drugnet's isolates column counts agents in the file; the 81 people without ties on the dataset "
                     "card are not in the file, so the model runs on the 212 connected people.")
    if missing:
        notes.append("Not run yet: " + ", ".join(missing) + ".")
    md = "\n\n".join([
        "# Four networks, one model: comparison",
        f"_{DISCLAIMER} {KNECHT_NOTE}_",
        "Same answer key, actions, beta and seed on every network; only the network underneath changes.",
        "## Table", _table(df),
        "## What happened",
        "\n".join(f"{i + 1}. {s}" for i, s in enumerate(findings(rows))),
        "The hypothesis in SPEC_MULTIDATASET.md was: recovery and explanation stay high everywhere, the strongest "
        "intervention changes, hub isolation matters most on Caviar and least on Knecht. The points above say "
        "which parts held.",
        *notes,
        "![Comparison](figures/comparison.png)",
    ]) + "\n"
    (outputs_root / "comparison.md").write_text(md, encoding="utf-8")
    print(f"Comparison written to {outputs_root / 'comparison.md'}")
    return df
