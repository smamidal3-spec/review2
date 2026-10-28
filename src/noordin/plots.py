"""Charts in the LOOP "Warm Minimalism" design language.

Plotly builders are used by the app; `save_static_figures` writes matplotlib PNGs for
the report. Every chart carries the disclaimer.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import DISCLAIMER

# LOOP palette
CREAM = "#FAF7F2"
CREAM_MUTED = "#F1EDE4"
CREAM_DARK = "#E7E1D5"
CARD = "#FFFFFF"
INK = "#0F172A"
INK_MUTED = "#5C6657"
SLATE_LIGHT = "#94A3B8"
GREEN = "#006e1c"
GREEN_BRIGHT = "#4caf50"
GREEN_SOFT = "#E8F5E9"
AMBER = "#B45309"
AMBER_LIGHT = "#D97706"
RUST = "#C2410C"
RUST_SOFT = "#FBE3D6"

FONT_BODY = "Inter, system-ui, sans-serif"
FONT_HEAD = "Plus Jakarta Sans, Inter, system-ui, sans-serif"

POLICY_COLORS = {"true": INK, "recovered": GREEN, "bc": AMBER}
POLICY_NAMES = {"true": "True reward", "recovered": "Recovered reward", "bc": "Behavioural cloning"}
ACTION_COLORS = [GREEN, GREEN_BRIGHT, AMBER_LIGHT, RUST]
CONCEPT_LABELS = {"peer_approval": "Peer approval", "material_gain": "Material gain",
                  "status_gain": "Status gain", "sanction_risk": "Sanction risk"}


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family=FONT_BODY, color=INK, size=13),
        margin=dict(l=56, r=24, t=24, b=96),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor=INK, font=dict(family=FONT_BODY, color=CREAM, size=12), bordercolor=INK),
        colorway=[GREEN, INK, AMBER, RUST, SLATE_LIGHT],
    )
    fig.update_xaxes(automargin=True, showgrid=False, linecolor=CREAM_DARK, tickcolor=CREAM_DARK, ticks="outside",
                     title_font=dict(size=12, color=INK_MUTED), zeroline=False)
    fig.update_yaxes(automargin=True, gridcolor=CREAM_MUTED, linecolor=CREAM_DARK, title_font=dict(size=12, color=INK_MUTED),
                     zerolinecolor=CREAM_DARK)
    import textwrap
    wrapped = "<br>".join(textwrap.wrap(DISCLAIMER, 66))
    fig.add_annotation(text=wrapped, xref="paper", yref="paper", x=0, y=0, yshift=-90, xanchor="left",
                       yanchor="bottom", showarrow=False, align="left", font=dict(size=10, color=INK_MUTED))
    return fig


# ---------------------------------------------------------------- graph

def layout_positions(labels: list[str], edges: pd.DataFrame, seed: int = 42) -> dict[str, tuple[float, float]]:
    g = nx.Graph()
    g.add_nodes_from(labels)
    g.add_edges_from(edges[["agent_i", "agent_j"]].itertuples(index=False, name=None))
    pos = nx.kamada_kawai_layout(g) if nx.is_connected(g) else _component_layout(g, seed)
    return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}


def _component_layout(g: nx.Graph, seed: int):
    """Largest group on the left; the other groups stacked in a column on its right;
    agents with no ties in a row underneath."""
    comps = sorted(nx.connected_components(g), key=lambda c: (-len(c), sorted(c)[0]))
    big = [c for c in comps if len(c) > 1]
    single = sorted(next(iter(c)) for c in comps if len(c) == 1)

    def place(comp, radius):
        p = nx.kamada_kawai_layout(g.subgraph(comp))
        arr = np.array(list(p.values()))
        arr = (arr - arr.mean(0)) / (np.abs(arr - arr.mean(0)).max() or 1) * radius
        return dict(zip(p.keys(), arr))

    pos = {}
    if big:
        r0 = 0.85 * np.sqrt(len(big[0]))
        for k, v in place(big[0], r0).items():
            pos[k] = (v[0], v[1])
        rest = big[1:]
        y = r0
        for comp in rest:
            r = 0.55 * np.sqrt(len(comp))
            for k, v in place(comp, r).items():
                pos[k] = (v[0] + r0 + 0.9 + r, v[1] + y - r)
            y -= 2 * r + 0.5
    ymin = min((v[1] for v in pos.values()), default=0.0) - 0.9
    xmin = min((v[0] for v in pos.values()), default=0.0)
    for i, label in enumerate(single):
        pos[label] = (xmin + 0.6 * (i % 12), ymin - 0.6 * (i // 12))
    return pos


def fig_network(labels, edges: pd.DataFrame, degree, pos, edge_values=None, edge_title="",
                highlight=None, height=520, tie_label: str = "Kinship") -> go.Figure:
    """Nodes sized by degree. Ties coloured by the tie feature, or by `edge_values` (a diverging scale)."""
    fig = go.Figure()
    deg = dict(zip(labels, degree))
    big = len(labels) > 40              # large networks: small nodes, thin ties, labels on hover only
    thin = 0.45 if big else 1.0
    if edge_values is None:
        strong = edges["tie"] >= 0.5
        groups = [(f"{tie_label} (strong)" if not edges["tie"].isin([0, 1]).all() else f"{tie_label} tie",
                   edges[strong], GREEN, 2.4),
                  ("Other tie", edges[~strong], SLATE_LIGHT, 1.6)]
        for name, sub, color, width in groups:
            xs, ys = [], []
            for a, b in sub[["agent_i", "agent_j"]].itertuples(index=False):
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width * thin),
                                     name=name, hoverinfo="skip"))
    else:
        vmax = float(np.max(np.abs(edge_values))) or 1.0
        for (a, b), v in zip(edges[["agent_i", "agent_j"]].itertuples(index=False), edge_values):
            t = v / vmax
            color = rgba(RUST, 0.25 + 0.75 * abs(t)) if t < 0 else rgba(GREEN, 0.25 + 0.75 * abs(t))
            fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                     line=dict(color=color, width=(1.5 + 5 * abs(t)) * thin), showlegend=False,
                                     hoverinfo="text", text=f"{a} to {b}<br>{edge_title}: {v:+.3f}"))
        # midpoints carry the hover so thin lines are still easy to inspect
        mx = [(pos[a][0] + pos[b][0]) / 2 for a, b in edges[["agent_i", "agent_j"]].itertuples(index=False)]
        my = [(pos[a][1] + pos[b][1]) / 2 for a, b in edges[["agent_i", "agent_j"]].itertuples(index=False)]
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="markers", showlegend=False,
            marker=dict(size=10, color=list(edge_values), colorscale=[[0, RUST], [0.5, CREAM_DARK], [1, GREEN]],
                        cmin=-vmax, cmax=vmax, opacity=0.0,
                        colorbar=dict(title=dict(text=edge_title, font=dict(size=11)), thickness=10, len=0.6,
                                      outlinewidth=0)),
            hovertext=[f"{a} to {b}<br>{edge_title}: {v:+.3f}" for (a, b), v in
                       zip(edges[["agent_i", "agent_j"]].itertuples(index=False), edge_values)],
            hoverinfo="text"))
    hl = set(highlight or [])
    xs = [pos[l][0] for l in labels]
    ys = [pos[l][1] for l in labels]
    sizes = [5 + 2.2 * np.sqrt(deg[l]) if big else 12 + 3.2 * deg[l] for l in labels]
    fill = [AMBER if l in hl else (INK if deg[l] > 0 else CARD) for l in labels]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers" if big else "markers+text", text=labels, textposition="top center",
        textfont=dict(size=11, color=INK, family=FONT_HEAD),
        marker=dict(size=sizes, color=fill, line=dict(color=CARD if big else INK, width=0.6 if big else 1.2)),
        hovertext=[f"<b>{l}</b><br>degree {deg[l]}" for l in labels], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return style(fig, height=height)


# ---------------------------------------------------------------- imitation learning

def fig_theta(recovery: pd.DataFrame, extra: dict | None = None) -> go.Figure:
    """True vs recovered theta. `extra` adds another named series, e.g. a live re-run."""
    th = recovery[recovery["parameter"].isin(CONCEPT_LABELS)]
    x = [CONCEPT_LABELS[p] for p in th["parameter"]]
    fig = go.Figure()
    fig.add_bar(x=x, y=th["true"], name="True (answer key)", marker_color=INK, marker_line_width=0)
    fig.add_bar(x=x, y=th["recovered"], name="Recovered from behaviour", marker_color=GREEN, marker_line_width=0)
    for name, vals in (extra or {}).items():
        fig.add_bar(x=x, y=vals, name=name, marker_color=AMBER, marker_line_width=0)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title="weight", zeroline=True, zerolinewidth=1.5)
    return style(fig, height=380)


def fig_data_efficiency(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df["episodes"], y=df["cosine"], mode="lines+markers", name="Cosine to true theta",
                    line=dict(color=GREEN, width=2.5), marker=dict(size=9, color=GREEN))
    fig.add_hline(y=0.90, line=dict(color=AMBER, dash="dot", width=1.5),
                  annotation_text="0.90 one-episode target", annotation_font=dict(color=AMBER, size=11))
    fig.update_xaxes(type="log", title="training episodes (log scale)", tickvals=list(df["episodes"]),
                     ticktext=[str(int(v)) for v in df["episodes"]], minor=dict(showgrid=False, ticks=""))
    fig.update_yaxes(title="cosine similarity", range=[min(0.85, float(df["cosine"].min()) - 0.02), 1.005])
    return style(fig, height=360, legend=False)


def fig_confusion(matrix, labels, title_x="predicted", title_y="actual", height=340) -> go.Figure:
    m = np.asarray(matrix, float)
    labels = [str(l).replace(" ", "<br>") for l in labels]
    row = m.sum(1, keepdims=True)
    pct = np.divide(m, row, out=np.zeros_like(m), where=row > 0)
    text = [[f"{int(v):,}<br><span style='font-size:10px'>{p:.0%}</span>" for v, p in zip(r, pr)]
            for r, pr in zip(m, pct)]
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, text=text, texttemplate="%{text}",
                               colorscale=[[0, CARD], [0.5, GREEN_SOFT], [1, GREEN]], zmin=0, zmax=1,
                               showscale=False, hovertemplate="actual %{y}<br>predicted %{x}<br>%{z:.1%}<extra></extra>",
                               xgap=3, ygap=3))
    fig.update_yaxes(autorange="reversed", title=title_y, showgrid=False)
    fig.update_xaxes(title=title_x, side="bottom", tickangle=0)
    return style(fig, height=height + 30, legend=False)


# ---------------------------------------------------------------- ablation and sensitivity

def fig_ablation_bars(abl: pd.DataFrame, top: int = 15, tie_label: str = "Kinship") -> go.Figure:
    d = abl.head(top).iloc[::-1]
    tag = tie_label.lower()
    y = [f"{a} to {b}" + (f" ({tag} {k:.2f})" if 0 < k < 1 else f" ({tag})" if k >= 1 else "")
         for a, b, k in d[["agent_i", "agent_j", "tie"]].itertuples(index=False)]
    colors = [RUST if v < 0 else GREEN for v in d["mean_delta"]]
    fig = go.Figure(go.Bar(
        x=d["mean_delta"], y=y, orientation="h", marker_color=colors, marker_line_width=0,
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["mean_delta"],
                     arrayminus=d["mean_delta"] - d["ci_low"], color=INK_MUTED, thickness=1.2, width=3),
        hovertemplate="%{y}<br>delta %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(title="change in final bad share when the tie is removed")
    return style(fig, height=max(320, 26 * len(d) + 120), legend=False)


def fig_sensitivity(sens: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=sens["setting"], y=sens["final_bad_share"], marker_color=[RUST, GREEN][: len(sens)], marker_line_width=0,
        error_y=dict(type="data", symmetric=False, array=sens["ci_high"] - sens["final_bad_share"],
                     arrayminus=sens["final_bad_share"] - sens["ci_low"], color=INK, thickness=1.2, width=6),
        text=[f"{v:.2f}" for v in sens["final_bad_share"]], textposition="inside", insidetextanchor="end",
        textfont=dict(family=FONT_HEAD, size=15, color=CARD)))
    fig.update_yaxes(title="final bad share", range=[0, 1])
    fig.update_layout(bargap=0.55)
    return style(fig, height=340, legend=False)


# ---------------------------------------------------------------- cascades

def _band(fig, steps, mean, low, high, color, name, dash=None):
    fig.add_scatter(x=np.r_[steps, steps[::-1]], y=np.r_[high, low[::-1]], fill="toself",
                    fillcolor=rgba(color, 0.12), line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=steps, y=mean, mode="lines", name=name, line=dict(color=color, width=2.6, dash=dash),
                    hovertemplate=name + "<br>step %{x}<br>bad share %{y:.2f}<extra></extra>")


def fig_cascades(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pol in ("true", "recovered", "bc"):
        d = curves[curves["policy"] == pol]
        if len(d):
            _band(fig, d["step"].to_numpy(), d["mean"].to_numpy(), d["low"].to_numpy(), d["high"].to_numpy(),
                  POLICY_COLORS[pol], POLICY_NAMES[pol], dash="dot" if pol == "true" else None)
    fig.update_xaxes(title="step")
    fig.update_yaxes(title="share of agents choosing a bad action", range=[0, 1])
    return style(fig, height=420)


def fig_scenario_curves(curves: pd.DataFrame, names=None) -> go.Figure:
    palette = [INK, RUST, GREEN, AMBER, SLATE_LIGHT, GREEN_BRIGHT]
    fig = go.Figure()
    for k, (name, d) in enumerate(curves.groupby("scenario", sort=False)):
        if names and name not in names:
            continue
        _band(fig, d["step"].to_numpy(), d["mean"].to_numpy(), d["low"].to_numpy(), d["high"].to_numpy(),
              palette[k % len(palette)], name, dash="dot" if name.startswith("Baseline") else None)
    fig.update_xaxes(title="step")
    fig.update_yaxes(title="share of agents choosing a bad action", range=[0, 1])
    return style(fig, height=420)


# ---------------------------------------------------------------- explanations

def fig_contributions(concepts: pd.DataFrame) -> go.Figure:
    """Stacked concept contributions (beta * theta_hat * phi) for every candidate action."""
    fig = go.Figure()
    colors = {"peer_approval": GREEN, "material_gain": AMBER_LIGHT, "status_gain": AMBER, "sanction_risk": RUST}
    names = [n.replace("_", " ") + (" (chosen)" if c else "") for n, c in zip(concepts["action"], concepts["chosen"])]
    for k, label in CONCEPT_LABELS.items():
        fig.add_bar(y=names, x=concepts[k], name=label, orientation="h", marker_color=colors[k], marker_line_width=0,
                    hovertemplate=label + ": %{x:+.2f}<extra></extra>")
    fig.add_scatter(y=names, x=concepts["utility"], mode="markers", name="Total utility",
                    marker=dict(symbol="diamond", size=12, color=INK, line=dict(color=CARD, width=1.5)),
                    hovertemplate="utility %{x:+.2f}<extra></extra>")
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_layout(barmode="relative", bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="contribution to utility (beta x theta_hat x phi)")
    return style(fig, height=340)


def fig_match_rates(g: dict) -> go.Figure:
    rows = [("Random neighbour", g["random"], SLATE_LIGHT), ("Most similar neighbour", g["most_similar"], AMBER),
            ("Recovered attention", g["recovered"], GREEN)]
    fig = go.Figure()
    for metric, opacity in (("top1", 1.0), ("top3", 0.45)):
        fig.add_bar(x=[r[0] for r in rows], y=[r[1][metric] for r in rows], name=metric.replace("top", "Top-"),
                    marker_color=[rgba(r[2], opacity) for r in rows], marker_line_width=0,
                    text=[f"{r[1][metric]:.2f}" for r in rows], textposition="outside",
                    textfont=dict(family=FONT_HEAD, size=13, color=INK))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(title="match rate with the true influencer", range=[0, 1.12])
    return style(fig, height=360)


# ---------------------------------------------------------------- interventions

def fig_interventions(table: pd.DataFrame) -> go.Figure:
    d = table.iloc[::-1]
    colors = [GREEN if v < 0 else RUST for v in d["delta"]]
    y = [f"{s}  <b>{v:+.3f}</b>" for s, v in zip(d["scenario"], d["delta"])]
    fig = go.Figure(go.Bar(
        x=d["delta"], y=y, orientation="h", marker_color=colors, marker_line_width=0,
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["delta"],
                     arrayminus=d["delta"] - d["ci_low"], color=INK, thickness=1.2, width=4),
        hovertemplate="%{y}<br>95% CI shown as whiskers<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    lo = min(0.0, float(table["ci_low"].min())) * 1.15
    hi = max(0.0, float(table["ci_high"].max())) * 1.15
    fig.update_xaxes(title="change in final bad share vs baseline (95% CI)", range=[lo - 0.01, hi + 0.01])
    fig.update_layout(bargap=0.45)
    return style(fig, height=340, legend=False)


# ---------------------------------------------------------------- static PNGs for the report

def save_static_figures(out: Path, recovery: pd.DataFrame, efficiency: pd.DataFrame,
                        curves: pd.DataFrame, interventions: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": CREAM_DARK, "axes.labelcolor": INK_MUTED,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": CARD,
                         "axes.facecolor": CARD, "axes.grid": True, "grid.color": CREAM_MUTED, "axes.axisbelow": True})
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paths = []

    def finish(fig, name, caption):
        fig.text(0.01, 0.055, caption, fontsize=9, color=INK)
        fig.text(0.01, 0.015, DISCLAIMER, fontsize=7.5, color=INK_MUTED)
        fig.subplots_adjust(bottom=0.24)
        p = figdir / name
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    th = recovery[recovery["parameter"].isin(CONCEPT_LABELS)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(th))
    ax.bar(x - 0.18, th["true"], 0.36, color=INK, label="True")
    ax.bar(x + 0.18, th["recovered"], 0.36, color=GREEN, label="Recovered")
    ax.set_xticks(x, [CONCEPT_LABELS[p] for p in th["parameter"]])
    ax.axhline(0, color=INK, lw=0.8)
    ax.legend(frameon=False)
    finish(fig, "theta_recovery.png", "True vs recovered reward weights.")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(efficiency["episodes"], efficiency["cosine"], "o-", color=GREEN, lw=2)
    ax.axhline(0.9, color=AMBER, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("cosine to true theta")
    finish(fig, "data_efficiency.png", "Reward recovery quality as training data grows.")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for pol in ("true", "recovered", "bc"):
        d = curves[curves["policy"] == pol]
        ax.fill_between(d["step"], d["low"], d["high"], color=POLICY_COLORS[pol], alpha=0.12, lw=0)
        ax.plot(d["step"], d["mean"], color=POLICY_COLORS[pol], lw=2, ls=":" if pol == "true" else "-",
                label=POLICY_NAMES[pol])
    ax.set_ylim(0, 1)
    ax.set_xlabel("step")
    ax.set_ylabel("bad share")
    ax.legend(frameon=False)
    finish(fig, "cascades.png", "Bad-behaviour cascades under the true, recovered and cloned policies.")

    d = interventions.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(d["scenario"], d["delta"], color=[GREEN if v < 0 else RUST for v in d["delta"]],
            xerr=[d["delta"] - d["ci_low"], d["ci_high"] - d["delta"]], ecolor=INK, capsize=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("change in final bad share")
    fig.subplots_adjust(left=0.36)
    finish(fig, "interventions.png", "Intervention effects with 95% confidence intervals.")
    return paths


def fig_paired_runs(intact, yours) -> go.Figure:
    """One row per simulated run: intact network vs the user's network, same people and dice."""
    pairs = sorted(zip(intact, yours), key=lambda p: p[0])
    fig = go.Figure()
    for k, (a, b) in enumerate(pairs, start=1):
        col = GREEN if b < a else RUST if b > a else SLATE_LIGHT
        fig.add_scatter(x=[a, b], y=[k, k], mode="lines", line=dict(color=rgba(col, 0.55), width=2.5),
                        showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[a for a, _ in pairs], y=list(range(1, len(pairs) + 1)), mode="markers", name="Intact network",
                    marker=dict(size=9, color=SLATE_LIGHT, line=dict(color=CARD, width=1)),
                    hovertemplate="run %{y}<br>intact: %{x:.0%}<extra></extra>")
    fig.add_scatter(x=[b for _, b in pairs], y=list(range(1, len(pairs) + 1)), mode="markers", name="Your network",
                    marker=dict(size=9, color=[GREEN if b < a else RUST if b > a else SLATE_LIGHT for a, b in pairs],
                                line=dict(color=CARD, width=1)),
                    hovertemplate="run %{y}<br>yours: %{x:.0%}<extra></extra>")
    fig.update_xaxes(title="share of agents behaving badly (last 10 turns)", tickformat=".0%")
    fig.update_yaxes(title="run (sorted)", showticklabels=False, showgrid=False)
    return style(fig, height=470)


def fig_groups(groups: list[dict]) -> go.Figure:
    """Bad share of each kind of agent on the intact network (grey) and on the edited one (colour)."""
    fig = go.Figure()
    names = [f"{g['label']} ({g['agents']})" for g in groups][::-1]
    gs = groups[::-1]
    for name, g in zip(names, gs):
        sig = g["ci_high"] < 0 or g["ci_low"] > 0
        col = SLATE_LIGHT if not sig else (GREEN if g["change"] < 0 else RUST)
        fig.add_scatter(x=[g["intact"], g["yours"]], y=[name, name], mode="lines",
                        line=dict(color=rgba(col, 0.6), width=5), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[g["intact"] for g in gs], y=names, mode="markers", name="Intact network",
                    marker=dict(size=13, color=SLATE_LIGHT, line=dict(color=CARD, width=1.5)),
                    hovertemplate="%{y}<br>intact: %{x:.0%}<extra></extra>")
    fig.add_scatter(x=[g["yours"] for g in gs], y=names, mode="markers+text", name="Your network",
                    marker=dict(size=13, color=[SLATE_LIGHT if not (g["ci_high"] < 0 or g["ci_low"] > 0)
                                                else GREEN if g["change"] < 0 else RUST for g in gs],
                                line=dict(color=CARD, width=1.5)),
                    text=[f"{g['change'] * 100:+.0f} pts" for g in gs], textposition="middle right",
                    textfont=dict(family=FONT_HEAD, size=12, color=INK),
                    hovertemplate="%{y}<br>yours: %{x:.0%}<extra></extra>")
    lo = min(min(g["intact"], g["yours"]) for g in groups) - 0.05
    hi = max(max(g["intact"], g["yours"]) for g in groups) + 0.1
    fig.update_xaxes(title="share behaving badly (last 10 turns)", tickformat=".0%", range=[max(0, lo), min(1, hi)])
    return style(fig, height=110 + 60 * len(groups))
