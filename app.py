"""Noordin network demo: Streamlit app.

    streamlit run app.py

Loads precomputed results from outputs/ so it opens instantly. Run `python run_all.py`
first. Styled in the LOOP "Warm Minimalism" design language.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1  # noqa: E402,F401

from noordin import DISCLAIMER, ablation, interventions, learn  # noqa: E402
from noordin import plots as P  # noqa: E402
from noordin.config import THETA_NAMES, dataset_names, load_answer_key, load_public_config  # noqa: E402
from noordin.explain import explain_decision  # noqa: E402
from noordin.grade import cosine  # noqa: E402
from noordin.load_graph import load_graph, tie_identification_note  # noqa: E402
from noordin.simulator import simulate_dataset, true_policy  # noqa: E402

st.set_page_config(page_title="Noordin Network Demo", page_icon="🌱", layout="wide",
                   initial_sidebar_state="expanded")

DS_NAMES = {"noordin": "Noordin Top (terror cell)", "caviar": "Caviar (drug trafficking)",
            "drugnet": "Drugnet (drug users)", "knecht": "Knecht (school class)"}
DATASETS = dataset_names()
OUT_ROOT = Path(load_public_config()["outputs_root"])
RAN = [d for d in DATASETS if (OUT_ROOT / d / "metrics.json").exists()]
with st.sidebar:
    st.markdown("### Network")
    DATASET = st.selectbox(
        "Dataset", DATASETS, index=DATASETS.index("noordin"), key="dataset", label_visibility="collapsed",
        format_func=lambda d: DS_NAMES.get(d, d) + ("" if d in RAN else " (not run yet)"))
    st.markdown('<div class="caption">Same hidden reward, same learner, different real network. '
                'Every tab shows the network chosen here; the Compare tab shows all four.</div>',
                unsafe_allow_html=True)

CFG = load_public_config(dataset=DATASET)
OUT = Path(CFG["outputs_dir"])
KNECHT_NOTE = "Knecht is a class of children; no individual's real data is used."

# Plain-language description of each network. Everything numeric comes from the graph itself.
DS_INFO = {
    "noordin": dict(
        title="the real Jakarta 2009 / Noordin Top network", main="pre-attack network",
        source="Four relationship matrices over the same {n} people, symmetrised and reduced to present or absent.",
        codes="Cell values in the raw files are relationship codes from a codebook we do not have, so any "
              "non-zero value counts as a tie and nothing more.",
        tie_note="The dataset has no tie-strength values. Ties are coloured by how many layers record them, "
                 "which is a fact about the data rather than a guess about how strong a relationship was."),
    "caviar": dict(
        title="the real Caviar drug-trafficking network (police phone intercepts)", main="full call network",
        source="Intercepted phone calls among {n} people across 11 investigation phases, symmetrised; "
               "each tie is weighted by the number of calls.",
        codes="A tie is a pair who called each other at least once. The tie feature is the log of the call count, "
              "scaled to 0 to 1.",
        tie_note="Call volume is the one tie strength these datasets actually record."),
    "drugnet": dict(
        title="the real Drugnet acquaintance network", main="acquaintance network",
        source="\"A says they know B\" among {n} people who use drugs, symmetrised. The 81 people with no ties are not "
               "in the file and are left out.",
        codes="A tie is a nomination in either direction; it counts as mutual when both people named each other.",
        tie_note="The tie feature marks mutual acquaintance, the stronger kind of tie in this data."),
    "knecht": dict(
        title="the real Knecht friendship network (one school class)", main="wave 1 friendship network",
        source="Best-friend nominations among {n} students in one class, wave 1, symmetrised.",
        codes="Missing answers cannot be told apart from \"not a friend\". Knecht is a class of children; "
              "no individual's real data is used.",
        tie_note="The tie feature marks a mutual friendship, where both students named each other."),
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# ---------------------------------------------------------------- design system

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {{
  --background: {P.CREAM};
  --foreground: {P.INK};
  --card: {P.CARD};
  --primary: {P.GREEN};
  --primary-bright: {P.GREEN_BRIGHT};
  --secondary: #F0F6EA;
  --muted: {P.CREAM_MUTED};
  --muted-foreground: {P.INK_MUTED};
  --accent: {P.GREEN_SOFT};
  --accent-foreground: #14532D;
  --amber: {P.AMBER};
  --rust: {P.RUST};
  --border: {P.CREAM_DARK};
  --radius: 1rem;
  --shadow: 0px 10px 30px rgba(10, 25, 47, 0.06);
}}

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, textarea, select, button {{
  font-family: 'Inter', system-ui, sans-serif;
}}
.stApp {{ background: var(--background); color: var(--foreground); }}
h1, h2, h3, h4, .loop-heading {{
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  letter-spacing: -0.02em; color: var(--foreground);
}}
h2 {{ font-weight: 700 !important; }}
h3 {{ font-weight: 700 !important; font-size: 1.15rem !important; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
[data-testid="stHeader"] {{ background: transparent; }}

::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-thumb {{ background: #D6D0C4; border-radius: 3px; }}

/* Page header: tinted panel with a soft blob, as in LOOP's PageHeader */
.loop-header {{
  position: relative; overflow: hidden; border-radius: 1.5rem; border: 1px solid var(--border);
  padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
  background: linear-gradient(135deg, var(--tint) 0%, var(--card) 100%);
}}
.loop-header::after {{
  content: ""; position: absolute; right: -24px; top: -40px; width: 140px; height: 140px;
  border-radius: 50%; background: rgba(0, 110, 28, 0.06); filter: blur(24px);
}}
.loop-header .eyebrow {{
  display: inline-flex; align-items: center; gap: .45rem; font-size: .72rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: var(--muted-foreground); margin-bottom: .35rem;
}}
.loop-header h1 {{ font-size: 1.85rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.2; }}
.loop-header p {{ color: var(--muted-foreground); font-size: .95rem; margin: .4rem 0 0; max-width: 46rem; }}

.dot {{ display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; }}

/* Cards: white, 1rem radius, warm border, soft shadow */
div[class*="st-key-card"] {{
  background: var(--card); border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow); padding: 1rem 1.1rem !important;
}}
.kv {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: .35rem 1.1rem; }}
.kv div {{ display: flex; justify-content: space-between; gap: 1rem; padding: .5rem 0;
  border-bottom: 1px solid var(--muted); font-size: .87rem; }}
.kv div:last-child {{ border-bottom: none; }}
.kv span:first-child {{ color: var(--muted-foreground); }}
.kv span:last-child {{ font-weight: 600; text-align: right; }}
.loop-card {{
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 1.1rem 1.2rem; height: 100%;
}}

/* Stat tiles */
.stat {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 1rem 1.15rem; height: 100%; }}
.stat .label {{ font-size: .78rem; font-weight: 600; color: var(--muted-foreground);
  display: flex; align-items: center; gap: .4rem; }}
.stat .value {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.75rem;
  letter-spacing: -0.03em; color: var(--foreground); margin-top: .2rem; line-height: 1.15; }}
.stat .sub {{ font-size: .8rem; color: var(--muted-foreground); margin-top: .2rem; }}
.stat.good .value {{ color: var(--primary); }}
.stat.warn .value {{ color: var(--amber); }}
.stat.bad .value {{ color: var(--rust); }}

/* Step cards on the overview */
.step {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 1rem 1.1rem; height: 100%; }}
.step .num {{ width: 1.9rem; height: 1.9rem; border-radius: 50%; background: var(--accent);
  color: var(--accent-foreground); font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
  display: flex; align-items: center; justify-content: center; font-size: .9rem; margin-bottom: .55rem; }}
.step h4 {{ margin: 0 0 .25rem; font-size: 1rem; font-weight: 700; }}
.step p {{ margin: 0; font-size: .86rem; color: var(--muted-foreground); line-height: 1.45; }}

.caption {{ font-size: .84rem; color: var(--muted-foreground); margin: -.2rem 0 .4rem; }}
.note {{ background: var(--secondary); border: 1px solid #DCE8D3; border-radius: .85rem;
  padding: .8rem 1rem; font-size: .87rem; color: #1F3D24; }}
.note.amber {{ background: #FDF4E7; border-color: #F1DDBF; color: #6B3A08; }}
.chip {{ display: inline-flex; align-items: center; gap: .35rem; padding: .18rem .6rem; border-radius: 999px;
  background: var(--muted); font-size: .76rem; font-weight: 600; color: var(--muted-foreground); }}
.chip.green {{ background: var(--accent); color: var(--accent-foreground); }}

/* Tabs as a soft pill bar */
.stTabs [data-baseweb="tab-list"] {{
  gap: .25rem; background: var(--muted); padding: .3rem; border-radius: 999px; width: fit-content;
  max-width: 100%; overflow-x: auto; margin-bottom: .6rem;
}}
.stTabs [data-baseweb="tab"] {{
  height: 2.3rem; padding: 0 1rem; border-radius: 999px; background: transparent; border: none;
  font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; color: var(--muted-foreground);
  white-space: nowrap;
}}
.stTabs [data-baseweb="tab"] p {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; font-size: .9rem; }}
.stTabs [aria-selected="true"] {{ background: var(--card) !important; color: var(--primary) !important;
  box-shadow: 0 2px 8px rgba(10, 25, 47, 0.08); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* Buttons: rounded, green, with LOOP's squish */
.stButton > button {{
  border-radius: 999px; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700;
  padding: .45rem 1.15rem; transition: transform 150ms cubic-bezier(0.34, 1.56, 0.64, 1);
  border: 1px solid var(--border);
}}
.stButton > button:active {{ transform: scale(0.96); }}
.stButton > button[kind="primary"] {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #005a17; border-color: #005a17; }}

/* Inputs */
[data-baseweb="select"] > div, .stNumberInput input {{ border-radius: .75rem !important; background: var(--card) !important; }}
[data-testid="stSidebar"] {{ background: var(--card); border-right: 1px solid var(--border); }}
[data-testid="stDataFrame"] {{ border-radius: .75rem; overflow: hidden; border: 1px solid var(--border); }}
[data-testid="stExpander"] details {{ border-radius: var(--radius); border: 1px solid var(--border); background: var(--card); }}

.footer {{ margin-top: 2.5rem; padding: 1rem 1.2rem; border-top: 1px solid var(--border);
  color: var(--muted-foreground); font-size: .8rem; display: flex; gap: .6rem; align-items: center; }}

.report h1 {{ font-size: 1.4rem; }} .report h2 {{ font-size: 1.1rem; margin-top: 1.4rem; }}
.report table {{ font-size: .85rem; }}
</style>
"""

TINTS = {"green": P.GREEN_SOFT, "amber": "#FDF1E2", "rose": "#FBE9E1", "sky": "#F0F6EA"}


def header(title: str, subtitle: str, eyebrow: str = "", tint: str = "green", dot: str = P.GREEN):
    eb = f'<div class="eyebrow"><span class="dot" style="background:{dot}"></span>{eyebrow}</div>' if eyebrow else ""
    st.markdown(f'<div class="loop-header" style="--tint:{TINTS[tint]}">{eb}<h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "", tone: str = "", dot: str | None = None):
    d = f'<span class="dot" style="background:{dot}"></span>' if dot else ""
    st.markdown(f'<div class="stat {tone}"><div class="label">{d}{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def caption(text: str):
    st.markdown(f'<div class="caption">{text}</div>', unsafe_allow_html=True)


def note(text: str, tone: str = ""):
    st.markdown(f'<div class="note {tone}">{text}</div>', unsafe_allow_html=True)


def chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


_CARD_COUNT = [0]


def card():
    """White LOOP card. The key gives the container a class the stylesheet can target."""
    _CARD_COUNT[0] += 1
    return st.container(border=True, key=f"card_{_CARD_COUNT[0]}")


# ---------------------------------------------------------------- data

REQUIRED = ["metrics.json", "graph_edges.csv", "graph_stats.csv", "recovered.json", "recovery_table.csv",
            "data_efficiency.csv", "ablation.csv", "sensitivity.csv", "cascade_curves.csv", "cascade_gap.csv",
            "time_to_escalation.csv", "interventions.csv", "intervention_curves.csv", "summary.md"]


@st.cache_data(show_spinner=False)
def load_outputs(stamp: float, dataset: str) -> dict:
    o = {}
    for name in REQUIRED:
        p = OUT / name
        if name.endswith(".csv"):
            o[name[:-4]] = pd.read_csv(p)
        elif name.endswith(".json"):
            o[name[:-5]] = json.loads(p.read_text(encoding="utf-8"))
        else:
            o[name.split(".")[0]] = p.read_text(encoding="utf-8")
    tree = OUT / "surrogate_tree.txt"
    o["surrogate_tree"] = tree.read_text(encoding="utf-8") if tree.exists() else ""
    return o


@st.cache_resource(show_spinner=False)
def graph_and_layout(dataset: str):
    g = load_graph(CFG)
    edges = pd.read_csv(OUT / "graph_edges.csv")
    return g, edges, P.layout_positions(g.labels, edges, CFG["seed"])


@st.cache_resource(show_spinner="Loading held-out decisions")
def held_out(stamp: float, test_eps: tuple, dataset: str):
    eps = set(test_eps)
    dec = pd.read_csv(OUT / "decisions.csv")
    dec = dec[dec["episode"].isin(eps)].reset_index(drop=True)
    nb = pd.read_csv(OUT / "neighbours.csv")
    nb = nb[nb["episode"].isin(eps)]
    data = learn.prepare(dec, nb)
    infl = pd.read_csv(OUT / "influence_log.csv", keep_default_na=False)
    infl = infl[infl["episode"].isin(eps)]
    # plain dict so a lookup always yields one name, never a pandas Series
    infl = {(int(e), int(s), str(a)): str(p) for e, s, a, p in
            zip(infl["episode"], infl["step"], infl["agent"], infl["peer_id"])}
    return data, infl


# Two-way component: the explorer sends back the ties left after the user's edits.
_explorer = st.components.v1.declare_component("network_explorer", path=str(ROOT / "src" / "noordin" / "web" / "explorer"))


@st.cache_data(show_spinner=False)
def explorer_data(dataset: str) -> dict:
    g = load_graph(CFG)
    edges = pd.read_csv(OUT / "graph_edges.csv")
    # the browser components use two flags per tie: kin = carries the tie feature (above the median
    # when the feature is continuous), post = still present in the later network
    binary = edges["tie"].isin([0, 1]).all()
    strong = edges["tie"] > 0 if binary else edges["tie"] >= edges["tie"].median()
    js_edges = edges.assign(kin=strong.astype(int), post=edges["later"])
    main = CFG["ds"].get("main_label", "main").capitalize()
    tie = g.tie_label if binary else f"High {g.tie_label.lower()}"
    if dataset == "noordin":
        note = ("Ties are coloured by how many of the dataset\u2019s layers record them (pre-attack, kinship, "
                "post-attack). The raw cell values are relationship codes from a codebook we do not have, so they "
                "are not read as strength. The dataset records only before and after; the order ties disappear "
                "during playback is illustrative.")
    else:
        layers = ", ".join(x for x in (main.lower(), tie.lower(), (g.later_label or "").lower()) if x)
        note = (f"Ties are coloured by how many of these layers record them: {layers}. "
                + DS_INFO[dataset]["codes"]
                + (f" The data records the {main.lower()} and {g.later_label.lower()} networks only; the order ties "
                   "disappear during playback is illustrative." if g.later is not None else ""))
    return {"labels": g.labels, "degree": g.degree.astype(int).tolist(),
            "edges": js_edges.to_dict("records"), "disclaimer": DISCLAIMER,
            "text": {"main": main, "later": g.later_label, "tie": tie}, "note": note,
            "pos": {k: [round(v[0], 4), round(v[1], 4)] for k, v in P.layout_positions(g.labels, edges).items()}}


@st.cache_data(show_spinner=False)
def simulate_network(dataset: str, edges: tuple, removed: tuple, added: tuple = ()) -> dict:
    return interventions.simulate_edited(G, CFG, list(edges), removed, list(added))


_live_demo = st.components.v1.declare_component("live_demo", path=str(ROOT / "src" / "noordin" / "web" / "live"))


@st.cache_data(show_spinner=False)
def live_demo_data(dataset: str) -> dict:
    g = load_graph(CFG)
    a = CFG["actions"]
    params = {"beta": CFG["choice"]["beta"], "level": a["level"], "severity": a["severity"], "material": a["material"],
              "bad": a["bad"], "kappa": CFG["agents"]["kappa"], "steps": CFG["run"]["steps"],
              "supRange": CFG["agents"]["supervision_range"], "devRange": CFG["agents"]["init_deviance_range"],
              "scRange": CFG["agents"]["self_control_range"], "seedDev": CFG["agents"]["seed_deviance"],
              "thetaTrue": KEY["theta"], "gammaTrue": {"gamma_status": KEY["gamma_status"], "gamma_kin": KEY["gamma_tie"]}}
    return explorer_data(dataset) | {"status": [round(float(x), 4) for x in g.status], "params": params}


def outputs_stamp() -> float | None:
    if not all((OUT / n).exists() for n in REQUIRED):
        return None
    return max((OUT / n).stat().st_mtime for n in REQUIRED)


def footer():
    st.markdown(f'<div class="footer"><span class="dot" style="background:{P.AMBER}"></span>'
                f'{DISCLAIMER} {KNECHT_NOTE}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- app

st.markdown(CSS, unsafe_allow_html=True)

stamp = outputs_stamp()
if stamp is None:
    header("Noordin network demo", f"Precomputed results were not found in <code>outputs/{CFG['dataset']}/</code>.",
           eyebrow="Setup needed", tint="amber", dot=P.AMBER)
    note("Run the pipeline once from the project folder, then refresh this page:<br><br>"
         f"<code>python run_all.py --dataset {CFG['dataset']}</code> or add <code>--quick</code> for a fast run.",
         tone="amber")
    footer()
    st.stop()

O = load_outputs(stamp, DATASET)
M = O["metrics"]
G, EDGES, POS = graph_and_layout(DATASET)
ACTS = M["action_names"]
ACT_LABELS = [a.replace("_", " ") for a in ACTS]
KEY = load_answer_key()

INFO = DS_INFO[CFG["dataset"]]
N_AGENTS = G.n
N_TIES = int(np.triu(G.adj, 1).sum())
TIE_BINARY = bool(np.isin(G.tie[np.triu(G.adj, 1)], [0.0, 1.0]).all())
N_TIE_ON = int(np.triu((G.tie > 0) & G.adj, 1).sum())
HUB, SECOND = (G.labels[int(i)] for i in G.ranked()[:2])
LATER = G.later is not None
N_LATER = int(np.triu(G.later & G.adj, 1).sum()) if LATER else 0
N_LOST = N_TIES - N_LATER
LATER_NAME = G.later_label.lower() if LATER else ""

tabs = st.tabs(["Overview", "Live demo", "1 Graph", "2 Ablation", "3 Imitation learning", "4 Adoption", "5 Explainable AI",
                "6 Report", "Compare"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    header("Can we learn why a network misbehaves, just by watching it?",
           f"An explainable imitation-learning demo on {INFO['title']}. "
           "Agents act on a hidden reward we wrote down; a learner sees only their behaviour and "
           "tries to recover that reward, who each agent listens to, and why each choice was made.",
           eyebrow="PROJ2999 capstone · GITAM")
    rec, e = M["recovery"], M["explanations"]
    best = min(M["interventions"], key=lambda x: x["delta"])
    c = st.columns(4)
    with c[0]:
        stat("Agents and ties", f"{N_AGENTS} · {N_TIES}", f"real {INFO['main']}", dot=P.INK)
    with c[1]:
        stat("Reward recovered", f"{rec['cosine']:.4f}", "cosine to the hidden weights", "good", P.GREEN)
    with c[2]:
        stat("True influencer found", f"{e['recovered']['top1']:.0%}",
             f"vs {e['most_similar']['top1']:.0%} naive, {e['random']['top1']:.0%} random", "good", P.GREEN)
    with c[3]:
        stat("Best intervention", f"{best['delta']:+.2f}", best["scenario"].split(" (")[0], "warn", P.AMBER)

    st.markdown("### The six-step workflow")
    steps = [
        ("Datasets to graph", f"The raw files become one network of {N_AGENTS} agents and {N_TIES} ties"
                              + (f", {N_TIE_ON} of them with the {G.tie_label.lower()} feature." if TIE_BINARY
                                 else f", each weighted by {G.tie_label.lower()}.")),
        ("Ablation study", "Remove each tie in turn and measure how much bad behaviour changes."),
        ("Imitation learning", "Recover the hidden reward and attention from actions alone, and compare with plain cloning."),
        ("Behaviour adoption", "Roll each learned policy forward and see whether it reproduces the cascade."),
        ("Explainable AI", "Break any single decision into motives and name the neighbour who swayed it."),
        ("Report and insights", "Test interventions" + (f", including the real {LATER_NAME} network." if LATER else ".")),
    ]
    for start in (0, 3):
        cols = st.columns(3)
        for col, (k, (t, d)) in zip(cols, enumerate(steps[start:start + 3], start=start + 1)):
            with col:
                st.markdown(f'<div class="step"><div class="num">{k}</div><h4>{t}</h4><p>{d}</p></div>',
                            unsafe_allow_html=True)
        st.write("")

    l, r = st.columns([1.35, 1])
    with l:
        st.markdown("### How it works")
        st.markdown(
            "Each agent picks one of four actions (prosocial, neutral, minor deviance, escalated deviance) "
            "by weighing four concepts: **peer approval**, **material gain**, **status gain** and "
            "**sanction risk**. Peer approval is an attention-weighted vote of the neighbours, where "
            "attention favours high-status and kin ties. Agents choose with a softmax over this linear "
            "reward, so recovering the reward is maximum-likelihood estimation of a multinomial-logit "
            "(Luce) choice model: inverse reinforcement learning in its simplest exact form.")
        st.markdown(
            "The learner never reads the answer key. A separate grading step compares what it found "
            "against the truth the simulator logged.")
    with r:
        st.markdown("### Ground rules")
        note("<b>Network structure is real.</b> Ties come from the published Jakarta 2009 dataset. "
             "Agents are shown only as A01 to A28.")
        st.write("")
        note("<b>Everything else is simulated.</b> Traits, actions and rewards are invented for the "
             "experiment. There is no per-person risk score anywhere in this demo.", tone="amber")

# ---------------------------------------------------------------- graph

# ---------------------------------------------------------------- live demo
with tabs[1]:
    header("Watch it happen, one move at a time",
           "Press Next step. The computer first guesses what each agent will do, then every agent makes its move "
           "using the hidden recipe. The computer watches, learns, and its guesses improve as the episode goes on.",
           eyebrow="Live demo")
    if DATASET == "noordin":
        _live_demo(data=live_demo_data(DATASET), key="live_demo", default=None)
    else:
        note(f"The step-by-step live demo is Noordin-only for now: it follows each of 28 agents move by move, "
             f"and {N_AGENTS} agents are too many to follow that way on screen. Switch the network to Noordin Top "
             f"in the sidebar to use it. Every other tab works on {DS_NAMES[DATASET]}.", tone="amber")

with tabs[2]:
    header("Step 1 · Datasets to graph",
           INFO["source"].format(n=N_AGENTS) + f" The {INFO['main']} is the one the agents live on.",
           eyebrow="Real structure")
    sent = st.session_state.get(f"explorer_{DATASET}")
    result = None
    if sent and sent.get("edges") is not None:
        edges = tuple(sorted(tuple(sorted(e)) for e in sent["edges"]))
        added = tuple(sorted(tuple(sorted(e)) for e in (sent.get("added") or [])))
        result = simulate_network(DATASET, edges, tuple(sorted(sent.get("removed") or [])), added)
        st.session_state[f"explorer_last_{DATASET}"] = result
    payload = None if result is None else {k: v for k, v in result.items() if k != "curves"} | {"nonce": sent["nonce"]}
    _explorer(data=explorer_data(DATASET), result=payload, key=f"explorer_{DATASET}", default=None)
    caption((f"Press play to watch the {INFO['main']} lose the {N_LOST} ties missing from the {LATER_NAME} data. "
             if LATER else "") +
            "Cut ties, remove agents or switch to Add ties to connect people, then press Simulate this network.")
    last = st.session_state.get(f"explorer_last_{DATASET}")
    if last:
        sig = last["ci_high"] < 0 or last["ci_low"] > 0
        rel = last["relative"]
        pc = lambda v: f"{abs(v):.1%}" if abs(v) < 0.1 else f"{abs(v):.0%}"
        verdict = ("no real change" if not sig else
                   f"{pc(rel)} {'less' if rel < 0 else 'more'} bad behaviour")
        edits = ", ".join(x for x in [f'{len(last["removed"])} removed' if last["removed"] else "",
                                      f'{last["added"]} added' if last.get("added") else ""] if x)
        st.markdown("### What your edits did")
        c = st.columns(3)
        with c[0]:
            stat("Intact network", f'{last["baseline"]:.0%}', "of agents behaving badly", dot=P.INK)
        with c[1]:
            stat("Your network", f'{last["edited"]:.0%}', f'{last["ties"]} ties' + (f" · {edits}" if edits else ""), dot=P.AMBER)
        with c[2]:
            tone = "" if not sig else ("good" if rel < 0 else "bad")
            lo, hi = sorted([abs(last["relative_low"]), abs(last["relative_high"])])
            stat("Change", verdict, (f"between {pc(lo)} and {pc(hi)} across the runs" if sig
                                     else "the 30 runs disagree, so it could be luck"), tone)
        st.write("")
        if last.get("groups"):
            st.markdown("#### Who changed")
            gl, gr = st.columns([1.3, 1])
            with gl:
                with card():
                    chart(P.fig_groups(last["groups"]), key="explorer_groups")
                    caption("Each row is one kind of agent: grey is the intact network, colour is yours. Green means that "
                            "group behaved better on your network, rust worse, grey no change beyond chance.")
            with gr:
                note(last["explanation"], tone="" if sig else "amber")
                st.write("")
                rows = "".join(f"<div><span>{g['label']}: {g['description']}</span><span>{g['agents']}</span></div>"
                               for g in last["groups"])
                st.markdown(f'<div class="kv">{rows}</div>', unsafe_allow_html=True)
            st.write("")
        l, r = st.columns([1.15, 1])
        with l:
            with card():
                chart(P.fig_paired_runs(last["runs"]["intact"], last["runs"]["yours"]), key="explorer_runs")
                caption("Each row is one of the 30 runs: the same people and the same dice, played once on the intact "
                        "network (grey) and once on yours (colour). Green means your network ended with less bad behaviour in that run, rust means more.")
        with r:
            st.markdown("#### What happens in each of the 30 runs")
            st.markdown(
                f"1. **Make {N_AGENTS} people.** Each gets random traits: how closely they are watched, how much self-control "
                "they have, and how bad they start out. One person per group starts out very bad.\n"
                "2. **Play 50 turns.** Every turn each person looks at what their friends did last turn and picks "
                "good, neutral, a bit bad or very bad by rolling weighted dice. Bad friends make bad moves more likely; "
                "being watched makes them less likely.\n"
                "3. **Score the run.** Count the share of people choosing a bad move over the last 10 turns.\n"
                f"4. **Replay it on your network.** Same {N_AGENTS} people, same traits, same dice. Only the ties are different, "
                "so any difference in the score is caused by your edits.\n\n"
                "We repeat this with 30 different sets of people and dice, then average. One run can be lucky; "
                "30 runs agreeing is evidence.")
        with card():
            chart(P.fig_scenario_curves(last["curves"]), key="explorer_curves")
            caption(f"Average share of the {last['counted']} agents still in the network behaving badly at each of the 50 turns, "
                    f"with a 95% band, over the {last['seeds']} paired runs.")
    st.write("")
    l, r = st.columns([1.2, 1])
    with l:
        st.markdown("### What each layer records")
        tl = G.tie_label.lower()
        if TIE_BINARY:
            strong, on, off = EDGES["tie"] > 0, f"with {tl}", f"without {tl}"
        else:
            strong = EDGES["tie"] >= EDGES["tie"].median()
            on, off = f"high {tl}", f"low {tl}"
        kept = EDGES["later"] == 1
        combos = ([(f"{off.capitalize()}, gone by {LATER_NAME}", ~strong & ~kept), (f"{on.capitalize()}, gone by {LATER_NAME}", strong & ~kept),
                   (f"{off.capitalize()}, still there in {LATER_NAME}", ~strong & kept), (f"{on.capitalize()}, still there in {LATER_NAME}", strong & kept)]
                  if LATER else [(off.capitalize(), ~strong), (on.capitalize(), strong)])
        rows = "".join(f"<div><span>{name}</span><span>{int(mask.sum())}</span></div>" for name, mask in combos)
        st.markdown(f'<div class="kv">{rows}</div>', unsafe_allow_html=True)
        st.write("")
        note(INFO["tie_note"])
    with r:
        st.markdown("### Dataset checks")
        names = {"nodes": "Agents", "ties": "Ties", "mutual_pairs": "Mutual pairs", "group_sizes": "Group sizes",
                 "groups": "Connected groups", "isolates": "Agents with no ties", "max_degree": "Highest degree",
                 "hub_share": "Share of ties touching the hub", "tie_feature": "Tie feature",
                 "tie_feature_nonzero": "Ties with the feature", "tie_feature_mean": "Mean tie feature",
                 "later_network": "Later network", "later_ties": "Ties in the later network",
                 "later_isolated_agents": "Agents with no ties later",
                 "pre_attack_ties": "Pre-attack ties", "kinship_ties": "Kinship ties",
                 "kinship_ties_in_pre": "Kinship ties also pre-attack", "post_attack_ties": "Post-attack ties",
                 "post_ties_in_pre": "Post-attack ties also pre-attack", "pre_attack_group_sizes": "Connected groups",
                 "post_attack_isolated_agents": "Agents with no ties after", "highest_degree_agent": "Highest degree",
                 "second_highest_degree_agent": "Second highest degree", "average_degree": "Average degree"}
        rows = "".join(f"<div><span>{names.get(k, k)}</span><span>{v}</span></div>"
                       for k, v in O["graph_stats"].itertuples(index=False))
        st.markdown(f'<div class="kv">{rows}</div>', unsafe_allow_html=True)
        st.write("")
        note(INFO["codes"])

# ---------------------------------------------------------------- ablation
with tabs[3]:
    header("Step 2 · Ablation study",
           "Remove one tie, re-run the same seeds, and measure the change in the final share of agents "
           "choosing a bad action. Paired seeds mean the only difference is the missing tie.",
           eyebrow="Which relationships matter", tint="sky")
    abl = O["ablation"]
    key_of = {f"{a}-{b}": i for i, (a, b) in enumerate(zip(EDGES["agent_i"], EDGES["agent_j"]))}
    vals = np.zeros(len(EDGES))
    for a, b, d in abl[["agent_i", "agent_j", "mean_delta"]].itertuples(index=False):
        k = key_of.get(f"{a}-{b}", key_of.get(f"{b}-{a}"))
        vals[k] = d
    l, r = st.columns([1.6, 1])
    with l:
        with card():
            chart(P.fig_network(G.labels, EDGES, G.degree, POS, edge_values=vals, edge_title="delta"), key="net_abl")
            caption("Ties coloured by their ablation effect: rust ties spread bad behaviour (removing them lowers it), green ties held it back."
                    + (f" Only the {len(abl)} ties with the highest edge betweenness were ablated; the rest are drawn neutral."
                       if len(abl) < N_TIES else ""))
    with r:
        with card():
            chart(P.fig_sensitivity(pd.DataFrame(M["sensitivity"])), key="sens")
            caption("Sensitivity check: doubling the weight on sanction risk sharply lowers the final bad share.")
    with card():
        chart(P.fig_ablation_bars(abl, tie_label=G.tie_label), key="abl_bars")
        caption(f"The 15 ties with the largest effect, with 95% intervals over {CFG['run']['ablation_seeds']} paired seeds"
                + (f", out of the {len(abl)} ablated." if len(abl) < N_TIES else "."))
    st.markdown("### Try one tie live")
    with card():
        ties = [f"{a} to {b}" for a, b in zip(EDGES["agent_i"], EDGES["agent_j"])]
        c1, c2 = st.columns([2, 1])
        with c1:
            pick = st.selectbox("Tie to remove", ties, index=next((k for k, t in enumerate(ties) if {HUB, SECOND} == set(t.split(" to "))), 0))
        with c2:
            st.write("")
            go_live = st.button("Run this ablation (quick)", type="primary", width="stretch")
        if go_live:
            a, b = pick.split(" to ")
            qcfg = load_public_config(quick=True)
            t0 = time.perf_counter()
            res = ablation.ablate_tie(G, qcfg, G.label_index(a), G.label_index(b))
            took = time.perf_counter() - t0
            cc = st.columns(3)
            with cc[0]:
                stat("Baseline bad share", f"{res['baseline']:.3f}", f"{qcfg['run']['ablation_seeds']} seeds")
            with cc[1]:
                stat("Without this tie", f"{res['ablated']:.3f}", f"ran in {took:.1f} s")
            with cc[2]:
                tone = "good" if res["mean_delta"] < 0 else "bad"
                stat("Change", f"{res['mean_delta']:+.3f}", f"95% CI {res['ci_low']:+.3f} to {res['ci_high']:+.3f}", tone)
            st.write("")
            note("Quick mode uses only a few seeds, so its interval is wide. The table above uses the full run.")
    with st.expander("Full ablation table"):
        st.dataframe(abl[["agent_i", "agent_j", "tie", "mean_delta", "ci_low", "ci_high"]], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%+.4f") for c in ("mean_delta", "ci_low", "ci_high")})

# ---------------------------------------------------------------- imitation learning
with st.sidebar:
    st.markdown("### Answer key lab")
    st.markdown('<div class="caption">Change the hidden reward, re-simulate in quick mode, and see '
                'whether the learner finds your new weights. Used by tab 3.</div>', unsafe_allow_html=True)
    slider_theta = []
    for name, v in zip(THETA_NAMES, KEY["theta"]):
        slider_theta.append(st.slider(P.CONCEPT_LABELS[name], -1.0, 1.0, float(v), 0.05, key=f"th_{name}"))
    rerun = st.button("Re-run (quick)", type="primary", width="stretch")
    if rerun:
        qcfg = load_public_config(quick=True)
        t0 = time.perf_counter()
        with st.spinner("Simulating and refitting"):
            data = simulate_dataset(G, qcfg, policy=true_policy(qcfg, theta=slider_theta))
            d = learn.prepare(data["decisions"], data["neighbours"])
            train, _, _, _ = learn.split_by_episode(d, qcfg["run"]["test_fraction"])
            fit = learn.fit_reward(train, qcfg)
        st.session_state["live"] = {"theta": list(slider_theta), "fit": fit, "secs": time.perf_counter() - t0}
    if "live" in st.session_state:
        lv = st.session_state["live"]
        st.markdown(f'<span class="chip green">cosine {cosine(lv["fit"]["theta_hat"], lv["theta"]):.4f}</span> '
                    f'<span class="chip">{lv["secs"]:.1f} s</span>', unsafe_allow_html=True)

with tabs[4]:
    header("Step 3 · Imitation learning",
           "Two learners see the same recorded behaviour. Behavioural cloning copies actions from summary "
           "features; reward recovery fits the six parameters of the choice model by maximum likelihood.",
           eyebrow="Good vs bad behaviour", tint="green")
    rec, e = M["recovery"], M["recovered_eval"]
    c = st.columns(4)
    with c[0]:
        stat("Cosine to true theta", f"{rec['cosine']:.4f}", "all four signs correct" if rec["signs_correct"] else "a sign is wrong", "good", P.GREEN)
    with c[1]:
        stat("Largest weight error", f"{rec['max_abs_error']:.3f}", "tolerance 0.05", "good", P.GREEN)
    with c[2]:
        stat("Recovered reward accuracy", f"{e['accuracy']:.2f}", f"good/bad {e['good_bad_accuracy']:.2f}", dot=P.GREEN)
    with c[3]:
        stat("Cloning accuracy", f"{M['bc']['accuracy']:.2f}", f"majority class {M['bc']['majority_accuracy']:.2f}", dot=P.AMBER)
    st.write("")
    l, r = st.columns([1.5, 1])
    with l:
        with card():
            live = st.session_state.get("live")
            if live:
                tbl = pd.DataFrame({"parameter": THETA_NAMES, "true": live["theta"], "recovered": live["fit"]["theta_hat"]})
                st.markdown('<span class="chip green">Showing your live re-run</span>', unsafe_allow_html=True)
                chart(P.fig_theta(tbl), key="theta_live")
                caption(f"Your answer key vs what the learner recovered from {live['fit']['n_decisions']:,} quick-mode decisions.")
            else:
                chart(P.fig_theta(O["recovery_table"]), key="theta")
                caption(f"Hidden reward weights vs the weights recovered from {M['recovered']['n_decisions']:,} observed decisions.")
    with r:
        st.markdown("### Attention parameters")
        rt = O["recovery_table"]
        gam = rt[rt["parameter"].str.startswith("gamma")].copy()
        gam["parameter"] = gam["parameter"].map({"gamma_status": "Status (gamma_status)",
                                                 "gamma_tie": f"{G.tie_label} (gamma_tie)"})
        st.dataframe(gam, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ("true", "recovered", "error")})
        note(tie_identification_note(G) + " It is reported, not tested.", tone="amber")
        st.write("")
        note("Open the sidebar (top left arrow) to change the answer key and re-run in quick mode.")
    l, r = st.columns(2)
    with l:
        with card():
            chart(P.fig_data_efficiency(O["data_efficiency"]), key="eff")
            caption("How recovery quality grows with training data; one episode is already enough here.")
        st.dataframe(O["data_efficiency"][["episodes", "decisions", "cosine", "gamma_status_hat", "gamma_tie_hat"]],
                     hide_index=True, width="stretch",
                     column_config={"cosine": st.column_config.NumberColumn(format="%.4f"),
                                    "gamma_status_hat": st.column_config.NumberColumn(format="%.2f"),
                                    "gamma_tie_hat": st.column_config.NumberColumn(format="%.2f")})
    with r:
        with card():
            mode = st.segmented_control("Confusion matrix", ["Four actions", "Good vs bad"], default="Four actions",
                                        label_visibility="collapsed")
            if mode == "Good vs bad":
                chart(P.fig_confusion(M["bc"]["good_bad_confusion"], ["good", "bad"], height=300), key="cm2")
                caption("Behavioural cloning on held-out episodes, collapsed to good (actions 0 and 1) vs bad (2 and 3).")
            else:
                chart(P.fig_confusion(M["bc"]["confusion"], ACT_LABELS), key="cm4")
                caption("Behavioural cloning on held-out episodes: rows are true actions, shading is the row share.")
        note("Neither model can reach 100%: agents sample from a softmax, so the same situation can produce "
             "different actions. Accuracy is capped by that randomness.")

# ---------------------------------------------------------------- adoption
with tabs[5]:
    header("Step 4 · Criminal behaviour adoption",
           "Each policy is rolled forward on the same seeds. A good imitation should reproduce the whole "
           "cascade, not just single choices.", eyebrow="Rollouts", tint="amber", dot=P.AMBER)
    gaps = M["cascade_gap"]
    c = st.columns(3)
    with c[0]:
        stat("Recovered reward gap", f"{gaps.get('recovered', float('nan')):.3f}", "mean distance from true curve", "good", P.GREEN)
    with c[1]:
        stat("Behavioural cloning gap", f"{gaps.get('bc', float('nan')):.3f}", "mean distance from true curve", "warn", P.AMBER)
    with c[2]:
        ratio = gaps["bc"] / gaps["recovered"] if gaps.get("recovered") else float("nan")
        stat("Cloning is further off by", f"{ratio:.1f}x", "on the bad-share curve", dot=P.INK)
    st.write("")
    with card():
        chart(P.fig_cascades(O["cascade_curves"]), key="cascade")
        caption("Share of agents choosing a bad action at each step, mean and 95% band over 20 seeds, for each policy.")
    st.markdown("### Time to escalation")
    tte = O["time_to_escalation"].copy()
    tte["policy"] = tte["policy"].map(P.POLICY_NAMES)
    piv = tte.pivot(index="agent", columns="policy", values="mean_step").reset_index()
    share = tte.pivot(index="agent", columns="policy", values="escalated_share")
    piv["Escalated in (true)"] = share[P.POLICY_NAMES["true"]].values
    st.dataframe(piv, hide_index=True, width="stretch", height=360,
                 column_config={**{n: st.column_config.NumberColumn(n, format="%.1f") for n in P.POLICY_NAMES.values()},
                                "Escalated in (true)": st.column_config.ProgressColumn(
                                    "Share of seeds escalating (true)", min_value=0, max_value=1, format="percent")})
    caption("First step at which each simulated agent takes the escalated action, averaged over seeds where it happened. "
            "These are properties of the simulation, not of the real people.")

# ---------------------------------------------------------------- explainable AI
with tabs[6]:
    header("Step 5 · Explainable AI",
           "Any held-out decision can be broken into the four motives, using only recovered parameters, "
           "and traced to the neighbour whose example weighed most.", eyebrow="Why did they do that?", tint="sky")
    ex = M["explanations"]
    c = st.columns(4)
    with c[0]:
        stat("Top-1 match", f"{ex['recovered']['top1']:.1%}", f"{ex['recovered']['n']:,} held-out decisions", "good", P.GREEN)
    with c[1]:
        stat("Most similar baseline", f"{ex['most_similar']['top1']:.1%}", "attention-free", "warn", P.AMBER)
    with c[2]:
        stat("Random baseline", f"{ex['random']['top1']:.1%}", "expected value", dot=P.SLATE_LIGHT)
    with c[3]:
        ag = M["ablation_agreement"]
        stat("Ablation agreement", f"{ag['spearman']:+.2f}", f"Spearman over {ag['n_ties']} ties", dot=P.INK)
    st.write("")

    data, infl = held_out(stamp, tuple(M["test_episodes"]), DATASET)
    meta = data.meta
    with card():
        c1, c2, c3 = st.columns(3)
        with c1:
            ep = st.selectbox("Episode (held out)", sorted(meta["episode"].unique()))
        with c2:
            stp = st.slider("Step", 0, int(meta["step"].max()), 25)
        with c3:
            ag_pick = st.selectbox("Agent", G.labels, index=G.labels.index(HUB))
    row = np.where((meta["episode"] == ep) & (meta["step"] == stp) & (meta["agent"] == ag_pick))[0]
    if len(row):
        exp = explain_decision(data, int(row[0]), CFG, M["recovered"])
        truth = infl.get((int(ep), int(stp), str(ag_pick)), "")
        hit = bool(truth) and str(exp["predicted_influencer"]) == truth
        l, r = st.columns([1.5, 1])
        with l:
            with card():
                st.markdown(f"**{ag_pick}** chose **{exp['chosen'].replace('_', ' ')}** "
                            f'<span class="chip">supervision {exp["meta"]["supervision"]:.2f}</span> '
                            f'<span class="chip">deviance {exp["meta"]["deviance"]:.2f}</span>',
                            unsafe_allow_html=True)
                chart(P.fig_contributions(exp["concepts"]), key="contrib")
                caption("How much each motive added to or took from every option; the diamond is the total utility.")
        with r:
            c1, c2 = st.columns(2)
            with c1:
                stat("Predicted influencer", exp["predicted_influencer"] or "none", "from recovered attention",
                     "good" if hit else "bad", P.GREEN if hit else P.RUST)
            with c2:
                stat("True influencer", truth or "none", "from the simulator log", dot=P.INK)
            st.write("")
            nb = exp["neighbours"].copy()
            nb["last_action"] = nb["last_action"].str.replace("_", " ")
            nb["tie"] = nb["tie"].round(2)
            st.dataframe(nb, hide_index=True, width="stretch",
                         column_order=["neighbour", "last_action", "tie", "attention", "contribution"],
                         column_config={"neighbour": "Neighbour", "last_action": "Last action", "tie": G.tie_label,
                                        "attention": st.column_config.ProgressColumn("Attention", min_value=0, max_value=1, format="%.2f"),
                                        "contribution": st.column_config.NumberColumn("Contribution", format="%.3f")})
            caption("Neighbour contributions to peer approval: recovered attention times how close their last action was.")
    l, r = st.columns([1.2, 1])
    with l:
        with card():
            chart(P.fig_match_rates(ex), key="match")
            caption("How often each method names the true influencer, for held-out agents with two or more neighbours.")
    with r:
        st.markdown("### Surrogate rules")
        st.markdown(f'<span class="chip green">fidelity {M["surrogate_tree_fidelity"]:.0%}</span>', unsafe_allow_html=True)
        st.code(O["surrogate_tree"] or "Not generated.", language=None)
        caption("A depth-3 decision tree that imitates the recovered policy's most likely action, readable as if-then rules.")

# ---------------------------------------------------------------- report
with tabs[7]:
    header("Step 6 · Report and insights",
           f"{len(M['interventions'])} interventions, each run on the same seeds as the baseline"
           + (f", including the real {LATER_NAME} network as a natural experiment." if LATER else "."),
           eyebrow="What would help", tint="green")
    st.markdown("### Insights")
    for i, s in enumerate(M["insights"], 1):
        st.markdown(f'<div class="loop-card" style="margin-bottom:.6rem"><span class="chip green">{i}</span> '
                    f'<span style="margin-left:.4rem">{s}</span></div>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        with card():
            chart(P.fig_interventions(O["interventions"]), key="iv")
            caption("Change in final bad share for each intervention, with 95% intervals over paired seeds.")
    with r:
        with card():
            ivc = O["intervention_curves"]
            pre_post = ivc[ivc["scenario"].str.startswith(("Baseline", "Real"))]
            chart(P.fig_scenario_curves(pre_post), key="prepost")
            caption(f"Bad share over time on the real {INFO['main']}"
                    + (f" vs the {LATER_NAME} network" if LATER else "") + ", mean and 95% band.")
    note(f"Mentor injection fixes {SECOND} at prosocial, which on its own removes one agent in {N_AGENTS} "
         f"(about {1 / N_AGENTS:.3f}) from the bad share. The rest of its effect is spread through {SECOND}'s neighbours.",
         tone="amber")
    st.write("")
    with st.expander("Full summary report (summary.md)", expanded=False):
        st.markdown(f'<div class="report">', unsafe_allow_html=True)
        st.markdown(O["summary"])
        st.markdown("</div>", unsafe_allow_html=True)
    st.download_button("Download summary.md", O["summary"], file_name="summary.md", mime="text/markdown")

# ---------------------------------------------------------------- compare
with tabs[8]:
    header("Four networks, one model",
           "The same hidden reward, learner and seed on four real networks. Only the network underneath changes, "
           "so any difference below comes from network shape.", eyebrow="Compare", tint="sky")
    comp_csv, comp_md = OUT_ROOT / "comparison.csv", OUT_ROOT / "comparison.md"
    if not comp_csv.exists():
        note("No comparison yet. Run <code>python run_all.py --all</code> (or <code>--compare</code> if every "
             "dataset has already been run) and refresh.", tone="amber")
    else:
        cmp = pd.read_csv(comp_csv)
        cols = st.columns(len(cmp))
        for col, (_, row) in zip(cols, cmp.iterrows()):
            with col:
                stat(DS_NAMES.get(row["dataset"], row["dataset"]), f"{row['top1_influencer_acc']:.0%}",
                     f"influencer found · cosine {row['cosine(theta)']:.4f} · {row['nodes']} agents", "good", P.GREEN)
        st.write("")
        with card():
            st.image(str(OUT_ROOT / "figures" / "comparison.png"), width="stretch")
            caption("Left: learning and explanation quality. Middle: effect of each intervention with 95% intervals. "
                    "Right: the recovered tie-feature weight against its true value of 2.")
        st.markdown("### What happened")
        md = comp_md.read_text(encoding="utf-8")
        found = md.split("## What happened", 1)[1].split("![Comparison]", 1)[0] if "## What happened" in md else ""
        st.markdown(found)
        with st.expander("Full comparison table"):
            st.dataframe(cmp, hide_index=True, width="stretch")
        st.download_button("Download comparison.md", md, file_name="comparison.md", mime="text/markdown")

footer()
