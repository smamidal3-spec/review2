"""D1: Noordin Top, Jakarta 2009. Four 28x28 matrices over the same people.

Agents keep the CSV column order (A11 is original ID 177), so every Noordin result is
unchanged by the multi-dataset refactor. Tie feature: kinship (0/1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..load_graph import Graph, components, degree_status, make_labels

FILES = {
    "kinship": "JAKARTA_NORDIN_KINSHIP.csv",
    "pre": "JAKARTA_NORDIN_PRE-ATTACK.csv",
    "post": "JAKARTA_NORDIN_POST-ATTACK.csv",
    "operation": "JAKARTA_NORDIN_OPERATION_ID.csv",
}


def _read_matrix(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    ids = [str(c).strip() for c in df.columns]
    rows = [str(r).strip() for r in df.index]
    if ids != rows:
        raise ValueError(f"{path.name}: row and column IDs differ")
    m = df.values
    # Cell values are relationship codes from an unread codebook: present/absent only.
    directed = m > 0
    np.fill_diagonal(directed, False)
    return ids, directed, directed | directed.T


def files_present(cfg: dict) -> bool:
    return all((Path(cfg["data_dir"]) / f).exists() for f in FILES.values())


def load(cfg: dict) -> Graph:
    data_dir = Path(cfg["data_dir"])
    ds = cfg["ds"]
    mats, ids, pre_directed = {}, None, None
    for key, name in FILES.items():
        these, directed, m = _read_matrix(data_dir / name)
        if ids is None:
            ids = these
        elif these != ids:
            raise ValueError(f"{name}: IDs are not in the same order as the other files")
        mats[key] = m
        if key == "pre":
            pre_directed = directed
    pre = mats["pre"]
    kin = mats["kinship"] & pre if ds.get("kinship_counts_only_if_in_pre", True) else mats["kinship"]
    post = mats["post"]
    sizes = [len(c) for c in components(pre) if len(c) > 1]
    extra = {
        "pre_attack_ties": int(np.triu(pre, 1).sum()),
        "kinship_ties": int(np.triu(mats["kinship"], 1).sum()),
        "kinship_ties_in_pre": int(np.triu(kin, 1).sum()),
        "post_attack_ties": int(np.triu(post, 1).sum()),
        "post_ties_in_pre": int(np.triu(post & pre, 1).sum()),
        "pre_attack_group_sizes": ", ".join(map(str, sizes)),
        "post_attack_isolated_agents": int((post.sum(1) == 0).sum()),
    }
    return Graph(
        name="noordin",
        labels=make_labels(len(ids)),
        adj=pre,
        tie=kin.astype(float),
        tie_label=ds.get("tie_label", "Kinship"),
        directed=pre_directed,
        later=post,
        later_label=ds.get("later_label", "Post-attack"),
        status=degree_status(pre),
        extra=extra,
        _original_ids=ids,
        _layers={"kin_raw": mats["kinship"], "operation": mats["operation"]},
    )
