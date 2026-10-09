"""Our loss variants next to every baseline: figure + tables for REPORT.md.

    python results/version_2/analysis/make_report.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from scripts.runner import Runner  # noqa: E402

OUT = Path("results/version_2/analysis")
OURS, OTHER, INK, MUTED, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
NAMES = {"metabric": "METABRIC", "support": "SUPPORT", "ebmt": "EBMT", "hsa_synthetic": "hsa synthetic",
         "deephit_synthetic": "DeepHit synthetic"}
BASE = {"survtrace": "SurvTRACE", "deephit": "DeepHit", "rsf": "RSF", "pchazard": "PC-Hazard",
        "deepsurv": "DeepSurv", "mensa": "MENSA", "cox": "Cox", "dsm": "DSM"}
MULTI = {"ebmt", "hsa_synthetic", "deephit_synthetic"}


def load(name):
    return Runner(repo=".", results_dir=f"results/version_2/{name}", verbose=False).collect()


bench = pd.concat([load("02_tabular_benchmark"), load("03_multievent_benchmark")])
l5, r6 = load("05_ablation_losses"), load("06_regression_head")

rows = []
for ds in NAMES:
    b = bench[(bench.dataset == ds) & (bench.model != "ours")]
    for m, lab in BASE.items():
        rows.append((ds, lab, "baseline", b[b.model == m]))
    for v, lab in [("S1", "Ours S1: L_PCH"), ("S2", "Ours S2: +L_rank"),
                   ("S3", "Ours S3: +L_mul"), ("S4", "Ours S4: +L_rank +L_mul")]:
        x = l5[(l5.dataset == ds) & (l5.variant == v)]
        if len(x):
            rows.append((ds, lab, "ours", x))
    joint = "joint_S4xR2" if ds in MULTI else "joint_S2xR2"
    rows.append((ds, "Ours full: survival (all losses) + regression head", "ours",
                 r6[(r6.dataset == ds) & (r6.variant == joint)]))

recs = []
for ds, lab, kind, x in rows:
    recs.append({"dataset": ds, "model": lab, "kind": kind,
                 **{f"{m}_{s}": getattr(x[m], s)() for m in ("ctd_weighted_avg", "ibs") for s in ("mean", "std")}})
tab = pd.DataFrame(recs)

# figure: C_td and IBS, one row of panels each
for metric, label, higher, fname in [("ctd_weighted_avg", "C_td (higher is better)", True, "report_ctd.png"),
                                     ("ibs", "Integrated Brier score (lower is better)", False, "report_ibs.png")]:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.4))
    for ax, ds in zip(axes, NAMES):
        t = tab[tab.dataset == ds].sort_values(f"{metric}_mean", ascending=higher)
        y = np.arange(len(t))
        for yi, (_, r) in zip(y, t.iterrows()):
            c = OURS if r.kind == "ours" else OTHER
            ax.errorbar(r[f"{metric}_mean"], yi, xerr=r[f"{metric}_std"], fmt="none", ecolor=c, elinewidth=2)
            ax.scatter(r[f"{metric}_mean"], yi, s=34, c=c, zorder=3, edgecolors="#fcfcfb", linewidths=1.5)
        ax.set_yticks(y, [m.split(":")[0] for m in t.model])
        ax.set_title(NAMES[ds], loc="left", color=INK, fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.supxlabel(f"{label}, mean ± sd over 10 splits", color=INK, fontsize=9)
    fig.suptitle("Our loss variants (blue) vs baselines (grey), best at the top", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)

# markdown tables
def md(metric, higher):
    cols = list(NAMES)
    order = tab[tab.dataset == "metabric"].sort_values(f"{metric}_mean", ascending=not higher).model.tolist()
    order += [m for m in tab.model.unique() if m not in order]
    best = {ds: (tab[tab.dataset == ds][f"{metric}_mean"].max() if higher
                 else tab[tab.dataset == ds][f"{metric}_mean"].min()) for ds in cols}
    lines = ["| Model | " + " | ".join(NAMES[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in order:
        cells = []
        for ds in cols:
            r = tab[(tab.dataset == ds) & (tab.model == m)]
            if r.empty or np.isnan(r[f"{metric}_mean"].iloc[0]):
                cells.append("—")
                continue
            v, s = r[f"{metric}_mean"].iloc[0], r[f"{metric}_std"].iloc[0]
            cell = f"{v:.3f} ({s:.3f})"
            cells.append(f"**{cell}**" if np.isclose(v, best[ds]) else cell)
        name = f"**{m}**" if m.startswith("Ours") else m
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


(OUT / "report_tables.md").write_text("### C_td\n\n" + md("ctd_weighted_avg", True) +
                                      "\n\n### IBS\n\n" + md("ibs", False) + "\n", encoding="utf-8")
print((OUT / "report_tables.md").read_text(encoding="utf-8"))
