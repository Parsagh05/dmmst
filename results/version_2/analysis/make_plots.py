"""Version-2 figures from notebooks 02-06 (run from the repo root):

    python results/version_2/analysis/make_plots.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from scripts.runner import Runner, paired_vs  # noqa: E402

OUT = Path("results/version_2/analysis")
OURS, OTHER, ACCENT, INK, MUTED, GRID = "#2a78d6", "#a3a29c", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
NAMES = {"metabric": "METABRIC", "support": "SUPPORT", "ebmt": "EBMT", "hsa_synthetic": "hsa synthetic",
         "deephit_synthetic": "DeepHit synthetic"}
MODEL = {"ours": "Ours", "survtrace": "SurvTRACE", "cox": "Cox", "rsf": "RSF", "deepsurv": "DeepSurv",
         "pchazard": "PC-Hazard", "deephit": "DeepHit", "dsm": "DSM", "mensa": "MENSA"}


def load(name):
    return Runner(repo=".", results_dir=f"results/version_2/{name}", verbose=False).collect()


def dots(ax, labels, mean, sd, colors, xlabel):
    y = np.arange(len(labels))
    for yi, m, s, c in zip(y, mean, sd, colors):
        ax.errorbar(m, yi, xerr=s, fmt="none", ecolor=c, elinewidth=2, capsize=0)
    ax.scatter(mean, y, s=36, c=colors, zorder=3, edgecolors="#fcfcfb", linewidths=1.5)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, series, ylabel):
    w = 0.8 / len(series)
    for j, (lab, col, vals) in enumerate(series):
        ax.bar(np.arange(len(groups)) + (j - (len(series) - 1) / 2) * w, vals, w - 0.04, color=col, label=lab)
    ax.set_xticks(range(len(groups)), groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)


# 1-2. benchmark: C_td and IBS per dataset, models sorted, ours highlighted
bench = pd.concat([load("02_tabular_benchmark"), load("03_multievent_benchmark")])
for metric, label, higher, fname in [
        ("ctd_weighted_avg", "C_td, mean ± sd over 10 splits (higher is better)", True, "benchmark_ctd.png"),
        ("ibs", "Integrated Brier score, mean ± sd (lower is better)", False, "benchmark_ibs.png")]:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    for ax, ds in zip(axes, NAMES):
        g = bench[bench.dataset == ds].groupby("model")[metric].agg(["mean", "std"]).dropna()
        g = g.sort_values("mean", ascending=higher)  # best at the top
        dots(ax, [MODEL[m] for m in g.index], g["mean"], g["std"],
             [OURS if m == "ours" else (ACCENT if m == "survtrace" else OTHER) for m in g.index], "")
        ax.set_title(NAMES[ds], color=INK, loc="left", fontsize=10)
    fig.supxlabel(label, color=INK, fontsize=9)
    fig.suptitle("Benchmark (notebooks 02, 03) - ours in blue, SurvTRACE in orange, best at the top",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)

# 3. loss ablation: paired change in C_td vs S1 on the same splits
l5 = load("05_ablation_losses")
t = paired_vs(l5, "ctd_weighted_avg", by="variant", baseline="S1", pair_on="seed", within=["dataset"])
t = t.iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.4))
dots(ax, [f"{NAMES[r.dataset]} · {r.variant}" for r in t.itertuples()], t.delta.values,
     (1.96 * t.delta_sd / np.sqrt(t.n)).values,
     [OURS if p < 0.05 else OTHER for p in t.p_wilcoxon], "Change in C_td vs S1 (L_PCH alone), 95% CI")
ax.axvline(0, color=MUTED, lw=1)
ax.set_title("Loss ablation (05): S2 = +L_rank, S3 = +L_mul, S4 = both\nblue: paired Wilcoxon p < 0.05",
             loc="left", color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "losses_delta_ctd.png")
plt.close(fig)

# 3b. stability: every split per recipe
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, (ds, metric, lab) in zip(axes, [("hsa_synthetic", "ctd_weighted_avg", "C_td"),
                                        ("ebmt", "brier_survtrace_weighted_avg", "Brier (lower is better)")]):
    x = l5[l5.dataset == ds]
    for i, v in enumerate(["S1", "S2", "S3", "S4"]):
        vals = x[x.variant == v][metric].values
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=18,
                   c=OTHER if v == "S1" else OURS, zorder=3)
    ax.set_xticks(range(4), ["S1\nL_PCH", "S2\n+rank", "S3\n+mul", "S4\n+both"])
    ax.set_ylabel(lab)
    ax.set_title(f"{NAMES[ds]}: one dot per split", loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT / "losses_stability.png")
plt.close(fig)

# 4. encoding and representation
e4 = load("04_ablation_encoding_representation")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, ds in zip(axes, ["metabric", "support"]):
    g = e4[e4.dataset == ds].groupby("variant").ctd_weighted_avg.agg(["mean", "std"]).sort_values("mean")
    lab = [v.replace("enc_", "encoding: ").replace("rep_", "repr: ") for v in g.index]
    dots(ax, lab, g["mean"], g["std"],
         [OURS if v == "enc_continuous" else (ACCENT if v.startswith("enc_") else OTHER) for v in g.index],
         "C_td, mean ± sd")
    ax.set_title(NAMES[ds], loc="left", color=INK, fontsize=10)
fig.suptitle("Encoding & representation (04): blue = paper's numeric embedding, orange = other encodings\n"
             "repr t2/t3/t4/t5 = sum/mean/concat/[CLS] over layers, s2/s3 = max/mean over tokens, all/last layers",
             x=0.01, ha="left", color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "encoding_representation.png")
plt.close(fig)

# 5. regression head: regression-only vs joint (means over recipes and splits)
r6 = load("06_regression_head")
r6["setup"] = np.where(r6.variant.str.startswith("joint"), "joint", "regression only")
dsl = list(NAMES)
groups = [NAMES[d] for d in dsl]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
mean = lambda df, d, s, m: df[(df.dataset == d) & (df.setup == s)][m].mean()  # noqa: E731
bars(axes[0], groups, [(s, c, [mean(r6, d, s, "reg_harrell") for d in dsl])
                       for s, c in [("regression only", OTHER), ("joint", OURS)]],
     "Harrell C of predicted time")
axes[0].set_ylim(0.4, 0.85)
axes[0].axhline(0.5, color=MUTED, lw=1, ls=":")
bars(axes[1], groups, [(s, c, [mean(r6, d, s, "reg_mae_margin") / mean(r6, d, "regression only", "reg_mae_margin")
                               for d in dsl]) for s, c in [("regression only", OTHER), ("joint", OURS)]],
     "MAE-margin relative to regression only")
axes[1].axhline(1.0, color=MUTED, lw=1, ls=":")
bars(axes[2], groups, [("survival only (05)", OTHER, [l5[l5.dataset == d].ctd_weighted_avg.mean() for d in dsl]),
                       ("joint (06)", OURS, [mean(r6, d, "joint", "ctd_weighted_avg") for d in dsl])],
     "C_td")
axes[2].set_ylim(0.45, 0.9)
fig.suptitle("Regression head (06): means over all recipes and 10 splits", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "regression_head.png")
plt.close(fig)

# 6. joint recipes on hsa synthetic
x = r6[(r6.dataset == "hsa_synthetic") & (r6.setup == "joint")].copy()
x["S"] = x.variant.str.extract(r"joint_(S\d)", expand=False)
x["R"] = x.variant.str.extract(r"x(R\d)", expand=False)
for metric, lab, fname, lo, hi in [("ctd_weighted_avg", "C_td", "joint_recipes_hsa_ctd.png", 0.70, 0.87),
                                   ("reg_harrell", "Harrell C of predicted time", "joint_recipes_hsa_harrell.png",
                                    0.58, 0.80)]:
    piv = x.groupby(["S", "R"])[metric].mean().unstack()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    im = ax.imshow(piv.values, cmap="Blues", vmin=lo, vmax=hi)
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > lo + 0.65 * (hi - lo) else INK, fontsize=8)
    ax.set_title(f"hsa synthetic, joint models: {lab}", loc="left", color=INK, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)

print("saved:", sorted(p.name for p in OUT.glob("*.png")))
