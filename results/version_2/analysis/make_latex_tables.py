"""LaTeX tables of the version-2 results (notebooks 01-06) for report_latex/full_report.tex.

    python results/version_2/analysis/make_latex_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.runner import Runner, paired_vs  # noqa: E402

OUT = Path("results/version_2/analysis/report_latex/tables")
OUT.mkdir(parents=True, exist_ok=True)
DS = {"metabric": "METABRIC", "support": "SUPPORT", "ebmt": "EBMT", "hsa_synthetic": "hsa synth.",
      "deephit_synthetic": "DeepHit synth."}
K = {"metabric": 1, "support": 1, "ebmt": 4, "hsa_synthetic": 2, "deephit_synthetic": 2}
EVENTS = {"ebmt": ["recovery", "adverse event", "relapse", "death"], "hsa_synthetic": ["event 1", "event 2"],
          "deephit_synthetic": ["event 1", "event 2"]}
MODEL = {"ours": "Ours (S1)", "survtrace": "SurvTRACE", "deephit": "DeepHit", "rsf": "RSF",
         "pchazard": "PC-Hazard", "deepsurv": "DeepSurv", "mensa": "MENSA", "cox": "Cox", "dsm": "DSM"}


def load(n):
    return Runner(repo=".", results_dir=f"results/version_2/{n}", verbose=False).collect()


def cell(x, best=False, d=3):
    x = x.dropna()
    if not len(x):
        return "---"
    s = f"{x.mean():.{d}f} ({x.std():.{d}f})"
    return f"\\textbf{{{x.mean():.{d}f}}} ({x.std():.{d}f})" if best else s


def table(df, rows, row_col, cols, metric, higher=True, colnames=None, d=3, row_names=None):
    """rows x cols of mean (sd); best per column in bold."""
    means = {c: df[df[c[0]] == c[1]].groupby(row_col)[metric].mean() if isinstance(c, tuple) else None
             for c in cols}
    lines = []
    for r in rows:
        cells = []
        for c in cols:
            sub = df[(df[c[0]] == c[1]) & (df[row_col] == r)][metric]
            m = means[c].dropna()
            best = len(sub.dropna()) and len(m) and np.isclose(sub.mean(), m.max() if higher else m.min())
            cells.append(cell(sub, best, d))
        lines.append(f"{(row_names or {}).get(r, r)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def write(name, header, body, colspec):
    (OUT / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n",
        encoding="utf-8")


bench = pd.concat([load("02_tabular_benchmark"), load("03_multievent_benchmark")])
l5, e4, r6 = load("05_ablation_losses"), load("04_ablation_encoding_representation"), load("06_regression_head")
models = list(MODEL)
dcols = [("dataset", d) for d in DS]
hdr = "Model & " + " & ".join(DS.values())

# benchmark: several metrics, one table each
for metric, higher, name in [("ctd_weighted_avg", True, "bench_ctd"), ("ibs", False, "bench_ibs"),
                             ("brier_survtrace_weighted_avg", False, "bench_brier"),
                             ("auc", True, "bench_auc"), ("antolini", True, "bench_antolini"),
                             ("ibll", False, "bench_ibll"), ("dcal_p", True, "bench_dcal"),
                             ("mae_linear_margin", False, "bench_mae")]:
    d = 1 if metric == "mae_linear_margin" else 3
    write(name, hdr, table(bench, models, "model", dcols, metric, higher, d=d, row_names=MODEL), "l" + "c" * 5)

# SurvTRACE-format: C_td at 25/50/75% per event
for ds in DS:
    cols, names = [], []
    for k in range(K[ds]):
        for q in (0.25, 0.5, 0.75):
            col = f"ctd_{k}th_event_{q}"
            if col in bench.columns:
                cols.append(col)
                names.append(f"{int(q * 100)}\\%")
    x = bench[bench.dataset == ds]
    lines = []
    for m in models:
        cells = []
        for col in cols:
            means = x.groupby("model")[col].mean()
            sub = x[x.model == m][col]
            cells.append(cell(sub, len(sub.dropna()) and np.isclose(sub.mean(), means.max())))
        lines.append(f"{MODEL[m]} & " + " & ".join(cells) + " \\\\")
    if K[ds] > 1:
        top = " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{e}}}" for e in EVENTS[ds])
        header = top + " \\\\\n" + "Model & " + " & ".join(names)
    else:
        header = "Model & " + " & ".join(names)
    write(f"horizons_{ds}", header, "\n".join(lines), "l" + "c" * len(cols))

# losses (05): several metrics
l5["variant_name"] = l5.variant
for metric, higher, name in [("ctd_weighted_avg", True, "loss_ctd"), ("ibs", False, "loss_ibs"),
                             ("brier_survtrace_weighted_avg", False, "loss_brier"), ("auc", True, "loss_auc")]:
    write(name, "Recipe & " + " & ".join(DS.values()),
          table(l5, ["S1", "S2", "S3", "S4"], "variant", dcols, metric, higher,
                row_names={"S1": "S1: $L_{PCH}$", "S2": "S2: $+L_{rank}$", "S3": "S3: $+L_{mul}$",
                           "S4": "S4: $+L_{rank}+L_{mul}$"}), "l" + "c" * 5)

# paired tests vs S1
lines = []
for metric, lab in [("ctd_weighted_avg", "$C_{td}$"), ("ibs", "IBS")]:
    t = paired_vs(l5, metric, by="variant", baseline="S1", pair_on="seed", within=["dataset"])
    for r in t.itertuples():
        lines.append(f"{lab} & {DS[r.dataset]} & {r.variant} & {r.delta:+.4f} & {r.wins} & "
                     f"{r.p_ttest:.3f} & {r.p_wilcoxon:.3f} \\\\")
write("loss_tests", "Metric & Dataset & Recipe & $\\Delta$ vs S1 & wins & $p$ (t-test) & $p$ (Wilcoxon)",
      "\n".join(lines), "lllcccc")

# encoding / representation (04)
order = ["enc_continuous", "enc_quantile", "enc_discretised"] + sorted(
    v for v in e4.variant.unique() if v.startswith("rep_"))
names = {"enc_continuous": "encoding: numeric embedding (paper)", "enc_quantile": "encoding: quantile",
         "enc_discretised": "encoding: bins"}
for v in order:
    if v.startswith("rep_"):
        _, t, s, layers = v.split("_")
        names[v] = (f"repr.: {dict(t2='sum', t3='mean', t4='concat', t5='[CLS]')[t]} / "
                    f"{dict(s1='-', s2='max', s3='mean')[s]} / {layers} layers")
e4c = [("dataset", "metabric"), ("dataset", "support")]
for metric, higher, name in [("ctd_weighted_avg", True, "enc_ctd"), ("ibs", False, "enc_ibs")]:
    pass
lines = []
for v in order:
    cells = []
    for ds in ("metabric", "support"):
        x = e4[e4.dataset == ds]
        for metric, higher in [("ctd_weighted_avg", True), ("ibs", False)]:
            means = x.groupby("variant")[metric].mean()
            sub = x[x.variant == v][metric]
            cells.append(cell(sub, np.isclose(sub.mean(), means.max() if higher else means.min())))
    lines.append(f"{names[v]} & " + " & ".join(cells) + " \\\\")
write("encoding", " & \\multicolumn{2}{c}{METABRIC} & \\multicolumn{2}{c}{SUPPORT} \\\\\n"
      "Variant & $C_{td}$ & IBS & $C_{td}$ & IBS", "\n".join(lines), "lcccc")

# regression head (06)
r6["setup"] = np.where(r6.variant.str.startswith("joint"), "joint", "reg")
lines = []
for ds in DS:
    x = r6[r6.dataset == ds]
    for v in sorted(x[x.setup == "reg"].variant.unique()):
        y = x[x.variant == v]
        lines.append(f"{DS[ds]} & regression only & {v.replace('reg_', '')} & {cell(y.reg_mae_margin, d=1)} & "
                     f"{cell(y.reg_mae_pseudo_obs, d=1)} & {cell(y.reg_harrell)} & --- \\\\")
    j = x[x.setup == "joint"]
    lines.append(f"{DS[ds]} & joint (all recipes) & mean & {cell(j.reg_mae_margin, d=1)} & "
                 f"{cell(j.reg_mae_pseudo_obs, d=1)} & {cell(j.reg_harrell)} & {cell(j.ctd_weighted_avg)} \\\\")
    lines.append("\\midrule")
write("regression", "Dataset & Setup & Recipe & MAE-margin & MAE-pseudo-obs. & Harrell $C$ & $C_{td}$",
      "\n".join(lines[:-1]), "lllcccc")

# joint recipes: C_td and Harrell per S x R, multi-event data
lines = []
for ds in ("ebmt", "hsa_synthetic", "deephit_synthetic"):
    x = r6[(r6.dataset == ds) & (r6.setup == "joint")]
    for s in ("S1", "S2", "S3", "S4"):
        cells = []
        for rr in ("R1", "R2", "R3", "R4"):
            y = x[x.variant == f"joint_{s}x{rr}"]
            cells.append(f"{y.ctd_weighted_avg.mean():.3f} / {y.reg_harrell.mean():.3f}")
        lines.append(f"{DS[ds] if s == 'S1' else ''} & {s} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
write("joint_recipes", "Dataset & Survival & R1 & R2 & R3 & R4", "\n".join(lines[:-1]), "llcccc")
x = r6[(r6.setup == "joint") & r6.dataset.isin(["metabric", "support"])]
lines = []
for ds in ("metabric", "support"):
    for s in ("S1", "S2"):
        y = [x[(x.dataset == ds) & (x.variant == f"joint_{s}x{rr}")] for rr in ("R1", "R2")]
        lines.append(f"{DS[ds] if s == 'S1' else ''} & {s} & " + " & ".join(
            f"{z.ctd_weighted_avg.mean():.3f} / {z.reg_harrell.mean():.3f}" for z in y) + " \\\\")
write("joint_recipes_single", "Dataset & Survival & R1 & R2", "\n".join(lines), "llcc")

# tuned settings (01)
t = json.loads(Path("results/version_2/01_tuning/tuned.json").read_text())
lines = []
for ds in DS:
    o = t[ds]["ours"]
    st = t[ds]["survtrace"]
    lines.append(f"{DS[ds]} & {o['learning_rate']:g} & {o['weight_decay']:g} & {o['transformer_num_hidden_layers']} & "
                 f"{o['transformer_hidden_size']} & {o['transformer_intermediate_size']} & "
                 f"{o['transformer_num_attention_heads']} & {st['st_learning_rate']:g} & "
                 f"{st['st_num_hidden_layers']} & {st['st_hidden_size']} \\\\")
write("tuned_models", "Dataset & lr & wd & layers & hidden & interm. & heads & ST lr & ST layers & ST hidden",
      "\n".join(lines), "lccccccccc")
lines = []
for ds in DS:
    L = t["losses"][ds]
    mul = f"{L['mul']['coeff']:g} / {L['mul']['sigma']:g}" if "mul" in L else "---"
    lines.append(f"{DS[ds]} & {L['rank']['coeff']:g} / {L['rank']['sigma']:g} & {mul} & "
                 f"{L.get('mm_coeff', '---')} & {'raw days' if L['time_scale_raw'] else 'end of follow-up'} \\\\")
write("tuned_losses", "Dataset & $L_{rank}$ weight / $\\sigma$ & $L_{mul}$ weight / $\\sigma$ & $L_{MM}$ weight & "
      "time unit (regression only)", "\n".join(lines), "lcccc")
print("tables:", sorted(p.name for p in OUT.glob("*.tex")))
