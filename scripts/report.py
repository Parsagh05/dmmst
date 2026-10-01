"""Tables and figures for the paper from one or more result folders written by
``scripts.runner`` (a folder of ``<tag>.json`` + ``<tag>.meta.json``).

    python scripts/report.py results/new/01_multievent results/new/02_scenarios ... \
        --out results/new/report

Writes markdown + LaTeX tables, CSVs and (for scenario sweeps) PDF/PNG figures.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # works both as `scripts.report` and as a plain script
    from scripts.runner import paired_vs
except ImportError:  # pragma: no cover
    from runner import paired_vs

RECIPE_NAMES = {
    "nllpch": "L_PCH",
    "nllpch_event_ranking": "L_PCH + L_mul",
    "nllpch_sample_ranking": "L_PCH + L_rank",
    "nllpch_sample_event_ranking": "L_PCH + L_rank + L_mul (paper)",
    "nllpch_mmv": "L_PCH + MMV",
    "nllpch_mmv_rank": "L_PCH + L_rank + L_mul + MMV",
}
BASELINE_NAMES = {
    "oracle": "Oracle (true CIF)",
    "coxph": "Cox PH (cause-specific)",
    "deephit_paper": "DeepHit",
    "dsm_paper": "DSM",
    "mensa_paper": "MENSA",
    "llm_pretrained": "End-to-end LLM (DistilGPT2, pretrained)",
    "llm_scratch": "End-to-end LLM (GPT-2, from scratch)",
}
METRIC_NAMES = {
    "ctd_weighted_avg": "C_td ↑",
    "brier_survtrace_weighted_avg": "Brier ↓",
    "within_ctd": "Within-subject C ↑",
    "within_ctd_km": "Within C, population KM",
    "within_ctd_gain": "Within C gain over KM ↑",
    "mae_margin": "MAE-margin ↓",
    "mae_uncensored": "MAE-uncens. ↓",
    "true_order_c": "True-order C ↑",
    "true_order_brier": "True-order Brier ↓",
    "true_order_logloss": "True-order log-loss ↓",
}
# fixed categorical order (validated palette); color follows the entity, never rank
SERIES_STYLE = {
    "Cox PH (cause-specific)": ("#2a78d6", "s"),
    "L_PCH": ("#eb6834", "o"),
    "L_PCH + L_mul": ("#1baf7a", "^"),
    "L_PCH + L_rank + L_mul (paper)": ("#eda100", "D"),
}
ORACLE_STYLE = ("#6b6b66", "--")


def load(dirs) -> pd.DataFrame:
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name.endswith(".meta.json"):
                continue
            meta_f = f.with_name(f"{f.stem}.meta.json")
            if not meta_f.is_file():  # environment.json, tuned_backbone.json, ...
                continue
            meta = json.loads(meta_f.read_text())
            test = json.loads(f.read_text()).get("test", {})
            row = {"tag": f.stem, "source": Path(d).name}
            row.update(meta.get("info", {}))
            row.update({k: v["mean"] for k, v in test.items() if isinstance(v, dict) and "mean" in v})
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        lv = df["lambda_v"] if "lambda_v" in df else pd.Series(np.nan, index=df.index)
        rec = df["recipe"] if "recipe" in df else pd.Series("", index=df.index)
        df["variant"] = [r if pd.isna(v) else f"{r}@{v:g}" for r, v in zip(rec, lv)]
        df["label"] = [
            RECIPE_NAMES.get(r, r) + (f" [λv={lv:g}]" if pd.notna(lv) else "")
            if m == "ours"
            else BASELINE_NAMES.get(m, m)
            for m, r, lv in zip(df["model"], rec, lv)
        ]
        if "stage" in df:  # notebook 08: tuned penalties must not merge with untuned runs
            df["label"] = [lab + " [tuned]" if st == "tuned" else lab for lab, st in zip(df["label"], df["stage"])]
            df["variant"] = [v + "@tuned" if st == "tuned" else v for v, st in zip(df["variant"], df["stage"])]
        if "variant_tag" in df:  # final runs: model configuration (e.g. large+CLS) in the label
            df["label"] = [lab if pd.isna(t) else f"{lab} ({t})" for lab, t in zip(df["label"], df["variant_tag"])]
        if "repr" in df:  # §4.2: same model on sequence / bag / static inputs
            df["label"] = [lab if pd.isna(rp) else f"{lab} [{rp}]" for lab, rp in zip(df["label"], df["repr"])]
    return df


def mean_sd(df, metrics, by, digits=3):
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(by, dropna=False, sort=False)[metrics]
    mean, sd, n = g.mean(), g.std().fillna(0.0), g.count()
    out = mean.astype(object).copy()
    for m in metrics:
        out[m] = [
            "" if c == 0 else (f"{a:.{digits}f} ± {b:.{digits}f}" if c > 1 else f"{a:.{digits}f}")
            for a, b, c in zip(mean[m], sd[m], n[m])
        ]
    out["n"] = n.max(axis=1)
    return out.rename(columns=METRIC_NAMES)


def write_table(df: pd.DataFrame, out: Path, name: str, title: str):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv")
    try:
        body = df.to_markdown()
    except ImportError:  # tabulate not installed
        body = df.to_string()
    md = f"### {title}\n\n{body}\n"
    (out / f"{name}.md").write_text(md, encoding="utf-8")
    try:
        (out / f"{name}.tex").write_text(df.to_latex(escape=True), encoding="utf-8")
    except Exception:  # noqa: BLE001  (jinja2 missing)
        pass
    return md


def benchmark_tables(df, out: Path, metrics, per_dataset=True):
    """One table per dataset: every model/recipe, mean ± sd over seeds."""
    mds = []
    for ds, sub in df.groupby("dataset", sort=False):
        t = mean_sd(sub, metrics, "label")
        mds.append(write_table(t, out, f"table_{ds}", f"{ds} (mean ± sd over seeds)"))
    return "\n".join(mds)


def ablation_tables(df, out: Path, metrics, baseline="nllpch"):
    """Each of our recipes vs L_PCH, paired on seed (= same split and init)."""
    ours = df[df["model"] == "ours"]
    within = ["dataset"] + (["repr"] if "repr" in ours and ours["repr"].notna().any() else [])
    parts = [
        paired_vs(ours, m, by="variant", baseline=baseline, pair_on="seed", within=within)
        for m in metrics
        if m in ours.columns
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return ""
    t = pd.concat(parts, ignore_index=True)
    def name(v):
        r, _, lv = v.partition("@")
        return RECIPE_NAMES.get(r, r) + (f" [λv={lv}]" if lv else "")

    t["recipe"] = t.pop("variant").map(name)
    t["vs"] = t["vs"].map(name)
    t["metric"] = t["metric"].map(lambda m: METRIC_NAMES.get(m, m))
    for c in ("delta", "delta_sd"):
        t[c] = t[c].round(4)
    for c in ("p_ttest", "p_wilcoxon"):
        t[c] = t[c].map(lambda p: f"{p:.3g}")
    return write_table(t.set_index(within + ["metric", "recipe"]), out, "ablation_paired",
                       "Paired differences vs L_PCH (same seed = same split and init)")


def _expand_base(df):
    """The base scenario is trained once but belongs to every sweep: copy its rows
    into each sweep at that sweep's base value (stored in info['base_knobs'])."""
    base = df[df["sweep"] == "base"]
    if base.empty or "base_knobs" not in df:
        return df
    copies = []
    for sweep in df.loc[df["sweep"].notna() & (df["sweep"] != "base"), "sweep"].unique():
        c = base.copy()
        c["sweep"] = sweep
        c["value"] = [bk.get(sweep) if isinstance(bk, dict) else np.nan for bk in c["base_knobs"]]
        copies.append(c)
    return pd.concat([df] + copies, ignore_index=True)


def representation_tables(df, out: Path, metrics):
    """§4.2: the same model and loss on the full sequence vs a bag-of-codes summary vs
    static features only, paired on seed (same split and init)."""
    if "repr" not in df or df["repr"].isna().all():
        return ""
    ours = df[(df["model"] == "ours") & df["repr"].notna()]
    parts = [
        paired_vs(ours, m, by="repr", baseline="bag", pair_on="seed", within=["dataset", "recipe"])
        for m in metrics if m in ours.columns
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return ""
    t = pd.concat(parts, ignore_index=True)
    t["recipe"] = t["recipe"].map(lambda r: RECIPE_NAMES.get(r, r))
    t["metric"] = t["metric"].map(lambda m: METRIC_NAMES.get(m, m))
    for c in ("delta", "delta_sd"):
        t[c] = t[c].round(4)
    for c in ("p_ttest", "p_wilcoxon"):
        t[c] = t[c].map(lambda p: f"{p:.3g}")
    return write_table(t.set_index(["dataset", "recipe", "metric", "repr"]), out, "representation_paired",
                       "Sequence / static vs bag-of-codes input (paired on seed)")


def sweep_figures(df, out: Path, metrics=("within_ctd", "ctd_weighted_avg", "within_ctd_gain", "true_order_brier")):
    """One panel per metric and sweep: metric vs knob value, mean ± sd over seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "sweep" not in df.columns:
        return []
    df = _expand_base(df)
    files = []
    for sweep, sub in df[df["sweep"].notna() & (df["sweep"] != "base")].groupby("sweep"):
        categorical = not pd.to_numeric(sub["value"], errors="coerce").notna().all()
        if categorical:
            order = list(dict.fromkeys(sub["value"]))
            sub = sub.assign(value=sub["value"].map({v: i for i, v in enumerate(order)}))
        ms = [m for m in metrics if m in sub.columns]
        fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.4), squeeze=False)
        for ax, m in zip(axes[0], ms):
            for label, (color, marker) in list(SERIES_STYLE.items()) + [(BASELINE_NAMES["oracle"], (None, None))]:
                s = sub[sub["label"] == label]
                if s.empty or s[m].isna().all():
                    continue
                g = s.groupby("value")[m].agg(["mean", "std"]).sort_index()
                x = g.index.values.astype(float)
                if marker is None:  # oracle: neutral dashed reference, not a series color
                    ax.plot(x, g["mean"], ORACLE_STYLE[1], color=ORACLE_STYLE[0], lw=1.5, marker="x", ms=5, label=label)
                    continue
                ax.errorbar(x, g["mean"], yerr=g["std"].fillna(0), color=color, marker=marker,
                            ms=6, lw=2, capsize=3, label=label)
            ax.set_title(METRIC_NAMES.get(m, m), fontsize=10)
            ax.set_xlabel(sweep)
            if categorical:
                ax.set_xticks(range(len(order)), order)
            ax.grid(alpha=0.25, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        for ext in ("pdf", "png"):
            f = out / f"sweep_{sweep}.{ext}"
            fig.savefig(f, dpi=200)
            files.append(f)
        plt.close(fig)
        # the table behind the figure (identity is never color-alone)
        t = sub.groupby(["value", "label"])[ms].agg(["mean", "std"]).round(4)
        t.to_csv(out / f"sweep_{sweep}.csv")
    return files


def build(dirs, out, metrics=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(dirs)
    if df.empty:
        print("no results found in", dirs)
        return df
    df.to_csv(out / "all_runs.csv", index=False)
    df = df[df["model"] != "tune"]  # backbone-tuning runs are selection, not results
    metrics = metrics or [m for m in METRIC_NAMES if m in df.columns]
    parts = [
        benchmark_tables(df[df.get("sweep").isna()] if "sweep" in df else df, out, metrics),
        ablation_tables(df[df.get("sweep").isna()] if "sweep" in df else df, out,
                        [m for m in ("ctd_weighted_avg", "within_ctd", "within_ctd_gain", "true_order_c",
                                     "true_order_brier", "true_order_logloss",
                                     "brier_survtrace_weighted_avg", "mae_margin") if m in df]),
    ]
    parts.append(representation_tables(df, out, [m for m in ("ctd_weighted_avg", "within_ctd",
                                                           "brier_survtrace_weighted_avg") if m in df]))
    figs = sweep_figures(df, out)
    (out / "REPORT.md").write_text("\n\n".join(p for p in parts if p), encoding="utf-8")
    print(f"{len(df)} runs -> {out} ({len(figs)} figure files)")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/new/report")
    args = ap.parse_args(argv)
    build(args.dirs, args.out)


if __name__ == "__main__":
    main()
