"""Generate the version-2 notebooks next to this file (shared cells written once).

    python notebooks/version_2/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


COMMON_OPTIONS = '''
# --- execution ------------------------------------------------------------------------
SMOKE_TEST   = False    # True: 2 epochs, 1 seed, 1-2 configurations - only to check that it runs
SMOKE_EPOCHS = 2
WORKERS      = None     # parallel runs; None = 2 on Kaggle (4 vCPU), 4 locally
TIME_BUDGET_MIN = 11 * 60   # stop launching new runs after this (Kaggle sessions end at 12 h);
                            # re-run with this notebook's output attached to continue
INSTALL_DEPS = True     # Kaggle only
GIT_URL      = "https://github.com/Parsagh05/dmmst.git"   # the code is cloned from here
CLEAN_UP_AT_END = True  # Kaggle: delete the working repo copy after the report
'''

SETUP = r'''
# ---- 1. get the code --------------------------------------------------------------
import os, sys, json, shutil, subprocess, time, hashlib
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").is_dir()

def _is_repo(p):
    return (p / "sat" / "finetune.py").is_file() and (p / "conf").is_dir()

if ON_KAGGLE:
    WORK = Path("/kaggle/working")
    REPO = WORK / "dmmst"
    if not _is_repo(REPO):
        subprocess.run(["git", "clone", "--depth", "1", GIT_URL, str(REPO)], check=True)
    RESULTS_ROOT = WORK / "results"
    INPUT_ROOTS = [Path("/kaggle/input")]
else:
    REPO = Path.cwd().resolve()
    while not _is_repo(REPO) and REPO != REPO.parent:
        REPO = REPO.parent
    RESULTS_ROOT = REPO / "results" / "version_2"
    INPUT_ROOTS = []
assert _is_repo(REPO), REPO
os.chdir(REPO)
sys.path.insert(0, str(REPO))
CODE_FINGERPRINT = hashlib.md5(b"".join(
    p.read_bytes() for p in sorted((REPO / "sat").rglob("*.py")) + sorted((REPO / "conf").rglob("*.yaml"))
)).hexdigest()[:10]
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
                        capture_output=True).stdout.strip()
print("repo:", REPO, "| commit:", COMMIT, "| code fingerprint:", CODE_FINGERPRINT)
'''

INSTALL = r'''
# ---- 2. dependencies (Kaggle) ------------------------------------------------------
# numpy and torch are NEVER moved: pandas/pyarrow and CUDA in the image are built for them.
if ON_KAGGLE and INSTALL_DEPS:
    import numpy as _np, torch
    _tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    pkgs = ["transformers==4.50.0", "datasets==3.4.1", "tokenizers>=0.21,<0.22", "evaluate>=0.4.3",
            "accelerate>=1.4.0", "hydra-core>=1.3.2", "hydra-colorlog>=1.2.0",
            "torchsurv" if _tv >= (2, 8) else "torchsurv==0.1.4",
            "h5py>=3.13", "logdecorator>=2.5", "einops==0.8.1", "torchtuples>=0.2.2",
            "numba", "nvidia-ml-py", "polars", "tqdm", "tabulate",
            "scikit-survival" if int(_np.__version__.split(".")[0]) >= 2 else "scikit-survival==0.22.2",
            "pycox==0.3.0", "peft", "easydict", "lifelines", "osqp", "patsy",
            f"numpy=={_np.__version__}", f"torch=={torch.__version__.split('+')[0]}"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    # SurvivalEVAL declares torch>=2.13; it runs on older torch, so its pins are skipped
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "SurvivalEVAL==0.8.5"], check=True)
import torch
print("torch", torch.__version__, "| cuda" if torch.cuda.is_available() else "| cpu")
'''

RUNNER = r'''
# ---- 3. runner ----------------------------------------------------------------------
from IPython.display import Markdown, display
import pandas as pd
from scripts.runner import Runner, select_best
from scripts import v2

RUN_NAME = NAME + ("_smoke" if SMOKE_TEST else "")
r = Runner(repo=REPO, results_dir=RESULTS_ROOT / RUN_NAME,
           workers=WORKERS or (2 if ON_KAGGLE else 4),
           smoke_epochs=SMOKE_EPOCHS if SMOKE_TEST else None, gpu="0")

# Resume: attach this notebook's previous output ("Add Input") and finished runs are
# copied back, so only the missing ones are trained.
n_copied = 0
for root in INPUT_ROOTS:
    for d in root.rglob(RUN_NAME):
        if d.is_dir() and d != r.results_dir:
            for f in d.glob("*.json"):
                if not (r.results_dir / f.name).exists():
                    shutil.copy2(f, r.results_dir / f.name); n_copied += 1
print(f"resumed {n_copied} files from attached inputs")
(r.results_dir / "environment.json").write_text(json.dumps({
    "commit": COMMIT, "code_fingerprint": CODE_FINGERPRINT, "torch": torch.__version__,
    "cuda": torch.cuda.is_available(), "smoke": SMOKE_TEST, "name": RUN_NAME}, indent=2))
print("results ->", r.results_dir)
'''

LOAD_TUNED = r'''
# ---- settings tuned on validation in notebook 01 -------------------------------------
# an attached 01 output wins; otherwise the copy committed to GitHub
# (results/version_2/01_tuning/tuned.json in the cloned repo)
TUNED_PATH = v2.find_tuned(RESULTS_ROOT, INPUT_ROOTS + [REPO / "results" / "version_2" / "01_tuning"])
print("tuned settings from", TUNED_PATH)
TUNED = v2.load_tuned(TUNED_PATH)
LOSS = TUNED["losses"]
print(json.dumps(TUNED, indent=1)[:1500])
'''

FINISH = r'''
# ---- tables + zip ---------------------------------------------------------------------
rep = r.results_dir / "report"
rep.mkdir(exist_ok=True)
res = r.collect()
res.to_csv(rep / "all_runs.csv", index=False)
if len(res):
    t = v2.table(res, ["dataset", "model", "variant"])
    t.to_csv(rep / "summary.csv", index=False)
    display(t)
print("zip:", r.zip())
'''

CLEANUP = r"""
if ON_KAGGLE and CLEAN_UP_AT_END:
    shutil.rmtree(REPO, ignore_errors=True)
    print("removed", REPO, "- results are in", r.results_dir)
"""


def notebook(name, title_md, options, body, needs_tuned=True):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [md(title_md), md("## Options"), code(f'NAME = "{name}"\n' + options + COMMON_OPTIONS),
                md("## Setup (identical in every notebook)"), code(SETUP), code(INSTALL), code(RUNNER)]
    if needs_tuned:
        nb.cells += [code(LOAD_TUNED)]
    nb.cells += body
    nb.cells += [md("## Clean up"), code(CLEANUP)]
    nbf.write(nb, OUT / f"{name}.ipynb")
    print("wrote", OUT / f"{name}.ipynb")


SEEDS_OPT = '''
SEEDS = list(range(10))   # protocol: 10 random 60/10/30 splits (as SurvTRACE); seed s = split s and init s
'''

# ================================================================== 01 tuning
notebook("01_tuning", """# 01 · Tuning (validation only)

Every setting the paper leaves open is chosen here, **on the validation split only** -
test numbers are never looked at. The result, `tuned.json`, is read by notebooks 02-07
(on Kaggle: add this notebook's output as an input there).

* **Stage A - models.** For every dataset: our transformer and official SurvTRACE on
  `N_CONFIGS` configurations drawn from SurvTRACE's grid (lr, weight decay, layers,
  embedding size, intermediate size, heads); Cox (ridge penalty), Random Survival Forest,
  DeepSurv and PC-Hazard on their own grids. DeepHit / DSM / MENSA use the transformer
  backbone tuned for our model (same encoder, different head).
* **Stage B - loss weights** (with the tuned backbone): weight and sigma of L_rank (Eq. 4)
  and L_mul (Eq. 5), the L_MM weight (Eq. 7) and the regression
  head's time unit (end of follow-up vs raw time).

Selection: mean validation C_td over `TUNE_SEEDS` (regression-only runs: validation
MAE-margin).""", SEEDS_OPT + '''
TUNE_SEEDS = [0]          # splits used for tuning
DATASETS = ["metabric", "support", "ebmt", "hsa_synthetic", "deephit_synthetic"]
N_CONFIGS = 24            # configurations drawn from the 216-point SurvTRACE grid
N_CONFIGS_NN = 16         # DeepSurv / PC-Hazard (36-point grid)
MODELS = ["ours", "survtrace", "cox", "rsf", "deepsurv", "pchazard"]
RANK_GRID = {"coeff": [0.1, 0.5, 1.0], "sigma": [0.1, 0.5, 1.0]}
MM_COEFF = [0.1, 0.5, 1.0]
''', [
    md("## Data (built once, label files per split seed)"), code(r'''
if SMOKE_TEST:
    TUNE_SEEDS, N_CONFIGS, N_CONFIGS_NN = TUNE_SEEDS[:1], 2, 2
    DATASETS = DATASETS[:1] + ["ebmt"]
v2.prepare(r, DATASETS, sorted(set(TUNE_SEEDS)), REPO)
'''),
    md("## Stage A · models"), code(r'''
runs = []
for name in DATASETS:
    for model in MODELS:
        n = N_CONFIGS if model in ("ours", "survtrace") else (N_CONFIGS_NN if model in ("deepsurv", "pchazard") else None)
        configs = v2.sample_grid(v2.TUNE_GRIDS[model], n, seed=0)
        runs += v2.tune_runs(model, name, configs, TUNE_SEEDS, REPO)
print(len(runs), "tuning runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)

TUNED = {name: {} for name in DATASETS}
for name in DATASETS:
    for model in MODELS:
        try:
            best = select_best(r, f"{name}__{model}__tune")
            TUNED[name][model] = best["config"]
            print(f"{name:18s} {model:10s} val C_td {best['score']:.4f}  {best['config']}")
        except RuntimeError as e:
            print(f"{name:18s} {model:10s} NOT TUNED ({e}) - defaults will be used")
'''),
    md("## Stage B · loss weights (tuned backbone)"), code(r'''
def ours(name):
    return v2.tuned_overrides(TUNED, "ours", name)

runs = []
rank_cfgs = v2.grid(RANK_GRID) if not SMOKE_TEST else v2.grid(RANK_GRID)[:2]
for name in DATASETS:
    multi = v2.DATASETS[name]["events"] > 1
    runs += v2.tune_runs("ours", name, [{"v2_rank_coeff": c["coeff"], "v2_rank_sigma": c["sigma"]} for c in rank_cfgs],
                         TUNE_SEEDS, REPO, extra=ours(name), prefix=f"{name}__rank__tune")
    # regression head time unit: end of follow-up (default) vs raw time, R2 regression-only
    reg = ours(name) + ["tasks=v2_regression", "selection_metric=eval_reg_mae_margin", "selection_greater=false",
                        "v2_l1_type=margin"]
    runs += v2.tune_runs("ours", name, [{}, {"v2_time_scale": 1.0}], TUNE_SEEDS, REPO, extra=reg,
                         prefix=f"{name}__timescale__tune")
    if multi:
        runs += v2.tune_runs("ours", name, [{"v2_mul_coeff": c["coeff"], "v2_mul_sigma": c["sigma"]} for c in rank_cfgs],
                             TUNE_SEEDS, REPO, extra=ours(name), prefix=f"{name}__mul__tune")
        runs += v2.tune_runs("ours", name, [{"v2_mm_coeff": c, "v2_mm_t_q": "observed"} for c in MM_COEFF],
                             TUNE_SEEDS, REPO, extra=reg, prefix=f"{name}__mm__tune")
print(len(runs), "loss-tuning runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)

def pick(prefix, metric="ctd_weighted_avg", greater=True, default=None):
    try:
        b = select_best(r, prefix, metric, greater)
        print(f"{prefix:32s} {b['config']}  (val {metric} {b['score']:.4f})")
        return b["config"]
    except RuntimeError:
        print(f"{prefix:32s} not available - default {default}")
        return default

TUNED["losses"] = {}
for name in DATASETS:
    multi = v2.DATASETS[name]["events"] > 1
    rk = pick(f"{name}__rank__tune", default={"v2_rank_coeff": 0.5, "v2_rank_sigma": 0.5})
    ts = pick(f"{name}__timescale__tune", "reg_mae_margin", False, default={})
    L = {"rank": {"coeff": rk["v2_rank_coeff"], "sigma": rk["v2_rank_sigma"]},
         "time_scale_raw": bool(ts.get("v2_time_scale") == 1.0)}
    if multi:
        mu = pick(f"{name}__mul__tune", default={"v2_mul_coeff": 0.5, "v2_mul_sigma": 0.5})
        mm = pick(f"{name}__mm__tune", "reg_mae_margin", False, default={"v2_mm_coeff": 0.5})
        L["mul"] = {"coeff": mu["v2_mul_coeff"], "sigma": mu["v2_mul_sigma"]}
        L["mm_coeff"] = mm["v2_mm_coeff"]
    TUNED["losses"][name] = L

(r.results_dir / "tuned.json").write_text(json.dumps(TUNED, indent=2))
print(json.dumps(TUNED, indent=1))
'''),
    md("## Tables + zip"), code(r'''
rep = r.results_dir / "report"
rep.mkdir(exist_ok=True)
val = r.collect(split="validation")
val.to_csv(rep / "tuning_validation.csv", index=False)
print("zip:", r.zip())
'''),
], needs_tuned=False)


# shared snippet: tuned overrides + loss recipes per dataset
HELPERS = r'''
def tuned(model, name):
    return v2.tuned_overrides(TUNED, model, name)

def loss_extra(name):
    """Per-dataset extras: time unit chosen in notebook 01."""
    return ["v2_time_scale=1.0"] if LOSS[name].get("time_scale_raw") else []

def surv_recipes(name):
    L = LOSS[name]
    multi = v2.DATASETS[name]["events"] > 1
    return v2.survival_recipes(multi, L["rank"], L.get("mul", {"coeff": 0, "sigma": 0.1}))

def reg_recipes(name):
    L = LOSS[name]
    return v2.regression_recipes(v2.DATASETS[name]["events"] > 1, L.get("mm_coeff", 0.5))
'''

BENCH_MODELS = '''
MODELS = ["ours", "cox", "rsf", "deepsurv", "pchazard", "deephit", "dsm", "mensa", "survtrace"]
'''

# ================================================================== 02 tabular
notebook("02_tabular_benchmark", """# 02 · Tabular benchmark (paper §4.1)

METABRIC and SUPPORT, the two datasets of SurvTRACE's Table 2, under its protocol:
10 random 60/10/30 splits, C_td (IPCW, censoring from training) at the 25/50/75% event-time
quantiles, mean (std). Our model uses L_PCH (recipe S1); the loss ablations are in 05.

Models: ours, Cox, Random Survival Forest, DeepSurv, PC-Hazard, DeepHit, DSM, MENSA and
SurvTRACE (authors' released code), all with the settings tuned in notebook 01.""",
    SEEDS_OPT + BENCH_MODELS + '''
DATASETS = ["metabric", "support"]
# published averages over 25/50/75% (SurvTRACE Table 2), for the report
PUBLISHED = {"metabric": {"SurvTRACE (paper)": 0.691}, "support": {"SurvTRACE (paper)": 0.640}}
''', [
    code(HELPERS),
    md("## Runs"), code(r'''
if SMOKE_TEST:
    SEEDS, MODELS = SEEDS[:1], ["ours", "cox", "survtrace"]
v2.prepare(r, DATASETS, SEEDS, REPO)
runs = []
for name in DATASETS:
    for model in MODELS:
        runs += v2.runs(model, name, SEEDS, REPO, extra=tuned(model, name), info={"stage": "benchmark"})
print(len(runs), "runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## SurvTRACE-format table (C_td at 25/50/75%) and all metrics"), code(FINISH + r'''
ht = v2.horizon_table(res, ["dataset", "model"])
ht.to_csv(rep / "ctd_horizons.csv", index=False)
display(ht)
'''),
])

# ================================================================== 03 multi-event
notebook("03_multievent_benchmark", """# 03 · Multi-event benchmark (paper §4.3)

EBMT (real, 4 non-exclusive events, death ends follow-up), hsa_synthetic (Tjandra et al.
2021, 2 events that can both occur) and DeepHit's synthetic competing-risks data
(Lee et al. 2018). C_td and Brier per event at that event's 25/50/75% horizons.

Baselines are fitted cause-specifically where they are single-event models (Cox, RSF,
DeepSurv, PC-Hazard, and SurvTRACE where events have their own times); DeepHit, DSM and
MENSA are multi-event heads on the tuned transformer.""",
    SEEDS_OPT + BENCH_MODELS + '''
DATASETS = ["ebmt", "hsa_synthetic", "deephit_synthetic"]
''', [
    code(HELPERS),
    md("## Runs"), code(r'''
if SMOKE_TEST:
    SEEDS, MODELS, DATASETS = SEEDS[:1], ["ours", "cox", "survtrace"], DATASETS[:1]
v2.prepare(r, DATASETS, SEEDS, REPO)
runs = []
for name in DATASETS:
    for model in MODELS:
        runs += v2.runs(model, name, SEEDS, REPO, extra=tuned(model, name), info={"stage": "benchmark"})
print(len(runs), "runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Tables"), code(FINISH + r'''
for name in DATASETS:
    K = v2.DATASETS[name]["events"]
    ht = v2.horizon_table(res[res.dataset == name], ["model"], K=K)
    ht.to_csv(rep / f"ctd_horizons_{name}.csv", index=False)
    print(name); display(ht)
'''),
])

# ================================================================== 04 encoding / representation
notebook("04_ablation_encoding_representation", """# 04 · Input encoding and shared representation (paper §2.2)

**Encoding** - the paper encodes numbers *without* discretisation (the token's embedding is
scaled by the standardised value). Compared on METABRIC and SUPPORT with the same model:
`continuous` (the paper), `quantile` (rank-normalised values, same embedding) and
`discretised` (each number cut into bins, one token per bin).

**Representation** - how the transformer's hidden states become one patient vector:
token pooling over layers (sum / mean / concatenate / BERT [CLS]) x pooling over tokens
(max / mean) x layers (all / last).""", SEEDS_OPT + '''
DATASETS = ["metabric", "support"]
ENCODINGS = ["continuous", "quantile", "discretised"]
# (token_emb, sentence_emb, layers): token_emb 2=sum 3=mean 4=concat 5=[CLS];
# sentence_emb 2=max 3=mean; layers "all" or "last"
REPRESENTATIONS = [(t, s, l) for t in (2, 3, 4) for s in (2, 3) for l in ("all", "last")] + [(5, 1, "all")]
''', [
    code(HELPERS + r'''
GROUP = {"metabric": "survtrace_metabric", "support": "survtrace_support"}
def enc_overrides(ds, enc):
    """(overrides, model-hub folder) of one input encoding."""
    if enc == "continuous":
        return [f"dataset={ds}_numeric"], f"{ds}_numeric"
    if enc == "quantile":
        return [f"dataset={ds}_quantile", "data.parse.scale_method=quantile"], f"{ds}_quantile"
    return [f"data/load={ds}", f"data/parse={ds}", f"tokenizers={ds}", f"dataset={ds}"], ds
'''),
    md("## Encodings"), code(r'''
from scripts.runner import Run
if SMOKE_TEST:
    SEEDS, REPRESENTATIONS = SEEDS[:1], REPRESENTATIONS[:2]
runs = []
for ds in DATASETS:
    for enc in ENCODINGS:
        ov, hub = enc_overrides(ds, enc)
        r.prepare(f"{GROUP[ds]}/survival", ov, label=f"{ds}_{enc}", split_seeds=SEEDS)
        for s in SEEDS:
            runs.append(Run(f"{ds}__enc_{enc}__s{s}", "finetune", f"{GROUP[ds]}/survival",
                            ov + v2.V2 + ["tasks=v2_survival"] + tuned("ours", ds) + [f"seed={s}", f"split_seed={s}"],
                            hub, {"dataset": ds, "model": "ours", "variant": f"enc_{enc}", "seed": s, "stage": "encoding"}))
print(len(runs), "encoding runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Representations"), code(r'''
runs = []
v2.prepare(r, DATASETS, SEEDS, REPO)
for ds in DATASETS:
    for t, s_emb, layers in REPRESENTATIONS:
        extra = tuned("ours", ds) + [f"token_emb={t}", f"sentence_emb={s_emb}"]
        if layers == "last":
            # hidden-state indices run 0 (embeddings) .. number of layers; -1 is rejected
            n_layers = TUNED[ds]["ours"]["transformer_num_hidden_layers"]
            extra.append(f"select_hidden_layers=[{n_layers}]")
        runs += v2.runs("ours", ds, SEEDS, REPO, extra=extra, tag=f"rep_t{t}_s{s_emb}_{layers}",
                        info={"stage": "representation", "token_emb": t, "sentence_emb": s_emb, "layers": layers})
print(len(runs), "representation runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Tables"), code(FINISH),
])

# ================================================================== 05 losses
notebook("05_ablation_losses", """# 05 · Every loss combination of the survival head (paper §2.3)

Our model always uses L_PCH. On top of it, with the weights and sigmas tuned in 01:

| | single event | multiple events |
|---|---|---|
| S1 L_PCH | yes | yes |
| S2 + L_rank (Eq. 4) | yes | yes |
| S3 + L_mul (Eq. 5) | - | yes |
| S4 + L_rank + L_mul | - | yes |

These runs also report the time error of the survival head's E[T] under the three tails of §2.4,
which notebook 06 uses as the "survival head only" row.""", SEEDS_OPT + '''
DATASETS = ["metabric", "support", "ebmt", "hsa_synthetic", "deephit_synthetic"]
''', [
    code(HELPERS),
    md("## Runs"), code(r'''
if SMOKE_TEST:
    SEEDS, DATASETS = SEEDS[:1], ["metabric", "ebmt"]
v2.prepare(r, DATASETS, SEEDS, REPO)
runs = []
for name in DATASETS:
    for rec, ov in surv_recipes(name).items():
        runs += v2.runs("ours", name, SEEDS, REPO, extra=tuned("ours", name) + loss_extra(name) + ov,
                        tag=rec, info={"stage": "losses", "recipe": rec})
print(len(runs), "runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Tables (paired against S1 on the same splits)"), code(FINISH + r'''
from scripts.runner import paired_vs
for metric in ["ctd_weighted_avg", "brier_survtrace_weighted_avg", "ibs"]:
    t = paired_vs(res, metric, by="variant", baseline="S1", pair_on="seed", within=["dataset"])
    t.to_csv(rep / f"paired_vs_S1_{metric}.csv", index=False)
    display(t)
'''),
])

# ================================================================== 06 regression head
notebook("06_regression_head", """# 06 · The regression ("time of event") head (paper §2.4)

| setup | what is trained |
|---|---|
| survival head only | rows of notebook 05: E[T] from S(t), closed by the truncated, T_max (Hu et al.) or linear (Haider et al.) tail |
| regression head only | R1 MAE on observed events · R2 best-guess MAE (Eq. 6) · R3 R2 + L_MM with t_q from Eq. 1 · R4 R2 + L_MM with the best-guess t_q |
| both heads | every survival recipe S x every regression recipe R |

L_MM needs >= 2 events, so R3/R4 only exist on the multi-event datasets. Regression-only
models select checkpoints on validation MAE-margin; models with a survival head on
validation C_td. Reported: MAE (uncensored / hinge / margin / pseudo-observation) and
Harrell's C of the predicted times, plus every survival metric where there is a survival head.""",
    SEEDS_OPT + '''
DATASETS = ["metabric", "support", "ebmt", "hsa_synthetic", "deephit_synthetic"]
''', [
    code(HELPERS),
    md("## Runs"), code(r'''
if SMOKE_TEST:
    SEEDS, DATASETS = SEEDS[:1], ["metabric", "ebmt"]
v2.prepare(r, DATASETS, SEEDS, REPO)
REG_ONLY = ["tasks=v2_regression", "selection_metric=eval_reg_mae_margin", "selection_greater=false"]
runs = []
for name in DATASETS:
    base = tuned("ours", name) + loss_extra(name)
    for rr, rov in reg_recipes(name).items():
        runs += v2.runs("ours", name, SEEDS, REPO, extra=base + REG_ONLY + rov, tag=f"reg_{rr}",
                        info={"stage": "regression", "setup": "regression only", "R": rr})
        for sr, sov in surv_recipes(name).items():
            runs += v2.runs("ours", name, SEEDS, REPO, extra=base + ["tasks=v2_survival_regression"] + sov + rov,
                            tag=f"both_{sr}x{rr}",
                            info={"stage": "regression", "setup": "both heads", "S": sr, "R": rr})
print(len(runs), "runs")
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Tables"), code(FINISH + r'''
cols = ["reg_mae_uncensored", "reg_mae_hinge", "reg_mae_margin", "reg_mae_pseudo_obs", "reg_harrell",
        "mae_linear_pseudo_obs", "mae_tmax_pseudo_obs", "mae_trunc_pseudo_obs", "ctd_weighted_avg", "ibs"]
t = v2.table(res, ["dataset", "variant"], cols)
t.to_csv(rep / "regression_table.csv", index=False)
display(t)
'''),
])

# ================================================================== 07 LLM
notebook("07_llm_end_to_end", """# 07 · End-to-end LLM (paper §3)

The patient record is the question; the answer is one token per time unit (`[N]` no event,
`[C]` censored, `[Ek]` event k). Three backbones x three training objectives:

* backbones: **GPT-2** (124M, pretrained), **Qwen2.5-0.5B** (pretrained, LoRA), and a small
  GPT-2 trained **from scratch** on the records;
* objectives: **ce** - token cross-entropy (Eq. 8); **deephit** - the DeepHit-style
  likelihood the paper proposes; **deephit+illegal** - plus the illegal-sequence penalty.

Curves are read from the answer-token probabilities and scored with exactly the metrics
of every other model. A T4 GPU is needed (fp16).""", SEEDS_OPT + '''
DATASETS = ["metabric", "support", "ebmt"]
BACKBONES = {
    "gpt2": ["llm_init=pretrained", "llm_model=gpt2", "llm_learning_rate=5e-5"],
    "qwen0.5b": ["llm_init=pretrained", "llm_model=Qwen/Qwen2.5-0.5B", "llm_lora_r=16",
                 "llm_learning_rate=2e-4", "llm_batch_size=16", "llm_grad_accum=2"],
    "scratch": ["llm_init=scratch", "llm_model=gpt2", "llm_learning_rate=5e-4"],
}
OBJECTIVES = {"ce": ["llm_loss=ce"], "deephit": ["llm_loss=deephit"],
              "deephit+illegal": ["llm_loss=deephit", "llm_illegal_coeff=1.0"]}
LLM_EPOCHS = 20
''', [
    md("## Runs"), code(r'''
if SMOKE_TEST:
    SEEDS, DATASETS = SEEDS[:1], DATASETS[:1]
    BACKBONES = {"scratch": BACKBONES["scratch"]}
v2.prepare(r, DATASETS, SEEDS, REPO)
runs = []
for name in DATASETS:
    for bb, bov in BACKBONES.items():
        for obj, oov in OBJECTIVES.items():
            runs += v2.runs("llm", name, SEEDS, REPO, extra=bov + oov + [f"llm_epochs={LLM_EPOCHS}"],
                            tag=f"llm_{bb}_{obj}", info={"stage": "llm", "backbone": bb, "objective": obj})
print(len(runs), "runs")
# one LLM at a time: two 0.5B models do not fit next to each other on a T4
r.workers = 1
r.run_many(runs, time_budget_min=TIME_BUDGET_MIN)
'''),
    md("## Tables"), code(FINISH),
], needs_tuned=False)

# ================================================================== 08 report
notebook("08_report", """# 08 · Report: all version-2 tables

Collects the results of notebooks 01-07 (on Kaggle: add their outputs as inputs; locally
they are read from `results/version_2/`) and writes every table as CSV plus one
`REPORT.md`. No training.""", '''
NOTEBOOKS = ["02_tabular_benchmark", "03_multievent_benchmark", "04_ablation_encoding_representation",
             "05_ablation_losses", "06_regression_head", "07_llm_end_to_end"]
''', [
    md("## Collect"), code(r'''
from scripts.runner import Runner
frames = {}
for nb in NOTEBOOKS:
    dirs = [RESULTS_ROOT / nb] + [d for root in INPUT_ROOTS for d in root.rglob(nb) if d.is_dir()]
    for d in dirs:
        if any(d.glob("*.meta.json")):
            df = Runner(repo=REPO, results_dir=d, verbose=False).collect()
            df["notebook"] = nb
            frames[nb] = df
            break
    print(f"{nb:38s} {len(frames.get(nb, [])):5d} runs")
allres = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
out = r.results_dir
allres.to_csv(out / "all_runs.csv", index=False)
'''),
    md("## Tables"), code(r'''
lines = ["# Version 2 results", "", f"code commit {COMMIT}", ""]
def add(title, df, fname):
    if df is None or not len(df):
        return
    df.to_csv(out / fname, index=False)
    lines.extend([f"## {title}", "", df.to_markdown(index=False), ""])
    display(Markdown(f"### {title}")); display(df)

if "02_tabular_benchmark" in frames:
    f = frames["02_tabular_benchmark"]
    add("§4.1 C_td at 25/50/75% (SurvTRACE Table 2 layout)", v2.horizon_table(f, ["dataset", "model"]), "t_tabular_horizons.csv")
    add("§4.1 all metrics", v2.table(f, ["dataset", "model"]), "t_tabular_all.csv")
if "03_multievent_benchmark" in frames:
    f = frames["03_multievent_benchmark"]
    for name in sorted(f.dataset.unique()):
        K = v2.DATASETS[name]["events"]
        add(f"§4.3 {name}: C_td per event", v2.horizon_table(f[f.dataset == name], ["model"], K=K), f"t_multi_{name}.csv")
    add("§4.3 all metrics", v2.table(f, ["dataset", "model"]), "t_multi_all.csv")
if "04_ablation_encoding_representation" in frames:
    add("§2.2 encoding and representation", v2.table(frames["04_ablation_encoding_representation"], ["dataset", "variant"]), "t_encoding_representation.csv")
if "05_ablation_losses" in frames:
    add("§2.3 loss combinations", v2.table(frames["05_ablation_losses"], ["dataset", "variant"]), "t_losses.csv")
if "06_regression_head" in frames:
    add("§2.4 regression head", v2.table(frames["06_regression_head"], ["dataset", "variant"]), "t_regression.csv")
if "07_llm_end_to_end" in frames:
    add("§3 end-to-end LLM", v2.table(frames["07_llm_end_to_end"], ["dataset", "variant"]), "t_llm.csv")
(out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("zip:", r.zip())
'''),
], needs_tuned=False)
