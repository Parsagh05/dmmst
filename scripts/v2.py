"""Shared plan for the version-2 notebooks (notebooks/version_2): datasets, how every
model is launched, search grids, loss recipes, tuned settings and result tables.

Keeping this in the repo (not in the notebooks) means every notebook builds runs the
same way, and a fix here reaches all of them through the git clone.
"""

__authors__ = ["Parsa"]

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.runner import Run, dataset_overrides, grid

# ---------------------------------------------------------------- datasets
# name -> experiment group, model-hub folder, number of events, how to build the data
DATASETS = {
    "metabric": dict(exp="survtrace_metabric", hub="metabric_numeric", events=1, build=None),
    "support": dict(exp="survtrace_support", hub="support_numeric", events=1, build=None),
    "ebmt": dict(exp="multievent", hub="ebmt4ev", events=4,
                 build=["scripts/prepare_ebmt.py", "--no-composite"]),
    "hsa_synthetic": dict(exp="multievent", hub="hsa_synthetic_me", events=2,
                          build=["scripts/prepare_public_synthetic.py", "--only", "hsa_synthetic_me"]),
    "deephit_synthetic": dict(exp="multievent", hub="deephit_synthetic", events=2,
                              build=["scripts/prepare_public_synthetic.py", "--only", "deephit_synthetic"]),
}
SINGLE = ["metabric", "support"]
MULTI = ["ebmt", "hsa_synthetic", "deephit_synthetic"]


def data_overrides(name: str, repo) -> List[str]:
    spec = DATASETS[name]
    return dataset_overrides(spec["hub"], str(repo)) if spec["exp"] == "multievent" else []


def ensure_data(name: str, repo) -> None:
    spec = DATASETS[name]
    if spec["build"] and not (Path(repo) / "data" / spec["hub"] / "meta.json").is_file():
        subprocess.run([sys.executable, *spec["build"]], cwd=repo, check=True,
                       stdout=subprocess.DEVNULL)


def prepare(runner, names, seeds, repo) -> None:
    """Build the data and, for every split seed, the training-split label files."""
    for name in names:
        ensure_data(name, repo)
        runner.prepare(f"{DATASETS[name]['exp']}/survival", data_overrides(name, repo),
                       label=name, split_seeds=list(seeds))


# ---------------------------------------------------------------- models
V2 = ["tasks/losses=v2", "tasks/metrics=v2"]
NEURAL_HEADS = {"deephit": "deephit_paper", "dsm": "dsm_paper", "mensa": "mensa_paper"}
CLASSICAL = ("cox", "rsf", "deepsurv", "pchazard")


def launch(model: str, name: str, repo) -> tuple:
    """(script, experiment, overrides) that start `model` on dataset `name`."""
    exp = DATASETS[name]["exp"]
    data = data_overrides(name, repo)
    if model == "ours":
        return "finetune", f"{exp}/survival", data + V2 + ["tasks=v2_survival"]
    if model in NEURAL_HEADS:
        return "finetune", f"{exp}/{NEURAL_HEADS[model]}", data + ["tasks/metrics=v2"]
    if model in CLASSICAL:
        return "baselines", f"{exp}/survival", data + [f"baseline={model}"]
    if model == "survtrace":
        return "survtrace_official", f"{exp}/survival", data
    if model == "llm":
        return "llm", f"{exp}/survival", data
    raise ValueError(model)


def runs(model: str, name: str, seeds, repo, extra: Optional[List[str]] = None,
         tag: Optional[str] = None, info: Optional[Dict] = None) -> List[Run]:
    """One Run per split seed (seed s = initialisation s and split s)."""
    script, exp, ov = launch(model, name, repo)
    tag = tag or model
    return [Run(f"{name}__{tag}__s{s}", script, exp,
                ov + list(extra or []) + [f"seed={s}", f"split_seed={s}"],
                DATASETS[name]["hub"],
                {"dataset": name, "model": model, "variant": tag, "seed": s} | (info or {}))
            for s in seeds]


# ---------------------------------------------------------------- search spaces
# SurvTRACE's grid (Wang & Sun 2022, Sec. 5.1.4) for every transformer.
TRANSFORMER_GRID = {
    "learning_rate": [1e-4, 1e-3],
    "weight_decay": [1e-3, 1e-4, 0.0],
    "transformer_num_hidden_layers": [2, 3, 4],
    "transformer_hidden_size": [8, 16],
    "transformer_intermediate_size": [32, 64],
    "transformer_num_attention_heads": [1, 2, 4],
}
SURVTRACE_GRID = {
    "st_learning_rate": [1e-4, 1e-3],
    "st_weight_decay": [1e-3, 1e-4, 0.0],
    "st_num_hidden_layers": [2, 3, 4],
    "st_hidden_size": [8, 16],
    "st_intermediate_size": [32, 64],
    "st_num_attention_heads": [1, 2, 4],
}
BASELINE_GRIDS = {
    "cox": {"bl_cox_alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0]},
    "rsf": {"bl_rsf_n_estimators": [100, 300], "bl_rsf_min_samples_leaf": [3, 10, 30],
            "bl_rsf_max_features": ["sqrt", 0.5]},
    "deepsurv": {"bl_nn_nodes": [32, 64, 128], "bl_nn_layers": [1, 2], "bl_nn_dropout": [0.0, 0.1, 0.3],
                 "bl_nn_lr": [1e-3, 1e-2]},
    "pchazard": {"bl_nn_nodes": [32, 64, 128], "bl_nn_layers": [1, 2], "bl_nn_dropout": [0.0, 0.1, 0.3],
                 "bl_nn_lr": [1e-3, 1e-2]},
}
TUNE_GRIDS = {"ours": TRANSFORMER_GRID, "survtrace": SURVTRACE_GRID, **BASELINE_GRIDS}


def sample_grid(space: Dict[str, list], n: Optional[int], seed: int = 0) -> List[Dict]:
    """All configurations, or n of them drawn without replacement (fixed seed) when the
    full grid is too large (SurvTRACE's grid has 216 points)."""
    configs = grid(space)
    if n is None or n >= len(configs):
        return configs
    return random.Random(seed).sample(configs, n)


# ---------------------------------------------------------------- loss recipes
def survival_recipes(multi: bool, rank: Dict, mul: Dict) -> Dict[str, List[str]]:
    """S1-S4. rank / mul: tuned {"coeff", "sigma"} of L_rank / L_mul."""
    r = [f"v2_rank_coeff={rank['coeff']}", f"v2_rank_sigma={rank['sigma']}"]
    m = [f"v2_mul_coeff={mul['coeff']}", f"v2_mul_sigma={mul['sigma']}"]
    rec = {"S1": [], "S2": r}
    if multi:
        rec.update({"S3": m, "S4": r + m})
    return rec


def regression_recipes(multi: bool, mm_coeff: float) -> Dict[str, List[str]]:
    """R1-R4 of the paper's regression head (Sec. 2.4)."""
    rec = {"R1": ["v2_l1_type=uncensored"], "R2": ["v2_l1_type=margin"]}
    if multi:
        rec.update({
            "R3": ["v2_l1_type=margin", f"v2_mm_coeff={mm_coeff}", "v2_mm_t_q=observed"],
            "R4": ["v2_l1_type=margin", f"v2_mm_coeff={mm_coeff}", "v2_mm_t_q=best_guess"],
        })
    return rec


# ---------------------------------------------------------------- tuned settings
TUNED_FILE = "tuned.json"


def find_tuned(results_root, extra_roots=()) -> Path:
    """tuned.json of notebook 01: in this results folder or in an attached input."""
    cands = [Path(results_root) / "01_tuning" / TUNED_FILE]
    for root in extra_roots:
        cands += sorted(Path(root).rglob(TUNED_FILE))
    for c in cands:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "tuned.json from notebook 01 not found. On Kaggle: Add Input -> the output of "
        "01_tuning. Locally: run 01_tuning first.")


def load_tuned(path) -> Dict:
    return json.loads(Path(path).read_text())


def tuned_overrides(tuned: Dict, model: str, name: str) -> List[str]:
    """Settings chosen on validation in notebook 01. DeepHit / DSM / MENSA use the
    transformer backbone tuned for our model (same encoder, different head)."""
    key = "ours" if model in NEURAL_HEADS else model
    conf = tuned.get(name, {}).get(key, {})
    return [f"{k}={v}" for k, v in conf.items()]


# ---------------------------------------------------------------- tables
HEADLINE = [
    "ctd_weighted_avg", "brier_survtrace_weighted_avg", "ibs", "ibll", "antolini", "auc",
    "dcal_p", "onecal_p", "mae_linear_pseudo_obs", "mae_linear_margin",
    "reg_mae_pseudo_obs", "reg_mae_margin", "reg_harrell",
]


def table(df: pd.DataFrame, by: List[str], metrics=None, digits: int = 3) -> pd.DataFrame:
    """mean (std) over split seeds, the SurvTRACE reporting format."""
    metrics = [m for m in (metrics or HEADLINE) if m in df.columns]
    g = df.groupby(by, dropna=False)[metrics]
    mean, sd, n = g.mean(), g.std(), g.count()
    out = mean.copy().astype(object)
    for m in metrics:
        out[m] = [f"{a:.{digits}f} ({b:.{digits}f})" if c > 1 else (f"{a:.{digits}f}" if c else "")
                  for a, b, c in zip(mean[m], sd[m].fillna(0), n[m])]
    out["n"] = n[metrics[0]] if metrics else 0
    return out.reset_index()


def horizon_table(df: pd.DataFrame, by: List[str], K: int = 1, digits: int = 3) -> pd.DataFrame:
    """C_td at the 25/50/75% horizons per event: the layout of SurvTRACE's Tables 2-3."""
    cols = [f"ctd_{k}th_event_{q}" for k in range(K) for q in (0.25, 0.5, 0.75)]
    return table(df, by, cols, digits)


# ---------------------------------------------------------------- tuning runs
def tune_runs(model: str, name: str, configs: List[Dict], seeds, repo,
              extra: Optional[List[str]] = None, prefix: Optional[str] = None) -> List[Run]:
    """Runs for a list of configurations; tags <prefix>__c###__s<seed> so that
    scripts.runner.select_best(runner, prefix) picks the best on VALIDATION."""
    script, exp, ov = launch(model, name, repo)
    prefix = prefix or f"{name}__{model}__tune"
    out = []
    for i, conf in enumerate(configs):
        for s in seeds:
            out.append(Run(
                f"{prefix}__c{i:03d}__s{s}", script, exp,
                ov + list(extra or []) + [f"{k}={v}" for k, v in conf.items()]
                + [f"seed={s}", f"split_seed={s}"],
                DATASETS[name]["hub"],
                {"dataset": name, "model": model, "stage": "tune", "config_id": f"c{i:03d}",
                 "config": conf, "seed": s},
            ))
    return out
