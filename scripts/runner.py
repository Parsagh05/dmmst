"""Experiment runner shared by the Kaggle notebooks and local runs.

Every run is a subprocess ``python -m sat.<script> experiments=<exp> ...`` with its
own ``modelname`` (= its tag), so parallel runs never overwrite each other. The
run's ``metrics.json`` is copied to ``<results_dir>/<tag>.json`` together with a
``<tag>.meta.json`` describing the run; a tag whose json already exists is skipped,
so a notebook can be re-run after a disconnect and continues where it stopped.

    from scripts.runner import Run, Runner, dataset_overrides
    r = Runner(repo=".", results_dir="results/lmul", workers=4)
    r.prepare("multievent/survival", dataset_overrides("sim_base"))
    r.run_many([Run("sim_base__nllpch__s0", "finetune", "multievent/survival",
                    dataset_overrides("sim_base") + ["seed=0", "split_seed=0"],
                    dataset="sim_base", info={"model": "ours", "recipe": "nllpch"})])
    df = r.collect(); summary(df, "ctd_weighted_avg", by=["dataset", "model", "recipe"])
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HEADLINE = [
    "ctd_weighted_avg",
    "brier_survtrace_weighted_avg",
    "mae_uncensored",
    "mae_margin",
    "within_ctd",
    "within_ctd_km",
    "within_ctd_gain",
    "within_ctd_subject",
    "within_subject_ipcw",
]
# lower is better for these; everything else higher is better
LOWER_IS_BETTER = ("brier", "mae", "loss")


@dataclass
class Run:
    tag: str
    script: str  # finetune | coxph | oracle
    experiment: str  # e.g. multievent/survival
    overrides: List[str]
    dataset: str  # directory under data/model-hub holding the run's output
    info: Dict = field(default_factory=dict)  # free-form labels (model, recipe, seed...)


def read_meta(dataset: str, repo: str = ".") -> dict:
    with (Path(repo) / "data" / dataset / "meta.json").open() as f:
        return json.load(f)


def dataset_overrides(dataset: str, repo: str = ".") -> List[str]:
    """Hydra overrides that point the generic multievent group at data/<dataset>/."""
    m = read_meta(dataset, repo)
    return [
        f"dataset={dataset}",
        f"me_num_events={m['num_events']}",
        f"me_num_features={m['num_features']}",
        f"survival_max_time={m['max_time']}",
    ]


def ehr_overrides(source: str, representation: str, repo: str = ".") -> List[str]:
    """Hydra overrides for the ehrseq group: data/<source>/ in one representation
    (sequence | bag | static). The model-hub dataset is <source>_<representation>."""
    m = read_meta(source, repo)
    return [
        f"ehr_source={source}",
        f"ehr_repr={representation}",
        f"ehr_max_tokens={m['max_tokens']}",
        f"me_num_events={m['num_events']}",
        f"me_num_features={m['num_features'][representation]}",
        f"survival_max_time={m['max_time']}",
    ]


class Runner:
    def __init__(
        self,
        repo: str = ".",
        results_dir: str = "results",
        workers: int = 1,
        threads_per_run: Optional[int] = None,
        smoke_epochs: Optional[int] = None,
        extra_overrides: Optional[List[str]] = None,
        gpu: Optional[str] = "0",
        keep_checkpoints: bool = False,
        verbose: bool = True,
    ):
        self.repo = Path(repo).resolve()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        self.workers = max(1, workers)
        self.smoke_epochs = smoke_epochs
        self.extra = list(extra_overrides or [])
        self.keep_checkpoints = keep_checkpoints
        self.verbose = verbose
        self._lock = threading.Lock()

        env = os.environ.copy()
        env.update(
            TOKENIZERS_PARALLELISM="false",
            HYDRA_FULL_ERROR="1",
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
        )
        if gpu is not None:
            # two visible GPUs make HF Trainer wrap the model in DataParallel, which
            # crashes on this model ("chunk expects at least a 1-dimensional tensor")
            env["CUDA_VISIBLE_DEVICES"] = gpu
        threads = threads_per_run or max(1, (os.cpu_count() or 2) // self.workers)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(threads)
        self.env = env

    # ------------------------------------------------------------------ running
    def _log(self, msg):
        if self.verbose:
            with self._lock:
                print(msg, flush=True)

    def sat(self, script, experiment, overrides=(), label=None, log_name=None):
        """Run one sat entry point; returns (ok, seconds, log text)."""
        cmd = [sys.executable, "-m", f"sat.{script}", f"experiments={experiment}", *overrides]
        t0 = time.time()
        p = subprocess.run(
            cmd, cwd=self.repo, env=self.env, text=True, errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        dt = time.time() - t0
        ok = p.returncode == 0
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", log_name or label or f"{script}_{experiment}")[:120]
        (self.results_dir / "logs" / f"{'' if ok else 'FAILED_'}{name}.log").write_text(
            " ".join(cmd) + "\n\n" + p.stdout, encoding="utf-8"
        )
        if not ok:
            tail = "\n".join(p.stdout.strip().splitlines()[-15:])
            self._log(f"  !! {label or name} FAILED ({dt:.0f}s)\n{tail}")
        return ok, dt, p.stdout

    def prepare(self, experiment: str, overrides=(), label: str = "prepare"):
        """prepare_data -> train_tokenizer -> train_labeltransform (once per dataset)."""
        for step in ("prepare_data", "train_tokenizer", "train_labeltransform"):
            ok, dt, _ = self.sat(step, experiment, overrides, label=f"{label}:{step}",
                                 log_name=f"{label}_{step}")
            if not ok:
                raise RuntimeError(f"{step} failed for {experiment} {overrides}; see logs/")
            self._log(f"  {label}: {step} ok ({dt:.0f}s)")

    def done(self, tag) -> bool:
        return (self.results_dir / f"{tag}.json").is_file()

    def run(self, r: Run) -> bool:
        if self.done(r.tag):
            return True
        ov = list(r.overrides) + [f"modelname={r.tag}"] + self.extra
        if self.smoke_epochs and r.script == "finetune":
            ov += [f"trainer.training_arguments.num_train_epochs={self.smoke_epochs}", "warmup_steps=0"]
        if self.smoke_epochs and r.script == "llm":
            ov += [f"llm_epochs={self.smoke_epochs}"]
        ok, dt, _ = self.sat(r.script, r.experiment, ov, label=r.tag, log_name=r.tag)
        out = self.repo / "data" / "model-hub" / r.dataset / r.tag
        if ok and (out / "metrics.json").is_file():
            shutil.copy2(out / "metrics.json", self.results_dir / f"{r.tag}.json")
            meta = asdict(r) | {"seconds": dt}
            (self.results_dir / f"{r.tag}.meta.json").write_text(json.dumps(meta, indent=2))
            self._log(f"  ok   {r.tag} ({dt:.0f}s)")
        elif ok:
            self._log(f"  !! {r.tag}: finished but no metrics.json in {out}")
            ok = False
        if not self.keep_checkpoints and out.is_dir():
            for ck in out.glob("checkpoint-*"):
                shutil.rmtree(ck, ignore_errors=True)
        return ok

    def run_many(self, runs: List[Run], time_budget_min: Optional[float] = None):
        """Run everything not yet done. Stops launching new runs after the budget."""
        todo = [r for r in runs if not self.done(r.tag)]
        self._log(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to go")
        if not todo:
            return
        t0 = time.time()
        failed = []
        # the first run alone: warms the HF datasets / evaluate caches, which are not
        # safe to populate from several processes at once
        if not self.run(todo[0]):
            failed.append(todo[0].tag)
        rest = todo[1:]
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = None
        bar = tqdm(total=len(rest), desc="runs") if (tqdm and rest) else None
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {}
            for r in rest:
                if time_budget_min and (time.time() - t0) / 60 > time_budget_min:
                    self._log("time budget reached - not launching more runs. Re-run to continue.")
                    break
                futures[ex.submit(self.run, r)] = r
            for f in as_completed(futures):
                if not f.result():
                    failed.append(futures[f].tag)
                if bar:
                    bar.update(1)
        if bar:
            bar.close()
        self._log(f"finished in {(time.time() - t0) / 60:.1f} min; {len(failed)} failed: {failed}")

    # --------------------------------------------------------------- collecting
    def collect(self, split: str = "test") -> pd.DataFrame:
        rows = []
        for f in sorted(self.results_dir.glob("*.json")):
            if f.name.endswith(".meta.json"):
                continue
            tag = f.stem
            meta_f = self.results_dir / f"{tag}.meta.json"
            if not meta_f.is_file():  # environment.json, tuned_backbone.json, ...
                continue
            meta = json.loads(meta_f.read_text())
            metrics = json.loads(f.read_text()).get(split, {})
            row = {"tag": tag, "dataset": meta.get("dataset")}
            row.update(meta.get("info", {}))
            for k, v in metrics.items():
                if isinstance(v, dict) and "mean" in v:
                    row[k] = v["mean"]
                elif isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def zip(self, name: Optional[str] = None) -> Path:
        name = name or f"{self.results_dir.name}.zip"
        out = self.results_dir.parent / name
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.results_dir.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(self.results_dir.parent))
        return out


# ------------------------------------------------------------------ statistics
def summary(df: pd.DataFrame, metrics, by, digits: int = 4) -> pd.DataFrame:
    """mean ± sd (n) per group."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(by, dropna=False)[metrics]
    mean, sd, n = g.mean(), g.std(), g.count()
    out = mean.copy().astype(object)
    for m in metrics:
        out[m] = [
            f"{a:.{digits}f} ± {b:.{digits}f} ({c})" if c > 1 else (f"{a:.{digits}f} (1)" if c else "")
            for a, b, c in zip(mean[m], sd[m].fillna(0), n[m])
        ]
    return out


def paired_vs(df: pd.DataFrame, metric: str, by: str, baseline, pair_on="seed", within=None):
    """Paired comparison of every level of `by` against `baseline`, matched on `pair_on`
    (same seed = same split and init). Returns delta mean ± sd, t-test and Wilcoxon p."""
    from scipy import stats

    within = [within] if isinstance(within, str) else list(within or [])
    rows = []
    groups = df.groupby(within) if within else [((), df)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        piv = sub.pivot_table(index=pair_on, columns=by, values=metric, aggfunc="mean")
        if baseline not in piv.columns:
            continue
        for level in piv.columns:
            if level == baseline:
                continue
            both = piv[[baseline, level]].dropna()
            if len(both) < 2:
                continue
            d = both[level] - both[baseline]
            t_p = stats.ttest_rel(both[level], both[baseline]).pvalue
            try:
                w_p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            except ValueError:
                w_p = np.nan
            better = (d < 0) if any(s in metric for s in LOWER_IS_BETTER) else (d > 0)
            rows.append(
                dict(zip(within, key))
                | {
                    by: level,
                    "vs": baseline,
                    "metric": metric,
                    "n": len(d),
                    "delta": d.mean(),
                    "delta_sd": d.std(),
                    "wins": f"{int(better.sum())}/{len(d)}",
                    "p_ttest": t_p,
                    "p_wilcoxon": w_p,
                }
            )
    return pd.DataFrame(rows)
