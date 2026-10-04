"""Phase-0 protocol check for every version-2 dataset (runs locally, no training).

For each dataset it prepares the data for a few split seeds and asserts:

1. the patient count matches the published one;
2. each split is 60/10/30 (+-0.5 points), the three splits are disjoint and cover every
   patient, and different split seeds give different splits;
3. the outcome-derived files (``split_<s>/transformed_train_labels.csv`` and
   ``imp_sample.csv``) contain exactly the training rows of that split - nothing from
   validation or test (they feed the IPCW weights and the Kaplan-Meier best guesses);
4. the evaluation horizons are the 25/50/75% quantiles of each event's observed times
   over the full data (SurvTRACE), and for a single event they equal the inner cuts.

    python scripts/verify_protocol.py                    # all datasets, seeds 0 1 2
    python scripts/verify_protocol.py --datasets metabric --seeds 0 1
"""

__authors__ = ["Parsa"]

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.runner import Runner, dataset_overrides  # noqa: E402

# name -> (experiment, overrides-builder, model-hub dir, published n, source of n)
DATASETS = {
    "metabric": ("survtrace_metabric/survival", lambda: [], "metabric_numeric", 1904,
                 "SurvTRACE Table 1 / pycox"),
    "support": ("survtrace_support/survival", lambda: [], "support_numeric", 8873,
                "SurvTRACE Table 1 / pycox"),
    "ebmt4ev": ("multievent/survival", lambda: dataset_overrides("ebmt4ev", str(REPO)),
                "ebmt4ev", 2279, "mstate::ebmt4"),
    "hsa_synthetic_me": ("multievent/survival",
                         lambda: dataset_overrides("hsa_synthetic_me", str(REPO)),
                         "hsa_synthetic_me", 5000, "Tjandra et al. 2021, Table 1"),
    "deephit_synthetic": ("multievent/survival",
                          lambda: dataset_overrides("deephit_synthetic", str(REPO)),
                          "deephit_synthetic", 30000, "Lee et al. 2018 (DeepHit), Table 1"),
}


def _splits(experiment, overrides, seed):
    """The exact split finetune.py uses for this split_seed."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from omegaconf import OmegaConf

    from sat.data.splitter import StreamingKFoldSplitter
    from sat.utils import config

    if not OmegaConf.has_resolver("sum"):
        config.Config()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
        cfg = compose("finetune", overrides=[f"experiments={experiment}", *overrides,
                                              f"split_seed={seed}"])
    sp = StreamingKFoldSplitter(
        id_field=cfg.data.id_col, k=None, val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio, test_split_strategy="hash",
        split_names=cfg.data.splits, split_seed=seed,
    )
    ds = sp.load_split(cfg=cfg.data.load)
    return cfg, {name: ds[name] for name in cfg.data.splits}


def _outcomes(split, cfg):
    e = np.asarray(split[cfg.data.event_col], dtype=float)
    d = np.asarray(split[cfg.data.duration_col], dtype=float)
    return e.reshape(len(e), -1), d.reshape(len(d), -1)


def check(name, seeds, results_dir):
    experiment, ov_fn, hub, n_pub, source = DATASETS[name]
    ov = ov_fn()
    r = Runner(repo=str(REPO), results_dir=results_dir, workers=1, verbose=False)
    r.prepare(experiment, ov, label=name, split_seeds=seeds)
    base = REPO / "data" / "model-hub" / hub
    report, failures = {"dataset": name}, []

    members = {}
    for s in seeds:
        cfg, sp = _splits(experiment, ov, s)
        ids = {k: set(np.asarray(v[cfg.data.id_col]).tolist()) for k, v in sp.items()}
        n = sum(len(v) for v in ids.values())
        tr, va, te = (ids[k] for k in cfg.data.splits)
        if n != n_pub:
            failures.append(f"seed {s}: n={n}, published {n_pub} ({source})")
        if tr & va or tr & te or va & te:
            failures.append(f"seed {s}: splits overlap")
        fr = [len(x) / n for x in (tr, va, te)]
        if any(abs(a - b) > 0.005 for a, b in zip(fr, (0.6, 0.1, 0.3))):
            failures.append(f"seed {s}: fractions {np.round(fr, 3)}")
        members[s] = te
        report[f"seed{s}_fractions"] = [round(x, 3) for x in fr]

        # outcome files == the training rows of this split
        e_tr, d_tr = _outcomes(sp[cfg.data.splits[0]], cfg)
        lab = pd.read_csv(base / f"split_{s}" / "transformed_train_labels.csv")
        K = e_tr.shape[1]
        for k in range(K):
            a = np.sort(lab[f"duration_event{k + 1}"].values)
            b = np.sort(d_tr[:, k])
            if len(a) != len(b) or not np.allclose(a, b, atol=1e-4):
                failures.append(f"seed {s}: training labels of event {k + 1} != train split")
            if int(lab[f"event{k + 1}"].sum()) != int(e_tr[:, k].sum()):
                failures.append(f"seed {s}: event {k + 1} count != train split")
        imp = pd.read_csv(base / f"split_{s}" / "imp_sample.csv", header=None)[0].values
        exp_imp = np.concatenate([[(~e_tr.astype(bool).any(1)).mean()], e_tr.mean(0)])
        if not np.allclose(imp, exp_imp, atol=1e-6):
            failures.append(f"seed {s}: imp_sample {imp} != train proportions {exp_imp}")

    if len(seeds) > 1 and all(members[seeds[0]] == members[s] for s in seeds[1:]):
        failures.append("test split identical across seeds")
    report["test_overlap_seed0_seed1"] = (
        round(len(members[seeds[0]] & members[seeds[1]]) / len(members[seeds[0]]), 3)
        if len(seeds) > 1 else None
    )

    # horizons: full-data quantiles per event
    cfg, sp = _splits(experiment, ov, seeds[0])
    e_all = np.vstack([_outcomes(v, cfg)[0] for v in sp.values()])
    d_all = np.vstack([_outcomes(v, cfg)[1] for v in sp.values()])
    hz = {int(k): v for k, v in json.loads((base / "event_horizons.json").read_text()).items()}
    for k in range(e_all.shape[1]):
        want = np.quantile(d_all[e_all[:, k] > 0, k], [0.25, 0.5, 0.75])
        if not np.allclose(hz[k], want, rtol=1e-6):
            failures.append(f"horizons event {k + 1}: {hz[k]} != {want}")
    cuts = pd.read_csv(base / "duration_cuts.csv", header=None)[0].values
    if e_all.shape[1] == 1 and len(cuts) == 5 and not np.allclose(cuts[1:-1], hz[0], rtol=1e-6):
        failures.append(f"single-event cuts {cuts[1:-1]} != horizons {hz[0]}")
    report["horizons"] = {k + 1: np.round(v, 2).tolist() for k, v in hz.items()}
    report["event_rates_full"] = np.round(e_all.mean(0), 3).tolist()
    report["n"] = int(len(e_all))
    report["failures"] = failures
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--results-dir", default="results/verify_protocol")
    args = ap.parse_args(argv)
    ok = True
    for name in args.datasets:
        rep = check(name, args.seeds, args.results_dir)
        ok &= not rep["failures"]
        print(json.dumps(rep, indent=1))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
