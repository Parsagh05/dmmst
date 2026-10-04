"""Write the two published synthetic multi-event datasets in the generic layout of
``sat.data.dataset.parse_multievent`` (id, x_*, event1..K, duration1..K + meta.json),
so they go through exactly the same pipeline and model config as EBMT.

* ``deephit_synthetic`` - Lee et al. (AAAI 2018), "SYNTHETIC": 30,000 patients,
  12 covariates, two competing risks (exponential hitting times), 50% censored.
  Source: ``synthetic_comprisk.csv`` from the authors' repository. Competing risks:
  the observed event's indicator is 1 and every event shares the observed time.
  The file also holds each patient's latent ``true_time`` / ``true_label``; they are
  kept in ``truth.csv`` (not used as features).
* ``hsa_synthetic_me`` - Tjandra et al. (AAAI 2021), "Synthetic": 5,000 patients,
  15 covariates, two events that can both occur in any order. Source: the copy shipped
  with the original sat repository (``data/hsa-synthetic/simulated_data.csv``).

    python scripts/prepare_public_synthetic.py            # both
    python scripts/prepare_public_synthetic.py --only deephit_synthetic
"""

__authors__ = ["Parsa"]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEEPHIT_URL = (
    "https://raw.githubusercontent.com/chl8856/DeepHit/master/"
    "sample%20data/SYNTHETIC/synthetic_comprisk.csv"
)
HSA_PATH = "data/hsa-synthetic/simulated_data.csv"
# zero durations would be dropped by the label transform (it keeps durations > 0);
# the same convention as scripts/prepare_ebmt.py
MIN_DURATION = 0.5


def _meta(out: pd.DataFrame, source: str, names, regime: str) -> dict:
    K = len(names)
    ev = out[[f"event{k}" for k in range(1, K + 1)]]
    return {
        "source": source,
        "num_events": K,
        "num_features": int(sum(c.startswith(("x_", "c_")) for c in out.columns)),
        "event_names": list(names),
        "max_time": float(out[[f"duration{k}" for k in range(1, K + 1)]].values.max()),
        "n": int(len(out)),
        "event_rates": ev.mean().round(4).tolist(),
        "frac_censored_all": float((ev.sum(axis=1) == 0).mean()),
        "frac_subjects_multiple_events": float((ev.sum(axis=1) > 1).mean()),
        "scenario": {"regime": regime},
    }


def build_deephit(raw: pd.DataFrame):
    feats = [c for c in raw.columns if c.startswith("feature")]
    out = pd.DataFrame({"id": np.arange(len(raw))})
    for c in feats:
        out[f"x_{c.replace('feature', '')}"] = raw[c].astype(float).values
    t = raw["time"].astype(float).clip(lower=MIN_DURATION).values
    for k in (1, 2):
        out[f"event{k}"] = (raw["label"] == k).astype(int).values
        out[f"duration{k}"] = t
    truth = pd.DataFrame(
        {"id": out["id"], "true_time": raw["true_time"], "true_label": raw["true_label"]}
    )
    return out, _meta(out, DEEPHIT_URL, ["event_1", "event_2"], "competing"), truth


def build_hsa(raw: pd.DataFrame):
    out = pd.DataFrame({"id": raw["id"].astype(int).values})
    for c in [c for c in raw.columns if c.startswith("x_")]:
        out[c] = raw[c].astype(float).values
    for k in (1, 2):
        out[f"event{k}"] = raw[f"event{k}"].astype(int).values
        out[f"duration{k}"] = raw[f"duration{k}"].astype(float).clip(lower=MIN_DURATION).values
    return out, _meta(out, HSA_PATH, ["event_1", "event_2"], "non-competing"), None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["deephit_synthetic", "hsa_synthetic_me"])
    ap.add_argument("--deephit-source", default=DEEPHIT_URL)
    ap.add_argument("--hsa-source", default=HSA_PATH)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)

    jobs = {
        "deephit_synthetic": lambda: build_deephit(pd.read_csv(args.deephit_source)),
        "hsa_synthetic_me": lambda: build_hsa(pd.read_csv(args.hsa_source)),
    }
    for name, job in jobs.items():
        if args.only and name != args.only:
            continue
        df, meta, truth = job()
        out = Path(args.data_dir) / name
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "data.csv", index=False)
        if truth is not None:
            truth.to_csv(out / "truth.csv", index=False)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(name, json.dumps(meta))


if __name__ == "__main__":
    main()
