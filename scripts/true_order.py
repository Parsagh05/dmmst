"""Within-subject event ordering scored against the TRUE order (simulated data only).

``within_ctd`` compares predicted CIFs at the observed time of the earlier event. A loss
trained on exactly those pairs (L_mul) can push it above the oracle, so it cannot
support a claim on its own. The simulators store every subject's latent (uncensored)
event times, so ordering can be scored against the truth with proper scores:

    p_ab = P(T_a < T_b | x) = int_0^H f_a(t) S_b(t) dt + 1/2 S_a(H) S_b(H)

from each model's per-event survival curves on the duration-cut grid (events taken as
independent given x - exact for the simulator's non-competing regime without frailty;
beyond the horizon H the order is unknown and split evenly). Then, over all pairs of
events of every test subject with distinct latent times,

    true_order_c        P(p_ab > 1/2 agrees with the true order), ties 1/2
    true_order_brier    mean (p_ab - 1[T_a < T_b])^2       (proper, lower is better)
    true_order_logloss  mean cross-entropy                   (proper, lower is better)

A model cannot beat the oracle on the proper scores in expectation.
The runner calls ``score_run`` after every run on a dataset with ``truth.npz``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_FILES = ("survival{k}.csv", "risk{k}.csv")


def _curves(run_dir: Path, K: int):
    """(ids, surv (n, K, G)) from a run directory, or None."""
    for pattern in GRID_FILES:
        files = [run_dir / pattern.format(k=k) for k in range(K)]
        if all(f.is_file() for f in files):
            frames = [pd.read_csv(f) for f in files]
            ids = frames[0]["id"].values
            vals = [fr.drop(columns=["id"]).values.astype(float) for fr in frames]
            s = np.stack(vals, axis=1)
            return ids, (1.0 - s if pattern.startswith("risk") else s)
    return None


def order_probabilities(surv: np.ndarray) -> np.ndarray:
    """p[i, a, b] = P(T_a < T_b) from survival on a grid starting at t = 0."""
    s = np.clip(surv, 0.0, 1.0)
    if not np.allclose(s[:, :, 0], 1.0):
        s = np.concatenate([np.ones(s.shape[:2] + (1,)), s], axis=2)
    dF = np.clip(s[:, :, :-1] - s[:, :, 1:], 0.0, None)  # (n, K, G-1)
    s_mid = 0.5 * (s[:, :, :-1] + s[:, :, 1:])
    p = np.einsum("nag,nbg->nab", dF, s_mid)
    p += 0.5 * s[:, :, -1][:, :, None] * s[:, :, -1][:, None, :]
    return np.clip(p, 1e-6, 1 - 1e-6)


def score(p: np.ndarray, latent: np.ndarray) -> dict:
    """p (n, K, K) predicted P(a before b); latent (n, K) true event times."""
    n, K = latent.shape
    a, b = np.triu_indices(K, k=1)
    pa = p[:, a, b]
    y = (latent[:, a] < latent[:, b]).astype(float)
    ok = latent[:, a] != latent[:, b]
    pa, y = pa[ok], y[ok]
    conc = np.where(pa == 0.5, 0.5, ((pa > 0.5) == (y == 1)).astype(float))
    return {
        "true_order_c": float(conc.mean()),
        "true_order_brier": float(np.mean((pa - y) ** 2)),
        "true_order_logloss": float(-np.mean(y * np.log(pa) + (1 - y) * np.log(1 - pa))),
        "true_order_pairs": int(ok.sum()),
    }


def score_run(run_dir, truth_file, K: int):
    """Scores for one finished run, or {} when it saved no curves."""
    got = _curves(Path(run_dir), K)
    if got is None:
        return {}
    ids, surv = got
    truth = np.load(truth_file)
    rows = np.searchsorted(truth["ids"], ids)
    return score(order_probabilities(surv), truth["latent_times"][rows])


def add_to_metrics(result_json: Path, scores: dict):
    """Merge the scores into a copied metrics.json (nested test/{metric: {mean}} format)."""
    if not scores:
        return
    d = json.loads(result_json.read_text())
    d.setdefault("test", {}).update({k: {"mean": v, "variance": 0.0, "sd": 0.0} for k, v in scores.items()})
    result_json.write_text(json.dumps(d, indent=2))
