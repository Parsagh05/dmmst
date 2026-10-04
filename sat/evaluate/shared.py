"""One scoring path for models that are not trained by finetune.py (Cox, RSF, DeepSurv,
PC-Hazard, official SurvTRACE, the LLM, the oracle).

The model supplies only its survival function; everything else - horizons, training-
split censoring weights, C_td / Brier (``score_at_horizons``) and the version-2 suite
(``V2Metrics``) - is the very code that scores the transformer, so every row of every
table has the same columns computed the same way.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from sat.evaluate.survtrace_metrics import (
    _structured,
    load_event_horizons,
    score_at_horizons,
)
from sat.evaluate.v2_metrics import V2Metrics

LABELS = [0.25, 0.5, 0.75]


def references(events: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """(n, K) events / durations -> the (n, 4K) reference layout of the metric modules."""
    events = np.asarray(events, dtype=float).reshape(len(events), -1)
    durations = np.asarray(durations, dtype=float).reshape(len(durations), -1)
    K = events.shape[1]
    ref = np.zeros((len(events), 4 * K))
    ref[:, K : 2 * K] = events
    ref[:, 3 * K :] = durations
    return ref


def evaluate_curves(
    cfg,
    surv_at: Callable[[np.ndarray], np.ndarray],
    refs: np.ndarray,
    time_to_event: Optional[np.ndarray] = None,
    fine_points: int = 200,
) -> Dict[str, float]:
    """surv_at(times) -> survival (n, K, len(times)) of the evaluated subjects."""
    K = int(cfg.data.num_events)
    save = cfg.data.label_transform.save_dir
    cuts_file = f"{save}/duration_cuts.csv"
    train_file = f"{cfg.data.label_transform.train_dir}/transformed_train_labels.csv"
    cuts = pd.read_csv(cuts_file, header=None, names=["c"], float_precision="round_trip").c.values
    train = pd.read_csv(train_file, header=0, float_precision="round_trip")
    per_event = (
        load_event_horizons(save, train_file, K) if cfg.get("per_event_horizons", False) else {}
    )

    out: Dict[str, float] = {}
    ctd, brier = [], []
    for k in range(K):
        times = np.asarray(per_event.get(k, cuts[1:-1]), dtype=float)
        labels = LABELS[: len(times)]
        et_train = _structured(train[f"event{k + 1}"] == 1, train[f"duration_event{k + 1}"])
        et_test = _structured(refs[:, K + k] > 0, refs[:, 3 * K + k])
        surv_h = np.asarray(surv_at(times))[:, k, :]
        m = score_at_horizons(et_train, et_test, 1.0 - surv_h, surv_h, times, labels, k)
        out.update(m)
        if f"ctd_{k}th_event" in m:
            ctd.append(m[f"ctd_{k}th_event"])
        if f"brier_{k}th_event" in m:
            brier.append(m[f"brier_{k}th_event"])
    out["ctd_weighted_avg"] = float(np.mean(ctd)) if ctd else 0.0
    if brier:
        out["brier_survtrace_weighted_avg"] = float(np.mean(brier))

    grid = np.linspace(0.0, float(cuts[-1]), fine_points)
    v2 = V2Metrics(cfg.data, cuts_file, train_file)
    v2.final = True
    preds = {"survival": np.asarray(surv_at(grid)), "grid": grid}
    if time_to_event is not None:
        preds["time_to_event"] = time_to_event
    out.update(v2.compute(preds, refs))
    return out


def write_metrics(out_dir, test: Dict[str, float], validation: Optional[Dict[str, float]] = None):
    """metrics.json in the layout finetune.py writes (mean / variance / sd per key)."""
    import json
    from pathlib import Path

    def wrap(d):
        return {"n": 1} | {k: {"mean": float(v), "variance": 0.0, "sd": 0.0} for k, v in d.items()}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"validation": wrap(validation if validation is not None else test), "test": wrap(test)}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=4))
