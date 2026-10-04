"""The shared evaluator (baselines) and the finetune path (SurvTRACEMetrics + V2Metrics)
must give the same numbers for the same model."""

import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from sat.evaluate.shared import evaluate_curves, references
from sat.evaluate.survtrace_metrics import SurvTRACEMetrics
from sat.evaluate.v2_metrics import V2Metrics


def _setup(tmp_path, per_event=False):
    rng = np.random.default_rng(5)
    n = 1500
    lam = 0.2 * np.exp(rng.normal(size=n))
    T = rng.exponential(1 / lam)
    C = rng.uniform(0, np.quantile(T, 0.95), size=n)
    t, e = np.minimum(T, C), (T <= C)
    tr, te = slice(0, 1000), slice(1000, None)
    cuts = np.concatenate([[0.0], np.quantile(t[e], [0.25, 0.5, 0.75]), [t.max()]])
    save = tmp_path / "hub"
    (save / "split_0").mkdir(parents=True)
    pd.DataFrame({"c": cuts}).to_csv(save / "duration_cuts.csv", index=False, header=False)
    pd.DataFrame({"event1": e[tr].astype(int), "duration_event1": t[tr]}).to_csv(
        save / "split_0" / "transformed_train_labels.csv", index=False)
    (save / "event_horizons.json").write_text(json.dumps({"0": cuts[1:-1].tolist()}))
    cfg = OmegaConf.create({
        "per_event_horizons": per_event,
        "data": {"num_events": 1, "label_transform": {
            "save_dir": str(save), "train_dir": str(save / "split_0")}},
    })
    return cfg, cuts, lam[te], t[te], e[te]


@pytest.mark.parametrize("per_event", [False, True])
def test_shared_equals_finetune_path(tmp_path, per_event):
    cfg, cuts, lam, t, e = _setup(tmp_path, per_event)
    refs = references(e[:, None], t[:, None])
    surv_at = lambda times: np.exp(-lam[:, None, None] * np.asarray(times)[None, None, :])  # noqa: E731
    shared = evaluate_curves(cfg, surv_at, refs)

    on_cuts = surv_at(cuts)  # (n, 1, len(cuts)) incl. t = 0
    preds = {"hazard": np.zeros_like(on_cuts), "risk": 1 - on_cuts, "survival": on_cuts}
    save, train = cfg.data.label_transform.save_dir, cfg.data.label_transform.train_dir
    st = SurvTRACEMetrics(cfg.data, f"{save}/duration_cuts.csv",
                          f"{train}/transformed_train_labels.csv", per_event).compute(preds, refs)
    for key in ("ctd_weighted_avg", "ctd_0th_event_0.25", "ctd_0th_event_0.75",
                "brier_survtrace_weighted_avg"):
        assert shared[key] == pytest.approx(st[key], abs=1e-6), key

    v2 = V2Metrics(cfg.data, f"{save}/duration_cuts.csv", f"{train}/transformed_train_labels.csv")
    v2.final = True
    ft = v2.compute(preds, refs)
    # the fine-grid path and the cut-grid path describe the same exponential curves
    for key, tol in (("antolini", 2e-3), ("ibs", 2e-3), ("mae_linear_uncensored", 0.05),
                     ("mae_tmax_margin", 0.05)):
        assert shared[key] == pytest.approx(ft[key], rel=tol), key
