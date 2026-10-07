"""SurvTRACEMetrics must equal scikit-survival called exactly as SurvTRACE's
evaluate_utils.py calls it: censoring distribution from the training rows, C_td
truncated at each horizon, Brier at the horizons, risk read at the right grid column."""

import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sksurv.metrics import brier_score, concordance_index_ipcw

from sat.evaluate.survtrace_metrics import SurvTRACEMetrics


def _structured(e, t):
    return np.array(list(zip(e.astype(bool), t.astype(float))), dtype=[("e", bool), ("t", float)])


def _cohort(n, seed):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    t = rng.exponential(1.0 / np.exp(score))
    c = rng.uniform(0, np.quantile(t, 0.9), size=n)
    return score, np.minimum(t, c), (t <= c).astype(int)


def _write(tmp_path, cuts, train_events, train_durations, horizons=None):
    pd.DataFrame({"cuts": cuts}).to_csv(tmp_path / "cuts.csv", index=False, header=False)
    K = train_events.shape[1]
    cols = {}
    for k in range(K):
        cols[f"event{k + 1}"] = train_events[:, k]
        cols[f"duration_event{k + 1}"] = train_durations[:, k]
    pd.DataFrame(cols).to_csv(tmp_path / "train.csv", index=False)
    if horizons is not None:
        (tmp_path / "event_horizons.json").write_text(json.dumps(horizons))
    return str(tmp_path / "cuts.csv"), str(tmp_path / "train.csv")


def _predictions(scores, grid):
    """Exponential model S(t) = exp(-exp(score) t) on the grid INCLUDING t = 0, the
    layout the survival head emits (pad_col(where='start'))."""
    surv = np.exp(-np.exp(scores)[:, None] * grid[None, :])  # (n, len(grid))
    risk = 1.0 - surv
    hazard = np.zeros_like(surv)
    return np.stack([hazard, risk, surv], axis=1)[:, :, None, :]  # (n, 3, 1, len(grid))


def test_single_event_equals_sksurv_with_training_censoring(tmp_path):
    s_tr, t_tr, e_tr = _cohort(600, 0)
    s_te, t_te, e_te = _cohort(300, 1)
    t_te = np.minimum(t_te, t_tr.max() * 0.999)  # inside the training follow-up
    times = np.quantile(np.concatenate([t_tr, t_te])[np.concatenate([e_tr, e_te]) == 1],
                        [0.25, 0.5, 0.75])
    cuts = np.concatenate([[0.0], times, [max(t_tr.max(), t_te.max())]])
    cuts_f, train_f = _write(tmp_path, cuts, e_tr[:, None], t_tr[:, None])

    cfg = OmegaConf.create({"num_events": 1})
    m = SurvTRACEMetrics(cfg, cuts_f, train_f)
    refs = np.zeros((len(t_te), 4))
    refs[:, 1], refs[:, 3] = e_te, t_te
    out = m.compute(_predictions(s_te, cuts), refs)

    et_train, et_test = _structured(e_tr, t_tr), _structured(e_te, t_te)
    surv_at = np.exp(-np.exp(s_te)[:, None] * times[None, :])
    want_c = [concordance_index_ipcw(et_train, et_test, 1 - surv_at[:, i], tau=times[i])[0]
              for i in range(3)]
    want_b = brier_score(et_train, et_test, surv_at, times)[1]
    for i, q in enumerate([0.25, 0.5, 0.75]):
        assert out[f"ctd_0th_event_{q}"] == pytest.approx(want_c[i], abs=1e-9)
        assert out[f"brier_0th_event_{q}"] == pytest.approx(want_b[i], abs=1e-6)
    assert out["ctd_weighted_avg"] == pytest.approx(np.mean(want_c), abs=1e-9)

    # the censoring distribution really is the TRAINING one: fitting it on the test
    # rows (the old behaviour) gives a different number
    test_ipcw = concordance_index_ipcw(et_test, et_test, 1 - surv_at[:, 1], tau=times[1])[0]
    assert test_ipcw != pytest.approx(out["ctd_0th_event_0.5"], abs=1e-12)


def test_per_event_horizons_come_from_event_horizons_json(tmp_path):
    s1, t1, e1 = _cohort(800, 2)
    s2, t2, e2 = _cohort(800, 3)
    t2 = t2 * 10.0  # second event on a 10x longer time scale
    n_tr = 500
    E = np.stack([e1, e2], 1)
    T = np.stack([t1, t2], 1)
    horizons = {str(k): np.quantile(T[E[:, k] == 1, k], [0.25, 0.5, 0.75]).tolist()
                for k in range(2)}
    grid = np.linspace(0, T.max(), 40)
    cuts_f, train_f = _write(tmp_path, grid, E[:n_tr], T[:n_tr], horizons)

    cfg = OmegaConf.create({"num_events": 2})
    m = SurvTRACEMetrics(cfg, cuts_f, train_f, per_event_horizons=True)
    assert np.allclose(m.event_times[1], horizons["1"])

    te = slice(n_tr, None)
    preds = np.concatenate(
        [_predictions(s1[te], grid), _predictions(s2[te] - np.log(10.0), grid)], axis=2
    )
    refs = np.zeros((800 - n_tr, 8))
    refs[:, 2:4] = E[te]
    refs[:, 6:8] = T[te]
    out = m.compute(preds, refs)
    et_train = _structured(E[:n_tr, 1], T[:n_tr, 1])
    et_test = _structured(E[te, 1], T[te, 1])
    h = np.asarray(horizons["1"])
    # risk read log-linearly between grid points; for an exponential model that is exact
    risk = 1 - np.exp(-np.exp(s2[te] - np.log(10.0))[:, None] * h[None, :])
    usable = [i for i in range(3) if h[i] < min(T[:n_tr, 1].max(), T[te, 1].max())]
    for i in usable:
        want = concordance_index_ipcw(et_train, et_test, risk[:, i], tau=h[i])[0]
        assert out[f"ctd_1th_event_{[0.25, 0.5, 0.75][i]}"] == pytest.approx(want, abs=1e-6)


def test_brier_keeps_subjects_censored_at_the_last_training_time():
    """Administrative censoring: many subjects end at the same final time, which is also
    the last training time (hsa_synthetic: ~70%). They are alive at every horizon and must
    count in the Brier score - not be dropped - whatever the float precision of the times."""
    from sat.evaluate.survtrace_metrics import _structured, score_at_horizons

    rng = np.random.default_rng(0)
    end = 574.2571974083571

    def sample(n):
        t = rng.uniform(10, 500, n)
        e = rng.random(n) < 0.7
        late = rng.random(n) < 0.6
        t[late], e[late] = end, False
        return t, e

    t_tr, e_tr = sample(800)
    t_te, e_te = sample(400)
    times = np.array([150.0, 300.0, 450.0])
    surv = np.exp(-np.outer(rng.uniform(0.001, 0.004, len(t_te)), times))
    et_tr = _structured(e_tr, np.float32(t_tr).astype(float))

    # reference: the same data with the final time nudged below the training maximum
    ref_t = np.where(t_te == end, end - 1.0, t_te)
    _, ref = brier_score(et_tr, _structured(e_te, ref_t), surv, times)

    for t in (t_te, np.float32(t_te).astype(float)):  # float64 and float32 labels
        out = score_at_horizons(et_tr, _structured(e_te, t), 1 - surv, surv, times,
                                [0.25, 0.5, 0.75], 0)
        got = [out[f"brier_0th_event_{q}"] for q in (0.25, 0.5, 0.75)]
        np.testing.assert_allclose(got, ref, rtol=1e-9)
