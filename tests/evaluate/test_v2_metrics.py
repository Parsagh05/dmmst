"""Version-2 metric suite: E[T] tails against closed forms, and every metric present
and in range on a model whose curves are the truth."""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from sat.evaluate.v2_metrics import V2Metrics, expected_time, ibll


def test_expected_time_exact_for_exponential():
    lam = np.array([0.5, 2.0])
    grid = np.linspace(0, 3.0, 7)
    S = np.exp(-lam[:, None] * grid[None, :])
    trunc = expected_time(grid, S, "trunc")
    assert np.allclose(trunc, (1 - np.exp(-lam * 3.0)) / lam, rtol=1e-6)
    # linear tail adds the triangle from (t_K, S_K) to where the (0,1)-(t_K,S_K) line hits 0
    SK = np.exp(-lam * 3.0)
    t0 = 3.0 / (1 - SK)
    assert np.allclose(expected_time(grid, S, "linear"), trunc + SK * (t0 - 3.0) / 2, rtol=1e-6)
    # tmax: E[T | T <= T_max] in closed form for the exponential
    T = 3.0
    cond = (1 / lam - np.exp(-lam * T) * (T + 1 / lam)) / (1 - np.exp(-lam * T))
    assert np.allclose(expected_time(grid, S, "tmax"), cond, rtol=1e-6)
    assert np.all(expected_time(grid, S, "tmax") <= trunc)
    # grid independent: a 10x finer sampling gives the same value
    fine = np.linspace(0, 3.0, 61)
    Sf = np.exp(-lam[:, None] * fine[None, :])
    for tail in ("trunc", "tmax", "linear"):
        assert np.allclose(expected_time(fine, Sf, tail), expected_time(grid, S, tail), rtol=1e-6)


def test_ibll_without_censoring_is_the_mean_log_loss():
    from tests.evaluate.test_survtrace_metrics import _structured

    t = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.ones(4, dtype=bool)
    grid = np.array([1.5, 2.5, 3.5])
    S = np.array([[0.2, 0.1, 0.1], [0.7, 0.3, 0.2], [0.8, 0.6, 0.3], [0.9, 0.8, 0.7]])
    et = _structured(e, t)
    died = t[:, None] <= grid[None, :]
    ll = -(died * np.log(1 - S) + (~died) * np.log(S)).mean(0)
    want = np.trapz(ll, grid) / (grid[-1] - grid[0])
    assert ibll(et, et, grid, S) == pytest.approx(want, rel=1e-6)


def _setup(tmp_path, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    lam = 0.2 * np.exp(score)
    T = rng.exponential(1 / lam)
    C = rng.uniform(0, np.quantile(T, 0.95), size=n)
    t, e = np.minimum(T, C), (T <= C)
    tr, te = slice(0, 1000), slice(1000, None)
    cuts = np.concatenate([[0.0], np.quantile(t[e], [0.25, 0.5, 0.75]), [t.max()]])
    pd.DataFrame({"c": cuts}).to_csv(tmp_path / "cuts.csv", index=False, header=False)
    pd.DataFrame({"event1": e[tr].astype(int), "duration_event1": t[tr]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    surv = np.exp(-lam[te, None] * cuts[None, :])  # true curves on the grid incl. t=0
    refs = np.zeros((len(t[te]), 4))
    refs[:, 1], refs[:, 3] = e[te], t[te]
    m = V2Metrics(OmegaConf.create({"num_events": 1}), str(tmp_path / "cuts.csv"),
                  str(tmp_path / "train.csv"))
    return m, surv, refs, lam[te]


def test_suite_on_true_curves(tmp_path):
    m, surv, refs, lam = _setup(tmp_path)
    m.final = True
    out = m.compute({"survival": surv[:, None, :], "time_to_event": 1 / lam}, refs)
    for key in ("ibs", "ibll", "antolini", "auc", "dcal_p", "onecal_p",
                "mae_linear_pseudo_obs", "mae_trunc_margin", "mae_tmax_hinge",
                "reg_mae_uncensored", "reg_mae_pseudo_obs", "reg_harrell"):
        assert key in out, key
    assert 0.0 < out["ibs"] < 0.25
    assert out["antolini"] > 0.65 and out["auc"] > 0.65 and out["reg_harrell"] > 0.65
    assert out["dcal_p"] > 0.01  # true curves are calibrated
    # an anti-correlated predictor ranks worse
    bad = m.compute({"time_to_event": lam}, refs)
    assert "reg_harrell" in bad and bad["reg_harrell"] < 0.5


def test_cheap_mode_only_scores_regression_margin(tmp_path):
    m, surv, refs, lam = _setup(tmp_path)
    out = m.compute({"survival": surv[:, None, :], "time_to_event": 1 / lam}, refs)
    assert set(out) == {"reg_mae_margin", "reg_mae_margin_0th_event"}


def test_antolini_matches_pycox_on_a_fine_grid_and_harrell_under_ph():
    from lifelines.utils import concordance_index
    from pycox.evaluation import EvalSurv

    from sat.evaluate.v2_metrics import antolini

    rng = np.random.default_rng(1)
    n = 400
    lam = 0.2 * np.exp(rng.normal(size=n))
    T = rng.exponential(1 / lam)
    C = rng.uniform(0, np.quantile(T, 0.9), size=n)
    t, e = np.minimum(T, C), (T <= C)
    cuts = np.linspace(0, t.max(), 6)
    ours = antolini(cuts, np.exp(-lam[:, None] * cuts[None, :]), t, e)
    fine = np.linspace(0, t.max(), 20000)  # pycox's step evaluation converges here
    ref = EvalSurv(pd.DataFrame(np.exp(-lam[:, None] * fine[None, :]).T, index=fine),
                   t, e.astype(int), censor_surv="km").concordance_td("antolini")
    assert ours == pytest.approx(ref, abs=0.005)
    # proportional hazards: Antolini's C equals Harrell's C on the risk score
    assert ours == pytest.approx(concordance_index(t, -lam, e), abs=0.005)
