"""Kaplan-Meier best guess (Haider et al. 2020), used by the best-guess MAE (Eq. 6)."""

import numpy as np
import pytest

from sat.utils.km import KaplanMeierArea


def _numeric_best_guess(km, c, n=200001):
    """c + integral_c^inf S_ext / S(c) on a fine grid, S_ext = KM steps then 0 after
    the extension point."""
    times = km.area_times[:-1]
    probs = km.area_probabilities
    grid = np.linspace(c, times[-1], n)
    idx = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
    S = np.where(idx >= len(times) - 1, 0.0, probs[idx])
    s_c = S[0]
    return c + np.trapz(S, grid) / s_c if s_c > 0 else c


@pytest.mark.parametrize("last_censored", [False, True])
def test_best_guess_equals_numeric_integral(last_censored):
    rng = np.random.default_rng(3)
    t = np.round(rng.exponential(10, 200), 2)
    e = rng.random(200) < 0.6
    e[np.argmax(t)] = not last_censored  # censored last time -> curve extended linearly
    km = KaplanMeierArea(t, e)
    assert (km.survival_probabilities[-1] > 0) == last_censored
    for c in np.quantile(t, [0.1, 0.4, 0.8]):
        assert km.best_guess(np.array([c]))[0] == pytest.approx(_numeric_best_guess(km, c), rel=1e-3)
    assert np.all(km.best_guess(np.quantile(t, [0.1, 0.5])) > np.quantile(t, [0.1, 0.5]))


def test_best_guess_at_and_after_the_end_returns_the_censoring_time():
    km = KaplanMeierArea(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1], dtype=bool))
    c = np.array([3.0, 10.0])
    assert np.allclose(km.best_guess(c), c)
