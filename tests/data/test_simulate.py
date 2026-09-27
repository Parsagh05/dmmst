"""The simulator's true CIFs are what the oracle rows of every table rest on, so they
are checked against Monte Carlo on the latent times, in every regime."""

import dataclasses

import numpy as np
import pytest

from sat.data.simulate import SCENARIOS, Scenario, blocking_sets, simulate, true_cif


def _empirical_cif(T, regime, times):
    """P(event k observed by t) from the latent times, ignoring censoring."""
    n, K = T.shape
    out = np.zeros((K, len(times)))
    for k in range(K):
        blocks = blocking_sets(regime, K)[k]
        first = T[:, k] < (T[:, blocks].min(1) if blocks else np.inf)
        out[k] = [(first & (T[:, k] <= t)).mean() for t in times]
    return out


@pytest.mark.parametrize("regime", ["non-competing", "competing", "semi-competing"])
@pytest.mark.parametrize("frailty", [0.0, 1.0])
def test_true_cif_matches_monte_carlo(regime, frailty):
    sc = Scenario(n=20000, num_events=3, regime=regime, frailty_var=frailty, seed=1)
    _, truth, _ = simulate(sc)
    times = np.quantile(truth["latent_times"], [0.1, 0.3, 0.5, 0.7])
    model = true_cif(truth, regime, times).mean(0)  # population CIF (K, T)
    emp = _empirical_cif(truth["latent_times"], regime, times)
    np.testing.assert_allclose(model, emp, atol=0.015)


def test_closed_form_without_frailty():
    sc = Scenario(n=200, num_events=2, frailty_var=0.0, seed=2)
    _, truth, _ = simulate(sc)
    times = np.array([10.0, 100.0, 400.0])
    H = np.exp(truth["g"])[:, :, None] * (
        (times[None, None, :] / truth["scales"][None, :, None]) ** truth["shapes"][None, :, None]
    )
    np.testing.assert_allclose(true_cif(truth, "non-competing", times), 1 - np.exp(-H), atol=2e-3)


def test_censor_rate_is_hit_and_regimes_differ():
    for regime in ["non-competing", "competing", "semi-competing"]:
        sc = Scenario(n=4000, regime=regime, censor_rate=0.5, seed=3)
        df, _, meta = simulate(sc)
        assert abs(meta["unobserved_fraction"] - 0.5) < 0.01
        events = df[[f"event{k + 1}" for k in range(sc.num_events)]].values
        if regime == "competing":
            assert events.sum(1).max() == 1
        else:
            assert (events.sum(1) > 1).mean() > 0.3


def test_semi_competing_death_censors_non_terminal():
    df, truth, _ = simulate(Scenario(n=3000, regime="semi-competing", seed=4))
    K = 4
    death = df[f"duration{K}"].values
    for k in range(1, K):
        assert (df[f"duration{k}"].values <= death + 1e-9).all()


def test_named_scenarios_differ_from_base_in_one_knob_only():
    base = dataclasses.asdict(SCENARIOS["base"])
    for name, sc in SCENARIOS.items():
        diff = {k for k, v in dataclasses.asdict(sc).items() if base[k] != v} - {"name"}
        assert len(diff) <= 1, (name, diff)
