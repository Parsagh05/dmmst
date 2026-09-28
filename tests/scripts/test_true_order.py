import numpy as np

from scripts.true_order import order_probabilities, score


def test_order_probability_matches_exponential_closed_form():
    lam = np.array([[0.5, 1.5], [2.0, 0.2]])  # (n, K) rates
    t = np.linspace(0, 40, 4001)
    surv = np.exp(-lam[:, :, None] * t[None, None, :])
    p = order_probabilities(surv)
    want = lam[:, 0] / (lam[:, 0] + lam[:, 1])
    np.testing.assert_allclose(p[:, 0, 1], want, atol=2e-3)
    np.testing.assert_allclose(p[:, 0, 1] + p[:, 1, 0], 1.0, atol=2e-3)


def test_true_curves_beat_distorted_curves_on_proper_scores():
    rng = np.random.default_rng(0)
    n, K = 4000, 3
    lam = rng.gamma(2.0, 0.5, size=(n, K))
    latent = rng.exponential(1.0 / lam)
    t = np.linspace(0, 30, 1501)
    true = order_probabilities(np.exp(-lam[:, :, None] * t))
    swapped = order_probabilities(np.exp(-lam[:, ::-1][:, :, None] * t))
    flat = np.full_like(true, 0.5)
    s_true, s_bad, s_flat = score(true, latent), score(swapped, latent), score(flat, latent)
    assert s_true["true_order_brier"] < s_flat["true_order_brier"] < s_bad["true_order_brier"]
    assert s_true["true_order_c"] > 0.6 > s_bad["true_order_c"]
