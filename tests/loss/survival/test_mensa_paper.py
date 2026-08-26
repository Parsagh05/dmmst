"""Tests for the paper-faithful MENSA implementation.

Reference: MENSA, arXiv:2409.06525 (ML4H 2025), and the authors' code at
https://github.com/thecml/mensa. Equation numbers refer to the paper.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sat.loss import MENSAPaperLoss
from sat.models.heads import SAOutput
from sat.models.heads.mensa_paper import weibull_log_f_s

NUM_EVENTS = 2
N_DISTS = 3
CUTS = np.array([0.0, 2.0, 4.0, 6.0, 10.0])


@pytest.fixture
def duration_cuts_file(tmp_path):
    path = tmp_path / "duration_cuts.csv"
    pd.DataFrame({"cuts": CUTS}).to_csv(path, index=False, header=False)
    return str(path)


@pytest.fixture
def training_set_file(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "duration_event1": rng.uniform(0.5, 9.0, n),
            "event1": rng.integers(0, 2, n),
            "duration_event2": rng.uniform(0.5, 9.0, n),
            "event2": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "transformed_train_labels.csv"
    df.to_csv(path, index=False)
    return str(path)


def make_refs(durations, events):
    durations = torch.as_tensor(durations, dtype=torch.float32)
    events = torch.as_tensor(events, dtype=torch.float32)
    filler = torch.zeros(durations.shape[0], NUM_EVENTS)
    return torch.cat([filler, events, filler, durations], dim=1)


def make_output(shape, scale, gate):
    return SAOutput(
        loss=None, logits=None, survival=None, hazard=None, risk=None,
        shape=shape, scale=scale, logits_g=gate,
    )


# --------------------------------------------------------------- distribution


def test_single_component_matches_closed_form_weibull():
    """With one dominant component the mixture must equal a plain Weibull.

    Reference parameterisation: S(t) = exp(-(exp(b) t)^exp(k)).
    """
    k = torch.tensor([[0.3]])          # log shape
    b = torch.tensor([[-1.2]])         # log inverse-scale
    gate = torch.tensor([[0.0]])       # single component
    t = torch.tensor([[1.5]])

    log_f, log_s = weibull_log_f_s(k, b, gate, t)

    ek, eb = float(np.exp(0.3)), float(np.exp(-1.2))
    tt = 1.5
    expected_log_s = -((eb * tt) ** ek)
    # log f = log(eta/beta) + (eta-1) log(t/beta) - (t/beta)^eta, with 1/beta = eb
    expected_log_f = (
        0.3 + (-1.2) + (ek - 1.0) * (-1.2 + np.log(tt)) + expected_log_s
    )

    assert log_s.item() == pytest.approx(expected_log_s, rel=1e-5)
    assert log_f.item() == pytest.approx(expected_log_f, rel=1e-5)


def test_survival_is_a_valid_decreasing_probability():
    torch.manual_seed(0)
    k = torch.randn(8, N_DISTS) * 0.3
    b = torch.randn(8, N_DISTS) * 0.3 - 1.0
    gate = torch.randn(8, N_DISTS)
    t = torch.linspace(0.1, 10.0, 6).unsqueeze(0).expand(8, -1)

    _, log_s = weibull_log_f_s(k, b, gate, t)
    s = log_s.exp()

    assert torch.all(s <= 1.0 + 1e-6) and torch.all(s >= 0.0)
    assert torch.all(s[:, 1:] <= s[:, :-1] + 1e-6), "S(t) must be non-increasing"


def test_mixture_collapses_to_the_dominant_component():
    """A hugely peaked gate must reproduce that single component."""
    k = torch.tensor([[0.3, 1.5, -0.7]])
    b = torch.tensor([[-1.2, 0.4, 2.0]])
    peaked = torch.tensor([[50.0, -50.0, -50.0]])
    t = torch.tensor([[2.0]])

    _, log_s_mix = weibull_log_f_s(k, b, peaked, t)
    _, log_s_one = weibull_log_f_s(k[:, :1], b[:, :1], torch.zeros(1, 1), t)

    assert log_s_mix.item() == pytest.approx(log_s_one.item(), rel=1e-4)


def test_no_overflow_on_raw_duration_scales():
    """Regression: t up to ~2000 (SUPPORT) previously made torch.pow return NaN.

    The published code computes -(exp(b) t)**exp(k) directly, which overflows to
    inf on raw durations and yields "PowBackward1 returned nan values" in the
    backward pass. The log-space form must stay finite and differentiable.
    """
    k = torch.zeros(4, N_DISTS, requires_grad=True)
    b = torch.zeros(4, N_DISTS, requires_grad=True)
    gate = torch.zeros(4, N_DISTS, requires_grad=True)
    t = torch.tensor([[2029.0], [355.0], [1.0], [0.01]])

    log_f, log_s = weibull_log_f_s(k, b, gate, t)
    assert torch.isfinite(log_s).all() and torch.isfinite(log_f).all()

    log_s.sum().backward()
    assert torch.isfinite(k.grad).all(), "gradient must not be NaN at large t"
    assert torch.isfinite(b.grad).all()


# ---------------------------------------------------------------------- loss


def test_eq7_matches_manual_likelihood(duration_cuts_file, training_set_file):
    """L_ME = sum_p w_p [delta log f_p + (1-delta) log S_p], averaged over N."""
    loss_fn = MENSAPaperLoss(
        duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
        event_weights=[1.0, 1.0], traj_lambda=0.0,
    )
    torch.manual_seed(1)
    n = 5
    shape = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2
    scale = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2 - 1.0
    gate = torch.randn(n, NUM_EVENTS, N_DISTS)
    durations = torch.rand(n, NUM_EVENTS) * 8 + 0.5
    events = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])

    got = loss_fn(make_output(shape, scale, gate), make_refs(durations, events))

    total = 0.0
    for p in range(NUM_EVENTS):
        f_p, s_p = weibull_log_f_s(shape[:, p], scale[:, p], gate[:, p], durations[:, p])
        total = total + (events[:, p] * f_p + (1 - events[:, p]) * s_p).sum()
    expected = -(total / n)

    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_event_weights_are_inverse_frequency(duration_cuts_file, training_set_file):
    """w_p from the inverse frequency of transitions into state p."""
    loss_fn = MENSAPaperLoss(
        duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
        training_set=training_set_file,
    )
    df = pd.read_csv(training_set_file)
    freq = np.array([(df[f"event{p+1}"] == 1).mean() for p in range(NUM_EVENTS)])
    expected = 1.0 / freq
    expected = expected / expected.mean()

    assert np.allclose(loss_fn.event_weights.numpy(), expected, rtol=1e-5)


def test_eq8_trajectory_uses_S_B_at_T_A(duration_cuts_file):
    """Eq. 8: sum over ordered pairs of delta_A delta_B log S_B(T_A)."""
    loss_fn = MENSAPaperLoss(
        duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
        event_weights=[1.0, 1.0], trajectories=[[0, 1]], traj_lambda=1.0,
        trajectory_time="paper",
    )
    torch.manual_seed(2)
    n = 4
    shape = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2
    scale = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2 - 1.0
    gate = torch.randn(n, NUM_EVENTS, N_DISTS)
    durations = torch.tensor([[1.0, 3.0], [2.0, 5.0], [4.0, 6.0], [1.5, 2.5]])
    # only rows 0 and 2 have BOTH events observed
    events = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    got = loss_fn(make_output(shape, scale, gate), make_refs(durations, events))

    mask = torch.tensor([True, False, True, False])
    _, log_s_b_at_ta = weibull_log_f_s(
        shape[mask, 1], scale[mask, 1], gate[mask, 1], durations[mask, 0]
    )
    expected = -log_s_b_at_ta.mean()  # lambda = 1 -> pure trajectory term

    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_reference_variant_uses_S_B_at_T_B(duration_cuts_file):
    """The released code evaluates S_B at T_B; the two must differ when T_A != T_B."""
    common = dict(
        duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
        event_weights=[1.0, 1.0], trajectories=[[0, 1]], traj_lambda=1.0,
    )
    paper = MENSAPaperLoss(**common, trajectory_time="paper")
    ref = MENSAPaperLoss(**common, trajectory_time="reference")

    torch.manual_seed(3)
    n = 4
    shape = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2
    scale = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2 - 1.0
    gate = torch.randn(n, NUM_EVENTS, N_DISTS)
    durations = torch.tensor([[1.0, 3.0], [2.0, 5.0], [4.0, 6.0], [1.5, 2.5]])
    events = torch.ones(n, NUM_EVENTS)

    out, refs = make_output(shape, scale, gate), make_refs(durations, events)
    a, b = paper(out, refs).item(), ref(out, refs).item()

    assert a != pytest.approx(b, rel=1e-6), "the two conventions must not coincide"

    _, s_at_tb = weibull_log_f_s(shape[:, 1], scale[:, 1], gate[:, 1], durations[:, 1])
    assert b == pytest.approx(-s_at_tb.mean().item(), rel=1e-5)


def test_eq9_lambda_interpolates(duration_cuts_file):
    """L_total = (1-lambda) L_ME + lambda L_traj."""
    torch.manual_seed(4)
    n = 6
    shape = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2
    scale = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2 - 1.0
    gate = torch.randn(n, NUM_EVENTS, N_DISTS)
    durations = torch.rand(n, NUM_EVENTS) * 8 + 0.5
    events = torch.ones(n, NUM_EVENTS)
    out, refs = make_output(shape, scale, gate), make_refs(durations, events)

    kw = dict(duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
              event_weights=[1.0, 1.0], trajectories=[[0, 1]])
    only_me = MENSAPaperLoss(**kw, traj_lambda=0.0)(out, refs).item()
    only_tr = MENSAPaperLoss(**kw, traj_lambda=1.0)(out, refs).item()
    half = MENSAPaperLoss(**kw, traj_lambda=0.5)(out, refs).item()

    assert half == pytest.approx(0.5 * only_me + 0.5 * only_tr, rel=1e-5)


def test_trajectory_term_is_zero_without_pairs(duration_cuts_file):
    """No subject with both events observed -> the term contributes nothing."""
    kw = dict(duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
              event_weights=[1.0, 1.0], trajectories=[[0, 1]])
    torch.manual_seed(5)
    n = 3
    shape = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2
    scale = torch.randn(n, NUM_EVENTS, N_DISTS) * 0.2 - 1.0
    gate = torch.randn(n, NUM_EVENTS, N_DISTS)
    durations = torch.rand(n, NUM_EVENTS) * 8 + 0.5
    events = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # never both
    out, refs = make_output(shape, scale, gate), make_refs(durations, events)

    with_traj = MENSAPaperLoss(**kw, traj_lambda=0.5)(out, refs).item()
    me_only = MENSAPaperLoss(**kw, traj_lambda=0.0)(out, refs).item()

    assert with_traj == pytest.approx(0.5 * me_only, rel=1e-5)


def test_gradients_flow(duration_cuts_file):
    loss_fn = MENSAPaperLoss(
        duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
        event_weights=[1.0, 1.0], trajectories=[[0, 1]], traj_lambda=0.3,
    )
    n = 4
    shape = torch.randn(n, NUM_EVENTS, N_DISTS, requires_grad=True)
    scale = torch.randn(n, NUM_EVENTS, N_DISTS, requires_grad=True)
    gate = torch.randn(n, NUM_EVENTS, N_DISTS, requires_grad=True)
    durations = torch.rand(n, NUM_EVENTS) * 8 + 0.5
    events = torch.ones(n, NUM_EVENTS)

    loss_fn(make_output(shape, scale, gate), make_refs(durations, events)).backward()

    for name, t in (("shape", shape), ("scale", scale), ("gate", gate)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    assert shape.grad.abs().sum() > 0


def test_invalid_lambda_and_trajectory_are_rejected(duration_cuts_file):
    with pytest.raises(ValueError):
        MENSAPaperLoss(duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
                       traj_lambda=1.5)
    with pytest.raises(ValueError):
        MENSAPaperLoss(duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
                       trajectories=[[0, 9]])
    with pytest.raises(ValueError):
        MENSAPaperLoss(duration_cuts=duration_cuts_file, num_events=NUM_EVENTS,
                       trajectory_time="nonsense")
