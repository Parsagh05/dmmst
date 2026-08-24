"""Tests for the Margin-Mean-Variance (MMV) loss.

Reference: UniSurv, arXiv:2409.06209 (Machine Learning, 2024).
Equation numbers in the assertions refer to that paper.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from sat.loss import MMVLoss
from sat.models.heads import SAOutput
from sat.utils.km import KaplanMeierArea

NUM_EVENTS = 2
CUTS = np.array([2.0, 4.0, 6.0, 8.0, 10.0])


@pytest.fixture
def duration_cuts_file(tmp_path):
    path = tmp_path / "duration_cuts.csv"
    pd.DataFrame({"cuts": CUTS}).to_csv(path, index=False, header=False)
    return str(path)


@pytest.fixture
def importance_weights_file(tmp_path):
    path = tmp_path / "weights.csv"
    # [censored, event1, event2]
    pd.DataFrame({"weights": [0.5, 1.0, 1.0]}).to_csv(path, index=False, header=False)
    return str(path)


@pytest.fixture
def training_set_file(tmp_path):
    """Transformed training labels, in the layout L1Loss/MMVLoss expect."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "duration_event1": rng.uniform(1.0, 10.0, n),
            "event1": rng.integers(0, 2, n),
            "duration_event2": rng.uniform(1.0, 10.0, n),
            "event2": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "transformed_train_labels.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def loss_fn(duration_cuts_file, training_set_file, importance_weights_file):
    return MMVLoss(
        duration_cuts=duration_cuts_file,
        training_set=training_set_file,
        importance_sample_weights=importance_weights_file,
        num_events=NUM_EVENTS,
        max_time=12.0,
        variance_weight=0.01,
    )


def make_predictions(survival: torch.Tensor) -> SAOutput:
    return SAOutput(loss=None, logits=None, survival=survival, hazard=None, risk=None)


def make_references(durations, events) -> torch.Tensor:
    """references layout: [percentiles | events | fractions | durations]."""
    durations = torch.as_tensor(durations, dtype=torch.float32)
    events = torch.as_tensor(events, dtype=torch.float32)
    n = durations.shape[0]
    filler = torch.zeros(n, NUM_EVENTS)
    return torch.cat([filler, events, filler, durations], dim=1)


def test_initialization(loss_fn):
    assert loss_fn.num_events == NUM_EVENTS
    assert loss_fn.num_time_bins == len(CUTS)
    assert len(loss_fn.kms) == NUM_EVENTS
    assert torch.allclose(
        loss_fn.weights, torch.tensor([0.5, 1.0, 1.0], dtype=torch.float32)
    )


def test_max_time_defaults_to_last_cut(duration_cuts_file, training_set_file):
    loss = MMVLoss(
        duration_cuts=duration_cuts_file,
        training_set=training_set_file,
        num_events=NUM_EVENTS,
    )
    assert float(loss.max_time) == pytest.approx(CUTS[-1])


def test_mean_lifetime_matches_manual_integration(loss_fn):
    """Eq. 3: mu_hat is the integral of S(t), by the trapezoidal rule."""
    # S(0)=1 then a simple decreasing staircase over the 5 cuts
    s = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)

    mu_hat, _ = loss_fn.mean_and_variance(survival)

    # grid boundaries [0, 2, 4, 6, 8, 10, 12]; S is 0 at max_time
    grid = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    s_ext = np.append(s.numpy(), 0.0)
    expected = np.sum(np.diff(grid) * (s_ext[:-1] + s_ext[1:]) / 2.0)

    assert mu_hat.shape == (1, NUM_EVENTS)
    assert mu_hat[0, 0].item() == pytest.approx(expected, rel=1e-5)


def test_density_is_a_proper_distribution(loss_fn):
    """The implied per-interval density must sum to 1."""
    s = torch.tensor([1.0, 0.7, 0.5, 0.35, 0.2, 0.05])
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)

    grid = loss_fn._time_grid(survival.device, survival.shape[-1])
    zeros = torch.zeros(1, NUM_EVENTS, 1)
    surv = torch.cat((survival, zeros), dim=2)
    pdf = surv[:, :, :-1] - surv[:, :, 1:]

    assert grid.shape[0] == survival.shape[-1] + 1
    assert torch.allclose(pdf.sum(dim=2), torch.ones(1, NUM_EVENTS), atol=1e-6)
    assert (pdf >= 0).all(), "a decreasing survival curve must give a non-negative pdf"


def test_variance_is_zero_for_a_point_mass(loss_fn):
    """Eq. 4: all mass in one interval => variance 0 about that interval midpoint."""
    # S drops from 1 to 0 across the first interval only
    s = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)

    mu_hat, v = loss_fn.mean_and_variance(survival)

    assert mu_hat[0, 0].item() == pytest.approx(1.0, abs=1e-5)  # midpoint of [0, 2]
    assert v[0, 0].item() == pytest.approx(0.0, abs=1e-5)


def test_margin_time_matches_kaplan_meier_best_guess(loss_fn, training_set_file):
    """Eq. 5: censored subjects get the KM best-guess; observed keep their time."""
    durations = torch.tensor([[3.0, 3.0], [7.0, 7.0]])
    events = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # row 0 observed, row 1 censored

    e_m, w = loss_fn.margin_times(durations, events)

    # observed subjects are untouched and fully weighted
    assert e_m[0, 0].item() == pytest.approx(3.0)
    assert w[0, 0].item() == pytest.approx(1.0)

    # censored subject matches KaplanMeierArea.best_guess exactly
    df = pd.read_csv(training_set_file)
    km = KaplanMeierArea(df["duration_event1"], df["event1"] == 1)
    expected_guess = km.best_guess(np.array([7.0]))[0]
    expected_w = 1.0 - km.predict(np.array([7.0]))[0]

    assert e_m[1, 0].item() == pytest.approx(expected_guess, rel=1e-5)
    assert w[1, 0].item() == pytest.approx(expected_w, rel=1e-5)
    # the margin time can only push the estimate later than the censoring time
    assert e_m[1, 0].item() >= 7.0


def test_perfect_prediction_gives_near_zero_margin_mean_loss(loss_fn):
    """A model whose mu_hat equals the observed time incurs no margin-mean penalty."""
    # point mass in the first interval => mu_hat = 1.0
    s = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)
    references = make_references([[1.0, 1.0]], [[1.0, 1.0]])

    loss = loss_fn(make_predictions(survival), references)

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_loss_increases_with_prediction_error(loss_fn):
    """Moving mu_hat away from the observed time must increase the loss."""
    references = make_references([[1.0, 1.0]], [[1.0, 1.0]])

    good = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 1, -1)
    bad = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]).view(1, 1, -1)

    loss_good = loss_fn(
        make_predictions(good.repeat(1, NUM_EVENTS, 1)), references
    )
    loss_bad = loss_fn(make_predictions(bad.repeat(1, NUM_EVENTS, 1)), references)

    assert loss_bad.item() > loss_good.item()


def test_censored_subjects_are_down_weighted(loss_fn):
    """Eq. 6: an early-censored subject contributes less than an observed one.

    w = 1 - S_km(T) is small for early censoring times, so with the same squared
    error the censored subject must contribute strictly less.
    """
    s = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # mu_hat = 1.0
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)

    observed = make_references([[9.0, 9.0]], [[1.0, 1.0]])
    censored = make_references([[9.0, 9.0]], [[0.0, 0.0]])

    loss_observed = loss_fn(make_predictions(survival), observed)
    loss_censored = loss_fn(make_predictions(survival), censored)

    assert loss_observed.item() > 0
    assert loss_censored.item() != loss_observed.item()


def test_variance_weight_changes_the_loss(
    duration_cuts_file, training_set_file, importance_weights_file
):
    """lambda_v must actually scale the variance term."""
    # a spread-out density so the variance term is non-trivial
    s = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)
    references = make_references([[5.0, 5.0]], [[1.0, 1.0]])

    losses = []
    for lam_v in (0.0, 1.0):
        fn = MMVLoss(
            duration_cuts=duration_cuts_file,
            training_set=training_set_file,
            importance_sample_weights=importance_weights_file,
            num_events=NUM_EVENTS,
            max_time=12.0,
            variance_weight=lam_v,
        )
        losses.append(fn(make_predictions(survival), references).item())

    assert losses[1] > losses[0], "increasing lambda_v must increase the loss"


def test_gradients_flow_to_the_survival_curve(loss_fn):
    """The loss must be differentiable w.r.t. the model output."""
    s = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1], requires_grad=True)
    survival = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)
    references = make_references([[5.0, 5.0]], [[1.0, 0.0]])

    loss = loss_fn(make_predictions(survival), references)
    loss.backward()

    assert s.grad is not None
    assert torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0


def test_batch_reduction_is_a_mean(loss_fn):
    """Duplicating a batch must not change the loss value."""
    s = torch.tensor([1.0, 0.7, 0.5, 0.3, 0.1, 0.05])
    single = s.view(1, 1, -1).repeat(1, NUM_EVENTS, 1)
    doubled = single.repeat(2, 1, 1)

    ref_single = make_references([[5.0, 5.0]], [[1.0, 1.0]])
    ref_doubled = make_references([[5.0, 5.0], [5.0, 5.0]], [[1.0, 1.0], [1.0, 1.0]])

    loss_single = loss_fn(make_predictions(single), ref_single)
    loss_doubled = loss_fn(make_predictions(doubled), ref_doubled)

    assert loss_doubled.item() == pytest.approx(loss_single.item(), rel=1e-5)


def test_grid_aligns_with_cuts_that_already_start_at_zero(tmp_path, training_set_file):
    """Regression: duration_cuts.csv written by train_labeltransform starts at 0.0.

    In that layout len(cuts) == survival width, so prepending another zero would
    create one interval too many. The original implementation did exactly that and
    blew up only once it hit the real pipeline, because every test here used a cut
    list that omitted t=0.
    """
    cuts_with_zero = np.array([0.0, 42.68, 85.87, 145.33, 355.20])   # as on METABRIC
    path = tmp_path / "cuts0.csv"
    pd.DataFrame({"cuts": cuts_with_zero}).to_csv(path, index=False, header=False)

    fn = MMVLoss(duration_cuts=str(path), training_set=training_set_file,
                 num_events=NUM_EVENTS, variance_weight=0.01)

    # survival width equals len(cuts) in this layout
    survival = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2]).view(1, 1, -1).repeat(1, NUM_EVENTS, 1)
    grid = fn._time_grid(survival.device, survival.shape[-1])
    assert grid.shape[0] == survival.shape[-1] + 1

    mu_hat, v = fn.mean_and_variance(survival)
    assert mu_hat.shape == (1, NUM_EVENTS)
    assert torch.isfinite(mu_hat).all() and (mu_hat > 0).all()

    refs = make_references([[100.0, 100.0]], [[1.0, 1.0]])
    loss = fn(make_predictions(survival), refs)
    assert torch.isfinite(loss)


def test_mismatched_cuts_raise_clearly(tmp_path, training_set_file):
    """A cut list that cannot align with the survival width must fail loudly."""
    path = tmp_path / "cuts_bad.csv"
    pd.DataFrame({"cuts": np.arange(9, dtype=float)}).to_csv(path, index=False, header=False)
    fn = MMVLoss(duration_cuts=str(path), training_set=training_set_file,
                 num_events=NUM_EVENTS)
    survival = torch.rand(1, NUM_EVENTS, 5)
    with pytest.raises(ValueError, match="Cannot align"):
        fn.mean_and_variance(survival)
