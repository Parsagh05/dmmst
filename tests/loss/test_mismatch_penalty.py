"""Eq. 7 (paper Sec. 2.4) computed by hand for small cases."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sat.loss.factory import regression_loss, survival_loss
from sat.loss.regression.mismatch import MismatchPenalty
from sat.utils.km import KaplanMeierArea


def _refs(events, durations):
    events = np.asarray(events, dtype=float)
    durations = np.asarray(durations, dtype=float)
    n, K = events.shape
    r = np.zeros((n, 4 * K))
    r[:, K : 2 * K] = events
    r[:, 3 * K :] = durations
    return torch.tensor(r, dtype=torch.float32)


def _out(y):
    return SimpleNamespace(predictions=torch.tensor(y, dtype=torch.float32))


def test_matched_subject_has_no_penalty():
    mm = MismatchPenalty(num_events=2)
    # event 1 first (t=2), predicted first too (3 < 9)
    assert mm(_out([[3.0, 9.0]]), _refs([[1, 1]], [[2.0, 5.0]])).item() == 0.0


def test_mismatch_by_hand_paper_t_q():
    mm = MismatchPenalty(num_events=2, t_q="observed")
    # true first: event 1 at t_j = 2; predicted first: event 2 (y = 1 < 4);
    # event 2 censored at T_c = 5 -> t_q = 5 (Eq. 1)
    # ReLU(4 - 2) + ReLU(5 - 1) + (4 - 1) = 2 + 4 + 3 = 9
    loss = mm(_out([[4.0, 1.0]]), _refs([[1, 0]], [[2.0, 5.0]]))
    assert loss.item() == pytest.approx(9.0)


def test_mismatch_best_guess_t_q(tmp_path):
    train = pd.DataFrame(
        {
            "event1": [1, 1, 0, 1, 0, 1],
            "duration_event1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event2": [1, 0, 1, 0, 1, 1],
            "duration_event2": [2.0, 3.0, 4.0, 6.0, 7.0, 9.0],
        }
    )
    f = tmp_path / "train.csv"
    train.to_csv(f, index=False)
    mm = MismatchPenalty(num_events=2, t_q="best_guess", training_set=str(f))
    km = KaplanMeierArea(train.duration_event2.values, (train.event2 == 1).values)
    t_q = float(km.best_guess(np.array([5.0]))[0])
    assert t_q > 5.0
    loss = mm(_out([[4.0, 1.0]]), _refs([[1, 0]], [[2.0, 5.0]]))
    assert loss.item() == pytest.approx(2.0 + (t_q - 1.0) + 3.0, rel=1e-5)


def test_subjects_without_events_are_not_counted():
    mm = MismatchPenalty(num_events=2)
    refs = _refs([[1, 0], [0, 0]], [[2.0, 5.0], [3.0, 3.0]])
    loss = mm(_out([[4.0, 1.0], [0.1, 9.0]]), refs)
    assert loss.item() == pytest.approx(9.0)  # mean over the one subject with an event


def test_tie_is_not_a_mismatch():
    mm = MismatchPenalty(num_events=2)
    # both events observed at t=2: predicting event 2 first is not wrong
    assert mm(_out([[4.0, 1.0]]), _refs([[1, 1]], [[2.0, 2.0]])).item() == 0.0


def test_time_scale_divides_the_penalty():
    a = MismatchPenalty(num_events=2)(_out([[4.0, 1.0]]), _refs([[1, 0]], [[2.0, 5.0]]))
    b = MismatchPenalty(num_events=2, time_scale=10.0)(
        _out([[4.0, 1.0]]), _refs([[1, 0]], [[2.0, 5.0]])
    )
    assert b.item() == pytest.approx(a.item() / 10.0)


def test_gradient_pushes_the_order_right():
    mm = MismatchPenalty(num_events=2)
    y = torch.tensor([[4.0, 1.0]], requires_grad=True)
    mm(SimpleNamespace(predictions=y), _refs([[1, 0]], [[2.0, 5.0]])).backward()
    assert y.grad[0, 0] > 0  # gradient descent lowers y_j (true first event)
    assert y.grad[0, 1] < 0  # and raises y_q


def test_single_event_is_rejected():
    with pytest.raises(ValueError):
        MismatchPenalty(num_events=1)


def _files(tmp_path, K):
    cuts = tmp_path / "cuts.csv"
    pd.DataFrame({"c": [0.0, 1.0, 2.0, 3.0]}).to_csv(cuts, index=False, header=False)
    imp = tmp_path / "imp.csv"
    pd.DataFrame({"w": [1.0] * (K + 1)}).to_csv(imp, index=False, header=False)
    train = tmp_path / "train.csv"
    cols = {}
    for k in range(K):
        cols[f"event{k + 1}"] = [1, 0, 1, 1]
        cols[f"duration_event{k + 1}"] = [0.5, 1.5, 2.0, 2.5]
    pd.DataFrame(cols).to_csv(train, index=False)
    return str(cuts), str(imp), str(train)


def test_factory_builds_only_enabled_terms(tmp_path):
    cuts, imp, train = _files(tmp_path, 2)
    names = lambda m: [type(x).__name__ for x in m.losses]  # noqa: E731
    assert names(survival_loss(cuts, imp, train, 2)) == ["SATNLLPCHazardLoss"]
    assert names(survival_loss(cuts, imp, train, 2, rank_coeff=1, mul_coeff=1, mmv_coeff=1)) == [
        "SATNLLPCHazardLoss", "SampleRankingLoss", "MultiEventRankingLoss", "MMVLoss",
    ]
    assert names(regression_loss(train, imp, 2, l1_type="uncensored")) == ["L1Loss"]
    assert names(regression_loss(train, imp, 2, mm_coeff=0.5, mm_t_q="best_guess")) == [
        "L1Loss", "MismatchPenalty",
    ]


def test_factory_refuses_l_mul_on_single_event(tmp_path):
    cuts, imp, train = _files(tmp_path, 1)
    with pytest.raises(ValueError):
        survival_loss(cuts, imp, train, 1, mul_coeff=1.0)
