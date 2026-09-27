import numpy as np

from sat.evaluate.multievent_metrics import _interp_cif, within_subject_concordance


def test_interp_cif_hits_grid_points_and_is_monotone():
    grid = np.array([0.0, 1.0, 2.0, 4.0])
    cif = np.array([[0.0, 0.2, 0.5, 0.9]])
    for t, want in [(0.0, 0.0), (1.0, 0.2), (2.0, 0.5), (4.0, 0.9)]:
        np.testing.assert_allclose(_interp_cif(grid, cif, np.array([t])), [want], atol=1e-12)
    ts = np.linspace(0, 4, 50)
    vals = _interp_cif(grid, np.repeat(cif, 50, 0), ts)
    assert np.all(np.diff(vals) >= -1e-12)


def _fn(risk):
    return lambda k, t: risk[:, k]


def test_perfect_and_reversed_ordering():
    # event 0 at t=1 (observed), event 1 at t=3 (observed) -> one pair (0 before 1)
    events = np.array([[1, 1]])
    durations = np.array([[1.0, 3.0]])
    good = np.array([[0.8, 0.2]])
    bad = np.array([[0.2, 0.8]])
    assert within_subject_concordance(events, durations, _fn(good))[0] == 1.0
    assert within_subject_concordance(events, durations, _fn(bad))[0] == 0.0


def test_censored_later_event_counts_and_ties_are_skipped():
    events = np.array([[1, 0], [1, 1]])
    durations = np.array([[1.0, 5.0], [2.0, 2.0]])  # row 2: tied times -> no pair
    risk = np.array([[0.9, 0.1], [0.1, 0.9]])
    pooled, subject, pairs = within_subject_concordance(events, durations, _fn(risk))
    assert pairs == 1 and pooled == 1.0 and subject == 1.0


def test_censored_event_cannot_be_the_earlier_one():
    events = np.array([[0, 1]])
    durations = np.array([[1.0, 5.0]])  # event 0 censored at 1: nothing is orderable
    assert within_subject_concordance(events, durations, _fn(np.array([[0.5, 0.5]])))[2] == 0
