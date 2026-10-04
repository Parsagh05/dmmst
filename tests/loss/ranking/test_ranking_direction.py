"""The ranking penalties must punish the WRONG order (paper Eqs. 4-5).

The manuscript writes eta(x, y) = exp((x - y) / sigma) with x = CIF of the subject /
event that happens first, which would reward the wrong order; the text and DeepHit
(Lee et al. 2018) use exp(-(x - y) / sigma). The implementation works on survival
(S = 1 - CIF) as exp((S_first - S_later) / sigma), i.e. the correct direction. These
tests pin that down for L_rank (SampleRankingLoss) and L_mul (MultiEventRankingLoss).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sat.loss.ranking.multievent import MultiEventRankingLoss
from sat.loss.ranking.sample import SampleRankingLoss

CUTS = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def cuts_file(tmp_path):
    p = tmp_path / "cuts.csv"
    pd.DataFrame({"c": CUTS}).to_csv(p, index=False, header=False)
    return str(p)


def _predictions(rates):
    """Constant-hazard curves S(t) = exp(-rate * t) on CUTS; rates: (n, K)."""
    rates = torch.tensor(rates, dtype=torch.float32)
    t = torch.tensor(CUTS, dtype=torch.float32)
    survival = torch.exp(-rates.unsqueeze(-1) * t)  # (n, K, len(CUTS))
    hazard = rates.unsqueeze(-1).expand_as(survival).clone()
    return SimpleNamespace(survival=survival, hazard=hazard)


def _references(events, durations):
    events = np.asarray(events, dtype=float)
    durations = np.asarray(durations, dtype=float)
    n, K = events.shape
    ref = np.zeros((n, 4 * K))
    ref[:, K : 2 * K] = events
    ref[:, 3 * K :] = durations
    return torch.tensor(ref, dtype=torch.float32)


def test_sample_ranking_penalises_wrong_order(cuts_file):
    """Subject 0 has the event at t=1, subject 1 later (t=3): subject 0 must be riskier."""
    loss = SampleRankingLoss(cuts_file, sigma=0.5, num_events=1)
    ref = _references([[1], [1]], [[1.0], [3.0]])
    right = loss(_predictions([[1.5], [0.2]]), ref)
    wrong = loss(_predictions([[0.2], [1.5]]), ref)
    assert right < wrong


def test_sample_ranking_uses_censored_later_subject(cuts_file):
    """Observed early vs censored later is an acceptable pair (set U of Eq. 4)."""
    loss = SampleRankingLoss(cuts_file, sigma=0.5, num_events=1)
    ref = _references([[1], [0]], [[1.0], [3.0]])
    right = loss(_predictions([[1.5], [0.2]]), ref)
    wrong = loss(_predictions([[0.2], [1.5]]), ref)
    assert right < wrong


def test_multievent_ranking_penalises_wrong_event_order(cuts_file):
    """One subject: event 1 at t=1, event 2 at t=3. Event 1 must carry the higher risk
    at t=1 (Eq. 5)."""
    loss = MultiEventRankingLoss(cuts_file, sigma=0.5, num_events=2)
    ref = _references([[1, 1]], [[1.0, 3.0]])
    right = loss(_predictions([[1.5, 0.2]]), ref)
    wrong = loss(_predictions([[0.2, 1.5]]), ref)
    assert right < wrong


def test_multievent_ranking_observed_vs_censored_event(cuts_file):
    """Event 1 observed at t=1, event 2 censored at t=3: also an acceptable pair."""
    loss = MultiEventRankingLoss(cuts_file, sigma=0.5, num_events=2)
    ref = _references([[1, 0]], [[1.0, 3.0]])
    right = loss(_predictions([[1.5, 0.2]]), ref)
    wrong = loss(_predictions([[0.2, 1.5]]), ref)
    assert right < wrong
