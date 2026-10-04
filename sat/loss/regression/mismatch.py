"""Mismatch penalty L_MM of the regression task head (paper Sec. 2.4, Eq. 7).

For subject i let j be the first event that actually occurred (smallest observed time)
and q the event the regression head predicts to happen first (smallest y_hat). When
q != j the prediction gets the event order wrong and is penalised by

    L_MM = ReLU(y_hat_j - t_j) + ReLU(t_q - y_hat_q) + (y_hat_j - y_hat_q)       (Eq. 7)

* first term  - the true first event is predicted later than it happened;
* second term - the wrongly-first event is predicted before its time t_q;
* third term  - pushes the two predictions into the right order (it is >= 0 whenever
  q is the predicted first event, so no ReLU is needed).

What t_q is when event q was not observed. The paper's data model (Eq. 1) gives every
event of every subject a time: the event time if it was observed, otherwise the
censoring time T_c (for competing risks: the time the subject left the risk set).
``t_q="observed"`` uses exactly that time - the paper's definition. Because the true
time of an unobserved event is at least T_c, the second term is then a valid one-sided
bound. ``t_q="best_guess"`` replaces T_c by the Kaplan-Meier best guess of Eq. 6
(Haider et al. 2020), the alternative tested in version 2.

Subjects with no observed event have no j and contribute nothing. The loss is averaged
over the subjects that have at least one observed event (matched subjects add 0), so
its scale does not jump with the number of mismatches in a batch. With
``time_scale`` > 1 times are measured in units of ``time_scale`` (see the regression
head), which keeps the penalty on the same scale as the likelihood.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from typing import List

import numpy as np
import pandas as pd
import torch

from sat.models.heads import TaskOutput
from sat.utils import logging
from sat.utils.km import KaplanMeierArea

from ..base import Loss

logger = logging.get_default_logger()

T_Q_OPTIONS = ("observed", "best_guess")


class MismatchPenalty(Loss):
    def __init__(
        self,
        num_events: int,
        t_q: str = "observed",
        training_set: str = None,
        time_scale: float = 1.0,
    ):
        super().__init__(num_events=num_events)
        if num_events < 2:
            raise ValueError("the mismatch penalty needs at least two events")
        if t_q not in T_Q_OPTIONS:
            raise ValueError(f"t_q must be one of {T_Q_OPTIONS}, got {t_q!r}")
        self.t_q = t_q
        self.time_scale = float(time_scale)
        self.kms: List[KaplanMeierArea] = []
        if t_q == "best_guess":
            if training_set is None:
                raise ValueError("t_q='best_guess' needs the training labels")
            df = pd.read_csv(training_set, header=0, float_precision="round_trip")
            for k in range(num_events):
                self.kms.append(
                    KaplanMeierArea(
                        df[f"duration_event{k + 1}"].values,
                        (df[f"event{k + 1}"] == 1).values,
                    )
                )

    def _best_guess(self, event: int, times: torch.Tensor) -> torch.Tensor:
        t = times.detach().cpu().numpy().astype(float)
        bg = self.kms[event].best_guess(t)
        return torch.as_tensor(np.asarray(bg, dtype=float), dtype=times.dtype, device=times.device)

    def penalty(self, y_hat: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """y_hat: (batch, K) predicted times. Returns (batch,) per-subject penalties
        (0 for matched subjects and subjects without an observed event)."""
        events = self.events(references).to(torch.bool)  # (B, K)
        durations = self.durations(references).to(y_hat.dtype)  # (B, K)
        has_event = events.any(dim=1)

        inf = torch.full_like(durations, float("inf"))
        observed_times = torch.where(events, durations, inf)
        j = observed_times.argmin(dim=1)  # true first event
        q = y_hat.argmin(dim=1)  # predicted first event
        rows = torch.arange(len(y_hat), device=y_hat.device)
        t_j = durations[rows, j]

        # ties: q is not a mismatch if it was observed at the same time as j
        tie = events[rows, q] & (durations[rows, q] == t_j)
        mismatch = has_event & (q != j) & ~tie

        t_q = durations[rows, q].clone()
        if self.t_q == "best_guess":
            unobserved = mismatch & ~events[rows, q]
            for k in range(self.num_events):
                sel = unobserved & (q == k)
                if sel.any():
                    t_q[sel] = self._best_guess(k, t_q[sel])

        y_j, y_q = y_hat[rows, j], y_hat[rows, q]
        per_subject = (
            torch.relu(y_j - t_j) + torch.relu(t_q - y_q) + (y_j - y_q)
        ) / self.time_scale
        return torch.where(mismatch, per_subject, torch.zeros_like(per_subject))

    def forward(self, predictions: TaskOutput, references: torch.Tensor) -> torch.Tensor:
        y_hat = predictions.predictions
        y_hat = y_hat.reshape(len(y_hat), -1)
        has_event = self.events(references).to(torch.bool).any(dim=1)
        n = has_event.sum()
        if n == 0:
            return y_hat.sum() * 0.0  # keeps the graph, no subject to score
        return self.penalty(y_hat, references).sum() / n
