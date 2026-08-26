"""MENSA loss, implemented to match the published method.

Reference
---------
MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
Likelihood Estimation. ML4H 2025. arXiv:2409.06525.
Reference implementation: https://github.com/thecml/mensa (`src/mensa/loss.py`).

Multi-event likelihood (paper Eq. 7)
------------------------------------
    L_ME = sum_i sum_p w_p [ delta_ip log f_p(t_ip | x_i)
                             + (1 - delta_ip) log S_p(t_ip | x_i) ]

with ``w_p`` the inverse frequency of transitions into state p. Each event has its
own observed time ``t_ip``; events are treated as conditionally independent given
the covariates.

Trajectory likelihood (paper Eq. 8)
-----------------------------------
    L_traj = sum_i sum_{(A,B) in T} delta_iA delta_iB log S_B(T_A | x_i)

For a known ordering A -> B, when both events are observed the model should still
assign high survival to B at the time A occurred.

Combined (paper Eq. 9)
----------------------
    L_total = (1 - lambda) * L_ME / N + lambda * L_traj / N

Both terms are log-likelihoods to be *maximised*; this module returns the negative.

A discrepancy worth knowing about
---------------------------------
The paper's Eq. 8 evaluates ``S_B`` at ``T_A``. The authors' released code passes the
survival array already evaluated at each event's *own* time, so it effectively uses
``S_B(T_B)``:

    f, s = self.compute_risks_multi(params, ti)     # s[:, j] = log S_j(t_j)
    traj_loss += trajectory_loss(i, j, ei, s)       # uses s[mask, j]

``trajectory_time`` selects between them. The default ``"paper"`` implements Eq. 8 as
written; ``"reference"`` reproduces the released code. They differ whenever
``T_A != T_B``, which is exactly the multi-event case the term is meant to address.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch

from sat.models.heads import SAOutput
from sat.models.heads.mensa_paper import weibull_log_f_s
from sat.utils import logging

from ..balancing import BalancingStrategy
from ..base import Loss

logger = logging.get_default_logger()


class MENSAPaperLoss(Loss):
    """Paper-faithful MENSA loss: Eq. 7 + Eq. 8, combined per Eq. 9.

    Args:
        duration_cuts: CSV of cut points. Only used to expose ``duration_cuts`` to
            the head so it can build its evaluation grid; the likelihood itself is
            evaluated at each subject's own observed time, as in the paper.
        num_events: number of modelled events (states).
        trajectories: known orderings as ``[[A, B], ...]`` (0-indexed). ``A -> B``
            means A is expected to precede B. Empty disables the trajectory term,
            leaving pure Eq. 7.
        traj_lambda: ``lambda`` in Eq. 9, in [0, 1].
        event_weights: ``w_p``. ``None`` uses the paper's inverse transition
            frequency, estimated from the training labels when available.
        training_set: transformed training labels, used to estimate ``w_p``.
        trajectory_time: ``"paper"`` (Eq. 8, S_B at T_A) or ``"reference"``
            (released code, S_B at T_B).
    """

    def __init__(
        self,
        duration_cuts: str,
        num_events: int = 1,
        trajectories: Optional[Sequence[Sequence[int]]] = None,
        traj_lambda: float = 0.0,
        event_weights: Optional[Sequence[float]] = None,
        training_set: Optional[str] = None,
        trajectory_time: str = "paper",
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super().__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )
        if trajectory_time not in ("paper", "reference"):
            raise ValueError("trajectory_time must be 'paper' or 'reference'")
        self.trajectory_time = trajectory_time

        if not 0.0 <= traj_lambda <= 1.0:
            raise ValueError(f"traj_lambda must be in [0, 1], got {traj_lambda}")
        self.traj_lambda = float(traj_lambda)

        self.trajectories: List[Tuple[int, int]] = [
            (int(a), int(b)) for a, b in (trajectories or [])
        ]
        for a, b in self.trajectories:
            if not (0 <= a < num_events and 0 <= b < num_events):
                raise ValueError(f"trajectory ({a}, {b}) outside [0, {num_events})")
        if self.traj_lambda > 0 and not self.trajectories:
            logger.warning(
                "traj_lambda > 0 but no trajectories given: the trajectory term is "
                "identically zero, so this is just Eq. 7 scaled by (1 - lambda)."
            )

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32)
        )

        # w_p: inverse frequency of transitions into each state (paper Sec. 3.3)
        if event_weights is not None:
            w = torch.tensor(list(event_weights), dtype=torch.float32)
        elif training_set is not None:
            try:
                labels = pd.read_csv(training_set, header=0)
                freq = torch.tensor(
                    [
                        float((labels[f"event{p + 1}"] == 1).mean())
                        for p in range(num_events)
                    ],
                    dtype=torch.float32,
                )
                w = 1.0 / torch.clamp(freq, min=1e-6)
                w = w / w.mean()  # keep the loss on a comparable scale
                logger.info(f"MENSA inverse-frequency event weights: {w.tolist()}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not estimate event weights ({e}); using 1.0")
                w = torch.ones(num_events)
        else:
            w = torch.ones(num_events)
        self.register_buffer("event_weights", w)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        shape = predictions.shape  # [n, events, dists]
        scale = predictions.scale
        gate = predictions.logits_g
        if shape is None or scale is None or gate is None:
            raise ValueError(
                "MENSAPaperLoss needs shape/scale/logits_g on the model output; "
                "use it with MENSAPaperTaskHead."
            )

        events = self.events(references).float()  # [n, events]
        durations = self.durations(references).float()  # [n, events]
        device = references.device
        n = events.shape[0]
        w = self.event_weights.to(device)

        # --- Eq. 7: multi-event likelihood, each event at its own observed time ---
        log_f, log_s = [], []
        for p in range(self.num_events):
            f_p, s_p = weibull_log_f_s(
                shape[:, p], scale[:, p], gate[:, p], durations[:, p]
            )
            log_f.append(f_p)
            log_s.append(s_p)
        log_f = torch.stack(log_f, dim=1)  # [n, events]
        log_s = torch.stack(log_s, dim=1)

        observed = events > 0
        ll = torch.where(observed, log_f, log_s)  # delta*log f + (1-delta)*log S
        l_me = (ll * w.unsqueeze(0)).sum() / n

        # --- Eq. 8: trajectory term ---
        l_traj = torch.zeros((), device=device)
        if self.traj_lambda > 0 and self.trajectories:
            for a, b in self.trajectories:
                mask = observed[:, a] & observed[:, b]
                if not mask.any():
                    continue
                if self.trajectory_time == "paper":
                    # S_B evaluated at T_A, as Eq. 8 states
                    _, s_b_at_ta = weibull_log_f_s(
                        shape[mask, b], scale[mask, b], gate[mask, b],
                        durations[mask, a],
                    )
                else:
                    s_b_at_ta = log_s[mask, b]  # released code: S_B(T_B)
                l_traj = l_traj + s_b_at_ta.mean()

        # Eq. 9. Both terms are log-likelihoods to maximise, so negate for a loss.
        total = (1.0 - self.traj_lambda) * l_me + self.traj_lambda * l_traj
        loss = -total

        if not torch.isfinite(loss):
            logger.warning("non-finite MENSA loss; returning zero for this batch")
            return torch.zeros((), device=device, requires_grad=True)
        return loss
