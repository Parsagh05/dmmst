"""Margin-Mean-Variance (MMV) loss for survival analysis.

Reference
---------
Adaptive Transformer Modelling of Density Function for Nonparametric Survival
Analysis (UniSurv), Machine Learning (2024), doi:10.1007/s10994-024-06686-w,
arXiv:2409.06209.

Definition (equation numbers follow the paper)
----------------------------------------------
Mean lifetime from the predicted density (Eq. 3):

    mu_hat = sum_{t=T_0}^{T_max} S_hat(t)

Variance of the predicted density about that mean (Eq. 4):

    v = sum_{t=T_0}^{T_max} p_hat_t * (t - mu_hat)^2

Margin ("best guess") event time for a censored subject, from the population
Kaplan-Meier estimator fitted on the *training* split (Eq. 5):

    e_m = T + [ integral_{T}^{T_max} S_km(t) dt ] / S_km(T)

Margin-mean loss (Eq. 6):

    L_mm = 1/2 * sum_i [ d_i * (mu_hat_i - T_i)^2
                       + (1 - d_i) * w_i * (mu_hat_i - e_m_i)^2 ]

with the censored-subject confidence weight

    w_i = 1 - S_km(T_i)

so that later censoring times -- where the best guess is better constrained --
carry more weight.

This class returns  L_mm + lambda_v * L_v.

Scope note
----------
The paper's full objective is  L_total = L_s + lam_m*L_mm + lam_v*L_v + lam_d*L_d
(Eq. 10). Only the margin-mean and variance terms are MMV-specific, so only those
live here. The other two already exist in this codebase and are composed via
``MetaLoss``:

  * L_s (softmax/likelihood) -> ``SATNLLPCHazardLoss``
  * L_d (discordant/ranking) -> ``SampleRankingLoss``

lambda_m is therefore the ``coeffs`` entry given to this loss inside ``MetaLoss``.

Note that Eq. 5 is the same "best guess" construction used in Eq. 6 of our own
paper, and is already implemented by ``sat.utils.km.KaplanMeierArea.best_guess``.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import Dict, List, Optional, Union

import pandas as pd
import torch

from sat.models.heads import SAOutput
from sat.utils import logging
from sat.utils.km import KaplanMeierArea

from ..balancing import BalancingStrategy
from ..base import Loss

logger = logging.get_default_logger()


class MMVLoss(Loss):
    """Margin-Mean-Variance loss (UniSurv, arXiv:2409.06209).

    Args:
        duration_cuts: CSV of duration cut points (one value per line, no header).
        training_set: CSV of transformed training labels, used to fit the
            per-event Kaplan-Meier curves that supply the margin times (Eq. 5).
        max_time: Time at which the survival function is taken to reach zero.
            Needed to close the final interval when integrating for mu_hat.
            Defaults to the largest duration cut if not given.
        importance_sample_weights: Optional CSV of per-event importance weights.
        num_events: Number of events modelled.
        variance_weight: lambda_v in Eq. 10. The paper grid-searches this over
            {0.001, 0.01, 0.1, 1}.
    """

    def __init__(
        self,
        duration_cuts: str,
        training_set: str,
        max_time: Optional[float] = None,
        importance_sample_weights: Optional[str] = None,
        num_events: int = 1,
        variance_weight: float = 0.01,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super(MMVLoss, self).__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )

        self.variance_weight = variance_weight

        # per-event importance weights (index 0 is censoring, as elsewhere)
        if importance_sample_weights is not None:
            df = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(df.weights.values).to(torch.float32)
        else:
            weights = torch.ones(self.num_events + 1)
        self.register_buffer("weights", weights)

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        cuts = torch.tensor(df.cuts.values, dtype=torch.float32)
        self.register_buffer("duration_cuts", cuts)
        self.num_time_bins = len(df.cuts)

        if max_time is None:
            # S is assumed to reach 0 at the final cut. This truncates E[T] at the
            # end of the study window - the standard limitation noted in the paper.
            max_time = float(cuts[-1])
        self.register_buffer("max_time", torch.tensor(float(max_time)))

        # Kaplan-Meier curves per event, fitted on the training split (Eq. 5)
        if training_set is None:
            raise ValueError(
                "MMVLoss requires the transformed training labels to fit the "
                "Kaplan-Meier curves used for the margin times."
            )
        self.kms: List[KaplanMeierArea] = []
        df = pd.read_csv(training_set, header=0)
        for event in range(self.num_events):
            training_event_times = df[f"duration_event{event + 1}"]
            training_event_indicators = df[f"event{event + 1}"] == 1
            self.kms.append(
                KaplanMeierArea(training_event_times, training_event_indicators)
            )

    def _time_grid(self, device, width: int) -> torch.Tensor:
        """Interval boundaries for a survival tensor of the given width.

        `survival` holds S at `width` time points, and we close the final
        interval by taking S = 0 at `max_time`, so we need exactly `width + 1`
        boundaries.

        The layout of duration_cuts.csv is not fixed: the file written by
        train_labeltransform *already begins at 0.0* (5 entries for 4 label
        transform cuts, matching the survival width), whereas a hand-written cut
        list may omit t=0. Handle both rather than assuming, since prepending a
        second zero silently produces one interval too many.
        """
        cuts = self.duration_cuts.to(device)
        if cuts.numel() == width - 1:
            cuts = torch.cat((torch.zeros(1, device=device), cuts))
        elif cuts.numel() != width:
            raise ValueError(
                f"Cannot align {cuts.numel()} duration cuts with a survival "
                f"tensor of width {width}; expected {width} or {width - 1}."
            )
        max_time = self.max_time.reshape(1).to(device)
        if float(max_time) <= float(cuts[-1]):
            # keep the final interval non-degenerate
            max_time = cuts[-1].reshape(1)
        return torch.cat((cuts, max_time))

    def mean_and_variance(self, survival: torch.Tensor):
        """Mean lifetime (Eq. 3) and density variance (Eq. 4).

        Args:
            survival: [batch, num_events, num_cuts + 1]; column 0 is S(0) = 1.

        Returns:
            (mu_hat, v), each [batch, num_events].
        """
        device = survival.device
        width = survival.shape[-1]
        grid = self._time_grid(device, width)  # [width + 1]
        widths = grid[1:] - grid[:-1]  # [width]

        # close the final interval: S is taken to be 0 at max_time
        zeros = torch.zeros(
            survival.shape[0],
            survival.shape[1],
            1,
            device=device,
            dtype=survival.dtype,
        )
        surv = torch.cat((survival, zeros), dim=2)  # [B, E, n + 2]

        # Eq. 3 -- mu_hat = integral of S, by the trapezoidal rule
        left, right = surv[:, :, :-1], surv[:, :, 1:]
        mu_hat = torch.sum(widths * (left + right) / 2.0, dim=2)

        # discrete density on each interval, and the interval midpoints
        pdf = left - right  # [B, E, n + 1], sums to 1
        midpoints = (grid[:-1] + grid[1:]) / 2.0

        # Eq. 4 -- variance of the predicted density about mu_hat
        centred = midpoints.view(1, 1, -1) - mu_hat.unsqueeze(2)
        v = torch.sum(pdf * centred.pow(2), dim=2)

        return mu_hat, v

    def margin_times(self, durations: torch.Tensor, events: torch.Tensor):
        """Margin event times e_m (Eq. 5) and censored weights w = 1 - S_km(T).

        Uncensored subjects keep their observed time and receive weight 1.
        """
        device = durations.device
        e_m = durations.clone()
        w = torch.ones_like(durations)

        durations_np = durations.detach().cpu().numpy()
        events_np = events.detach().cpu().numpy()

        for event in range(self.num_events):
            censored = events_np[:, event] == 0
            if not censored.any():
                continue
            censor_times = durations_np[censored, event]
            km = self.kms[event]
            best_guess = km.best_guess(censor_times)
            surv_at_censor = km.predict(censor_times)

            e_m[censored, event] = torch.tensor(
                best_guess, dtype=e_m.dtype, device=device
            )
            # w = 1 - S_km(T): later censoring times are trusted more
            w[censored, event] = torch.tensor(
                1.0 - surv_at_censor, dtype=w.dtype, device=device
            )

        return e_m, w

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        """Compute L_mm + lambda_v * L_v.

        Args:
            predictions: model output; ``survival`` is [batch, events, cuts + 1].
            references: [batch, 4 * num_events].
        """
        survival = predictions.survival
        durations = self.durations(references)  # [B, E]
        events = self.events(references)  # [B, E]

        mu_hat, v = self.mean_and_variance(survival)
        e_m, w = self.margin_times(durations, events)

        delta = events.to(mu_hat.dtype)

        # Eq. 6 -- observed subjects against the true time, censored subjects
        # against the margin time, down-weighted by w
        observed_term = delta * (mu_hat - durations).pow(2)
        censored_term = (1.0 - delta) * w * (mu_hat - e_m).pow(2)
        per_event = 0.5 * (observed_term + censored_term)

        # per-event importance weighting (index 0 is censoring)
        event_weights = self.weights[1:].to(per_event.device).view(1, -1)
        l_mm = torch.mean(torch.sum(per_event * event_weights, dim=1))
        l_v = torch.mean(torch.sum(v * event_weights, dim=1))

        loss = l_mm + self.variance_weight * l_v

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"MMV: L_mm={l_mm.item():.4f} L_v={l_v.item():.4f}")

        return self.ensure_tensor(loss, device=references.device)
