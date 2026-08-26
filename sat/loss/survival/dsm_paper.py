"""Deep Survival Machines loss, matching the published method.

Reference
---------
Nagpal, Li, Dubrawski, IEEE JBHI 2021. Reference implementation:
https://github.com/autonlab/auton-survival, `models/dsm/losses.py`.

`_conditional_weibull_loss` computes, per risk::

    ll = sum_uncensored log f(t) + alpha * sum_censored log S(t)
    loss = -ll / N

where ``alpha`` is DSM's ``discount`` on the censored contribution, and the mixture
is aggregated either exactly (``elbo=False``) or via DSM's ELBO surrogate
(``elbo=True``, the reference default for the main training phase).

DSM is a single-transition model: for risk ``p`` a subject is "uncensored" only if
that exact risk was the observed event, and contributes the survival term
otherwise. With SAT's per-event binary indicators that is ``events[:, p] == 1`` vs
``== 0``, which coincides with the reference's ``e == risk`` / ``e != risk`` under
the competing-risks assumption that at most one event fires.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import Dict, Optional, Union

import pandas as pd
import torch

from sat.models.heads import SAOutput
from sat.models.heads.dsm_paper import dsm_log_f_s
from sat.utils import logging

from ..balancing import BalancingStrategy
from ..base import Loss

logger = logging.get_default_logger()


class DSMPaperLoss(Loss):
    """DSM conditional Weibull loss, faithful to the reference.

    Args:
        duration_cuts: CSV of cut points; exposed so the head can build its grid.
        num_events: number of competing risks.
        discount: ``alpha``, the weight on the censored log-survival term.
        elbo: use DSM's ELBO surrogate for the mixture (reference default True).
        importance_sample_weights: optional per-event weights CSV.
    """

    def __init__(
        self,
        duration_cuts: str,
        num_events: int = 1,
        discount: float = 1.0,
        elbo: bool = True,
        importance_sample_weights: Optional[str] = None,
        balance_strategy: Optional[Union[str, BalancingStrategy]] = "fixed",
        balance_params: Optional[Dict] = None,
    ):
        super().__init__(
            num_events=num_events,
            balance_strategy=balance_strategy,
            balance_params=balance_params,
        )
        self.discount = float(discount)
        self.elbo = bool(elbo)

        df = pd.read_csv(duration_cuts, header=None, names=["cuts"])
        self.register_buffer(
            "duration_cuts", torch.tensor(df.cuts.values, dtype=torch.float32)
        )

        if importance_sample_weights is not None:
            w = pd.read_csv(importance_sample_weights, header=None, names=["weights"])
            weights = torch.tensor(w.weights.values, dtype=torch.float32)
        else:
            weights = torch.ones(num_events + 1)
        self.register_buffer("weights", weights)

    def forward(self, predictions: SAOutput, references: torch.Tensor) -> torch.Tensor:
        shape, scale, gate = predictions.shape, predictions.scale, predictions.logits_g
        if shape is None or scale is None or gate is None:
            raise ValueError(
                "DSMPaperLoss needs shape/scale/logits_g on the model output; "
                "use it with DSMPaperTaskHead."
            )

        events = self.events(references).float()
        durations = self.durations(references).float()
        device = references.device
        n = events.shape[0]
        w = self.weights[1:].to(device)

        total = torch.zeros((), device=device)
        for p in range(self.num_events):
            log_f, log_s = dsm_log_f_s(
                shape[:, p], scale[:, p], gate[:, p], durations[:, p], elbo=self.elbo
            )
            uncens = events[:, p] > 0
            # ll = sum_uncens log f + alpha * sum_cens log S   (reference)
            ll = (uncens * log_f).sum() + self.discount * ((~uncens) * log_s).sum()
            total = total + w[p] * ll

        loss = -total / n
        if not torch.isfinite(loss):
            logger.warning("non-finite DSM loss; returning zero for this batch")
            return torch.zeros((), device=device, requires_grad=True)
        return loss
