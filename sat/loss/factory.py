"""Loss recipes for version 2: every combination in one place.

The paper's survival loss is L_PCH + L_rank + L_mul (Sec. 2.3) and its regression loss
L_MAE + L_MM (Sec. 2.4). Instead of one config file per combination, the two
builders below take one weight per term and include a term only when its weight is
> 0, so every recipe of the version-2 plan is a set of overrides:

    S1 L_PCH              rank_coeff=0  mul_coeff=0
    S2 L_PCH + L_rank     rank_coeff>0  mul_coeff=0
    S3 L_PCH + L_mul      rank_coeff=0  mul_coeff>0      (needs >= 2 events)
    S4 L_PCH + both       rank_coeff>0  mul_coeff>0
    + MMV                 mmv_coeff>0                    (UniSurv reference, not in the paper)

    R1 MAE, observed only        l1_type=uncensored  mm_coeff=0
    R2 best-guess MAE (Eq. 6)    l1_type=margin      mm_coeff=0
    R3 R2 + L_MM, t_q from Eq. 1 l1_type=margin      mm_coeff>0  mm_t_q=observed
    R4 R2 + L_MM, best-guess t_q l1_type=margin      mm_coeff>0  mm_t_q=best_guess

Terms are combined with fixed weights (MetaLoss, balance_strategy="fixed"): the paper
simply adds them.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from typing import Optional

from .meta import MetaLoss
from .ranking.multievent import MultiEventRankingLoss
from .ranking.sample import SampleRankingLoss
from .regression.l1 import L1Loss
from .regression.mismatch import MismatchPenalty
from .survival.mmv import MMVLoss
from .survival.nllpchazard import SATNLLPCHazardLoss


def survival_loss(
    duration_cuts: str,
    importance_sample_weights: str,
    training_set: str,
    num_events: int,
    max_time: Optional[float] = None,
    likelihood_coeff: float = 1.0,
    rank_coeff: float = 0.0,
    rank_sigma: float = 0.1,
    mul_coeff: float = 0.0,
    mul_sigma: float = 0.1,
    mmv_coeff: float = 0.0,
    mmv_variance_weight: float = 0.01,
    time_scale: float = 1.0,
) -> MetaLoss:
    losses = [SATNLLPCHazardLoss(importance_sample_weights, num_events)]
    coeffs = [float(likelihood_coeff)]
    if rank_coeff > 0:
        losses.append(
            SampleRankingLoss(duration_cuts, importance_sample_weights, rank_sigma, num_events)
        )
        coeffs.append(float(rank_coeff))
    if mul_coeff > 0:
        if num_events < 2:
            raise ValueError("L_mul (within-subject event ranking) needs >= 2 events")
        losses.append(
            MultiEventRankingLoss(duration_cuts, importance_sample_weights, mul_sigma, num_events)
        )
        coeffs.append(float(mul_coeff))
    if mmv_coeff > 0:
        losses.append(
            MMVLoss(
                duration_cuts=duration_cuts,
                training_set=training_set,
                max_time=max_time,
                importance_sample_weights=importance_sample_weights,
                num_events=num_events,
                variance_weight=mmv_variance_weight,
                time_scale=time_scale,
            )
        )
        coeffs.append(float(mmv_coeff))
    return MetaLoss(losses, coeffs, balance_strategy="fixed", num_events=num_events)


def regression_loss(
    training_set: str,
    importance_sample_weights: str,
    num_events: int,
    l1_type: str = "margin",
    mm_coeff: float = 0.0,
    mm_t_q: str = "observed",
    time_scale: float = 1.0,
) -> MetaLoss:
    if l1_type not in ("uncensored", "margin", "hinge"):
        raise ValueError(f"unknown l1_type {l1_type!r}")
    losses = [
        L1Loss(
            training_set=training_set,
            importance_sample_weights=importance_sample_weights,
            l1_type=l1_type,
            num_events=num_events,
            time_scale=time_scale,
        )
    ]
    coeffs = [1.0]
    if mm_coeff > 0:
        losses.append(
            MismatchPenalty(
                num_events=num_events,
                t_q=mm_t_q,
                training_set=training_set,
                time_scale=time_scale,
            )
        )
        coeffs.append(float(mm_coeff))
    return MetaLoss(losses, coeffs, balance_strategy="fixed", num_events=num_events)
