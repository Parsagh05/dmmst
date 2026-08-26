"""Initialization of the package

Scope: the loss components described in
"Dynamic Multi-modal & Multi-event Survival Transformer" (Sec. 2.3, 2.4).

    Loss = L_PCH + L_rank + L_mul                (Eq. 2, 4, 5)
    LossReg = L_MAE + L_MM                       (Eq. 6, 7)
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from .base import Loss, RankingLoss
from .classification.bce import CrossEntropyLoss
from .meta import MetaLoss
from .ranking.multievent import MultiEventRankingLoss
from .ranking.sample import SampleRankingLoss
from .regression.l1 import L1Loss
from .regression.mse import MSELoss
from .survival.mismatch import MismatchLoss
from .survival.mmv import MMVLoss
from .survival.nllpchazard import SATNLLPCHazardLoss

# --- baseline losses (comparison only; not part of the paper's method) ---
from .survival.deephit import DeepHitCalibrationLoss, DeepHitLikelihoodLoss
from .survival.dsm import DSMLoss
from .survival.dsm_paper import DSMPaperLoss
from .survival.mensa import MENSALoss
from .survival.mensa_paper import MENSAPaperLoss

__all__ = [
    "Loss",
    "RankingLoss",
    "MetaLoss",
    "CrossEntropyLoss",
    # L_rank (Eq. 4): ranks observations within an event type
    "SampleRankingLoss",
    # L_mul (Eq. 5): ranks event types within an observation
    "MultiEventRankingLoss",
    # L_PCH: piece-wise constant hazard likelihood
    "SATNLLPCHazardLoss",
    # L_MM (Eq. 7): mismatch penalty for the regression head
    "MismatchLoss",
    # L_MAE (Eq. 6): best-guess MAE with Kaplan-Meier extension
    "L1Loss",
    "MSELoss",
    # L_MMV: Margin-Mean-Variance loss (UniSurv, arXiv:2409.06209)
    "MMVLoss",
    # baselines
    "DeepHitLikelihoodLoss",
    "DeepHitCalibrationLoss",
    "DSMLoss",
    "DSMPaperLoss",
    "MENSALoss",
    "MENSAPaperLoss",
]
