"""Pair-wise learning-to-rank penalty terms (paper Sec. 2.3)

L_rank (Eq. 4) enforces globally consistent ranking of individual risk;
L_mul  (Eq. 5) enforces locally consistent ranking of event risk within an individual.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from .multievent import MultiEventRankingLoss  # noqa
from .sample import SampleRankingLoss  # noqa

__all__ = ["SampleRankingLoss", "MultiEventRankingLoss"]
