"""Within-subject event-ordering concordance, defined exactly like the paper's L_mul.

Why this exists
---------------
``ComputeWithinSubjectCIndex`` (``within_subject_ipcw``) scores each subject with
torchsurv's C-index at a *fixed* horizon and only over pairs of *observed* events. It
therefore drops every observed-vs-censored pair and compares risks at tau rather than
at the time the ordering is decided. Neither matches Eq. 5.

Here, for every subject i the admissible pairs are exactly the set E_i of Eq. 5:
event a observed at t_a, and event b either observed later or censored after t_a
(ties are not orderable and are skipped). The pair is concordant when

    CIF_a(t_a | x_i) > CIF_b(t_a | x_i)

i.e. at t_a the model already considers the event that did happen more likely than
the one that had not. Ties in prediction count 1/2.

``within_ctd_km`` scores the same pairs with population Kaplan-Meier curves fitted on
the training labels - the best a model can do *without* using covariates. A model has
learned subject-specific ordering only if it beats this reference, which is what
distinguishes real within-subject learning from reproducing the population order.

Reported:
    within_ctd           pair-pooled concordance
    within_ctd_subject   mean over subjects of their own concordance
    within_ctd_km        the same pairs scored by population KM curves
    within_ctd_gain      within_ctd - within_ctd_km
    within_ctd_pairs     number of admissible pairs
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from typing import Optional

import numpy as np
import pandas as pd

from sat.utils import logging

logger = logging.get_default_logger()


def _interp_cif(grid: np.ndarray, cif: np.ndarray, t: np.ndarray) -> np.ndarray:
    """CIF at time t for each row, log-linear in survival between grid points
    (the piecewise-constant-hazard assumption of the model).

    grid: (G,) starting at 0; cif: (n, G) with cif[:, 0] = 0; t: (n,).
    """
    surv = np.clip(1.0 - cif, 1e-12, 1.0)
    t = np.clip(t, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    t0, t1 = grid[j], grid[j + 1]
    rows = np.arange(len(t))
    s0, s1 = surv[rows, j], surv[rows, j + 1]
    frac = np.where(t1 > t0, (t - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
    return 1.0 - s0 * (s1 / s0) ** frac


def within_subject_concordance(events, durations, cif_fn):
    """Pair-pooled and subject-averaged concordance.

    events, durations: (n, K). cif_fn(k, t) -> CIF of event k at times t (n,).
    """
    n, K = events.shape
    conc_sum = np.zeros(n)
    pair_cnt = np.zeros(n)
    for a in range(K):
        ta = durations[:, a]
        ra = cif_fn(a, ta)
        for b in range(K):
            if a == b:
                continue
            ok = (events[:, a] == 1) & (ta < durations[:, b])
            if not ok.any():
                continue
            rb = cif_fn(b, ta)
            c = (ra > rb) + 0.5 * (ra == rb)
            conc_sum += np.where(ok, c, 0.0)
            pair_cnt += ok
    total = pair_cnt.sum()
    if total == 0:
        return float("nan"), float("nan"), 0
    has = pair_cnt > 0
    return (
        float(conc_sum.sum() / total),
        float(np.mean(conc_sum[has] / pair_cnt[has])),
        int(total),
    )


def _km_cif(durations: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """1 - Kaplan-Meier survival evaluated on grid."""
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    uniq = np.unique(d[e == 1])
    if len(uniq) == 0:
        return np.zeros_like(grid, dtype=float)
    at_risk = len(d) - np.searchsorted(d, uniq, side="left")
    deaths = np.array([np.sum((d == u) & (e == 1)) for u in uniq])
    steps = np.cumprod(1.0 - deaths / np.maximum(at_risk, 1))
    idx = np.searchsorted(uniq, grid, side="right") - 1
    surv = np.where(idx >= 0, steps[np.clip(idx, 0, None)], 1.0)
    return 1.0 - surv


class WithinSubjectOrdering:
    """Evaluation module; plug into a metrics yaml next to SurvTRACEMetrics."""

    def __init__(self, cfg, duration_cuts: str, training_set: Optional[str] = None):
        self.cfg = cfg
        self.cuts = pd.read_csv(duration_cuts, header=None, names=["cuts"], float_precision="round_trip").cuts.values.astype(float)
        self.km = None
        if training_set is not None:
            df = pd.read_csv(training_set, header=0, float_precision="round_trip")
            fine = np.linspace(0.0, self.cuts[-1], 512)
            self.km = (
                fine,
                np.stack(
                    [
                        _km_cif(
                            df[f"duration_event{k + 1}"].values.astype(float),
                            (df[f"event{k + 1}"].values == 1).astype(int),
                            fine,
                        )
                        for k in range(cfg.num_events)
                    ]
                ),
            )

    def compute(self, predictions, references):
        from sat.evaluate.eval_modules import SurvivalEvaluationModule

        K = self.cfg.num_events
        if K < 2:
            return {}
        preds = SurvivalEvaluationModule.survival_predictions(self, predictions)
        risk = np.asarray(preds[:, 1], dtype=float)  # (n, K, C) aligned to cuts[1:]
        grid = self.cuts
        if risk.shape[-1] == len(grid) - 1:
            risk = np.concatenate([np.zeros(risk.shape[:2] + (1,)), risk], axis=-1)
        elif risk.shape[-1] != len(grid):
            logger.warning(f"risk has {risk.shape[-1]} columns for {len(grid)} cuts; skipping")
            return {}

        refs = np.asarray(references, dtype=float)
        events = refs[:, K : 2 * K].astype(int)
        durations = refs[:, 3 * K : 4 * K]

        pooled, subject, pairs = within_subject_concordance(
            events, durations, lambda k, t: _interp_cif(grid, risk[:, k, :], t)
        )
        out = {
            "within_ctd": pooled,
            "within_ctd_subject": subject,
            "within_ctd_pairs": pairs,
        }
        if self.km is not None:
            fine, km = self.km
            n = len(events)
            km_pooled, _, _ = within_subject_concordance(
                events,
                durations,
                lambda k, t: _interp_cif(fine, np.broadcast_to(km[k], (n, len(fine))), t),
            )
            out["within_ctd_km"] = km_pooled
            out["within_ctd_gain"] = pooled - km_pooled
        return out
