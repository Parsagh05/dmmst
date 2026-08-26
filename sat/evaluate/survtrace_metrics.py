"""Evaluation metrics matched to the SurvTRACE protocol.

Why this exists
---------------
The metrics in ``eval_modules.py`` are not comparable with the numbers published in
SurvTRACE (Wang & Sun, BCB '22, Table 2), DeepHit, DSM etc., for two reasons:

1. **No truncation.** ``ComputeCIndex`` loops over the duration cuts and takes the
   risk estimate at each one, but never passes ``tmax``/``tau`` to the concordance
   calculator. Every "horizon" therefore scores concordance over the *whole*
   follow-up. The published protocol truncates at ``tau``. The tell-tale symptom is
   a C-index that stays flat across horizons (0.571 / 0.577 / 0.583) where the
   published values fall (0.713 / 0.680 / 0.644).

2. **Censoring distribution taken from the test set.** IPCW weights were estimated
   with ``get_ipcw(e_test, t_test)``. The standard estimator fits the censoring
   distribution on the *training* split.

This module reproduces the reference protocol exactly, using the same library the
SurvTRACE authors used:

    from sksurv.metrics import concordance_index_ipcw, brier_score
    concordance_index_ipcw(et_train, et_test, estimate=risk[:, i + 1], tau=times[i])
    brier_score(et_train, et_test, surv[:, 1:-1], times)

with ``times`` the 25/50/75% quantiles of the *uncensored* event times, which is
already what ``train_labeltransform`` writes into ``duration_cuts.csv``.
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import Optional

import numpy as np
import pandas as pd

from sat.utils import logging

logger = logging.get_default_logger()

try:
    from sksurv.metrics import brier_score as sksurv_brier_score
    from sksurv.metrics import concordance_index_ipcw

    HAVE_SKSURV = True
except ImportError:  # pragma: no cover
    HAVE_SKSURV = False
    logger.warning(
        "scikit-survival is not installed; the SurvTRACE-matched ctd_* metrics "
        "will be absent (everything else still runs). Install the release that "
        "matches your numpy: numpy>=2 -> 'scikit-survival', numpy<2 -> "
        "'scikit-survival==0.22.2'. Do NOT move numpy to satisfy it: pandas and "
        "pyarrow are compiled against it."
    )


def _structured(events, durations):
    """Build the (bool event, float time) structured array sksurv expects."""
    events = np.asarray(events).astype(bool)
    durations = np.asarray(durations).astype(float)
    return np.array(
        list(zip(events, durations)), dtype=[("e", bool), ("t", float)]
    )


class SurvTRACEMetrics:
    """C-index (IPCW, truncated) and Brier score, per the SurvTRACE protocol.

    Args:
        cfg: the ``data`` config node (supplies ``num_events``).
        duration_cuts: CSV of cut points ``[0, q25, q50, q75, max]``.
        training_set: ``transformed_train_labels.csv``. Supplies the *raw* training
            durations and event indicators used to fit the censoring distribution.
            Without it the estimator falls back to the test split, which is what
            made the previous numbers incomparable.
    """

    def __init__(self, cfg, duration_cuts: str, training_set: Optional[str] = None):
        self.cfg = cfg

        cuts = pd.read_csv(duration_cuts, header=None, names=["cuts"]).cuts.values
        # evaluation horizons: drop the leading 0 and the trailing max, exactly as
        # SurvTRACE does with duration_index[1:-1]
        self.times = cuts[1:-1]
        self.horizons = [0.25, 0.5, 0.75][: len(self.times)]

        self.train = {}
        if training_set is None:
            logger.warning(
                "No training_set given: IPCW will be estimated on the test split, "
                "which is NOT the published protocol and is not comparable."
            )
        else:
            df = pd.read_csv(training_set, header=0)
            for event in range(self.cfg.num_events):
                self.train[event] = _structured(
                    df[f"event{event + 1}"] == 1, df[f"duration_event{event + 1}"]
                )

    def compute_event(self, predictions, references, event):
        """predictions: [n, 3, num_events, n_cuts] as (hazard, risk, survival)."""
        out = {}
        if not HAVE_SKSURV:
            return out
        if predictions.size == 0 or predictions.ndim < 4:
            logger.warning(f"invalid predictions for event {event}: {predictions.shape}")
            return out

        events_test = references[:, (1 * self.cfg.num_events + event)].astype(bool)
        durations_test = references[:, (3 * self.cfg.num_events + event)]
        et_test = _structured(events_test, durations_test)
        et_train = self.train.get(event, et_test)

        risk = predictions[:, 1, event]  # [n, n_cuts]
        surv = predictions[:, 2, event]

        # sksurv refuses times outside the follow-up of the training data
        t_max_train = et_train["t"].max()
        t_max_test = durations_test.max()
        usable = [
            i
            for i, t in enumerate(self.times)
            if t < min(t_max_train, t_max_test) and (durations_test >= t).any()
        ]
        if not usable:
            logger.warning("no usable evaluation horizons for event %d", event)
            return out

        cis, brs = [], []
        for i in usable:
            tau = float(self.times[i])
            try:
                ci = concordance_index_ipcw(
                    et_train, et_test, estimate=risk[:, i], tau=tau
                )[0]
                cis.append(ci)
                out[f"ctd_{event}th_event_{self.horizons[i]}"] = float(ci)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"C-index failed at tau={tau}: {e}")

        try:
            idx = np.array(usable)
            # sksurv's Brier needs every test time inside the training censoring
            # support; the hash splitter does not guarantee the longest-duration
            # subject lands in train, so restrict rather than fail.
            keep = durations_test < et_train["t"].max()
            if keep.sum() > 0:
                _, bs = sksurv_brier_score(
                    et_train, et_test[keep], surv[keep][:, idx],
                    self.times[idx].astype(float),
                )
                for j, i in enumerate(usable):
                    brs.append(bs[j])
                    out[f"brier_{event}th_event_{self.horizons[i]}"] = float(bs[j])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Brier score failed: {e}")

        if cis:
            out[f"ctd_{event}th_event"] = float(np.mean(cis))
        if brs:
            out[f"brier_{event}th_event"] = float(np.mean(brs))
        return out

    def compute(self, predictions, references):
        from sat.evaluate.eval_modules import SurvivalEvaluationModule

        predictions = SurvivalEvaluationModule.survival_predictions(self, predictions)
        metrics = {}
        ctd, brier = [], []
        for event in range(self.cfg.num_events):
            m = self.compute_event(predictions, references, event)
            metrics.update(m)
            if f"ctd_{event}th_event" in m:
                ctd.append(m[f"ctd_{event}th_event"])
            if f"brier_{event}th_event" in m:
                brier.append(m[f"brier_{event}th_event"])
        if ctd:
            metrics["ctd_weighted_avg"] = float(np.mean(ctd))
        if brier:
            metrics["brier_survtrace_weighted_avg"] = float(np.mean(brier))
        return metrics


class SurvivalMAEMetrics:
    """Time-to-event error implied by the predicted survival curve.

    Why this exists
    ---------------
    MMV (UniSurv Eq. 3-6) optimises the *mean lifetime* of the predicted density,
    not a ranking. Judging it by concordance alone measures something it does not
    target, and the survival task head emits no ``time_to_event``, so the existing
    ``ComputeL1`` regression metric cannot be used with it.

    This derives the same quantity MMV optimises directly from the survival curve

        mu_hat = integral of S(t) dt        (UniSurv Eq. 3)

    and scores it two ways:

    * ``mae_uncensored`` - |mu_hat - T| over observed events only. Simple, but
      biased: it silently discards the censored majority.
    * ``mae_margin`` - Haider et al.'s margin MAE. Observed subjects are scored
      against T, censored subjects against the Kaplan-Meier best guess
      ``e_m = T + integral_T^inf S_km / S_km(T)``, weighted by ``1 - S_km(T)``.
      This is exactly the target of MMV's L_mm, so it is the fair test.

    Note the same truncation caveat as everywhere else: S is only known out to the
    last duration cut, so mu_hat is a lower bound on the true expected lifetime.
    Fine for *comparing* models on one dataset, not an absolute lifetime estimate.
    """

    def __init__(self, cfg, duration_cuts: str, training_set: Optional[str] = None):
        from sat.utils.km import KaplanMeierArea

        self.cfg = cfg
        cuts = pd.read_csv(duration_cuts, header=None, names=["cuts"]).cuts.values
        self.cuts = np.asarray(cuts, dtype=float)

        self.kms = {}
        if training_set is not None:
            df = pd.read_csv(training_set, header=0)
            for event in range(self.cfg.num_events):
                self.kms[event] = KaplanMeierArea(
                    df[f"duration_event{event + 1}"], df[f"event{event + 1}"] == 1
                )

    def _mean_lifetime(self, surv):
        """UniSurv Eq. 3: mu_hat = integral of S, trapezoidal over the cut grid."""
        grid = self.cuts
        if surv.shape[-1] == len(grid) - 1:
            grid = grid[1:]
        elif surv.shape[-1] != len(grid):
            grid = grid[: surv.shape[-1]]
        widths = np.diff(grid)
        left, right = surv[:, :-1], surv[:, 1:]
        return np.sum(widths * (left + right) / 2.0, axis=1)

    def compute_event(self, predictions, references, event):
        out = {}
        if predictions.size == 0 or predictions.ndim < 4:
            return out

        events_test = references[:, (1 * self.cfg.num_events + event)].astype(bool)
        durations_test = references[:, (3 * self.cfg.num_events + event)].astype(float)
        surv = predictions[:, 2, event]

        mu = self._mean_lifetime(surv)

        obs = events_test
        if obs.any():
            out[f"mae_uncensored_{event}th_event"] = float(
                np.mean(np.abs(mu[obs] - durations_test[obs]))
            )

        km = self.kms.get(event)
        if km is not None:
            target = durations_test.astype(float).copy()
            weight = np.ones_like(target)
            cens = ~events_test
            if cens.any():
                ct = durations_test[cens]
                try:
                    target[cens] = km.best_guess(ct)
                    weight[cens] = 1.0 - km.predict(ct)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"margin MAE unavailable: {e}")
                    weight[cens] = 0.0
            denom = weight.sum()
            if denom > 0:
                out[f"mae_margin_{event}th_event"] = float(
                    np.sum(weight * np.abs(mu - target)) / denom
                )
        return out

    def compute(self, predictions, references):
        from sat.evaluate.eval_modules import SurvivalEvaluationModule

        predictions = SurvivalEvaluationModule.survival_predictions(self, predictions)
        metrics, unc, mar = {}, [], []
        for event in range(self.cfg.num_events):
            m = self.compute_event(predictions, references, event)
            metrics.update(m)
            if f"mae_uncensored_{event}th_event" in m:
                unc.append(m[f"mae_uncensored_{event}th_event"])
            if f"mae_margin_{event}th_event" in m:
                mar.append(m[f"mae_margin_{event}th_event"])
        if unc:
            metrics["mae_uncensored"] = float(np.mean(unc))
        if mar:
            metrics["mae_margin"] = float(np.mean(mar))
        return metrics
