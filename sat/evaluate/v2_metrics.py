"""Version-2 metric suite (everything beyond the SurvTRACE C_td / Brier columns).

Survival curves (any model with a survival head), per event k:

* ``ibs`` - integrated Brier score (scikit-survival ``integrated_brier_score``),
  censoring distribution from the TRAINING rows, over the 1st-99th percentile of the
  test follow-up;
* ``ibll`` - integrated binomial log-likelihood (Kvamme & Borgan), same weights and grid;
* ``antolini`` - Antolini's time-dependent C-index on the whole curve, curves read at
  the exact event times (see ``antolini``);
* ``auc`` - cumulative/dynamic AUC at the 25/50/75% horizons (scikit-survival);
* ``dcal_p`` / ``onecal_p`` - D-calibration and 1-calibration at the median horizon
  (Haider et al. 2020; SurvivalEVAL), as p-values (> 0.05 = calibrated);
* ``mae_<tail>_<method>`` - error of the expected time E[T] = integral of S, under the
  three ways of closing the curve that paper Sec. 2.4 discusses:
  ``trunc`` (stop at the last cut, a lower bound), ``tmax`` (the curve conditioned on
  T <= end of follow-up, Hu et al. 2021) and ``linear`` (extend the line from (0, 1) through the
  last point to S = 0, Haider et al. 2020); scored with SurvivalEVAL's ``uncensored``,
  ``hinge``, ``margin`` and ``pseudo_obs`` MAE.

Regression-head times (Sec. 2.4), per event k:

* ``reg_mae_<method>`` - the same four MAE definitions on the predicted times;
* ``reg_harrell`` - Harrell's C on the predicted times (Hu et al. 2021 use it).

Cheap metrics are computed at every evaluation; the rest only when ``self.final`` is
True, which finetune.py sets before the final validation / test predictions.
``reg_mae_margin`` is always computed: regression-only models select checkpoints on it.
Keys without an event suffix average over events.
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from typing import Dict, Optional

import numpy as np
import pandas as pd

from sat.utils import logging

logger = logging.get_default_logger()

MAE_METHODS = ("uncensored", "hinge", "margin", "pseudo_obs")
TAILS = ("trunc", "tmax", "linear")
_EPS = 1e-7


# ----------------------------------------------------------------- curve helpers
def expected_time(grid: np.ndarray, surv: np.ndarray, tail: str = "linear") -> np.ndarray:
    """E[T] = integral of S(t) dt for survival values on ``grid`` (grid[0] = 0, S = 1).

    Inside each interval S decays exponentially (the piecewise-constant hazard the model
    assumes), so the area is exact: (b - a) (S_a - S_b) / ln(S_a / S_b).
    Beyond the last grid point the curve is closed by ``tail``:
      trunc  - no area after the last point (lower bound);
      tmax   - S(T_max) = 0 with T_max = the last grid point (Hu et al. 2021): their
               model is a distribution over [0, T_max], i.e. the curve conditioned on
               T <= T_max, so E[T] = (integral_0^T_max S - S(T_max) T_max) / (1 - S(T_max)).
               This does not depend on how finely the curve is sampled;
      linear - continue the line from (0, 1) through (t_K, S_K) down to 0
               (Haider et al. 2020).
    """
    grid = np.asarray(grid, dtype=float)
    S = np.clip(np.asarray(surv, dtype=float), _EPS, 1.0)
    width = np.diff(grid)[None, :]
    a, b = S[:, :-1], S[:, 1:]
    ratio = np.log(a / b)
    flat = np.abs(ratio) < 1e-9
    area = np.where(flat, width * a, width * (a - b) / np.where(flat, 1.0, ratio))
    total = area.sum(axis=1)
    if tail == "tmax":
        SK = np.clip(S[:, -1], None, 1 - _EPS)
        return (total - SK * grid[-1]) / (1.0 - SK)
    if tail == "trunc":
        return total
    if tail == "linear":
        tK, SK = grid[-1], S[:, -1]
        t0 = tK / np.clip(1.0 - SK, _EPS, None)  # where the line hits 0
        return total + SK * (t0 - tK) / 2.0
    raise ValueError(f"unknown tail {tail!r}")


def fine_curves(cuts: np.ndarray, surv_on_cuts: np.ndarray, n: int = 100):
    """Survival on a fine grid, log-linear between cuts (exact under the PCH model).
    surv_on_cuts: (n_subjects, len(cuts)) including t = 0."""
    from sat.evaluate.multievent_metrics import _interp_cif

    cuts = np.asarray(cuts, dtype=float)
    grid = np.linspace(0.0, cuts[-1], n)
    cif = 1.0 - surv_on_cuts
    m = len(cif)
    fine = np.stack([1.0 - _interp_cif(cuts, cif, np.full(m, t)) for t in grid], axis=1)
    return grid, np.clip(fine, 0.0, 1.0)


def _structured(e, t):
    return np.array(
        list(zip(np.asarray(e).astype(bool), np.asarray(t, dtype=float))),
        dtype=[("e", bool), ("t", float)],
    )


def _at(grid, surv, times):
    """S at arbitrary times by linear interpolation on a fine grid."""
    return np.stack([np.interp(times, grid, s) for s in surv])


# --------------------------------------------------------------------- metrics
def ibll(et_train, et_test, grid, surv_grid) -> float:
    """Integrated binomial log-likelihood with IPCW (censoring from training)."""
    from sksurv.nonparametric import CensoringDistributionEstimator

    cens = CensoringDistributionEstimator().fit(et_train)
    t, e = et_test["t"], et_test["e"]
    G_T = np.clip(cens.predict_proba(t), _EPS, None)
    G_t = np.clip(cens.predict_proba(grid), _EPS, None)
    S = np.clip(surv_grid, _EPS, 1 - _EPS)  # (n, len(grid))
    died = (t[:, None] <= grid[None, :]) & e[:, None]
    alive = t[:, None] > grid[None, :]
    bll = -(died * np.log(1 - S) / G_T[:, None] + alive * np.log(S) / G_t[None, :]).mean(0)
    return float(np.trapz(bll, grid) / (grid[-1] - grid[0]))


def antolini(cuts: np.ndarray, surv_on_cuts: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    """Antolini's time-dependent C (Antolini et al. 2005) with every curve evaluated at
    the exact event times (log-linear between cuts, i.e. under the model's piecewise-
    constant hazard). pycox reads curves at the last grid point <= t, which turns every
    event before the first grid step into a tie at S = 1 and biases C towards 0.5.

    Comparable pairs: subject i with an event and T_i < T_j (or T_i = T_j with j
    censored); concordant if S_i(T_i) < S_j(T_i); ties count 1/2."""
    from sat.evaluate.multievent_metrics import _interp_cif

    cuts = np.asarray(cuts, dtype=float)
    cif = 1.0 - np.asarray(surv_on_cuts, dtype=float)
    t = np.asarray(t, dtype=float)
    e = np.asarray(e).astype(bool)
    n = len(t)
    num = den = 0.0
    for i in np.flatnonzero(e):
        later = (t > t[i]) | ((t == t[i]) & ~e)
        if not later.any():
            continue
        S_at = 1.0 - _interp_cif(cuts, cif, np.full(n, t[i]))  # every curve at T_i
        si, sj = S_at[i], S_at[later]
        num += np.sum(si < sj) + 0.5 * np.sum(si == sj)
        den += later.sum()
    return float(num / den) if den else float("nan")


def _mae_point(pred, t, e, t_tr, e_tr) -> Dict[str, float]:
    from SurvivalEVAL import PointEvaluator

    ev = PointEvaluator(pred, t, e, t_tr, e_tr)
    out = {}
    for m in MAE_METHODS:
        try:
            out[m] = float(ev.mae(method=m))
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"MAE {m} failed: {ex}")
    return out


class V2Metrics:
    def __init__(
        self,
        cfg,
        duration_cuts: str,
        training_set: str,
        fine_grid: int = 100,
    ):
        self.cfg = cfg
        self.K = int(cfg.num_events)
        self.cuts = pd.read_csv(
            duration_cuts, header=None, names=["c"], float_precision="round_trip"
        ).c.values.astype(float)
        df = pd.read_csv(training_set, header=0, float_precision="round_trip")
        self.train = {
            k: (df[f"duration_event{k + 1}"].values.astype(float),
                (df[f"event{k + 1}"] == 1).values)
            for k in range(self.K)
        }
        self.fine_grid = fine_grid
        self.final = False
        # the same horizons as the C_td columns (full-data quantiles per event)
        from pathlib import Path

        from sat.evaluate.survtrace_metrics import load_event_horizons

        hz = load_event_horizons(Path(duration_cuts).parent, training_set, self.K)
        self.horizons = {k: np.asarray(hz.get(k, self.cuts[1:-1][:3]), dtype=float)
                         for k in range(self.K)}

    # ----------------------------------------------------------- survival part
    def _survival_event(self, surv, t, e, k, grid=None) -> Dict[str, float]:
        """surv: (n, len(grid)) incl. t = 0 for event k. ``grid`` defaults to the duration
        cuts (the PCH model's own grid, refined log-linearly); baselines whose curves are
        not piecewise exponential pass their survival on a fine grid instead."""
        out = {}
        t_tr, e_tr = self.train[k]
        et_train, et_test = _structured(e_tr, t_tr), _structured(e, t)
        if grid is None:
            cuts = self.cuts
            grid, fine = fine_curves(self.cuts, surv, self.fine_grid)
        else:
            cuts = grid = np.asarray(grid, dtype=float)
            fine = np.clip(surv, 0.0, 1.0)

        lo = np.percentile(t, 1)
        hi = min(np.percentile(t, 99), t_tr.max() * (1 - 1e-9))
        # a fixed integration grid, independent of how the model's curve is sampled
        inner = np.linspace(lo, hi, 100) if hi > lo else np.array([])
        if len(inner) >= 3:
            from sksurv.metrics import integrated_brier_score

            S_in = _at(grid, fine, inner)
            try:
                out["ibs"] = float(integrated_brier_score(et_train, et_test, S_in, inner))
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"IBS failed: {ex}")
            try:
                out["ibll"] = ibll(et_train, et_test, inner, S_in)
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"IBLL failed: {ex}")

        try:
            out["antolini"] = antolini(cuts, surv, t, e)
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"Antolini failed: {ex}")

        try:
            from sksurv.metrics import cumulative_dynamic_auc

            h = np.asarray(self.horizons[k], dtype=float)
            h = h[(h > t.min()) & (h < min(t.max(), t_tr.max()))]
            if len(h):
                risk = 1.0 - _at(grid, fine, h)
                aucs = [cumulative_dynamic_auc(et_train, et_test, risk[:, i], h[i])[0][0]
                        for i in range(len(h))]
                out["auc"] = float(np.mean(aucs))
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"AUC failed: {ex}")

        try:
            from SurvivalEVAL import SurvivalEvaluator

            se = SurvivalEvaluator(fine, grid, t, e.astype(int), t_tr, e_tr.astype(int))
            out["dcal_p"] = float(se.d_calibration()[0])
            out["onecal_p"] = float(se.one_calibration(float(self.horizons[k][1]))[0])
        except Exception as ex:  # noqa: BLE001
            logger.warning(f"calibration failed: {ex}")

        for tail in TAILS:
            et = expected_time(cuts, surv, tail)
            for m, v in _mae_point(et, t, e.astype(int), t_tr, e_tr.astype(int)).items():
                out[f"mae_{tail}_{m}"] = v
        return out

    # --------------------------------------------------------- regression part
    def _regression_event(self, pred, t, e, k) -> Dict[str, float]:
        t_tr, e_tr = self.train[k]
        out = {}
        if self.final:
            for m, v in _mae_point(pred, t, e.astype(int), t_tr, e_tr.astype(int)).items():
                out[f"reg_mae_{m}"] = v
            try:
                from SurvivalEVAL import PointEvaluator

                out["reg_harrell"] = float(
                    PointEvaluator(pred, t, e.astype(int), t_tr, e_tr.astype(int)).concordance()[0]
                )
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"Harrell C failed: {ex}")
        else:
            from SurvivalEVAL import PointEvaluator

            try:
                out["reg_mae_margin"] = float(
                    PointEvaluator(pred, t, e.astype(int), t_tr, e_tr.astype(int)).mae(method="margin")
                )
            except Exception as ex:  # noqa: BLE001
                logger.warning(f"MAE margin failed: {ex}")
        return out

    # ------------------------------------------------------------------ compute
    def compute(self, predictions: Dict, references: np.ndarray) -> Dict[str, float]:
        K = self.K
        events = np.asarray(references[:, K : 2 * K], dtype=float) > 0
        durations = np.asarray(references[:, 3 * K : 4 * K], dtype=float)
        per_event: Dict[str, list] = {}
        out: Dict[str, float] = {}

        surv = predictions.get("survival") if isinstance(predictions, dict) else None
        grid = predictions.get("grid") if isinstance(predictions, dict) else None
        if self.final and surv is not None and np.ndim(surv) == 3:
            for k in range(K):
                for name, v in self._survival_event(np.asarray(surv[:, k]), durations[:, k],
                                                    events[:, k], k, grid).items():
                    out[f"{name}_{k}th_event"] = v
                    per_event.setdefault(name, []).append(v)

        tte = predictions.get("time_to_event") if isinstance(predictions, dict) else None
        if tte is not None:
            tte = np.asarray(tte, dtype=float).reshape(len(durations), -1)
            for k in range(K):
                for name, v in self._regression_event(tte[:, k], durations[:, k],
                                                      events[:, k], k).items():
                    out[f"{name}_{k}th_event"] = v
                    per_event.setdefault(name, []).append(v)

        for name, vals in per_event.items():
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                out[name] = float(np.mean(vals))
        return out
