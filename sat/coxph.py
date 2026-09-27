"""Cox proportional hazards reference baseline.

Rationale
---------
Every deep model in this repo should, at minimum, beat a linear Cox model. In the
first full run none of them did (all scored below the 0.628 CPH figure SurvTRACE
reports on METABRIC), which is the kind of thing that is only visible if the
reference is actually computed on the *same split, same features, same metric*.

This runs `sksurv.linear_model.CoxPHSurvivalAnalysis` on the same parsed dataset the
transformers use, and scores it with the identical SurvTRACE protocol
(`concordance_index_ipcw` truncated at the 25/50/75% uncensored-event-time
quantiles, censoring distribution fitted on train).

Usage
-----
    python -m sat.coxph experiments=survtrace_metabric/survival
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sat.utils import config, logging, rand

logger = logging.get_default_logger()


def _structured(events, durations):
    events = np.asarray(events).astype(bool)
    durations = np.asarray(durations).astype(float)
    return np.array(list(zip(events, durations)), dtype=[("e", bool), ("t", float)])


def _feature_matrix(split, feature_cols):
    """Numeric design matrix from a parsed SAT split."""
    df = pd.DataFrame({c: split[c] for c in feature_cols})
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)


class _Design:
    """Cox design matrix: numeric features as values, categorical features one-hot.

    Categorical positions (modality 0) carry a placeholder 1.0 in ``numerics``; their
    information is in the token (e.g. ``x4_x4_1.0``). Using ``numerics`` alone - as
    this baseline used to - silently gave Cox a constant column for every categorical
    feature (4 of 9 on METABRIC, 6 of 14 on SUPPORT). The vocabulary is fit on train
    and the first level of each feature is dropped (reference category).
    """

    def __init__(self, train, numeric_col):
        self.numeric_col = numeric_col
        modality = np.asarray(train["modality"][0]) if "modality" in train.column_names else None
        width = len(train[numeric_col][0])
        self.num_pos = [i for i in range(width) if modality is None or modality[i] == 1]
        self.cat_pos = [i for i in range(width) if modality is not None and modality[i] == 0]
        toks = [x.split() for x in train["x"]] if self.cat_pos else []
        self.levels = {
            i: sorted({t[i] for t in toks})[1:] for i in self.cat_pos
        }
        self.n_numeric = len(self.num_pos)
        self.n_onehot = sum(len(v) for v in self.levels.values())

    def __call__(self, split):
        num = np.asarray([np.asarray(v, dtype=float) for v in split[self.numeric_col]])
        cols = [num[:, self.num_pos]]
        if self.cat_pos:
            toks = [x.split() for x in split["x"]]
            for i in self.cat_pos:
                col = np.array([t[i] for t in toks])
                cols.append(np.stack([col == lv for lv in self.levels[i]], 1).astype(float)
                            if self.levels[i] else np.zeros((len(col), 0)))
        return np.concatenate(cols, axis=1)


@rand.seed
def _coxph(cfg: DictConfig):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import brier_score, concordance_index_ipcw

    from sat.data import splitter

    # use the exact same split the transformers see, so the comparison is honest
    ds_splitter = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col,
        k=cfg.cv.k,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        test_split_strategy="hash",
        split_names=cfg.data.splits,
        split_seed=cfg.get("split_seed"),
    )
    dataset = ds_splitter.load_split(
        cfg=cfg.data.load, fold_index=cfg.replication if cfg.cv.k else None
    )
    logger.info(f"Loaded splits: {list(dataset.keys())}")

    duration_col = cfg.data.duration_col
    event_col = cfg.data.event_col
    numeric_col = cfg.data.get("numerics_col", "numerics")

    cuts = pd.read_csv(
        f"{cfg.data.label_transform.save_dir}/duration_cuts.csv",
        header=None,
        names=["cuts"],
    ).cuts.values
    times = cuts[1:-1]  # 25/50/75% quantiles of uncensored event times
    horizons = [0.25, 0.5, 0.75][: len(times)]

    num_events = int(cfg.data.num_events)

    design = _Design(dataset[cfg.data.splits[0]], numeric_col)
    logger.info(f"Cox design: {design.n_numeric} numeric + {design.n_onehot} one-hot columns")

    def xy(split_name, event_idx=0):
        """Design matrix plus the duration/indicator for one event.

        With num_events > 1 each event has its own column, and we fit a separate
        cause-specific Cox model per event - the CS-CPH baseline SurvTRACE uses,
        which treats the other events as censored.
        """
        split = dataset[split_name]
        x = design(split)
        d = np.asarray(split[duration_col], dtype=float)
        e = np.asarray(split[event_col], dtype=float)
        d = d[:, event_idx] if d.ndim > 1 else d.reshape(-1)
        e = e[:, event_idx] if e.ndim > 1 else e.reshape(-1)
        return x, d, e

    metrics = {}
    all_ctd, all_brier = [], []
    curves = {}  # event -> test survival on the full cut grid
    per_event = {}
    if cfg.get("per_event_horizons", False):
        from sat.evaluate.survtrace_metrics import event_horizons

        per_event = event_horizons(
            f"{cfg.data.label_transform.save_dir}/transformed_train_labels.csv", num_events
        )
    for event_idx in range(num_events):
        ev_times, ev_horizons = (
            (per_event[event_idx], [0.25, 0.5, 0.75])
            if event_idx in per_event
            else (times, horizons)
        )
        m = _fit_one_event(
            xy, event_idx, np.asarray(ev_times, dtype=float), ev_horizons,
            CoxPHSurvivalAnalysis, concordance_index_ipcw, brier_score,
            grid=cuts, curves=curves,
        )
        metrics.update(m)
        if f"ctd_{event_idx}th_event" in m:
            all_ctd.append(m[f"ctd_{event_idx}th_event"])
        if f"brier_{event_idx}th_event" in m:
            all_brier.append(m[f"brier_{event_idx}th_event"])

    metrics["ctd_weighted_avg"] = (
        float(np.mean(all_ctd)) if all_ctd else float("nan")
    )
    metrics["brier_survtrace_weighted_avg"] = (
        float(np.mean(all_brier)) if all_brier else float("nan")
    )
    metrics.update(_curve_metrics(cfg, xy, curves, num_events))

    out_dir = Path(f"{cfg.modelhub}/{cfg.dataset}/{cfg.modelname}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"test": {k: {"mean": v, "variance": 0.0, "sd": 0.0}
                        for k, v in metrics.items()}}
    payload["validation"] = payload["test"]
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"Cox PH metrics -> {out_dir/'metrics.json'}")
    for k, v in metrics.items():
        logger.info(f"  {k} = {v:.4f}")
    return metrics


def _curve_metrics(cfg, xy, curves, num_events):
    """MAE and within-subject ordering from the Cox curves, with the very same modules
    the neural models are scored with, so every row of a table has every column."""
    from sat.evaluate.multievent_metrics import WithinSubjectOrdering
    from sat.evaluate.survtrace_metrics import SurvivalMAEMetrics

    if len(curves) != num_events:
        return {}
    surv = np.stack([curves[k] for k in range(num_events)], axis=1)  # (n, K, G)
    predictions = np.stack([np.zeros_like(surv), 1.0 - surv, surv], axis=1)
    references = np.zeros((surv.shape[0], 4 * num_events))
    for k in range(num_events):
        _, d, e = xy("test", k)
        references[:, num_events + k] = e
        references[:, 3 * num_events + k] = d
    save = cfg.data.label_transform.save_dir
    cuts_file, train = f"{save}/duration_cuts.csv", f"{save}/transformed_train_labels.csv"
    out = {}
    for module in (SurvivalMAEMetrics(cfg.data, cuts_file, train),
                   WithinSubjectOrdering(cfg.data, cuts_file, train)):
        try:
            out.update(module.compute(predictions, references))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{type(module).__name__} unavailable for Cox: {e}")
    return out


def _fit_one_event(xy, event_idx, times, horizons, CoxPHSurvivalAnalysis,
                   concordance_index_ipcw, brier_score, grid=None, curves=None):
    """Fit and score a single cause-specific Cox model. If ``curves`` is given, the
    test survival on ``grid`` is stored in it under ``event_idx``."""
    x_train, d_train, e_train = xy("train", event_idx)
    x_test, d_test, e_test = xy("test", event_idx)
    logger.info(f"train {x_train.shape}, test {x_test.shape}")

    et_train = _structured(e_train, d_train)
    et_test = _structured(e_test, d_test)

    model = CoxPHSurvivalAnalysis(alpha=1e-2)
    model.fit(x_train, et_train)
    logger.info("Cox PH fitted")

    # survival function evaluated on the same grid as the transformers
    surv_fns = model.predict_survival_function(x_test)
    surv = np.asarray([[fn(t) for t in times] for fn in surv_fns])
    risk = 1.0 - surv
    if curves is not None and grid is not None:
        lo, hi = surv_fns[0].domain
        curves[event_idx] = np.asarray(
            [[1.0 if t < lo else fn(min(t, hi)) for t in grid] for fn in surv_fns]
        )

    metrics = {}
    cis, brs = [], []
    t_lim = min(et_train["t"].max(), d_test.max())
    usable = [i for i, t in enumerate(times) if t < t_lim]

    for i in usable:
        tau = float(times[i])
        try:
            ci = concordance_index_ipcw(et_train, et_test, estimate=risk[:, i], tau=tau)[0]
        except Exception as e:  # noqa: BLE001  e.g. no event before tau in the test split
            logger.warning(f"C-index failed for event {event_idx} at tau={tau}: {e}")
            continue
        cis.append(ci)
        metrics[f"ctd_{event_idx}th_event_{horizons[i]}"] = float(ci)

    # sksurv's Brier requires every test time to lie inside the training
    # censoring distribution's support. SurvTRACE sidesteps this by forcing the
    # longest-duration subject into the training split; our hash-based splitter
    # makes no such guarantee, so restrict the Brier evaluation instead.
    if usable:
        idx = np.array(usable)
        keep = d_test < et_train["t"].max()
        if keep.sum() > 0:
            try:
                _, bs = brier_score(
                    et_train, et_test[keep], surv[keep][:, idx],
                    times[idx].astype(float),
                )
                for j, i in enumerate(usable):
                    brs.append(bs[j])
                    metrics[f"brier_{event_idx}th_event_{horizons[i]}"] = float(bs[j])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Brier score unavailable: {e}")

    metrics[f"ctd_{event_idx}th_event"] = float(np.mean(cis)) if cis else float("nan")
    metrics[f"brier_{event_idx}th_event"] = float(np.mean(brs)) if brs else float("nan")
    return metrics



@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def coxph(cfg: DictConfig) -> None:
    config.Config()
    _coxph(cfg)


if __name__ == "__main__":
    coxph()
