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

    def xy(split_name):
        split = dataset[split_name]
        x = np.asarray([np.asarray(v, dtype=float) for v in split[numeric_col]])
        d = np.asarray(split[duration_col], dtype=float).reshape(-1)
        e = np.asarray(split[event_col], dtype=float).reshape(-1)
        if d.ndim > 1:
            d = d[:, 0]
        if e.ndim > 1:
            e = e[:, 0]
        return x, d, e

    x_train, d_train, e_train = xy("train")
    x_test, d_test, e_test = xy("test")
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

    metrics = {}
    cis, brs = [], []
    t_lim = min(et_train["t"].max(), d_test.max())
    usable = [i for i, t in enumerate(times) if t < t_lim]

    for i in usable:
        tau = float(times[i])
        ci = concordance_index_ipcw(et_train, et_test, estimate=risk[:, i], tau=tau)[0]
        cis.append(ci)
        metrics[f"ctd_0th_event_{horizons[i]}"] = float(ci)

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
                    metrics[f"brier_0th_event_{horizons[i]}"] = float(bs[j])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Brier score unavailable: {e}")

    metrics["ctd_0th_event"] = float(np.mean(cis)) if cis else float("nan")
    metrics["ctd_weighted_avg"] = metrics["ctd_0th_event"]
    metrics["brier_0th_event"] = float(np.mean(brs)) if brs else float("nan")
    metrics["brier_survtrace_weighted_avg"] = metrics["brier_0th_event"]

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


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def coxph(cfg: DictConfig) -> None:
    config.Config()
    _coxph(cfg)


if __name__ == "__main__":
    coxph()
