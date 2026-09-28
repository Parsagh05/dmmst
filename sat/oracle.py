"""Oracle reference: the TRUE cumulative incidence functions of a simulated dataset,
scored with exactly the metrics and test split the trained models see.

    python -m sat.oracle experiments=multievent/survival dataset=sim_base \
        me_num_events=4 me_num_features=15 modelname=oracle [split_seed=...]

Needs ``data/<dataset>/truth.npz`` and ``meta.json`` from ``sat.data.simulate``.
Writes ``metrics.json`` in the same nested format as ``finetune``.
"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sat.data.simulate import true_cif
from sat.evaluate.multievent_metrics import WithinSubjectOrdering
from sat.evaluate.survtrace_metrics import SurvivalMAEMetrics, SurvTRACEMetrics
from sat.utils import config, logging, rand

logger = logging.get_default_logger()


@rand.seed
def _oracle(cfg: DictConfig):
    from sat.data import splitter

    src = Path(cfg.data_source)
    src = src if src.is_dir() else src.parent  # csv (multievent) or directory (ehrseq)
    truth = dict(np.load(src / "truth.npz"))
    with (src / "meta.json").open() as f:
        meta = json.load(f)
    regime = meta["scenario"]["regime"]

    ds = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col,
        k=cfg.cv.k,
        val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio,
        test_split_strategy="hash",
        split_names=cfg.data.splits,
        split_seed=cfg.get("split_seed"),
    ).load_split(cfg=cfg.data.load, fold_index=cfg.replication if cfg.cv.k else None)
    test = ds[cfg.data.splits[-1]]

    save_dir = cfg.data.label_transform.save_dir
    cuts = pd.read_csv(f"{save_dir}/duration_cuts.csv", header=None, names=["cuts"]).cuts.values
    K = int(cfg.data.num_events)

    rows = np.searchsorted(truth["ids"], np.asarray(test[cfg.data.id_col]))
    cif = true_cif(truth, regime, cuts, rows=rows)  # (n, K, len(cuts))
    predictions = np.stack([np.zeros_like(cif), cif, 1.0 - cif], axis=1)
    out_dir = Path(f"{cfg.modelhub}/{cfg.dataset}/{cfg.modelname}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(cif.shape[1]):  # curves on the cut grid, for scripts/true_order.py
        pd.DataFrame(1.0 - cif[:, k], columns=[f"t{j}" for j in range(cif.shape[2])]).assign(
            id=np.asarray(test[cfg.data.id_col])).to_csv(out_dir / f"survival{k}.csv", index=False)

    events = np.asarray(test[cfg.data.event_col], dtype=float).reshape(len(rows), K)
    durations = np.asarray(test[cfg.data.duration_col], dtype=float).reshape(len(rows), K)
    references = np.zeros((len(rows), 4 * K))
    references[:, K : 2 * K] = events
    references[:, 3 * K : 4 * K] = durations

    training_set = f"{save_dir}/transformed_train_labels.csv"
    cuts_file = f"{save_dir}/duration_cuts.csv"
    metrics = {}
    for module in (
        SurvTRACEMetrics(cfg.data, cuts_file, training_set, cfg.get("per_event_horizons", False)),
        SurvivalMAEMetrics(cfg.data, cuts_file, training_set),
        WithinSubjectOrdering(cfg.data, cuts_file, training_set),
    ):
        metrics.update(module.compute(predictions, references))

    out_dir = Path(f"{cfg.modelhub}/{cfg.dataset}/{cfg.modelname}")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"test": {k: {"mean": float(v), "variance": 0.0, "sd": 0.0} for k, v in metrics.items()}}
    payload["validation"] = payload["test"]
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(payload, f, indent=4)
    logger.info(f"Oracle metrics -> {out_dir / 'metrics.json'}")
    for k, v in metrics.items():
        logger.info(f"  {k} = {v:.4f}")
    return metrics


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def oracle(cfg: DictConfig) -> None:
    config.Config()
    _oracle(cfg)


if __name__ == "__main__":
    oracle()
