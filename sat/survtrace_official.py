"""Official SurvTRACE (Wang & Sun, BCB '22) on our splits, scored like every other model.

    python scripts/fetch_survtrace.py            # once: the authors' code, pinned commit
    python -m sat.survtrace_official experiments=survtrace_metabric/survival split_seed=0

The released implementation (``third_party/SurvTRACE``) is used unchanged: its
``SurvTraceSingle`` / ``SurvTraceMulti`` models, ``LabelTransform``, ``Trainer`` (BERT-Adam,
early stopping on the validation loss). What this wrapper adds:

* the data: our split of this ``split_seed``; categorical features first, label-encoded
  over the full data with offsets into one vocabulary, then the standardised numeric
  features - exactly the layout of SurvTRACE's ``load_data``;
* the time grid: our duration cuts (the same horizons every model is scored at);
* multi-event data: SurvTRACE's multi-event model assumes one shared time per subject
  (competing risks). It is used as is on competing-risk data (DeepHit synthetic); where
  each event has its own time (EBMT, hsa_synthetic) one single-event SurvTRACE is
  trained per event (cause-specific), like CS-PC-Hazard in their paper;
* scoring through ``sat.evaluate.shared.evaluate_curves``.

The released code trains with the PC-Hazard likelihood only (the paper's multi-task
heads and IPS are not in the repository), so this is "SurvTRACE (released code)".
"""

__authors__ = ["Parsa"]
__status__ = "Development"

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sat.utils import config, logging, rand

logger = logging.get_default_logger()
REPO = Path(__file__).resolve().parents[1]


def _import_survtrace():
    root = REPO / "third_party" / "SurvTRACE"
    if not (root / "survtrace" / "model.py").is_file():
        sys.path.insert(0, str(REPO))
        from scripts.fetch_survtrace import fetch

        fetch()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # the released code uses np.Inf, removed in NumPy 2 (Kaggle's image); restore the alias
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    import survtrace  # noqa: F401
    from survtrace.config import STConfig
    from survtrace.model import SurvTraceMulti, SurvTraceSingle
    from survtrace.train_utils import Trainer
    from survtrace.utils import LabelTransform

    return STConfig, SurvTraceSingle, SurvTraceMulti, Trainer, LabelTransform


def _cpu_shim():
    """SurvTRACE's multi-event trainer calls .cuda() unconditionally; on a CPU-only
    machine (local smoke tests) make that a no-op. Kaggle runs on a GPU."""
    import torch

    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]


def _features(ds, splits):
    """SurvTRACE layout: [categorical ids..., numeric values...] for every split."""
    first = ds[splits[0]]
    modality = list(first["modality"][0]) if "modality" in first.column_names else None
    width = len(first["numerics"][0])
    cat_pos = [i for i in range(width) if modality is not None and modality[i] == 0]
    num_pos = [i for i in range(width) if modality is None or modality[i] == 1]

    frames = {}
    for name in splits:
        sp = ds[name]
        nums = np.asarray([np.asarray(v, dtype=float) for v in sp["numerics"]])
        toks = [x.split() for x in sp["x"]] if cat_pos else []
        df = pd.DataFrame({f"c{i}": [t[i] for t in toks] for i in cat_pos})
        for i in num_pos:
            df[f"n{i}"] = nums[:, i]
        frames[name] = df
    # label-encode categories over all rows with offsets (SurvTRACE's load_data)
    vocab = 0
    allc = pd.concat([frames[n] for n in splits], keys=splits)
    for i in cat_pos:
        codes = {v: j + vocab for j, v in enumerate(sorted(allc[f"c{i}"].unique()))}
        for n in splits:
            frames[n][f"c{i}"] = frames[n][f"c{i}"].map(codes).astype(float)
        vocab += len(codes)
    return frames, len(cat_pos), len(num_pos), max(vocab, 1)


@rand.seed
def _survtrace(cfg: DictConfig):
    import torch

    from sat.data import splitter
    from sat.evaluate.multievent_metrics import _interp_cif
    from sat.evaluate.shared import evaluate_curves, references, write_metrics

    STConfig, SurvTraceSingle, SurvTraceMulti, Trainer, LabelTransform = _import_survtrace()
    _cpu_shim()
    torch.manual_seed(int(cfg.seed))

    sp = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col, k=None, val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio, test_split_strategy="hash",
        split_names=cfg.data.splits, split_seed=cfg.get("split_seed"),
    )
    ds = sp.load_split(cfg=cfg.data.load)
    names = list(cfg.data.splits)
    tr, va, te = names
    frames, n_cat, n_num, vocab = _features(ds, names)
    K = int(cfg.data.num_events)
    cuts = pd.read_csv(f"{cfg.data.label_transform.save_dir}/duration_cuts.csv", header=None,
                       names=["c"], float_precision="round_trip").c.values.astype(float)

    def labels(name):
        sp_ = ds[name]
        e = np.asarray(sp_[cfg.data.event_col], dtype=float).reshape(len(sp_), -1)
        d = np.asarray(sp_[cfg.data.duration_col], dtype=float).reshape(len(sp_), -1)
        return e, d

    lab = {n: labels(n) for n in names}
    shared_time = K > 1 and all(np.allclose(lab[n][1], lab[n][1][:, :1]) for n in names)

    def st_config(num_event):
        c = STConfig.copy() if hasattr(STConfig, "copy") else dict(STConfig)
        c = type(STConfig)(c)
        c.update({
            "data": cfg.dataset, "num_event": num_event, "seed": int(cfg.seed),
            "checkpoint": str(Path(cfg.modelhub) / cfg.dataset / cfg.modelname / "survtrace.pt"),
            "vocab_size": int(vocab), "num_categorical_feature": n_cat,
            "num_numerical_feature": n_num, "num_feature": n_cat + n_num,
            "hidden_size": int(cfg.st_hidden_size), "intermediate_size": int(cfg.st_intermediate_size),
            "num_hidden_layers": int(cfg.st_num_hidden_layers),
            "num_attention_heads": int(cfg.st_num_attention_heads),
            "hidden_dropout_prob": float(cfg.st_hidden_dropout_prob),
            "early_stop_patience": int(cfg.st_early_stop_patience),
            "duration_index": cuts, "out_feature": len(cuts) - 1,
            "horizons": [0.25, 0.5, 0.75],
        })
        return c

    def y_frame(name, k, lt):
        e, d = lab[name]
        idx, ev, frac = lt.transform(d[:, k], e[:, k])
        return pd.DataFrame({"duration": idx, "event": ev, "proportion": frac},
                            index=frames[name].index)

    train_kw = dict(batch_size=int(cfg.st_batch_size), epochs=int(cfg.st_epochs),
                    learning_rate=float(cfg.st_learning_rate),
                    weight_decay=float(cfg.st_weight_decay), val_batch_size=10000)
    surv_on_cuts = {n: [None] * K for n in (va, te)}

    if shared_time:  # competing risks: the released multi-event model
        lt = LabelTransform(cuts=cuts)
        lt.fit(lab[tr][1][:, 0], lab[tr][0][:, 0])
        model = SurvTraceMulti(st_config(K))
        ys = {}
        for n in names:
            e, d = lab[n]
            idx, _, frac = lt.transform(d[:, 0], np.ones(len(d)))
            y = pd.DataFrame({"duration": idx, "proportion": frac}, index=frames[n].index)
            for k in range(K):
                y[f"event_{k}"] = e[:, k]
            ys[n] = y[["duration"] + [f"event_{k}" for k in range(K)] + ["proportion"]]
        Trainer(model).fit((frames[tr], ys[tr]), (frames[va], ys[va]), **train_kw)
        for n in (va, te):
            for k in range(K):
                surv_on_cuts[n][k] = model.predict_surv(frames[n], batch_size=10000, event=k).cpu().numpy()
    else:  # single event, or cause-specific when events have their own times
        for k in range(K):
            lt = LabelTransform(cuts=cuts)
            lt.fit(lab[tr][1][:, k], lab[tr][0][:, k])
            model = SurvTraceSingle(st_config(1))
            Trainer(model).fit((frames[tr], y_frame(tr, k, lt)), (frames[va], y_frame(va, k, lt)),
                               **train_kw)
            for n in (va, te):
                surv_on_cuts[n][k] = model.predict_surv(frames[n], batch_size=10000).cpu().numpy()

    def surv_at(name):
        S = np.stack(surv_on_cuts[name], axis=1)  # (n, K, len(cuts)), S at the cuts
        n = len(S)

        def f(times):
            return np.stack([
                np.stack([1.0 - _interp_cif(cuts, 1.0 - S[:, k], np.full(n, t)) for t in times], 1)
                for k in range(K)], axis=1)
        return f

    val = evaluate_curves(cfg, surv_at(va), references(*lab[va]))
    test = evaluate_curves(cfg, surv_at(te), references(*lab[te]))
    out_dir = Path(cfg.modelhub) / cfg.dataset / cfg.modelname
    write_metrics(out_dir, test, val)
    ids = ds[te][cfg.data.id_col]
    for k in range(K):
        pd.DataFrame(surv_on_cuts[te][k], columns=[f"t{j}" for j in range(len(cuts))]).assign(
            id=ids).to_csv(out_dir / f"survival{k}.csv", index=False)
    logger.info(f"SurvTRACE: test C_td {test.get('ctd_weighted_avg'):.4f} -> {out_dir}")
    return test


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def survtrace_official(cfg: DictConfig) -> None:
    config.Config()
    _survtrace(cfg)


if __name__ == "__main__":
    survtrace_official()
