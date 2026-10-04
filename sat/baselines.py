"""Classical and pycox baselines on exactly the splits and features the transformer sees.

    python -m sat.baselines experiments=survtrace_metabric/survival baseline=rsf split_seed=0

``baseline`` is one of

* ``cox``      - Cox PH (scikit-survival), ridge penalty ``bl_cox_alpha``;
* ``rsf``      - Random Survival Forest (scikit-survival, Ishwaran et al. 2008);
* ``deepsurv`` - DeepSurv (Katzman et al. 2018) via pycox ``CoxPH`` with an MLP;
* ``pchazard`` - PC-Hazard (Kvamme & Borgan 2019) via pycox, on the same duration cuts
  as our model (the closest classical relative of L_PCH).

With several events every model is fitted cause-specifically - one model per event,
the other events treated as censored - which is how SurvTRACE runs CS-CPH and
CS-PC-Hazard. Neural baselines early-stop on the validation split. Every model only
provides its survival function; scoring is ``sat.evaluate.shared.evaluate_curves``,
the same code that scores the transformer. Output: ``metrics.json`` (validation and
test) and the test curves on the cut grid (``survival<k>.csv``).
"""

__authors__ = ["Parsa"]
__status__ = "Development"

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from sat.utils import config, logging, rand

logger = logging.get_default_logger()

BASELINES = ("cox", "rsf", "deepsurv", "pchazard")


def _structured(e, t):
    return np.array(list(zip(np.asarray(e).astype(bool), np.asarray(t, dtype=float))),
                    dtype=[("e", bool), ("t", float)])


def _step_at(times_index: np.ndarray, values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Right-continuous step function S(t) = values[last index <= t] (1 before the first)."""
    idx = np.searchsorted(times_index, times, side="right") - 1
    out = np.ones((values.shape[0], len(times)))
    ok = idx >= 0
    out[:, ok] = values[:, idx[ok]]
    return out


class _SkSurvModel:
    """Cox or RSF; survival functions are step functions."""

    def __init__(self, kind, cfg, seed):
        if kind == "cox":
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            self.model = CoxPHSurvivalAnalysis(alpha=float(cfg.bl_cox_alpha))
        else:
            from sksurv.ensemble import RandomSurvivalForest

            self.model = RandomSurvivalForest(
                n_estimators=int(cfg.bl_rsf_n_estimators),
                min_samples_leaf=int(cfg.bl_rsf_min_samples_leaf),
                max_features=cfg.bl_rsf_max_features,
                n_jobs=-1,
                random_state=seed,
            )

    def fit(self, x, t, e, x_val=None, t_val=None, e_val=None):
        self.model.fit(x, _structured(e, t))
        self.event_times = np.asarray(self.model.unique_times_, dtype=float)
        return self

    def survival(self, x, times):
        S = self.model.predict_survival_function(x, return_array=True)  # (n, len(event_times))
        return _step_at(self.event_times, S, np.asarray(times, dtype=float))


class _PycoxModel:
    """DeepSurv (CoxPH) or PC-Hazard with an MLP, Adam, early stopping on validation."""

    def __init__(self, kind, cfg, seed, cuts):
        import torch
        import torchtuples as tt
        from pycox.models import CoxPH, PCHazard

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.kind, self.cfg, self.cuts = kind, cfg, np.asarray(cuts, dtype=float)
        self.tt = tt
        self._CoxPH, self._PCHazard = CoxPH, PCHazard

    def _net(self, in_features, out_features):
        return self.tt.practical.MLPVanilla(
            in_features,
            [int(self.cfg.bl_nn_nodes)] * int(self.cfg.bl_nn_layers),
            out_features,
            batch_norm=bool(self.cfg.bl_nn_batch_norm),
            dropout=float(self.cfg.bl_nn_dropout),
        )

    def fit(self, x, t, e, x_val, t_val, e_val):
        tt, cfg = self.tt, self.cfg
        x, x_val = x.astype("float32"), x_val.astype("float32")
        if self.kind == "deepsurv":
            self.model = self._CoxPH(self._net(x.shape[1], 1), tt.optim.Adam)
            y = (t.astype("float32"), e.astype("float32"))
            y_val = (t_val.astype("float32"), e_val.astype("float32"))
        else:
            from pycox.preprocessing.discretization import DiscretizeUnknownC, Duration2Idx

            self.labtrans = self._PCHazard.label_transform(self.cuts.astype("float32"))
            # pycox 0.3 bug: with predefined cuts fit() returns before building these
            self.labtrans.duc = DiscretizeUnknownC(self.labtrans.cuts, right_censor=True,
                                                   censor_side="right")
            self.labtrans.di = Duration2Idx(self.labtrans.cuts)
            y = self.labtrans.transform(t.astype("float32"), e.astype("float32"))
            y_val = self.labtrans.transform(t_val.astype("float32"), e_val.astype("float32"))
            self.model = self._PCHazard(self._net(x.shape[1], self.labtrans.out_features),
                                        tt.optim.Adam, duration_index=self.labtrans.cuts)
        self.model.optimizer.set_lr(float(cfg.bl_nn_lr))
        self.model.fit(
            x, y, batch_size=int(cfg.bl_nn_batch), epochs=int(cfg.bl_nn_epochs),
            callbacks=[tt.callbacks.EarlyStopping(patience=int(cfg.bl_nn_patience))],
            val_data=(x_val, y_val), verbose=False,
        )
        if self.kind == "deepsurv":
            self.model.compute_baseline_hazards()
        return self

    def survival(self, x, times):
        times = np.asarray(times, dtype=float)
        df = self.model.predict_surv_df(x.astype("float32"))  # index = time, columns = subjects
        S = df.values.T
        if self.kind == "deepsurv":
            return _step_at(df.index.values.astype(float), S, times)
        # PC-Hazard: piecewise-constant hazard -> log-linear between the cuts
        from sat.evaluate.multievent_metrics import _interp_cif

        grid = df.index.values.astype(float)
        n = len(S)
        return np.stack([1.0 - _interp_cif(grid, 1.0 - S, np.full(n, tt_)) for tt_ in times], axis=1)


@rand.seed
def _baselines(cfg: DictConfig):
    from sat.coxph import _Design
    from sat.data import splitter
    from sat.evaluate.shared import evaluate_curves, references, write_metrics

    kind = str(cfg.baseline)
    if kind not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {kind!r}")
    sp = splitter.StreamingKFoldSplitter(
        id_field=cfg.data.id_col, k=None, val_ratio=cfg.data.validation_ratio,
        test_ratio=cfg.data.test_ratio, test_split_strategy="hash",
        split_names=cfg.data.splits, split_seed=cfg.get("split_seed"),
    )
    ds = sp.load_split(cfg=cfg.data.load)
    tr_name, va_name, te_name = cfg.data.splits
    design = _Design(ds[tr_name], cfg.data.get("numerics_col", "numerics"))
    K = int(cfg.data.num_events)
    cuts = pd.read_csv(f"{cfg.data.label_transform.save_dir}/duration_cuts.csv",
                       header=None, names=["c"], float_precision="round_trip").c.values

    def xy(name):
        split = ds[name]
        e = np.asarray(split[cfg.data.event_col], dtype=float).reshape(len(split), -1)
        d = np.asarray(split[cfg.data.duration_col], dtype=float).reshape(len(split), -1)
        return design(split), d, e

    (x_tr, d_tr, e_tr), (x_va, d_va, e_va), (x_te, d_te, e_te) = (
        xy(tr_name), xy(va_name), xy(te_name))
    logger.info(f"{kind}: train {x_tr.shape}, validation {x_va.shape}, test {x_te.shape}")

    models = []
    for k in range(K):
        if kind in ("cox", "rsf"):
            m = _SkSurvModel(kind, cfg, int(cfg.seed))
        else:
            m = _PycoxModel(kind, cfg, int(cfg.seed) + k, cuts)
        models.append(m.fit(x_tr, d_tr[:, k], e_tr[:, k], x_va, d_va[:, k], e_va[:, k]))
        logger.info(f"{kind}: event {k + 1}/{K} fitted")

    def surv_at(x):
        return lambda times: np.stack([m.survival(x, times) for m in models], axis=1)

    val = evaluate_curves(cfg, surv_at(x_va), references(e_va, d_va))
    test = evaluate_curves(cfg, surv_at(x_te), references(e_te, d_te))
    out_dir = Path(f"{cfg.modelhub}/{cfg.dataset}/{cfg.modelname}")
    write_metrics(out_dir, test, val)
    ids = ds[te_name][cfg.data.id_col]
    S = surv_at(x_te)(cuts)
    for k in range(K):  # test curves on the cut grid, as finetune writes them
        pd.DataFrame(S[:, k], columns=[f"t{j}" for j in range(len(cuts))]).assign(
            id=ids).to_csv(out_dir / f"survival{k}.csv", index=False)
    logger.info(f"{kind}: test C_td {test.get('ctd_weighted_avg'):.4f} -> {out_dir}")
    return test


@hydra.main(version_base=None, config_path="../conf", config_name="finetune.yaml")
def baselines(cfg: DictConfig) -> None:
    config.Config()
    _baselines(cfg)


if __name__ == "__main__":
    baselines()
