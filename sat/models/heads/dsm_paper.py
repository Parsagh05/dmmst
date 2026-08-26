"""Deep Survival Machines, implemented to match the published method.

Reference
---------
Nagpal, Li, Dubrawski. "Deep Survival Machines: Fully Parametric Survival
Regression and Representation Learning for Censored Data with Competing Risks."
IEEE JBHI 2021. Reference implementation:
https://github.com/autonlab/auton-survival (`auton_survival/models/dsm/`).

Why this module exists
----------------------
`heads/dsm.py` is not the published parameterisation. It emits

    shape = softplus(net(x)) + 0.01
    scale = softplus(net(x)) + 0.01

i.e. positive parameters of order 1, and then evaluates the Weibull at raw
durations. DSM instead emits values in **log space** on top of a learnable base
parameter and exponentiates downstream (`dsm_torch.py::forward`):

    shape = SELU(shapeg(xrep)) + shape_base      # shape_base init -1
    scale = SELU(scaleg(xrep)) + scale_base
    gate  = gateg(xrep) / temp                   # bias=False

Because `exp(b)` can reach any magnitude, DSM has no time-scale problem. The
softplus version does: `exp(-(t/scale)^shape)` underflows to zero when t ~ 100 and
scale ~ 1, which is what forced the time-normalisation workaround in `heads/dsm.py`
and what made that model's Brier score 0.68. This module removes the need for the
workaround by being faithful.

Mixture and likelihood (`losses.py::_conditional_weibull_loss`)
---------------------------------------------------------------
Per component, with k = log shape and b = log inverse-scale:

    log S_g(t) = -(exp(b) t)^exp(k)
    log f_g(t) = k + b + (exp(k) - 1)(b + log t) + log S_g(t)

aggregated either as an exact mixture (``elbo=False``)::

    log S = logsumexp_g[ log_softmax(gate)_g + log S_g ]

or as DSM's ELBO surrogate (``elbo=True``, the reference default for the main
training phase), which averages the log-terms with the softmax weights instead::

    log S = sum_g softmax(gate)_g * log S_g
"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

from typing import List, Optional

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sat.utils import logging

from .base import SurvivalTask
from .output import SAOutput
from .survival import SurvivalConfig

logger = logging.get_default_logger()


def dsm_log_f_s(shape, scale, gate_logits, t, elbo: bool = False):
    """Log-density and log-survival of the DSM Weibull mixture.

    Mirrors `_conditional_weibull_loss` in the reference, including its two
    aggregation modes.

    Args:
        shape: log-shape ``k``, [batch, k_dists]
        scale: log-inverse-scale ``b``, [batch, k_dists]
        gate_logits: mixture logits (already divided by ``temp``), [batch, k_dists]
        t: times, [batch] or [batch, n_times]
        elbo: use DSM's ELBO surrogate instead of the exact log-mixture.

    Returns:
        (log_f, log_s), each matching ``t``'s shape.
    """
    single = t.dim() == 1
    if single:
        t = t.unsqueeze(1)
    n_times = t.shape[1]
    n_dists = shape.shape[1]

    t_e = torch.clamp(t, min=1e-12).unsqueeze(2).expand(-1, -1, n_dists)
    k = shape.unsqueeze(1).expand(-1, n_times, -1)
    b = scale.unsqueeze(1).expand(-1, n_times, -1)

    ek = torch.exp(torch.clamp(k, max=20.0))

    # The reference computes -(exp(b) * t) ** exp(k) directly. On raw durations
    # (355 on METABRIC, 2029 on SUPPORT) that overflows to inf and torch.pow's
    # backward returns NaN. Algebraically identical in log space, with the
    # exponent clamped *before* exponentiating:
    #     (exp(b) t)^exp(k) = exp( exp(k) * (b + log t) )
    log_base = b + torch.log(t_e)
    s_comp = -torch.exp(torch.clamp(ek * log_base, max=30.0))
    f_comp = k + b + (ek - 1.0) * log_base + s_comp

    if elbo:
        g = F.softmax(gate_logits, dim=1).unsqueeze(1).expand(-1, n_times, -1)
        log_s = (g * s_comp).sum(dim=2)
        log_f = (g * f_comp).sum(dim=2)
    else:
        g = F.log_softmax(gate_logits, dim=1).unsqueeze(1).expand(-1, n_times, -1)
        log_s = torch.logsumexp(s_comp + g, dim=2)
        log_f = torch.logsumexp(f_comp + g, dim=2)

    if single:
        return log_f.squeeze(1), log_s.squeeze(1)
    return log_f, log_s


class DSMPaperConfig(SurvivalConfig):
    """Configuration for the paper-faithful DSM head."""

    model_type = "sat-dsm-paper"

    def __init__(
        self,
        num_mixtures: int = 4,
        temp: float = 1000.0,
        discount: float = 1.0,
        elbo: bool = True,
        mlp_layers: Optional[List[int]] = None,
        mlp_dropout: float = 0.0,
        scale_init: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        # DSM's own default is 1000. The gate logits are divided by it, so the
        # mixture is deliberately smoothed towards uniform; discrimination comes
        # from the per-subject shape/scale, not the gate.
        self.temp = temp
        self.discount = discount
        self.elbo = elbo
        self.mlp_layers = mlp_layers
        self.mlp_dropout = mlp_dropout
        # None -> derive from the duration cuts; -1.0 reproduces the reference.
        self.scale_init = scale_init


class DSMPaperTaskHead(SurvivalTask):
    """DSM head: shared MLP -> per-risk Weibull mixture parameters."""

    config_class = DSMPaperConfig

    def __init__(self, config: DSMPaperConfig):
        super().__init__(config)
        self.num_events = config.num_events
        self.k_dists = config.num_mixtures
        self.temp = float(config.temp)
        self.elbo = bool(config.elbo)

        in_dim = config.num_features
        layers = list(config.mlp_layers) if config.mlp_layers else [
            config.intermediate_size
        ]

        mods, prev = [], in_dim
        for h in layers:
            mods += [nn.Linear(prev, h, bias=True), nn.ReLU6(),
                     nn.Dropout(p=config.mlp_dropout)]
            prev = h
        self.embedding = nn.Sequential(*mods)
        last = prev

        cuts = self._initial_cuts(config)
        if config.scale_init is not None:
            scale_init = float(config.scale_init)
        elif cuts is not None and float(cuts.max()) > 0:
            positive = cuts[cuts > 0]
            t_typical = float(positive.median()) if positive.numel() else 1.0
            scale_init = -float(torch.log(torch.tensor(max(t_typical, 1e-6))))
            logger.info(
                f"DSM base log-scale initialised to {scale_init:.3f} "
                f"(t_typical={t_typical:.3f}); reference default is -1.0"
            )
        else:
            scale_init = -1.0

        # learnable per-(risk, component) base parameters; reference inits both -1
        self.shape = nn.Parameter(-torch.ones(self.num_events, self.k_dists))
        self.scale = nn.Parameter(
            torch.full((self.num_events, self.k_dists), scale_init)
        )

        self.act = nn.SELU()
        self.shapeg = nn.ModuleList(
            [nn.Linear(last, self.k_dists, bias=True) for _ in range(self.num_events)]
        )
        self.scaleg = nn.ModuleList(
            [nn.Linear(last, self.k_dists, bias=True) for _ in range(self.num_events)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(last, self.k_dists, bias=False) for _ in range(self.num_events)]
        )

        self.loss = None
        if config.loss and "survival" in config.loss:
            self.loss = hydra.utils.instantiate(config.loss["survival"])

        cuts_buf = cuts if cuts is not None else torch.linspace(0.0, 1.0, steps=5)
        self.register_buffer("duration_cuts", cuts_buf)

        if self.__class__.__name__ == "DSMPaperTaskHead":
            self.post_init()

    @staticmethod
    def _initial_cuts(config):
        try:
            loss_cfg = config.loss.get("survival") if config.loss else None
            path = loss_cfg.get("duration_cuts") if loss_cfg else None
            if path:
                df = pd.read_csv(path, header=None, names=["cuts"])
                return torch.tensor(df.cuts.values, dtype=torch.float32)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"could not read duration cuts for scale init: {e}")
        return None

    def params(self, sequence_output):
        """Per-risk (log-shape, log-scale, gate logits). Reference `forward`."""
        x = sequence_output
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        xrep = self.embedding(x)
        n = x.shape[0]

        shapes, scales, gates = [], [], []
        for p in range(self.num_events):
            shapes.append(self.act(self.shapeg[p](xrep)) + self.shape[p].expand(n, -1))
            scales.append(self.act(self.scaleg[p](xrep)) + self.scale[p].expand(n, -1))
            gates.append(self.gate[p](xrep) / self.temp)
        return (
            torch.stack(shapes, dim=1),
            torch.stack(scales, dim=1),
            torch.stack(gates, dim=1),
        )

    def forward(self, sequence_output, labels=None, **kwargs):
        shape, scale, gate = self.params(sequence_output)
        n = shape.shape[0]
        device = shape.device
        cuts = self.duration_cuts.to(device)

        surv, haz = [], []
        for p in range(self.num_events):
            t = cuts.unsqueeze(0).expand(n, -1)
            # the survival curve is reported with the exact mixture; the ELBO
            # surrogate is a training device, not a predictive distribution
            _, log_s = dsm_log_f_s(shape[:, p], scale[:, p], gate[:, p], t, elbo=False)
            s = torch.exp(torch.clamp(log_s, max=0.0))
            if float(cuts[0]) <= 0.0:
                s = torch.cat([torch.ones(n, 1, device=device), s[:, 1:]], dim=1)
            surv.append(s)
            prev = torch.cat([torch.ones(n, 1, device=device), s[:, :-1]], dim=1)
            haz.append(torch.clamp(1.0 - s / torch.clamp(prev, min=1e-12), min=0.0))

        survival = torch.stack(surv, dim=1)
        hazard = torch.stack(haz, dim=1)

        output = SAOutput(
            loss=None,
            logits=survival,
            hazard=hazard,
            risk=1.0 - survival,
            survival=survival,
            hidden_states=sequence_output,
            shape=shape,
            scale=scale,
            logits_g=gate,
        )
        if labels is not None and self.loss is not None:
            output.loss = self.loss(output, labels)
        return output
