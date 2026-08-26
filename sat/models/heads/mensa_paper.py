"""MENSA, implemented to match the published method.

Reference
---------
MENSA: A Multi-Event Network for Survival Analysis with Trajectory-based
Likelihood Estimation. ML4H 2025. arXiv:2409.06525.
Reference implementation: https://github.com/thecml/mensa (`src/mensa/`).

Why this module exists
----------------------
`heads/mensa.py` is a MENSA-*inspired* reimplementation, not the published method:
it uses a learnable event-dependency matrix with sparsity regularisation, and the
word "trajectory" appears nowhere in it. Reporting its numbers under the name
"MENSA" would misattribute a result to a published paper. This module follows the
paper and the authors' code.

Architecture (paper Sec. 3.2, Eq. 3-6; reference `mensa/mlp.py`)
---------------------------------------------------------------
A shared representation, then per-state residual adapters and per-state heads:

    xrep = Phi(x)                                  shared MLP, ReLU6
    xrep_p = xrep + A_p(xrep)                      per-state residual adapter
    log eta_p = eta_tilde_p + SELU(W^shape_p xrep_p)     (Eq. 4)  shape
    log beta_p = beta_tilde_p + SELU(W^scale_p xrep_p)   (Eq. 3)  scale
    gate_p = (W^gate_p xrep_p) / temp                    mixture logits

`eta_tilde`/`beta_tilde` are learnable per-(state, component) base parameters,
initialised to -1 as in the reference.

Crucially the shape/scale heads emit values in **log space** which are exponentiated
downstream, so the distribution can reach any time scale. The other parametric heads
in this repo emit `softplus(...) + 0.01` (order 1) and then evaluate at raw durations
(0-355 on METABRIC), which underflows `exp(-(t/scale)^shape)` to zero -- see the note
in `heads/dsm.py`. MENSA has no such problem by construction.

Mixture, in log space (reference `compute_risks_multi`):

    log S_p(t) = logsumexp_psi[ -(exp(b)*t)^exp(k) + log g_psi ]
    log f_p(t) = logsumexp_psi[ k + b + (exp(k)-1)(b + log t) - (exp(b)*t)^exp(k)
                                + log g_psi ]
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


def safe_exp(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """exp with the clamping the reference implementation uses."""
    return torch.exp(torch.clamp(x, max=max_val))


def safe_log(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def weibull_log_f_s(shape, scale, gate_logits, t):
    """Log-density and log-survival of the Weibull mixture at times ``t``.

    Mirrors `MensaModel.compute_risks_multi` in the reference implementation.

    Args:
        shape: log-shape, [batch, n_dists]
        scale: log-scale, [batch, n_dists]
        gate_logits: mixture logits, [batch, n_dists]
        t: evaluation times, [batch] or [batch, n_times]

    Returns:
        (log_f, log_s), each matching ``t``'s shape.
    """
    single = t.dim() == 1
    if single:
        t = t.unsqueeze(1)  # [batch, 1]
    n_times = t.shape[1]
    n_dists = shape.shape[1]

    log_gate = F.log_softmax(gate_logits, dim=1)  # [batch, n_dists]

    # [batch, n_times, n_dists]
    t_e = torch.clamp(t, min=1e-12).unsqueeze(2).expand(-1, -1, n_dists)
    k = shape.unsqueeze(1).expand(-1, n_times, -1)
    b = scale.unsqueeze(1).expand(-1, n_times, -1)
    g = log_gate.unsqueeze(1).expand(-1, n_times, -1)

    ek = safe_exp(k)

    # The reference computes -(exp(b) * t) ** exp(k) directly. With raw durations
    # (t up to ~355 on METABRIC, ~2029 on SUPPORT) that overflows to inf and
    # torch.pow's backward then returns NaN ("PowBackward1 returned nan values").
    # Do it in log space instead - algebraically identical, but the exponent is
    # clamped before exponentiation rather than after it has already blown up:
    #     (exp(b) * t) ** exp(k) = exp( exp(k) * (b + log t) )
    log_base = b + safe_log(t_e)  # = log(exp(b) * t)
    z = torch.clamp(ek * log_base, max=30.0)  # exp(30) ~ 1e13, still finite
    s_comp = -torch.exp(z)
    f_comp = k + b + (ek - 1.0) * log_base + s_comp

    log_s = torch.logsumexp(s_comp + g, dim=2)
    log_f = torch.logsumexp(f_comp + g, dim=2)

    if single:
        return log_f.squeeze(1), log_s.squeeze(1)
    return log_f, log_s


class MENSAPaperConfig(SurvivalConfig):
    """Configuration for the paper-faithful MENSA head."""

    model_type = "sat-mensa-paper"

    def __init__(
        self,
        num_mixtures: int = 4,
        temp: float = 1.0,
        adapter_hidden: Optional[int] = None,
        mlp_layers: Optional[List[int]] = None,
        mlp_dropout: float = 0.0,
        scale_init: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mixtures = num_mixtures
        # NOTE: the reference divides the gate logits by temp. Its default is 1.0,
        # not the 1000.0 that DSM uses - a large temp here would flatten the gate.
        self.temp = temp
        self.adapter_hidden = adapter_hidden
        self.mlp_layers = mlp_layers
        self.mlp_dropout = mlp_dropout
        # None -> derive from the duration cuts; set -1.0 for the reference default
        self.scale_init = scale_init


class MENSAPaperTaskHead(SurvivalTask):
    """MENSA head: shared representation -> per-event Weibull mixtures.

    Emits an ``SAOutput`` whose ``survival``/``hazard``/``risk`` are evaluated on the
    duration cuts, so the existing metrics work unchanged, plus ``shape``/``scale``/
    ``logits_g`` which ``MENSAPaperLoss`` needs to evaluate the likelihood at each
    subject's own observed time.
    """

    config_class = MENSAPaperConfig

    def __init__(self, config: MENSAPaperConfig):
        # SurvivalTask is the HF PreTrainedModel base the other heads use; going
        # through it is what makes AutoModel.register / from_config work.
        super().__init__(config)
        self.num_events = config.num_events
        self.n_dists = config.num_mixtures
        self.temp = float(config.temp)

        in_dim = config.num_features
        layers = list(config.mlp_layers) if config.mlp_layers else [
            config.intermediate_size
        ]

        # shared representation Phi: Linear -> BatchNorm -> ReLU6 -> Dropout
        mods, prev = [], in_dim
        for h in layers:
            mods += [
                nn.Linear(prev, h, bias=True),
                nn.BatchNorm1d(h),
                nn.ReLU6(),
                nn.Dropout(p=config.mlp_dropout),
            ]
            prev = h
        self.embedding = nn.Sequential(*mods)
        last = prev

        # Learnable per-(event, component) base parameters. The reference
        # initialises both to -1, which implicitly assumes event times of order 1:
        # the model starts at exp(b) ~ 0.37, so exp(b)*t ~ 37 when t ~ 100 and the
        # survival curve is pinned at ~0 for every subject, giving no useful
        # gradient. METABRIC runs to 355 and SUPPORT to 2029, and MENSA trains on
        # raw durations (their loader does no time scaling), so the -1 init only
        # works when a dataset happens to live near t ~ 1.
        #
        # The functional form and the loss are exactly as published; only the
        # STARTING POINT of the scale parameter is set from the data, so that
        # exp(b) * t_typical ~ 1 at initialisation. Set `scale_init` explicitly to
        # reproduce the reference's -1.
        cuts_for_init = self._initial_cuts(config)
        if config.scale_init is not None:
            scale_init = float(config.scale_init)
        elif cuts_for_init is not None and float(cuts_for_init.max()) > 0:
            positive = cuts_for_init[cuts_for_init > 0]
            t_typical = float(positive.median()) if positive.numel() else 1.0
            scale_init = -float(torch.log(torch.tensor(max(t_typical, 1e-6))))
            logger.info(
                f"MENSA base log-scale initialised to {scale_init:.3f} "
                f"(t_typical={t_typical:.3f}); reference default is -1.0"
            )
        else:
            scale_init = -1.0

        self.shape = nn.Parameter(-torch.ones(self.num_events * self.n_dists))
        self.scale = nn.Parameter(
            torch.full((self.num_events * self.n_dists,), scale_init)
        )

        self.act = nn.SELU()
        self.shapeg = nn.ModuleList(
            [nn.Linear(last, self.n_dists, bias=True) for _ in range(self.num_events)]
        )
        self.scaleg = nn.ModuleList(
            [nn.Linear(last, self.n_dists, bias=True) for _ in range(self.num_events)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(last, self.n_dists, bias=False) for _ in range(self.num_events)]
        )
        adapter_hidden = config.adapter_hidden or max(16, last // 2)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(last, adapter_hidden, bias=True),
                    nn.ReLU6(),
                    nn.Linear(adapter_hidden, last, bias=True),
                )
                for _ in range(self.num_events)
            ]
        )

        self.loss = None
        if config.loss and "survival" in config.loss:
            self.loss = hydra.utils.instantiate(config.loss["survival"])

        if self.__class__.__name__ == "MENSAPaperTaskHead":
            self.post_init()

        cuts = None
        if self.loss is not None and getattr(self.loss, "duration_cuts", None) is not None:
            cuts = self.loss.duration_cuts.clone().detach()
        if cuts is None:
            cuts = torch.linspace(0.0, 1.0, steps=5)
        self.register_buffer("duration_cuts", cuts)

    @staticmethod
    def _initial_cuts(config):
        """Duration cuts, read early so the scale init can be data-aware."""
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
        """Per-event (log-shape, log-scale, gate logits). Paper Eq. 3-4."""
        x = sequence_output
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        xrep_shared = self.embedding(x)
        n = x.shape[0]

        base_shape = self.shape.view(self.num_events, self.n_dists)
        base_scale = self.scale.view(self.num_events, self.n_dists)

        shapes, scales, gates = [], [], []
        for p in range(self.num_events):
            xrep = xrep_shared + self.adapters[p](xrep_shared)
            shapes.append(self.act(self.shapeg[p](xrep)) + base_shape[p].expand(n, -1))
            scales.append(self.act(self.scaleg[p](xrep)) + base_scale[p].expand(n, -1))
            gates.append(self.gate[p](xrep) / self.temp)
        return (
            torch.stack(shapes, dim=1),  # [n, events, dists]
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
            _, log_s = weibull_log_f_s(shape[:, p], scale[:, p], gate[:, p], t)
            s = torch.exp(torch.clamp(log_s, max=0.0))
            # S(0) is exactly 1; the mixture is only defined for t > 0
            if float(cuts[0]) <= 0.0:
                s = torch.cat([torch.ones(n, 1, device=device), s[:, 1:]], dim=1)
            surv.append(s)
            # discrete hazard implied by the curve, for metric compatibility
            prev = torch.cat([torch.ones(n, 1, device=device), s[:, :-1]], dim=1)
            haz.append(torch.clamp(1.0 - s / torch.clamp(prev, min=1e-12), min=0.0))

        survival = torch.stack(surv, dim=1)  # [n, events, cuts]
        hazard = torch.stack(haz, dim=1)
        risk = 1.0 - survival

        output = SAOutput(
            loss=None,
            logits=survival,
            hazard=hazard,
            risk=risk,
            survival=survival,
            hidden_states=sequence_output,
            shape=shape,
            scale=scale,
            logits_g=gate,
        )
        if labels is not None and self.loss is not None:
            output.loss = self.loss(output, labels)
        return output
