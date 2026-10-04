"""Regression Task heads for survival analysis"""

__authors__ = ["Dominik Dahlem", "Mahed Abroshan"]
__status__ = "Development"

import hydra
import torch
from torch import nn

from sat.models.nets import CauseSpecificNet, CauseSpecificNetCompRisk
from sat.utils import logging

from .base import BaseConfig, RegressionTask
from .output import TaskOutput

logger = logging.get_default_logger()


class EventDurationTaskConfig(BaseConfig):
    """Configuration for the event-duration (regression) task head, paper Sec. 2.4.

    Same constructor pattern as SurvivalConfig: shared settings (initializer,
    loss, loss_weight, num_events, ...) go to BaseConfig, unknown keys (e.g.
    num_features, num_labels) to PretrainedConfig. It used to be a @dataclass, which
    cannot take those keyword arguments, so the head could not be built from config.
    """

    model_type = "regression"

    def __init__(
        self,
        intermediate_size: int = 32,
        num_hidden_layers: int = 0,
        indiv_intermediate_size: int = 32,
        indiv_num_hidden_layers: int = 1,
        batch_norm: bool = False,
        hidden_dropout_prob: float = 0.0,
        bias: bool = True,
        time_scale: float = 1.0,
        **kwargs,
    ):
        kwargs.setdefault("num_labels", 1)
        super().__init__(**kwargs)
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.indiv_intermediate_size = indiv_intermediate_size
        self.indiv_num_hidden_layers = indiv_num_hidden_layers
        self.batch_norm = batch_norm
        self.hidden_dropout_prob = hidden_dropout_prob
        self.bias = bias
        # predicted time = time_scale * softplus(z). With time_scale ~ the follow-up length
        # the network works on an O(1) scale instead of having to reach raw days
        self.time_scale = float(time_scale)


class EventDurationTaskHead(RegressionTask):
    config_class = EventDurationTaskConfig

    def __init__(self, config: EventDurationTaskConfig):
        super().__init__(config)

        if self.config.num_events > 1:
            self.nets = CauseSpecificNetCompRisk(
                in_features=self.config.num_features,
                shared_intermediate_size=self.config.intermediate_size,
                shared_num_hidden_layers=self.config.num_hidden_layers,
                indiv_intermediate_size=self.config.indiv_intermediate_size,
                indiv_num_hidden_layers=self.config.indiv_num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )
        else:
            self.nets = CauseSpecificNet(
                in_features=self.config.num_features,
                intermediate_size=self.config.intermediate_size,
                num_hidden_layers=self.config.num_hidden_layers,
                bias=self.config.bias,
                batch_norm=self.config.batch_norm,
                dropout=self.config.hidden_dropout_prob,
                out_features=self.config.num_labels,
                num_events=self.config.num_events,
            )

        loss = config.loss[config.model_type]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Instantiate the loss {loss}")
        self.loss = hydra.utils.instantiate(loss)

    def forward(self, sequence_output, labels=None, **kwargs):
        # softplus, not ReLU: a ReLU output dies (every pre-activation < 0 -> zero
        # gradient) and then predicts 0 for every subject, which is what happened in
        # the first smoke run. Softplus is positive and always has a gradient.
        logits = nn.functional.softplus(self.nets(sequence_output))
        # (batch, events): one predicted time per event, in the data's time unit
        predictions = torch.squeeze(logits, dim=2) * self.config.time_scale

        loss = None
        output = TaskOutput(loss=loss, logits=logits, predictions=predictions)
        if labels is not None:
            loss = self.loss(
                output,
                labels,
            )
            output.loss = loss

        return output
