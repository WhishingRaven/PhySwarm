"""Reusable distributions for task-named MAPPO action channels."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from learning.rmappo.utils import to_torch


def _linear(input_dim: int, output_dim: int, gain: float) -> nn.Linear:
    layer = nn.Linear(input_dim, output_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


class NamedDiagGaussian(nn.Module):
    """Independent Gaussian heads whose names define the action contract.

    Dynamic ``setattr`` retains the historical state-dict keys such as
    ``branch_food.0.weight`` and ``food_mu.bias`` without copying an entire
    distribution implementation for every task.
    """

    channel_names: tuple[str, ...] = ()
    sanitize_inputs = False
    bounded_log_std = False

    def __init__(self, args, act_dim, rnn_input_dim, obs_dim, device):
        del args
        super().__init__()
        if act_dim != len(self.channel_names):
            raise ValueError(
                f"{type(self).__name__} expects {len(self.channel_names)} actions, got {act_dim}"
            )
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.logstd_min = math.log(0.08)
        self.logstd_max = math.log(0.35)
        self.logstd_bias = math.log(0.25)
        input_dim = rnn_input_dim + obs_dim

        for name in self.channel_names:
            setattr(
                self,
                f"branch_{name}",
                nn.Sequential(_linear(input_dim, 64, math.sqrt(2.0)), nn.Tanh()),
            )
            setattr(self, f"{name}_mu", _linear(64, 1, 0.01))
            logstd = _linear(64, 1, 0.01)
            if self.bounded_log_std:
                nn.init.constant_(logstd.bias, self.logstd_bias)
            setattr(self, f"{name}_logstd", logstd)
        self.to(device)

    def forward(self, rnn_states, obs):
        rnn_states = rnn_states.to(**self.tpdv)
        obs = obs.to(**self.tpdv)
        if self.sanitize_inputs:
            rnn_states = torch.nan_to_num(rnn_states, nan=0.0, posinf=10.0, neginf=-10.0)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        inputs = torch.cat((rnn_states, obs), dim=-1)
        if self.sanitize_inputs:
            inputs = torch.clamp(inputs, -10.0, 10.0)

        means = []
        log_stds = []
        for name in self.channel_names:
            hidden = getattr(self, f"branch_{name}")(inputs)
            means.append(getattr(self, f"{name}_mu")(hidden))
            log_stds.append(getattr(self, f"{name}_logstd")(hidden))
        mean = torch.cat(means, dim=-1)
        log_std = torch.cat(log_stds, dim=-1)

        if self.bounded_log_std:
            mean = torch.clamp(
                torch.nan_to_num(mean, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0
            )
            log_std = torch.clamp(
                torch.nan_to_num(
                    log_std,
                    nan=self.logstd_bias,
                    posinf=self.logstd_max,
                    neginf=self.logstd_min,
                ),
                self.logstd_min,
                self.logstd_max,
            )
        return Normal(mean, torch.exp(log_std), validate_args=not self.sanitize_inputs)


class Categorical(nn.Module):
    """Shared historical discrete output layer."""

    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = nn.init.orthogonal_ if args.use_orthogonal else nn.init.xavier_uniform_
        layer = nn.Linear(input_dim, output_dim)
        init_method(layer.weight, gain=args.gain)
        nn.init.constant_(layer.bias, 0)
        self.layer = nn.Sequential(layer, nn.Sigmoid())
        self.to(device)

    def forward(self, inputs):
        return self.layer(to_torch(inputs).to(**self.tpdv))
