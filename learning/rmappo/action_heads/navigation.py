"""Navigation action channels in projector/checkpoint order."""

from learning.rmappo.action_heads.base import Categorical, NamedDiagGaussian


class DiagGaussian(NamedDiagGaussian):
    channel_names = ("flow", "shape", "diff", "beta")
    sanitize_inputs = True
    bounded_log_std = True


__all__ = ["Categorical", "DiagGaussian"]
