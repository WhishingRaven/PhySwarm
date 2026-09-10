"""Foraging action channels in projector/checkpoint order."""

from learning.rmappo.action_heads.base import Categorical, NamedDiagGaussian


class DiagGaussian(NamedDiagGaussian):
    channel_names = ("food", "nest", "rand", "info", "diff", "pick", "drop")


__all__ = ["Categorical", "DiagGaussian"]
