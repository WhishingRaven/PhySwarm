"""Rescue action channels in projector/checkpoint order."""

from learning.rmappo.action_heads.base import Categorical, NamedDiagGaussian


class DiagGaussian(NamedDiagGaussian):
    channel_names = ("target", "center", "rand", "diff", "anchor", "release")


__all__ = ["Categorical", "DiagGaussian"]
