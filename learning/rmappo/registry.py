"""Resolve task-specific MAPPO action and physics strategies."""

from __future__ import annotations

from typing import NamedTuple


class AlgorithmComponents(NamedTuple):
    action_head: type
    categorical_head: type
    trainer: type


def components_for(scenario: str) -> AlgorithmComponents:
    if scenario == "foraging":
        from learning.rmappo.action_heads.foraging import Categorical, DiagGaussian
        from learning.rmappo.trainers.foraging import R_MAPPO
    elif scenario == "navigation":
        from learning.rmappo.action_heads.navigation import Categorical, DiagGaussian
        from learning.rmappo.trainers.navigation import R_MAPPO
    elif scenario == "rescue":
        from learning.rmappo.action_heads.rescue import Categorical, DiagGaussian
        from learning.rmappo.trainers.rescue import R_MAPPO
    else:
        raise KeyError(f"unknown MAPPO scenario {scenario!r}")
    return AlgorithmComponents(DiagGaussian, Categorical, R_MAPPO)
