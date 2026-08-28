"""Shared Gymnasium episode-boundary semantics."""

from __future__ import annotations

import numpy as np


def episode_end_flags(
    alive_agents: np.ndarray,
    early_stop: np.ndarray,
    steps: np.ndarray,
    episode_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate task termination from the episode time limit."""
    alive_agents = np.asarray(alive_agents, dtype=np.bool_)
    shape = (*alive_agents.shape, 1)

    terminated = np.logical_not(alive_agents)[..., np.newaxis]
    environment_terminated = (
        np.logical_not(alive_agents.any(axis=-1))
        | np.asarray(early_stop, dtype=np.bool_).reshape(-1)
    )
    terminated = np.where(
        environment_terminated[:, np.newaxis, np.newaxis],
        np.ones(shape, dtype=np.bool_),
        terminated,
    )

    time_limit = np.asarray(steps) >= episode_length
    truncated = np.broadcast_to(
        time_limit[:, np.newaxis, np.newaxis], shape
    ).copy()
    return terminated, truncated
