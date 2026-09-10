"""Shared, opt-in trajectory CSV path handling for simulator episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryLogPlan:
    """Resolve one exclusive CSV path per episode.

    The existing trajectory schema has no environment identifier, so combining
    vectorized environments would silently mix unrelated agents and timesteps.
    Logging therefore stays disabled by default and rejects that ambiguous case.
    """

    scenario: str
    enabled: bool
    output: Path

    @classmethod
    def from_options(
        cls,
        *,
        scenario: str,
        save_trajectory: bool = False,
        trajectory_output: str | Path | None = None,
        num_envs: int = 1,
    ) -> "TrajectoryLogPlan":
        """Build a validated logging plan from supervisor CLI options."""

        enabled = bool(save_trajectory or trajectory_output is not None)
        if enabled and num_envs != 1:
            raise ValueError(
                "trajectory CSV logging requires --n_rollout_threads 1 because "
                "the trajectory schema does not contain an env_id column"
            )

        output = (
            Path(trajectory_output)
            if trajectory_output is not None
            else Path("data")
        )
        return cls(scenario=scenario, enabled=enabled, output=output)

    def path_for_episode(self, episode_index: int) -> Path:
        """Return a fresh path for ``episode_index`` and create its parent."""

        if not self.enabled:
            raise RuntimeError("trajectory logging is disabled")
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")

        if self.output.suffix.lower() == ".csv":
            if episode_index == 0:
                path = self.output
            else:
                path = self.output.with_name(
                    f"{self.output.stem}_episode_{episode_index:04d}.csv"
                )
        else:
            path = self.output / (
                f"{self.scenario}_trajectory_episode_{episode_index:04d}.csv"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"trajectory output already exists: {path}; choose a new "
                "--trajectory_output to preserve the existing experiment"
            )
        return path
