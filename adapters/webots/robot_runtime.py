"""Task-independent runtime options for the shared e-puck controller.

The motor/sensor process is launched by Webots, separately from a scenario's
Python trainer.  Importing one scenario's full MAPPO argument parser here made
the supposedly shared robot controller depend on Foraging and also pulled many
irrelevant learning flags into the robot process.  This narrow parser owns only
the timing and swarm-size values the e-puck process actually reads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RobotRuntimeOptions:
    """Validated timing contract shared by the Webots robot processes.

    ``timestep`` is one supervisor control interval in milliseconds and
    ``interval`` is the number of basic Webots steps inside it.  Exact integer
    division matters: both the supervisor and robot controller must advance by
    the same basic duration or sensor samples and wheel commands drift apart.
    """

    timestep: int = 100
    interval: int = 5
    num_agents: int = 6

    @property
    def basic_timestep(self) -> int:
        """Duration of one Webots ``Robot.step`` and sensor sample in ms."""

        return self.timestep // self.interval


def parse_robot_runtime_options(
    argv: Sequence[str] | None = None,
) -> RobotRuntimeOptions:
    """Parse known robot flags while leaving Webots-specific arguments alone."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timestep", type=int, default=100)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--num_agents", type=int, default=6)
    parsed, _unknown = parser.parse_known_args(argv)

    if parsed.timestep <= 0:
        raise ValueError("timestep must be positive")
    if parsed.interval <= 0:
        raise ValueError("interval must be positive")
    if parsed.timestep % parsed.interval != 0:
        raise ValueError("timestep must be divisible by interval")
    if parsed.num_agents <= 0:
        raise ValueError("num_agents must be positive")
    return RobotRuntimeOptions(
        timestep=parsed.timestep,
        interval=parsed.interval,
        num_agents=parsed.num_agents,
    )
