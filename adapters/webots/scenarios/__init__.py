"""Webots environments for the three paper tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.webots.gymnasium_env import WebotsSupervisorEnv


def environment_class(scenario: str) -> type[WebotsSupervisorEnv]:
    """Load one environment lazily after the Webots runtime is configured."""

    if scenario == "foraging":
        from adapters.webots.scenarios.foraging import Epuck2Supervisor
    elif scenario == "navigation":
        from adapters.webots.scenarios.navigation import Epuck2Supervisor
    elif scenario == "rescue":
        from adapters.webots.scenarios.rescue import Epuck2Supervisor
    else:
        raise KeyError(f"unknown Webots scenario {scenario!r}")
    return Epuck2Supervisor
