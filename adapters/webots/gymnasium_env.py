"""Small Gymnasium-compatible Webots controller bases."""

from __future__ import annotations

from gymnasium import Env
from controller import Supervisor


class WebotsSupervisorEnv(Supervisor, Env):
    """Supervisor base that combines Webots with the Gymnasium interface."""

    metadata = {"render_modes": []}

    def __init__(self, timestep: int | None = None):
        Supervisor.__init__(self)
        Env.__init__(self)
        self.timestep = int(timestep or self.getBasicTimeStep())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        Env.reset(self, seed=seed)
