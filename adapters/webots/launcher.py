"""Start and stop Webots without a shell wrapper."""

from __future__ import annotations

import getpass
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from adapters.webots.runtime import find_webots_home


def external_controller_endpoint(controller_url: str) -> Path:
    """Translate ``ipc://PORT/NAME`` to Webots' local extern endpoint."""

    match = re.fullmatch(r"ipc://([0-9]+)/([^/]+)", controller_url)
    if match is None:
        raise ValueError(
            f"invalid Webots controller URL {controller_url!r}; expected ipc://PORT/NAME"
        )
    port, controller_name = match.groups()
    return (
        Path("/tmp/webots")
        / getpass.getuser()
        / port
        / "ipc"
        / controller_name
        / "extern"
    )


@dataclass
class WebotsSession:
    """Context manager owning one optional Webots subprocess."""

    mode: str
    project_root: Path
    controller_url: str = "ipc://1234/supervisor"
    startup_timeout: float = 30.0
    webots_home: Path | None = None

    def __post_init__(self) -> None:
        aliases = {"slow": "realtime"}
        self.mode = aliases.get(self.mode, self.mode)
        if self.mode not in {"existing", "fast", "realtime"}:
            raise ValueError("mode must be existing, fast, or realtime")
        self.project_root = Path(self.project_root).resolve()
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None
        self._log_file = None

    @property
    def endpoint(self) -> Path:
        return external_controller_endpoint(self.controller_url)

    def environment(self) -> dict[str, str]:
        """Return the process environment shared by Webots and its controller."""

        home = Path(self.webots_home or find_webots_home()).resolve()
        controller_lib = home / "Contents" / "lib" / "controller"
        python_api = controller_lib / "python"
        env = os.environ.copy()
        env["WEBOTS_HOME"] = str(home)
        env["WEBOTS_CONTROLLER_URL"] = self.controller_url
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.project_root), str(python_api), env.get("PYTHONPATH")) if part
        )
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (str(controller_lib), env.get("DYLD_LIBRARY_PATH")) if part
        )
        return env

    def start(self) -> "WebotsSession":
        endpoint = self.endpoint
        if self.mode == "existing":
            if not endpoint.exists():
                raise RuntimeError(
                    f"no Webots external controller at {endpoint}; start the world or use --webots fast"
                )
            return self

        if endpoint.exists():
            raise RuntimeError(
                f"Webots external controller already exists at {endpoint}; use --webots existing"
            )

        env = self.environment()
        home = Path(env["WEBOTS_HOME"])
        executable = home / "Contents" / "MacOS" / "webots"
        world = self.project_root / "worlds" / "generated_world.wbt"
        if not executable.is_file():
            raise FileNotFoundError(f"Webots executable not found: {executable}")
        if not world.is_file():
            raise FileNotFoundError(f"Webots world not found: {world}")

        options = ["--batch", "--mode=fast", "--no-rendering"] if self.mode == "fast" else ["--mode=realtime"]
        fd, log_name = tempfile.mkstemp(prefix="physwarm-webots.")
        os.close(fd)
        self.log_path = Path(log_name)
        self._log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [str(executable), *options, "--stdout", "--stderr", "--extern-urls", str(world)],
            cwd=self.project_root,
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"Webots {self.mode} mode starting (PID {self.process.pid})")
        print(f"Webots log: {self.log_path}")

        deadline = time.monotonic() + self.startup_timeout
        while not endpoint.exists():
            if self.process.poll() is not None:
                raise RuntimeError(self._startup_error("Webots exited during startup"))
            if time.monotonic() >= deadline:
                raise TimeoutError(self._startup_error(f"timed out waiting for {endpoint}"))
            time.sleep(0.1)
        print(f"Webots external controller is ready: {endpoint}")
        return self

    def _startup_error(self, message: str) -> str:
        if self._log_file is not None:
            self._log_file.flush()
        tail = ""
        if self.log_path and self.log_path.exists():
            tail = self.log_path.read_text(errors="replace")[-4000:]
        return f"{message}\n{tail}" if tail else message

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
        self.process = None
        self._log_file = None

    def __enter__(self) -> "WebotsSession":
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
