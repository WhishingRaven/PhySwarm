"""Discover and configure the native Webots controller runtime on macOS."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_WEBOTS_HOME = Path("/Applications/Webots.app")


def find_webots_home() -> Path:
    """Return a usable Webots application bundle."""
    configured = os.environ.get("WEBOTS_HOME")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((DEFAULT_WEBOTS_HOME, Path.home() / "Applications" / "Webots.app"))

    for candidate in candidates:
        python_api = candidate / "Contents" / "lib" / "controller" / "python"
        if python_api.is_dir():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Webots R2025a was not found. Install Webots.app or set WEBOTS_HOME "
        f"to its application bundle. Checked: {checked}"
    )


def configure_webots_runtime() -> Path:
    """Expose Webots' Python API and native controller library to this process."""
    webots_home = find_webots_home()
    controller_lib = webots_home / "Contents" / "lib" / "controller"
    python_api = controller_lib / "python"

    os.environ["WEBOTS_HOME"] = str(webots_home)
    existing_library_path = os.environ.get("DYLD_LIBRARY_PATH")
    library_parts = [str(controller_lib)]
    if existing_library_path:
        library_parts.append(existing_library_path)
    os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(library_parts)

    python_api_string = str(python_api)
    if python_api_string not in sys.path:
        sys.path.insert(0, python_api_string)
    return webots_home

