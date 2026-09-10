"""Run a task controller and own its Webots process lifecycle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from adapters.webots.launcher import WebotsSession
from tasks.profiles import TASK_PROFILES, get_task_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"PhySwarm {action} using the Python Webots adapter",
        epilog="Unknown options are forwarded to the MAPPO configuration parser.",
    )
    parser.add_argument("scenario", choices=sorted(TASK_PROFILES))
    parser.add_argument(
        "--webots",
        choices=("existing", "fast", "realtime"),
        default="existing",
        help="reuse Webots or launch a headless/GUI process",
    )
    parser.add_argument("--controller-url", default="ipc://1234/supervisor")
    parser.add_argument("--webots-home", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true", help="print the controller command only")
    parser.add_argument("--model-dir", type=Path)
    if action == "train":
        parser.add_argument("--resume", action="store_true")
    else:
        parser.add_argument("--episodes", type=int, default=1 if action == "infer" else None)
    return parser


def controller_command(
    action: str,
    scenario: str,
    *,
    model_dir: Path | None = None,
    resume: bool = False,
    episodes: int | None = None,
    extra: Sequence[str] = (),
) -> tuple[list[str], Path]:
    """Build the exact Python controller invocation for one operation."""

    profile = get_task_profile(scenario)
    command = [
        sys.executable,
        "-m",
        "app.controller_process",
        action,
        scenario,
        *profile.training_arguments(),
    ]

    if action in {"evaluate", "infer"} and model_dir is None:
        raise ValueError(f"{action} requires --model-dir")
    if resume and model_dir is None:
        raise ValueError("--resume requires --model-dir")
    if model_dir is not None:
        # Current checkpoint reader appends the policy id. Preserve its on-disk
        # contract while accepting paths with or without a trailing separator.
        command.extend(("--model_dir", str(Path(model_dir).resolve()) + os.sep))
    if resume:
        command.append("--resume_training")
    if episodes is not None:
        if episodes <= 0:
            raise ValueError("--episodes must be positive")
        command.extend(("--num_eval_episodes", str(episodes)))
    forwarded = list(extra)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    command.extend(forwarded)
    return command, PROJECT_ROOT


def main(action: str, argv: Sequence[str] | None = None) -> int:
    parser = build_parser(action)
    options, extra = parser.parse_known_args(argv)
    try:
        command, working_directory = controller_command(
            action,
            options.scenario,
            model_dir=options.model_dir,
            resume=getattr(options, "resume", False),
            episodes=getattr(options, "episodes", None),
            extra=extra,
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error))

    if options.dry_run:
        print(subprocess.list2cmdline(command))
        return 0

    session = WebotsSession(
        mode=options.webots,
        project_root=PROJECT_ROOT,
        controller_url=options.controller_url,
        startup_timeout=options.startup_timeout,
        webots_home=options.webots_home,
    )
    with session:
        completed = subprocess.run(
            command,
            cwd=working_directory,
            env=session.environment(),
            check=False,
        )
    return completed.returncode
