"""Internal Python process that connects the experiment to Webots."""

from __future__ import annotations

import sys

from app.training import run_experiment


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        raise SystemExit("usage: python -m app.controller_process ACTION SCENARIO [MAPPO options]")
    action, scenario, *options = arguments
    return run_experiment(action, scenario, options)


if __name__ == "__main__":
    raise SystemExit(main())
