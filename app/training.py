"""Shared train/evaluate/infer lifecycle for all PhySwarm tasks."""

from __future__ import annotations

import os
import random
import socket
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import setproctitle
import torch
import wandb

from adapters.webots.runtime import configure_webots_runtime
from learning.device import configure_torch, select_torch_device
from learning.rmappo.config import get_config
from learning.rmappo.policy import R_MAPPOPolicy
from learning.rmappo.registry import components_for
from learning.rmappo.rollout import SwarmRolloutRunner
from learning.rmappo.utils import get_cent_act_dim, get_dim_from_space


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _new_run_directory(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    run_numbers = [
        int(path.name[3:])
        for path in base.iterdir()
        if path.is_dir() and path.name.startswith("run") and path.name[3:].isdigit()
    ]
    run_dir = base / f"run{max(run_numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def run_experiment(action: str, scenario: str, argv: Sequence[str]) -> int:
    """Execute a configured MAPPO operation after Webots is ready."""

    configure_webots_runtime()
    from adapters.webots.scenarios import environment_class

    parser = get_config(scenario)
    args = parser.parse_args(list(argv))
    args.scenario = scenario
    if action not in {"train", "evaluate", "infer"}:
        raise ValueError(f"unknown action {action!r}")
    if action != "train" and not args.model_dir:
        parser.error(f"{action} requires --model_dir")
    if action == "train" and args.resume_training and not args.model_dir:
        parser.error("resume training requires --model_dir")

    device = select_torch_device(args.device)
    configure_torch(args.seed, args.n_training_threads, args.deterministic)
    print(f"PyTorch device: {device}")

    base_run_dir = (
        PROJECT_ROOT
        / "results"
        / scenario
        / args.env_name
        / args.algorithm_name
        / args.experiment_name
    )
    if args.resume_training:
        run_dir = Path(args.model_dir).resolve().parent
        if not run_dir.exists():
            raise FileNotFoundError(f"run directory not found: {run_dir}")
    elif action in {"evaluate", "infer"}:
        run_dir = _new_run_directory(PROJECT_ROOT / "results" / scenario / action)
        print(f"Writing {action} outputs to: {run_dir}")
    else:
        run_dir = base_run_dir if args.use_wandb else _new_run_directory(base_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating new run directory: {run_dir}")

    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            config=args,
            project=args.env_name,
            entity=args.user_name,
            notes=socket.gethostname(),
            name=f"{scenario}_{args.algorithm_name}_{args.experiment_name}_seed{args.seed}",
            dir=str(run_dir),
            job_type=action,
            reinit=False,
            resume="allow",
            id=run_dir.name if args.resume_training else None,
        )

    setproctitle.setproctitle(
        f"physwarm-{scenario}-{action}-{args.algorithm_name}@{args.user_name}"
    )
    np.random.seed(args.seed)
    random.seed(args.seed)

    environment_type = environment_class(scenario)
    environment = environment_type(args)
    components = components_for(scenario)
    num_agents = args.num_agents
    if args.share_policy:
        print(environment.agent_state_spaces[0])
        policy_info = {
            "policy_0": {
                "cent_obs_dim": get_dim_from_space(environment.agent_state_spaces[0]),
                "cent_act_dim": get_cent_act_dim(environment.agent_action_spaces),
                "obs_space": environment.agent_observation_spaces[0],
                "share_obs_space": environment.agent_state_spaces[0],
                "act_space": environment.agent_action_spaces[0],
            }
        }
        policy_mapping_fn = lambda _agent_id: "policy_0"
    else:
        policy_info = {
            f"policy_{agent_id}": {
                "cent_obs_dim": get_dim_from_space(environment.agent_state_spaces[agent_id]),
                "cent_act_dim": get_cent_act_dim(environment.agent_action_spaces),
                "obs_space": environment.agent_observation_spaces[agent_id],
                "share_obs_space": environment.agent_state_spaces[agent_id],
                "act_space": environment.agent_action_spaces[agent_id],
            }
            for agent_id in range(num_agents)
        }
        policy_mapping_fn = lambda agent_id: f"policy_{agent_id}"

    runner_config = {
        "args": args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": environment,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "use_same_share_obs": args.use_same_share_obs,
        "use_available_actions": args.use_available_actions,
        "adj": torch.zeros((args.num_agents, args.num_factor), dtype=torch.int64),
        "policy_class": R_MAPPOPolicy,
        "trainer_class": components.trainer,
        "action_head_class": components.action_head,
        "categorical_head_class": components.categorical_head,
    }
    runner = SwarmRolloutRunner(config=runner_config)
    pd.DataFrame(list(vars(args).items()), columns=["Name", "Value"]).to_csv(
        run_dir / "config.csv", index=False
    )

    try:
        if action in {"evaluate", "infer"}:
            runner.restore()
            runner.eval()
        elif args.resume_training:
            if not os.path.exists(runner.checkpoint_path):
                raise FileNotFoundError(
                    f"checkpoint file not found: {runner.checkpoint_path}"
                )
            runner.restore()
            runner.load_checkpoint()
            while runner.total_env_steps < args.num_env_steps:
                runner.run()
        else:
            while runner.total_env_steps < args.num_env_steps:
                runner.run()
    except KeyboardInterrupt:
        if action == "train":
            print("\nTraining interrupted; saving a final checkpoint.")
            runner.save(is_checkpoint=True)
    finally:
        environment.close()
        if wandb_run is not None:
            wandb_run.finish()
        else:
            runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
            runner.writter.close()
    return 0
