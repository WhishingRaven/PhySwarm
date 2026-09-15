from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from adapters.webots.launcher import external_controller_endpoint
from app.execute import controller_command
from learning.rmappo.config import get_config
from learning.rmappo.runner import RecurrentRunner
from tasks.profiles import get_task_profile


def test_webots_controller_url_maps_to_extern_endpoint():
    endpoint = external_controller_endpoint("ipc://1234/supervisor")
    assert endpoint == Path(f"/tmp/webots/{getpass.getuser()}/1234/ipc/supervisor/extern")


@pytest.mark.parametrize("invalid", ["", "tcp://1234/supervisor", "ipc://x/name", "ipc://1234/"])
def test_webots_controller_url_rejects_invalid_values(invalid):
    with pytest.raises(ValueError):
        external_controller_endpoint(invalid)


def test_task_profiles_preserve_scenario_specific_rollout_defaults():
    assert get_task_profile("foraging").episode_length == 1000
    assert get_task_profile("navigation").extra[-2:] == ("--num_obs_targets", "0")
    assert get_task_profile("rescue").targets == 1


@pytest.mark.parametrize("scenario", ("foraging", "navigation", "rescue"))
def test_task_profile_arguments_are_all_owned_by_shared_config(scenario):
    get_config(scenario).parse_args(get_task_profile(scenario).training_arguments())


def test_train_command_uses_python_and_forwards_overrides():
    command, cwd = controller_command(
        "train", "foraging", extra=("--num_env_steps", "4")
    )
    assert command[0] == sys.executable
    assert command[1:5] == ["-m", "app.controller_process", "train", "foraging"]
    assert cwd.name == "PhySwarm"
    assert command[-2:] == ["--num_env_steps", "4"]


def test_evaluation_normalizes_model_directory_and_episode_count(tmp_path):
    command, _ = controller_command(
        "evaluate", "navigation", model_dir=tmp_path / "models", episodes=3
    )
    index = command.index("--model_dir")
    assert command[index + 1].endswith(os.sep)
    assert command[-2:] == ["--num_eval_episodes", "3"]


def test_resume_requires_checkpoint_directory():
    with pytest.raises(ValueError, match="requires --model-dir"):
        controller_command("train", "rescue", resume=True)


def test_training_checkpoint_precedes_periodic_evaluation(capsys):
    runner = RecurrentRunner.__new__(RecurrentRunner)
    events = []
    runner.trainer = SimpleNamespace(prep_rollout=lambda: None)
    runner.use_linear_lr_decay = False
    runner.collecter = lambda **kwargs: {}
    runner.env_infos = {}
    runner.num_episodes_collected = 0
    runner.last_train_episode = 0
    runner.train_interval_episode = 32
    runner.total_env_steps = 100
    runner.num_env_steps = 100
    runner.num_envs = 1
    runner.episode_length = 100
    runner.buffer_size = 32
    runner.last_log_T = 100
    runner.log_interval = 100
    runner.use_save = True
    runner.last_save_T = 0
    runner.save_interval = 100
    runner.saver = lambda **kwargs: events.append("save")
    runner.use_eval = True
    runner.last_eval_T = 0
    runner.eval_interval = 100
    runner.eval = lambda: events.append("eval")

    runner.run()

    assert events == ["save", "eval"]
    assert "[rollout] step=100/100 (100.00%)" in capsys.readouterr().out


def test_checkpoint_resume_restarts_empty_on_policy_collection_window(tmp_path):
    class OptimizerStub:
        def __init__(self):
            self.loaded_state = None

        def load_state_dict(self, state):
            self.loaded_state = state

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "total_env_steps": 1_950_000,
            "num_episodes_collected": 1_950,
            "last_train_episode": 1_920,
            "last_save_T": 1_950_000,
            "last_log_T": 1_950_000,
            "last_eval_T": 1_900_000,
            "actor_optimizer_state_dict": {"state": {}, "param_groups": []},
            "critic_optimizer_state_dict": {"state": {}, "param_groups": []},
        },
        checkpoint_path,
    )
    runner = RecurrentRunner.__new__(RecurrentRunner)
    runner.checkpoint_path = str(checkpoint_path)
    runner.device = torch.device("cpu")
    runner.trainer = SimpleNamespace(
        actor_optimizer=OptimizerStub(),
        critic_optimizer=OptimizerStub(),
        _use_valuenorm=False,
    )

    runner.load_checkpoint()

    assert runner.total_env_steps == 1_950_000
    assert runner.num_episodes_collected == 1_950
    assert runner.last_train_episode == 1_950


def test_old_scenario_execution_trees_have_no_python_or_shell_entrypoints():
    root = Path(__file__).resolve().parents[1]
    for directory in ("Swarm_Foraging", "Swarm_Navigation", "Swarm_Rescue"):
        old_root = root / "controllers" / directory / "supervisor_controller"
        assert not list(old_root.rglob("*.py"))
        assert not list(old_root.rglob("*.sh"))


def test_learning_core_does_not_import_webots_adapter():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "learning").rglob("*.py")
    )
    assert "adapters.webots" not in source
