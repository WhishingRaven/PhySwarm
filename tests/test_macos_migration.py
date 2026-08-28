from __future__ import annotations

import importlib.util
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ROOT / "controllers"
sys.path.insert(0, str(CONTROLLERS))

from common.device import select_torch_device
from common.gymnasium_api import episode_end_flags
from common.webots_runtime import configure_webots_runtime


SCENARIOS = ("Swarm_Foraging", "Swarm_Navigation", "Swarm_Rescue")


def test_conda_and_pip_manifests_match() -> None:
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    environment_lines = (ROOT / "environment.yml").read_text().splitlines()
    pip_requirements = {
        line.strip()[2:]
        for line in environment_lines
        if re.match(r"^\s{6}- \S", line)
    }

    assert "python=3.13" in {line.strip()[2:] for line in environment_lines if line.startswith("  - ")}
    assert requirements == pip_requirements


def test_runtime_sources_have_no_legacy_platform_assumptions() -> None:
    sources = [ROOT / "environment.yml", ROOT / "requirements.txt"]
    sources.extend(
        path
        for root in (CONTROLLERS, ROOT / "worlds")
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".sh", ".yml", ".txt", ".wbt", ".wbproj"}
    )
    text = "\n".join(path.read_text(errors="ignore") for path in sources).lower()

    forbidden = (
        "cu" + "da",
        "nvi" + "dia",
        "tri" + "ton",
        "deep" + "bots",
        "from " + "gym.",
        "import " + "gym\n",
        "/usr/local/" + "webots",
        "r2023" + "b",
        "epuck_" + "rlcontroller_nonros",
        "torch_" + "scatter",
        "torch" + "viz",
    )
    assert not [token for token in forbidden if token in text]


def test_episode_end_flags_separate_termination_and_time_limit() -> None:
    alive = np.array([[True, False], [False, False]], dtype=np.bool_)
    early_stop = np.array([[False], [False]], dtype=np.bool_)
    steps = np.array([5, 3])

    terminated, truncated = episode_end_flags(alive, early_stop, steps, 5)

    np.testing.assert_array_equal(
        terminated[..., 0], np.array([[False, True], [True, True]])
    )
    np.testing.assert_array_equal(
        truncated[..., 0], np.array([[True, True], [False, False]])
    )
    assert terminated.dtype == np.bool_
    assert truncated.dtype == np.bool_


def test_early_stop_terminates_entire_environment() -> None:
    terminated, truncated = episode_end_flags(
        np.ones((1, 3), dtype=np.bool_),
        np.array([[True]]),
        np.array([1]),
        10,
    )
    assert terminated.all()
    assert not truncated.any()


def test_selected_torch_backend_executes_recurrent_workload() -> None:
    device = select_torch_device("auto")
    gru = torch.nn.GRU(4, 8, batch_first=True).to(device)
    inputs = torch.randn(2, 3, 4, device=device, requires_grad=True)
    output, _ = gru(inputs)
    output.square().mean().backward()

    assert output.device.type == device.type
    assert np.isfinite(output.detach().cpu().numpy()).all()
    assert select_torch_device("cpu").type == "cpu"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_gumbel_softmax_is_device_agnostic(scenario: str) -> None:
    util_path = CONTROLLERS / scenario / "supervisor_controller" / "utils" / "util.py"
    spec = importlib.util.spec_from_file_location(f"{scenario}_util", util_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device = select_torch_device("auto")
    logits = torch.zeros((4, 3), device=device)
    result = module.gumbel_softmax_sample(logits, None, temperature=1.0, device=device)

    assert result.device.type == device.type
    torch.testing.assert_close(result.sum(dim=-1).cpu(), torch.ones(4))


def test_all_checkpoints_deserialize_on_selected_backend() -> None:
    device = select_torch_device("auto")
    model_files = sorted(CONTROLLERS.glob("Swarm_*/models/policy_0/*.pt"))
    assert len(model_files) == 12
    for model_file in model_files:
        state = torch.load(model_file, map_location=device, weights_only=True)
        assert isinstance(state, dict)
        assert state


@pytest.mark.parametrize(
    ("scenario", "obs_dim", "act_dim", "extra_args"),
    (
        ("Swarm_Foraging", 17, 7, ("--num_target", "2")),
        ("Swarm_Navigation", 16, 4, ("--num_target", "0", "--num_obs_targets", "0")),
        ("Swarm_Rescue", 15, 6, ("--num_target", "1")),
    ),
)
def test_policy_checkpoints_load_and_run(
    scenario: str,
    obs_dim: int,
    act_dim: int,
    extra_args: tuple[str, ...],
) -> None:
    from gymnasium.spaces import Box

    for module_name in tuple(sys.modules):
        if module_name == "supervisor_controller" or module_name.startswith("supervisor_controller."):
            del sys.modules[module_name]

    scenario_root = CONTROLLERS / scenario
    sys.path.insert(0, str(scenario_root))
    try:
        config_module = importlib.import_module("supervisor_controller.config")
        policy_module = importlib.import_module(
            "supervisor_controller.r_mappo.algorithm.rMAPPOPolicy"
        )
        args = config_module.get_config().parse_args(
            [
                "--algorithm_name",
                "mappo",
                "--num_agents",
                "8",
                "--use_feature_normalization",
                "--use_vfunction",
                *extra_args,
            ]
        )
        device = select_torch_device("auto")
        policy = policy_module.R_MAPPOPolicy(
            {"args": args, "device": device, "num_agents": 8},
            {
                "obs_space": Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
                "act_space": Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
                "cent_obs_dim": obs_dim * 8,
            },
        )

        model_dir = scenario_root / "models" / "policy_0"
        modules = {
            "rnn_network.pt": policy.rnn_network,
            "act.pt": policy.act,
            "rnn_critic.pt": policy.rnn_critic_network,
            "v_network.pt": policy.v_network,
        }
        for filename, model in modules.items():
            state = torch.load(model_dir / filename, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=True)

        batch = 8
        observations = torch.zeros((batch, obs_dim), device=device)
        recurrent_state = torch.zeros((batch, args.hidden_size), device=device)
        neighbor_counts = torch.zeros(
            (batch, args.num_obs_targets + args.num_obs_agents),
            dtype=torch.int64,
            device=device,
        )
        hidden, next_state, _ = policy.get_hidden_states(
            observations,
            torch.zeros((batch, act_dim), device=device),
            recurrent_state,
            neighbor_counts,
        )
        assert hidden.shape == (1, batch, args.hidden_size)
        assert next_state.shape == recurrent_state.shape
    finally:
        sys.path.remove(str(scenario_root))
        for module_name in tuple(sys.modules):
            if module_name == "supervisor_controller" or module_name.startswith("supervisor_controller."):
                del sys.modules[module_name]


def test_world_targets_webots_r2025a() -> None:
    world = (ROOT / "worlds" / "generated_world.wbt").read_text()
    project = (ROOT / "worlds" / ".generated_world.wbproj").read_text()
    generator = (ROOT / "worlds" / "generate_wbt.py").read_text()

    assert world.startswith("#VRML_SIM R2025a utf8\n")
    assert "/R2025a/projects/" in world
    assert project.startswith("Webots Project File version R2025a\n")
    assert 'WEBOTS_VERSION = "R2025a"' in generator


@pytest.mark.skipif(not Path("/Applications/Webots.app").is_dir(), reason="Webots is not installed")
def test_native_webots_python_api_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBOTS_HOME", "/Applications/Webots.app")
    webots_home = configure_webots_runtime()
    import controller

    assert webots_home == Path("/Applications/Webots.app").resolve()
    assert str(webots_home / "Contents/lib/controller/python") in str(Path(controller.__file__))
    assert os.environ["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(
        webots_home / "Contents/lib/controller"
    )
