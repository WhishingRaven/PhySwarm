from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTROLLERS = ROOT / "controllers"

from adapters.gymnasium import episode_end_flags
from adapters.webots.robot_runtime import parse_robot_runtime_options
from adapters.webots.runtime import configure_webots_runtime
from experiments.trajectory import TrajectoryLogPlan
from learning.device import select_torch_device
from learning.rmappo.action_heads.foraging import DiagGaussian
from learning.rmappo.config import get_config
from learning.rmappo.policy import R_MAPPOPolicy
from learning.rmappo.registry import components_for
from learning.rmappo.utils import gumbel_softmax_sample
from physics.layouts import get_parameter_layout
from physics.parameters import project_numpy_parameters
from tasks.specs import get_runtime_macro_spec


SCENARIOS = ("foraging", "navigation", "rescue")
CONTROLLER_DIRECTORIES = {
    "foraging": "Swarm_Foraging",
    "navigation": "Swarm_Navigation",
    "rescue": "Swarm_Rescue",
}


def test_python_entrypoints_replace_scenario_shell_launchers() -> None:
    for entrypoint in ("train.py", "evaluate.py", "infer.py"):
        assert (ROOT / entrypoint).is_file()
    for directory in CONTROLLER_DIRECTORIES.values():
        assert not (
            CONTROLLERS / directory / "supervisor_controller" / "train_mappo.sh"
        ).exists()


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


def test_shared_epuck_runtime_has_no_scenario_dependency() -> None:
    defaults = parse_robot_runtime_options([])
    overridden = parse_robot_runtime_options(
        [
            "--timestep",
            "80",
            "--interval",
            "4",
            "--num_agents",
            "9",
            "--webots-owned-flag",
            "ignored",
        ]
    )

    assert (defaults.timestep, defaults.interval, defaults.num_agents) == (100, 5, 6)
    assert defaults.basic_timestep == 20
    assert overridden.basic_timestep == 20
    assert overridden.num_agents == 9
    with pytest.raises(ValueError, match="divisible"):
        parse_robot_runtime_options(["--timestep", "100", "--interval", "6"])

    source = (CONTROLLERS / "epuck_controller" / "epuck_controller.py").read_text()
    assert "Swarm_Foraging" not in source
    assert "supervisor_controller.config" not in source


def test_trajectory_log_plan_is_explicit_single_env_and_non_overwriting(
    tmp_path: Path,
) -> None:
    disabled = TrajectoryLogPlan.from_options(scenario="foraging")
    assert disabled.enabled is False
    with pytest.raises(RuntimeError, match="disabled"):
        disabled.path_for_episode(0)

    with pytest.raises(ValueError, match="n_rollout_threads 1"):
        TrajectoryLogPlan.from_options(
            scenario="foraging", save_trajectory=True, num_envs=2
        )

    explicit = TrajectoryLogPlan.from_options(
        scenario="navigation",
        trajectory_output=tmp_path / "run" / "trajectory.csv",
    )
    first = explicit.path_for_episode(0)
    second = explicit.path_for_episode(1)
    assert first == tmp_path / "run" / "trajectory.csv"
    assert second == tmp_path / "run" / "trajectory_episode_0001.csv"
    assert first.parent.is_dir()

    first.write_text("step,agent_id\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        explicit.path_for_episode(0)

    directory = TrajectoryLogPlan.from_options(
        scenario="rescue", save_trajectory=True, trajectory_output=tmp_path / "episodes"
    )
    assert directory.path_for_episode(3) == (
        tmp_path / "episodes" / "rescue_trajectory_episode_0003.csv"
    )
    with pytest.raises(ValueError, match="non-negative"):
        directory.path_for_episode(-1)


@pytest.mark.parametrize("scenario", ("foraging", "rescue"))
def test_trajectory_logger_uses_the_current_wheel_command(scenario: str) -> None:
    source = (ROOT / "adapters" / "webots" / "scenarios" / f"{scenario}.py").read_text(
        encoding="utf-8"
    )

    assert "self.last_wheel_speeds[" not in source
    assert "save_test_data(pde_params, target_wheel_speeds, done)" in source
    assert "float(target_wheel_speeds[env_idx, i, 0])" in source
    assert "float(target_wheel_speeds[env_idx, i, 1])" in source


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


def test_gumbel_softmax_is_device_agnostic() -> None:
    device = select_torch_device("auto")
    logits = torch.zeros((4, 3), device=device)
    result = gumbel_softmax_sample(logits, None, temperature=1.0, device=device)

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
        ("foraging", 17, 7, ("--num_target", "2")),
        ("navigation", 16, 4, ("--num_target", "0", "--num_obs_targets", "0")),
        ("rescue", 15, 6, ("--num_target", "1")),
    ),
)
def test_policy_checkpoints_load_and_run(
    scenario: str,
    obs_dim: int,
    act_dim: int,
    extra_args: tuple[str, ...],
) -> None:
    from gymnasium.spaces import Box

    args = get_config(scenario).parse_args(
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
    components = components_for(scenario)
    policy = R_MAPPOPolicy(
        {
            "args": args,
            "device": device,
            "num_agents": 8,
            "action_head_class": components.action_head,
            "categorical_head_class": components.categorical_head,
        },
        {
            "obs_space": Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
            "act_space": Box(-1.0, 1.0, (act_dim,), dtype=np.float32),
            "cent_obs_dim": obs_dim * 8,
        },
    )

    model_dir = CONTROLLERS / CONTROLLER_DIRECTORIES[scenario] / "models" / "policy_0"
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


def test_foraging_actor_channel_order_matches_environment_contract() -> None:
    """정책 head와 환경의 action index가 같은 물리량을 뜻해야 한다.

    Foraging 환경과 PINN mapper의 계약은
    ``food, nest, random, information, diffusion, pick, drop`` 순서다.
    각 mean head에 구별되는 bias를 넣어 실제 결합 순서를 검증한다.
    """

    layer = DiagGaussian(
        args=None,
        act_dim=7,
        rnn_input_dim=1,
        obs_dim=1,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        ordered_heads = (
            layer.food_mu,
            layer.nest_mu,
            layer.rand_mu,
            layer.info_mu,
            layer.diff_mu,
            layer.pick_mu,
            layer.drop_mu,
        )
        for expected_index, head in enumerate(ordered_heads, start=1):
            head.bias.fill_(float(expected_index))

    distribution = layer(torch.zeros((1, 1)), torch.zeros((1, 1)))
    torch.testing.assert_close(
        distribution.loc,
        torch.arange(1.0, 8.0).reshape(1, 7),
    )


def test_foraging_actor_recovers_from_nonfinite_recurrent_features() -> None:
    layer = DiagGaussian(
        args=None,
        act_dim=7,
        rnn_input_dim=64,
        obs_dim=17,
        device=torch.device("cpu"),
    )

    distribution = layer(
        torch.full((8, 64), float("nan")),
        torch.zeros((8, 17)),
    )

    assert torch.isfinite(distribution.loc).all()
    assert torch.isfinite(distribution.scale).all()
    assert torch.all(distribution.scale > 0)


@pytest.mark.parametrize(
    "scenario_key",
    SCENARIOS,
)
def test_runtime_and_trainer_reference_one_legacy_projection_contract(
    scenario_key: str,
) -> None:
    """환경/학습기가 같은 이름 있는 profile을 참조하도록 회귀 고정한다."""

    runtime_source = (
        ROOT / "adapters" / "webots" / "scenarios" / f"{scenario_key}.py"
    ).read_text()
    assert "project_numpy_parameters" in runtime_source
    assert f'get_parameter_layout("{scenario_key}", contract="legacy")' in runtime_source

    trainer_type = components_for(scenario_key).trainer
    layout = get_parameter_layout(scenario_key, contract="legacy")
    raw = np.linspace(
        -1.5,
        1.5,
        num=2 * layout.action_dim,
        dtype=np.float32,
    ).reshape(2, layout.action_dim)
    trainer = trainer_type.__new__(trainer_type)
    actual = trainer._torch_map_actions_to_params(torch.from_numpy(raw))
    expected = project_numpy_parameters(raw, layout).as_named_dict(layout)

    for name, expected_values in expected.items():
        actual_values = actual[name].detach().cpu().numpy()
        if actual_values.ndim == expected_values.ndim + 1:
            actual_values = np.squeeze(actual_values, axis=-1)
        np.testing.assert_allclose(actual_values, expected_values, rtol=1e-6, atol=1e-7)
    if scenario_key == "navigation":
        np.testing.assert_allclose(actual["alpha"].numpy(), 1.0)


@pytest.mark.parametrize(
    "scenario_key",
    ("foraging", "navigation"),
)
def test_new_runs_can_select_paper_parameter_manifold(
    scenario_key: str,
) -> None:
    """같은 actor 차원을 유지하는 과제는 paper sigmoid 계약을 end-to-end 선택한다."""

    args = get_config(scenario_key).parse_args(
        [
            "--algorithm_name",
            "mappo",
            "--parameter_contract",
            "paper",
            "--phase_contract",
            "paper",
        ]
    )
    assert args.parameter_contract == "paper"
    assert args.phase_contract == "paper"
    assert len(get_runtime_macro_spec(scenario_key, phase_contract="paper").phase_names) == 4

    trainer_type = components_for(scenario_key).trainer
    layout = get_parameter_layout(scenario_key, contract="paper")
    trainer = trainer_type.__new__(trainer_type)
    trainer.parameter_layout = layout
    raw = np.zeros((2, layout.action_dim), dtype=np.float32)
    actual = trainer._torch_map_actions_to_params(torch.from_numpy(raw))
    expected = project_numpy_parameters(raw, layout).as_named_dict(layout)
    for name, expected_values in expected.items():
        values = actual[name].detach().cpu().numpy()
        if values.ndim == expected_values.ndim + 1:
            values = np.squeeze(values, axis=-1)
        np.testing.assert_allclose(values, expected_values, rtol=1e-6)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_trajectory_logging_cli_is_explicit(scenario: str) -> None:
    parser = get_config(scenario)
    defaults = parser.parse_args(["--algorithm_name", "mappo"])
    enabled = parser.parse_args(
        [
            "--algorithm_name",
            "mappo",
            "--save_trajectory",
            "--trajectory_output",
            "/tmp/trajectory.csv",
        ]
    )
    assert defaults.save_trajectory is False
    assert defaults.trajectory_output is None
    assert enabled.save_trajectory is True
    assert enabled.trajectory_output == "/tmp/trajectory.csv"


def test_rescue_robustness_is_explicit_and_standard_task_is_default() -> None:
    parser = get_config("rescue")
    assert parser.parse_args(["--algorithm_name", "mappo"]).robust_test is False
    assert parser.parse_args(
        ["--algorithm_name", "mappo", "--robust_test"]
    ).robust_test is True


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
