"""논문 지표와 통합 CSV 분석기의 회귀 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import (
    RunSpec,
    analyze_csv,
    compare_training_runs,
    load_manifest,
    load_paper_values,
)
from experiments.metrics import (
    algebraic_connectivity,
    control_smoothness,
    formation_error,
    navigation_time,
    relay_chain_connectivity,
    task_success_rate,
    time_to_connectivity,
)


def test_formation_error_ignores_translation_and_rotation() -> None:
    target = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    quarter_turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    actual = target @ quarter_turn + np.array([4.0, -3.0])

    assert formation_error(actual, target) < 1e-12


def test_relay_connectivity_requires_base_target_path() -> None:
    connected_robots = np.array([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]])
    lambda2 = relay_chain_connectivity(
        connected_robots,
        base_position=np.array([-1.0, 0.0]),
        target_position=np.array([1.0, 0.0]),
        communication_range=0.75,
    )
    assert lambda2 > 0.0
    assert relay_chain_connectivity(
        connected_robots[[0]],
        base_position=np.array([-1.0, 0.0]),
        target_position=np.array([1.0, 0.0]),
        communication_range=0.75,
    ) == 0.0

    # 완전 그래프 K3의 Laplacian 고유값은 [0, 3, 3]이다.
    np.testing.assert_allclose(algebraic_connectivity(np.ones((3, 3)) - np.eye(3)), 3.0)


def test_paper_execution_time_and_smoothness_metrics_keep_failures() -> None:
    commands = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, -1.0], [1.0, -1.0]],
            [[3.0, -1.0], [3.0, -1.0]],
        ]
    )
    np.testing.assert_allclose(control_smoothness(commands), 2.0)

    positions = np.array(
        [
            [[-1.1, 0.0], [-0.9, 0.0]],
            [[-0.1, 0.0], [0.1, 0.0]],
            [[0.9, 0.0], [1.1, 0.0]],
        ]
    )
    assert navigation_time(
        positions,
        np.array([1.0, 0.0]),
        0.1,
        steps=np.array([2, 4, 6]),
        max_time=20,
    ) == 6
    assert navigation_time(
        positions,
        np.array([5.0, 0.0]),
        0.1,
        max_time=20,
    ) == 20

    trials = np.array([[0.0, 0.2, 0.3], [0.0, 0.0, 0.0]])
    assert task_success_rate(trials) == 0.5
    assert time_to_connectivity(trials[0], steps=np.array([5, 10, 15])) == 10
    assert time_to_connectivity(trials[1], max_time=50) == 50


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_trajectory_analysis_generates_metrics_and_multiple_views(tmp_path: Path) -> None:
    rows = []
    for step in range(1, 5):
        for agent_id in range(3):
            rows.append(
                {
                    "step": step,
                    "agent_id": agent_id,
                    "pos_x": -0.6 + 0.3 * agent_id + 0.05 * step,
                    "pos_y": 0.1 * (agent_id - 1),
                    "role": min(agent_id, 2),
                    "is_alive": 1,
                    "is_colliding": int(step == 2 and agent_id == 0),
                    "w_target": 0.02 * step,
                    "w_center": 0.03,
                    "w_rand": 0.08 - 0.01 * step,
                    "k_diff": 0.01,
                    "lambda_anchor": 0.4,
                    "lambda_release": 0.2,
                    "base_x": -1.0,
                    "base_y": 0.0,
                    "target_0_x": 1.0,
                    "target_0_y": 0.0,
                    "comm_view": 0.7,
                    "chain_connected": int(step >= 3),
                    "rescued_count": int(step == 4),
                }
            )
    input_path = tmp_path / "rescue.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    original = input_path.read_bytes()

    result = analyze_csv(input_path, tmp_path / "report")

    assert result.kind == "trajectory"
    assert result.scenario == "rescue"
    assert result.summary["task_success"] is True
    assert result.summary["time_to_connectivity"] == 3
    assert input_path.read_bytes() == original
    assert (result.output_directory / "summary.json").is_file()
    assert (result.output_directory / "per_step.csv").is_file()
    png_paths = [path for path in result.generated_files if path.suffix == ".png"]
    assert len(png_paths) >= 4
    for path in png_paths:
        _assert_png(path)


def test_training_analysis_handles_sparse_evaluation_log(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "step": [0, 100, 200],
            "average_episode_rewards": [1.0, 2.0, 3.5],
            "Reward/Collision": [-0.3, -0.2, -0.1],
            "Param_Explore/w_food": [0.02, 0.03, 0.04],
            "pinn/loss_macro": [1.2, 0.9, 0.6],
        }
    )
    input_path = tmp_path / "progress_eval.csv"
    frame.to_csv(input_path, index=False)

    result = analyze_csv(input_path, tmp_path / "training-report", scenario="foraging")

    assert result.kind == "training"
    assert result.scenario == "foraging"
    summary = json.loads((result.output_directory / "summary.json").read_text())
    assert summary["metrics"]["average_episode_rewards"]["last"] == 3.5
    png_paths = [path for path in result.generated_files if path.suffix == ".png"]
    assert len(png_paths) >= 3
    for path in png_paths:
        _assert_png(path)


def test_navigation_trajectory_reports_s174_and_s181(tmp_path: Path) -> None:
    rows = []
    for step, center_x in ((1, 0.0), (2, 0.5), (3, 1.0)):
        for agent_id, offset in enumerate((-0.05, 0.05)):
            rows.append(
                {
                    "step": step,
                    "agent_id": agent_id,
                    "pos_x": center_x + offset,
                    "pos_y": 0.0,
                    "cmd_omega_l": float(step),
                    "cmd_omega_r": float(-step),
                    "is_alive": 1,
                    "w_flow": 0.05,
                    "goal_x": 1.0,
                    "goal_y": 0.0,
                    "goal_threshold": 0.1,
                    "episode_max_steps": 5,
                }
            )
    path = tmp_path / "navigation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    result = analyze_csv(path, tmp_path / "navigation-report")

    assert result.scenario == "navigation"
    assert result.summary["navigation_time"] == 3
    assert result.summary["navigation_time_censored"] is False
    np.testing.assert_allclose(result.summary["control_smoothness"], 2.0)
    assert result.summary["control_smoothness_columns"] == [
        "cmd_omega_l",
        "cmd_omega_r",
    ]

    failure_frame = pd.DataFrame(rows)
    failure_frame["goal_x"] = 5.0
    failure_path = tmp_path / "navigation-failure.csv"
    failure_frame.to_csv(failure_path, index=False)
    failure = analyze_csv(failure_path, tmp_path / "navigation-failure-report")
    assert failure.summary["navigation_time"] == 5
    assert failure.summary["navigation_time_censored"] is True


def test_training_comparison_aggregates_seeds_with_traceable_ci(
    tmp_path: Path,
) -> None:
    """다른 logging 간격을 보간하되 seed를 독립 표본으로 집계한다."""

    run_specs: list[RunSpec] = []
    manifest_rows = []
    for variant_index, variant in enumerate(("mappo", "physics")):
        for seed in (1, 2):
            # seed 2의 중간 step을 생략해 alignment/interpolation도 함께 검증한다.
            steps = [0, 20] if seed == 2 else [0, 10, 20]
            reward_offset = (
                2.0 * variant_index
                + 0.1 * seed
                + 0.05 * variant_index * seed
            )
            frame = pd.DataFrame(
                {
                    "step": steps,
                    "average_episode_rewards": [
                        reward_offset + 0.2 * step for step in steps
                    ],
                    "pinn/loss_macro": [
                        1.0 - 0.02 * step + 0.05 * seed for step in steps
                    ],
                    "Param_Explore/w_food": [
                        0.02 + 0.001 * step + 0.002 * variant_index
                        for step in steps
                    ],
                }
            )
            run_path = tmp_path / f"{variant}-{seed}.csv"
            frame.to_csv(run_path, index=False)
            run_specs.append(RunSpec(variant, seed, run_path))
            manifest_rows.append(
                {"variant": variant, "seed": seed, "path": run_path.name}
            )

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    assert load_manifest(manifest_path) == tuple(run_specs)

    paper_path = tmp_path / "paper-values.csv"
    pd.DataFrame(
        {"metric": ["average_episode_rewards"], "paper_value": [6.0]}
    ).to_csv(paper_path, index=False)
    paper_values = load_paper_values(paper_path)

    result = compare_training_runs(
        run_specs,
        tmp_path / "comparison",
        baseline_variant="mappo",
        paper_values=paper_values,
        bootstrap_samples=2_000,
        bootstrap_seed=17,
    )

    aggregate = pd.read_csv(result.output_directory / "aggregate.csv")
    final_metrics = pd.read_csv(result.output_directory / "final_metrics.csv")
    comparisons = pd.read_csv(
        result.output_directory / "statistical_comparisons.csv"
    )
    paper_comparison = pd.read_csv(
        result.output_directory / "paper_comparison.csv"
    )
    final_reward = aggregate[
        (aggregate["variant"] == "physics")
        & (aggregate["metric"] == "average_episode_rewards")
        & (aggregate["step"] == 20)
    ].iloc[0]
    assert final_reward["n"] == 2
    np.testing.assert_allclose(final_reward["mean"], 6.225)
    assert np.isfinite(final_reward["ci95_low"])
    assert np.isfinite(final_reward["ci95_high"])
    assert len(final_metrics) == 12
    assert result.summary["variants"] == {"mappo": [1, 2], "physics": [1, 2]}
    reward_comparison = comparisons[
        comparisons["metric"] == "average_episode_rewards"
    ].iloc[0]
    assert reward_comparison["comparison"] == "paired-by-seed"
    assert reward_comparison["effect_size_name"] == "cohen_dz"
    np.testing.assert_allclose(reward_comparison["mean_difference"], 2.075)
    np.testing.assert_allclose(
        [
            reward_comparison["bootstrap_ci95_low"],
            reward_comparison["bootstrap_ci95_high"],
        ],
        [2.05, 2.10],
    )
    physics_paper_gap = paper_comparison[
        paper_comparison["variant"] == "physics"
    ].iloc[0]
    np.testing.assert_allclose(physics_paper_gap["signed_gap"], 0.225)
    assert result.summary["statistical_comparison"]["baseline"] == "mappo"
    assert result.summary["paper_values"] == paper_values

    png_paths = [path for path in result.generated_files if path.suffix == ".png"]
    assert {path.name for path in png_paths} >= {
        "outcomes_comparison.png",
        "losses_comparison.png",
        "parameters_comparison.png",
        "final_metric_distributions.png",
    }
    for path in png_paths:
        _assert_png(path)


def test_training_comparison_uses_record_order_for_constant_legacy_steps(
    tmp_path: Path,
) -> None:
    paths = []
    for seed, rewards in ((1, [1.0, 2.0, 4.0]), (2, [2.0, 3.0, 5.0])):
        path = tmp_path / f"legacy-{seed}.csv"
        pd.DataFrame(
            {"step": [0, 0, 0], "average_episode_rewards": rewards}
        ).to_csv(path, index=False)
        paths.append(path)

    result = compare_training_runs(
        [RunSpec("legacy", seed, path) for seed, path in enumerate(paths, start=1)],
        tmp_path / "constant-step-comparison",
    )
    aggregate = pd.read_csv(result.output_directory / "aggregate.csv")

    assert aggregate["step"].tolist() == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(aggregate["mean"], [1.5, 2.5, 4.5])


def test_training_comparison_uses_welch_for_different_seed_sets(
    tmp_path: Path,
) -> None:
    """seed pair가 성립하지 않으면 표본을 버리지 않고 Welch 비교를 사용한다."""

    runs: list[RunSpec] = []
    for variant, seed_values in {
        "baseline": {1: 1.0, 2: 2.0},
        "candidate": {3: 2.5, 4: 4.0},
    }.items():
        for seed, value in seed_values.items():
            path = tmp_path / f"{variant}-{seed}.csv"
            pd.DataFrame({"step": [10], "score": [value]}).to_csv(
                path, index=False
            )
            runs.append(RunSpec(variant, seed, path))

    result = compare_training_runs(
        runs,
        tmp_path / "welch-comparison",
        baseline_variant="baseline",
        bootstrap_samples=1_000,
        bootstrap_seed=3,
    )
    comparisons = pd.read_csv(
        result.output_directory / "statistical_comparisons.csv"
    )
    row = comparisons.iloc[0]

    assert row["comparison"] == "unpaired-welch"
    assert row["test"] == "welch-t"
    assert row["effect_size_name"] == "hedges_g"
    assert row["n_shared_seeds"] == 0
    assert np.isfinite(row["p_value"])
    assert np.isfinite(row["effect_size"])
    assert np.isfinite(row["bootstrap_ci95_low"])
    assert np.isfinite(row["bootstrap_ci95_high"])
